"""Denetleyici: belge, nesne algılama, silme ve görsel ekleme akışını yönetir.

Arayüz parçaları (tuval, menüler) birbirini tanımaz; hepsi bu sınıfın sinyallerini dinler
ve ona istek yollar. Böylece:

* Mantık tek yerde toplanır, pencere kodu yalnızca bağlantı kurar.
* Ağır işler (algılama, doldurma, uyumlama) burada ``Isci`` ile arka planda çalışır.
* Bu sınıf pencere bileşeni içermediği için (QObject) ekransız da denenebilir.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal

from .. import dosya
from ..belge import Belge
from ..cekirdek import doldurma, filtre, kesme, kontur, nesne_bulucu, uyumlama
from ..cekirdek.nesne_bulucu import Analiz, Nesne
from ..cekirdek.uyumlama import EklenenGorsel, UyumAyarlari
from .isci import Isci

EKLEME_EN_BUYUK_KENAR = 1600       # eklenen görsel bundan büyükse çalışma kopyası küçültülür


def duyarliliktan_esige(d: float) -> float:
    """Arayüz duyarlılığı (0..1) -> nesne_bulucu göreli eşiği. Yüksek duyarlılık = düşük eşik."""
    return 0.8 - 0.6 * d


class Denetleyici(QObject):
    belge_acildi = Signal(object)              # rgb uint8; görünüm sıfırlanır
    goruntu_degisti = Signal(object)           # rgb uint8; görünüm korunur
    nesneler_degisti = Signal()
    secim_degisti = Signal(int)                # -1 = yok
    kontur_degisti = Signal(object)            # list[np.ndarray] | None
    ekleme_degisti = Signal(object)            # EklenenGorsel | None
    ekleme_boyutu_degisti = Signal(float)
    ekleme_arka_plan_secenegi = Signal(bool)   # görselin kendi alfası yoksa True
    onizleme_hazir = Signal(object)            # ((y0,x0,y1,x1), rgb_u8) | None
    durum_degisti = Signal()                   # menü etkinliklerini yenilemek için
    mesgul_degisti = Signal(bool, str)
    mesaj = Signal(str)
    hata = Signal(str)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self.belge: Belge | None = None
        self.analiz: Analiz | None = None
        self.nesneler: list[Nesne] = []
        self.secili = -1
        self.ekleme: EklenenGorsel | None = None
        self.ayar = UyumAyarlari()
        self.duyarlilik = 0.58
        self.arka_plan_temizle = True

        self._isci: Isci | None = None
        self._onizleme_isci: Isci | None = None
        self._onizleme_bekliyor = False
        self._ekleme_ham: tuple[np.ndarray, np.ndarray | None] | None = None
        self._ekleme_yol: str | None = None

        self._algila_zamanlayici = QTimer(self)
        self._algila_zamanlayici.setSingleShot(True)
        self._algila_zamanlayici.setInterval(500)
        self._algila_zamanlayici.timeout.connect(lambda: self.algila())
        self._onizleme_zamanlayici = QTimer(self)
        self._onizleme_zamanlayici.setSingleShot(True)
        self._onizleme_zamanlayici.setInterval(120)
        self._onizleme_zamanlayici.timeout.connect(self._onizleme_baslat)

    # ------------------------------------------------------------------ durum
    @property
    def mesgul(self) -> bool:
        return self._isci is not None

    @property
    def ekleme_aktif(self) -> bool:
        return self.ekleme is not None

    @property
    def secili_nesne(self) -> Nesne | None:
        return self.nesneler[self.secili] if 0 <= self.secili < len(self.nesneler) else None

    def baslik(self) -> str:
        if self.belge is None:
            return "Nesne Silici"
        ad = self.belge.yol.name if self.belge.yol else "Adsız"
        return f"{'● ' if self.belge.degisti else ''}{ad} — Nesne Silici"

    def _calistir(self, mesaj: str, fonksiyon, *args, bitti) -> bool:
        """Ağır işi arka planda başlatır. Bitince ``bitti(sonuc)`` çağrılır."""
        if self.mesgul:
            return False
        isci = Isci(fonksiyon, *args, parent=self)

        def tamam(sonuc) -> None:
            self._isci = None
            self.mesgul_degisti.emit(False, "")
            bitti(sonuc)

        def basarisiz(izleme: str) -> None:
            self._isci = None
            self.mesgul_degisti.emit(False, "")
            self.hata.emit(izleme)

        isci.bitti.connect(tamam)
        isci.hata.connect(basarisiz)
        isci.finished.connect(isci.deleteLater)
        self._isci = isci
        self.mesgul_degisti.emit(True, mesaj)
        isci.start()
        return True

    # ------------------------------------------------------------------ dosya
    def ac(self, yol: str) -> None:
        try:
            rgb, alfa = dosya.ac(yol)
        except Exception as e:                                   # noqa: BLE001
            self.hata.emit(f"Dosya açılamadı:\n{yol}\n\n{e}")
            return
        self._ekleme_birak()
        self._algila_zamanlayici.stop()
        self.belge = Belge(rgb, alfa, Path(yol))
        self.analiz, self.nesneler, self.secili = None, [], -1
        self.belge_acildi.emit(self.belge.rgb)
        self.kontur_degisti.emit(None)
        self.nesneler_degisti.emit()
        self.durum_degisti.emit()
        self.mesaj.emit(f"{Path(yol).name} açıldı ({rgb.shape[1]}×{rgb.shape[0]})")
        self.algila()

    def kaydet(self, kalite: int = 95) -> None:
        """Açıldığı dosyanın üzerine yazar (arayüz onayı aldıktan sonra çağrılır)."""
        if self.belge is not None and self.belge.yol is not None:
            self.farkli_kaydet(str(self.belge.yol), kalite)

    def farkli_kaydet(self, yol: str, kalite: int = 95) -> None:
        if self.belge is None:
            return
        b = self.belge
        hedef = Path(yol)

        def bitti(_sonuc) -> None:
            b.kaydedildi(hedef)
            self.durum_degisti.emit()
            self.mesaj.emit(f"Kaydedildi: {hedef.name}")

        self._calistir("Kaydediliyor…", dosya.kaydet, hedef, b.rgb, b.alfa, kalite, bitti=bitti)

    # ------------------------------------------------------------------ algılama / seçim
    def algila(self, duyarlilik: float | None = None) -> None:
        if self.belge is None or self.ekleme_aktif:
            return
        if duyarlilik is not None:
            self.duyarlilik = duyarlilik
        self._algila_zamanlayici.stop()
        girdi = self.belge.analiz_girdisi()
        esik = duyarliliktan_esige(self.duyarlilik)

        def bitti(analiz: Analiz) -> None:
            self.analiz, self.nesneler, self.secili = analiz, list(analiz.nesneler), -1
            self.kontur_degisti.emit(None)
            self.nesneler_degisti.emit()
            self.secim_degisti.emit(-1)
            self.durum_degisti.emit()
            self.mesaj.emit(f"{len(self.nesneler)} nesne bulundu. Listeden seç ya da görüntüde tıkla.")

        self._calistir("Nesneler algılanıyor…", nesne_bulucu.analiz_et, girdi, 16, 0.004, 0.55, esik, bitti=bitti)

    def _algilamayi_planla(self) -> None:
        """Geri al/yinele gibi ardışık işlemlerde her birinde algılama koşmasın diye bekletir."""
        self.analiz, self.nesneler, self.secili = None, [], -1
        self.kontur_degisti.emit(None)
        self.nesneler_degisti.emit()
        self._algila_zamanlayici.start()

    def sec(self, indeks: int) -> None:
        self.secili = indeks if 0 <= indeks < len(self.nesneler) else -1
        n = self.secili_nesne
        self.kontur_degisti.emit(kontur.konturlar(n.maske) if n else None)
        self.secim_degisti.emit(self.secili)
        self.durum_degisti.emit()
        if n:
            self.mesaj.emit(f"{n.ad} seçildi — alan {n.alan:,} piksel".replace(",", "."))

    def noktadan_sec(self, x: int, y: int) -> None:
        """Görüntüde tıklanan noktadaki nesneyi seçer; listede yoksa ağaçtan yenisini üretir."""
        if self.belge is None or self.mesgul or self.ekleme_aktif:
            return
        icinde = [i for i, n in enumerate(self.nesneler) if n.maske[y, x]]
        if icinde:
            self.sec(min(icinde, key=lambda i: self.nesneler[i].alan))
            return
        if self.analiz is None:
            return
        kimlik = max((n.kimlik for n in self.nesneler), default=0) + 1
        yeni = nesne_bulucu.nokta_ile_nesne(self.analiz, self.belge.analiz_girdisi(), x, y, kimlik)
        if yeni is None:
            self.sec(-1)
            self.mesaj.emit("Bu noktada ayrı bir nesne bulunamadı.")
            return
        yeni.ad = f"Nesne {kimlik} (elle)"
        self.nesneler.append(yeni)
        self.nesneler_degisti.emit()
        self.sec(len(self.nesneler) - 1)

    def _secili_degistir(self, yukari: bool) -> None:
        n = self.secili_nesne
        if n is None or self.analiz is None or self.belge is None or self.mesgul:
            return
        yeni = nesne_bulucu.nesneyi_genislet(self.analiz, self.belge.analiz_girdisi(), n, yukari)
        if yeni is None:
            self.mesaj.emit("Daha fazla büyütülemiyor." if yukari else "Daha fazla daraltılamıyor.")
            return
        yeni.ad = n.ad
        self.nesneler[self.secili] = yeni
        self.nesneler_degisti.emit()
        self.sec(self.secili)

    def genislet(self) -> None:
        self._secili_degistir(True)

    def daralt(self) -> None:
        self._secili_degistir(False)

    # ------------------------------------------------------------------ silme / geri al
    def secileni_sil(self) -> None:
        n, b = self.secili_nesne, self.belge
        if n is None or b is None or self.ekleme_aktif:
            return

        def bitti(sonuc) -> None:
            kutu, yama = sonuc
            b.yama_uygula(kutu, yama, f"{n.ad} silindi")
            self.goruntu_degisti.emit(b.rgb)
            self.durum_degisti.emit()
            self.mesaj.emit(f"{n.ad} silindi. Geri almak için Ctrl+Z.")
            self.algila()                       # görüntü değişti: nesne listesi yenilenir

        self._calistir(f"{n.ad} siliniyor ve arkası dolduruluyor…", doldurma.nesneyi_sil, b.rgb, n.maske, bitti=bitti)

    def geri_al(self) -> None:
        if self.belge is None or self.mesgul or self.ekleme_aktif:
            return
        adim = self.belge.geri_al()
        if adim:
            self._belge_degisti(f"Geri alındı: {adim.aciklama}")

    def yinele(self) -> None:
        if self.belge is None or self.mesgul or self.ekleme_aktif:
            return
        adim = self.belge.yinele()
        if adim:
            self._belge_degisti(f"Yinelendi: {adim.aciklama}")

    def _belge_degisti(self, mesaj: str) -> None:
        self.goruntu_degisti.emit(self.belge.rgb)
        self.durum_degisti.emit()
        self.mesaj.emit(mesaj)
        self._algilamayi_planla()

    # ------------------------------------------------------------------ görsel ekleme
    def gorsel_ekle(self, yol: str) -> None:
        """Bir görüntü dosyasını okuyup yerleştirme moduna geçer."""
        if self.belge is None or self.mesgul:
            return
        try:
            rgb_u8, alfa_u8 = dosya.ac(yol)
        except Exception as e:                                   # noqa: BLE001
            self.hata.emit(f"Görsel açılamadı:\n{yol}\n\n{e}")
            return
        self._ekleme_yol = yol
        self._ekleme_ham = (rgb_u8.astype(np.float32) / 255.0,
                            None if alfa_u8 is None else alfa_u8.astype(np.float32) / 255.0)
        self.ekleme_arka_plan_secenegi.emit(alfa_u8 is None)
        self._ekleme_hazirla(ilk=True)

    def arka_plan_temizlemeyi_ayarla(self, acik: bool) -> None:
        self.arka_plan_temizle = acik
        if self.ekleme_aktif and self._ekleme_ham is not None:
            self._ekleme_hazirla(ilk=False)

    def _ekleme_hazirla(self, ilk: bool) -> None:
        """Ham görselden alfa üretir (gerekirse), boyutlandırır ve tuvale koyar."""
        rgb, alfa = self._ekleme_ham
        temizle = self.arka_plan_temizle and alfa is None
        W, H = self.belge.boyut
        eski = self.ekleme

        def hazirla():
            r, a = rgb, alfa
            if max(r.shape[:2]) > EKLEME_EN_BUYUK_KENAR:                 # çalışma kopyası
                oran = EKLEME_EN_BUYUK_KENAR / max(r.shape[:2])
                yh, yw = round(r.shape[0] * oran), round(r.shape[1] * oran)
                r = filtre.boyutlandir(r, yh, yw)
                a = None if a is None else filtre.boyutlandir(a, yh, yw)
            if a is None:
                a = kesme.arka_plani_kaldir(r) if temizle else np.ones(r.shape[:2], np.float32)
                r, a = kesme.kirp(r, a)
            return r, a

        def bitti(sonuc) -> None:
            r, a = sonuc
            if eski is not None:                       # arka plan seçeneği değişti: konum/ölçek korunur
                g = EklenenGorsel(r, a, eski.x, eski.y, eski.olcek)
            else:                                      # ilk yerleştirme: ortada, tuvalin ~%35'i
                olcek = min(1.0, 0.35 * min(W, H) / max(r.shape[:2]))
                g = EklenenGorsel(r, a, (W - r.shape[1] * olcek) / 2, (H - r.shape[0] * olcek) / 2, olcek)
            self.ekleme = g
            self._algila_zamanlayici.stop()
            self.kontur_degisti.emit(None)
            self.ekleme_degisti.emit(g)
            self.ekleme_boyutu_degisti.emit(g.olcek)
            self.durum_degisti.emit()
            if ilk:
                self.mesaj.emit("Görseli sürükleyip konumlandır; hazır olunca Uygula.")
            self.onizleme_iste()

        self._calistir("Görsel hazırlanıyor…", hazirla, bitti=bitti)

    def ekleme_ayari_degisti(self, ayar: UyumAyarlari) -> None:
        self.ayar = ayar
        self.onizleme_iste()

    def ekleme_tasindi(self) -> None:
        self.onizleme_iste()

    def ekleme_olceklendi(self, yeni_olcek: float) -> None:
        g = self.ekleme
        if g is None:
            return
        yeni_olcek = float(np.clip(yeni_olcek, 0.05, 4.0))
        # Ölçek, görselin merkezi sabit kalacak şekilde değişir.
        h, w = g.alfa.shape
        g.x += w * (g.olcek - yeni_olcek) / 2
        g.y += h * (g.olcek - yeni_olcek) / 2
        g.olcek = yeni_olcek
        self.ekleme_boyutu_degisti.emit(yeni_olcek)
        self.onizleme_iste()

    def onizleme_iste(self) -> None:
        """Parametre değişimlerini birleştirip (debounce) uyumlanmış önizlemeyi ister."""
        self.onizleme_hazir.emit(None)                   # eskimiş önizlemeyi gizle
        self._onizleme_zamanlayici.start()

    def _onizleme_baslat(self) -> None:
        if self.ekleme is None or self.belge is None:
            return
        if self._onizleme_isci is not None:              # biri çalışıyor: bitince tekrar dene
            self._onizleme_bekliyor = True
            return
        g = copy.copy(self.ekleme)                       # iş sürerken kullanıcı sürüklese de tutarlı kalsın
        isci = Isci(uyumlama.yerlestir, self.belge.rgb, g, self.ayar, parent=self)

        def tamam(sonuc) -> None:
            self._onizleme_isci = None
            if self.ekleme is not None and (g.x, g.y, g.olcek) == (self.ekleme.x, self.ekleme.y, self.ekleme.olcek):
                self.onizleme_hazir.emit(sonuc)
            if self._onizleme_bekliyor:
                self._onizleme_bekliyor = False
                self._onizleme_baslat()

        def basarisiz(izleme: str) -> None:
            self._onizleme_isci = None
            self.hata.emit(izleme)

        isci.bitti.connect(tamam)
        isci.hata.connect(basarisiz)
        isci.finished.connect(isci.deleteLater)
        self._onizleme_isci = isci
        isci.start()

    def ekleme_uygula(self) -> None:
        g, b = self.ekleme, self.belge
        if g is None or b is None:
            return
        g = copy.copy(g)

        def bitti(sonuc) -> None:
            if sonuc is None:
                self.mesaj.emit("Görsel tuvalin dışında; önce görüntünün üzerine getir.")
                return
            kutu, yama = sonuc
            b.yama_uygula(kutu, yama, "Görsel eklendi")
            self._ekleme_birak()
            self.goruntu_degisti.emit(b.rgb)
            self.durum_degisti.emit()
            self.mesaj.emit("Görsel eklendi. Geri almak için Ctrl+Z.")
            self.algila()

        self._calistir("Görsel sahneye uyduruluyor…", uyumlama.yerlestir, b.rgb, g, self.ayar, bitti=bitti)

    def ekleme_iptal(self) -> None:
        if self.ekleme_aktif:
            self._ekleme_birak()
            self.durum_degisti.emit()
            self.mesaj.emit("Görsel ekleme iptal edildi.")
            self.algila()

    def _ekleme_birak(self) -> None:
        self._onizleme_zamanlayici.stop()
        self.ekleme = None
        self._ekleme_ham = None
        self.ekleme_degisti.emit(None)
        self.onizleme_hazir.emit(None)
