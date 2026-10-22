"""Ana pencere: menü, araç çubuğu, durum çubuğu ve parçaların birbirine bağlanması.

Bu dosyada iş mantığı yoktur. "Şu olursa şunu çağır" bağlantıları vardır; mantık
``Denetleyici``'dedir.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QFileDialog, QInputDialog, QLabel, QMainWindow, QMessageBox, QProgressBar,
                               QToolBar)

from .. import dosya
from .denetleyici import Denetleyici
from .donusum import nesne_ikonu
from .ekleme_paneli import EklemePaneli
from .nesne_menusu import NesneMenusu
from .tuval import Tuval
from .video_penceresi import VIDEO_FILTRESI, VideoPenceresi

KISAYOL_METNI = """<b>Görüntü</b><br>
Fare tekerleği: yakınlaştır &nbsp;·&nbsp; Sağ/orta tuş sürükle ya da Boşluk+sürükle: kaydır<br>
Ctrl+0: ekrana sığdır &nbsp;·&nbsp; Ctrl+1: %100<br><br>
<b>Nesneler</b><br>
Listeden seç ya da görüntüde tıkla &nbsp;·&nbsp; Ctrl+= genişlet &nbsp;·&nbsp; Ctrl+- daralt &nbsp;·&nbsp; Del sil<br><br>
<b>Görsel ekleme</b><br>
Sürükle: konumlandır &nbsp;·&nbsp; Ctrl+tekerlek: boyutlandır &nbsp;·&nbsp; Enter: uygula &nbsp;·&nbsp; Esc: iptal"""


class AnaPencere(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(1280, 800)
        self.setAcceptDrops(True)

        self.d = Denetleyici(self)
        self.tuval = Tuval()
        self.setCentralWidget(self.tuval)
        self.menu = NesneMenusu(self.tuval)
        self.panel = EklemePaneli(self.tuval)
        self.video_penceresi: VideoPenceresi | None = None
        self._ustune_yazma_onaylari: set[Path] = set()

        self._eylemleri_olustur()
        self._menuleri_kur()
        self._durum_cubugunu_kur()
        self._baglantilari_kur()
        self.d.ayar = self.panel.ayar()
        self._eylemleri_guncelle()
        self.setWindowTitle(self.d.baslik())

    # ================================================================== kurulum
    def _eylem(self, metin: str, kisayol: str | QKeySequence | None, slot, ipucu: str = "") -> QAction:
        e = QAction(metin, self)
        if kisayol:
            e.setShortcut(QKeySequence(kisayol) if isinstance(kisayol, str) else kisayol)
        e.setStatusTip(ipucu or metin)
        e.setToolTip(ipucu or metin)
        e.triggered.connect(lambda _=False: slot())
        return e

    def _eylemleri_olustur(self) -> None:
        d = self.d
        self.a_ac = self._eylem("Aç…", "Ctrl+O", self._ac_dialog, "Görüntü aç")
        self.a_kaydet = self._eylem("Kaydet", "Ctrl+S", self._kaydet, "Değişiklikleri kaydet")
        self.a_farkli = self._eylem("Farklı kaydet…", "Ctrl+Shift+S", self._farkli_kaydet,
                                    "Başka bir ad ya da biçimde kaydet (PNG, JPG, WEBP, BMP, TIFF…)")
        self.a_video = self._eylem("Videodaki nesneleri listele…", "Ctrl+Shift+V", self._video_ac,
                                   "Videodan kare örnekleyip nesneleri listeler")
        self.a_cikis = self._eylem("Çıkış", "Ctrl+Q", self.close)
        self.a_geri = self._eylem("Geri al", QKeySequence.StandardKey.Undo, d.geri_al)
        self.a_yinele = self._eylem("Yinele", QKeySequence.StandardKey.Redo, d.yinele)
        self.a_sil = self._eylem("Seçili nesneyi sil", "Del", d.secileni_sil,
                                 "Nesneyi siler ve arkasını çevreye uygun doldurur")
        self.a_genislet = self._eylem("Nesneyi genişlet", "Ctrl+=", d.genislet)
        self.a_daralt = self._eylem("Nesneyi daralt", "Ctrl+-", d.daralt)
        self.a_algila = self._eylem("Nesneleri yeniden algıla", "F5", lambda: d.algila(self.menu.duyarlilik()))
        self.a_ekle = self._eylem("Görsel ekle…", "Ctrl+I", self._gorsel_ekle_dialog,
                                  "Görüntüye başka bir görsel ekle; ışık ve renk sahneye uydurulur")
        self.a_sigdir = self._eylem("Ekrana sığdır", "Ctrl+0", lambda: self.tuval.sigdir())
        self.a_yuzde100 = self._eylem("%100", "Ctrl+1", lambda: self.tuval.yuzde_yuz())
        self.a_kisayollar = self._eylem("Kısayollar", "F1", lambda: QMessageBox.information(
            self, "Kısayollar", KISAYOL_METNI))
        self.a_uygula = self._eylem("Eklemeyi uygula", "Return", d.ekleme_uygula)
        self.a_iptal = self._eylem("Eklemeyi iptal et", "Escape", d.ekleme_iptal)
        for e in (self.a_uygula, self.a_iptal):          # yalnızca ekleme sırasında etkin
            self.addAction(e)
        # Enter ve Esc başka yerlerde (liste, kaydırıcı) tüketilmesin diye pencere kapsamlı
        for e in (self.a_uygula, self.a_iptal):
            e.setShortcutContext(Qt.ShortcutContext.WindowShortcut)

    def _menuleri_kur(self) -> None:
        m = self.menuBar()
        dosya_m = m.addMenu("&Dosya")
        for e in (self.a_ac, self.a_kaydet, self.a_farkli):
            dosya_m.addAction(e)
        dosya_m.addSeparator()
        dosya_m.addAction(self.a_video)
        dosya_m.addSeparator()
        dosya_m.addAction(self.a_cikis)

        duzen = m.addMenu("&Düzen")
        for e in (self.a_geri, self.a_yinele):
            duzen.addAction(e)
        duzen.addSeparator()
        for e in (self.a_sil, self.a_genislet, self.a_daralt):
            duzen.addAction(e)

        nesne = m.addMenu("&Nesne")
        for e in (self.a_algila, self.a_ekle):
            nesne.addAction(e)

        gorunum = m.addMenu("&Görünüm")
        for e in (self.a_sigdir, self.a_yuzde100):
            gorunum.addAction(e)
        m.addMenu("&Yardım").addAction(self.a_kisayollar)

        arac = QToolBar("Araçlar")
        arac.setMovable(False)
        arac.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.addToolBar(arac)
        for e in (self.a_ac, self.a_kaydet, self.a_farkli):
            arac.addAction(e)
        arac.addSeparator()
        for e in (self.a_geri, self.a_yinele):
            arac.addAction(e)
        arac.addSeparator()
        for e in (self.a_algila, self.a_sil, self.a_ekle):
            arac.addAction(e)
        arac.addSeparator()
        arac.addAction(self.a_video)

    def _durum_cubugunu_kur(self) -> None:
        sb = self.statusBar()
        self._zoom_etiketi = QLabel("")
        self._boyut_etiketi = QLabel("")
        self._mesgul = QProgressBar()
        self._mesgul.setRange(0, 0)                  # belirsiz ilerleme
        self._mesgul.setMaximumWidth(140)
        self._mesgul.setTextVisible(False)
        self._mesgul.hide()
        sb.addPermanentWidget(self._mesgul)
        sb.addPermanentWidget(self._boyut_etiketi)
        sb.addPermanentWidget(self._zoom_etiketi)

    def _baglantilari_kur(self) -> None:
        d, t, mn, p = self.d, self.tuval, self.menu, self.panel
        # denetleyici -> arayüz
        d.belge_acildi.connect(lambda rgb: (t.goruntu_ayarla(rgb, True), self._boyut_etiketini_yaz()))
        d.goruntu_degisti.connect(lambda rgb: t.goruntu_ayarla(rgb, False))
        d.kontur_degisti.connect(t.kontur_ayarla)
        d.nesneler_degisti.connect(self._nesne_listesini_yenile)
        d.secim_degisti.connect(mn.secimi_ayarla)
        d.ekleme_degisti.connect(self._ekleme_degisti)
        d.ekleme_boyutu_degisti.connect(lambda f: (p.boyutu_ayarla(f), t.ekleme_boyutu_degisti()))
        d.ekleme_arka_plan_secenegi.connect(p.arka_plan_secenegini_ayarla)
        d.onizleme_hazir.connect(self._onizleme_hazir)
        d.durum_degisti.connect(self._eylemleri_guncelle)
        d.mesgul_degisti.connect(self._mesgul_degisti)
        d.mesaj.connect(lambda s: self.statusBar().showMessage(s, 8000))
        d.hata.connect(lambda s: QMessageBox.critical(self, "Hata", s[-1800:]))
        # tuval -> denetleyici
        t.nokta_tiklandi.connect(d.noktadan_sec)
        t.gorsel_tasindi.connect(d.ekleme_tasindi)
        t.gorsel_olceklendi.connect(d.ekleme_olceklendi)
        t.zoom_degisti.connect(lambda z: self._zoom_etiketi.setText(f"%{z * 100:.0f}"))
        # nesne menüsü -> denetleyici
        mn.secildi.connect(d.sec)
        mn.sil_istendi.connect(d.secileni_sil)
        mn.genislet_istendi.connect(d.genislet)
        mn.daralt_istendi.connect(d.daralt)
        mn.yeniden_algila.connect(d.algila)
        # ekleme paneli -> denetleyici
        p.ayar_degisti.connect(lambda: d.ekleme_ayari_degisti(p.ayar()))
        p.boyut_degisti.connect(d.ekleme_olceklendi)
        p.arka_plan_degisti.connect(d.arka_plan_temizlemeyi_ayarla)
        p.uygula.connect(d.ekleme_uygula)
        p.iptal.connect(d.ekleme_iptal)

    # ================================================================== durum güncelleme
    def _eylemleri_guncelle(self) -> None:
        d = self.d
        belge = d.belge is not None
        bos = not d.mesgul and not d.ekleme_aktif           # kullanıcı serbestçe işlem yapabilir mi
        self.a_ac.setEnabled(bos)
        self.a_kaydet.setEnabled(belge and bos)
        self.a_farkli.setEnabled(belge and bos)
        self.a_geri.setEnabled(belge and bos and d.belge.geri_alinabilir)
        self.a_yinele.setEnabled(belge and bos and d.belge.yinelenebilir)
        self.a_geri.setText(f"Geri al: {d.belge.son_adim_adi}" if self.a_geri.isEnabled() else "Geri al")
        self.a_yinele.setText(f"Yinele: {d.belge.yinelenecek_adim_adi}" if self.a_yinele.isEnabled() else "Yinele")
        secili = d.secili_nesne is not None
        self.a_sil.setEnabled(secili and bos)
        self.a_genislet.setEnabled(secili and bos)
        self.a_daralt.setEnabled(secili and bos)
        self.a_algila.setEnabled(belge and bos)
        self.a_ekle.setEnabled(belge and bos)
        self.a_uygula.setEnabled(d.ekleme_aktif and not d.mesgul)
        self.a_iptal.setEnabled(d.ekleme_aktif and not d.mesgul)
        self.setWindowTitle(d.baslik())

    def _mesgul_degisti(self, mesgul: bool, mesaj: str) -> None:
        self._mesgul.setVisible(mesgul)
        self.menu.mesgul_ayarla(mesgul)
        self.panel.mesgul_ayarla(mesgul)
        if mesgul:
            self.statusBar().showMessage(mesaj)
            self.tuval.setCursor(Qt.CursorShape.BusyCursor)
        else:
            self.tuval.unsetCursor()
        self._eylemleri_guncelle()

    def _boyut_etiketini_yaz(self) -> None:
        if self.d.belge is not None:
            w, h = self.d.belge.boyut
            self._boyut_etiketi.setText(f"{w} × {h} px   ")

    def _nesne_listesini_yenile(self) -> None:
        d = self.d
        if d.belge is None:
            self.menu.liste_doldur([])
            return
        toplam = d.belge.rgb.shape[0] * d.belge.rgb.shape[1]
        ogeler = [(f"{n.ad}\n%{100.0 * n.alan / toplam:.1f} alan", nesne_ikonu(d.belge.rgb, n.maske, n.kutu))
                  for n in d.nesneler]
        self.menu.liste_doldur(ogeler, d.secili)

    def _ekleme_degisti(self, gorsel) -> None:
        aktif = gorsel is not None
        self.tuval.ekleme_ayarla(gorsel)
        self.menu.setVisible(not aktif)
        if aktif:
            self.panel.goster_ve_yerles()
            self.panel.boyutu_ayarla(gorsel.olcek)
        else:
            self.panel.hide()
        self._eylemleri_guncelle()

    def _onizleme_hazir(self, sonuc) -> None:
        if sonuc is None:
            self.tuval.onizleme_ayarla(None, None)
        else:
            self.tuval.onizleme_ayarla(sonuc[0], sonuc[1])

    # ================================================================== dosya işlemleri
    def _ac_dialog(self) -> None:
        if not self._degisiklik_kontrolu():
            return
        yol, _ = QFileDialog.getOpenFileName(self, "Görüntü aç", "", dosya.acma_filtresi())
        if yol:
            self.d.ac(yol)

    def ac_dosya(self, yol: str) -> None:
        """Komut satırından/sürükle-bırakla açma."""
        if Path(yol).suffix.lower() in {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".webm", ".m4v", ".mpg", ".mpeg"}:
            self._video_ac(yol)
        elif self._degisiklik_kontrolu():
            self.d.ac(yol)

    def _gorsel_ekle_dialog(self) -> None:
        yol, _ = QFileDialog.getOpenFileName(self, "Eklenecek görseli seç", "", dosya.acma_filtresi())
        if yol:
            self.d.gorsel_ekle(yol)

    def _kalite_sor(self, yol: str) -> int | None:
        """Kayıplı biçimlerde (JPEG/WEBP/HEIC) kalite sorar. İptalde None."""
        bicim = dosya.bicim_bul(yol)
        if bicim is None or not bicim.kayipli:
            return 95
        kalite, tamam = QInputDialog.getInt(self, f"{bicim.ad} kalitesi", "Kalite (1–100):", 95, 1, 100)
        return kalite if tamam else None

    def _kaydet(self) -> None:
        b = self.d.belge
        if b is None:
            return
        if b.yol is None or dosya.bicim_bul(b.yol) is None:
            self._farkli_kaydet()
            return
        if b.yol not in self._ustune_yazma_onaylari:
            cevap = QMessageBox.question(
                self, "Üzerine yaz", f"Orijinal dosyanın üzerine yazılacak:\n{b.yol}\n\nDevam edilsin mi?\n"
                "(Orijinali korumak için 'Farklı kaydet' kullanın.)")
            if cevap != QMessageBox.StandardButton.Yes:
                return
            self._ustune_yazma_onaylari.add(b.yol)
        kalite = self._kalite_sor(str(b.yol))
        if kalite is not None:
            self.d.kaydet(kalite)

    def _farkli_kaydet(self) -> None:
        b = self.d.belge
        if b is None:
            return
        onerilen = str((b.yol.with_name(b.yol.stem + "_duzenlendi.png")) if b.yol else "adsiz.png")
        yol, filtre = QFileDialog.getSaveFileName(self, "Farklı kaydet", onerilen, dosya.kaydetme_filtreleri())
        if not yol:
            return
        if dosya.bicim_bul(yol) is None:                      # uzantı yazılmadıysa seçili süzgeçten türet
            for bicim in dosya.KAYIT_BICIMLERI:
                if filtre.startswith(bicim.ad):
                    yol += bicim.uzantilar[0]
                    break
        bicim = dosya.bicim_bul(yol)
        if bicim is None:
            QMessageBox.warning(self, "Farklı kaydet", "Bu uzantı desteklenmiyor. PNG, JPG, WEBP, BMP, TIFF… seçin.")
            return
        if b.alfa is not None and not bicim.alfa:
            QMessageBox.information(self, "Saydamlık", f"{bicim.ad} saydamlığı desteklemez; "
                                    "saydam alanlar beyaza dönüştürülecek.")
        kalite = self._kalite_sor(yol)
        if kalite is not None:
            self.d.farkli_kaydet(yol, kalite)

    def _degisiklik_kontrolu(self) -> bool:
        """Kaydedilmemiş değişiklik varsa kullanıcıya sorar. Devam edilebilirse True."""
        b = self.d.belge
        if b is None or not b.degisti:
            return True
        cevap = QMessageBox.question(
            self, "Kaydedilmemiş değişiklik", "Değişiklikler kaydedilmedi. Yine de devam edilsin mi?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        return cevap == QMessageBox.StandardButton.Yes

    # ================================================================== video
    def _video_ac(self, yol: str | None = None) -> None:
        if self.video_penceresi is None:
            self.video_penceresi = VideoPenceresi(self, self.menu.duyarlilik())
        self.video_penceresi.duyarliligi_ayarla(self.menu.duyarlilik())
        self.video_penceresi.show()
        self.video_penceresi.raise_()
        if yol:
            self.video_penceresi.video_ac(yol)
        else:
            self.video_penceresi._video_sec()

    # ================================================================== sürükle-bırak / kapanış
    def dragEnterEvent(self, e) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e) -> None:
        yollar = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if yollar:
            self.ac_dosya(yollar[0])

    def closeEvent(self, e) -> None:
        if self._degisiklik_kontrolu():
            e.accept()
        else:
            e.ignore()
