"""Video penceresi: videodan örneklenen karelerde nesneleri algılayıp listeler.

Üç aşama:

1. **Kare yakalama** (``KareYakalayici``): Qt Multimedia ile video belirli aralıklarla
   (varsayılan her 1 sn) konumlanır, gelen kare küçültülüp saklanır. Qt'nin kare
   çözmesi arayüz iş parçacığında çalıştığı için bu aşama olay döngüsüyle yürür.
2. **Algılama** (``VideoAnalizi``): Her karede ``nesne_bulucu`` çalışır; ağır olduğu için
   ayrı iş parçacığındadır. Nesneler ``izleyici`` ile kareler arası eşleşir.
3. **Liste**: Her iz (aynı nesne) bir satırdır: önizleme, görüldüğü süre, kare sayısı.
   Satırı seçince en net karesi turuncu konturla gösterilir; kaydırıcıyla nesnenin diğer
   karelerine bakılır.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QSize, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtMultimedia import QMediaPlayer, QVideoSink
from PySide6.QtWidgets import (QDialog, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QMessageBox, QProgressBar, QPushButton, QSlider, QSpinBox,
                               QSplitter, QVBoxLayout, QWidget)

from ..cekirdek import filtre, izleyici, kontur, nesne_bulucu
from .donusum import nesne_ikonu, qimage_den_numpy, saniye_metni
from .tuval import Tuval

VIDEO_FILTRESI = ("Videolar (*.mp4 *.mkv *.avi *.mov *.wmv *.webm *.m4v *.mpg *.mpeg *.flv *.3gp);;"
                  "Tüm dosyalar (*)")
KARE_EN_BUYUK_KENAR = 480           # algılama için kare bu boyuta indirilir
KARE_BEKLEME_MS = 2500              # bir kare bu sürede gelmezse atlanır


# --------------------------------------------------------------------------- 1. kare yakalama
class KareYakalayici(QObject):
    """Videodan eşit aralıklı kareler alır (olay döngüsüyle, bloklamadan)."""

    kare_geldi = Signal(float, object)        # saniye, rgb uint8
    ilerleme = Signal(int, int)               # alınan, toplam
    bitti = Signal()
    hata = Signal(str)

    def __init__(self, yol: str, aralik_sn: float, en_fazla: int, parent=None):
        super().__init__(parent)
        self._yol, self._aralik, self._en_fazla = yol, aralik_sn, en_fazla
        self._oynatici = QMediaPlayer(self)
        self._sink = QVideoSink(self)
        self._oynatici.setVideoSink(self._sink)
        self._oynatici.mediaStatusChanged.connect(self._durum_degisti)
        self._oynatici.errorOccurred.connect(self._oynatici_hatasi)
        self._sink.videoFrameChanged.connect(self._kare)
        self._zamanlar: list[int] = []        # ms
        self._sira = 0
        self._hedef_ms: int | None = None
        self._zaman_asimi = QTimer(self)
        self._zaman_asimi.setSingleShot(True)
        self._zaman_asimi.setInterval(KARE_BEKLEME_MS)
        self._zaman_asimi.timeout.connect(self._atla)
        self._bitti_mi = False

    def baslat(self) -> None:
        self._oynatici.setSource(QUrl.fromLocalFile(self._yol))

    def iptal(self) -> None:
        self._bitir(sinyal=False)

    def _durum_degisti(self, durum) -> None:
        if durum == QMediaPlayer.MediaStatus.LoadedMedia and not self._zamanlar:
            sure = self._oynatici.duration()
            if sure <= 0:
                self.hata.emit("Videonun süresi okunamadı.")
                return
            adim = max(1, int(self._aralik * 1000))
            self._zamanlar = list(range(0, max(1, sure - 50), adim))[: self._en_fazla]
            self._oynatici.pause()
            QTimer.singleShot(150, self._siradaki)
        elif durum == QMediaPlayer.MediaStatus.InvalidMedia:
            self.hata.emit("Video dosyası açılamadı ya da biçimi desteklenmiyor.")

    def _oynatici_hatasi(self, _kod, metin: str) -> None:
        self.hata.emit(f"Video oynatıcı hatası: {metin}")

    def _siradaki(self) -> None:
        if self._bitti_mi:
            return
        if self._sira >= len(self._zamanlar):
            self._bitir()
            return
        self._hedef_ms = self._zamanlar[self._sira]
        self._zaman_asimi.start()
        self._oynatici.setPosition(self._hedef_ms)

    def _kare(self, kare) -> None:
        if self._hedef_ms is None or not kare.isValid():
            return
        baslangic = kare.startTime()                 # mikrosaniye; bilinmiyorsa -1
        if baslangic >= 0 and baslangic / 1000.0 < self._hedef_ms - 120:
            return                                   # konumlanmadan önceki eski kare
        img = kare.toImage()
        if img.isNull():
            return
        rgb = qimage_den_numpy(img)
        oran = KARE_EN_BUYUK_KENAR / max(rgb.shape[:2])
        if oran < 1:
            kucuk = filtre.boyutlandir(rgb.astype(np.float32), round(rgb.shape[0] * oran), round(rgb.shape[1] * oran))
            rgb = np.clip(kucuk + 0.5, 0, 255).astype(np.uint8)
        self.kare_geldi.emit(self._hedef_ms / 1000.0, rgb)
        self._ilerlet()

    def _atla(self) -> None:
        self._ilerlet()

    def _ilerlet(self) -> None:
        self._zaman_asimi.stop()
        self._hedef_ms = None
        self._sira += 1
        self.ilerleme.emit(self._sira, len(self._zamanlar))
        QTimer.singleShot(0, self._siradaki)

    def _bitir(self, sinyal: bool = True) -> None:
        self._bitti_mi = True
        self._zaman_asimi.stop()
        self._oynatici.stop()
        self._oynatici.setSource(QUrl())
        if sinyal:
            self.bitti.emit()


# --------------------------------------------------------------------------- 2. algılama
class VideoAnalizi(QThread):
    """Yakalanan karelerin hepsinde nesne bulur, sonra izleri çıkarır."""

    ilerleme = Signal(int, int)
    bitti = Signal(object)           # list[Iz]
    hata = Signal(str)

    def __init__(self, kareler: list[tuple[float, np.ndarray]], duyarlilik: float, parent=None):
        super().__init__(parent)
        self._kareler, self._esik = kareler, 0.8 - 0.6 * duyarlilik

    def run(self) -> None:
        import traceback
        try:
            gozlemler = []
            for i, (zaman, rgb) in enumerate(self._kareler):
                analiz = nesne_bulucu.analiz_et(rgb, goreli_esik=self._esik)
                gozlemler.append([izleyici.gozlem_olustur(n, i, zaman, rgb.shape[:2]) for n in analiz.nesneler])
                self.ilerleme.emit(i + 1, len(self._kareler))
            self.bitti.emit(izleyici.izle(gozlemler))
        except Exception:                                      # noqa: BLE001
            self.hata.emit(traceback.format_exc())


# --------------------------------------------------------------------------- 3. pencere
class VideoPenceresi(QDialog):
    def __init__(self, parent=None, duyarlilik: float = 0.58):
        super().__init__(parent)
        self.setWindowTitle("Videodaki nesneler")
        self.resize(1020, 660)
        self._duyarlilik = duyarlilik
        self._yol: str | None = None
        self._kareler: list[tuple[float, np.ndarray]] = []
        self._izler: list[izleyici.Iz] = []
        self._yakalayici: KareYakalayici | None = None
        self._analiz: VideoAnalizi | None = None

        # --- üst çubuk
        self._ac = QPushButton("Video aç…")
        self._ac.clicked.connect(self._video_sec)
        self._yol_etiketi = QLabel("Henüz video seçilmedi")
        self._yol_etiketi.setStyleSheet("color: #9a9aa2;")
        self._aralik = QDoubleSpinBox()
        self._aralik.setRange(0.2, 30.0)
        self._aralik.setValue(1.0)
        self._aralik.setSingleStep(0.5)
        self._aralik.setSuffix(" sn")
        self._aralik.setToolTip("Kaç saniyede bir kare incelensin")
        self._en_fazla = QSpinBox()
        self._en_fazla.setRange(3, 300)
        self._en_fazla.setValue(60)
        self._en_fazla.setToolTip("İncelenecek en çok kare sayısı (uzun videolarda süreyi sınırlar)")
        self._basla = QPushButton("Nesneleri bul")
        self._basla.setEnabled(False)
        self._basla.clicked.connect(self._baslat)
        self._csv = QPushButton("Listeyi CSV olarak kaydet")
        self._csv.setEnabled(False)
        self._csv.clicked.connect(self._csv_kaydet)

        ust = QHBoxLayout()
        ust.addWidget(self._ac)
        ust.addWidget(self._yol_etiketi, 1)
        ust.addWidget(QLabel("Aralık"))
        ust.addWidget(self._aralik)
        ust.addWidget(QLabel("En fazla kare"))
        ust.addWidget(self._en_fazla)
        ust.addWidget(self._basla)

        self._ilerleme = QProgressBar()
        self._ilerleme.setTextVisible(True)
        self._ilerleme.hide()
        self._durum = QLabel("")

        # --- liste + önizleme
        self._liste = QListWidget()
        self._liste.setIconSize(QSize(64, 64))
        self._liste.setMinimumWidth(300)
        self._liste.currentRowChanged.connect(self._iz_secildi)
        self._tuval = Tuval()
        self._kaydirici = QSlider(Qt.Orientation.Horizontal)
        self._kaydirici.setEnabled(False)
        self._kaydirici.valueChanged.connect(self._gozlem_goster)
        self._kare_etiketi = QLabel("")
        alt_sag = QHBoxLayout()
        alt_sag.addWidget(QLabel("Nesnenin kareleri"))
        alt_sag.addWidget(self._kaydirici, 1)
        alt_sag.addWidget(self._kare_etiketi)
        sag = QWidget()
        sag_d = QVBoxLayout(sag)
        sag_d.setContentsMargins(0, 0, 0, 0)
        sag_d.addWidget(self._tuval, 1)
        sag_d.addLayout(alt_sag)

        bolucu = QSplitter()
        bolucu.addWidget(self._liste)
        bolucu.addWidget(sag)
        bolucu.setStretchFactor(1, 1)

        alt = QHBoxLayout()
        alt.addWidget(self._durum, 1)
        alt.addWidget(self._csv)

        d = QVBoxLayout(self)
        d.addLayout(ust)
        d.addWidget(self._ilerleme)
        d.addWidget(bolucu, 1)
        d.addLayout(alt)

    # ------------------------------------------------------------------ akış
    def duyarliligi_ayarla(self, d: float) -> None:
        self._duyarlilik = d

    def video_ac(self, yol: str) -> None:
        self._yol = yol
        self._yol_etiketi.setText(Path(yol).name)
        self._basla.setEnabled(True)
        self._durum.setText("Hazır. 'Nesneleri bul' ile başlayın.")

    def _video_sec(self) -> None:
        yol, _ = QFileDialog.getOpenFileName(self, "Video seç", "", VIDEO_FILTRESI)
        if yol:
            self.video_ac(yol)

    def _baslat(self) -> None:
        if self._yol is None:
            return
        self._liste.clear()
        self._izler, self._kareler = [], []
        self._tuval.goruntu_ayarla(None)
        self._csv.setEnabled(False)
        self._basla.setEnabled(False)
        self._ac.setEnabled(False)
        self._ilerleme.setRange(0, 0)               # belirsiz: süre henüz bilinmiyor
        self._ilerleme.show()
        self._durum.setText("Kareler okunuyor…")

        self._yakalayici = KareYakalayici(self._yol, self._aralik.value(), self._en_fazla.value(), self)
        self._yakalayici.kare_geldi.connect(lambda t, rgb: self._kareler.append((t, rgb)))
        self._yakalayici.ilerleme.connect(self._yakalama_ilerlemesi)
        self._yakalayici.bitti.connect(self._yakalama_bitti)
        self._yakalayici.hata.connect(self._hata)
        self._yakalayici.baslat()

    def _yakalama_ilerlemesi(self, sira: int, toplam: int) -> None:
        self._ilerleme.setRange(0, toplam)
        self._ilerleme.setValue(sira)
        self._ilerleme.setFormat(f"Kare okunuyor %v / %m")

    def _yakalama_bitti(self) -> None:
        if not self._kareler:
            self._hata("Videodan hiç kare alınamadı. Biçim bu bilgisayarda desteklenmiyor olabilir.")
            return
        self._durum.setText(f"{len(self._kareler)} kare okundu. Nesneler aranıyor…")
        self._analiz = VideoAnalizi(self._kareler, self._duyarlilik, self)
        self._analiz.ilerleme.connect(self._analiz_ilerlemesi)
        self._analiz.bitti.connect(self._analiz_bitti)
        self._analiz.hata.connect(self._hata)
        self._analiz.finished.connect(self._analiz.deleteLater)
        self._analiz.start()

    def _analiz_ilerlemesi(self, sira: int, toplam: int) -> None:
        self._ilerleme.setRange(0, toplam)
        self._ilerleme.setValue(sira)
        self._ilerleme.setFormat("Kare inceleniyor %v / %m")

    def _analiz_bitti(self, izler: list) -> None:
        self._izler = izler
        self._ilerleme.hide()
        self._ac.setEnabled(True)
        self._basla.setEnabled(True)
        self._csv.setEnabled(bool(izler))
        for iz in izler:
            g = iz.en_iyi
            kare_rgb = self._kareler[g.kare][1]
            metin = (f"Nesne {iz.kimlik}\n{saniye_metni(iz.ilk_zaman)} – {saniye_metni(iz.son_zaman)}"
                     f" · {iz.kare_sayisi} karede · %{iz.ortalama_alan * 100:.0f} alan")
            self._liste.addItem(QListWidgetItem(nesne_ikonu(kare_rgb, g.maske, g.kutu, 64), metin))
        self._durum.setText(f"{len(izler)} farklı nesne bulundu ({len(self._kareler)} karede).")
        if izler:
            self._liste.setCurrentRow(0)

    def _hata(self, metin: str) -> None:
        if self._yakalayici is not None:
            self._yakalayici.iptal()
        self._ilerleme.hide()
        self._ac.setEnabled(True)
        self._basla.setEnabled(self._yol is not None)
        self._durum.setText("Hata oluştu.")
        QMessageBox.warning(self, "Video", metin[-1500:])

    # ------------------------------------------------------------------ gösterim
    def _iz_secildi(self, satir: int) -> None:
        if not 0 <= satir < len(self._izler):
            return
        iz = self._izler[satir]
        en_iyi = iz.gozlemler.index(iz.en_iyi)
        self._kaydirici.blockSignals(True)
        self._kaydirici.setRange(0, len(iz.gozlemler) - 1)
        self._kaydirici.setValue(en_iyi)
        self._kaydirici.blockSignals(False)
        self._kaydirici.setEnabled(len(iz.gozlemler) > 1)
        self._gozlem_goster(en_iyi)

    def _gozlem_goster(self, indeks: int) -> None:
        satir = self._liste.currentRow()
        if not 0 <= satir < len(self._izler):
            return
        g = self._izler[satir].gozlemler[indeks]
        self._tuval.goruntu_ayarla(self._kareler[g.kare][1])
        self._tuval.kontur_ayarla(kontur.konturlar(g.maske))
        self._kare_etiketi.setText(f"{saniye_metni(g.zaman)}  ({indeks + 1}/{len(self._izler[satir].gozlemler)})")

    # ------------------------------------------------------------------ dışa aktarma
    def _csv_kaydet(self) -> None:
        yol, _ = QFileDialog.getSaveFileName(self, "Listeyi kaydet", "video_nesneleri.csv", "CSV (*.csv)")
        if yol:
            self.csv_yaz(yol)
            self._durum.setText(f"Kaydedildi: {Path(yol).name}")

    def csv_yaz(self, yol: str | Path) -> None:
        """Bulunan nesnelerin özetini CSV'ye yazar (noktalı virgül ayraçlı, Excel Türkçe uyumlu)."""
        with open(yol, "w", newline="", encoding="utf-8-sig") as f:     # BOM: Excel Türkçe karakterleri doğru açar
            w = csv.writer(f, delimiter=";")
            w.writerow(["nesne", "ilk_saniye", "son_saniye", "kare_sayisi", "ortalama_alan_yuzde",
                        "ortalama_L", "ortalama_a", "ortalama_b"])
            for iz in self._izler:
                lab = np.mean([g.lab for g in iz.gozlemler], axis=0)
                w.writerow([f"Nesne {iz.kimlik}", f"{iz.ilk_zaman:.2f}", f"{iz.son_zaman:.2f}", iz.kare_sayisi,
                            f"{iz.ortalama_alan * 100:.1f}", f"{lab[0]:.1f}", f"{lab[1]:.1f}", f"{lab[2]:.1f}"])

    def closeEvent(self, e) -> None:
        if self._yakalayici is not None:
            self._yakalayici.iptal()
        super().closeEvent(e)
