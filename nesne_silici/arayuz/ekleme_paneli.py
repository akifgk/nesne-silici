"""Görsel eklerken görünen yüzen panel: boyut, uyum ayarları, uygula/iptal."""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (QCheckBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QSlider,
                               QVBoxLayout, QWidget)

from ..cekirdek.uyumlama import UyumAyarlari

STIL = """
#EklemePaneli { background: rgba(32, 33, 38, 235); border: 1px solid #55565c; border-radius: 10px; }
#EklemePaneli QLabel { color: #e6e6ea; }
#EklemePaneli QLabel#baslik { font-weight: 600; font-size: 13px; }
#EklemePaneli QLabel#ipucu { color: #9a9aa2; font-size: 11px; }
#EklemePaneli QCheckBox { color: #e6e6ea; }
#EklemePaneli QPushButton { background: #3a3b42; color: #e6e6ea; border: 1px solid #55565c;
                            border-radius: 6px; padding: 6px 10px; }
#EklemePaneli QPushButton:hover { background: #46474f; }
#EklemePaneli QPushButton#uygula { background: #1f6f3d; border-color: #2f9a58; }
#EklemePaneli QPushButton#uygula:hover { background: #26864a; }
"""

# (anahtar, etiket, en küçük, en büyük, başlangıç, açıklama)  değerler yüzde/10'luk birim
_KAYDIRICILAR = (
    ("renk", "Renk uyumu", 0, 100, 80, "Nesnenin renk ve parlaklığını çevreye yaklaştırır."),
    ("isik", "Işık yönü", 0, 100, 80, "Sahnenin ışık yönüne göre nesneye parlaklık gradyanı uygular."),
    ("gren", "Gren", 0, 100, 100, "Fotoğraftaki gürültü/gren düzeyini nesneye ekler."),
    ("golge", "Gölge", 0, 100, 50, "Işığın ters yönünde yumuşak gölge ve temas gölgesi."),
    ("kenar", "Kenar", 0, 40, 10, "Nesne kenarını yumuşatır, eski arka plan saçağını siler."),
)


class EklemePaneli(QFrame):
    ayar_degisti = Signal()                 # sürükleme sırasında değil, kısa gecikmeyle yayınlanır
    boyut_degisti = Signal(float)           # ölçek (1.0 = %100)
    arka_plan_degisti = Signal(bool)
    uygula = Signal()
    iptal = Signal()

    GENISLIK = 330
    KENAR_BOSLUGU = 12

    def __init__(self, tuval: QWidget):
        super().__init__(tuval)
        self.setObjectName("EklemePaneli")
        self.setStyleSheet(STIL)
        self.setFixedWidth(self.GENISLIK)
        self._tuval = tuval
        self._gecikme = QTimer(self)
        self._gecikme.setSingleShot(True)
        self._gecikme.setInterval(160)
        self._gecikme.timeout.connect(self.ayar_degisti.emit)

        baslik = QLabel("Görsel ekle")
        baslik.setObjectName("baslik")
        ipucu = QLabel("Sürükleyerek konumlandır. Ctrl + fare tekerleği ile boyutlandır.")
        ipucu.setObjectName("ipucu")
        ipucu.setWordWrap(True)

        self._boyut = QSlider(Qt.Orientation.Horizontal)
        self._boyut.setRange(5, 400)
        self._boyut.setValue(100)
        self._boyut_metni = QLabel("%100")
        self._boyut.valueChanged.connect(self._boyut_kaydi)

        izgara = QGridLayout()
        izgara.setHorizontalSpacing(8)
        izgara.setColumnStretch(1, 1)
        izgara.setColumnMinimumWidth(2, 40)
        izgara.addWidget(QLabel("Boyut"), 0, 0)
        izgara.addWidget(self._boyut, 0, 1)
        izgara.addWidget(self._boyut_metni, 0, 2)
        self._kaydiricilar: dict[str, QSlider] = {}
        for sira, (anahtar, etiket, lo, hi, ilk, aciklama) in enumerate(_KAYDIRICILAR, start=1):
            k = QSlider(Qt.Orientation.Horizontal)
            k.setRange(lo, hi)
            k.setValue(ilk)
            k.setToolTip(aciklama)
            k.valueChanged.connect(lambda _v: self._gecikme.start())
            e = QLabel(etiket)
            e.setToolTip(aciklama)
            izgara.addWidget(e, sira, 0)
            izgara.addWidget(k, sira, 1, 1, 2)
            self._kaydiricilar[anahtar] = k

        self._arka_plan = QCheckBox("Arka planı otomatik temizle")
        self._arka_plan.setChecked(True)
        self._arka_plan.setToolTip("Düz renkli arka planı siler. Saydam PNG'lerde etkisizdir.")
        self._arka_plan.toggled.connect(self.arka_plan_degisti.emit)
        self._poisson = QCheckBox("Kenar renk karışımı (deneysel)")
        self._poisson.setToolTip("Poisson karıştırma: dikişi siler ama nesne rengini çevreye sızdırıp soldurabilir.")
        self._poisson.toggled.connect(lambda _v: self._gecikme.start())

        uygula = QPushButton("Uygula")
        uygula.setObjectName("uygula")
        uygula.clicked.connect(self.uygula.emit)
        iptal = QPushButton("İptal")
        iptal.clicked.connect(self.iptal.emit)
        dugmeler = QHBoxLayout()
        dugmeler.addWidget(uygula, 1)
        dugmeler.addWidget(iptal)

        d = QVBoxLayout(self)
        d.setContentsMargins(12, 10, 12, 12)
        d.setSpacing(8)
        d.addWidget(baslik)
        d.addWidget(ipucu)
        d.addLayout(izgara)
        d.addWidget(self._arka_plan)
        d.addWidget(self._poisson)
        d.addLayout(dugmeler)

        self.hide()

    # ------------------------------------------------------------------ dışarıdan
    def ayar(self) -> UyumAyarlari:
        k = self._kaydiricilar
        return UyumAyarlari(
            renk=k["renk"].value() / 100.0, isik=k["isik"].value() / 100.0,
            gren=k["gren"].value() / 100.0, golge=k["golge"].value() / 100.0,
            kenar=k["kenar"].value() / 10.0, poisson=self._poisson.isChecked())

    def arka_plan_temizle(self) -> bool:
        return self._arka_plan.isChecked()

    def arka_plan_secenegini_ayarla(self, etkin: bool) -> None:
        """Görselin kendi alfa kanalı varsa seçenek anlamsızdır."""
        self._arka_plan.setEnabled(etkin)

    def boyutu_ayarla(self, olcek: float) -> None:
        self._boyut.blockSignals(True)
        self._boyut.setValue(int(round(olcek * 100)))
        self._boyut.blockSignals(False)
        self._boyut_metni.setText(f"%{int(round(olcek * 100))}")

    def _boyut_kaydi(self, v: int) -> None:
        self._boyut_metni.setText(f"%{v}")
        self.boyut_degisti.emit(v / 100.0)

    def goster_ve_yerles(self) -> None:
        self.show()
        self.kosesine_yerles()

    def mesgul_ayarla(self, mesgul: bool) -> None:
        self.setEnabled(not mesgul)

    def kosesine_yerles(self) -> None:
        """Sol alt köşeye yaslan."""
        self.adjustSize()
        self.move(self.KENAR_BOSLUGU, self._tuval.height() - self.height() - self.KENAR_BOSLUGU)
        self.raise_()
