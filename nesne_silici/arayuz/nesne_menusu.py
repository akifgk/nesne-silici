"""Köşede sabit duran nesne menüsü.

Tuval'in çocuğudur ve her yeniden boyutlandırmada sağ üst köşeye yaslanır. Görüntü
kaydırılıp yakınlaştırılsa bile yerinde durur. Fare olaylarını kendi tüketir, yani menüye
tıklamak altındaki görüntüde nesne seçimi tetiklemez.
"""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton,
                               QSlider, QVBoxLayout, QWidget)

STIL = """
#NesneMenusu { background: rgba(32, 33, 38, 235); border: 1px solid #55565c; border-radius: 10px; }
#NesneMenusu QLabel { color: #e6e6ea; }
#NesneMenusu QLabel#baslik { font-weight: 600; font-size: 13px; }
#NesneMenusu QLabel#ipucu { color: #9a9aa2; font-size: 11px; }
#NesneMenusu QListWidget { background: transparent; border: none; color: #e6e6ea; outline: 0; }
#NesneMenusu QListWidget::item { padding: 4px; border-radius: 6px; }
#NesneMenusu QListWidget::item:hover { background: rgba(255,255,255,18); }
#NesneMenusu QListWidget::item:selected { background: rgba(255,140,0,70); color: white; }
#NesneMenusu QPushButton { background: #3a3b42; color: #e6e6ea; border: 1px solid #55565c;
                           border-radius: 6px; padding: 5px 8px; }
#NesneMenusu QPushButton:hover { background: #46474f; }
#NesneMenusu QPushButton:disabled { color: #77777f; background: #2d2e33; }
#NesneMenusu QPushButton#sil { background: #8a2f1f; border-color: #b04a37; }
#NesneMenusu QPushButton#sil:hover { background: #a23a28; }
#NesneMenusu QPushButton#sil:disabled { background: #3a2a27; color: #86706b; border-color: #4a3a36; }
"""


class NesneMenusu(QFrame):
    secildi = Signal(int)               # satır numarası (-1: seçim yok)
    sil_istendi = Signal()
    genislet_istendi = Signal()
    daralt_istendi = Signal()
    yeniden_algila = Signal(float)      # duyarlılık 0..1

    GENISLIK = 270
    KENAR_BOSLUGU = 12

    def __init__(self, tuval: QWidget):
        super().__init__(tuval)
        self.setObjectName("NesneMenusu")
        self.setStyleSheet(STIL)
        self.setFixedWidth(self.GENISLIK)
        self._tuval = tuval

        self._baslik = QLabel("Nesneler")
        self._baslik.setObjectName("baslik")
        self._ipucu = QLabel("Listeden seç ya da görüntüde bir yere tıkla.")
        self._ipucu.setObjectName("ipucu")
        self._ipucu.setWordWrap(True)

        self._liste = QListWidget()
        self._liste.setIconSize(QSize(52, 52))
        self._liste.setMinimumHeight(150)
        self._liste.currentRowChanged.connect(self.secildi.emit)

        self._daralt = QPushButton("Daralt")
        self._daralt.setToolTip("Seçili nesneyi bir alt bölgeye küçült (Ctrl+-)")
        self._genislet = QPushButton("Genişlet")
        self._genislet.setToolTip("Seçili nesneyi komşu bölgeyle birleştirip büyüt (Ctrl++)")
        self._sil = QPushButton("Seçileni sil")
        self._sil.setObjectName("sil")
        self._sil.setToolTip("Nesneyi sil ve arkasını çevreye uygun doldur (Del)")
        self._daralt.clicked.connect(self.daralt_istendi.emit)
        self._genislet.clicked.connect(self.genislet_istendi.emit)
        self._sil.clicked.connect(self.sil_istendi.emit)

        self._duyarlilik = QSlider(Qt.Orientation.Horizontal)
        self._duyarlilik.setRange(0, 100)
        self._duyarlilik.setValue(58)
        self._duyarlilik.setToolTip("Yüksek: silik nesneleri de listeler. Düşük: yalnızca belirgin olanlar.")
        self._yenile = QPushButton("Yeniden algıla")
        self._yenile.clicked.connect(lambda: self.yeniden_algila.emit(self._duyarlilik.value() / 100.0))

        satir1 = QHBoxLayout()
        satir1.addWidget(self._daralt)
        satir1.addWidget(self._genislet)
        satir2 = QHBoxLayout()
        satir2.addWidget(QLabel("Duyarlılık"))
        satir2.addWidget(self._duyarlilik, 1)

        d = QVBoxLayout(self)
        d.setContentsMargins(12, 10, 12, 12)
        d.setSpacing(8)
        d.addWidget(self._baslik)
        d.addWidget(self._ipucu)
        d.addWidget(self._liste, 1)
        d.addLayout(satir1)
        d.addWidget(self._sil)
        d.addLayout(satir2)
        d.addWidget(self._yenile)

        self.secim_durumunu_guncelle()
        self.kosesine_yerles()

    # ------------------------------------------------------------------ dışarıdan
    def duyarlilik(self) -> float:
        return self._duyarlilik.value() / 100.0

    def liste_doldur(self, ogeler: list[tuple[str, QIcon]], secili: int = -1) -> None:
        """``ogeler``: (metin, ikon) çiftleri."""
        self._liste.blockSignals(True)
        self._liste.clear()
        for metin, ikon in ogeler:
            self._liste.addItem(QListWidgetItem(ikon, metin))
        self._liste.setCurrentRow(secili)
        self._liste.blockSignals(False)
        self._baslik.setText(f"Nesneler ({len(ogeler)})")
        self.secim_durumunu_guncelle()

    def secimi_ayarla(self, satir: int) -> None:
        self._liste.blockSignals(True)
        self._liste.setCurrentRow(satir)
        self._liste.blockSignals(False)
        self.secim_durumunu_guncelle()

    def secim_durumunu_guncelle(self) -> None:
        var = self._liste.currentRow() >= 0
        for d in (self._daralt, self._genislet, self._sil):
            d.setEnabled(var)

    def mesgul_ayarla(self, mesgul: bool) -> None:
        self.setEnabled(not mesgul)

    # ------------------------------------------------------------------ konum
    def kosesine_yerles(self) -> None:
        """Sağ üst köşeye yaslan; yükseklik tuvalin yüksekliğini aşmaz."""
        ust = self._tuval.height() - 2 * self.KENAR_BOSLUGU
        self.setFixedHeight(max(300, min(520, ust)))
        self.move(self._tuval.width() - self.width() - self.KENAR_BOSLUGU, self.KENAR_BOSLUGU)
        self.raise_()
