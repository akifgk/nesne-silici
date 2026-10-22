"""Tuval: görüntüyü gösterir; turuncu konturu ve yerleştirilen görseli çizer.

Koordinat sistemleri
--------------------
* **Görüntü koordinatı**: görüntünün kendi pikseli, (0, 0) sol üst.
* **Ekran koordinatı**: widget pikseli.

``ekran = görüntü * zoom + ofset``. Çizim sırasında painter'a bu dönüşüm bir kez
uygulanır; konturlar görüntü koordinatında verilir ve zoom'a göre ölçeklenir, ama çizgi
kalınlığı "kozmetik" olduğundan her zoom'da aynı kalır.
"""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPolygonF, QWheelEvent
from PySide6.QtWidgets import QWidget

from ..cekirdek.uyumlama import EklenenGorsel
from .donusum import numpy_den_qimage, rgba_den_qimage

TURUNCU = QColor(255, 140, 0)


class Tuval(QWidget):
    nokta_tiklandi = Signal(int, int)       # görüntü koordinatı
    gorsel_tasindi = Signal()               # sürükleme bırakıldı
    gorsel_olceklendi = Signal(float)       # yeni ölçek
    zoom_degisti = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(320, 240)
        self.setAcceptDrops(False)

        self._resim: QImage | None = None
        self._zoom = 1.0
        self._ofset = QPointF(0, 0)
        self._konturlar: list[QPolygonF] = []

        self._ekleme: EklenenGorsel | None = None
        self._ekleme_qimg: QImage | None = None        # ölçeklenmiş ham görsel (hızlı sürükleme için)
        self._onizleme: tuple[QPointF, QImage] | None = None   # uyumlanmış önizleme

        self._surukleme: str | None = None             # "pan" | "gorsel"
        self._son_fare = QPointF()
        self._bosluk_basili = False

    # ------------------------------------------------------------------ dışarıdan ayarlar
    def goruntu_ayarla(self, rgb_u8: np.ndarray | None, sifirla: bool = True) -> None:
        """Gösterilecek görüntüyü değiştirir. ``sifirla`` ise ekrana sığdırır."""
        self._resim = None if rgb_u8 is None else numpy_den_qimage(rgb_u8)
        if sifirla:
            self.sigdir()
        self.update()

    def kontur_ayarla(self, konturlar: list[np.ndarray] | None) -> None:
        self._konturlar = [QPolygonF([QPointF(float(x), float(y)) for x, y in k]) for k in (konturlar or [])]
        self.update()

    def ekleme_ayarla(self, gorsel: EklenenGorsel | None) -> None:
        """Yerleştirilecek görseli verir (None = yerleştirme modundan çık)."""
        self._ekleme = gorsel
        self._onizleme = None
        self._ekleme_qimg_yenile()
        self.update()

    def onizleme_ayarla(self, kutu: tuple[int, int, int, int] | None, rgb_u8: np.ndarray | None) -> None:
        """Uyumlanmış sonucu gösterir (kutu: y0, x0, y1, x1)."""
        if kutu is None or rgb_u8 is None:
            self._onizleme = None
        else:
            self._onizleme = (QPointF(kutu[1], kutu[0]), numpy_den_qimage(rgb_u8))
        self.update()

    def ekleme_boyutu_degisti(self) -> None:
        """Ölçek değiştiğinde ham görsel önizlemesi yeniden üretilir."""
        self._onizleme = None
        self._ekleme_qimg_yenile()
        self.update()

    def _ekleme_qimg_yenile(self) -> None:
        if self._ekleme is None:
            self._ekleme_qimg = None
            return
        rgb, alfa = self._ekleme.olcekli()
        self._ekleme_qimg = rgba_den_qimage(rgb, alfa)

    # ------------------------------------------------------------------ görünüm
    @property
    def zoom(self) -> float:
        return self._zoom

    def sigdir(self) -> None:
        if self._resim is None:
            return
        pay = 24
        w, h = max(1, self.width() - pay), max(1, self.height() - pay)
        self._zoom = min(w / self._resim.width(), h / self._resim.height(), 4.0)   # küçük resimleri en çok 4x büyüt
        self._ortala()
        self.zoom_degisti.emit(self._zoom)
        self.update()

    def yuzde_yuz(self) -> None:
        self._yakinlastir(1.0, QPointF(self.width() / 2, self.height() / 2), mutlak=True)

    def _ortala(self) -> None:
        if self._resim is None:
            return
        self._ofset = QPointF((self.width() - self._resim.width() * self._zoom) / 2,
                              (self.height() - self._resim.height() * self._zoom) / 2)

    def _yakinlastir(self, deger: float, odak: QPointF, mutlak: bool = False) -> None:
        """``odak`` (ekran noktası) yerinde kalacak şekilde zoom değiştirir."""
        yeni = float(np.clip(deger if mutlak else self._zoom * deger, 0.02, 64.0))
        goruntu_noktasi = (odak - self._ofset) / self._zoom
        self._zoom = yeni
        self._ofset = odak - goruntu_noktasi * yeni
        self.zoom_degisti.emit(self._zoom)
        self.update()

    def ekran_den_goruntuye(self, p: QPointF) -> QPointF:
        return (p - self._ofset) / self._zoom

    def _gorsel_dikdortgeni(self) -> QRectF | None:
        """Yerleştirilen görselin görüntü koordinatındaki dikdörtgeni."""
        if self._ekleme is None or self._ekleme_qimg is None:
            return None
        return QRectF(self._ekleme.x, self._ekleme.y, self._ekleme_qimg.width(), self._ekleme_qimg.height())

    # ------------------------------------------------------------------ çizim
    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(30, 31, 34))
        if self._resim is None:
            p.setPen(QColor(150, 150, 155))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "Bir görüntü açmak için Dosya > Aç (Ctrl+O)\nya da dosyayı buraya sürükleyin")
            return
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self._zoom < 2.0)
        p.translate(self._ofset)
        p.scale(self._zoom, self._zoom)
        p.drawImage(0, 0, self._resim)

        # Yerleştirilen görsel: uyumlanmış önizleme varsa o, yoksa ham görsel
        if self._ekleme is not None:
            if self._onizleme is not None and self._surukleme != "gorsel":
                p.drawImage(self._onizleme[0], self._onizleme[1])
            elif self._ekleme_qimg is not None:
                p.drawImage(QPointF(self._ekleme.x, self._ekleme.y), self._ekleme_qimg)
            r = self._gorsel_dikdortgeni()
            if r is not None:
                kalem = QPen(QColor(120, 190, 255), 1.5, Qt.PenStyle.DashLine)
                kalem.setCosmetic(True)
                p.setPen(kalem)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRect(r)

        # Turuncu kontur: altta açık bir hale, üstte turuncu çizgi (her zeminde seçilir)
        if self._konturlar:
            hale = QPen(QColor(255, 255, 255, 110), 5.0)
            hale.setCosmetic(True)
            hale.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            cizgi = QPen(TURUNCU, 2.6)
            cizgi.setCosmetic(True)
            cizgi.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            p.setBrush(QBrush(QColor(255, 140, 0, 18)))
            for poli in self._konturlar:
                p.setPen(hale)
                p.drawPolygon(poli)
            for poli in self._konturlar:
                p.setPen(cizgi)
                p.drawPolygon(poli)

    # ------------------------------------------------------------------ fare / klavye
    def wheelEvent(self, e: QWheelEvent) -> None:
        adim = e.angleDelta().y() / 120.0
        if adim == 0:
            return
        if (e.modifiers() & Qt.KeyboardModifier.ControlModifier) and self._ekleme is not None:
            self.gorsel_olceklendi.emit(self._ekleme.olcek * (1.06 ** adim))
        else:
            self._yakinlastir(1.15 ** adim, e.position())

    def mousePressEvent(self, e) -> None:
        self.setFocus()
        self._son_fare = e.position()
        sol = e.button() == Qt.MouseButton.LeftButton
        pan = e.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton) or \
            (sol and self._bosluk_basili)
        if pan:
            self._surukleme = "pan"
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif sol and self._resim is not None:
            nokta = self.ekran_den_goruntuye(e.position())
            r = self._gorsel_dikdortgeni()
            if r is not None and r.contains(nokta):
                self._surukleme = "gorsel"
                self.setCursor(Qt.CursorShape.SizeAllCursor)
            elif r is None and 0 <= nokta.x() < self._resim.width() and 0 <= nokta.y() < self._resim.height():
                self.nokta_tiklandi.emit(int(nokta.x()), int(nokta.y()))

    def mouseMoveEvent(self, e) -> None:
        fark = e.position() - self._son_fare
        self._son_fare = e.position()
        if self._surukleme == "pan":
            self._ofset += fark
            self.update()
        elif self._surukleme == "gorsel" and self._ekleme is not None:
            self._ekleme.x += fark.x() / self._zoom
            self._ekleme.y += fark.y() / self._zoom
            self.update()

    def mouseReleaseEvent(self, _e) -> None:
        onceki = self._surukleme
        self._surukleme = None
        self.unsetCursor()
        if onceki == "gorsel":
            self.gorsel_tasindi.emit()
        self.update()

    def keyPressEvent(self, e) -> None:
        if e.key() == Qt.Key.Key_Space:
            self._bosluk_basili = True
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            super().keyPressEvent(e)

    def keyReleaseEvent(self, e) -> None:
        if e.key() == Qt.Key.Key_Space:
            self._bosluk_basili = False
            self.unsetCursor()
        else:
            super().keyReleaseEvent(e)

    def resizeEvent(self, e) -> None:
        # Köşede sabit duran yüzen paneller (nesne menüsü, ekleme paneli) yeniden konumlanır.
        for cocuk in self.children():
            yerlestir = getattr(cocuk, "kosesine_yerles", None)
            if callable(yerlestir):
                yerlestir()
        super().resizeEvent(e)
