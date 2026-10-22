"""numpy <-> Qt dönüşümleri ve küçük resim üretimi."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QImage, QPixmap

from ..cekirdek import filtre


def numpy_den_qimage(rgb_u8: np.ndarray) -> QImage:
    """uint8 (h, w, 3) -> QImage. Kopyalanır: numpy dizisi silinse de görüntü geçerli kalır."""
    rgb = np.ascontiguousarray(rgb_u8)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def rgba_den_qimage(rgb: np.ndarray, alfa: np.ndarray) -> QImage:
    """float RGB (0..1) + float alfa (0..1) -> önceden çarpılmış RGBA QImage."""
    a = np.clip(alfa, 0, 1)[..., None]
    dizi = np.ascontiguousarray(np.dstack([np.clip(rgb, 0, 1) * a, a]) * 255.0 + 0.5).astype(np.uint8)
    h, w = dizi.shape[:2]
    return QImage(dizi.data, w, h, 4 * w, QImage.Format.Format_RGBA8888_Premultiplied).copy()


def qimage_den_numpy(img: QImage) -> np.ndarray:
    """QImage -> uint8 (h, w, 3) RGB."""
    img = img.convertToFormat(QImage.Format.Format_RGB888)
    w, h, adim = img.width(), img.height(), img.bytesPerLine()
    ham = np.frombuffer(img.constBits(), dtype=np.uint8, count=adim * h).reshape(h, adim)
    return np.array(ham[:, :w * 3].reshape(h, w, 3))      # kopya: QImage belleği fonksiyon bitince serbest kalır


def nesne_ikonu(rgb_u8: np.ndarray, maske: np.ndarray, kutu: tuple[int, int, int, int],
                boyut: int = 56) -> QIcon:
    """Nesnenin küçük önizlemesi: dışı koyulaştırılmış, nesne olduğu gibi."""
    y0, x0, y1, x1 = kutu
    pay = max(2, int(0.08 * max(y1 - y0, x1 - x0)))
    y0, x0 = max(0, y0 - pay), max(0, x0 - pay)
    y1, x1 = min(rgb_u8.shape[0], y1 + pay), min(rgb_u8.shape[1], x1 + pay)
    parca = rgb_u8[y0:y1, x0:x1].astype(np.float32)
    m = maske[y0:y1, x0:x1][..., None]
    parca = np.where(m, parca, parca * 0.25 + 12.0)
    oran = boyut / max(parca.shape[:2])
    kucuk = filtre.boyutlandir(parca, max(1, round(parca.shape[0] * oran)), max(1, round(parca.shape[1] * oran)))
    return QIcon(QPixmap.fromImage(numpy_den_qimage(np.clip(kucuk, 0, 255).astype(np.uint8))))


def saniye_metni(s: float) -> str:
    """72.4 -> '1:12'"""
    s = int(round(s))
    return f"{s // 60}:{s % 60:02d}"


__all__ = ["numpy_den_qimage", "rgba_den_qimage", "qimage_den_numpy", "nesne_ikonu", "saniye_metni", "Qt"]
