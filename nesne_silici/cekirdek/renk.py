"""Renk uzayı dönüşümleri: sRGB <-> CIE Lab.

Neden Lab? RGB'de iki rengin sayısal uzaklığı gözün "bu iki renk ne kadar farklı"
algısıyla uyuşmaz. Lab uzayında ise öklid uzaklığı algıya yakındır:

* ``L``: parlaklık (0 = siyah, 100 = beyaz)
* ``a``: yeşil(-) <-> kırmızı(+)
* ``b``: mavi(-) <-> sarı(+)

Bölütleme, renk uyumlama ve doldurma algoritmalarının hepsi benzerliği Lab'de ölçer.
"""
from __future__ import annotations

import numpy as np

# D65 referans beyazı ve sRGB <-> XYZ matrisleri (standart değerler).
_BEYAZ = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
_RGB_XYZ = np.array(
    [[0.4124564, 0.3575761, 0.1804375],
     [0.2126729, 0.7151522, 0.0721750],
     [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
_XYZ_RGB = np.linalg.inv(_RGB_XYZ).astype(np.float32)

_DELTA = 6.0 / 29.0


def _dogrusala(c: np.ndarray) -> np.ndarray:
    """sRGB'deki gama eğrisini kaldırır; ışık şiddetiyle orantılı değer verir."""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _gamaya_cevir(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def _f(t: np.ndarray) -> np.ndarray:
    return np.where(t > _DELTA ** 3, np.cbrt(t), t / (3 * _DELTA ** 2) + 4.0 / 29.0)


def _f_ters(t: np.ndarray) -> np.ndarray:
    return np.where(t > _DELTA, t ** 3, 3 * _DELTA ** 2 * (t - 4.0 / 29.0))


def rgb_den_lab(rgb: np.ndarray) -> np.ndarray:
    """RGB (0..1) -> Lab. Şekil korunur."""
    xyz = _dogrusala(rgb.astype(np.float32)) @ _RGB_XYZ.T
    f = _f(xyz / _BEYAZ)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1).astype(np.float32)


def lab_den_rgb(lab: np.ndarray) -> np.ndarray:
    """Lab -> RGB (0..1, sınırlara kırpılmış)."""
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = fy + lab[..., 1] / 500.0
    fz = fy - lab[..., 2] / 200.0
    xyz = _f_ters(np.stack([fx, fy, fz], axis=-1)) * _BEYAZ
    return _gamaya_cevir(xyz @ _XYZ_RGB.T).astype(np.float32)


def parlaklik(rgb: np.ndarray) -> np.ndarray:
    """Algısal parlaklık (0..1). Hızlı gri tonlama için."""
    return rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
