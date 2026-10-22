"""Testlerde kullanılan yapay sahne üreticileri (gerçek fotoğraf gerekmeden deneme için)."""
from __future__ import annotations

import numpy as np


def sahne(h: int = 360, w: int = 480, tohum: int = 1,
          nesne_ciz: bool = True) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Gradyanlı, gürültülü bir zemin üstünde üç nesne çizer.

    Dönen: (rgb float32, {ad: gerçek_maske}). Gerçek maskeler bölütleme başarısını ölçer.
    """
    rng = np.random.default_rng(tohum)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Zemin: üstte açık, altta koyu mavi-gri geçiş + ince doku
    zemin = np.zeros((h, w, 3), np.float32)
    zemin[..., 0] = 0.55 - 0.15 * yy / h
    zemin[..., 1] = 0.62 - 0.17 * yy / h
    zemin[..., 2] = 0.72 - 0.20 * yy / h
    doku = np.sin(xx * 0.35) * np.sin(yy * 0.31) * 0.012
    zemin += doku[..., None]

    img = zemin.copy()
    maskeler: dict[str, np.ndarray] = {}
    cizim_oncesi = img.copy()

    # Kırmızı daire (gölgelendirilmiş)
    cy, cx, r = h * 0.40, w * 0.28, h * 0.17
    daire = (yy - cy) ** 2 + (xx - cx) ** 2 < r * r
    isik = 1.0 - 0.45 * np.clip(((xx - cx) + (yy - cy)) / (2 * r) * 0.5 + 0.5, 0, 1)
    img[daire] = (np.array([0.80, 0.12, 0.10], np.float32) * isik[..., None])[daire]
    maskeler["daire"] = daire

    # Sarı dikdörtgen
    dik = np.zeros((h, w), bool)
    dik[int(h * 0.55):int(h * 0.88), int(w * 0.55):int(w * 0.80)] = True
    img[dik] = np.array([0.92, 0.78, 0.15], np.float32) * (0.9 + 0.1 * np.sin(xx[dik] * 0.05))[:, None]
    maskeler["dikdortgen"] = dik

    # Koyu yeşil elips
    ey, ex = h * 0.28, w * 0.72
    elips = ((yy - ey) / (h * 0.12)) ** 2 + ((xx - ex) / (w * 0.13)) ** 2 < 1
    img[elips] = np.array([0.10, 0.45, 0.22], np.float32)
    maskeler["elips"] = elips

    if not nesne_ciz:
        img = cizim_oncesi
    img += rng.normal(0, 0.01, img.shape).astype(np.float32)
    return np.clip(img, 0, 1), maskeler


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Kesişim / birleşim: iki maskenin ne kadar örtüştüğü (1 = birebir)."""
    return float((a & b).sum() / max(1, (a | b).sum()))
