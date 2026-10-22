"""Kaba (düşük çözünürlüklü) maskeyi tam çözünürlükte gerçek kenara oturtur.

Bölütleme 384 piksellik küçük görüntüde yapılır; maskeyi büyütünce kenar basamaklı ve
birkaç piksel kayık olur. Çözüm: kenarın çevresinde dar bir "belirsiz şerit" bırakıp
şeritteki her pikseli, *yakınındaki içeri renginin mi yoksa dışarı renginin mi*
ortalamasına daha yakın olduğuna göre sınıflandırmak. Böylece sınır, resimdeki gerçek
renk geçişine yapışır.
"""
from __future__ import annotations

import numpy as np

from . import filtre, renk


def tam_cozunurluge_tasi(kaba: np.ndarray, rgb_u8: np.ndarray,
                         olcek: float) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Kaba maskeyi ``rgb_u8`` çözünürlüğüne taşır ve kenarını iyileştirir.

    Döner: (tam boyutlu bool maske, (y0, x0, y1, x1) sınır kutusu)."""
    H, W = rgb_u8.shape[:2]
    h, w = kaba.shape
    ys, xs = np.nonzero(kaba)
    ky0, ky1, kx0, kx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1

    serit = max(2, int(round(1.5 * olcek)))          # belirsiz şerit yarıçapı (tam çöz. piksel)
    pay = serit * 3 + 4
    y0 = max(0, int(ky0 * olcek) - pay)
    x0 = max(0, int(kx0 * olcek) - pay)
    y1 = min(H, int(np.ceil(ky1 * olcek)) + pay)
    x1 = min(W, int(np.ceil(kx1 * olcek)) + pay)

    # Kaba maskeyi yumuşatıp yalnızca nesne çevresindeki kesitte büyüt, 0.5'ten kes:
    # basamaklar yuvarlanır, büyük görüntüde tam boyutlu dizi üretilmez.
    yumusak = filtre.gauss_bulaniklik(kaba.astype(np.float32), 0.7)
    ys_k = (np.arange(y0, y1, dtype=np.float32) + 0.5) / olcek - 0.5
    xs_k = (np.arange(x0, x1, dtype=np.float32) + 0.5) / olcek - 0.5
    m = filtre.bilineer_ornekle(yumusak, ys_k, xs_k) > 0.5

    lab = renk.rgb_den_lab(rgb_u8[y0:y1, x0:x1].astype(np.float32) / 255.0)
    lab = filtre.gauss_bulaniklik(lab, 0.8)

    ic_guvenli = filtre.daralt(m, serit)
    dis_guvenli = ~filtre.genislet(m, serit)
    belirsiz = ~(ic_guvenli | dis_guvenli)

    # Yerel ortalama renkler: pencere içindeki güvenli içeri/dışarı piksellerinden.
    r = max(4, serit * 3)
    agirlik_ic = filtre.kutu_ortalama(ic_guvenli.astype(np.float32), r)
    agirlik_dis = filtre.kutu_ortalama(dis_guvenli.astype(np.float32), r)
    ort_ic = filtre.kutu_ortalama(lab * ic_guvenli[..., None], r) / np.maximum(agirlik_ic, 1e-4)[..., None]
    ort_dis = filtre.kutu_ortalama(lab * dis_guvenli[..., None], r) / np.maximum(agirlik_dis, 1e-4)[..., None]

    d_ic = ((lab - ort_ic) ** 2).sum(-1)
    d_dis = ((lab - ort_dis) ** 2).sum(-1)
    ic_mi = d_ic < d_dis
    # Yerelde örnek yoksa (ortalama bilinmiyor) kaba maskeye güven.
    ic_mi = np.where(agirlik_ic < 1e-3, m, ic_mi)
    ic_mi = np.where(agirlik_dis < 1e-3, m, ic_mi)

    sonuc = np.where(belirsiz, ic_mi, ic_guvenli)
    # Pürüzsüzleştir: küçük kırıntıları at, kenarı hafif yumuşat.
    sonuc = filtre.gauss_bulaniklik(sonuc.astype(np.float32), 1.0) > 0.5
    sonuc = filtre.en_buyuk_bilesen(sonuc)
    sonuc = filtre.delikleri_doldur(sonuc) | (sonuc & m)   # küçük içsel boşlukları kapat

    tam = np.zeros((H, W), dtype=bool)
    tam[y0:y1, x0:x1] = sonuc
    ys, xs = np.nonzero(sonuc)
    kutu = (y0 + int(ys.min()), x0 + int(xs.min()), y0 + int(ys.max()) + 1, x0 + int(xs.max()) + 1)
    return tam, kutu
