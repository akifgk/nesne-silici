"""Temel görüntü filtreleri ve morfoloji (elle yazılmış, yalnızca numpy).

İçindekiler
-----------
* ``gauss_bulaniklik``  : Gauss bulanıklığı (ayrıştırılabilir evrişim)
* ``kutu_ortalama``     : Kaydırmalı pencere ortalaması (integral görüntü ile O(1))
* ``gradyan``           : Sobel kenar gücü
* ``boyutlandir``       : Bilineer yeniden örnekleme
* ``genislet/daralt``   : İkili morfoloji (maske büyütme/küçültme)
* ``etiketle``          : Bağlantılı bileşen etiketleme
* ``delikleri_doldur``  : Maskedeki iç boşlukları kapatır
"""
from __future__ import annotations

import math

import numpy as np


# --------------------------------------------------------------------------- evrişim
def gauss_cekirdegi(sigma: float) -> np.ndarray:
    """Tek boyutlu, toplamı 1 olan Gauss çekirdeği."""
    yaricap = max(1, int(math.ceil(3.0 * sigma)))
    x = np.arange(-yaricap, yaricap + 1, dtype=np.float32)
    k = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return (k / k.sum()).astype(np.float32)


def _eksen_evrisim(img: np.ndarray, k: np.ndarray, eksen: int) -> np.ndarray:
    """Tek eksende evrişim. Kenarlar aynalanır (reflect), böylece koyu çerçeve oluşmaz."""
    r = len(k) // 2
    pad = [(0, 0)] * img.ndim
    pad[eksen] = (r, r)
    p = np.pad(img, pad, mode="reflect") if img.shape[eksen] > r else np.pad(img, pad, mode="edge")
    n = img.shape[eksen]
    sonuc = np.zeros(img.shape, dtype=np.float32)
    for i, agirlik in enumerate(k):
        dilim = [slice(None)] * img.ndim
        dilim[eksen] = slice(i, i + n)
        sonuc += agirlik * p[tuple(dilim)]
    return sonuc


def gauss_bulaniklik(img: np.ndarray, sigma: float) -> np.ndarray:
    """2B Gauss bulanıklığı. 2B Gauss ayrıştırılabilir olduğundan önce satır, sonra sütun
    işlenir: maliyet K² yerine 2K olur."""
    if sigma <= 0:
        return img.astype(np.float32, copy=True)
    k = gauss_cekirdegi(sigma)
    img = img.astype(np.float32, copy=False)
    return _eksen_evrisim(_eksen_evrisim(img, k, 0), k, 1)


def kutu_ortalama(img: np.ndarray, yaricap: int) -> np.ndarray:
    """(2r+1)x(2r+1) pencere ortalaması. Integral görüntü (kümülatif toplam) kullanır:
    pencere ne kadar büyük olursa olsun piksel başına 4 işlem yapılır."""
    r = int(yaricap)
    h, w = img.shape[:2]
    p = np.pad(img.astype(np.float64), [(r + 1, r), (r + 1, r)] + [(0, 0)] * (img.ndim - 2), mode="edge")
    c = p.cumsum(0).cumsum(1)
    d = 2 * r + 1
    toplam = c[d:d + h, d:d + w] - c[0:h, d:d + w] - c[d:d + h, 0:w] + c[0:h, 0:w]
    return (toplam / (d * d)).astype(np.float32)


# --------------------------------------------------------------------------- kenar
def gradyan(gri: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sobel gradyanı (gx, gy). Kenarlar yumuşak değişimden güçlü diye ayrışır."""
    p = np.pad(gri.astype(np.float32), 1, mode="edge")
    gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
    gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])
    return gx / 8.0, gy / 8.0


def kenar_gucu(lab: np.ndarray) -> np.ndarray:
    """Çok kanallı görüntüde piksel başına kenar gücü: kanal gradyanlarının bileşkesi."""
    toplam = np.zeros(lab.shape[:2], dtype=np.float32)
    for kanal in range(lab.shape[2]):
        gx, gy = gradyan(lab[..., kanal])
        toplam += gx * gx + gy * gy
    return np.sqrt(toplam)


# --------------------------------------------------------------------------- yeniden örnekleme
def alan_ortalamasi(img: np.ndarray, faktor: int) -> np.ndarray:
    """Tam sayı ``faktor`` kadar küçültür; her blok ortalama alınır (en doğru küçültme).
    Büyük uint8 görüntülerde bellek taşmasın diye satır dilimleri halinde çalışır."""
    if faktor <= 1:
        return img.astype(np.float32)
    h, w = img.shape[:2]
    hh, ww = -(-h // faktor), -(-w // faktor)
    pad = [(0, hh * faktor - h), (0, ww * faktor - w)] + [(0, 0)] * (img.ndim - 2)
    p = np.pad(img, pad, mode="edge")
    c = 1 if img.ndim == 2 else img.shape[2]
    cikis = np.empty((hh, ww, c) if img.ndim == 3 else (hh, ww), dtype=np.float32)
    adim = max(1, 4_000_000 // max(1, ww * faktor * faktor * c))
    for y0 in range(0, hh, adim):
        y1 = min(hh, y0 + adim)
        blok = p[y0 * faktor:y1 * faktor].astype(np.float32)
        if img.ndim == 3:
            blok = blok.reshape(y1 - y0, faktor, ww, faktor, c)
            cikis[y0:y1] = blok.mean(axis=(1, 3))
        else:
            blok = blok.reshape(y1 - y0, faktor, ww, faktor)
            cikis[y0:y1] = blok.mean(axis=(1, 3))
    return cikis


def boyutlandir(img: np.ndarray, yeni_h: int, yeni_w: int) -> np.ndarray:
    """Bilineer yeniden örnekleme. Belirgin küçültmede önce Gauss ile süzer (aliasing önlemi)."""
    h, w = img.shape[:2]
    yeni_h, yeni_w = max(1, int(yeni_h)), max(1, int(yeni_w))
    if (h, w) == (yeni_h, yeni_w):
        return img.astype(np.float32, copy=True)
    img = img.astype(np.float32, copy=False)
    kucultme = max(h / yeni_h, w / yeni_w)
    if kucultme > 1.5:
        img = gauss_bulaniklik(img, 0.5 * kucultme)
    # Piksel merkezi hizalı koordinat eşlemesi.
    ys = (np.arange(yeni_h, dtype=np.float32) + 0.5) * (h / yeni_h) - 0.5
    xs = (np.arange(yeni_w, dtype=np.float32) + 0.5) * (w / yeni_w) - 0.5
    ys = np.clip(ys, 0, h - 1)
    xs = np.clip(xs, 0, w - 1)
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    fy = (ys - y0)[:, None]
    fx = (xs - x0)[None, :]
    if img.ndim == 3:
        fy, fx = fy[..., None], fx[..., None]
    ust = img[y0][:, x0] * (1 - fx) + img[y0][:, x1] * fx
    alt = img[y1][:, x0] * (1 - fx) + img[y1][:, x1] * fx
    return (ust * (1 - fy) + alt * fy).astype(np.float32)


def bilineer_ornekle(img: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """``img`` üzerinden (ys[:,None], xs[None,:]) ızgarasında bilineer örnekleme.

    ``ys``/``xs`` ondalıklı piksel koordinatlarıdır; görüntünün yalnızca istenen kesitini
    üretir, bu yüzden büyük görüntülerin küçük bölgesini büyütmek için idealdir."""
    h, w = img.shape[:2]
    ys = np.clip(ys.astype(np.float32), 0, h - 1)
    xs = np.clip(xs.astype(np.float32), 0, w - 1)
    y0, x0 = np.floor(ys).astype(np.int64), np.floor(xs).astype(np.int64)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    fy, fx = (ys - y0)[:, None], (xs - x0)[None, :]
    if img.ndim == 3:
        fy, fx = fy[..., None], fx[..., None]
    ust = img[y0][:, x0] * (1 - fx) + img[y0][:, x1] * fx
    alt = img[y1][:, x0] * (1 - fx) + img[y1][:, x1] * fx
    return (ust * (1 - fy) + alt * fy).astype(np.float32)


# --------------------------------------------------------------------------- morfoloji
def _kaydir_max(m: np.ndarray, r: int, kullan_max: bool) -> np.ndarray:
    """Kare yapı elemanıyla genişletme/daraltma; iki ayrık geçiş (yatay+dikey)."""
    f = np.maximum if kullan_max else np.minimum
    dolgu = False if kullan_max else True
    sonuc = m
    for eksen in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[eksen] = (r, r)
        p = np.pad(sonuc, pad, mode="constant", constant_values=dolgu)
        n = sonuc.shape[eksen]
        acc = None
        for i in range(2 * r + 1):
            dilim = [slice(None), slice(None)]
            dilim[eksen] = slice(i, i + n)
            parca = p[tuple(dilim)]
            acc = parca if acc is None else f(acc, parca)
        sonuc = acc
    return sonuc


def genislet(maske: np.ndarray, yaricap: int) -> np.ndarray:
    """Maskeyi her yöne ``yaricap`` piksel büyütür (kare yapı elemanı)."""
    if yaricap <= 0:
        return maske.astype(bool, copy=True)
    return _kaydir_max(maske.astype(bool), int(yaricap), True)


def daralt(maske: np.ndarray, yaricap: int) -> np.ndarray:
    """Maskeyi her yönden ``yaricap`` piksel küçültür."""
    if yaricap <= 0:
        return maske.astype(bool, copy=True)
    return _kaydir_max(maske.astype(bool), int(yaricap), False)


def sinir(maske: np.ndarray) -> np.ndarray:
    """Maskenin iç sınır pikselleri (4-komşuluğa göre)."""
    p = np.pad(maske, 1, mode="constant", constant_values=False)
    ic = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return maske & ~ic


# --------------------------------------------------------------------------- etiketleme
def etiketle(maske: np.ndarray, sekiz_komsu: bool = False) -> tuple[np.ndarray, int]:
    """Bağlantılı bileşen etiketleme (0 = arka plan, 1..n = bileşenler).

    Piksel piksel gezmek yerine her satırdaki ardışık ``True`` dizilerini (run) bulur;
    yalnızca alt alta çakışan diziler birleştirilir (union-find). Böylece python döngüsü
    piksel sayısına değil dizi sayısına bağlıdır.
    """
    h, w = maske.shape
    m = np.pad(maske.astype(np.int8), ((0, 0), (1, 1)))
    fark = np.diff(m, axis=1)
    y_bas, x_bas = np.nonzero(fark == 1)
    _, x_son = np.nonzero(fark == -1)
    n = len(y_bas)
    etiket = np.zeros((h, w), dtype=np.int32)
    if n == 0:
        return etiket, 0

    ebeveyn = list(range(n))

    def bul(i: int) -> int:
        while ebeveyn[i] != i:
            ebeveyn[i] = ebeveyn[ebeveyn[i]]
            i = ebeveyn[i]
        return i

    # Her satırın dizilerinin başlangıç indeksi.
    satir_bas = np.searchsorted(y_bas, np.arange(h + 1))
    ek = 1 if sekiz_komsu else 0
    for y in range(1, h):
        a0, a1 = satir_bas[y - 1], satir_bas[y]
        b0, b1 = satir_bas[y], satir_bas[y + 1]
        i, j = a0, b0
        while i < a1 and j < b1:
            # [x_bas, x_son) yarı açık aralıkların çakışması
            if x_bas[j] < x_son[i] + ek and x_bas[i] < x_son[j] + ek:
                ra, rb = bul(i), bul(j)
                if ra != rb:
                    ebeveyn[rb] = ra
            if x_son[i] < x_son[j]:
                i += 1
            else:
                j += 1

    kok = np.array([bul(i) for i in range(n)])
    _, yeni = np.unique(kok, return_inverse=True)
    for i in range(n):
        etiket[y_bas[i], x_bas[i]:x_son[i]] = yeni[i] + 1
    return etiket, int(yeni.max()) + 1


def en_buyuk_bilesen(maske: np.ndarray) -> np.ndarray:
    etiket, n = etiketle(maske)
    if n <= 1:
        return maske.copy()
    sayim = np.bincount(etiket.ravel())
    sayim[0] = 0
    return etiket == int(sayim.argmax())


def delikleri_doldur(maske: np.ndarray) -> np.ndarray:
    """Görüntü kenarına bağlı olmayan arka plan parçaları 'delik'tir; onları maskeye katar."""
    arka, n = etiketle(~maske)
    if n == 0:
        return maske.copy()
    kenar_etiketleri = np.unique(np.concatenate([arka[0], arka[-1], arka[:, 0], arka[:, -1]]))
    dis = np.isin(arka, kenar_etiketleri[kenar_etiketleri > 0])
    return ~dis
