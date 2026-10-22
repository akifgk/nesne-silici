"""Maskenin sınırını yumuşak, kapalı eğrilere (kontur) çevirir.

Yöntem: **marching squares**. Maske önce hafifçe bulanıklaştırılır (0..1 arası yumuşak
bir yüzey olur). Sonra her 2x2 piksel hücresinde, yüzeyin 0.5 seviyesinin hücreyi nasıl
kestiği bakılır ve kesişim noktaları doğrusal ara değerlemeyle piksel altı hassasiyetle
bulunur. Ortaya çıkan küçük doğru parçaları uç uca eklenerek kapalı çokgenler olur.

Çıktı, arayüzde ``QPainterPath`` olarak çizilir; böylece yakınlaştırınca bile kenar
pürüzsüz kalır (piksel basamağı görünmez).
"""
from __future__ import annotations

import numpy as np

from . import filtre

# Marching squares: bir hücrenin 4 köşesi iç/dış durumuna göre 4 bitlik indeks verir:
#   üst-sol=1, üst-sağ=2, alt-sağ=4, alt-sol=8.
# Tablo, o hücrede eğrinin hangi kenarları birleştirdiğini söyler.
# Kenarlar: 0=üst, 1=sağ, 2=alt, 3=sol. (5 ve 10 "eyer" durumudur: iki ayrı parça.)
_TABLO: dict[int, list[tuple[int, int]]] = {
    0: [], 15: [],
    1: [(0, 3)], 2: [(0, 1)], 3: [(3, 1)], 4: [(1, 2)],
    5: [(0, 3), (1, 2)], 6: [(0, 2)], 7: [(3, 2)], 8: [(3, 2)],
    9: [(0, 2)], 10: [(0, 1), (3, 2)], 11: [(1, 2)], 12: [(3, 1)],
    13: [(0, 1)], 14: [(0, 3)],
}


def konturlar(maske: np.ndarray, yumusatma: float = 1.2, sadelestir: float = 0.6,
              min_uzunluk: int = 12) -> list[np.ndarray]:
    """Maskenin sınır eğrilerini döndürür: her biri ``(N, 2)`` boyutlu ``(x, y)`` dizisi.

    ``yumusatma`` bulanıklık sigması, ``sadelestir`` Douglas-Peucker toleransıdır (piksel)."""
    ys, xs = np.nonzero(maske)
    if len(ys) == 0:
        return []
    pay = 3
    y0, y1 = max(0, ys.min() - pay), min(maske.shape[0], ys.max() + 1 + pay)
    x0, x1 = max(0, xs.min() - pay), min(maske.shape[1], xs.max() + 1 + pay)
    kes = np.pad(maske[y0:y1, x0:x1].astype(np.float32), 2)
    f = filtre.gauss_bulaniklik(kes, yumusatma) if yumusatma > 0 else kes

    ic = f > 0.5
    ust_sol, ust_sag = ic[:-1, :-1], ic[:-1, 1:]
    alt_sag, alt_sol = ic[1:, 1:], ic[1:, :-1]
    indeks = (ust_sol * 1 + ust_sag * 2 + alt_sag * 4 + alt_sol * 8).astype(np.int8)
    cy, cx = np.nonzero((indeks != 0) & (indeks != 15))

    def kenar_noktasi(y: int, x: int, kenar: int) -> tuple[float, float]:
        # Hücrenin dört köşesindeki yumuşak değerler
        a, b = f[y, x], f[y, x + 1]            # üst-sol, üst-sağ
        d, c = f[y + 1, x], f[y + 1, x + 1]    # alt-sol, alt-sağ

        def ara(v0: float, v1: float) -> float:
            return 0.5 if v0 == v1 else float(np.clip((0.5 - v0) / (v1 - v0), 0.0, 1.0))

        if kenar == 0:
            return x + ara(a, b), float(y)
        if kenar == 1:
            return x + 1.0, y + ara(b, c)
        if kenar == 2:
            return x + ara(d, c), y + 1.0
        return float(x), y + ara(a, d)

    # Parçaları uç noktalarına göre anahtarla (kenarın kimliği: iki hücre paylaşır).
    def kenar_anahtari(y: int, x: int, kenar: int) -> tuple[int, int, int]:
        if kenar == 0:
            return (y, x, 0)
        if kenar == 1:
            return (y, x + 1, 1)
        if kenar == 2:
            return (y + 1, x, 0)
        return (y, x, 1)

    komsu: dict[tuple, list[tuple]] = {}
    nokta: dict[tuple, tuple[float, float]] = {}
    for y, x in zip(cy.tolist(), cx.tolist()):
        for e1, e2 in _TABLO[int(indeks[y, x])]:
            k1, k2 = kenar_anahtari(y, x, e1), kenar_anahtari(y, x, e2)
            nokta.setdefault(k1, kenar_noktasi(y, x, e1))
            nokta.setdefault(k2, kenar_noktasi(y, x, e2))
            komsu.setdefault(k1, []).append(k2)
            komsu.setdefault(k2, []).append(k1)

    gorulen: set = set()
    sonuc: list[np.ndarray] = []
    for baslangic in list(komsu):
        if baslangic in gorulen:
            continue
        yol = [baslangic]
        gorulen.add(baslangic)
        onceki, simdi = None, baslangic
        while True:
            adaylar = [k for k in komsu[simdi] if k != onceki and k not in gorulen]
            if not adaylar:
                break
            onceki, simdi = simdi, adaylar[0]
            gorulen.add(simdi)
            yol.append(simdi)
        if len(yol) < min_uzunluk:
            continue
        # Çerçeve dolgusu (2) ve kırpma kaymasını geri al: (x, y) tam görüntü koordinatı.
        pts = np.array([(nokta[k][0] - 2 + x0 + 0.5, nokta[k][1] - 2 + y0 + 0.5) for k in yol],
                       dtype=np.float32)
        sonuc.append(_sadelestir(pts, sadelestir) if sadelestir > 0 else pts)
    sonuc.sort(key=lambda p: -len(p))
    return sonuc


def _sadelestir(pts: np.ndarray, tolerans: float) -> np.ndarray:
    """Douglas-Peucker: gereksiz ara noktaları atar, şekli ``tolerans`` piksel içinde korur."""
    if len(pts) < 8:
        return pts
    tut = np.zeros(len(pts), dtype=bool)
    tut[0] = tut[-1] = True
    yigin = [(0, len(pts) - 1)]
    while yigin:
        a, b = yigin.pop()
        if b <= a + 1:
            continue
        p, q = pts[a], pts[b]
        v = q - p
        uz = float(np.hypot(*v))
        ara = pts[a + 1:b]
        if uz < 1e-6:
            d = np.hypot(*(ara - p).T)
        else:
            d = np.abs(v[0] * (ara[:, 1] - p[1]) - v[1] * (ara[:, 0] - p[0])) / uz
        i = int(d.argmax())
        if d[i] > tolerans:
            tut[a + 1 + i] = True
            yigin.append((a, a + 1 + i))
            yigin.append((a + 1 + i, b))
    return pts[tut]
