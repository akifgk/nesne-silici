"""Görüntüyü bölgelere ayırma: SLIC süperpiksel + hiyerarşik bölge birleştirme.

Fikir iki aşamalıdır:

1. **SLIC**: Görüntüyü yüzlerce küçük, rengi birbirine uyan "süperpiksele" böler.
   Her süperpikselin merkezi vardır; piksel, kendine "renk + konum olarak" en yakın
   merkeze katılır. Konum terimi parçaların derli toplu (kompakt) kalmasını sağlar.
2. **Birleştirme ağacı**: Komşu süperpiksellerden birbirine en çok benzeyenler
   sırayla birleştirilir (Ward maliyeti, arada güçlü kenar varsa maliyet artar).
   Her birleşme ağaca yeni bir düğüm ekler. Ağacın yaprakları süperpiksel, kökü
   tüm görüntüdür; aradaki her düğüm "olası bir nesne"dir. Hangisinin gerçekten
   nesne olduğunu ``nesne_bulucu`` seçer.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np

from . import filtre


# --------------------------------------------------------------------------- SLIC
def slic(lab: np.ndarray, parca_sayisi: int = 600, kompaktlik: float = 12.0,
         tekrar: int = 6) -> np.ndarray:
    """SLIC süperpiksel bölütleme. Dönen dizi: her pikselin süperpiksel numarası (0..n-1).

    ``kompaktlik`` büyüdükçe parçalar kareye/dairesele yaklaşır, küçüldükçe kenarlara
    daha sadık ama daha düzensiz olur.
    """
    h, w, _ = lab.shape
    S = max(4, int(math.sqrt(h * w / parca_sayisi)))
    cy = np.arange(S // 2, h, S, dtype=np.float32)
    cx = np.arange(S // 2, w, S, dtype=np.float32)
    gy, gx = np.meshgrid(cy, cx, indexing="ij")
    merkez = np.zeros((gy.size, 5), dtype=np.float32)   # y, x, L, a, b
    merkez[:, 0], merkez[:, 1] = gy.ravel(), gx.ravel()
    merkez[:, 2:] = lab[gy.ravel().astype(int), gx.ravel().astype(int)]
    k = len(merkez)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    agirlik = (kompaktlik / S) ** 2
    etiket = np.zeros((h, w), dtype=np.int32)

    for _ in range(tekrar):
        uzaklik = np.full((h, w), np.inf, dtype=np.float32)
        etiket = np.full((h, w), -1, dtype=np.int32)
        for i in range(k):
            y, x = merkez[i, 0], merkez[i, 1]
            y0, y1 = int(max(0, y - S)), int(min(h, y + S + 1))
            x0, x1 = int(max(0, x - S)), int(min(w, x + S + 1))
            renk_f = ((lab[y0:y1, x0:x1] - merkez[i, 2:]) ** 2).sum(-1)
            konum_f = (yy[y0:y1, x0:x1] - y) ** 2 + (xx[y0:y1, x0:x1] - x) ** 2
            d = renk_f + agirlik * konum_f
            alt = uzaklik[y0:y1, x0:x1]
            sec = d < alt
            alt[sec] = d[sec]
            etiket[y0:y1, x0:x1][sec] = i
        etiket[etiket < 0] = 0
        # Merkezleri üyelerinin ortalamasına taşı.
        sayim = np.bincount(etiket.ravel(), minlength=k).astype(np.float32)
        ok = sayim > 0
        for kanal, deger in enumerate((yy, xx, lab[..., 0], lab[..., 1], lab[..., 2])):
            toplam = np.bincount(etiket.ravel(), weights=deger.ravel(), minlength=k)
            merkez[ok, kanal] = toplam[ok] / sayim[ok]

    return _bagli_hale_getir(etiket, min_alan=max(4, S * S // 5))


def _bagli_hale_getir(etiket: np.ndarray, min_alan: int) -> np.ndarray:
    """SLIC bazen aynı etiketi birbirinden kopuk parçalara verir. Her parçayı ayrı
    süperpiksel yapar; çok küçük kalanları en çok temas ettikleri komşuya katar."""
    h, w = etiket.shape
    yeni = np.full((h, w), -1, dtype=np.int32)
    sayac = 0
    kucukler: list[tuple[int, int, int, int, np.ndarray]] = []  # (id, y0, x0, ..., maske)
    for e in np.unique(etiket):
        ys, xs = np.nonzero(etiket == e)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        kutu = etiket[y0:y1, x0:x1] == e
        bil, n = filtre.etiketle(kutu)
        for b in range(1, n + 1):
            m = bil == b
            yeni[y0:y1, x0:x1][m] = sayac
            if m.sum() < min_alan:
                kucukler.append((sayac, y0, x0, y1, m))  # type: ignore[arg-type]
            sayac += 1
    # Küçükleri komşuya birleştir (en çok sınır paylaşılan etiket).
    for kimlik, y0, x0, y1, m in kucukler:
        mm = np.zeros((h, w), dtype=bool)
        mm[y0:y1, x0:x0 + m.shape[1]] = m
        halka = filtre.genislet(mm, 1) & ~mm
        komsu = yeni[halka]
        komsu = komsu[(komsu >= 0) & (komsu != kimlik)]
        if len(komsu):
            yeni[mm] = np.bincount(komsu).argmax()
    # Etiketleri 0..n-1 aralığına sıkıştır.
    _, sikis = np.unique(yeni, return_inverse=True)
    return sikis.reshape(h, w).astype(np.int32)


# --------------------------------------------------------------------------- birleştirme ağacı
@dataclass
class BolgeAgaci:
    """Hiyerarşik bölütleme sonucu.

    Düğüm numaraları: ``0..n_yaprak-1`` süperpiksel (yaprak), sonrası birleşmeyle doğan
    iç düğümler. ``ebeveyn[d]`` düğümün üstünü, ``cocuklar[d]`` altındaki ikiliyi verir.
    """
    etiket: np.ndarray                    # (h, w) yaprak numarası
    n_yaprak: int
    ebeveyn: list[int] = field(default_factory=list)
    cocuklar: dict[int, tuple[int, int]] = field(default_factory=dict)
    yapraklar: list[np.ndarray] = field(default_factory=list)   # düğüm -> yaprak numaraları
    alan: list[int] = field(default_factory=list)
    kutu: list[tuple[int, int, int, int]] = field(default_factory=list)  # y0, x0, y1, x1
    maliyet: list[float] = field(default_factory=list)

    @property
    def n_dugum(self) -> int:
        return len(self.ebeveyn)

    def maske(self, dugum: int) -> np.ndarray:
        """Düğümün tam boyutlu maskesi."""
        m = np.zeros(self.etiket.shape, dtype=bool)
        y0, x0, y1, x1 = self.kutu[dugum]
        m[y0:y1, x0:x1] = np.isin(self.etiket[y0:y1, x0:x1], self.yapraklar[dugum])
        return m

    def kirpik_maske(self, dugum: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """Yalnızca sınır kutusu içindeki maske ve kutu (büyük görüntülerde hızlı)."""
        y0, x0, y1, x1 = self.kutu[dugum]
        return np.isin(self.etiket[y0:y1, x0:x1], self.yapraklar[dugum]), (y0, x0, y1, x1)

    def nokta_zinciri(self, x: int, y: int) -> list[int]:
        """(x, y) noktasını içeren düğümler: en küçükten (yaprak) en büyüğe (kök)."""
        d = int(self.etiket[y, x])
        zincir = [d]
        while self.ebeveyn[d] != -1:
            d = self.ebeveyn[d]
            zincir.append(d)
        return zincir


def agac_kur(lab: np.ndarray, etiket: np.ndarray, kenar_etkisi: float = 1.0) -> BolgeAgaci:
    """Süperpikselleri Ward maliyetiyle birleştirip ağacı çıkarır.

    Maliyet(a, b) = (na·nb / (na+nb)) · ‖ortalama_a − ortalama_b‖² · (1 + kenar_etkisi · k)

    * İlk çarpan bölgeler büyüdükçe birleşmeyi zorlaştırır (küçükler önce yutulur).
    * İkinci çarpan renk farkıdır.
    * ``k``, iki bölgenin ortak sınırındaki ortalama kenar gücüdür: aralarında net bir
      kenar varsa maliyet yükselir ve bölgeler ayrı kalmayı tercih eder.
    """
    h, w, _ = lab.shape
    n = int(etiket.max()) + 1
    düz = etiket.ravel()
    sayim = np.bincount(düz, minlength=n).astype(np.float64)
    toplam = np.stack([np.bincount(düz, weights=lab[..., c].ravel(), minlength=n) for c in range(3)], 1)

    # Sınır kutuları
    ys, xs = np.mgrid[0:h, 0:w]
    y_min = np.full(n, h, dtype=np.int64)
    y_max = np.zeros(n, dtype=np.int64)
    x_min = np.full(n, w, dtype=np.int64)
    x_max = np.zeros(n, dtype=np.int64)
    np.minimum.at(y_min, düz, ys.ravel())
    np.maximum.at(y_max, düz, ys.ravel())
    np.minimum.at(x_min, düz, xs.ravel())
    np.maximum.at(x_max, düz, xs.ravel())
    kutu = [(int(y_min[i]), int(x_min[i]), int(y_max[i]) + 1, int(x_max[i]) + 1) for i in range(n)]

    # Komşuluklar: yatay ve dikey komşu piksel çiftleri farklı etiketteyse sınırdır.
    komsu: list[dict[int, list[float]]] = [dict() for _ in range(n)]
    for a, b, f in (
        (etiket[:, :-1], etiket[:, 1:], np.linalg.norm(lab[:, :-1] - lab[:, 1:], axis=-1)),
        (etiket[:-1, :], etiket[1:, :], np.linalg.norm(lab[:-1, :] - lab[1:, :], axis=-1)),
    ):
        fark = a != b
        i, j, g = a[fark].astype(np.int64), b[fark].astype(np.int64), f[fark]
        lo, hi = np.minimum(i, j), np.maximum(i, j)
        anahtar = lo * n + hi
        essiz, ters = np.unique(anahtar, return_inverse=True)
        say = np.bincount(ters)
        top = np.bincount(ters, weights=g)
        for k_, c_, t_ in zip(essiz, say, top):
            u, v = int(k_ // n), int(k_ % n)
            kayit = komsu[u].setdefault(v, [0.0, 0.0])
            kayit[0] += c_
            kayit[1] += t_
            komsu[v][u] = kayit      # iki yönde aynı liste: tek yerden güncellenir

    tum = [kk[1] / kk[0] for d in komsu for kk in d.values() if kk[0] > 0]
    olcek = float(np.mean(tum)) if tum else 1.0

    def maliyet(a: int, b: int, kayit: list[float]) -> float:
        na, nb = sayim[a], sayim[b]
        ma, mb = toplam[a] / na, toplam[b] / nb
        ward = (na * nb / (na + nb)) * float(((ma - mb) ** 2).sum())
        kenar = kayit[1] / max(kayit[0], 1.0)
        return ward * (1.0 + kenar_etkisi * kenar / olcek)

    agac = BolgeAgaci(etiket=etiket, n_yaprak=n)
    agac.ebeveyn = [-1] * n
    agac.yapraklar = [np.array([i]) for i in range(n)]
    agac.alan = [int(s) for s in sayim]
    agac.kutu = list(kutu)
    agac.maliyet = [0.0] * n

    canli = [True] * n
    yigin: list[tuple[float, int, int]] = []
    for a in range(n):
        for b, kayit in komsu[a].items():
            if a < b:
                heapq.heappush(yigin, (maliyet(a, b, kayit), a, b))

    sayim = list(sayim)
    toplam = [toplam[i] for i in range(n)]
    while yigin:
        c, a, b = heapq.heappop(yigin)
        if not (canli[a] and canli[b]):
            continue
        yeni = len(canli)
        canli[a] = canli[b] = False
        canli.append(True)
        agac.ebeveyn[a] = agac.ebeveyn[b] = yeni
        agac.ebeveyn.append(-1)
        agac.cocuklar[yeni] = (a, b)
        agac.yapraklar.append(np.concatenate([agac.yapraklar[a], agac.yapraklar[b]]))
        agac.alan.append(agac.alan[a] + agac.alan[b])
        ka, kb = agac.kutu[a], agac.kutu[b]
        agac.kutu.append((min(ka[0], kb[0]), min(ka[1], kb[1]), max(ka[2], kb[2]), max(ka[3], kb[3])))
        agac.maliyet.append(float(c))
        sayim.append(sayim[a] + sayim[b])
        toplam.append(toplam[a] + toplam[b])

        # a ve b'nin komşuluklarını yeni düğüme taşı (ortak komşularda toplanır).
        yeni_komsu: dict[int, list[float]] = {}
        for eski in (a, b):
            for v, kayit in komsu[eski].items():
                if v in (a, b):
                    continue
                hedef = yeni_komsu.setdefault(v, [0.0, 0.0])
                hedef[0] += kayit[0]
                hedef[1] += kayit[1]
                komsu[v].pop(eski, None)
        komsu.append(yeni_komsu)
        for v, kayit in yeni_komsu.items():
            komsu[v][yeni] = kayit
            heapq.heappush(yigin, (maliyet(yeni, v, kayit), yeni, v))
        # maliyet() 'sayim'/'toplam' listelerini okuduğu için numpy dönüşümü gerekmez.
    return agac
