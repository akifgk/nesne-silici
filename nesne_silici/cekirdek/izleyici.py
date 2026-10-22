"""Videoda nesneleri kareler arası eşleyip "kimlik" verir (izleme).

Her karede ``nesne_bulucu`` bağımsız çalışır; "bu karedeki Nesne 2" ile "önceki karedeki
Nesne 5" aynı şey mi, onu burada karar veririz. Eşleştirme maliyeti üç farkın toplamıdır:

* **Renk**: Lab ortalama renklerinin uzaklığı (kıyaslanan nesne aynı renkte kalır).
* **Konum**: merkezlerin görüntü boyuna oranlı uzaklığı (nesne kareler arasında az hareket eder).
* **Boyut**: alan oranının logaritması (aniden büyüyüp küçülmez).

Eşleştirme açgözlüdür: en düşük maliyetli çift önce bağlanır. Eşik aşılırsa nesne yeni bir
iz olur. Bir iz birkaç kare kaybolup geri gelebilir (``kayip_toleransi``): oyuncunun
önünden bir şey geçince nesne kısa süre algılanamaz.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .nesne_bulucu import Nesne

KOMSU_ESIK = 2.2          # bu maliyetin üstü eşleşme sayılmaz
RENK_OLCEK = 25.0         # Lab uzaklığının 1 maliyet sayıldığı değer
KONUM_AGIRLIK = 3.0
BOYUT_AGIRLIK = 1.0


@dataclass
class Gozlem:
    """Bir nesnenin bir karedeki durumu."""
    kare: int
    zaman: float                 # saniye
    lab: np.ndarray              # (3,) ortalama Lab
    merkez: tuple[float, float]  # (x, y) 0..1 normalleştirilmiş
    alan_orani: float            # nesne alanı / görüntü alanı
    kutu: tuple[int, int, int, int]
    maske: np.ndarray            # çalışma çözünürlüğünde bool maske
    puan: float


@dataclass
class Iz:
    """Aynı nesnenin videodaki yolculuğu."""
    kimlik: int
    gozlemler: list[Gozlem] = field(default_factory=list)

    @property
    def ilk_zaman(self) -> float:
        return self.gozlemler[0].zaman

    @property
    def son_zaman(self) -> float:
        return self.gozlemler[-1].zaman

    @property
    def kare_sayisi(self) -> int:
        return len(self.gozlemler)

    @property
    def ortalama_alan(self) -> float:
        return float(np.mean([g.alan_orani for g in self.gozlemler]))

    @property
    def en_iyi(self) -> Gozlem:
        """Listede gösterilecek temsilci kare: nesnenin en net/büyük göründüğü an."""
        return max(self.gozlemler, key=lambda g: g.puan * math.sqrt(g.alan_orani))


def gozlem_olustur(nesne: Nesne, kare: int, zaman: float, sekil: tuple[int, int]) -> Gozlem:
    h, w = sekil
    ys, xs = np.nonzero(nesne.maske)
    return Gozlem(
        kare=kare, zaman=zaman, lab=np.array(nesne.ortalama_lab, dtype=np.float32),
        merkez=(float(xs.mean()) / w, float(ys.mean()) / h),
        alan_orani=float(nesne.alan) / (h * w), kutu=nesne.kutu, maske=nesne.maske, puan=nesne.puan)


def _maliyet(iz: Iz, g: Gozlem) -> float:
    son = iz.gozlemler[-1]
    renk = float(np.linalg.norm(son.lab - g.lab)) / RENK_OLCEK
    konum = math.hypot(son.merkez[0] - g.merkez[0], son.merkez[1] - g.merkez[1]) * KONUM_AGIRLIK
    boyut = abs(math.log(max(g.alan_orani, 1e-6) / max(son.alan_orani, 1e-6))) * BOYUT_AGIRLIK
    return renk + konum + boyut


def izle(kareler: list[list[Gozlem]], kayip_toleransi: int = 3) -> list[Iz]:
    """Kare kare gözlem listelerinden izleri çıkarır. Dönen izler ilk görülme sırasına göredir."""
    izler: list[Iz] = []
    for gozlemler in kareler:
        if not gozlemler:
            continue
        kare_no = gozlemler[0].kare
        # Aday izler: yakın zamanda görülmüş olanlar.
        aktif = [iz for iz in izler if kare_no - iz.gozlemler[-1].kare <= kayip_toleransi + 1]
        ciftler = sorted(
            ((_maliyet(iz, g), i, j) for i, iz in enumerate(aktif) for j, g in enumerate(gozlemler)),
            key=lambda c: c[0])
        kullanilan_iz, kullanilan_g = set(), set()
        for maliyet, i, j in ciftler:
            if maliyet > KOMSU_ESIK:
                break
            if i in kullanilan_iz or j in kullanilan_g:
                continue
            aktif[i].gozlemler.append(gozlemler[j])
            kullanilan_iz.add(i)
            kullanilan_g.add(j)
        for j, g in enumerate(gozlemler):
            if j not in kullanilan_g:
                izler.append(Iz(kimlik=len(izler) + 1, gozlemler=[g]))
    return izler
