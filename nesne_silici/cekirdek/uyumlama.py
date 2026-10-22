"""Eklenen görseli sahneye uydurma: renk, ışık yönü, gren, gölge ve kenar.

Bir nesne arka plana "yapıştırıldığı" belli olur çünkü şu işaretler tutmaz:

1. **Renk/parlaklık**: Nesne başka bir ışıkta çekilmiştir. Çevredeki halkanın Lab
   istatistiğine (Reinhard renk aktarımı) kısmen yaklaştırılır. Kromaya (a, b) daha az
   dokunulur; yoksa kırmızı elma gri-yeşile döner.
2. **Işık yönü**: Sahnenin parlaklığına düzlem uydurulur (L ≈ a·x + b·y + c). Düzlemin
   eğimi ışığın nereden geldiğini verir; nesneye aynı yönde yumuşak bir parlaklık
   gradyanı uygulanır.
3. **Gren**: Fotoğraflarda sensör gürültüsü vardır, yapıştırılan temiz bir görselde
   yoktur. Sahnenin gren düzeyi (yüksek frekanslı bileşenin sağlam std'si) ölçülür,
   eksik kalan kadar gürültü eklenir.
4. **Gölge**: Işığın zıt yönünde, nesnenin alfa şeklinden türetilmiş yumuşak bir gölge
   ve nesnenin tabanında daha sıkı bir temas gölgesi.
5. **Kenar**: Alfa bir piksel daraltılıp yumuşatılır (eski arka plan renginden saçak
   kalmaz). İstenirse Poisson karıştırma ile renk sızıntısı da giderilir.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import filtre, poisson, renk


@dataclass
class UyumAyarlari:
    renk: float = 0.8        # 0..1 renk/parlaklık aktarımı
    isik: float = 0.8        # 0..1 ışık yönü gradyanı
    gren: float = 1.0        # 0..1 gren eşleştirme
    golge: float = 0.5       # 0..1 gölge şiddeti
    kenar: float = 1.0       # 0..4 kenar yumuşatma (piksel)
    poisson: bool = False    # Poisson ile renk sızıntısı gider


@dataclass
class EklenenGorsel:
    """Yerleştirilecek görsel ve konumu (konum: sol üst köşe, tuval koordinatı)."""
    rgb: np.ndarray          # float32 (h, w, 3)
    alfa: np.ndarray         # float32 (h, w)
    x: float = 0.0
    y: float = 0.0
    olcek: float = 1.0

    def olcekli(self) -> tuple[np.ndarray, np.ndarray]:
        h = max(1, round(self.rgb.shape[0] * self.olcek))
        w = max(1, round(self.rgb.shape[1] * self.olcek))
        if (h, w) == self.rgb.shape[:2]:
            return self.rgb, self.alfa
        return filtre.boyutlandir(self.rgb, h, w), np.clip(filtre.boyutlandir(self.alfa, h, w), 0, 1)


# --------------------------------------------------------------------------- yardımcılar
def _agirlikli_istatistik(lab: np.ndarray, maske: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    deger = lab[maske]
    return deger.mean(axis=0), deger.std(axis=0) + 1e-3


def _isik_duzlemi(L: np.ndarray, gecerli: np.ndarray) -> tuple[float, float]:
    """Parlaklığa L ≈ a·x + b·y + c düzlemi uydurur (en küçük kareler). (a, b) piksel başına
    parlaklık değişimidir. Çok sayıda piksel varsa seyreltilir."""
    ys, xs = np.nonzero(gecerli)
    if len(ys) < 50:
        return 0.0, 0.0
    if len(ys) > 6000:
        sec = np.random.default_rng(0).choice(len(ys), 6000, replace=False)
        ys, xs = ys[sec], xs[sec]
    A = np.stack([xs - xs.mean(), ys - ys.mean(), np.ones(len(xs))], axis=1)
    katsayi, *_ = np.linalg.lstsq(A, L[ys, xs], rcond=None)
    return float(katsayi[0]), float(katsayi[1])


def _yuksek_frekans_std(L: np.ndarray, maske: np.ndarray) -> float:
    """Gren düzeyi: bulanıklığı çıkarılmış parlaklığın sağlam (MAD) std'si."""
    yf = L - filtre.gauss_bulaniklik(L, 1.0)
    v = yf[maske]
    if len(v) < 30:
        return 0.0
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def _kaydir(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    return np.roll(np.roll(a, dy, 0), dx, 1)


# --------------------------------------------------------------------------- ana işlev
def yerlestir(arka_u8: np.ndarray, gorsel: EklenenGorsel, ayar: UyumAyarlari
              ) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
    """Görseli sahneye uyduruyp yerleştirir.

    Döner: ((y0, x0, y1, x1), yeni_parca_uint8) ya da görsel tuvalin dışındaysa None."""
    H, W = arka_u8.shape[:2]
    nesne_rgb, nesne_alfa = gorsel.olcekli()
    nh, nw = nesne_alfa.shape
    x0, y0 = int(round(gorsel.x)), int(round(gorsel.y))
    if x0 >= W or y0 >= H or x0 + nw <= 0 or y0 + nh <= 0:
        return None

    # 1) Çalışma kutusu (ROI): nesne + gölge/halka için çevre payı.
    pay = int(max(nh, nw) * 0.6) + 24
    ry0, rx0 = max(0, y0 - pay), max(0, x0 - pay)
    ry1, rx1 = min(H, y0 + nh + pay), min(W, x0 + nw + pay)
    arka = arka_u8[ry0:ry1, rx0:rx1].astype(np.float32) / 255.0
    h, w = arka.shape[:2]

    # 2) Nesneyi ROI boyutlu tuvale yerleştir (tuval dışına taşan kısım kırpılır).
    obj = np.zeros((h, w, 3), dtype=np.float32)
    alfa = np.zeros((h, w), dtype=np.float32)
    ty0, tx0 = max(y0, ry0), max(x0, rx0)
    ty1, tx1 = min(y0 + nh, ry1), min(x0 + nw, rx1)
    obj[ty0 - ry0:ty1 - ry0, tx0 - rx0:tx1 - rx0] = nesne_rgb[ty0 - y0:ty1 - y0, tx0 - x0:tx1 - x0]
    alfa[ty0 - ry0:ty1 - ry0, tx0 - rx0:tx1 - rx0] = nesne_alfa[ty0 - y0:ty1 - y0, tx0 - x0:tx1 - x0]
    ic = alfa > 0.5
    if ic.sum() < 16:
        return None
    boyut = float(max(nh, nw))

    arka_lab = renk.rgb_den_lab(arka)
    obj_lab = renk.rgb_den_lab(obj)

    # 3) Çevre halkası: nesnenin hemen dışındaki piksellerin istatistiği.
    halka_yaricap = max(8, int(0.25 * boyut))
    halka = filtre.genislet(ic, halka_yaricap) & ~filtre.genislet(alfa > 0.02, 2)
    if halka.sum() < 100:
        halka = ~filtre.genislet(alfa > 0.02, 2)
    if halka.sum() < 30:
        halka = np.ones_like(ic)

    # 4) Renk / parlaklık aktarımı (Reinhard, kısmi).
    mu_o, sd_o = _agirlikli_istatistik(obj_lab, ic)
    mu_r, sd_r = _agirlikli_istatistik(arka_lab, halka)
    k = ayar.renk
    oran_L = np.clip(sd_r[0] / sd_o[0], 0.7, 1.3) ** (0.5 * k)
    yeni = obj_lab.copy()
    yeni[..., 0] = (obj_lab[..., 0] - mu_o[0]) * oran_L + mu_o[0] + (mu_r[0] - mu_o[0]) * k
    yeni[..., 1:] = obj_lab[..., 1:] + (mu_r[1:] - mu_o[1:]) * (0.4 * k)

    # 5) Işık yönü: sahne parlaklığı düzleminin eğimini nesneye uygula.
    a_egim, b_egim = _isik_duzlemi(arka_lab[..., 0], ~filtre.genislet(alfa > 0.02, 2))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy_, cx_ = np.nonzero(ic)
    gradyan_L = a_egim * (xx - cx_.mean()) + b_egim * (yy - cy_.mean())
    yeni[..., 0] += ayar.isik * np.clip(gradyan_L, -15.0, 15.0)

    # 6) Gren: sahnede olup nesnede eksik kalan gürültüyü ekle.
    sd_sahne = _yuksek_frekans_std(arka_lab[..., 0], halka)
    sd_nesne = _yuksek_frekans_std(yeni[..., 0], ic)
    eksik = max(sd_sahne ** 2 - sd_nesne ** 2, 0.0) ** 0.5 * ayar.gren
    if eksik > 0:
        rng = np.random.default_rng(7)                    # sabit tohum: önizleme titremez
        g = filtre.gauss_bulaniklik(rng.standard_normal((h, w, 3)).astype(np.float32), 0.6)
        g /= g.std() + 1e-6
        yeni += g * np.array([eksik, eksik * 0.4, eksik * 0.4], dtype=np.float32)
    obj_h = renk.lab_den_rgb(yeni)

    # 7) Kenar: alfayı daralt + yumuşat.
    ke = float(ayar.kenar)
    a = alfa
    if ke > 0:
        a = alfa * filtre.daralt(ic, 1)
        a = filtre.gauss_bulaniklik(a, 0.4 + 0.6 * ke)
    a = np.clip(a, 0.0, 1.0)

    # 8) Gölge: ışığın ters yönünde yumuşak + tabanda sıkı temas gölgesi.
    egim = np.hypot(a_egim, b_egim) * boyut
    yon = np.array([a_egim, b_egim]) / (np.hypot(a_egim, b_egim) + 1e-9) if egim > 1.0 else np.array([0.0, 0.2])
    uzunluk = 0.05 * boyut + 2
    dx, dy = int(round(-yon[0] * uzunluk)), int(round(-yon[1] * uzunluk))
    yumusak_golge = filtre.gauss_bulaniklik(_kaydir(alfa, dy, dx), 0.03 * boyut + 1.5)
    temas = filtre.gauss_bulaniklik(_kaydir(alfa, dy // 3, dx // 3), 1.5)
    golge = np.clip(0.65 * yumusak_golge + 0.5 * temas, 0, 1) * ayar.golge * 0.5
    arka_g = arka * (1.0 - golge * (1.0 - a))[..., None]

    # 9) Birleştir.
    if ayar.poisson:
        ic_guvenli = filtre.daralt(a > 0.5, 1)
        if ic_guvenli.any():
            obj_h = poisson.poisson_karistir(arka_g, obj_h, ic_guvenli)
    sonuc = arka_g * (1.0 - a[..., None]) + obj_h * a[..., None]
    return (ry0, rx0, ry1, rx1), np.clip(sonuc * 255.0 + 0.5, 0, 255).astype(np.uint8)
