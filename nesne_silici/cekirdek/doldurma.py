"""Silinen nesnenin yerini çevresine uygun şekilde doldurma (inpainting).

Doldurma iki katmanda yapılır, çünkü iki ayrı sorun vardır:

* **Yumuşak taban (düşük frekans)**: Parlaklık eğimi, renk geçişi, vinyet gibi yavaş
  değişimler. Başka yerden yama kopyalamak bunu bozar (yama kendi parlaklığını getirir,
  dikiş/blok görünür). Bu yüzden taban, boşluğa sınırdaki değerlerden *pürüzsüzce*
  yayılır (Laplace denklemi, ``poisson.harmonik_doldur``).
* **Detay (yüksek frekans)**: Doku, gren, çizgi, kenar. Bunlar yama kopyalayarak iyi
  üretilir. Detay katmanının ortalaması sıfıra yakın olduğu için kopyalanan yamalar
  renk kayması getirmez.

Detay katmanında kullanılan yama kopyalama algoritması Criminisi ve ark.,
"Region Filling and Object Removal by Exemplar-Based Inpainting" (2004)'dir:

1. **Öncelik**: Boşluğun sınırındaki piksellerden birini seç, ``P = C · D``:
   * ``C`` (güven): yamanın ne kadarı zaten bilinen/doldurulmuş. Güvenilir yerden
     başlamak hataların yayılmasını azaltır.
   * ``D`` (veri): oraya gelen çizgi sınıra ne kadar dik giriyor. Böylece çizgiler
     (ufuk, duvar köşesi) boşluğun içinden devam eder.
2. **Eşleştirme**: Seçilen noktanın çevresindeki yamanın bilinen kısmına en çok benzeyen
   tam bilinen yamayı çevrede bul (kareler farkı toplamı, SSD).
3. **Kopyalama**: Yamanın boş kısmını eşleşenden doldur, güveni güncelle. Boşluk bitene
   dek tekrarla.

Büyük görüntülerde Criminisi küçültülmüş kopyada çalışır; her pikselin nereden
kopyalandığı (ofset) kaydedilir ve aynı kopyalamalar tam çözünürlükte tekrarlanır.
Böylece sonuç bulanık olmaz, gerçek doku detayıyla dolar.
"""
from __future__ import annotations

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from . import filtre, poisson, renk

YAMA_YARI = 4              # yama = 9x9
ARAMA_YARICAPI = 40        # eşleşme önce hedef çevresindeki bu pencerede aranır
CALISMA_BOYUTU = 360       # Criminisi bu uzun kenara indirgenmiş kopyada çalışır
HEDEF_DELIK_PIKSELI = 12000  # çalışma kopyasındaki boşluk bu kadarı geçmesin (hız sınırı)
TABAN_SIGMASI = 2.0        # taban/detay ayrımı (çalışma çözünürlüğünde piksel); küçük = mozaik olmaz

# 5x5 pencerede x ve y yönünde "ağırlıklı fark" çekirdekleri (boşluk normali için)
_DX = np.tile(np.arange(-2, 3, dtype=np.float32), (5, 1))
_DY = _DX.T.copy()


class _Doldurucu:
    """Tek bir (küçük) detay görüntüsü üzerinde Criminisi algoritması."""

    def __init__(self, detay: np.ndarray, delik: np.ndarray):
        self.h, self.w = delik.shape
        self.r = YAMA_YARI
        self.img = detay.astype(np.float32).copy()
        self.img[delik] = 0.0
        self.delik = delik.copy()
        # Kenarı "boş" sayan dolgulu kopya: cephe ve normal hesabı için.
        self.delik_pad = np.pad(self.delik, 2, constant_values=True)
        self.guven = (~delik).astype(np.float32)
        self.ofset = np.zeros((self.h, self.w, 2), dtype=np.int32)   # kaynak - hedef konumu
        self.kopyalandi = np.zeros((self.h, self.w), dtype=bool)
        self.gecerli = self._gecerli_merkezler(delik)
        self.gx = np.zeros((self.h, self.w), dtype=np.float32)
        self.gy = np.zeros((self.h, self.w), dtype=np.float32)
        self.oncelik = np.full((self.h, self.w), -1.0, dtype=np.float32)
        self._gradyan_guncelle(0, self.h, 0, self.w)
        self.gmax = float(np.hypot(self.gx, self.gy).max()) + 1e-6   # veri terimini 0..1'e ölçekler
        self._oncelik_guncelle(0, self.h, 0, self.w)

    # ------------------------------------------------------------------ hazırlık
    def _gecerli_merkezler(self, delik: np.ndarray) -> np.ndarray:
        """Kaynak olabilecek yama merkezleri: pencerede hiç boş piksel olmayanlar."""
        r = self.r
        ok = filtre.kutu_ortalama(delik.astype(np.float32), r) < 1e-6
        ok[:r], ok[-r:], ok[:, :r], ok[:, -r:] = False, False, False, False
        return ok

    def _gradyan_guncelle(self, y0: int, y1: int, x0: int, x1: int) -> None:
        """Parlaklık (kanal 0) gradyanı; yalnızca iki komşusu da bilinen pikselde tanımlıdır."""
        y0, x0 = max(0, y0), max(0, x0)
        y1, x1 = min(self.h, y1), min(self.w, x1)
        a0, b0 = max(0, y0 - 1), max(0, x0 - 1)
        a1, b1 = min(self.h, y1 + 1), min(self.w, x1 + 1)
        Lp = np.pad(self.img[a0:a1, b0:b1, 0], 1)
        bp = np.pad(~self.delik[a0:a1, b0:b1], 1)
        gx = np.where(bp[1:-1, 2:] & bp[1:-1, :-2] & bp[1:-1, 1:-1], (Lp[1:-1, 2:] - Lp[1:-1, :-2]) * 0.5, 0.0)
        gy = np.where(bp[2:, 1:-1] & bp[:-2, 1:-1] & bp[1:-1, 1:-1], (Lp[2:, 1:-1] - Lp[:-2, 1:-1]) * 0.5, 0.0)
        oy, ox = y0 - a0, x0 - b0
        self.gx[y0:y1, x0:x1] = gx[oy:oy + (y1 - y0), ox:ox + (x1 - x0)]
        self.gy[y0:y1, x0:x1] = gy[oy:oy + (y1 - y0), ox:ox + (x1 - x0)]

    # ------------------------------------------------------------------ öncelik
    def _cephe(self, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        """[y0:y1, x0:x1] içinde, en az bir 4-komşusu bilinen boş pikseller (doldurma cephesi)."""
        p = self.delik_pad          # 2 piksel dolgu: gerçek (y, x) -> p[y+2, x+2]
        merkez = p[y0 + 2:y1 + 2, x0 + 2:x1 + 2]
        hepsi_bos = (p[y0 + 1:y1 + 1, x0 + 2:x1 + 2] & p[y0 + 3:y1 + 3, x0 + 2:x1 + 2]
                     & p[y0 + 2:y1 + 2, x0 + 1:x1 + 1] & p[y0 + 2:y1 + 2, x0 + 3:x1 + 3])
        return merkez & ~hepsi_bos

    def _oncelik_guncelle(self, y0: int, y1: int, x0: int, x1: int) -> None:
        y0, x0 = max(0, y0), max(0, x0)
        y1, x1 = min(self.h, y1), min(self.w, x1)
        self.oncelik[y0:y1, x0:x1] = -1.0
        for y, x in zip(*np.nonzero(self._cephe(y0, y1, x0, x1))):
            self.oncelik[y0 + y, x0 + x] = self._oncelik(y0 + int(y), x0 + int(x))

    def _yama_kutusu(self, y: int, x: int) -> tuple[int, int, int, int]:
        """(y, x) etrafındaki yama; görüntü dışına taşıyorsa içeri kaydırılır."""
        r = self.r
        cy = min(max(y, r), self.h - r - 1)
        cx = min(max(x, r), self.w - r - 1)
        return cy - r, cy + r + 1, cx - r, cx + r + 1

    def _oncelik(self, y: int, x: int) -> float:
        a0, a1, b0, b1 = self._yama_kutusu(y, x)
        guven = float(self.guven[a0:a1, b0:b1].mean())
        gx, gy = self.gx[a0:a1, b0:b1], self.gy[a0:a1, b0:b1]
        i = np.unravel_index(int((gx * gx + gy * gy).argmax()), gx.shape)
        izo_x, izo_y = -gy[i], gx[i]                     # izofot: gradyana dik yön
        pencere = self.delik_pad[y:y + 5, x:x + 5].astype(np.float32)
        nx, ny = float((pencere * _DX).sum()), float((pencere * _DY).sum())   # boşluk normali
        n = math.hypot(nx, ny)
        veri = 0.0 if n < 1e-6 else abs(izo_x * nx + izo_y * ny) / n / self.gmax
        return guven * (veri + 0.01)     # +0.01: düz bölgede de güven sırası belirleyici olsun

    # ------------------------------------------------------------------ eşleştirme
    def _en_iyi_kaynak(self, cy: int, cx: int, hedef: np.ndarray,
                       bilinen: np.ndarray) -> tuple[int, int] | None:
        """Hedef yamanın bilinen pikselleriyle SSD'si en küçük geçerli kaynak merkezi."""
        r, P = self.r, 2 * self.r + 1
        yaricap = ARAMA_YARICAPI
        while True:
            y0, y1 = max(r, cy - yaricap), min(self.h - r - 1, cy + yaricap)
            x0, x1 = max(r, cx - yaricap), min(self.w - r - 1, cx + yaricap)
            if y1 >= y0 and x1 >= x0 and self.gecerli[y0:y1 + 1, x0:x1 + 1].any():
                break
            if yaricap > max(self.h, self.w):
                return None
            yaricap *= 2                      # yakında kaynak yoksa arama alanını genişlet
        parca = self.img[y0 - r:y1 + r + 1, x0 - r:x1 + r + 1]
        pencereler = sliding_window_view(parca, (P, P), axis=(0, 1))     # (nh, nw, 3, P, P)
        fark = pencereler - hedef.transpose(2, 0, 1)
        ssd = np.einsum("ijcab,ijcab,ab->ij", fark, fark, bilinen.astype(np.float32))
        ssd = np.where(self.gecerli[y0:y1 + 1, x0:x1 + 1], ssd, np.inf)
        i, j = np.unravel_index(int(ssd.argmin()), ssd.shape)
        return y0 + int(i), x0 + int(j)

    # ------------------------------------------------------------------ ana döngü
    def calistir(self) -> None:
        kalan = int(self.delik.sum())
        r = self.r
        while kalan > 0:
            y, x = divmod(int(self.oncelik.argmax()), self.w)
            if self.oncelik[y, x] < 0:
                break                                       # cephe kalmadı (olağandışı)
            a0, a1, b0, b1 = self._yama_kutusu(y, x)
            cy, cx = (a0 + a1) // 2, (b0 + b1) // 2
            kaynak = self._en_iyi_kaynak(cy, cx, self.img[a0:a1, b0:b1], ~self.delik[a0:a1, b0:b1])
            if kaynak is None:
                break
            ky, kx = kaynak
            yeni = self.delik[a0:a1, b0:b1].copy()          # yamanın hâlâ boş pikselleri
            guven = float(self.guven[a0:a1, b0:b1].mean())
            self.img[a0:a1, b0:b1][yeni] = self.img[ky - r:ky + r + 1, kx - r:kx + r + 1][yeni]
            self.guven[a0:a1, b0:b1][yeni] = guven
            self.ofset[a0:a1, b0:b1][yeni] = (ky - cy, kx - cx)
            self.kopyalandi[a0:a1, b0:b1][yeni] = True
            self.delik[a0:a1, b0:b1][yeni] = False
            self.delik_pad[a0 + 2:a1 + 2, b0 + 2:b1 + 2][yeni] = False
            kalan -= int(yeni.sum())
            # Yalnızca etkilenen pencerede gradyan ve öncelikleri yeniden hesapla.
            g = 2 * r + 2
            self._gradyan_guncelle(a0 - g, a1 + g, b0 - g, b1 + g)
            self._oncelik_guncelle(a0 - g, a1 + g, b0 - g, b1 + g)
        # Eşleşme bulunamayan pikseller sıfır detay (yalnızca taban) olarak kalır.


# --------------------------------------------------------------------------- taban / detay
def _taban_cikar(lab: np.ndarray, delik: np.ndarray, sigma: float) -> np.ndarray:
    """Yumuşak taban: yalnızca bilinen piksellerle (normalleştirilmiş Gauss) bulanıklaştırılır,
    sonra boşluk pürüzsüzce doldurulur."""
    bilinen = (~delik).astype(np.float32)
    pay = filtre.gauss_bulaniklik(lab * bilinen[..., None], sigma)
    payda = filtre.gauss_bulaniklik(bilinen, sigma)[..., None]
    taban = pay / np.maximum(payda, 1e-4)
    return poisson.harmonik_doldur(taban, delik)


# --------------------------------------------------------------------------- genel arayüz
def silme_kutusu(maske: np.ndarray) -> tuple[int, int, int, int]:
    """Doldurma için bağlam içeren çalışma kutusu: nesne + etrafında yeterli çevre."""
    H, W = maske.shape
    ys, xs = np.nonzero(maske)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    pay = int(max(y1 - y0, x1 - x0) * 0.8) + 32
    return (max(0, y0 - pay), max(0, x0 - pay), min(H, y1 + pay), min(W, x1 + pay))


def nesneyi_sil(rgb_u8: np.ndarray, maske: np.ndarray
                ) -> tuple[tuple[int, int, int, int], np.ndarray]:
    """``maske`` ile işaretli alanı siler ve çevreye uygun doldurur.

    Döner: ((y0, x0, y1, x1), yeni_parca_uint8). Yalnızca bu kutu değişir; çağıran
    yamayı belgeye yazar (geri alma için eski yama da saklanır)."""
    y0, x0, y1, x1 = silme_kutusu(maske)
    roi = rgb_u8[y0:y1, x0:x1].astype(np.float32) / 255.0
    m = maske[y0:y1, x0:x1]
    ys, xs = np.nonzero(m)
    boy = max(ys.max() - ys.min(), xs.max() - xs.min()) + 1
    # Anti-alias kenar halesi kalmasın diye maske nesne boyuna göre biraz büyütülür.
    hole = filtre.genislet(m, max(2, int(round(boy * 0.012))))
    yeni = _doldur_roi(roi, hole)
    sonuc = roi.copy()
    sonuc[hole] = yeni[hole]
    return (y0, x0, y1, x1), np.clip(sonuc * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _doldur_roi(roi: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """Çalışma kutusunu doldurur: taban (harmonik) + detay (Criminisi ofsetleriyle)."""
    H, W = hole.shape
    # Küçültme katsayısı: hem görüntü boyutu hem boşluk alanı sınırlanır (hız için).
    s = max(1, -(-max(H, W) // CALISMA_BOYUTU),
            int(math.ceil(math.sqrt(hole.sum() / HEDEF_DELIK_PIKSELI))))
    if s > 1:
        dusuk = filtre.alan_ortalamasi(roi, s)
        delik_d = filtre.alan_ortalamasi(hole.astype(np.float32), s) > 0.0
    else:
        dusuk, delik_d = roi, hole
    lh, lw = delik_d.shape
    lab_d = renk.rgb_den_lab(dusuk)

    # 1) Düşük çözünürlükte taban + detay ayrımı; detay katmanında yama kopyalama.
    taban_d = _taban_cikar(lab_d, delik_d, TABAN_SIGMASI)
    detay_d = (lab_d - taban_d) * (~delik_d)[..., None]
    d = _Doldurucu(detay_d, delik_d)
    d.calistir()

    # 2) Tam çözünürlük: taban bilinear büyütülür, detay gerçek piksellerden kopyalanır.
    taban = filtre.boyutlandir(taban_d, lh * s, lw * s)[:H, :W]
    detay_tam = renk.rgb_den_lab(roi) - taban
    sonuc = taban.copy()

    ys, xs = np.nonzero(hole)
    ly, lx = np.minimum(ys // s, lh - 1), np.minimum(xs // s, lw - 1)
    sy = ys + d.ofset[ly, lx, 0] * s
    sx = xs + d.ofset[ly, lx, 1] * s
    icinde = (sy >= 0) & (sy < H) & (sx >= 0) & (sx < W)
    sy, sx = np.clip(sy, 0, H - 1), np.clip(sx, 0, W - 1)
    uygun = d.kopyalandi[ly, lx] & icinde & ~filtre.genislet(hole, s)[sy, sx]  # kaynak boşluğa değmesin

    sonuc[ys[uygun], xs[uygun]] += detay_tam[sy[uygun], sx[uygun]]
    # Tam çözünürlükte kaynak bulunamayan pikseller düşük çözünürlüklü detayı büyütüp kullanır.
    kalan = ~uygun
    if kalan.any():
        detay_yukari = filtre.boyutlandir(d.img, lh * s, lw * s)[:H, :W]
        sonuc[ys[kalan], xs[kalan]] += detay_yukari[ys[kalan], xs[kalan]]
    return renk.lab_den_rgb(sonuc)
