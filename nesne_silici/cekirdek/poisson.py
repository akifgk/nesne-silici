"""Poisson denklemi çözücüsü: gradyan alanı (kenar/doku) korunarak dikişsiz birleştirme.

Bir ``maske`` bölgesinin içini, **sınırdaki değerler** sabit kalacak ve **iç gradyanlar**
bir kılavuz görüntünün gradyanına olabildiğince yaklaşacak şekilde yeniden hesaplar:

    minimize  Σ ‖∇f − ∇g‖²   (maske içinde),   f = hedef  (maske sınırında)

Çözüm, maske içindeki her piksel için doğrusal bir denklemdir (n = görüntü içindeki
komşu sayısı, genelde 4):

    n·f(p) − Σ f(komşu) = Σ hedef(maske dışı komşu) + (n·g(p) − Σ g(komşu))

Bu büyük ama seyrek bir sistemdir; **eşlenik gradyan (CG)** yöntemiyle çözülür. Matrisi
hiç kurmayız: "matris x vektör" işlemi, komşu piksel toplamlarıyla yapılır.

Görüntü kenarına dayanan maskelerde, kenarın ötesi *bilinmeyen* sayılır ve hiçbir şey
dayatmaz (Neumann koşulu): komşu sayısı n kenarda 3 ya da 2 olur. Aksi halde nesne
kenara değdiğinde, nesnenin kendi pikselleri sınır değeri sanılırdı.

İki kullanım:

* ``poisson_karistir``: yapıştırılan nesnenin rengi arka plana, gradyanları nesneden
  gelecek şekilde birleşir (görsel ekleme).
* ``harmonik_doldur``: kılavuz yok (gradyan 0) → boşluk, sınırdan pürüzsüzce doldurulur
  (Laplace denklemi). Doldurmadaki "yumuşak taban" budur.
"""
from __future__ import annotations

import numpy as np

_YONLER = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _kaydir(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """``a`` dizisini (dy, dx) kaydırır; boşalan kenar sıfırla dolar (sarmalama yok)."""
    sonuc = np.zeros_like(a)
    h, w = a.shape[:2]
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    sonuc[yd, xd] = a[ys, xs]
    return sonuc


def _komsu_toplam(a: np.ndarray) -> np.ndarray:
    """Görüntü içindeki komşuların toplamı (dışarısı 0)."""
    return sum(_kaydir(a, dy, dx) for dy, dx in _YONLER)


def _komsu_sayisi(sekil: tuple[int, int]) -> np.ndarray:
    return _komsu_toplam(np.ones(sekil, dtype=np.float32))


def _eslenik_gradyan(maske: np.ndarray, derece: np.ndarray, b: np.ndarray,
                     baslangic: np.ndarray, tekrar: int, tol: float) -> np.ndarray:
    """(derece·I − komşu) · x = b sistemini maske içinde çözer (tek kanal)."""
    m = maske.astype(np.float32)

    def A(x: np.ndarray) -> np.ndarray:
        return (derece * x - _komsu_toplam(x)) * m

    x = baslangic * m
    r = b - A(x)
    p = r.copy()
    rs = float((r * r).sum())
    hedef = tol * tol * max(float((b * b).sum()), 1e-12)
    for _ in range(tekrar):
        if rs <= hedef:
            break
        Ap = A(p)
        alfa = rs / max(float((p * Ap).sum()), 1e-20)
        x += alfa * p
        r -= alfa * Ap
        rs_yeni = float((r * r).sum())
        p = r + (rs_yeni / rs) * p
        rs = rs_yeni
    return x


def _kirp(maske: np.ndarray, pay: int = 2) -> tuple[slice, slice]:
    ys, xs = np.nonzero(maske)
    return (slice(max(0, ys.min() - pay), ys.max() + 1 + pay),
            slice(max(0, xs.min() - pay), xs.max() + 1 + pay))


def _coz(hedef: np.ndarray, kilavuz: np.ndarray | None, maske: np.ndarray,
         karisik: bool, tekrar: int, tol: float) -> np.ndarray:
    """Ortak çözücü. ``kilavuz`` None ise gradyan 0 (Laplace)."""
    sonuc = hedef.astype(np.float32, copy=True)
    if not maske.any():
        return sonuc
    ky, kx = _kirp(maske)
    t = hedef[ky, kx].astype(np.float32)
    g = None if kilavuz is None else kilavuz[ky, kx].astype(np.float32)
    m = maske[ky, kx]
    dis = ~m
    derece = _komsu_sayisi(m.shape)
    if not dis.any():                       # tüm kesit maske: sınır değeri yok
        return sonuc
    for c in range(t.shape[2]):
        tc = t[..., c]
        b = _komsu_toplam(tc * dis)                       # sınır koşulu: maske dışı komşular
        if g is not None:
            gc = g[..., c]
            if karisik:
                # Her yönde arka plan ve ön plan gradyanından güçlü olan seçilir.
                for dy, dx in _YONLER:
                    gecerli = _kaydir(np.ones(m.shape, np.float32), dy, dx)
                    gg = (gc - _kaydir(gc, dy, dx)) * gecerli
                    gt = (tc - _kaydir(tc, dy, dx)) * gecerli
                    b = b + np.where(np.abs(gg) > np.abs(gt), gg, gt)
            else:
                b = b + derece * gc - _komsu_toplam(gc)   # kılavuzun Laplace'ı
        b = b * m
        # İyi bir başlangıç çözümü yakınsamayı hızlandırır.
        ilk = gc if g is not None else np.full(m.shape, float(tc[dis].mean()), np.float32)
        x = _eslenik_gradyan(m, derece, b, ilk, tekrar, tol)
        t[..., c] = np.where(m, x, tc)
    sonuc[ky, kx] = t
    return sonuc


def harmonik_doldur(img: np.ndarray, delik: np.ndarray, tekrar: int = 1500,
                    tol: float = 1e-3) -> np.ndarray:
    """``delik``i, çevresindeki değerlerden pürüzsüz (Laplace) biçimde doldurur."""
    return _coz(img, None, delik, False, tekrar, tol)


def poisson_karistir(arka: np.ndarray, on: np.ndarray, maske: np.ndarray,
                     karisik_gradyan: bool = False, tekrar: int = 600,
                     tol: float = 1e-3) -> np.ndarray:
    """``on`` görüntüsünün ``maske`` içindeki kısmını ``arka`` üzerine dikişsiz yapıştırır.

    Renk/parlaklık sınırda arka plana uyar; kenar ve doku ``on``dan gelir.
    ``karisik_gradyan=True`` iken her yerde iki görüntüden *güçlü olan* gradyan alınır
    (ince çizgili/saydam nesnelerde arka plan dokusunu da korur)."""
    return _coz(arka, on, maske, karisik_gradyan, tekrar, tol)
