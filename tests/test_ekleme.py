"""Görsel ekleme: kesme ve uyumlama testleri."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nesne_silici.cekirdek import kesme, poisson, renk, uyumlama  # noqa: E402
from tests.sentetik import sahne  # noqa: E402


def _u8(a):
    return (a * 255).round().astype(np.uint8)


def _beyaz_zeminde_elma(h=120, w=140):
    img = np.ones((h, w, 3), np.float32) * 0.97
    yy, xx = np.mgrid[0:h, 0:w]
    daire = (yy - h / 2) ** 2 + (xx - w / 2) ** 2 < 40 ** 2
    img[daire] = (0.85, 0.1, 0.1)
    return img, daire


def test_arka_plan_kaldirma():
    img, daire = _beyaz_zeminde_elma()
    alfa = kesme.arka_plani_kaldir(img)
    ic = alfa > 0.5
    kesisim = (ic & daire).sum() / (ic | daire).sum()
    assert kesisim > 0.95, kesisim


def test_ic_beyaz_korunur():
    """Nesnenin içindeki beyaz (arka plan rengine yakın) bölge silinmemeli."""
    img, daire = _beyaz_zeminde_elma()
    img[55:65, 65:75] = 0.97                      # nesnenin içinde beyaz leke
    alfa = kesme.arka_plani_kaldir(img)
    assert alfa[60, 70] > 0.5


def test_poisson_sinir_ve_ic():
    h = w = 60
    arka = np.full((h, w, 3), 0.3, np.float32)
    on = np.full((h, w, 3), 0.8, np.float32)
    m = np.zeros((h, w), bool)
    m[20:40, 20:40] = True
    sonuc = poisson.poisson_karistir(arka, on, m)
    # Düz kılavuz: iç, sınırdaki arka plan değerine (0.3) çözülür.
    assert abs(sonuc[30, 30].mean() - 0.3) < 0.02


def test_yerlestir_renk_yaklasir():
    arka, _ = sahne(300, 400)
    obj = np.zeros((60, 60, 3), np.float32)
    obj[...] = (1.0, 0.95, 0.9)                    # çok parlak nesne
    alfa = np.ones((60, 60), np.float32)
    g = uyumlama.EklenenGorsel(obj, alfa, 170, 120)
    kutu, yama = uyumlama.yerlestir(_u8(arka), g, uyumlama.UyumAyarlari(poisson=False))
    y0, x0, _, _ = kutu
    ic = yama[120 - y0 + 20:120 - y0 + 40, 170 - x0 + 20:170 - x0 + 40].astype(np.float32) / 255
    ham = np.array([1.0, 0.95, 0.9])
    arka_ort = arka[110:190, 160:240].mean(axis=(0, 1))
    # Uyumlanmış nesne rengi, ham renkten arka plana doğru yaklaşmış olmalı.
    assert np.linalg.norm(ic.mean(axis=(0, 1)) - arka_ort) < np.linalg.norm(ham - arka_ort) - 0.1


def test_yerlestir_tuval_disi():
    arka, _ = sahne(100, 100)
    g = uyumlama.EklenenGorsel(np.ones((10, 10, 3), np.float32), np.ones((10, 10), np.float32), 500, 500)
    assert uyumlama.yerlestir(_u8(arka), g, uyumlama.UyumAyarlari()) is None


def test_yerlestir_kismen_disarida_ve_poisson():
    arka, _ = sahne(200, 300)
    g = uyumlama.EklenenGorsel(np.full((50, 50, 3), 0.6, np.float32), np.ones((50, 50), np.float32), -20, 170)
    sonuc = uyumlama.yerlestir(_u8(arka), g, uyumlama.UyumAyarlari(poisson=True))
    assert sonuc is not None and sonuc[1].dtype == np.uint8
