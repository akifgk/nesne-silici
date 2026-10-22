"""Renk, filtre ve bölütleme temellerinin testleri. Çalıştırma: python -m pytest tests"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nesne_silici.cekirdek import bolutleme, filtre, renk  # noqa: E402
from tests.sentetik import iou, sahne  # noqa: E402


def test_lab_gidis_donus():
    rng = np.random.default_rng(0)
    rgb = rng.random((20, 20, 3)).astype(np.float32)
    geri = renk.lab_den_rgb(renk.rgb_den_lab(rgb))
    assert np.abs(geri - rgb).max() < 2e-3


def test_lab_bilinen_degerler():
    lab = renk.rgb_den_lab(np.array([[[1, 1, 1], [0, 0, 0]]], np.float32))
    assert abs(lab[0, 0, 0] - 100) < 0.1 and abs(lab[0, 1, 0]) < 0.1


def test_gauss_toplam_korunur():
    x = np.random.default_rng(1).random((30, 40)).astype(np.float32)
    assert abs(filtre.gauss_bulaniklik(x, 2.0).mean() - x.mean()) < 0.01


def test_kutu_ortalama_dogru():
    x = np.random.default_rng(2).random((20, 20)).astype(np.float32)
    k = filtre.kutu_ortalama(x, 1)
    assert abs(k[10, 10] - x[9:12, 9:12].mean()) < 1e-5


def test_etiketle_iki_bilesen():
    m = np.zeros((10, 10), bool)
    m[1:4, 1:4] = True
    m[6:9, 6:9] = True
    _, n = filtre.etiketle(m)
    assert n == 2
    _, n8 = filtre.etiketle(np.eye(5, dtype=bool), sekiz_komsu=True)
    assert n8 == 1


def test_delik_doldur():
    m = np.zeros((20, 20), bool)
    m[3:17, 3:17] = True
    m[8:12, 8:12] = False
    assert filtre.delikleri_doldur(m)[10, 10]


def test_morfoloji():
    m = np.zeros((21, 21), bool)
    m[10, 10] = True
    assert filtre.genislet(m, 3).sum() == 49
    assert filtre.daralt(filtre.genislet(m, 3), 3).sum() == 1


def test_boyutlandir_boyut():
    x = np.random.default_rng(3).random((50, 80, 3)).astype(np.float32)
    assert filtre.boyutlandir(x, 25, 40).shape == (25, 40, 3)


def test_bolutleme_nesneleri_ayirir():
    rgb, gercek = sahne()
    lab = renk.rgb_den_lab(rgb)
    etiket = bolutleme.slic(lab, 500)
    agac = bolutleme.agac_kur(lab, etiket)
    assert agac.ebeveyn[agac.n_dugum - 1] == -1
    # Her gerçek nesne için ağaçta onu iyi kapsayan bir düğüm bulunmalı.
    for ad, g in gercek.items():
        en_iyi = max(iou(agac.maske(d), g) for d in range(agac.n_dugum)
                     if agac.alan[d] < 0.6 * g.size)
        assert en_iyi > 0.85, (ad, en_iyi)
