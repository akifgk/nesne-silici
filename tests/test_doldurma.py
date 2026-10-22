"""Doldurma testleri: gerçek arka plan bilindiği için hata sayısal ölçülür."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nesne_silici.cekirdek import doldurma  # noqa: E402
from tests.sentetik import sahne  # noqa: E402


def _u8(a):
    return (a * 255).round().astype(np.uint8)


def _dene(h, w):
    rgb, gercek = sahne(h, w)
    temiz, _ = sahne(h, w, nesne_ciz=False)
    maske = gercek["daire"]
    kutu, yama = doldurma.nesneyi_sil(_u8(rgb), maske)
    y0, x0, y1, x1 = kutu
    sonuc = _u8(rgb).copy()
    sonuc[y0:y1, x0:x1] = yama
    hata = np.abs(sonuc.astype(np.float32) / 255 - temiz)[maske].mean()
    disi = np.abs(sonuc.astype(np.int16) - _u8(rgb).astype(np.int16))[~maske]
    return hata, sonuc, disi


def test_kucuk_goruntu():
    hata, _, _ = _dene(360, 480)
    assert hata < 0.04, hata


def test_buyuk_goruntu_ofset_yolu():
    hata, _, _ = _dene(720, 960)
    assert hata < 0.04, hata


def test_kenara_dayanan_nesne_karanlik_sizdirmaz():
    """Nesne görüntü kenarına değerken doldurma nesnenin renklerini sınır sanmamalı."""
    rgb, gercek = sahne(360, 480)
    maske = np.zeros(rgb.shape[:2], bool)
    maske[250:, 150:320] = True                 # alt kenara dayalı
    rgb = rgb.copy()
    rgb[maske] = (0.02, 0.02, 0.05)             # çok koyu "nesne"
    kutu, yama = doldurma.nesneyi_sil(_u8(rgb), maske)
    y0, x0, y1, x1 = kutu
    ort = yama[maske[y0:y1, x0:x1]].astype(np.float32).mean() / 255
    assert ort > 0.3, ort                       # arka plan açık mavi-gri, koyu değil
