"""Nesne algılama, kenar iyileştirme ve kontur testleri."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nesne_silici.cekirdek import kontur, nesne_bulucu  # noqa: E402
from tests.sentetik import iou, sahne  # noqa: E402


def _u8(rgb):
    return (rgb * 255).round().astype(np.uint8)


def test_nesneler_bulunur():
    rgb, gercek = sahne()
    analiz = nesne_bulucu.analiz_et(_u8(rgb))
    assert 3 <= len(analiz.nesneler) <= 8
    for ad, g in gercek.items():
        en_iyi = max(iou(n.maske, g) for n in analiz.nesneler)
        assert en_iyi > 0.9, (ad, en_iyi)


def test_tam_cozunurluk_kenari():
    """Büyük görüntüde (2x) maske yine gerçek kenara oturmalı."""
    rgb, gercek = sahne(720, 960)
    analiz = nesne_bulucu.analiz_et(_u8(rgb))
    for ad, g in gercek.items():
        assert max(iou(n.maske, g) for n in analiz.nesneler) > 0.93, ad


def test_kontur_daire():
    yy, xx = np.mgrid[0:100, 0:100]
    m = (yy - 50) ** 2 + (xx - 50) ** 2 < 30 ** 2
    c = kontur.konturlar(m)
    assert len(c) == 1
    r = np.hypot(c[0][:, 0] - 50, c[0][:, 1] - 50)
    assert abs(r.mean() - 30) < 1.0 and r.std() < 1.0
    assert np.allclose(c[0][0], c[0][-1], atol=1.5) or len(c[0]) > 8


def test_nokta_ile_secim_ve_genisletme():
    rgb, gercek = sahne()
    u8 = _u8(rgb)
    analiz = nesne_bulucu.analiz_et(u8)
    cy, cx = np.argwhere(gercek["dikdortgen"]).mean(0).astype(int)
    n = nesne_bulucu.nokta_ile_nesne(analiz, u8, cx, cy, 99)
    assert n is not None and n.maske[cy, cx]
    buyuk = nesne_bulucu.nesneyi_genislet(analiz, u8, n, yukari=True)
    assert buyuk is None or buyuk.alan > n.alan
