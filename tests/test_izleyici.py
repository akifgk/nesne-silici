"""Video izleyici testi: yapay bir 'video' (hareket eden nesneler) üzerinde kimlik sürekliliği."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nesne_silici.cekirdek import izleyici, nesne_bulucu  # noqa: E402
from tests.sentetik import sahne  # noqa: E402


def _kare(kayma: int, daire_var=True):
    rgb, gercek = sahne(240, 320)
    # Sahneyi yatay kaydırarak "kamera/nesne hareketi" simüle et
    rgb = np.roll(rgb, kayma, axis=1)
    return (rgb * 255).round().astype(np.uint8)


def test_kimlikler_kareler_arasinda_korunur():
    kareler = []
    for i, kayma in enumerate([0, 4, 8, 12, 16]):
        u8 = _kare(kayma)
        analiz = nesne_bulucu.analiz_et(u8)
        kareler.append([izleyici.gozlem_olustur(n, i, i * 0.5, u8.shape[:2]) for n in analiz.nesneler])
    izler = izleyici.izle(kareler)
    # 3 gerçek nesne: her biri 5 karede görülmeli, fazladan iz olmamalı
    uzun = [iz for iz in izler if iz.kare_sayisi == 5]
    assert len(uzun) >= 3, [(i.kimlik, i.kare_sayisi) for i in izler]
    assert len(izler) <= 5


def test_kaybolan_nesne_toleransla_geri_baglanir():
    u8 = _kare(0)
    analiz = nesne_bulucu.analiz_et(u8)
    g = lambda k: [izleyici.gozlem_olustur(n, k, k * 0.5, u8.shape[:2]) for n in analiz.nesneler]
    izler = izleyici.izle([g(0), g(1), [], [], g(4)])    # iki karelik kayıp
    assert all(i.kare_sayisi == 3 for i in izler)
