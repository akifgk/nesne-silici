"""Dosya G/Ç (tüm formatlar) ve belge geçmişi testleri."""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nesne_silici import belge, dosya  # noqa: E402


def _rgb(h=40, w=60):
    rng = np.random.default_rng(0)
    y = np.linspace(0, 255, h, dtype=np.uint8)[:, None, None]
    return np.clip(y + rng.integers(0, 20, (h, w, 3)), 0, 255).astype(np.uint8)


@pytest.mark.parametrize("uzanti", [".png", ".bmp", ".tiff", ".webp"])
def test_kayipsiz_gidis_donus(tmp_path, uzanti):
    a = _rgb()
    yol = tmp_path / f"a{uzanti}"
    dosya.kaydet(yol, a, None, kalite=100)
    b, alfa = dosya.ac(yol)
    assert alfa is None and b.shape == a.shape
    if uzanti != ".webp":                          # webp kayıplı
        assert np.array_equal(a, b)


def test_jpeg_yaklasik(tmp_path):
    a = _rgb()
    dosya.kaydet(tmp_path / "a.jpg", a, None, 95)
    b, _ = dosya.ac(tmp_path / "a.jpg")
    assert np.abs(a.astype(int) - b.astype(int)).mean() < 6


def test_alfa_png_korunur_jpg_beyaza_yedirilir(tmp_path):
    a = _rgb()
    alfa = np.zeros(a.shape[:2], np.uint8)
    alfa[:, 30:] = 255
    dosya.kaydet(tmp_path / "x.png", a, alfa)
    _, a2 = dosya.ac(tmp_path / "x.png")
    assert a2 is not None and np.array_equal(a2, alfa)
    dosya.kaydet(tmp_path / "x.jpg", a, alfa)
    b, a3 = dosya.ac(tmp_path / "x.jpg")
    assert a3 is None and b[:, :20].mean() > 240   # saydam yarı beyaza döndü


def test_gif_ico_ppm(tmp_path):
    a = _rgb()
    for uz in (".gif", ".ico", ".ppm"):
        dosya.kaydet(tmp_path / f"a{uz}", a)
        b, _ = dosya.ac(tmp_path / f"a{uz}")
        assert b.ndim == 3 and b.shape[2] == 3


def test_gri_ve_16bit_okunur(tmp_path):
    Image.fromarray(np.arange(2400, dtype=np.uint8).reshape(40, 60), "L").save(tmp_path / "g.png")
    b, _ = dosya.ac(tmp_path / "g.png")
    assert b.shape == (40, 60, 3)
    Image.fromarray(np.arange(2400, dtype=np.uint16).reshape(40, 60) * 20).save(tmp_path / "s.png")
    b, _ = dosya.ac(tmp_path / "s.png")
    assert b.shape == (40, 60, 3) and b.max() == 255


def test_bilinmeyen_uzanti():
    with pytest.raises(ValueError):
        dosya.kaydet("x.xyz", _rgb())


def test_acma_filtresi_ana_bicimleri_icerir():
    f = dosya.acma_filtresi()
    for u in ("*.png", "*.jpg", "*.webp", "*.bmp", "*.tiff"):
        assert u in f


def test_gecmis_geri_al_yinele():
    b = belge.Belge(np.zeros((50, 50, 3), np.uint8))
    assert not b.degisti
    b.yama_uygula((10, 10, 20, 20), np.full((10, 10, 3), 200, np.uint8), "boya")
    assert b.degisti and b.rgb[15, 15, 0] == 200
    b.geri_al()
    assert b.rgb[15, 15, 0] == 0 and not b.degisti
    b.yinele()
    assert b.rgb[15, 15, 0] == 200 and b.degisti
    b.kaydedildi()
    assert not b.degisti
    b.geri_al()
    assert b.degisti


def test_gecmis_yeni_islem_ileriyi_siler():
    b = belge.Belge(np.zeros((30, 30, 3), np.uint8))
    b.yama_uygula((0, 0, 5, 5), np.ones((5, 5, 3), np.uint8), "bir")
    b.geri_al()
    b.yama_uygula((0, 0, 5, 5), np.full((5, 5, 3), 2, np.uint8), "iki")
    assert not b.yinelenebilir


def test_acilan_goruntu_yazilabilir(tmp_path):
    """Pillow'dan gelen dizi salt okunur olabilir; belge düzenlenebilir olmalı."""
    dosya.kaydet(tmp_path / "a.png", _rgb())
    rgb, _ = dosya.ac(tmp_path / "a.png")
    assert rgb.flags.writeable
    b = belge.Belge(rgb)
    b.yama_uygula((0, 0, 4, 4), np.zeros((4, 4, 3), np.uint8), "x")
