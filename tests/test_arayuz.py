"""Uçtan uca arayüz testi (ekransız Qt): gerçek pencere, gerçek iş parçacıkları."""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication  # noqa: E402

from nesne_silici import dosya  # noqa: E402
from nesne_silici.arayuz.ana_pencere import AnaPencere  # noqa: E402
from tests.sentetik import iou, sahne  # noqa: E402

EKRAN_KLASORU = os.environ.get("NSC_EKRAN")


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def uyu(sn: float) -> None:
    """Olay döngüsünü işletip bekler. QTest.qWait kullanılmaz: bekleme boyunca Python'un GIL
    kilidini tutup arka plan iş parçacıklarını aç bırakıyor (gerçek uygulamada app.exec() bırakır)."""
    bitis = time.time() + sn
    while time.time() < bitis:
        QApplication.processEvents()
        time.sleep(0.01)


def bekle(kosul, sn=90.0):
    bitis = time.time() + sn
    while time.time() < bitis:
        uyu(0.04)
        if kosul():
            return
    raise AssertionError("zaman aşımı")


def ekran(w, ad):
    if EKRAN_KLASORU:
        w.grab().save(str(Path(EKRAN_KLASORU) / f"{ad}.png"))


def test_uctan_uca(app, tmp_path):
    rgb, gercek = sahne(480, 640)
    kaynak = tmp_path / "sahne.png"
    Image.fromarray((rgb * 255).round().astype(np.uint8)).save(kaynak)

    w = AnaPencere()
    w.resize(1200, 760)
    w.show()
    d = w.d

    # --- aç + algıla
    d.ac(str(kaynak))
    bekle(lambda: d.analiz is not None and not d.mesgul)
    assert len(d.nesneler) >= 3
    ekran(w, "1_algilandi")

    # --- listeden daireyi seç: kontur çıkmalı
    daire_idx = max(range(len(d.nesneler)), key=lambda i: iou(d.nesneler[i].maske, gercek["daire"]))
    assert iou(d.nesneler[daire_idx].maske, gercek["daire"]) > 0.9
    w.menu._liste.setCurrentRow(daire_idx)
    uyu(0.1)
    assert d.secili == daire_idx and w.tuval._konturlar
    ekran(w, "2_secili_turuncu_kontur")

    # --- tıklamayla başka nesne seç (sarı dikdörtgenin ortası)
    cy, cx = np.argwhere(gercek["dikdortgen"]).mean(0).astype(int)
    d.noktadan_sec(int(cx), int(cy))
    assert d.secili_nesne is not None and d.secili_nesne.maske[cy, cx]

    # --- daireyi sil
    d.sec(daire_idx if daire_idx < len(d.nesneler) else 0)
    onceki = d.belge.rgb.copy()
    d.secileni_sil()
    bekle(lambda: d.belge.geri_alinabilir and not d.mesgul)
    assert not np.array_equal(onceki, d.belge.rgb)
    kalan = d.belge.rgb[gercek["daire"]].astype(np.float32) / 255
    assert kalan[..., 0].mean() < 0.7 and (kalan[..., 2] > kalan[..., 0]).mean() > 0.8   # artık kırmızı değil
    bekle(lambda: d.analiz is not None and not d.mesgul)                                  # otomatik yeniden algılama
    ekran(w, "3_silindi")

    # --- geri al / yinele
    d.geri_al()
    assert np.array_equal(d.belge.rgb, onceki)
    d.yinele()
    assert not np.array_equal(d.belge.rgb, onceki)
    bekle(lambda: d.analiz is not None and not d.mesgul, 30)

    # --- saydam PNG ekle
    ekle = np.zeros((90, 90, 4), np.uint8)
    yy, xx = np.mgrid[0:90, 0:90]
    ekle[..., :3] = (255, 140, 30)
    ekle[..., 3] = (np.hypot(yy - 45, xx - 45) < 40) * 255
    ekle_yol = tmp_path / "top.png"
    Image.fromarray(ekle, "RGBA").save(ekle_yol)
    onceki2 = d.belge.rgb.copy()
    d.gorsel_ekle(str(ekle_yol))
    bekle(lambda: d.ekleme_aktif and not d.mesgul)
    assert w.panel.isVisible() or not w.panel.isHidden()
    d.ekleme.x, d.ekleme.y = 120, 300
    d.ekleme_tasindi()
    bekle(lambda: w.tuval._onizleme is not None, 30)
    ekran(w, "4_ekleme_onizleme")
    d.ekleme_uygula()
    bekle(lambda: not d.ekleme_aktif and not d.mesgul)
    assert not np.array_equal(onceki2, d.belge.rgb)
    bekle(lambda: d.analiz is not None and not d.mesgul, 30)

    # --- farklı kaydet: her biçim
    for uz in (".png", ".jpg", ".webp", ".bmp", ".tiff"):
        hedef = tmp_path / f"cikti{uz}"
        d.farkli_kaydet(str(hedef), 90)
        bekle(lambda: not d.mesgul)
        okunan, _ = dosya.ac(hedef)
        assert okunan.shape == d.belge.rgb.shape
    assert not d.belge.degisti
    w.close()
