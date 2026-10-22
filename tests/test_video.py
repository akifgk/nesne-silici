"""Video hattı: gerçek Qt Multimedia ile kare okuma + algılama + izleme."""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from PySide6.QtWidgets import QApplication  # noqa: E402

from nesne_silici.arayuz.video_penceresi import VideoPenceresi  # noqa: E402
from ornek_video_uret import avi_yaz, kareler_uret  # noqa: E402


def test_video_nesneleri_listelenir(tmp_path):
    app = QApplication.instance() or QApplication([])
    yol = tmp_path / "ornek.avi"
    avi_yaz(yol, kareler_uret(45), fps=15)                 # 3 sn

    w = VideoPenceresi()
    hatalar = []
    w._hata = lambda m: hatalar.append(m)                  # hata penceresi açılmasın
    w.video_ac(str(yol))
    w._aralik.setValue(0.5)
    w._baslat()
    bitis = time.time() + 90
    while time.time() < bitis and not w._izler and not hatalar:
        app.processEvents()
        time.sleep(0.01)

    assert not hatalar, hatalar
    assert len(w._kareler) >= 5
    assert w._izler and w._liste.count() == len(w._izler)
    assert max(iz.kare_sayisi for iz in w._izler) >= 4     # en az bir nesne birçok karede izlendi
    csv_yolu = tmp_path / "liste.csv"
    w.csv_yaz(csv_yolu)
    satirlar = csv_yolu.read_text(encoding="utf-8-sig").splitlines()
    assert satirlar[0].startswith("nesne;ilk_saniye") and len(satirlar) == len(w._izler) + 1
