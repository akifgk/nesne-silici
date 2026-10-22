"""Deneme için küçük bir video üretir (MJPEG-AVI): hareket eden renkli nesneler.

Kullanım:  python scripts/ornek_video_uret.py cikti.avi

Dış araç (ffmpeg) gerektirmez: AVI kapsayıcısı elle yazılır, kareler Pillow ile JPEG'e
sıkıştırılır. Video penceresini denemek için yeterlidir.
"""
from __future__ import annotations

import io
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.sentetik import sahne  # noqa: E402


def kareler_uret(sayi: int, h: int = 240, w: int = 320):
    """Sabit zemin üstünde yatay kayan sahne; her karede sahnenin kaydırılmış hali."""
    rgb, _ = sahne(h, w)
    for i in range(sayi):
        yield (np.roll(rgb, i * 3, axis=1) * 255).round().astype(np.uint8)


def avi_yaz(yol: str | Path, kareler, fps: int = 15, kalite: int = 90) -> None:
    jpegler = []
    for k in kareler:
        b = io.BytesIO()
        Image.fromarray(k).save(b, format="JPEG", quality=kalite)
        jpegler.append(b.getvalue())
    h, w = k.shape[:2]
    n = len(jpegler)

    def chunk(tur: bytes, veri: bytes) -> bytes:
        return tur + struct.pack("<I", len(veri)) + veri + (b"\0" if len(veri) % 2 else b"")

    def liste(tur: bytes, veri: bytes) -> bytes:
        return b"LIST" + struct.pack("<I", len(veri) + 4) + tur + veri

    en_buyuk = max(len(j) for j in jpegler)
    avih = struct.pack("<IIIIIIIIIIIIII", 1_000_000 // fps, en_buyuk * fps, 0, 0x10, n, 0, 1, en_buyuk, w, h, 0, 0, 0, 0)
    strh = b"vids" + b"MJPG" + struct.pack("<IHHIIIIIIIIhhhh", 0, 0, 0, 0, 1, fps, 0, n, en_buyuk, 0xFFFFFFFF, 0, 0, 0, w, h)
    strf = struct.pack("<IiiHH4sIiiII", 40, w, h, 1, 24, b"MJPG", w * h * 3, 0, 0, 0, 0)
    hdrl = liste(b"hdrl", chunk(b"avih", avih) + liste(b"strl", chunk(b"strh", strh) + chunk(b"strf", strf)))

    movi_veri, idx, konum = b"", b"", 4
    for j in jpegler:
        c = chunk(b"00dc", j)
        idx += b"00dc" + struct.pack("<III", 0x10, konum, len(j))
        movi_veri += c
        konum += len(c)
    govde = b"AVI " + hdrl + liste(b"movi", movi_veri) + chunk(b"idx1", idx)
    Path(yol).write_bytes(b"RIFF" + struct.pack("<I", len(govde)) + govde)


if __name__ == "__main__":
    hedef = sys.argv[1] if len(sys.argv) > 1 else "ornek.avi"
    avi_yaz(hedef, kareler_uret(90))
    print("yazıldı:", hedef)
