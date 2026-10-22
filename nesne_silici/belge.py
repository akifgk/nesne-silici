"""Açık belge (görüntü) ve geri al/yinele geçmişi.

Geri almayı tüm görüntünün kopyalarını tutarak yapmak 12 megapikselde adım başına
36 MB eder. Bunun yerine her işlem yalnızca **değişen dikdörtgenin eski ve yeni
halini** saklar (``Adim``). Silme/ekleme zaten bir ``kutu + yama`` ürettiği için bu doğal
bir uyum sağlar. Toplam bellek tavanı aşılırsa en eski adımlar unutulur.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

Kutu = tuple[int, int, int, int]       # y0, x0, y1, x1

GECMIS_BELLEK_TAVANI = 600 * 1024 * 1024    # bayt


@dataclass
class Adim:
    aciklama: str
    kutu: Kutu
    eski: np.ndarray
    yeni: np.ndarray

    @property
    def bayt(self) -> int:
        return self.eski.nbytes + self.yeni.nbytes


class Belge:
    """Düzenlenen görüntü. ``rgb`` her zaman uint8 (h, w, 3); ``alfa`` varsa uint8 (h, w)."""

    def __init__(self, rgb: np.ndarray, alfa: np.ndarray | None = None, yol: Path | None = None):
        # Düzenlenebilir olmalı: salt okunur (örn. Pillow'dan gelen) diziler kopyalanır.
        self.rgb = rgb if rgb.flags.writeable else rgb.copy()
        self.alfa = alfa if alfa is None or alfa.flags.writeable else alfa.copy()
        self.yol = yol
        self._geri: list[Adim] = []
        self._ileri: list[Adim] = []
        self._kayitli_adim = 0          # kaydedildiği andaki geri-yığın uzunluğu
        self._degisim_sayaci = 0        # geri alınıp aynı uzunluğa dönmeyi ayırt etmek için

    # ------------------------------------------------------------------ bilgi
    @property
    def boyut(self) -> tuple[int, int]:
        """(genişlik, yükseklik)"""
        return self.rgb.shape[1], self.rgb.shape[0]

    @property
    def degisti(self) -> bool:
        return self._degisim_sayaci != self._kayitli_adim

    @property
    def geri_alinabilir(self) -> bool:
        return bool(self._geri)

    @property
    def yinelenebilir(self) -> bool:
        return bool(self._ileri)

    @property
    def son_adim_adi(self) -> str:
        return self._geri[-1].aciklama if self._geri else ""

    @property
    def yinelenecek_adim_adi(self) -> str:
        return self._ileri[-1].aciklama if self._ileri else ""

    def analiz_girdisi(self) -> np.ndarray:
        """Nesne algılama için RGB: saydam alanlar nötr griyle doldurulur."""
        if self.alfa is None:
            return self.rgb
        a = (self.alfa.astype(np.float32) / 255.0)[..., None]
        return (self.rgb * a + 128.0 * (1.0 - a) + 0.5).astype(np.uint8)

    # ------------------------------------------------------------------ düzenleme
    def yama_uygula(self, kutu: Kutu, yeni_yama: np.ndarray, aciklama: str) -> None:
        """Dikdörtgeni yeni yamayla değiştirir ve geri alınabilir kaydeder."""
        y0, x0, y1, x1 = kutu
        eski = self.rgb[y0:y1, x0:x1].copy()
        self.rgb[y0:y1, x0:x1] = yeni_yama
        if self.alfa is not None:                # doldurulan/eklenen alan artık opak
            self.alfa[y0:y1, x0:x1] = 255        # (saydam görüntülerde basit, öngörülebilir kural)
        self._geri.append(Adim(aciklama, kutu, eski, yeni_yama.copy()))
        self._ileri.clear()
        self._degisim_sayaci += 1
        self._bellegi_sinirla()

    def geri_al(self) -> Adim | None:
        if not self._geri:
            return None
        adim = self._geri.pop()
        y0, x0, y1, x1 = adim.kutu
        self.rgb[y0:y1, x0:x1] = adim.eski
        self._ileri.append(adim)
        self._degisim_sayaci -= 1
        return adim

    def yinele(self) -> Adim | None:
        if not self._ileri:
            return None
        adim = self._ileri.pop()
        y0, x0, y1, x1 = adim.kutu
        self.rgb[y0:y1, x0:x1] = adim.yeni
        self._geri.append(adim)
        self._degisim_sayaci += 1
        return adim

    def kaydedildi(self, yol: Path | None = None) -> None:
        if yol is not None:
            self.yol = yol
        self._kayitli_adim = self._degisim_sayaci

    def _bellegi_sinirla(self) -> None:
        while len(self._geri) > 1 and sum(a.bayt for a in self._geri) > GECMIS_BELLEK_TAVANI:
            self._geri.pop(0)
            self._kayitli_adim -= 1     # kayıtlı duruma artık dönülemez: "değişti" kalır
