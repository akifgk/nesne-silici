"""Ağır işleri arayüzü dondurmadan çalıştıran arka plan iş parçacığı."""
from __future__ import annotations

import traceback
from typing import Any, Callable

from PySide6.QtCore import QThread, Signal


class Isci(QThread):
    """``fonksiyon(*argumanlar)``'ı ayrı iş parçacığında çalıştırır.

    Bitince ``bitti(sonuc)``, istisna olursa ``hata(izleme_metni)`` sinyali gelir. Sinyaller
    arayüz iş parçacığına güvenli biçimde iletilir; çağıran slot'ta pencere
    bileşenlerine doğrudan dokunulabilir."""

    bitti = Signal(object)
    hata = Signal(str)

    def __init__(self, fonksiyon: Callable[..., Any], *argumanlar: Any, parent=None):
        super().__init__(parent)
        self._fonksiyon = fonksiyon
        self._argumanlar = argumanlar

    def run(self) -> None:
        try:
            self.bitti.emit(self._fonksiyon(*self._argumanlar))
        except Exception:                                   # noqa: BLE001 - arayüze iletilir
            self.hata.emit(traceback.format_exc())
