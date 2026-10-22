"""Nesne Silici: görüntüdeki nesneleri bulan, silen ve yeni görsel ekleyen masaüstü program.

Paket iki katmandan oluşur:

* ``cekirdek``: Saf numpy ile yazılmış görüntü işleme algoritmaları. Arayüzü tanımaz,
  bu yüzden tek başına test edilebilir.
* ``arayuz``: PySide6 ile yazılmış pencere, tuval ve menüler. Algoritmaları çağırır,
  kendisi görüntü işlemez.
"""
