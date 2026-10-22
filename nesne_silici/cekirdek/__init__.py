"""Görüntü işleme çekirdeği (yalnızca numpy).

Ortak sözleşme (tüm modüller buna uyar):

* Renkli görüntü: ``float32``, şekil ``(yükseklik, genişlik, 3)``, RGB, değer aralığı 0..1.
* Maske: ``bool``, şekil ``(yükseklik, genişlik)``, ``True`` = "bu piksel nesnenin içinde".
* Koordinat: dizi indeksi her zaman ``[y, x]``; nokta/konum ise ``(x, y)`` yazılır.
"""
