"""Eklenecek görselin arka planını temizleme (alfa kanalı üretme).

PNG gibi zaten saydamlığı olan dosyalarda bu adım gerekmez; alfa dosyadan gelir.
Düz renkli arka planlı bir fotoğrafta ise şöyle yapılır:

1. Görüntünün **çerçevesindeki** piksellerin medyan rengi "arka plan rengi" sayılır.
2. Bu renge Lab uzayında yakın pikseller "arka plan adayı" olur.
3. Adaylardan yalnızca **görüntü kenarına bağlı olanlar** gerçek arka plandır. İçerideki,
   nesnenin kendi parçası olan açık renkli bölgeler (gözün beyazı gibi) bu sayede korunur.
4. Arka planın hemen yanındaki piksellerde renk uzaklığına göre yumuşak saydamlık verilir
   (kenarda arka plan renginde saçak kalmaz).
"""
from __future__ import annotations

import numpy as np

from . import filtre, renk


def arka_plani_kaldir(rgb: np.ndarray) -> np.ndarray:
    """Döner: ``(h, w)`` alfa (0 = arka plan, 1 = nesne). Ayırt edilemezse hepsi 1."""
    h, w = rgb.shape[:2]
    lab = renk.rgb_den_lab(rgb)
    cerceve = np.zeros((h, w), dtype=bool)
    cerceve[:2], cerceve[-2:], cerceve[:, :2], cerceve[:, -2:] = True, True, True, True
    arka_renk = np.median(lab[cerceve], axis=0)
    uzaklik = np.linalg.norm(lab - arka_renk, axis=-1)

    # Çerçevenin kendi rengi ne kadar tutarlıysa eşik o kadar dar olabilir.
    tol = float(np.clip(2.0 * np.percentile(uzaklik[cerceve], 90) + 4.0, 8.0, 30.0))
    aday = uzaklik < tol
    etiket, n = filtre.etiketle(aday)
    if n == 0:
        return np.ones((h, w), dtype=np.float32)
    kenar_etiketleri = np.unique(np.concatenate([etiket[0], etiket[-1], etiket[:, 0], etiket[:, -1]]))
    arka = np.isin(etiket, kenar_etiketleri[kenar_etiketleri > 0])

    on = ~arka
    oran = on.mean()
    if oran < 0.02 or oran > 0.98:        # arka plan/nesne ayrımı bulunamadı
        return np.ones((h, w), dtype=np.float32)

    alfa = on.astype(np.float32)
    kenar_seridi = on & filtre.genislet(arka, 2)
    yumusak = np.clip((uzaklik - 0.5 * tol) / (0.5 * tol), 0.0, 1.0)
    alfa[kenar_seridi] = yumusak[kenar_seridi]

    # Küçük kırıntıları at: en büyük parçanın %2'sinden küçük adacıkları sil.
    etiket2, n2 = filtre.etiketle(alfa > 0.5)
    if n2 > 1:
        sayim = np.bincount(etiket2.ravel())
        sayim[0] = 0
        tut = np.nonzero(sayim >= 0.02 * sayim.max())[0]
        alfa[~np.isin(etiket2, tut) & (alfa > 0)] = 0.0
    return np.clip(filtre.gauss_bulaniklik(alfa, 0.7), 0.0, 1.0)


def kirp(rgb: np.ndarray, alfa: np.ndarray, esik: float = 0.02
         ) -> tuple[np.ndarray, np.ndarray]:
    """Nesne dışındaki boş kenarları keser."""
    ys, xs = np.nonzero(alfa > esik)
    if len(ys) == 0:
        return rgb, alfa
    s = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    return rgb[s], alfa[s]
