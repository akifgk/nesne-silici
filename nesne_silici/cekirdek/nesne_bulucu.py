"""Birleştirme ağacındaki düğümlerden "nesne" olanları seçer.

Eğitilmiş model olmadığı için "bu bir kedi" denemez. Bunun yerine bir bölgenin
*nesne gibi davranıp davranmadığı* ölçülür:

* **Sınırı belirgin mi?** Nesnenin çevresinde güçlü renk/kenar farkı olmalı.
* **İçi sakin mi?** Bir bütünün içinde çok kenar yoksa birden fazla parçaya bölünmemiştir.
* **Arka plan mı?** Görüntü kenarına geniş ölçüde yaslanan büyük bölgeler arka plandır.
* **Makul büyüklük ve biçimde mi?** Birkaç pikselden ibaret ya da ip gibi ince şeyler elenir.

Düğümler puanlanır, en yüksek puandan başlayarak birbiriyle çakışmayanlar seçilir.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import filtre, maske_iyilestir, renk
from .bolutleme import BolgeAgaci, agac_kur, slic


@dataclass
class Nesne:
    """Algılanan tek bir nesne. Maske ve kutu tam çözünürlüktedir."""
    kimlik: int
    ad: str
    maske: np.ndarray                      # (H, W) bool, tam çözünürlük
    kutu: tuple[int, int, int, int]        # y0, x0, y1, x1
    puan: float
    ortalama_lab: tuple[float, float, float]
    dugum: int = -1                        # ağaç düğümü (ince ayar için)

    @property
    def alan(self) -> int:
        return int(self.maske.sum())


@dataclass
class Analiz:
    """Bir görüntünün bölütleme sonucu; ince ayar ve yeni nesne seçimi için saklanır."""
    agac: BolgeAgaci
    lab_kucuk: np.ndarray
    olcek: float                           # tam çözünürlük / çalışma çözünürlüğü
    kenar_olcek: float
    nesneler: list[Nesne]


CALISMA_BOYUTU = 384


def kucult(rgb_u8: np.ndarray, uzun_kenar: int = CALISMA_BOYUTU) -> tuple[np.ndarray, float]:
    """Çalışma çözünürlüğüne indirger. (küçük_rgb_float, tam/küçük oranı) döndürür."""
    h, w = rgb_u8.shape[:2]
    oran = max(h, w) / uzun_kenar
    if oran <= 1:
        return rgb_u8.astype(np.float32) / 255.0, 1.0
    faktor = int(oran)
    img = filtre.alan_ortalamasi(rgb_u8, faktor) / 255.0 if faktor > 1 else rgb_u8.astype(np.float32) / 255.0
    yeni_h, yeni_w = round(h * uzun_kenar / max(h, w)), round(w * uzun_kenar / max(h, w))
    img = filtre.boyutlandir(img, yeni_h, yeni_w)
    return img.astype(np.float32), h / yeni_h


def _ozellikler(agac: BolgeAgaci, dugum: int, kenar: np.ndarray, kenar_olcek: float):
    """Düğümün nesnelik ölçütlerini hesaplar. Yalnızca sınır kutusunda çalışır (hızlı)."""
    m, (y0, x0, y1, x1) = agac.kirpik_maske(dugum)
    H, W = agac.etiket.shape
    alan = int(m.sum())
    if alan < 12:
        return None
    sinir = filtre.sinir(m)
    ic = filtre.daralt(m, 2)
    k = kenar[y0:y1, x0:x1]
    sinir_gucu = float(k[sinir].mean()) / kenar_olcek if sinir.any() else 0.0
    ic_gucu = float(k[ic].mean()) / kenar_olcek if ic.any() else 0.0
    # Görüntü çerçevesine temas: kutu çerçeveye değiyorsa temas eden piksel sayısı.
    temas = 0
    if y0 == 0:
        temas += int(m[0].sum())
    if y1 == H:
        temas += int(m[-1].sum())
    if x0 == 0:
        temas += int(m[:, 0].sum())
    if x1 == W:
        temas += int(m[:, -1].sum())
    cevre = int(sinir.sum()) + 1
    kenar_temas = temas / cevre
    kompaktlik = 4 * math.pi * alan / (cevre * cevre)
    return alan, sinir_gucu, ic_gucu, kenar_temas, kompaktlik


def analiz_et(rgb_u8: np.ndarray, en_fazla: int = 16, min_alan: float = 0.004,
              maks_alan: float = 0.55, goreli_esik: float = 0.45) -> Analiz:
    """Görüntüyü bölütler, ağacı kurar ve nesneleri seçer.

    ``goreli_esik`` küçüldükçe daha silik nesneler de listeye girer (duyarlılık artar)."""
    kucuk, olcek = kucult(rgb_u8)
    lab = renk.rgb_den_lab(kucuk)
    lab_yumusak = filtre.gauss_bulaniklik(lab, 1.0)
    kenar = filtre.kenar_gucu(lab_yumusak)
    kenar_olcek = float(np.percentile(kenar, 95)) + 1e-3

    etiket = slic(lab_yumusak, parca_sayisi=700)
    agac = agac_kur(lab_yumusak, etiket)
    toplam_alan = lab.shape[0] * lab.shape[1]

    adaylar = []
    for d in range(agac.n_dugum):
        oran = agac.alan[d] / toplam_alan
        if not (min_alan <= oran <= maks_alan):
            continue
        ozellik = _ozellikler(agac, d, kenar, kenar_olcek)
        if ozellik is None:
            continue
        alan, sinir_gucu, ic_gucu, kenar_temas, kompaktlik = ozellik
        if kenar_temas > 0.30 or kompaktlik < 0.03:
            continue
        # Puan: net sınır iyi, kalabalık iç kötü, büyük bütünler küçük parçalara tercih edilir.
        buyukluk = math.log(oran / min_alan) / math.log(maks_alan / min_alan)
        puan = sinir_gucu - 0.7 * ic_gucu + 0.35 * buyukluk + 0.25 * min(kompaktlik, 1.0)
        adaylar.append((puan, d))

    adaylar.sort(reverse=True)
    # Eşik göreli: görüntüdeki en belirgin nesnenin puanının bir oranı. Mutlak bir sayı
    # her fotoğrafta (pürüzlü/sade) aynı anlama gelmediği için böyle yapılır.
    en_iyi = adaylar[0][0] if adaylar else 0.0
    esik = max(0.9, goreli_esik * en_iyi)
    secilen: list[tuple[float, int, np.ndarray]] = []
    for puan, d in adaylar:
        if puan < esik or len(secilen) >= en_fazla:
            continue
        m = agac.maske(d)
        cakisti = False
        for _, _, diger in secilen:
            kesisim = int((m & diger).sum())
            if kesisim > 0.15 * min(int(m.sum()), int(diger.sum())):
                cakisti = True
                break
        if not cakisti:
            secilen.append((puan, d, m))

    # Büyükten küçüğe, soldan sağa sıralanmış kalıcı numaralar
    secilen.sort(key=lambda t: -agac.alan[t[1]])
    nesneler = [
        nesne_olustur(i + 1, d, puan, agac, lab_yumusak, rgb_u8, olcek)
        for i, (puan, d, _) in enumerate(secilen)
    ]
    return Analiz(agac, lab_yumusak, olcek, kenar_olcek, nesneler)


def nesne_olustur(kimlik: int, dugum: int, puan: float, agac: BolgeAgaci,
                  lab_kucuk: np.ndarray, rgb_u8: np.ndarray, olcek: float) -> Nesne:
    """Ağaç düğümünden tam çözünürlüklü, kenarı iyileştirilmiş nesne üretir."""
    kaba = agac.maske(dugum)
    tam, kutu = maske_iyilestir.tam_cozunurluge_tasi(kaba, rgb_u8, olcek)
    lab_ort = lab_kucuk[kaba].mean(axis=0)
    return Nesne(kimlik, f"Nesne {kimlik}", tam, kutu, float(puan),
                 (float(lab_ort[0]), float(lab_ort[1]), float(lab_ort[2])), dugum)


def nokta_ile_nesne(analiz: Analiz, rgb_u8: np.ndarray, x: int, y: int, kimlik: int,
                    seviye: int = 0) -> Nesne | None:
    """Tıklanan noktadaki nesneyi ağaçtan seçer (tam çözünürlük koordinatı verilir).

    ``seviye`` ağaçtaki yükseliş sayısıdır: 0 = o noktayı içeren en küçük *anlamlı*
    bölge, her artış nesneyi bir adım büyütür (üst bölgeyle birleştirir)."""
    h, w = analiz.agac.etiket.shape
    kx = int(np.clip(x / analiz.olcek, 0, w - 1))
    ky = int(np.clip(y / analiz.olcek, 0, h - 1))
    zincir = analiz.agac.nokta_zinciri(kx, ky)
    toplam = h * w
    uygun = [d for d in zincir if analiz.agac.alan[d] >= 0.004 * toplam
             and analiz.agac.alan[d] <= 0.7 * toplam]
    if not uygun:
        return None
    d = uygun[min(seviye, len(uygun) - 1)]
    return nesne_olustur(kimlik, d, 0.0, analiz.agac, analiz.lab_kucuk, rgb_u8, analiz.olcek)


def nesneyi_genislet(analiz: Analiz, rgb_u8: np.ndarray, nesne: Nesne, yukari: bool) -> Nesne | None:
    """Seçili nesneyi ağaçta bir adım büyütür (yukarı) ya da küçültür (aşağı)."""
    agac = analiz.agac
    toplam = agac.etiket.size
    if yukari:
        d = agac.ebeveyn[nesne.dugum]
        while d != -1 and agac.alan[d] < agac.alan[nesne.dugum] * 1.15:
            d = agac.ebeveyn[d]
        if d == -1 or agac.alan[d] > 0.7 * toplam:
            return None
    else:
        if nesne.dugum not in agac.cocuklar:
            return None
        # Büyük çocuğu seçmek yerine en az bölgenin yarısını koruyan çocuğu al.
        a, b = agac.cocuklar[nesne.dugum]
        d = a if agac.alan[a] >= agac.alan[b] else b
        while agac.alan[d] < 0.004 * toplam and d in agac.cocuklar:
            d = max(agac.cocuklar[d], key=lambda c: agac.alan[c])
        if agac.alan[d] < 0.004 * toplam:
            return None
    return nesne_olustur(nesne.kimlik, d, nesne.puan, agac, analiz.lab_kucuk, rgb_u8, analiz.olcek)
