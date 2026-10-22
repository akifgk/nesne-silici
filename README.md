# Nesne Silici

Görüntüdeki nesneleri **hazır bir yapay zekâ modeli kullanmadan**, tamamen elle yazılmış
görüntü işleme algoritmalarıyla bulan, silen ve yeni görsel ekleyen masaüstü programı.

* Fotoğraf aç → nesneler sağ üst köşedeki **sabit menüde** listelenir.
* Menüden (ya da görüntüde tıklayarak) nesne seç → etrafı **turuncu kontürle** işaretlenir.
* **Sil** → nesne kalkar, arkası çevreye uygun şekilde doldurulur.
* **Görsel ekle** → başka bir görsel sahnenin **ışık, renk, gren ve gölgesine** uydurularak eklenir.
* **Farklı kaydet** (PNG, JPG, WEBP, BMP, TIFF, GIF, ICO, HEIC) ve tüm bu biçimleri **açma**.
* **Video**: karelerden nesneleri algılayıp zaman damgasıyla listeler (silme/ekleme yok), CSV'ye aktarır.

## Çalıştırma

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
baslat.bat                       # ya da:  .venv\Scripts\python -m nesne_silici [dosya]
```

Testler: `.venv\Scripts\python -m pytest tests`

Videoyu denemek için örnek video: `python scripts/ornek_video_uret.py ornek.avi`

## Dürüst sınırlar

Model olmadığı için program nesnenin **ne olduğunu** (kedi, araba…) bilmez; "arkasından
ayrışan bir bütün" arar. Bu yüzden:

* Sade, kontrastlı sahnelerde iyi çalışır. Kalabalık ve dokulu sahnelerde nesneyi
  eksik/fazla alabilir. Bu durumda **Genişlet / Daralt** ile ağaçtaki bir üst/alt bölgeye
  geçilir ya da görüntüde **doğrudan tıklanır**; **Duyarlılık** kaydırıcısı eşiği ayarlar.
* Doldurma; gökyüzü, duvar, çim, masa gibi düzgün ve dokulu zeminlerde çok iyidir. Çizgisel
  desen/yazı/yüz gibi karmaşık yapıların boşluktan geçtiği yerlerde yapay görünebilir.
* Gerçek bir "ışık" bilgisi yoktur; ışık yönü sahne parlaklığına uydurulan düzlemin
  eğiminden tahmin edilir. Çok karmaşık ışıklandırmada tahmin yanılabilir.

## Mimari

```
nesne_silici/
  cekirdek/          arayüzden bağımsız, yalnızca numpy ile yazılmış algoritmalar
    renk.py            sRGB <-> Lab
    filtre.py          Gauss, kutu ortalaması, Sobel, yeniden boyutlandırma, morfoloji, etiketleme
    bolutleme.py       SLIC süperpiksel + Ward maliyetli birleştirme ağacı
    nesne_bulucu.py    ağaçtan "nesne gibi davranan" bölgeleri seçer (sınır gücü, iç sakinlik, kenar teması)
    maske_iyilestir.py kaba maskeyi tam çözünürlükte gerçek kenara oturtur
    kontur.py          marching squares + Douglas-Peucker
    poisson.py         eşlenik gradyanla Poisson/Laplace çözücü (Neumann kenar koşulu)
    doldurma.py        taban (harmonik) + detay (Criminisi örnek tabanlı) doldurma
    kesme.py           eklenen görselin arka planını temizleme
    uyumlama.py        renk (Reinhard), ışık düzlemi, gren, gölge, kenar
    izleyici.py        videoda kareler arası nesne eşleştirme
  arayuz/            PySide6: tuval, nesne menüsü, ekleme paneli, denetleyici, video penceresi
  dosya.py           tüm formatları okuma/yazma (Pillow codec)
  belge.py           açık belge + yama tabanlı geri al/yinele
tests/               sentetik sahnelerle algoritma ve uçtan uca arayüz testleri
```

Çekirdek arayüzü tanımaz; arayüz görüntü işlemez. Her algoritma dosyasının başında "ne
yapıyor, neden böyle" açıklaması vardır. Okumaya `cekirdek/renk.py` ve `filtre.py`'den
başlayıp `bolutleme.py` → `nesne_bulucu.py` → `doldurma.py` sırasıyla ilerlemek en kolay yoldur.

## Kısayollar

Ctrl+O aç · Ctrl+S kaydet · Ctrl+Shift+S farklı kaydet · Ctrl+Z / Ctrl+Y geri al/yinele ·
Del sil · Ctrl+= / Ctrl+- genişlet/daralt · F5 yeniden algıla · Ctrl+I görsel ekle ·
Ctrl+Shift+V video · Ctrl+0 sığdır · Ctrl+1 %100 · F1 yardım ·
(eklerken) sürükle konumlandır, Ctrl+tekerlek boyutlandır, Enter uygula, Esc iptal.
