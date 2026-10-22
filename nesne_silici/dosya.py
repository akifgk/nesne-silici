"""Görüntü dosyası okuma/yazma: PNG, JPG, BMP, WEBP, TIFF, GIF, ICO, HEIC/HEIF ...

Tek sorumluluk: dosya <-> numpy dizisi dönüşümü. Görüntü *işleme* burada yapılmaz;
kodlama/çözme (codec) işini Pillow yapar, çünkü onlarca biçimin ikili düzenini elle
yazmak amaç dışıdır.

* ``ac``     : her biçimi ``(rgb uint8, alfa uint8 | None)`` olarak okur. EXIF dönme
  bilgisi uygulanır (telefon fotoğrafları yan görünmez), 16 bit/gri/paletli/CMYK
  görüntüler RGB'ye çevrilir. Animasyonlu dosyalarda ilk kare alınır.
* ``kaydet`` : uzantıya göre biçim seçer. Alfa desteklemeyen biçimlerde (JPG, BMP...)
  saydam alan beyaza yedirilir; sessizce siyah kalmaz.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

try:                                           # iPhone fotoğrafları (HEIC) için isteğe bağlı
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_VAR = True
except Exception:                              # pragma: no cover - paket yoksa sorun değil
    HEIC_VAR = False

Image.MAX_IMAGE_PIXELS = 400_000_000           # çok büyük panoramalara izin ver


@dataclass(frozen=True)
class Bicim:
    ad: str            # arayüzde görünen ad
    uzantilar: tuple[str, ...]
    pillow_adi: str    # Pillow'un biçim adı
    alfa: bool         # saydamlık destekler mi
    kayipli: bool      # kalite ayarı anlamlı mı


KAYIT_BICIMLERI: tuple[Bicim, ...] = (
    Bicim("PNG", (".png",), "PNG", True, False),
    Bicim("JPEG", (".jpg", ".jpeg", ".jpe"), "JPEG", False, True),
    Bicim("WEBP", (".webp",), "WEBP", True, True),
    Bicim("BMP", (".bmp",), "BMP", False, False),
    Bicim("TIFF", (".tif", ".tiff"), "TIFF", True, False),
    Bicim("GIF", (".gif",), "GIF", False, False),
    Bicim("ICO", (".ico",), "ICO", True, False),
    Bicim("PPM", (".ppm",), "PPM", False, False),
) + ((Bicim("HEIC", (".heic", ".heif"), "HEIF", True, True),) if HEIC_VAR else ())

# Okunabilen uzantılar: Pillow'un kayıtlı tüm biçimleri (sistemdeki kuruluma göre değişir).
def okunabilir_uzantilar() -> list[str]:
    Image.init()
    return sorted(Image.registered_extensions())


def acma_filtresi() -> str:
    """QFileDialog süzgeci: 'Görüntüler (*.png *.jpg ...);;Tüm dosyalar (*)'."""
    onemli = [u for u in okunabilir_uzantilar()
              if u not in {".pdf", ".eps", ".ps", ".msp", ".mpo", ".xbm", ".xpm"}]
    return f"Görüntüler ({' '.join('*' + u for u in onemli)});;Tüm dosyalar (*)"


def kaydetme_filtreleri() -> str:
    return ";;".join(f"{b.ad} ({' '.join('*' + u for u in b.uzantilar)})" for b in KAYIT_BICIMLERI)


def bicim_bul(yol: str | Path) -> Bicim | None:
    uzanti = Path(yol).suffix.lower()
    return next((b for b in KAYIT_BICIMLERI if uzanti in b.uzantilar), None)


# --------------------------------------------------------------------------- okuma
def ac(yol: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Görüntüyü okur. Döner: (rgb uint8 (h,w,3), alfa uint8 (h,w) ya da None)."""
    with Image.open(yol) as im:
        im.load()
        im = ImageOps.exif_transpose(im)         # telefon fotoğrafı yönü
        if im.mode in ("I;16", "I;16L", "I;16B", "I", "F"):
            im = _yuksek_bit_derinligi_cevir(im)
        saydam_var = im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info
        im = im.convert("RGBA" if saydam_var else "RGB")   # P, L, CMYK, 1... hepsi buradan geçer
        dizi = np.array(im)                      # kopya: np.asarray Pillow'un salt okunur belleğini verirdi
    if dizi.shape[2] == 4:
        alfa = np.ascontiguousarray(dizi[..., 3])
        return np.ascontiguousarray(dizi[..., :3]), (alfa if alfa.min() < 255 else None)
    return np.ascontiguousarray(dizi), None


def _yuksek_bit_derinligi_cevir(im: Image.Image) -> Image.Image:
    """16 bit / float görüntüyü 8 bit griye ölçekler (en parlak değer 255 olur)."""
    a = np.asarray(im).astype(np.float64)
    ust = a.max() if a.max() > 0 else 1.0
    return Image.fromarray(np.clip(a / ust * 255.0, 0, 255).astype(np.uint8), "L")


# --------------------------------------------------------------------------- yazma
def kaydet(yol: str | Path, rgb: np.ndarray, alfa: np.ndarray | None = None,
           kalite: int = 95) -> None:
    """Görüntüyü uzantısına uygun biçimde yazar. Uzantı tanınmazsa ValueError fırlatır."""
    bicim = bicim_bul(yol)
    if bicim is None:
        raise ValueError(f"Desteklenmeyen kayıt biçimi: {Path(yol).suffix or '(uzantı yok)'}")
    if alfa is not None and bicim.alfa:
        im = Image.fromarray(np.dstack([rgb, alfa]), "RGBA")
    elif alfa is not None:                       # alfa yok sayılamaz: beyaza yedir
        a = (alfa.astype(np.float32) / 255.0)[..., None]
        im = Image.fromarray((rgb * a + 255.0 * (1 - a) + 0.5).astype(np.uint8), "RGB")
    else:
        im = Image.fromarray(rgb, "RGB")

    secenek: dict = {}
    if bicim.pillow_adi == "JPEG":
        secenek = {"quality": int(kalite), "optimize": True, "subsampling": 0 if kalite >= 90 else 2}
    elif bicim.pillow_adi in ("WEBP", "HEIF"):
        secenek = {"quality": int(kalite)}
    elif bicim.pillow_adi == "GIF":
        im = im.convert("P", palette=Image.ADAPTIVE)
    elif bicim.pillow_adi == "ICO":
        if max(im.size) > 256:                   # ICO en çok 256x256 olabilir
            im.thumbnail((256, 256))
    elif bicim.pillow_adi == "TIFF":
        secenek = {"compression": "tiff_lzw"}
    im.save(yol, format=bicim.pillow_adi, **secenek)
