from __future__ import annotations

from typing import Iterable

from PIL import Image

Image.MAX_IMAGE_PIXELS = 16777216

MAX_FILE_SIZE = 5 * 1024 * 1024
MAX_DIMENSION = 4096
ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
}
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "TIFF"}


def check_file_size(size: int) -> bool:
    return size < MAX_FILE_SIZE


def check_dimensions(width: int, height: int) -> bool:
    return width < MAX_DIMENSION and height < MAX_DIMENSION


def check_content_type(content_type: str) -> bool:
    return content_type in ALLOWED_MIME_TYPES


def check_magic_bytes(data: bytes, content_type: str | None = None) -> bool:
    if len(data) < 12:
        return False

    is_jpeg = data[:3] == b"\xff\xd8\xff"
    is_png = data[:8] == b"\x89PNG\r\n\x1a\n"
    is_webp = data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    is_bmp = data[:2] == b"BM"
    is_tiff = data[:4] in (b"\x49\x49\x2A\x00", b"\x4D\x4D\x00\x2A")

    detected = {
        "image/jpeg": is_jpeg,
        "image/png": is_png,
        "image/webp": is_webp,
        "image/bmp": is_bmp,
        "image/tiff": is_tiff,
    }

    if content_type is not None:
        return detected.get(content_type, False)

    return any(detected.values())
