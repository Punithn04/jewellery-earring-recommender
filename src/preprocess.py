"""Image preprocessing: isolate the jewellery object from its (varied) background.

Product photos in this set sit on pink card, beige cloth, navy velvet busts, wood +
leaves, grey fabric, etc. If we feed raw pixels to an embedding model or a colour
histogram, we mostly match on background and lighting. So step 1 is always:

    remove background -> crop to the object -> pad to square -> resize

We use `rembg` (U^2-Net) when available; otherwise fall back to a GrabCut-based
segmentation that assumes the object is roughly centred.
"""
from __future__ import annotations

import functools
import os
from pathlib import Path

import numpy as np
from PIL import Image

# rembg (U^2-Net) gives the cleanest cutout but its ONNX session costs ~600 MB RAM
# and ~10 s/image on CPU. Set DISABLE_REMBG=1 (done on the memory-limited Cloud Run
# deploy) to skip it and use the fast OpenCV GrabCut fallback instead.
if os.getenv("DISABLE_REMBG", "").lower() in ("1", "true", "yes"):
    _HAVE_REMBG = False
else:
    try:
        from rembg import remove as _rembg_remove, new_session as _rembg_session

        _SESSION = _rembg_session("u2net")
        _HAVE_REMBG = True
    except Exception:  # pragma: no cover - fallback path
        _HAVE_REMBG = False

import cv2

TARGET = 224
_PAD_RGB = (255, 255, 255)


def _grabcut_mask(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (int(w * 0.06), int(h * 0.06), int(w * 0.88), int(h * 0.88))
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(rgb, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    except Exception:
        m = np.zeros((h, w), np.uint8)
        m[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = 1
        return m
    return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == biggest).astype(np.uint8)


@functools.lru_cache(maxsize=256)
def load_object(path: str) -> tuple:
    """Return (rgb_on_white [H,W,3] uint8, mask [H,W] uint8 0/1), object cropped+squared."""
    img = Image.open(path).convert("RGB")
    rgb = np.array(img)

    if _HAVE_REMBG:
        cut = _rembg_remove(img, session=_SESSION).convert("RGBA")
        arr = np.array(cut)
        mask = (arr[:, :, 3] > 40).astype(np.uint8)
        if mask.sum() < 0.01 * mask.size:  # rembg wiped everything -> fall back
            mask = _grabcut_mask(rgb)
    else:
        mask = _grabcut_mask(rgb)

    mask = _largest_component(mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:  # degenerate: keep whole frame
        ys, xs = np.where(np.ones_like(mask))
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    pad = 8
    y0, x0 = max(0, y0 - pad), max(0, x0 - pad)
    y1, x1 = min(rgb.shape[0], y1 + pad), min(rgb.shape[1], x1 + pad)

    crop_rgb = rgb[y0:y1, x0:x1].copy()
    crop_mask = mask[y0:y1, x0:x1].copy()
    crop_rgb[crop_mask == 0] = _PAD_RGB  # paint background white

    # pad to square
    h, w = crop_rgb.shape[:2]
    side = max(h, w)
    sq_rgb = np.full((side, side, 3), _PAD_RGB, np.uint8)
    sq_mask = np.zeros((side, side), np.uint8)
    oy, ox = (side - h) // 2, (side - w) // 2
    sq_rgb[oy:oy + h, ox:ox + w] = crop_rgb
    sq_mask[oy:oy + h, ox:ox + w] = crop_mask

    sq_rgb = cv2.resize(sq_rgb, (TARGET, TARGET), interpolation=cv2.INTER_AREA)
    sq_mask = cv2.resize(sq_mask, (TARGET, TARGET), interpolation=cv2.INTER_NEAREST)
    return sq_rgb, sq_mask


def pil_on_white(path: str) -> Image.Image:
    rgb, _ = load_object(path)
    return Image.fromarray(rgb)


def object_pixels(path: str) -> np.ndarray:
    """[N,3] uint8 RGB pixels that belong to the object (for colour stats)."""
    rgb, mask = load_object(path)
    return rgb[mask > 0].reshape(-1, 3)
