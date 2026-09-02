"""Feature extraction + the three matching signals.

Signal 1 - CLIP embedding similarity (weight 0.55)
    Fashion-CLIP (patrickjohncyh/fashion-clip) image encoder. It is a CLIP ViT-B/32
    fine-tuned on ~700k fashion product image/description pairs, so its embedding
    space already separates jewellery by material, motif, silhouette and "vibe"
    far better than vanilla ImageNet features. Compared with cosine similarity.

Signal 2 - Colour-palette similarity (weight 0.30)
    Dominant colours of the *masked object* via k-means in CIELAB (perceptually
    uniform). Necklace vs earring palettes compared with a weighted nearest-colour
    distance (an approximate Earth Mover's Distance). This is what keeps an antique
    matte-gold temple necklace away from a bright rhodium AD earring even when the
    shapes rhyme.

Signal 3 - Attribute agreement (weight 0.15)
    Zero-shot CLIP text probes along 3 axes (metal tone / stone palette / style).
    Each axis becomes a probability distribution; agreement = Bhattacharyya
    coefficient. Cheap, and it gives us human-readable "why it matched" chips.

Final score = weighted sum of the three, each min-max normalised across the 15
candidate earrings for the query necklace (so every query uses the full 0..1 range).
"""
from __future__ import annotations

import functools

import numpy as np

# NOTE: torch / transformers / sklearn / cv2 are imported lazily inside the
# functions that need them. Ranking a *provided* necklace only touches the cached
# feature vectors + numpy, so a low-memory host never has to load the models.

_MODEL_NAME = "patrickjohncyh/fashion-clip"


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"

WEIGHTS = {"clip": 0.55, "colour": 0.30, "attr": 0.15}

ATTRIBUTE_AXES = {
    "metal tone": [
        "antique matte gold jewellery",
        "bright polished yellow gold jewellery",
        "rose gold jewellery",
        "silver white rhodium jewellery",
    ],
    "stone palette": [
        "jewellery with white diamond and cubic zirconia stones",
        "jewellery with green emerald stones",
        "jewellery with red ruby stones",
        "jewellery with uncut polki kundan stones",
        "plain gold jewellery with pearls and no coloured stones",
    ],
    "style": [
        "South Indian temple jewellery with goddess Lakshmi motifs",
        "Kundan polki bridal jewellery",
        "modern diamond cocktail jewellery",
        "minimalist delicate everyday jewellery",
    ],
}
_FLAT_PROMPTS = [(ax, i, p) for ax, ps in ATTRIBUTE_AXES.items() for i, p in enumerate(ps)]


def _vec(out):
    """transformers>=5 returns an object from get_*_features; <5 returns a tensor."""
    import torch

    if torch.is_tensor(out):
        return out
    return out.pooler_output


@functools.lru_cache(maxsize=1)
def _load():
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(_MODEL_NAME).to(_device()).eval()
    proc = CLIPProcessor.from_pretrained(_MODEL_NAME)
    return model, proc


@functools.lru_cache(maxsize=1)
def _text_bank() -> np.ndarray:
    import torch

    model, proc = _load()
    prompts = [p for _, _, p in _FLAT_PROMPTS]
    with torch.no_grad():
        t = proc(text=prompts, return_tensors="pt", padding=True).to(_device())
        emb = _vec(model.get_text_features(**t))
    emb = torch.nn.functional.normalize(emb, dim=-1)
    return emb.cpu().numpy()


# ---------------------------------------------------------------- CLIP embedding
def clip_embedding(path: str) -> np.ndarray:
    import torch

    from .preprocess import pil_on_white

    model, proc = _load()
    img = pil_on_white(path)
    with torch.no_grad():
        inp = proc(images=img, return_tensors="pt").to(_device())
        emb = _vec(model.get_image_features(**inp))
    emb = torch.nn.functional.normalize(emb, dim=-1)
    return emb.cpu().numpy()[0].astype(np.float32)


# ---------------------------------------------------------------- colour palette
def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    import cv2

    arr = rgb.reshape(-1, 1, 3).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)


def colour_palette(path: str, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Return (centres_lab [k,3], weights [k]) for the object's dominant colours."""
    from sklearn.cluster import KMeans

    from .preprocess import object_pixels

    px = object_pixels(path)
    if len(px) > 20000:
        px = px[np.random.RandomState(0).choice(len(px), 20000, replace=False)]
    lab = _rgb_to_lab(px)
    k = min(k, max(1, len(np.unique(lab, axis=0))))
    km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(lab)
    counts = np.bincount(km.labels_, minlength=k).astype(np.float32)
    return km.cluster_centers_.astype(np.float32), counts / counts.sum()


def palette_similarity(a: tuple, b: tuple) -> float:
    """Symmetric weighted nearest-colour distance in LAB -> similarity in [0,1]."""
    (ca, wa), (cb, wb) = a, b

    def directed(c1, w1, c2):
        d = np.linalg.norm(c1[:, None, :] - c2[None, :, :], axis=2)  # deltaE-ish
        return float((w1 * d.min(axis=1)).sum())

    dist = 0.5 * (directed(ca, wa, cb) + directed(cb, wb, ca))
    return float(np.exp(-dist / 22.0))  # ~22 LAB units -> 0.37


# ---------------------------------------------------------------- attributes
def attribute_profile(path: str) -> dict:
    """{axis: probability vector over that axis's prompts}."""
    import torch

    from .preprocess import pil_on_white

    model, proc = _load()
    img = pil_on_white(path)
    with torch.no_grad():
        inp = proc(images=img, return_tensors="pt").to(_device())
        emb = torch.nn.functional.normalize(_vec(model.get_image_features(**inp)), dim=-1)
    sims = emb.cpu().numpy()[0] @ _text_bank().T  # cosine to every prompt

    out: dict[str, np.ndarray] = {}
    for ax in ATTRIBUTE_AXES:
        idx = [j for j, (a, _, _) in enumerate(_FLAT_PROMPTS) if a == ax]
        logits = sims[idx] / 0.07
        p = np.exp(logits - logits.max())
        out[ax] = p / p.sum()
    return out


def attribute_similarity(pa: dict, pb: dict) -> float:
    vals = [float(np.sqrt(pa[ax] * pb[ax]).sum()) for ax in pa]  # Bhattacharyya
    return float(np.mean(vals))


def describe(profile: dict) -> list[str]:
    chips = []
    for ax, prompts in ATTRIBUTE_AXES.items():
        i = int(np.argmax(profile[ax]))
        label = prompts[i].replace(" jewellery", "").replace("jewellery ", "")
        chips.append(f"{ax}: {label}")
    return chips
