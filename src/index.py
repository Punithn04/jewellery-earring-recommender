"""Build / load the feature index and rank earrings for a query necklace."""
from __future__ import annotations

import csv
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import features as F

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
IMAGES = DATA / "images"
CACHE = ROOT / "outputs" / "index.pkl"


@dataclass
class Item:
    id: str
    product_type: str
    image_file: str
    clip: np.ndarray
    palette: tuple
    attrs: dict

    @property
    def path(self) -> str:
        return str(IMAGES / self.image_file)


def _read_csv() -> list[dict]:
    with open(DATA / "candidate_dataset.csv", newline="") as fh:
        return list(csv.DictReader(fh))


def build(verbose: bool = True) -> dict[str, Item]:
    from .preprocess import load_object

    items: dict[str, Item] = {}
    for row in _read_csv():
        path = str(IMAGES / row["image_file"])
        if verbose:
            print(f"  features: {row['id']:<5} {row['image_file']}")
        load_object(path)  # warm cache / segmentation
        items[row["id"]] = Item(
            id=row["id"],
            product_type=row["product_type"],
            image_file=row["image_file"],
            clip=F.clip_embedding(path),
            palette=F.colour_palette(path),
            attrs=F.attribute_profile(path),
        )
    CACHE.parent.mkdir(exist_ok=True)
    with open(CACHE, "wb") as fh:
        pickle.dump(items, fh)
    if verbose:
        print(f"saved {CACHE}")
    return items


def load(rebuild: bool = False) -> dict[str, Item]:
    if rebuild or not CACHE.exists():
        return build()
    with open(CACHE, "rb") as fh:
        return pickle.load(fh)


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x) if hi - lo < 1e-9 else (x - lo) / (hi - lo)


# Below this raw spread a signal carries little real information for the query and
# its min-max-normalised version is damped so noise is not amplified to full weight.
_INFORMATIVE_RANGE = {"clip": 0.04, "colour": 0.08, "attr": 0.03}


def _damped(raw: np.ndarray, key: str) -> np.ndarray:
    rng = float(raw.max() - raw.min())
    return _minmax(raw) * min(1.0, rng / _INFORMATIVE_RANGE[key])


def recommend(query_id: str | None, items: dict[str, Item], top_k: int = 5,
              query_path: str | None = None) -> dict:
    """Rank earrings for a necklace given by id (in inventory) or by image path."""
    if query_path:
        q_clip = F.clip_embedding(query_path)
        q_pal = F.colour_palette(query_path)
        q_attr = F.attribute_profile(query_path)
        q_meta = {"id": "UPLOAD", "image_file": Path(query_path).name}
    else:
        q = items[query_id]
        q_clip, q_pal, q_attr = q.clip, q.palette, q.attrs
        q_meta = {"id": q.id, "image_file": q.image_file}

    ear = [it for it in items.values() if it.product_type.lower().startswith("earr")]
    clip_s = np.array([float(q_clip @ it.clip) for it in ear])
    col_s = np.array([F.palette_similarity(q_pal, it.palette) for it in ear])
    att_s = np.array([F.attribute_similarity(q_attr, it.attrs) for it in ear])

    final = (F.WEIGHTS["clip"] * _damped(clip_s, "clip")
             + F.WEIGHTS["colour"] * _damped(col_s, "colour")
             + F.WEIGHTS["attr"] * _damped(att_s, "attr"))

    order = np.argsort(-final)
    results = []
    for rank, i in enumerate(order[:top_k], 1):
        it = ear[i]
        results.append({
            "rank": rank,
            "id": it.id,
            "image_file": it.image_file,
            "score": round(float(final[i]), 4),
            "breakdown": {
                "clip": round(float(clip_s[i]), 4),
                "colour": round(float(col_s[i]), 4),
                "attr": round(float(att_s[i]), 4),
            },
            "why": F.describe(it.attrs),
        })
    return {
        "query": {**q_meta, "why": F.describe(q_attr)},
        "weights": F.WEIGHTS,
        "results": results,
    }
