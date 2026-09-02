"""Recommend matching earrings for a necklace.

    python recommend.py --necklace N01                 # necklace from the inventory
    python recommend.py --necklace N01 --top 5
    python recommend.py --image path/to/necklace.jpg    # any external necklace photo
    python recommend.py --all                           # contact sheet for all 5

Prints JSON and saves a contact-sheet PNG under outputs/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.index import load, recommend
from src.render import contact_sheet

ROOT = Path(__file__).resolve().parent


def _run(query_id, image, top, rebuild):
    items = load(rebuild=rebuild)
    res = recommend(query_id, items, top_k=top,
                    query_path=str(image) if image else None)
    name = (query_id or Path(image).stem).lower()
    out = contact_sheet(res, ROOT / "outputs" / f"reco_{name}.png")
    print(json.dumps(res, indent=2))
    print(f"\nContact sheet -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--necklace", help="necklace id from candidate_dataset.csv, e.g. N01")
    g.add_argument("--image", type=Path, help="path to an external necklace image")
    g.add_argument("--all", action="store_true", help="run every necklace in the inventory")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--rebuild", action="store_true", help="recompute the feature index")
    args = ap.parse_args()

    if args.all:
        items = load(rebuild=args.rebuild)
        neck = [k for k, v in items.items() if v.product_type.lower().startswith("neck")]
        for nid in neck:
            res = recommend(nid, items, top_k=args.top)
            out = contact_sheet(res, ROOT / "outputs" / f"reco_{nid.lower()}.png")
            top = ", ".join(f"{r['id']}({r['score']:.2f})" for r in res["results"])
            print(f"{nid}: {top}   -> {out.name}")
        return

    _run(args.necklace, args.image, args.top, args.rebuild)


if __name__ == "__main__":
    main()
