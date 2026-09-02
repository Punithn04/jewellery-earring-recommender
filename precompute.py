"""Build the feature index for every product in candidate_dataset.csv.

    python precompute.py

Writes outputs/index.pkl (CLIP embedding + colour palette + attribute profile per
item). Run once; recommend.py and app.py load the cache.
"""
from src.index import build

if __name__ == "__main__":
    build()
