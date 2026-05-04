"""
One-time script: encode all Google Product Taxonomy categories into a
pre-computed embedding index used by CategoryMatcher at runtime.

Run once after cloning:
    uv run python generate_embeddings.py

Output: category_embeddings.npz (~13 MB, gitignored — rebuild after cloning)
"""

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

TAXONOMY = Path("categories.txt")
OUT = Path("category_embeddings.npz")
MODEL = "all-MiniLM-L6-v2"  # 80 MB download, 384-dim, fast on CPU

categories = [
    line.strip()
    for line in TAXONOMY.read_text().splitlines()
    if line.strip() and not line.startswith("#")
]

print(f"Encoding {len(categories)} taxonomy categories with {MODEL}...")
model = SentenceTransformer(MODEL)
embeddings = model.encode(categories, show_progress_bar=True, convert_to_numpy=True)

np.savez(OUT, embeddings=embeddings, categories=np.array(categories))
print(f"Saved → {OUT}  ({OUT.stat().st_size / 1024 / 1024:.1f} MB)")
