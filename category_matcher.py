"""
Matches a product payload to the closest Google Product Taxonomy categories
using a hybrid of word-overlap scoring and semantic embeddings.

Word-overlap runs first using a synonym table that maps product-world terms
(tee, trousers, runner) to taxonomy-world terms (shirt, pants, shoes). Its
results are merged with the top-N categories from a pre-computed embedding
index (category_embeddings.npz), giving the LLM candidates from both signals.

The embedding index is built once by scripts/generate_embeddings.py. When the
index is not present, the matcher falls back to word-overlap only.
"""

import json
import logging
import re
import warnings
from pathlib import Path

import numpy as np

from models import VALID_CATEGORIES

logger = logging.getLogger(__name__)

MAX_RESULTS = 20
_MODEL_NAME = "all-MiniLM-L6-v2"
_INDEX_PATH = Path(__file__).parent / "category_embeddings.npz"

# ---------------------------------------------------------------------------
# Payload text extraction
# ---------------------------------------------------------------------------

def _extract_product_text(payload: str) -> str:
    """
    Return the product name string from the preprocessed payload.

    Prefers the JSON-LD `name` field from [STRUCTURED DATA] — it's concise,
    unambiguous, and carries the most category signal. Falls back to the
    first 200 characters of the payload if no JSON-LD name is found.
    """
    structured_match = re.search(
        r"\[STRUCTURED DATA\](.*?)(?:\[IMAGES\]|\[CONTENT\]|$)",
        payload,
        re.DOTALL,
    )
    if structured_match:
        for chunk in structured_match.group(1).split("---"):
            try:
                data = json.loads(chunk.strip())
                name = data.get("name", "").strip()
                if name:
                    return name
            except (json.JSONDecodeError, AttributeError):
                continue

    return payload[:200].strip()


# ---------------------------------------------------------------------------
# Word-overlap fallback (used when .npz index is not present)
# ---------------------------------------------------------------------------

_STOP_WORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for",
    "with", "by", "from", "is", "it", "its", "this", "that", "are", "was",
    "be", "as", "up", "out", "but", "not", "can", "has", "have", "been",
}

_SYNONYMS: dict[str, str] = {
    "trousers": "pants", "slacks": "pants", "chinos": "pants", "leggings": "pants",
    "tee": "shirt", "tshirt": "shirt", "henley": "shirt", "polo": "shirt", "blouse": "shirt",
    "sneakers": "shoes", "trainers": "shoes", "loafers": "shoes", "heels": "shoes",
    "runner": "shoes", "runners": "shoes",
    "hoodie": "sweatshirt", "parka": "coat", "anorak": "jacket",
    "shelf": "shelving", "shelves": "shelving", 
    "toolkit": "tools", "sofa": "sofa", "couch": "sofa", "loveseat": "sofa", "armchair": "chair",
}

_FIT_PATTERN = re.compile(
    r"\b(traditional|slim|classic|regular|relaxed|loose|fitted|straight|athletic|modern|original)\s+fit\b",
    re.IGNORECASE,
)


def _word_overlap_score(keywords: set[str], category: str) -> float:
    category_words = set(re.findall(r"[a-zA-Z]+", category.lower()))
    if not category_words:
        return 0.0
    return len(keywords & category_words) / len(category_words)


def _extract_keywords(text: str) -> set[str]:
    text = _FIT_PATTERN.sub("", text)
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    # Length filter is applied to the SYNONYM RESULT, not the original token.
    # This lets short tokens that map to meaningful synonyms survive
    # (e.g. "tee" → "shirt", len 5, kept) while filtering out short ambiguous
    # words whose synonyms are also short (e.g. "air" → "air", len 3, dropped).
    # Without this, "Air Max 90" spuriously matches HVAC categories via "air".
    keywords = {
        syn
        for t in tokens
        if t not in _STOP_WORDS
        for syn in (_SYNONYMS.get(t, t),)
        if len(syn) > 3
    }
    # Add plural/singular counterparts so "shirt" matches "Shirts & Tops"
    # and "drill" matches "Handheld Power Drills"
    expanded: set[str] = set()
    for kw in keywords:
        expanded.add(kw)
        expanded.add(kw[:-1] if kw.endswith("s") and len(kw) > 3 else kw + "s")
    return expanded


# ---------------------------------------------------------------------------
# CategoryMatcher
# ---------------------------------------------------------------------------

class CategoryMatcher:
    """
    Matches a preprocessed product payload to Google Product Taxonomy categories.

    Uses cosine similarity against a pre-computed embedding index when
    category_embeddings.npz is present; degrades to word-overlap scoring
    otherwise.
    """

    def __init__(self, categories: set[str] = VALID_CATEGORIES) -> None:
        self._valid_categories = categories
        self._model = None
        self._embeddings: np.ndarray | None = None
        self._index_categories: list[str] = []
        self._norms: np.ndarray | None = None
        self._use_embeddings = False

        if _INDEX_PATH.exists():
            try:
                self._load_index()
                self._use_embeddings = True
            except Exception as exc:
                warnings.warn(
                    f"Failed to load embedding index ({exc}); falling back to word-overlap.",
                    stacklevel=2,
                )
        else:
            warnings.warn(
                f"Embedding index not found at {_INDEX_PATH}. "
                "Run scripts/generate_embeddings.py to build it. Falling back to word-overlap matching.",
                stacklevel=2,
            )

    def _load_index(self) -> None:
        from sentence_transformers import SentenceTransformer

        npz = np.load(_INDEX_PATH, allow_pickle=True)
        self._embeddings = npz["embeddings"].astype(np.float32)
        self._index_categories = npz["categories"].tolist()
        self._norms = np.linalg.norm(self._embeddings, axis=1)
        self._model = SentenceTransformer(_MODEL_NAME)
        logger.debug(
            "Loaded embedding index: %d categories, dim=%d",
            len(self._index_categories),
            self._embeddings.shape[1],
        )

    def match(self, payload: str, n: int = MAX_RESULTS) -> list[str]:
        """
        Return the top n taxonomy categories most relevant to the payload.

        Strategy:
        - Both word-overlap and embeddings run independently and their results
          are merged. Word-overlap candidates appear first (they're more precise
          for known synonym patterns), followed by any additional categories
          surfaced by embeddings that weren't already in the overlap set.
        - Falls back to word-overlap-only when the embedding index is not present.

        Args:
            payload: preprocessed product page text (output of preprocessor.preprocess)
            n: maximum number of categories to return

        Returns:
            list of category strings from Google Product Taxonomy, ranked by relevance
        """
        if not payload.strip():
            return []

        overlap_results = self._match_word_overlap(payload, n)

        if self._use_embeddings:
            embedding_results = self._match_embeddings(payload, n)
            # Merge: word-overlap first, then any embedding results not already present
            seen = set(overlap_results)
            combined = overlap_results + [c for c in embedding_results if c not in seen]
            candidates = combined[:n]
        else:
            candidates = overlap_results

        if not candidates:
            return []

        return self._drop_superseded_parents(candidates)

    @staticmethod
    def _drop_superseded_parents(candidates: list[str]) -> list[str]:
        """
        Remove any candidate that is a strict prefix of another candidate in
        the same list. Prevents the LLM from picking a vague parent category
        (e.g. "Hardware > Tools > Drills") when a more specific child
        (e.g. "Hardware > Tools > Drills > Handheld Power Drills") is available.
        """
        candidate_set = set(candidates)
        return [
            c for c in candidates
            if not any(
                other != c and other.startswith(c + " > ")
                for other in candidate_set
            )
        ]

    def _match_embeddings(self, payload: str, n: int) -> list[str]:
        text = _extract_product_text(payload)
        query = self._model.encode([text], convert_to_numpy=True)[0].astype(np.float32)
        q_norm = float(np.linalg.norm(query))
        if q_norm < 1e-9:
            return []

        scores = (self._embeddings @ query) / (self._norms * q_norm + 1e-9)
        top_idx = np.argsort(scores)[::-1][:n]
        # Keep only categories that are still in VALID_CATEGORIES
        return [
            self._index_categories[i]
            for i in top_idx
            if self._index_categories[i] in self._valid_categories
        ]

    def _match_word_overlap(self, payload: str, n: int) -> list[str]:
        text = _extract_product_text(payload)
        keywords = _extract_keywords(text)
        if not keywords:
            return []
        scored = (
            (_word_overlap_score(keywords, cat), cat)
            for cat in self._valid_categories
        )
        top = sorted(scored, key=lambda x: x[0], reverse=True)[:n]
        return [cat for score, cat in top if score > 0.0]
