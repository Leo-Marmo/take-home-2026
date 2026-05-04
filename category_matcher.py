import json
import re
from models import VALID_CATEGORIES

# Common words that add no signal for category matching
_STOP_WORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for",
    "with", "by", "from", "is", "it", "its", "this", "that", "are", "was",
    "be", "as", "up", "out", "but", "not", "can", "has", "have", "been",
}

MAX_RESULTS = 20


def _extract_keywords(text: str) -> set[str]:
    """Tokenize text into lowercase keywords, stripping stop words."""
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    return {t for t in tokens if t not in _STOP_WORDS and len(t) > 2}


def _extract_product_keywords(payload: str) -> set[str]:
    """
    Extract keywords from product name and description only.

    Prefers JSON-LD name/description fields over raw page text to avoid
    pollution from brand names, product codes, and navigation content.
    Falls back to first 500 chars of payload if no JSON-LD is found.
    """
    # Try to pull name + description from JSON-LD in [STRUCTURED DATA] section
    structured_match = re.search(
        r"\[STRUCTURED DATA\](.*?)(?:\[IMAGES\]|\[CONTENT\]|$)",
        payload,
        re.DOTALL,
    )
    if structured_match:
        block = structured_match.group(1)
        # Try each JSON block (separated by ---)
        for chunk in block.split("---"):
            try:
                data = json.loads(chunk.strip())
                name = data.get("name", "")
                description = data.get("description", "")
                combined = f"{name} {description}"
                if combined.strip():
                    return _extract_keywords(combined)
            except (json.JSONDecodeError, AttributeError):
                continue

    # Fallback to raw payload prefix
    return _extract_keywords(payload[:500])


def _score(keywords: set[str], category: str) -> float:
    """
    Score a category by normalized word overlap with the keyword set.

    Normalizes by category length so short, specific categories
    (e.g. 'Apparel & Accessories > Shoes') aren't penalized against
    longer strings that match incidentally.
    """
    category_words = set(re.findall(r"[a-zA-Z]+", category.lower()))
    if not category_words:
        return 0.0
    overlap = len(keywords & category_words)
    return overlap / len(category_words)


class CategoryMatcher:
    """
    Matches a text payload to the closest Google Product Taxonomy categories.

    Currently implemented as normalized word-overlap scoring against product
    name and description. Can be swapped for BM25 or embedding-based search
    without changing the interface.

    TODO: Improve matching algorithm. Current word-overlap approach has known
    limitations with brand names and ambiguous terms. Suggested progression:
      1. BM25 (rank_bm25) — better term weighting, drop-in replacement
      2. Embeddings (sentence-transformers) — semantic matching, handles
         synonyms (e.g. "sneakers" → shoes) without exact word overlap
    """

    def __init__(self, categories: set[str] = VALID_CATEGORIES):
        self._categories = categories

    def match(self, payload: str, n: int = MAX_RESULTS) -> list[str]:
        """
        Return the top n categories most relevant to the payload.

        Args:
            payload: preprocessed product page text
            n: maximum number of categories to return

        Returns:
            list of category strings from Google Product Taxonomy, ranked by relevance
        """
        keywords = _extract_product_keywords(payload)
        if not keywords:
            return []

        scored = (
            (self._score(keywords, category), category)
            for category in self._categories
        )
        top = sorted(scored, key=lambda x: x[0], reverse=True)[:n]
        return [category for score, category in top if score > 0.0]

    def _score(self, keywords: set[str], category: str) -> float:
        return _score(keywords, category)
