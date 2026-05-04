import asyncio
import logging
import os
from pathlib import Path

from pydantic import ValidationError

import ai
from category_matcher import CategoryMatcher
from models import Product
from preprocessor import preprocess

logger = logging.getLogger(__name__)

# Default model per provider
OLLAMA_MODEL = "llama3.2:latest"
OPENROUTER_MODEL = "google/gemini-2.5-flash-lite"

_category_matcher = CategoryMatcher()

SYSTEM_PROMPT = """\
You are an expert product data extractor. Given a preprocessed product page, \
you MUST return a single structured Product object.

Rules:
- You MUST extract only the primary product on the page. You MUST NOT include \
data from related, recommended, or upsell products.
- You MUST prefer [STRUCTURED DATA] over [CONTENT] when values conflict. \
JSON-LD structured data is authoritative.
- You MUST source image_urls from [STRUCTURED DATA] first — these are \
intentionally set by the site and are always product images. You MUST only \
fall back to [IMAGES] if [STRUCTURED DATA] contains no image URLs. \
You MUST NOT include UI chrome, logos, navigation icons, or thumbnails.
- Each variant MUST represent a single purchasable combination of options \
(e.g. Size=10, Color=Black). You MUST NOT list option dimensions as variants.
- The price field MUST reflect the current purchase price. You MUST NOT use \
compare_at_price as the main price.
- Colors MUST only reflect what is explicitly visible on the page. \
You MUST NOT infer or hallucinate colors.
- For any field where data is not available, you MUST use null rather than \
guessing or fabricating a value.
"""

USER_PROMPT = """\
IMPORTANT — CATEGORY SELECTION:
You MUST set category.name to EXACTLY one of the strings listed below, \
character for character. Do NOT paraphrase, shorten, or invent a category. \
Copy it exactly as written.

{candidate_categories}

Now extract the product from the following page:

{payload}\
"""

RETRY_SUFFIX = """

Your previous response failed validation with the following error:
{error}

Please fix the issue and try again.
"""


def _get_model() -> str:
    provider = os.environ.get("LLM_PROVIDER", "openrouter").lower()
    return OLLAMA_MODEL if provider == "ollama" else OPENROUTER_MODEL


async def extract(html: str, source: str) -> Product | None:
    """
    Extract a Product from raw HTML.

    Args:
        html: raw HTML string of a product detail page
        source: identifier for logging (e.g. filename stem)

    Returns:
        Product instance on success, None if extraction fails after retry
    """
    payload = preprocess(html)
    candidates = _category_matcher.match(payload)
    numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(candidates))
    user_message = USER_PROMPT.format(candidate_categories=numbered, payload=payload)
    model = _get_model()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    # First attempt
    first_error = None
    try:
        return await ai.responses(model=model, input=messages, text_format=Product)
    except (ValidationError, Exception) as e:
        first_error = e
        logger.warning(f"[{source}] First attempt failed: {e}")

    # Retry with validation error hint
    messages.append({"role": "user", "content": RETRY_SUFFIX.format(error=str(first_error))})
    try:
        return await ai.responses(model=model, input=messages, text_format=Product)
    except (ValidationError, Exception) as e:
        logger.error(f"[{source}] Second attempt failed, skipping: {e}")
        return None


async def run(data_dir: Path) -> None:
    """
    Extract Product data from all HTML files in data_dir and write JSON output.

    Args:
        data_dir: directory containing *.html product pages
    """
    output_dir = data_dir / "output"
    output_dir.mkdir(exist_ok=True)

    html_files = list(data_dir.glob("*.html"))
    if not html_files:
        logger.warning(f"No HTML files found in {data_dir}")
        return

    logger.info(f"Processing {len(html_files)} files...")

    async def _extract_and_save(html_file: Path) -> bool:
        source = html_file.stem
        html = html_file.read_text(encoding="utf-8")
        product = await extract(html, source)
        if product is None:
            return False
        out_path = output_dir / f"{source}.json"
        out_path.write_text(product.model_dump_json(indent=2), encoding="utf-8")
        logger.info(f"[{source}] Written to {out_path}")
        return True

    results = await asyncio.gather(*[_extract_and_save(f) for f in html_files])

    succeeded = sum(results)
    failed = len(results) - succeeded
    logger.info(f"Done: {succeeded} succeeded, {failed} failed")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "data"
    asyncio.run(run(data_dir))
