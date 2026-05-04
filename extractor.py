import asyncio
import logging
from pathlib import Path

from pydantic import ValidationError

import ai
from category_matcher import CategoryMatcher
from models import Product
from preprocessor import preprocess

logger = logging.getLogger(__name__)

MODEL = "google/gemini-2.5-flash-lite"
MAX_RETRIES = 2

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


async def extract(html: str, source: str) -> Product | None:
    """
    Extract a Product from raw HTML.

    Args:
        html: raw HTML string of a product detail page
        source: identifier for logging (e.g. filename stem)

    Returns:
        Product instance on success, None if all attempts fail
    """
    payload = preprocess(html)
    candidates = _category_matcher.match(payload)
    numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(candidates))
    user_message = USER_PROMPT.format(candidate_categories=numbered, payload=payload)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            return await ai.responses(model=MODEL, messages=messages, text_format=Product)
        except Exception as e:
            last_error = e
            if attempt <= MAX_RETRIES:
                logger.warning(f"[{source}] Attempt {attempt} failed: {e}")
                messages.append({"role": "user", "content": RETRY_SUFFIX.format(error=str(e))})
            else:
                logger.error(f"[{source}] Failed after {MAX_RETRIES} retries, skipping: {e}")

    return None


async def run(data_dir: Path, only: set[str] | None = None) -> None:
    """
    Extract Product data from all HTML files in data_dir and write JSON output.

    Args:
        data_dir: directory containing *.html product pages
        only: if provided, restrict extraction to files whose stem is in this set
    """
    output_dir = data_dir / "output"
    output_dir.mkdir(exist_ok=True)

    html_files = list(data_dir.glob("*.html"))
    if only:
        html_files = [f for f in html_files if f.stem in only]
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

    args = sys.argv[1:]
    only: set[str] = set()
    remaining: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == "--only" and i + 1 < len(args):
            only.add(Path(args[i + 1]).stem)
            i += 2
        else:
            remaining.append(args[i])
            i += 1

    data_dir = Path(remaining[0]) if remaining else Path(__file__).parent / "data"
    asyncio.run(run(data_dir, only=only or None))
