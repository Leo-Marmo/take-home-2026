import json
import re
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import extruct
from bs4 import BeautifulSoup

# Image extensions we consider real product images (for <img> tag filtering only)
_IMAGE_EXTENSIONS = re.compile(r'\.(jpg|jpeg|png|webp|avif)(\?|$)', re.IGNORECASE)

# URL fragments that indicate UI/chrome images — applied to all sources
# _NNNx. is the Shopify embedded-size thumbnail suffix (e.g. _271x.png, _600x.jpg);
# these appear in JSON-LD recommendation lists and <img> browse carousels, never in
# the primary product JSON-LD image field which always points to the full-size asset.
_NOISE_PATTERNS = re.compile(
    r'(\.svg|/resources/|/icons?/|logo|favicon|spinner|loading|blank\.gif|no-image|-carousel-|_\d+x\.|analytics)',
    re.IGNORECASE,
)


def _is_product_image(url: str) -> bool:
    """Full check for <img> tag URLs: requires a known image extension and no noise patterns."""
    return bool(_IMAGE_EXTENSIONS.search(url)) and not _NOISE_PATTERNS.search(url)


def _is_structured_image(url: str) -> bool:
    """Relaxed check for structured data images (JSON-LD, microdata, og:image).

    Structured data images are always intentionally set by the site, so we
    skip the extension requirement and only filter obvious noise.
    """
    return bool(url) and not _NOISE_PATTERNS.search(url)


def _normalize_url(url: str) -> str:
    """Normalize a URL for deduplication and display.

    - Promotes protocol-relative URLs (//host/path) to https://
    - Strips query params so CDN variants of the same asset collapse to one key
    """
    if url.startswith("//"):
        url = "https:" + url
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))


def _walk_for_images(data: object, seen: set[str], out: list[str], _in_image_key: bool = False) -> None:
    """Recursively walk any JSON-like structure and collect structured data image URLs.

    Strings are only collected when nested inside an 'image'/'images' key so that
    non-URL string values (product names, types, etc.) are not mistakenly included.
    """
    if isinstance(data, str):
        if _in_image_key:
            normalized = _normalize_url(data)
            if _is_structured_image(normalized) and normalized not in seen:
                seen.add(normalized)
                out.append(normalized)
    elif isinstance(data, list):
        for item in data:
            _walk_for_images(item, seen, out, _in_image_key)
    elif isinstance(data, dict):
        for key, value in data.items():
            is_image_key = key.lower() in ("image", "images", "image_url", "imageurl")
            _walk_for_images(value, seen, out, is_image_key)


def preprocess(html: str) -> str:
    """
    Strip a raw HTML page down to a plain text payload for LLM extraction.

    Output sections:
      [STRUCTURED DATA]  — JSON-LD and microdata blocks found in the page
      [IMAGES]           — deduplicated product image URLs
      [VIDEO]            — video URL if found in OpenGraph
      [CONTENT]          — visible text after stripping noise tags
    """
    soup = BeautifulSoup(html, "lxml")
    seen: set[str] = set()

    # --- 1. Extract all structured data via extruct ---
    # Handles JSON-LD, Schema.org microdata, and OpenGraph in one pass
    extracted = extruct.extract(
        html,
        syntaxes=["json-ld", "microdata", "opengraph"],
        uniform=True,
        errors="ignore",
    )

    # --- 2. Build [STRUCTURED DATA] from JSON-LD and microdata ---
    # Only keep Product/ProductGroup blocks — BreadcrumbList, 3DModel, WebSite,
    # FAQPage, etc. add no extraction signal and waste tokens.
    # Keep only the FIRST qualifying block: the primary product is always listed
    # first; subsequent blocks are related/upsell products injected by the page.
    _PRODUCT_TYPES = {"Product", "ProductGroup"}
    # Keys that add no extraction signal and can be large (reviews, ratings).
    _STRIP_KEYS = {"review", "aggregateRating"}

    def _clean_product_block(item: dict) -> dict:
        cleaned = {k: v for k, v in item.items() if k not in _STRIP_KEYS}
        # Remove noise placeholder image URLs from the structured data dump so
        # the LLM doesn't copy them into image_urls.
        for img_key in ("image", "images"):
            if img_key not in cleaned:
                continue
            val = cleaned[img_key]
            if isinstance(val, str) and _NOISE_PATTERNS.search(val):
                del cleaned[img_key]
            elif isinstance(val, list):
                filtered = [v for v in val if not (isinstance(v, str) and _NOISE_PATTERNS.search(v))]
                if filtered:
                    cleaned[img_key] = filtered
                else:
                    del cleaned[img_key]
        # Trim hasVariant to entries with actual data — URL-only stubs
        # ({"@type": "Product", "url": "..."}) carry no useful information.
        if "hasVariant" in cleaned:
            rich_variants = [
                v for v in cleaned["hasVariant"]
                if isinstance(v, dict) and len(v) > 2
            ]
            if rich_variants:
                cleaned["hasVariant"] = rich_variants
            else:
                del cleaned["hasVariant"]
        return cleaned

    structured_blocks = []
    for item in extracted.get("json-ld", []) + extracted.get("microdata", []):
        item_type = item.get("@type", "")
        if isinstance(item_type, list):
            item_type = item_type[0] if item_type else ""
        if item_type not in _PRODUCT_TYPES:
            continue
        structured_blocks.append(json.dumps(_clean_product_block(item), indent=2))
        break  # primary product is always first; skip related/upsell blocks

    # --- 3. Walk structured data recursively for image URLs ---
    structured_image_urls: list[str] = []
    _walk_for_images(extracted.get("json-ld", []), seen, structured_image_urls)
    _walk_for_images(extracted.get("microdata", []), seen, structured_image_urls)

    # --- 4. OpenGraph: images as fallback, video for video_url field ---
    og_image_urls: list[str] = []
    video_url: str | None = None
    for og in extracted.get("opengraph", []):
        og_img = (og.get("og:image") or "").strip()
        if og_img:
            normalized = _normalize_url(og_img)
            if _is_structured_image(normalized) and normalized not in seen:
                seen.add(normalized)
                og_image_urls.append(normalized)
        if not video_url:
            video_url = og.get("og:video") or og.get("og:video:url")

    # --- 5. __NEXT_DATA__ for NextJS sites (not handled by extruct) ---
    # Recursively find the product node in pageProps to surface variants/colors to the LLM.
    # Walk for images when JSON-LD images are sparse.
    _PRODUCT_KEYS = {"name", "price", "sku", "description", "variants", "variantName", "relatedProducts", "items"}

    def _find_product_node(obj: object, depth: int = 0) -> dict | None:
        if depth > 8 or not isinstance(obj, dict):
            return None
        if len(_PRODUCT_KEYS & obj.keys()) >= 3:
            return obj
        for v in obj.values():
            if isinstance(v, (dict, list)):
                result = _find_product_node(v, depth + 1) if isinstance(v, dict) else next(
                    (r for item in v if isinstance(item, dict) for r in [_find_product_node(item, depth + 1)] if r), None
                )
                if result:
                    return result
        return None

    next_data_image_urls: list[str] = []
    next_data_tag = soup.find("script", id="__NEXT_DATA__")
    if next_data_tag:
        try:
            next_data = json.loads(next_data_tag.string or "")
            # Walk for images when structured data images are sparse (fewer than 2 found)
            if len(structured_image_urls) < 2:
                _walk_for_images(next_data, seen, next_data_image_urls)
            # Find and slim the product node for structured data
            product_node = _find_product_node(next_data)
            if product_node:
                _KEEP_KEYS = {
                    "name", "variantName", "sku", "description", "price", "priceAsNumber",
                    "priceBeforeDiscount", "discountPercent", "showAsOnSale", "available",
                }
                slim = {k: v for k, v in product_node.items() if k in _KEEP_KEYS}
                item_table = product_node.get("itemTable", {})
                if item_table.get("x"):
                    slim["sizes"] = item_table["x"]
                base_sku = product_node.get("productSku", "")
                related = product_node.get("relatedProducts", [])
                if related:
                    color_variants = [
                        p.get("variantName") for p in related
                        if isinstance(p, dict)
                        and (not base_sku or p.get("productSku") == base_sku)
                        and p.get("variantName")
                    ]
                    current_color = product_node.get("variantName")
                    if current_color and current_color not in color_variants:
                        color_variants.insert(0, current_color)
                    if color_variants:
                        slim["availableColors"] = color_variants
                structured_blocks.append(json.dumps(slim, indent=2))
        except (json.JSONDecodeError, TypeError):
            pass

    # --- 6. Strip noise tags before img/text extraction ---
    for tag in soup.find_all(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    # --- 7. Scope <img> extraction to the main product container if possible ---
    product_scope = None
    for candidate in soup.find_all(True):
        if "schema.org/Product" in (candidate.get("itemtype") or ""):
            product_scope = candidate
            break
    if product_scope is None:
        for candidate in soup.find_all(True):
            testid = (candidate.get("data-testid") or candidate.get("data-test") or "").lower()
            if "product" in testid:
                product_scope = candidate
                break
    img_search_root = product_scope if product_scope is not None else soup

    # --- 8. Collect image URLs from <img> tags within the product scope ---
    img_tag_urls: list[str] = []
    for img in img_search_root.find_all("img"):
        for attr in ("src", "data-src", "data-lazy-src"):
            raw = img.get(attr, "").strip()
            if not raw:
                continue
            url = _normalize_url(raw)
            if _is_product_image(url) and url not in seen:
                seen.add(url)
                img_tag_urls.append(url)

        srcset = img.get("srcset", "").strip()
        if srcset:
            for part in srcset.split(","):
                tokens = part.strip().split()
                if not tokens:
                    continue
                raw = tokens[0]
                url = _normalize_url(raw)
                if _is_product_image(url) and url not in seen:
                    seen.add(url)
                    img_tag_urls.append(url)

    # Merge in priority order: structured data → img tags → og:image → __NEXT_DATA__
    # Cap at 20 URLs to control token spend
    image_urls = (structured_image_urls + img_tag_urls + og_image_urls + next_data_image_urls)[:20]

    # --- 9. Extract visible text ---
    # Truncate [CONTENT] to control token spend. Structured data covers the key fields;
    # [CONTENT] is a fallback, so less is needed when structured data is rich.
    content_limit = 2000 if structured_blocks else 5000
    text = soup.get_text(separator="\n", strip=True)
    lines = [line for line in text.splitlines() if line.strip()]
    visible_text = "\n".join(lines)[:content_limit]

    # --- 10. Assemble payload ---
    sections = []

    if structured_blocks:
        sections.append("[STRUCTURED DATA]\n" + "\n---\n".join(structured_blocks))

    if image_urls:
        sections.append("[IMAGES]\n" + "\n".join(image_urls))

    if video_url:
        sections.append(f"[VIDEO]\n{video_url}")

    sections.append("[CONTENT]\n" + visible_text)

    return "\n\n".join(sections)


if __name__ == "__main__":
    data_dir = Path(__file__).parent / "data"
    output_dir = data_dir / "preprocessed"
    output_dir.mkdir(exist_ok=True)

    html_files = list(data_dir.glob("*.html"))
    if not html_files:
        print("No HTML files found in data/")
    else:
        for html_file in html_files:
            print(f"Processing {html_file.name}...")
            html = html_file.read_text(encoding="utf-8")
            result = preprocess(html)
            out_path = output_dir / html_file.with_suffix(".txt").name
            out_path.write_text(result, encoding="utf-8")
            print(f"  → {out_path} ({len(result):,} chars)")
