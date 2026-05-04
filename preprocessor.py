import json
import re
from pathlib import Path
from bs4 import BeautifulSoup

# Image extensions we consider real product images
_IMAGE_EXTENSIONS = re.compile(r'\.(jpg|jpeg|png|webp|avif)(\?|$)', re.IGNORECASE)

# URL fragments that indicate UI/chrome images rather than product images
_NOISE_PATTERNS = re.compile(
    r'(\.svg|/resources/|/icons?/|logo|favicon|spinner|loading|blank\.gif|analytics)',
    re.IGNORECASE,
)


def _is_product_image(url: str) -> bool:
    """Return True if the URL looks like a real product image, not UI chrome."""
    return bool(_IMAGE_EXTENSIONS.search(url)) and not _NOISE_PATTERNS.search(url)


def _walk_json_ld_images(data: object, seen: set[str], out: list[str]) -> None:
    """Recursively walk a JSON-LD object and collect all image URLs."""
    if isinstance(data, str):
        if _is_product_image(data) and data not in seen:
            seen.add(data)
            out.append(data)
    elif isinstance(data, list):
        for item in data:
            _walk_json_ld_images(item, seen, out)
    elif isinstance(data, dict):
        for key, value in data.items():
            if key.lower() in ("image", "images"):
                _walk_json_ld_images(value, seen, out)
            else:
                _walk_json_ld_images(value, seen, out)


def preprocess(html: str) -> str:
    """
    Strip a raw HTML page down to a plain text payload for LLM extraction.

    Output sections:
      [STRUCTURED DATA]  — JSON-LD blocks found in the page
      [IMAGES]           — deduplicated product image URLs
      [CONTENT]          — visible text after stripping noise tags
    """
    soup = BeautifulSoup(html, "lxml")
    seen: set[str] = set()

    # --- 1. Extract JSON-LD before stripping scripts ---
    json_ld_blocks = []
    json_ld_image_urls: list[str] = []
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
            json_ld_blocks.append(json.dumps(data, indent=2))
            _walk_json_ld_images(data, seen, json_ld_image_urls)
        except (json.JSONDecodeError, TypeError):
            pass

    # --- 2. OpenGraph meta tags (og:image fallback + og:video for video_url) ---
    og_image_urls: list[str] = []
    video_url: str | None = None
    for meta in soup.find_all("meta"):
        prop = meta.get("property", "") or meta.get("name", "")
        content = (meta.get("content") or "").strip()
        if not content:
            continue
        if prop == "og:image" and _is_product_image(content) and content not in seen:
            seen.add(content)
            og_image_urls.append(content)
        elif prop == "og:video" and not video_url:
            video_url = content

    # --- 3. __NEXT_DATA__ script targeting (NextJS sites with no <img src>) ---
    next_data_image_urls: list[str] = []
    if len(json_ld_image_urls) < 2:
        next_data_tag = soup.find("script", id="__NEXT_DATA__")
        if next_data_tag:
            try:
                next_data = json.loads(next_data_tag.string or "")
                _walk_json_ld_images(next_data, seen, next_data_image_urls)
            except (json.JSONDecodeError, TypeError):
                pass

    # --- 4. Strip noise tags ---
    for tag in soup.find_all(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    # --- 5. Find main product container to scope <img> extraction ---
    product_scope = None
    for candidate in soup.find_all(True):
        itemtype = candidate.get("itemtype", "")
        if "schema.org/Product" in itemtype:
            product_scope = candidate
            break
    if product_scope is None:
        for candidate in soup.find_all(True):
            testid = candidate.get("data-testid", "") or candidate.get("data-test", "")
            if "product" in testid.lower():
                product_scope = candidate
                break
    img_search_root = product_scope if product_scope is not None else soup

    # --- 6. Collect image URLs from <img> tags within product scope ---
    img_tag_urls: list[str] = []
    for img in img_search_root.find_all("img"):
        for attr in ("src", "data-src", "data-lazy-src"):
            url = img.get(attr, "").strip()
            if url and _is_product_image(url) and url not in seen:
                seen.add(url)
                img_tag_urls.append(url)

        srcset = img.get("srcset", "").strip()
        if srcset:
            for part in srcset.split(","):
                url = part.strip().split()[0]
                if url and _is_product_image(url) and url not in seen:
                    seen.add(url)
                    img_tag_urls.append(url)

    # Merge in priority order: JSON-LD (intentional) → img tags → og:image → __NEXT_DATA__
    # Cap at 20 URLs to control token spend
    image_urls = (json_ld_image_urls + img_tag_urls + og_image_urls + next_data_image_urls)[:20]

    # --- 7. Extract visible text ---
    text = soup.get_text(separator="\n", strip=True)
    lines = [line for line in text.splitlines() if line.strip()]
    visible_text = "\n".join(lines)

    # --- 8. Assemble payload ---
    sections = []

    if json_ld_blocks:
        sections.append("[STRUCTURED DATA]\n" + "\n---\n".join(json_ld_blocks))

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
