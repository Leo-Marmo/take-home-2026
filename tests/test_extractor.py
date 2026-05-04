import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch
from pydantic import ValidationError

from models import Product, Price, Category, Variant
from extractor import extract, run


def _make_product() -> Product:
    return Product(
        name="Test Drill",
        price=Price(price=99.99, currency="USD"),
        description="A great drill.",
        key_features=["Cordless", "20V"],
        image_urls=["https://example.com/drill.jpg"],
        category=Category(name="Hardware > Tools > Drills"),
        brand="DeWalt",
        colors=["Black"],
        variants=[],
    )


MINIMAL_HTML = """
<html><body>
<script type="application/ld+json">{"@type": "Product", "name": "Test Drill"}</script>
<p>A great drill. Cordless. 20V.</p>
</body></html>
"""


@pytest.mark.asyncio
async def test_extract_returns_product_on_success():
    product = _make_product()
    with patch("extractor.ai.responses", new=AsyncMock(return_value=product)):
        result = await extract(MINIMAL_HTML, "test")
    assert isinstance(result, Product)
    assert result.name == "Test Drill"


@pytest.mark.asyncio
async def test_extract_retries_on_first_failure():
    product = _make_product()
    mock = AsyncMock(side_effect=[ValidationError.from_exception_data("Product", []), product])
    with patch("extractor.ai.responses", new=mock):
        result = await extract(MINIMAL_HTML, "test")
    assert mock.call_count == 2
    assert isinstance(result, Product)


@pytest.mark.asyncio
async def test_extract_returns_none_after_all_retries():
    from extractor import MAX_RETRIES
    error = Exception("LLM failed")
    total_attempts = MAX_RETRIES + 1
    mock = AsyncMock(side_effect=[error] * total_attempts)
    with patch("extractor.ai.responses", new=mock):
        result = await extract(MINIMAL_HTML, "test")
    assert result is None
    assert mock.call_count == total_attempts


@pytest.mark.asyncio
async def test_run_writes_json_output(tmp_path):
    # Write a minimal HTML file to tmp_path
    html_file = tmp_path / "test.html"
    html_file.write_text(MINIMAL_HTML)

    product = _make_product()
    with patch("extractor.ai.responses", new=AsyncMock(return_value=product)):
        await run(tmp_path)

    out_file = tmp_path / "output" / "test.json"
    assert out_file.exists()
    data = json.loads(out_file.read_text())
    assert data["name"] == "Test Drill"
