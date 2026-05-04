from pathlib import Path
from urllib.parse import urlparse, urlunparse

from pydantic import BaseModel, ValidationInfo, field_validator


def _normalize_image_url(url: str) -> str:
    """Promote protocol-relative URLs to https and strip query params."""
    if url.startswith("//"):
        url = "https:" + url
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))


# Load categories once at module level
CATEGORIES_FILE = Path(__file__).parent / "categories.txt"
VALID_CATEGORIES: set[str] = set()
if CATEGORIES_FILE.exists():
    with open(CATEGORIES_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                VALID_CATEGORIES.add(line)


class Category(BaseModel):
    # https://www.google.com/basepages/producttype/taxonomy.en-US.txt
    name: str

    @field_validator("name")
    @classmethod
    def validate_name_exists(cls, v: str) -> str:
        if v not in VALID_CATEGORIES:
            raise ValueError(f"Category '{v}' is not a valid category in categories.txt")
        return v


class Price(BaseModel):
    price: float
    currency: str
    compare_at_price: float | None = None

    @field_validator("compare_at_price")
    @classmethod
    def nullify_equal_compare_at(cls, v: float | None, info: ValidationInfo) -> float | None:
        if v is not None and info.data.get("price") is not None and v <= info.data["price"]:
            return None
        return v


class VariantOption(BaseModel):
    name: str   # e.g. "Color", "Size", "Voltage"
    value: str  # e.g. "Black", "10", "20V"

    @field_validator("name")
    @classmethod
    def normalize_name(cls, v: str) -> str:
        return v.strip().title()


class Variant(BaseModel):
    options: list[VariantOption]
    sku: str | None = None
    price: Price | None = None      # override Product.price if this variant differs
    image_url: str | None = None    # variant-specific image
    in_stock: bool = True

    @field_validator("image_url")
    @classmethod
    def normalize_image_url(cls, v: str | None) -> str | None:
        return _normalize_image_url(v.strip()) if v else None


class Product(BaseModel):
    name: str
    price: Price
    description: str
    key_features: list[str]
    image_urls: list[str]
    video_url: str | None = None
    category: Category
    brand: str
    colors: list[str]
    variants: list[Variant]

    @field_validator("image_urls")
    @classmethod
    def normalize_and_deduplicate(cls, v: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for url in v:
            normalized = _normalize_image_url(url.strip())
            if normalized and normalized not in seen:
                seen.add(normalized)
                result.append(normalized)
        return result


class ProductSummary(BaseModel):
    id: str
    product: Product
