import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import Product, ProductSummary

OUTPUT_DIR = Path(__file__).parent / "data" / "output"

_products: dict[str, ProductSummary] = {}


def _load_products() -> None:
    for path in sorted(OUTPUT_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            product = Product.model_validate(data)
            _products[path.stem] = ProductSummary(id=path.stem, product=product)
        except Exception as e:
            print(f"[server] Skipping {path.name}: {e}")
    print(f"[server] Loaded {len(_products)} products: {', '.join(_products)}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_products()
    yield


app = FastAPI(title="Channel3 Product API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/products", response_model=list[ProductSummary])
def list_products() -> list[ProductSummary]:
    return list(_products.values())


@app.get("/products/{id}", response_model=ProductSummary)
def get_product(id: str) -> ProductSummary:
    if id not in _products:
        raise HTTPException(status_code=404, detail=f"Product '{id}' not found")
    return _products[id]
