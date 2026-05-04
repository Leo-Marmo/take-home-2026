# Channel3 Take-Home

An LLM-powered pipeline that extracts structured product data from raw HTML pages and serves it through a React web app.

---

## Setup

Requires Python 3.12+ and Node 18+.

Create a `.env` file in the root with your OpenRouter key:

```
OPEN_ROUTER_API_KEY=your_key_here
```

Install Python dependencies:

```bash
uv sync --extra embeddings
# or: pip install -e ".[embeddings]"
```

Build the category embedding index (one-time, ~30s):

```bash
uv run python scripts/generate_embeddings.py
```

Install frontend dependencies:

```bash
cd frontend && npm install
```

---

## Running the Pipeline

The pipeline has four steps. Run them in order when processing new HTML files.

**1. Preprocess** — strips HTML noise and extracts structured signals into a plain-text payload:

```bash
uv run python preprocessor.py
# reads:  data/*.html
# writes: data/preprocessed/*.txt
```

**2. Extract** — runs the LLM over each preprocessed payload and writes structured JSON:

```bash
uv run python extractor.py
# reads:  data/*.html  (preprocesses internally)
# writes: data/output/*.json
```

To process specific files only, use `--only` (can be repeated):

```bash
uv run python extractor.py --only llbean --only allbirds
```

**3. Serve** — starts the FastAPI server on port 8000:

```bash
uv run uvicorn server:app --port 8000
```

**4. Frontend** — starts the React dev server on port 5173:

```bash
cd frontend && npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

---

## Adding New Products

Drop any product page HTML file into `data/` and rerun the extractor:

```bash
uv run python extractor.py
```

The server reloads products at startup, so restart it after extraction completes.

---

## Running Tests

```bash
uv run pytest
```

---

## Architecture

```
data/*.html
     ↓
preprocessor.py   — extracts JSON-LD, microdata, OpenGraph, __NEXT_DATA__,
                    image URLs, and visible text into a labeled plain-text payload
     ↓
extractor.py      — category matcher narrows the 5,596-entry Google taxonomy to
                    ~20 candidates; single LLM call returns a validated Product
     ↓
data/output/*.json
     ↓
server.py         — FastAPI: GET /products, GET /products/{id}
     ↓
frontend/         — React catalog grid + PDP
```

### Preprocessor

The preprocessor is the most important component for generalization. It runs entirely in Python with no LLM involvement, so it's fast, deterministic, and free to run. The goal is to extract every signal the LLM might need while stripping everything it doesn't.

It uses `extruct` to handle JSON-LD, Schema.org microdata, and OpenGraph in a single pass — covering the structured data formats used by the vast majority of e-commerce sites. For Next.js sites that don't serialize product data into JSON-LD, it falls back to parsing `__NEXT_DATA__`. Image URLs are deduplicated and normalized before being passed to the LLM so it can copy them verbatim without guessing.

### Category Matching

Sending all 5,596 Google taxonomy strings to the LLM on every call would be expensive and slow. Instead, a two-stage approach narrows the field first:

1. **Word-overlap with synonym expansion** — product-world terms (`tee`, `trousers`, `runner`) are mapped to taxonomy-world terms (`shirt`, `pants`, `shoes`) before scoring. Fit descriptors like "Traditional Fit" are stripped so they don't pollute results with ceremonial clothing categories.
2. **Semantic embeddings** (`all-MiniLM-L6-v2`) — the pre-computed index is queried in parallel and its results are merged with the word-overlap set, adding candidates that keyword matching would miss for unusual or branded product names.

The LLM is then asked to pick exactly one string from the merged ~20 candidates.

---

## System Design

The core extraction logic — preprocess, match categories, call the LLM, validate with Pydantic — is already stateless and parallelizable. What doesn't scale is everything around it, including the HTML files downloaded by hand, extraction triggered manually, and a server that loads all products into memory at startup. To scale this, I would introduce a crawl layer using Playwright (with per-domain rate limiting and robots.txt compliance), swap out the JSON document store for a Postgres db with JSONB columns, and make extraction event-driven using a messaging queue like SQS (or Kafka in order to reprocess events from a certain point in time). The preprocessor in Python can run on any worker horizontally, but the LLM call is the cost bottleneck. Content-hashing the preprocessed payload and skipping re-extraction when the hash hasn't changed reduces redundant LLM calls. Structured output validation is already in place with retry strategy so faulty LLM responses will get caught and retried instead of silently written to the data store. Another bottleneck currently is the category matching which runs in O(N). Moving category matching to an approximate nearest neighbors index would reduce this to near constant time. On monitoring, we'd stream real-time pipeline metrics and track validation failure rate, category distribution drift, and image extraction rate per domain. A sudden spike in validation failures is the earliest signal that a site's HTML structure changed and the preprocessor needs updating.

The current REST API is enough for a browser UI, but agents need richer query interfaces. I'd expose a semantic search endpoint that accepts a natural language query and returns ranked products using embedding similarity, a structured filter API (category, price range, in-stock, brand), and a comparison endpoint that takes a list of product IDs and returns a normalized side-by-side schema. The existing Product model with its Google taxonomy category, structured variants, and explicit pricing is already well-suited to agent consumption — agents can reason over a consistent schema rather than scraping raw text. Long-term, exposing the taxonomy tree itself as a navigable API lets developers build category-aware browsing and recommendation features without coupling to our internal representation.
