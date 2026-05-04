from preprocessor import preprocess


MINIMAL_HTML = """
<html>
<head>
  <script type="application/ld+json">{"@type": "Product", "name": "Test Product"}</script>
  <style>body { color: red; }</style>
</head>
<body>
  <nav>Nav noise</nav>
  <header>Header noise</header>
  <img src="https://example.com/product.jpg" alt="Product">
  <p>Product description here.</p>
  <footer>Footer noise</footer>
  <script>var x = 1;</script>
</body>
</html>
"""


def test_extracts_json_ld():
    result = preprocess(MINIMAL_HTML)
    assert "[STRUCTURED DATA]" in result
    assert '"name": "Test Product"' in result


def test_extracts_img_src():
    result = preprocess(MINIMAL_HTML)
    assert "[IMAGES]" in result
    assert "https://example.com/product.jpg" in result


def test_filters_svg_noise():
    html = """
    <html><body>
    <img src="/resources/images/logo.svg" alt="logo">
    <img src="https://cdn.example.com/product.jpg" alt="product">
    </body></html>
    """
    result = preprocess(html)
    assert "logo.svg" not in result
    assert "product.jpg" in result


def test_strips_noise_tags():
    result = preprocess(MINIMAL_HTML)
    assert "Nav noise" not in result
    assert "Header noise" not in result
    assert "Footer noise" not in result


def test_retains_visible_content():
    result = preprocess(MINIMAL_HTML)
    assert "[CONTENT]" in result
    assert "Product description here." in result


def test_no_json_ld_omits_structured_data_section():
    html = "<html><body><p>Just text</p></body></html>"
    result = preprocess(html)
    assert "[STRUCTURED DATA]" not in result
    assert "[CONTENT]" in result


def test_no_images_omits_images_section():
    html = "<html><body><p>Just text, no images</p></body></html>"
    result = preprocess(html)
    assert "[IMAGES]" not in result


def test_recursive_json_ld_images():
    """Images nested inside hasVariant are extracted."""
    html = """
    <html><head>
    <script type="application/ld+json">{
      "@type": "Product",
      "name": "Shoe",
      "hasVariant": [
        {"image": ["https://cdn.example.com/variant1.jpg"]},
        {"image": ["https://cdn.example.com/variant2.jpg"]}
      ]
    }</script>
    </head><body><p>Shoe</p></body></html>
    """
    result = preprocess(html)
    assert "https://cdn.example.com/variant1.jpg" in result
    assert "https://cdn.example.com/variant2.jpg" in result


def test_microdata_emits_structured_data():
    """Schema.org microdata is extracted into [STRUCTURED DATA]."""
    html = """
    <html><body>
    <div itemscope itemtype="https://schema.org/Product">
      <span itemprop="name">Microdata Product</span>
      <span itemprop="description">A great product</span>
    </div>
    </body></html>
    """
    result = preprocess(html)
    assert "[STRUCTURED DATA]" in result
    assert "Microdata Product" in result


def test_og_image_fallback():
    """og:image is used when no structured data images exist."""
    html = """
    <html><head>
    <meta property="og:image" content="https://cdn.example.com/og.jpg">
    </head><body><p>Product</p></body></html>
    """
    result = preprocess(html)
    assert "[IMAGES]" in result
    assert "https://cdn.example.com/og.jpg" in result


def test_og_image_without_extension():
    """og:image URLs without file extensions are still included (e.g. image server URLs)."""
    html = """
    <html><head>
    <meta property="og:image" content="https://cdn.example.com/is/image/product/123">
    </head><body><p>Product</p></body></html>
    """
    result = preprocess(html)
    assert "https://cdn.example.com/is/image/product/123" in result


def test_og_video_emits_video_section():
    """og:video produces a [VIDEO] section."""
    html = """
    <html><head>
    <meta property="og:video" content="https://cdn.example.com/product.mp4">
    </head><body><p>Product</p></body></html>
    """
    result = preprocess(html)
    assert "[VIDEO]" in result
    assert "https://cdn.example.com/product.mp4" in result


def test_cdn_deduplication():
    """CDN variants of the same image (different query params) collapse to one URL in [IMAGES]."""
    html = """
    <html><head>
    <script type="application/ld+json">{
      "@type": "Product",
      "name": "Sofa",
      "image": "https://cdn.example.com/products/sofa.jpg"
    }</script>
    </head><body>
    <img src="https://cdn.example.com/products/sofa.jpg?w=320&q=80" alt="sofa">
    <img src="https://cdn.example.com/products/sofa.jpg?w=640&q=40" alt="sofa">
    </body></html>
    """
    result = preprocess(html)
    images_section = result.split("[IMAGES]")[1].split("\n\n")[0]
    urls = [l.strip() for l in images_section.strip().splitlines() if "sofa.jpg" in l]
    assert len(urls) == 1
    assert "?" not in urls[0]


def test_protocol_relative_urls_promoted():
    """Protocol-relative image URLs are promoted to https:// in [IMAGES]."""
    html = """
    <html><head>
    <script type="application/ld+json">{
      "@type": "Product",
      "name": "Drill",
      "image": "//cdn.example.com/drill.jpg"
    }</script>
    </head><body><p>Drill</p></body></html>
    """
    result = preprocess(html)
    images_section = result.split("[IMAGES]")[1].split("\n\n")[0]
    assert "https://cdn.example.com/drill.jpg" in images_section


def test_next_data_images_when_structured_sparse():
    """__NEXT_DATA__ images are extracted when structured data has fewer than 2 images."""
    html = """
    <html><head></head><body>
    <script id="__NEXT_DATA__">{"props": {"pageProps": {"product": {
      "images": ["https://cdn.example.com/next1.jpg", "https://cdn.example.com/next2.jpg"]
    }}}}</script>
    <p>Product</p>
    </body></html>
    """
    result = preprocess(html)
    assert "https://cdn.example.com/next1.jpg" in result
    assert "https://cdn.example.com/next2.jpg" in result


def test_product_container_scoping():
    """Images outside the product container are excluded when schema.org/Product is found."""
    html = """
    <html><body>
    <img src="https://cdn.example.com/banner.jpg" alt="banner">
    <div itemscope itemtype="https://schema.org/Product">
      <img src="https://cdn.example.com/product.jpg" alt="product">
    </div>
    </body></html>
    """
    result = preprocess(html)
    assert "https://cdn.example.com/product.jpg" in result
    assert "https://cdn.example.com/banner.jpg" not in result
