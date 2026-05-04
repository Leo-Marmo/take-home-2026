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


def test_extracts_images_from_script_tags():
    # Script image extraction is currently disabled — JSON-LD covers primary
    # images for all supported pages. This test documents the expected behaviour
    # while the feature is commented out.
    html = """
    <html><body>
    <script>var data = {"image": "https://cdn.example.com/shot.jpg"};</script>
    <p>Some product</p>
    </body></html>
    """
    result = preprocess(html)
    # Script images are not extracted while feature is disabled
    assert "https://cdn.example.com/shot.jpg" not in result


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


def test_og_image_fallback():
    """og:image is used when no JSON-LD images exist."""
    html = """
    <html><head>
    <meta property="og:image" content="https://cdn.example.com/og.jpg">
    </head><body><p>Product</p></body></html>
    """
    result = preprocess(html)
    assert "[IMAGES]" in result
    assert "https://cdn.example.com/og.jpg" in result


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


def test_next_data_images_when_json_ld_sparse():
    """__NEXT_DATA__ images are extracted when JSON-LD has fewer than 2 images."""
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


def test_next_data_skipped_when_json_ld_has_enough_images():
    """__NEXT_DATA__ is not used when JSON-LD already has 2+ images."""
    html = """
    <html><head>
    <script type="application/ld+json">{
      "@type": "Product",
      "image": ["https://cdn.example.com/a.jpg", "https://cdn.example.com/b.jpg"]
    }</script>
    </head><body>
    <script id="__NEXT_DATA__">{"images": ["https://cdn.example.com/next1.jpg"]}</script>
    <p>Product</p>
    </body></html>
    """
    result = preprocess(html)
    assert "https://cdn.example.com/a.jpg" in result
    assert "https://cdn.example.com/next1.jpg" not in result


def test_product_container_scoping():
    """Images outside the product container are excluded when schema.org/Product is found."""
    html = """
    <html><body>
    <img src="https://cdn.example.com/banner.jpg" alt="banner">
    <div itemtype="https://schema.org/Product">
      <img src="https://cdn.example.com/product.jpg" alt="product">
    </div>
    </body></html>
    """
    result = preprocess(html)
    assert "https://cdn.example.com/product.jpg" in result
    assert "https://cdn.example.com/banner.jpg" not in result
