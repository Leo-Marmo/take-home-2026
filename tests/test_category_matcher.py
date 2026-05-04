from category_matcher import CategoryMatcher, MAX_RESULTS

matcher = CategoryMatcher()


def test_returns_list_of_strings():
    results = matcher.match("cordless drill dewalt 20v battery")
    assert isinstance(results, list)
    assert all(isinstance(r, str) for r in results)


def test_returns_at_most_max_results():
    results = matcher.match("cordless drill dewalt 20v battery")
    assert len(results) <= MAX_RESULTS


def test_drill_payload_returns_power_tool_categories():
    results = matcher.match("cordless drill dewalt 20v battery power tool")
    combined = " ".join(results).lower()
    assert "drill" in combined or "power" in combined or "tool" in combined


def test_footwear_payload_returns_shoe_categories():
    results = matcher.match("nike air force shoes sneakers footwear size")
    combined = " ".join(results).lower()
    assert "shoe" in combined or "footwear" in combined or "athletic" in combined


def test_empty_payload_returns_empty_list():
    results = matcher.match("")
    assert results == []


def test_results_are_valid_taxonomy_categories():
    from models import VALID_CATEGORIES
    results = matcher.match("cotton trousers pants clothing apparel")
    for r in results:
        assert r in VALID_CATEGORIES
