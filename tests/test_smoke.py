from src.label_mapping import map_article_type, map_color, map_gender


def test_article_type_mapping():
    selected = {"Shoes": ["Casual Shoes", "Sports Shoes"], "Shirt": ["Shirts"]}
    assert map_article_type("Casual Shoes", selected) == "Shoes"
    assert map_article_type("Shirts", selected) == "Shirt"
    assert map_article_type("Unknown", selected) is None


def test_gender_mapping():
    mapping = {"Men": "Men", "Women": "Women", "Boys": "Kids"}
    assert map_gender("Boys", mapping) == "Kids"
    assert map_gender("Men", mapping) == "Men"


def test_color_mapping():
    keep = ["Black", "Blue"]
    assert map_color("Black", keep) == "Black"
    assert map_color("Maroon", keep) == "Other"
