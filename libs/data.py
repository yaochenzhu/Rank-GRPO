import pickle


def _normalize_catalog_item(item):
    if not isinstance(item, (list, tuple)) or len(item) < 2:
        return item
    title, year = item[0], item[1]
    try:
        year = int(year)
    except Exception:
        pass
    return (title, year)


def load_catalog(catalog_path):
    with open(catalog_path, "rb") as f:
        return {_normalize_catalog_item(item) for item in pickle.load(f)}