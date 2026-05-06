def catalog_contains(gt_catalog, title, year) -> bool:
    """Return True when a catalog contains a title/year pair.

    Catalogs have historically stored release years as either integers or
    strings, while parsed recommendations usually provide integer years.
    """
    return (title, year) in gt_catalog or (title, str(year)) in gt_catalog
