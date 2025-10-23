import re
import numpy as np
from functools import wraps, lru_cache

def _safe_int_year(y):
    try:
        return int(str(y))
    except Exception:
        return None

def _default_title_normalizer(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s).strip())
    return s.casefold()

@lru_cache(maxsize=32)
def _discounts(rec_num: int) -> np.ndarray:
    rec_num = int(rec_num)
    d = np.ones(rec_num, dtype=np.float64)
    if rec_num > 1:
        idx = np.arange(1, rec_num, dtype=np.float64)
        d[1:] = 1.0 / np.log2(idx + 2.0)
    return d


def evaluate_direct_match_aligned(
    item,
    rec_num: int,
    seen_field: str,
    rec_field: str,
    gt_field: str,
    gt_catalog,
    *,
    title_normalizer=None,
    year_tolerance: int = 2,
):
    recs = item[rec_field]
    seen_titles = set(item[seen_field])
    groundtruths = item[gt_field]

    norm = title_normalizer or _default_title_normalizer
    rec_num = int(rec_num)
    L = min(len(recs), rec_num)
    hits = np.zeros(rec_num, dtype=np.int32) 

    gt_years_by_title = {}
    for gt_title, gt_year in groundtruths:
        key = norm(gt_title)
        y = _safe_int_year(gt_year)
        if y is None:
            continue
        bucket = gt_years_by_title.setdefault(key, {})
        bucket[y] = bucket.get(y, 0) + 1

    for pos in range(L):
        rec_title, rec_year = recs[pos]

        if rec_title in seen_titles:
            continue

        if (rec_title, rec_year) not in gt_catalog:
            continue

        y = _safe_int_year(rec_year)
        if y is None:
            continue

        key = norm(rec_title)
        bucket = gt_years_by_title.get(key)
        if not bucket:
            continue

        matched_year = None
        if year_tolerance <= 0:
            if bucket.get(y, 0) > 0:
                matched_year = y
        else:
            if bucket.get(y, 0) > 0:
                matched_year = y
            else:
                for d in range(1, year_tolerance + 1):
                    yp, ym = y + d, y - d
                    if bucket.get(yp, 0) > 0:
                        matched_year = yp
                        break
                    if bucket.get(ym, 0) > 0:
                        matched_year = ym
                        break

        if matched_year is None:
            continue

        hits[pos] = 1
        cnt = bucket[matched_year] - 1
        if cnt <= 0:
            del bucket[matched_year]
            if not bucket:
                del gt_years_by_title[key]
        else:
            bucket[matched_year] = cnt
    return hits