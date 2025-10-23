
import numpy as np
from tqdm.autonotebook import tqdm
from editdistance import eval as distance

def recall_at_k(hits, num_gt, k):
    if len(hits) == 0:
        return 0
    hits_at_k = hits[:k]
    return sum(hits_at_k) / num_gt

def dcg_at_k(hits, k):
    if len(hits) == 0:
        return 0
    if len(hits) == 1:
        return hits[0]
    k = min(k, len(hits))
    return hits[0] + sum(hits[i] / np.log2(i + 2) for i in range(1, k))

def ndcg_at_k(hits, num_gt, k):
    if len(hits) == 0:
        return 0
    idea_hits = np.zeros(len(hits), dtype=int)
    idea_hits[:num_gt] = 1
    idea_dcg = dcg_at_k(idea_hits, k)
    dcg = dcg_at_k(hits, k)
    return dcg/idea_dcg

def remove_seen(seen_titles, rec_list):
    return [rec for rec in rec_list if rec[0] not in seen_titles]

def remove_gt_catalog(rec_list, gt_catalog):
    return [rec for rec in rec_list if rec in gt_catalog]

def evaluate_direct_match(item, k, seen_field, rec_field, gt_field, gt_catalog):
    rec_list_raw = item[rec_field]
    seen_titles = item[seen_field]
    rec_list_raw = remove_gt_catalog(remove_seen(seen_titles, rec_list_raw), gt_catalog)
    groundtruths = item[gt_field]
    
    hits = np.zeros(len(rec_list_raw), dtype=int)
    for gt in groundtruths:
        name_match_results = [distance(gt[0], rec[0]) for rec in rec_list_raw]
        year_match_results = [np.abs(int(gt[1])-int(rec[1])) for rec in rec_list_raw]
        matched = False
        for i, (name_res, year_res) in enumerate(
            zip(name_match_results, year_match_results)
        ):
            if name_res == 0 and year_res <= 2 and not matched:
                hits[i] = 1
                matched = True

    num_gt = len(groundtruths)
    recall = recall_at_k(hits, num_gt, k)
    ndcg = ndcg_at_k(hits, num_gt, k)
    return recall, ndcg