# libs/reward_funcs.py
import numpy as np
from functools import wraps
from libs.utils import process_rec_raw
from libs.metrics_align import _discounts, evaluate_direct_match_aligned


def reward_func_exp_inf(completions, 
                        groundtruth_with_release_year, 
                        seen_titles, 
                        rec_num, 
                        gt_catalog, 
                        **kwargs):
    title_normalizer = kwargs.get("title_normalizer")
    year_tolerance = int(kwargs.get("year_tolerance", 2))
    rec_num = int(rec_num)

    rewards = []
    for recs, gt_with_year, seen in zip(completions, 
                                        groundtruth_with_release_year, 
                                        seen_titles):
        recs_text = recs[0]["content"]
        item = {
            "raw_recs": recs_text, 
            "groundtruth_with_release_year": gt_with_year, 
            "seen_titles": seen
        }
        
        error, item = process_rec_raw(item, "raw_recs", "recs")
        if error:
            rewards.append([0.0] * rec_num)
            continue
        
        hits = evaluate_direct_match_aligned(
            item=item,
            rec_num=rec_num,
            seen_field="seen_titles",
            rec_field="recs",
            gt_field="groundtruth_with_release_year",
            gt_catalog=gt_catalog,
            title_normalizer=title_normalizer,
            year_tolerance=year_tolerance,
        ).astype(np.float64)
        
        rewards.append(hits.tolist())
    return rewards

def make_reward_func_individual(rec_num, gt_catalog):
    @wraps(reward_func_exp_inf)
    def wrapped(completions, groundtruth_with_release_year, seen_titles, **kwargs):
        return reward_func_exp_inf(completions, 
                                   groundtruth_with_release_year, 
                                   seen_titles, 
                                   rec_num, 
                                   gt_catalog, 
                                   **kwargs)
    return wrapped


def reward_func_log_decay(
    completions,
    groundtruth_with_release_year,
    seen_titles,
    rec_num,
    gt_catalog,
    **kwargs
):
    if not (
        len(completions)
        == len(groundtruth_with_release_year)
        == len(seen_titles)
    ):
        raise ValueError(
            "Batch inputs must have equal lengths: "
            f"{len(completions)=}, "
            f"{len(groundtruth_with_release_year)=}, "
            f"{len(seen_titles)=}"
        )

    title_normalizer = kwargs.get("title_normalizer", None)
    year_tolerance = int(kwargs.get("year_tolerance", 2))
    rec_num = int(rec_num)
    discounts = _discounts(rec_num)

    batch_rewards = []
    for recs, gt_with_year, seen in zip(
        completions, groundtruth_with_release_year, seen_titles
    ):
        recs_text = recs[0]["content"]
        item = {
            "raw_recs": recs_text,
            "groundtruth_with_release_year": gt_with_year,
            "seen_titles": seen,
        }

        error, item = process_rec_raw(item, "raw_recs", "recs")
        if error:
            batch_rewards.append([0.0] * rec_num)
            continue

        hits = evaluate_direct_match_aligned(
            item=item,
            rec_num=rec_num,
            seen_field="seen_titles",
            rec_field="recs",
            gt_field="groundtruth_with_release_year",
            gt_catalog=gt_catalog,
            title_normalizer=title_normalizer,
            year_tolerance=year_tolerance,
        ).astype(np.float64)

        gains = hits * discounts
        total_dcg = float(gains.sum())
        prefix_excl = np.concatenate(([0.0], np.cumsum(gains)[:-1]))
        rewards_i = (total_dcg - prefix_excl).tolist()

        batch_rewards.append(rewards_i)

    return batch_rewards


def make_reward_func(rec_num, gt_catalog):
    @wraps(reward_func_log_decay)
    def wrapped(completions, groundtruth_with_release_year, seen_titles, **kwargs):
        return reward_func_log_decay(
            completions,
            groundtruth_with_release_year,
            seen_titles,
            rec_num=rec_num,
            gt_catalog=gt_catalog,
            **kwargs
        )
    return wrapped
