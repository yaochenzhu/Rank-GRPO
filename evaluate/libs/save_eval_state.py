import json, time, re
from pathlib import Path

def _list_to_map(list_of_tuples):
    m = {}
    for t in list_of_tuples:
        if not isinstance(t, (list, tuple)) or len(t) < 2:
            continue
        step = int(t[0])
        avg = float(t[1])
        std = float(t[2]) if len(t) > 2 and t[2] is not None else None
        m[step] = (avg, std)
    return m

def save_analysis_state(model_root: str,
                        rec_nums,
                        avg_metrics,
                        filename: str = "analysis_state.json",
                        analysis_name: str = "posthoc_rec_eval"):
    model_dir = Path(model_root).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    recnum_map = _list_to_map(rec_nums)
    steps = sorted(set(avg_metrics.keys()) | set(recnum_map.keys()))

    state = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_dir": str(model_dir),
        "analysis_name": analysis_name,
        "log_history": []
    }

    for step in steps:
        entry = {
            "step": int(step),
            "checkpoint": f"checkpoint-{int(step)}",
        }

        # rec num
        if step in recnum_map:
            avg_n, std_n = recnum_map[step]
            entry["eval_rec_num"] = float(avg_n)
            if std_n is not None:
                entry["eval_rec_num_std"] = float(std_n)

        # recall/ndcg
        rec_dict, ndcg_dict = avg_metrics.get(step, ({}, {}))
        if rec_dict:
            for K, v in rec_dict.items():
                entry[f"eval_recall@{K}"] = float(v)
        if ndcg_dict:
            for K, v in ndcg_dict.items():
                entry[f"eval_ndcg@{K}"] = float(v)

        state["log_history"].append(entry)

    out_path = model_dir / filename
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    tmp.replace(out_path)
    print(f"[OK] Saved analysis state → {out_path}")
    return out_path