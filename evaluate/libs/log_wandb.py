import os, json, hashlib
from pathlib import Path
from collections import defaultdict
import wandb


def find_latest_trainer_state(model_dir: Path) -> Path:
    """Find trainer_state.json inside the highest-numbered checkpoint-N directory."""
    checkpoints = []
    for p in model_dir.glob("checkpoint-*"):
        if p.is_dir() and p.name.startswith("checkpoint-"):
            try:
                step = int(p.name.split("-")[-1])
                checkpoints.append((step, p))
            except ValueError:
                pass
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint-* directories in {model_dir}")
    _, latest_ckpt = max(checkpoints, key=lambda x: x[0])
    ts_path = latest_ckpt / "trainer_state.json"
    if not ts_path.exists():
        raise FileNotFoundError(f"No trainer_state.json in {latest_ckpt}")
    return ts_path

def stable_run_id(model_root: str, suffix: str, extra: str = "merged") -> str:
    """Deterministic run_id: model path + suffix + 'merged'"""
    model_root = str(model_root) + "__merged__"
    h = hashlib.sha1()
    h.update(str(Path(model_root).resolve()).encode("utf-8"))
    h.update(b"|")
    h.update((extra + "|" + suffix).encode("utf-8"))
    return h.hexdigest()[:32]

def load_json(path: Path):
    return json.load(open(path)) if path.exists() else None

def merge_log_histories(trainer_hist, analysis_hist):
    merged = defaultdict(dict)
    for e in trainer_hist:
        step = int(e.get("step", -1))
        merged[step].update(e)
    for e in analysis_hist:
        step = int(e.get("step", -1))
        merged[step].update(e)
    return [merged[s] for s in sorted(merged.keys()) if s >= 0]

def merge_states(trainer_state, analysis_state, merged_filename):
    out = dict(trainer_state)  # shallow copy
    out["analysis_name"] = analysis_state.get("analysis_name", "posthoc_rec_eval")
    out["log_history"] = merge_log_histories(
        trainer_state.get("log_history", []),
        analysis_state.get("log_history", [])
    )
    # strip best metrics (optional)
    out.pop("best_metric", None)
    out.pop("best_model_checkpoint", None)
    return out

def save_json(state, path: Path):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    tmp.replace(path)
    
def extract_model_name(model_dir: Path) -> str:
    """
    Extract a clean model name from the path, e.g. 'Llama-3.2-3B-Instruct' or 'Qwen-1.5B'.
    Falls back to model_dir.name if nothing matches.
    """
    parts = [p for p in model_dir.parts]
    for p in parts:
        if p.startswith("Llama-") or p.startswith("Qwen-"):
            return p
    return model_dir.name

def log_to_wandb(model_dir: Path, merged_state, project, run_name_suffix, merged_filename):
    model_name = extract_model_name(model_dir)   # <- use our extractor
    run_id = stable_run_id(model_dir, run_name_suffix)
    run_name = f"{model_name}-{run_name_suffix}"

    os.environ["WANDB_PROJECT"] = project
    run = wandb.init(
        project=project,
        name=run_name,
        id=run_id,
        resume="allow",
        reinit=True,
    )

    # save merged file locally + log as artifact
    merged_path = model_dir / merged_filename
    save_json(merged_state, merged_path)
    art = wandb.Artifact(name=f"{model_name}-{merged_filename}", type="merged-state")
    art.add_file(str(merged_path))
    run.log_artifact(art)

    # log metrics by step
    for e in merged_state["log_history"]:
        step = e.get("step", None)
        payload = {}
        for k, v in e.items():
            if k in ("step", "epoch"): 
                continue
            if k.startswith("eval_"):
                payload[f"val/{k[5:]}"] = v
            elif k.startswith("train_") or k in ("loss", "learning_rate", "grad_norm"):
                payload[f"train/{k}"] = v
            else:
                payload[k] = v
        if step is not None:
            wandb.log(payload, step=step)
        else:
            wandb.log(payload)

    run.finish()
    print(f"[OK] Uploaded merged run → {project}/{run_name} (run_id={run_id})")

def merge_and_upload(model_dir: str,
                     project: str,
                     run_name_suffix: str,
                     merged_filename: str,
                     upload=True):
    model_dir = Path(model_dir).resolve()

    # 1. get trainer_state.json from the latest checkpoint
    trainer_path = find_latest_trainer_state(model_dir)

    # 2. analysis_state.json is always in the model root
    analysis_path = model_dir / "analysis_state.json"
    if not analysis_path.exists():
        raise FileNotFoundError(f"Missing analysis_state.json in {model_dir}")

    trainer = load_json(trainer_path)
    analysis = load_json(analysis_path)
    merged = merge_states(trainer, analysis, merged_filename=merged_filename)

    if upload:
        log_to_wandb(model_dir, merged, project=project,
                     run_name_suffix=run_name_suffix,
                     merged_filename=merged_filename)
    else:
        save_json(merged, model_dir / merged_filename)
        print(f"[OK] Saved merged JSON only → {model_dir/merged_filename}")