import os
import hashlib
from datetime import datetime
from accelerate import Accelerator


def uses_wandb(report_to):
    return "wandb" in normalize_report_to(report_to)


def normalize_report_to(report_to):
    if report_to in (None, "none"):
        return []
    if report_to == "both":
        return ["tensorboard", "wandb"]
    if isinstance(report_to, str):
        return [report_to]
    return list(report_to)

def _stable_wandb_run_id(output_dir, model_name, checkpoint, seed):
    """
    Persist a deterministic run id under output_dir so restarts reuse it.
    """
    rid_file = os.path.join(output_dir, ".wandb_run_id")
    if os.path.exists(rid_file):
        with open(rid_file, "r") as f:
            return f.read().strip()

    raw = f"{os.path.abspath(output_dir)}|{model_name}|sft{checkpoint}|seed{seed}"
    rid = hashlib.sha1(raw.encode()).hexdigest()[:16]
    os.makedirs(output_dir, exist_ok=True)
    with open(rid_file, "w") as f:
        f.write(rid)
    return rid

def setup_environment(wandb_project=None, report_to="tensorboard"):
    accelerator = Accelerator()
    if uses_wandb(report_to):
        os.environ["WANDB_PROJECT"] = wandb_project or "rank_grpo"
    else:
        os.environ.setdefault("WANDB_DISABLED", "true")
    return accelerator

def setup_run(accelerator, output_dir, model_name, sft_checkpoint, seed, project_name, report_to="tensorboard"):
    run_name = f"{model_name}-sft{sft_checkpoint}-seed{seed}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    if not uses_wandb(report_to):
        return run_name

    try:
        import wandb
    except ImportError as exc:
        raise ImportError("W&B logging requested but `wandb` is not installed. Install wandb or use --report_to tensorboard.") from exc

    if accelerator.is_main_process:
        wandb_id = _stable_wandb_run_id(output_dir, model_name, sft_checkpoint, seed)
        wandb.init(project=project_name, id=wandb_id, resume="allow", name=run_name, reinit=True)
        return run_name

    os.environ["WANDB_DISABLED"] = "true"
    return None
