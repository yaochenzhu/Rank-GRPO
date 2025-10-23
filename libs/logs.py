import os, hashlib, wandb
from datetime import datetime
from accelerate import Accelerator
from libs.logs import _stable_wandb_run_id

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

def setup_environment(wandb_project):
    accelerator = Accelerator()
    os.environ["WANDB_PROJECT"] = wandb_project
    return accelerator

def setup_wandb(accelerator, output_dir, model_name, sft_checkpoint, seed, project_name):
    if accelerator.is_main_process:
        wandb_id = _stable_wandb_run_id(output_dir, model_name, sft_checkpoint, seed)
        run_name = f"{model_name}-sft{sft_checkpoint}-seed{seed}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        wandb.init(project=project_name, id=wandb_id, resume="allow", name=run_name, reinit=True)
        return run_name
    else:
        os.environ["WANDB_DISABLED"] = "true"
        return None
