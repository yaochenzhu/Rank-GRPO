import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "libs", "trl"))

import json
import math
import shutil

from datasets import load_from_disk
from rank_grpo_trainer import GRPOConfig, RankGRPOTrainer
import argparse
from transformers import TrainerCallback

from libs.data import load_catalog
from libs.utils import StepLRSchedulerCallback
from libs.reward_funcs import make_reward_func, make_reward_func_individual
from libs.logs import normalize_report_to, setup_environment, setup_run


def _safe_name(value):
    return str(value).rstrip("/").replace(os.sep, "_").replace("/", "_")


def resolve_sft_model_path(model_name, sft_checkpoint):
    if os.path.isdir(model_name):
        checkpoint_path = os.path.join(model_name, f"checkpoint-{sft_checkpoint}")
        return checkpoint_path if os.path.isdir(checkpoint_path) else model_name
    return os.path.join("./results", model_name, f"checkpoint-{sft_checkpoint}")


def load_dataset_path(path, split=None):
    if split and os.path.isdir(os.path.join(path, split)):
        return load_from_disk(os.path.join(path, split))
    return load_from_disk(path)


class SaveTimeValidationTopKCallback(TrainerCallback):
    def __init__(self, top_k, metric_name, greater_is_better=True):
        self.trainer = None
        self.top_k = int(top_k)
        self.metric_name = metric_name
        self.greater_is_better = bool(greater_is_better)
        self._is_evaluating = False

    def set_trainer(self, trainer):
        self.trainer = trainer
        return self

    def _state_path(self, output_dir):
        return os.path.join(output_dir, "topk_checkpoints.json")

    def _load_state(self, output_dir):
        try:
            with open(self._state_path(output_dir)) as f:
                state = json.load(f)
        except FileNotFoundError:
            return []
        return state if isinstance(state, list) else []

    def _save_state(self, output_dir, state):
        with open(self._state_path(output_dir), "w") as f:
            json.dump(state, f, indent=2, sort_keys=True)

    def _metric_value(self, metrics):
        metric_name = self.metric_name
        if metric_name not in metrics and not metric_name.startswith("eval_"):
            metric_name = f"eval_{metric_name}"
        value = metrics.get(metric_name)
        if value is None:
            return metric_name, None
        try:
            return metric_name, float(value)
        except (TypeError, ValueError):
            return metric_name, None

    def on_save(self, args, state, control, **kwargs):
        if self.trainer is None or self._is_evaluating or state.global_step <= 0:
            return control

        self._is_evaluating = True
        try:
            metrics = self.trainer.evaluate(metric_key_prefix="eval")
        finally:
            self._is_evaluating = False

        if self.top_k <= 0:
            return control

        metric_name, metric_value = self._metric_value(metrics)
        if metric_value is None or not math.isfinite(metric_value):
            if self.trainer.is_world_process_zero():
                print(f"[topk] Metric {metric_name} missing or non-finite; keeping checkpoint unranked.")
            return control

        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        self.trainer.accelerator.wait_for_everyone()
        if not self.trainer.is_world_process_zero() or not os.path.isdir(checkpoint_dir):
            return control

        topk_state = [
            entry for entry in self._load_state(args.output_dir)
            if int(entry.get("step", -1)) != int(state.global_step)
        ]
        topk_state.append(
            {
                "step": int(state.global_step),
                "metric": metric_name,
                "value": metric_value,
                "path": checkpoint_dir,
            }
        )
        topk_state.sort(key=lambda entry: float(entry["value"]), reverse=self.greater_is_better)
        keep = topk_state[: self.top_k]
        drop = topk_state[self.top_k :]
        keep_paths = {entry["path"] for entry in keep}

        for entry in drop:
            path = entry.get("path")
            if path and path not in keep_paths and os.path.isdir(path):
                shutil.rmtree(path)
                print(f"[topk] Removed checkpoint outside top-{self.top_k}: {path}")

        self._save_state(args.output_dir, keep)
        print(f"[topk] Kept top-{self.top_k} checkpoints by {metric_name}: {keep}")
        return control


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train Rank-GRPO with user feedback as verifiable reward.\n\n"
            "This script post-trains an SFT model using offline reinforcement learning (GRPO), "
            "where rank-level rewards are derived from user preference data."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    core = parser.add_argument_group("Core setup")
    core.add_argument(
        "--train_path",
        required=True,
        help=(
            "Path to the training dataset directory (load_from_disk). "
            "Only the training split is loaded — validation is performed via saved checkpoints."
        ),
    )
    core.add_argument(
        "--val_path",
        default=None,
        help=(
            "Path to the validation dataset directory. Accepts either a split directory or a "
            "dataset directory containing a validation split."
        ),
    )
    core.add_argument(
        "--val_max_samples",
        type=int,
        default=None,
        help="Maximum number of validation examples to evaluate on each checkpoint save.",
    )
    core.add_argument(
        "--val_shuffle",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to shuffle validation examples in the eval sampler.",
    )
    core.add_argument(
        "--model_name",
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Base model name or path (Hugging Face format).",
    )
    core.add_argument(
        "--sft_checkpoint",
        type=int,
        default=1500,
        help="Checkpoint ID for the supervised fine-tuned (SFT) reference policy.",
    )
    core.add_argument(
        "--reward_func",
        default="exp_inf",
        choices=["exp_inf", "log_decay"],
        help=(
            "Reward shaping function:\n"
            " - exp_inf: per-rank independent exponential/indicator reward.\n"
            " - log_decay: rank-weighted logarithmic decay reward."
        ),
    )
    core.add_argument(
        "--catalog_path",
        default="gt_catalog.pkl",
        help="Path to the ground-truth catalog pickle (used for hit/match validation).",
    )

    opt = parser.add_argument_group("Optimization hyperparameters")
    opt.add_argument("--lr", type=float, default=1e-6, help="Initial learning rate for policy update.")
    opt.add_argument("--adam_beta1", type=float, default=0.9, help="Adam optimizer β₁ parameter.")
    opt.add_argument("--adam_beta2", type=float, default=0.99, help="Adam optimizer β₂ parameter.")
    opt.add_argument("--kl_beta", type=float, default=1e-3, help="KL-divergence penalty coefficient.")
    opt.add_argument("--gradient_accumulation_steps", type=int, default=12, help="Gradient accumulation steps.")
    opt.add_argument("--optim", default="paged_adamw_8bit", help="Optimizer type (paged_adamw_8bit recommended).")

    sched = parser.add_argument_group("Training schedule")
    sched.add_argument("--per_device_train_batch_size", type=int, default=8, help="Training batch size per GPU.")
    sched.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Evaluation batch size per GPU.")
    sched.add_argument("--num_train_epochs", type=int, default=2, help="Number of training epochs.")
    sched.add_argument("--mu", type=int, default=1, help="Number of GRPO iterations (μ=1 = strictly on-policy).")

    gen = parser.add_argument_group("Model / generation parameters")
    gen.add_argument("--max_prompt_length", type=int, default=2048, help="Maximum prompt token length.")
    gen.add_argument("--max_completion_length", type=int, default=1024, help="Maximum generation token length.")
    gen.add_argument("--num_generations", type=int, default=8, help="Number of sampled completions per prompt.")
    gen.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing.")

    vllm = parser.add_argument_group("vLLM inference backend")
    vllm.add_argument("--use_vllm", action="store_true", help="Use vLLM as the inference backend.")
    vllm.add_argument("--vllm_mode", default="colocate", help="vLLM deployment mode: colocate or separate.")
    vllm.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.5,
        help="Fraction of GPU memory allocated to vLLM inference.",
    )

    ### For Qwen2.5-0.5B, please modify this to 2
    vllm.add_argument("--vllm_tensor_parallel_size", type=int, default=4, help="vLLM tensor parallel size.")

    log = parser.add_argument_group("Logging / checkpointing")
    log.add_argument(
        "--report_to",
        default="tensorboard",
        choices=["tensorboard", "wandb", "both", "none"],
        help="Logging backend. Use 'both' to log to TensorBoard and W&B.",
    )
    log.add_argument("--wandb_project", default="rank_grpo", help="Weights & Biases project name.")
    log.add_argument("--logging_steps", type=int, default=10, help="Logging frequency in steps.")
    log.add_argument("--save_strategy", default="steps", help="Checkpoint save strategy: steps or epoch.")
    log.add_argument("--save_steps", type=int, default=200, help="Save model every N steps.")
    log.add_argument("--topk_checkpoints", type=int, default=3, help="Keep top-K checkpoints by validation metric.")
    log.add_argument("--topk_metric", default="eval_reward_total", help="Validation metric used for top-K checkpointing.")
    log.add_argument("--topk_greater_is_better", action=argparse.BooleanOptionalAction, default=True)
    log.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint.")

    misc = parser.add_argument_group("Miscellaneous")
    misc.add_argument("--seed", type=int, default=3407, help="Random seed for reproducibility.")
    misc.add_argument("--bf16", action="store_true", help="Enable bfloat16 precision.")
    misc.add_argument("--verbose", action="store_true", help="Enable verbose LR scheduler output.")

    return parser.parse_args()


def main():
    args = parse_args()
    report_to = normalize_report_to(args.report_to)
    accelerator = setup_environment(args.wandb_project, report_to)

    # Load datasets and catalog
    gt_catalog = load_catalog(args.catalog_path)
    train_dataset = load_dataset_path(args.train_path, split="train")
    val_dataset = load_dataset_path(args.val_path, split="validation") if args.val_path else None
    if val_dataset is not None and args.val_max_samples is not None:
        if args.val_max_samples <= 0:
            raise ValueError("--val_max_samples must be a positive integer.")
        val_max_samples = min(args.val_max_samples, len(val_dataset))
        val_dataset = val_dataset.select(range(val_max_samples))
        accelerator.print(f"Using {val_max_samples} validation samples.")

    # Reward function selection
    if args.reward_func == "exp_inf":
        reward_func = make_reward_func_individual(rec_num=20, gt_catalog=gt_catalog)
    elif args.reward_func == "log_decay":
        reward_func = make_reward_func(rec_num=20, gt_catalog=gt_catalog)
    else:
        raise ValueError(f"{args.reward_func} not implemented!")

    # Define model paths and W&B run
    sft_model_path = resolve_sft_model_path(args.model_name, args.sft_checkpoint)
    output_dir = os.path.join("./results/grpo", f"{_safe_name(args.model_name)}_lr{args.lr}_kl{args.kl_beta}_mu{args.mu}")
    run_name = setup_run(
        accelerator,
        output_dir,
        args.model_name,
        args.sft_checkpoint,
        args.seed,
        args.wandb_project,
        report_to,
    )

    # ---------------- Learning rate scheduler ----------------
    # According to the paper setup: constant LR for small model, decay for larger models.
    if "0.5B" in args.model_name:
        schedule = [(8000, 1e-6)]  # constant for Qwen2.5-0.5B
    else:
        schedule = [(8000, 1e-7)]  # decay for Llama-3.2-1B/3B
    callback = StepLRSchedulerCallback(schedule=schedule, verbose=args.verbose)

    # ---------------- GRPO Configuration ----------------
    config = GRPOConfig(
        importance_sampling_level="item",
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        beta=args.kl_beta,
        epsilon=0.06,
        epsilon_high=0.08,
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        num_iterations=args.mu,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        logging_dir=os.path.join(output_dir, "tensorboard"),
        report_to=report_to,
        bf16=args.bf16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        optim=args.optim,
        lr_scheduler_type="constant",
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        seed=args.seed,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        gradient_checkpointing=args.gradient_checkpointing,
        run_name=run_name,
    )
    config.val_shuffle = args.val_shuffle

    # ---------------- Trainer ----------------
    trainer = RankGRPOTrainer(
        model=sft_model_path,
        reward_funcs=reward_func,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[callback],
    )
    if val_dataset is not None:
        trainer.add_callback(
            SaveTimeValidationTopKCallback(
                top_k=args.topk_checkpoints,
                metric_name=args.topk_metric,
                greater_is_better=args.topk_greater_is_better,
            ).set_trainer(trainer)
        )

    accelerator.print("🚀 Training …")
    trainer.train(resume_from_checkpoint=args.resume)
    accelerator.print("✅ Done")


if __name__ == "__main__":
    main()