import os
import sys
import argparse
from datasets import load_from_disk
from accelerate import Accelerator
from trl import SFTConfig, SFTTrainer

sys.path.append("libs")
from utils import process_rec_raw
from metrics import evaluate_direct_match_truncate


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train Supervised Fine-Tuning (SFT) model for Rank-GRPO.\n\n"
            "This stage grounds the base model to the recommendation catalog "
            "and prepares the reference policy for subsequent GRPO training."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    core = parser.add_argument_group("Core setup")
    core.add_argument(
        "--dataset_path",
        required=True,
        help="Path to the dataset directory (load_from_disk). Must contain 'train' and 'validation' splits.",
    )
    core.add_argument(
        "--model_name",
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Base model name or path from Hugging Face Hub.",
    )

    opt = parser.add_argument_group("Optimization hyperparameters")
    opt.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate.")
    opt.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup ratio for LR scheduler.")
    opt.add_argument("--optim", default="paged_adamw_8bit", help="Optimizer type.")
    opt.add_argument("--lr_scheduler_type", default="cosine", help="Type of learning rate scheduler.")

    sched = parser.add_argument_group("Training schedule")
    sched.add_argument("--num_train_epochs", type=int, default=10, help="Number of full training epochs.")
    sched.add_argument("--per_device_train_batch_size", type=int, default=12, help="Batch size per device for training.")
    sched.add_argument("--per_device_eval_batch_size", type=int, default=12, help="Batch size per device for evaluation.")
    sched.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Number of gradient accumulation steps.")
    sched.add_argument("--dataset_num_proc", type=int, default=64, help="Number of processes for dataset preprocessing.")
    sched.add_argument("--max_length", type=int, default=1024, help="Maximum input sequence length.")

    log = parser.add_argument_group("Checkpointing / logging")
    log.add_argument("--save_strategy", default="steps", help="Save strategy for checkpoints.")
    log.add_argument("--save_steps", type=int, default=50, help="Number of steps between checkpoints.")
    log.add_argument("--logging_steps", type=int, default=10, help="Number of steps between log outputs.")
    log.add_argument("--eval_strategy", default="steps", help="Evaluation strategy (steps or epoch).")
    log.add_argument("--eval_steps", type=int, default=10, help="Number of steps between evaluations.")
    log.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint if available.")

    misc = parser.add_argument_group("Miscellaneous")
    misc.add_argument("--seed", type=int, default=3407, help="Random seed for reproducibility.")
    misc.add_argument("--bf16", action="store_true", help="Enable bfloat16 precision training.")
    misc.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing.")

    return parser.parse_args()


def main():
    args = parse_args()
    accelerator = Accelerator()
    os.environ["WANDB_PROJECT"] = "sft-rec-uva"

    # Load datasets
    accelerator.print(f"📚 Loading datasets from {args.dataset_path}")
    train_dataset = load_from_disk(os.path.join(args.dataset_path, "train"))
    val_dataset = load_from_disk(os.path.join(args.dataset_path, "validation"))

    # SFT configuration
    config = SFTConfig(
        output_dir=f"./results/{args.model_name}",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        bf16=args.bf16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        learning_rate=args.learning_rate,
        optim=args.optim,
        lr_scheduler_type=args.lr_scheduler_type,
        seed=args.seed,
        dataset_num_proc=args.dataset_num_proc,
        max_length=args.max_length,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    # Trainer
    trainer = SFTTrainer(
        model=args.model_name,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    accelerator.print("🚀 Training …")
    trainer.train(resume_from_checkpoint=args.resume)
    accelerator.print("✅ Done")


if __name__ == "__main__":
    main()
