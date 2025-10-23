import os
import re
import json
from collections import defaultdict

import seaborn as sns
import matplotlib.pyplot as plt


def find_latest_checkpoint(model_dir: str) -> str:
    checkpoint_dirs = [d for d in os.listdir(model_dir) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(model_dir, d))]
    if not checkpoint_dirs:
        print(f"Error: No checkpoint directories found in {model_dir}")
        return None

    # Extract step numbers and find the max
    max_step = -1
    latest_checkpoint_dir = None
    for directory in checkpoint_dirs:
        match = re.search(r"checkpoint-(\d+)", directory)
        if match:
            step = int(match.group(1))
            if step > max_step:
                max_step = step
                latest_checkpoint_dir = directory
    
    if latest_checkpoint_dir:
        return os.path.join(model_dir, latest_checkpoint_dir)
    else:
        print(f"Error: Could not determine the latest checkpoint in {model_dir}")
        return None

    
def parse_log_history(trainer_state_path: str) -> tuple:
    try:
        with open(trainer_state_path, 'r') as f:
            state = json.load(f)
    except FileNotFoundError:
        print(f"Error: trainer_state.json not found at {trainer_state_path}")
        return [], [], [], []

    log_history = state.get("log_history", [])
    if not log_history:
        print("Warning: Log history is empty.")
        return [], [], [], []

    train_steps, train_losses = [], []
    eval_steps, eval_losses = [], []

    for entry in log_history:
        step = entry.get("step")
        if step is None:
            continue
        
        # Training loss is logged under the 'loss' key
        if 'loss' in entry:
            train_steps.append(step)
            train_losses.append(entry['loss'])
        
        # Evaluation loss is logged under the 'eval_loss' key
        if 'eval_loss' in entry:
            eval_steps.append(step)
            eval_losses.append(entry['eval_loss'])

    return train_steps, train_losses, eval_steps, eval_losses


def plot_losses(train_steps, train_losses, eval_steps, eval_losses, model_name, output_dir):
    if not train_steps and not eval_steps:
        print("No data to plot.")
        return

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 7))

    # Plot training loss
    if train_steps:
        plt.plot(train_steps, train_losses, label="Training Loss", color="dodgerblue", alpha=0.8, linewidth=2)

    # Plot evaluation loss
    if eval_steps:
        plt.plot(eval_steps, eval_losses, label="Validation Loss", color="darkorange", marker='o', linestyle='--', markersize=5)

    plt.title(f"Training and Validation Loss for\n{model_name}", fontsize=16)
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # Save the plot
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plot_path = os.path.join(output_dir, f"{model_name.replace('/','-')}_loss_curve.png")
    plt.savefig(plot_path, dpi=300)
    print(f"✅ Plot saved to {plot_path}")
    plt.show()