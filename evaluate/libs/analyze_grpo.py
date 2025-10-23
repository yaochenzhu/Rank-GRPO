import os
import re
import json
from collections import defaultdict

import numpy as np
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

def parse_grpo_log_history(trainer_state_path: str) -> tuple:
    try:
        with open(trainer_state_path, 'r') as f:
            state = json.load(f)
    except FileNotFoundError:
        print(f"Error: trainer_state.json not found at {trainer_state_path}")
        return [], [], []

    log_history = state.get("log_history", [])
    if not log_history:
        print("Warning: Log history is empty.")
        return [], [], []

    steps, rewards, reward_stds = [], [], []

    for entry in log_history:
        # Check if the required keys for GRPO logs exist in the log entry
        if 'step' in entry and 'reward' in entry and 'reward_std' in entry:
            steps.append(entry['step'])
            rewards.append(entry['reward'])
            reward_stds.append(entry['reward_std'])

    return steps, rewards, reward_stds

def plot_rewards(steps, rewards, reward_stds, model_name, output_dir):
    if not steps:
        print("No data to plot.")
        return

    # Convert lists to numpy arrays for vectorized operations
    steps_arr = np.array(steps)
    rewards_arr = np.array(rewards)
    reward_stds_arr = np.array(reward_stds)

    # Sort data by step number to ensure the plot is drawn correctly
    sort_indices = np.argsort(steps_arr)
    steps_arr = steps_arr[sort_indices]
    rewards_arr = rewards_arr[sort_indices]
    reward_stds_arr = reward_stds_arr[sort_indices]

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 7))

    # Plot the average reward line
    plt.plot(steps_arr, rewards_arr, label="Average Reward", color="dodgerblue", linewidth=2)

    # Create a shaded region for the standard deviation (reward ± std)
    plt.fill_between(
        steps_arr, 
        rewards_arr - reward_stds_arr, 
        rewards_arr + reward_stds_arr, 
        color="dodgerblue", 
        alpha=0.2, 
        label="Reward Std. Dev."
    )

    plt.title(f"Training Reward for {model_name}", fontsize=16)
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Reward", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(output_dir, f"{model_name.replace('/','-')}_reward_curve.png")
    plt.savefig(plot_path, dpi=300)
    print(f"✅ Plot saved to {plot_path}")
    plt.show()