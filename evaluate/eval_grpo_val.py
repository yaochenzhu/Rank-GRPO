import os
import sys
import json
import pickle
import numpy as np
import argparse
from tqdm import tqdm
from collections import defaultdict
from vllm import LLM, SamplingParams
from datasets import load_from_disk

# --- GRPO-specific analysis utilities ---
from libs.analyze_grpo import find_latest_checkpoint, parse_log_history, plot_losses
from libs.save_eval_state import save_analysis_state
from libs.log_wandb import merge_and_upload

# --- global libs for evaluation ---
sys.path.insert(0, "../../libs")
from utils import process_rec_raw
from metrics import evaluate_direct_match_truncate


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate GRPO-aligned models using vLLM inference on validation set."
    )
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Base model or checkpoint name.")
    parser.add_argument("--model_root", required=True,
                        help="Path to the GRPO training results directory containing checkpoints.")
    parser.add_argument("--dataset_path", default="../processed_datasets/sft_dataset",
                        help="Validation dataset path (load_from_disk).") # validation and test data are stored in sft folder
    parser.add_argument("--catalog_path", default="../processed_datasets/gt_catalog.pkl",
                        help="Ground-truth catalog pickle file.")
    parser.add_argument("--output_dir", default="figs", help="Directory to save plots and analysis.")
    parser.add_argument("--wandb_project", default="grpo_eval_val",
                        help="Weights & Biases project name for upload.")
    parser.add_argument("--upload_wandb", action="store_true",
                        help="Upload merged results to Weights & Biases.")
    parser.add_argument("--max_tokens", type=int, default=1024,
                        help="Maximum generation length for vLLM inference.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8,
                        help="Fraction of GPU memory to allocate for vLLM.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of tensor-parallel GPUs for vLLM.")
    parser.add_argument("--step_interval", type=int, default=200,
                        help="Interval between evaluated checkpoints (in training steps).")
    return parser.parse_args()

def main():
    args = parse_args()

    print(f"📂 Loading validation dataset from {args.dataset_path} ...")
    val_dataset = load_from_disk(os.path.join(args.dataset_path, "validation"))

    # --- Step 1: Load catalog ---
    with open(args.catalog_path, "rb") as f:
        gt_catalog = set(pickle.load(f))
    print(f"Loaded catalog of size: {len(gt_catalog)}")

    # --- Step 2: Group contexts ---
    context_to_indices = defaultdict(list)
    for i, item in enumerate(val_dataset):
        context = item["prompt"][0]["content"]
        context_to_indices[context].append(i)
    all_contexts = list(context_to_indices.keys())
    all_input_texts = [[{"role": "user", "content": ctx}] for ctx in all_contexts]
    print(f"Total unique contexts: {len(all_input_texts)}")

    # --- Step 3: Find latest checkpoint & parse logs ---
    model_root = args.model_root
    latest_ckpt = find_latest_checkpoint(model_root)
    print(f"Latest checkpoint: {latest_ckpt}")
    trainer_state_path = os.path.join(latest_ckpt, "trainer_state.json")

    print("Parsing GRPO training log history ...")
    train_steps, train_rewards, eval_steps, eval_rewards = parse_log_history(trainer_state_path)
    plot_losses(train_steps, train_rewards, eval_steps, eval_rewards,
                args.model_name, args.output_dir)

    last_step = int(latest_ckpt.split("-")[-1])
    step_list = list(range(0, last_step + args.step_interval, args.step_interval))
    llm_outputs = {}
    test_data_with_rec = [item for item in val_dataset]

    # --- Step 4: vLLM inference across checkpoints ---
    print("🧠 Running vLLM inference across checkpoints ...")
    model_path_tmpl = os.path.join(model_root, "checkpoint-{}")

    for step in step_list:
        if step in llm_outputs:
            continue
        print(f"Processing checkpoint {step} ...")
        model_to_load = args.model_name if step == 0 else model_path_tmpl.format(step)

        llm = LLM(
            model=model_to_load,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=8192,
        )
        sampling_params = SamplingParams(
            temperature=0, top_p=0.9, max_tokens=args.max_tokens
        )
        llm_outputs[step] = llm.chat(all_input_texts, sampling_params)
        del llm  # free GPU memory

    # --- Step 5: Attach outputs to dataset ---
    for step in step_list:
        field = f"raw_rec_after_step_{step}"
        context_to_rec = {ctx: out.outputs[0].text for ctx, out in zip(all_contexts, llm_outputs[step])}
        for ctx, indices in context_to_indices.items():
            for idx in indices:
                test_data_with_rec[idx][field] = context_to_rec[ctx]

    # --- Step 6: Postprocess & compute hit ratios ---
    print("🔍 Evaluating catalog match ratios ...")
    ratios = []
    gt_index = defaultdict(list)
    for name, year in gt_catalog:
        gt_index[name.lower()].append(year)

    for step in step_list:
        field = f"raw_rec_after_step_{step}"
        test_data_with_rec = [process_rec_raw(item, field, f"rec_after_step_{step}")[1]
                              for item in test_data_with_rec]
        rec_field = f"rec_after_step_{step}"

        step_ratios = []
        for item in tqdm(test_data_with_rec, desc=f"Evaluating step {step}"):
            recs = item.get(rec_field, [])[:20]
            matches = 0
            for movie, year in recs:
                years = gt_index.get(movie.lower(), [])
                if any(abs(year - y) <= 2 for y in years):
                    matches += 1
            step_ratios.append(matches / len(recs) if recs else 0)
        ratios.append((step, np.mean(step_ratios), np.std(step_ratios)))

    # --- Step 7: Compute Recall / NDCG ---
    print("📊 Calculating Recall and NDCG ...")
    k_list = [5, 10, 15, 20]
    metrics, avg_metrics = {}, {}

    for step in step_list:
        rec_field = f"rec_after_step_{step}"
        recalls, ndcgs = {}, {}
        for k in k_list:
            recall_k, ndcg_k = [], []
            for item in tqdm(test_data_with_rec, desc=f"Step {step} Top-{k}"):
                r, n = evaluate_direct_match_truncate(
                    item, k,
                    seen_field="seen_titles",
                    rec_field=rec_field,
                    gt_field="groundtruth_with_release_year",
                    gt_catalog=gt_catalog
                )
                recall_k.append(r)
                ndcg_k.append(n)
            recalls[k] = recall_k
            ndcgs[k] = ndcg_k
        metrics[step] = (recalls, ndcgs)
        avg_metrics[step] = (
            {k: np.mean(recalls[k]) for k in k_list},
            {k: np.mean(ndcgs[k]) for k in k_list},
        )

    # --- Step 8: Save results ---
    print("💾 Saving evaluation results ...")
    analysis_json = save_analysis_state(model_root, ratios, avg_metrics)
    with open(os.path.join(model_root, "output_val.pkl"), "wb") as f:
        pickle.dump(test_data_with_rec, f)

    if args.upload_wandb:
        merge_and_upload(
            model_dir=model_root,
            project=args.wandb_project,
            run_name_suffix="grpo_eval",
            merged_filename="trainer_plus_analysis.json",
            upload=True,
        )

    print(f"✅ GRPO Evaluation complete. Results saved to {model_root}")
    
if __name__ == "__main__":
    main()
