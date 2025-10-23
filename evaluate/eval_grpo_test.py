"""
grpo_test.py — Generalized GRPO Test Evaluation Script (vLLM-based)

Evaluates a GRPO-aligned model on the test set:
  • Loads the dataset and catalog
  • Runs vLLM inference on all test prompts
  • Processes model generations
  • Computes Recall@K and NDCG@K metrics
  • Saves results and optionally uploads to Weights & Biases
"""

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
        description="Evaluate GRPO-aligned models on test set using vLLM inference."
    )
    parser.add_argument("--model_name", required=True, help="Base model or checkpoint name.")
    parser.add_argument("--model_root", required=True, help="Path to the GRPO training directory.")
    parser.add_argument("--dataset_path", required=True, help="Dataset directory (load_from_disk).")
    parser.add_argument("--catalog_path", required=True, help="Ground-truth catalog pickle path.")
    parser.add_argument("--checkpoint", type=int, default=None, help="Checkpoint step to evaluate (default = latest).")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum generation length.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8, help="GPU memory fraction for vLLM.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallelism size.")
    parser.add_argument("--save_dir", default="results_grpo_test", help="Output directory for results.")
    return parser.parse_args()


def main():
    args = parse_args()

    # ---------- Load Dataset ----------
    print(f"📂 Loading test dataset from {args.dataset_path} ...")
    val_dataset = load_from_disk(os.path.join(args.dataset_path, "test"))

    context_to_indices = defaultdict(list)
    for i, item in enumerate(val_dataset):
        context = item["prompt"][0]["content"]
        context_to_indices[context].append(i)
    all_contexts = list(context_to_indices.keys())
    all_input_texts = [[{"role": "user", "content": ctx}] for ctx in all_contexts]
    print(f"Loaded {len(all_input_texts)} unique test contexts.")

    # ---------- Load Catalog ----------
    with open(args.catalog_path, "rb") as f:
        gt_catalog = set(pickle.load(f))
    print(f"Catalog size: {len(gt_catalog)}")

    # ---------- Determine Checkpoint ----------
    if args.checkpoint is not None:
        ckpt_path = os.path.join(args.model_root, f"checkpoint-{args.checkpoint}")
    else:
        ckpt_path = find_latest_checkpoint(args.model_root)
    print(f"Evaluating checkpoint: {ckpt_path}")

    # ---------- Prepare Model ----------
    sampling_params = SamplingParams(temperature=0, top_p=0.9, max_tokens=args.max_tokens)
    print("🧠 Initializing vLLM...")
    llm = LLM(
        model=ckpt_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=8192,
    )
    print("✅ Model loaded successfully.")

    # ---------- Generate Recommendations ----------
    print("💬 Generating recommendations ...")
    llm_outputs = llm.chat(all_input_texts, sampling_params)
    del llm

    test_data_with_rec = [item for item in val_dataset]
    for ctx, out in zip(all_contexts, llm_outputs):
        rec_text = out.outputs[0].text
        for idx in context_to_indices[ctx]:
            test_data_with_rec[idx]["raw_rec"] = rec_text

    # ---------- Postprocess ----------
    print("🔧 Processing generated outputs ...")
    test_data_with_rec = [process_rec_raw(item, "raw_rec", "rec") for item in test_data_with_rec]
    errors = [item[0] for item in test_data_with_rec if item[0]]
    test_data_with_rec = [item[1] for item in test_data_with_rec]
    print(f"# parse errors: {len(errors)}")

    # ---------- Compute Metrics ----------
    print("📊 Computing Recall and NDCG metrics ...")
    k_list = [5, 10, 15, 20]
    recalls, ndcgs = {}, {}
    for k in k_list:
        recall_k, ndcg_k = [], []
        print(f"Top-{k}")
        for item in tqdm(test_data_with_rec, total=len(test_data_with_rec)):
            r, n = evaluate_direct_match_truncate(
                item,
                k,
                seen_field="seen_titles",
                rec_field="rec",
                gt_field="groundtruth_with_release_year",
                gt_catalog=gt_catalog,
            )
            recall_k.append(r)
            ndcg_k.append(n)
        recalls[k] = recall_k
        ndcgs[k] = ndcg_k

    avg_recalls = {k: np.mean(recalls[k]) for k in k_list}
    avg_ndcgs = {k: np.mean(ndcgs[k]) for k in k_list}

    # ---------- Display Results ----------
    print("\n✅ Final Average Metrics:")
    for k in k_list:
        print(f"  Top-{k}: Recall={avg_recalls[k]:.4f}, NDCG={avg_ndcgs[k]:.4f}")

    # ---------- Save Results ----------
    print("\n💾 Saving results ...")
    os.makedirs(args.save_dir, exist_ok=True)
    output_pkl = os.path.join(args.save_dir, f"{args.model_name.replace('/', '_')}_test.pkl")
    with open(output_pkl, "wb") as f:
        pickle.dump(test_data_with_rec, f)
    save_analysis_state(args.save_dir, [], {args.model_name: (avg_recalls, avg_ndcgs)})

    print(f"\n🎯 Evaluation complete for {args.model_name}")
    print(f"Results saved in: {args.save_dir}")


if __name__ == "__main__":
    main()
