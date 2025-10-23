import os
import sys
import json
import pickle
import numpy as np
import argparse
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt

from vllm import LLM, SamplingParams
from datasets import load_from_disk

# Local libs
from libs.analyze import find_latest_checkpoint, parse_log_history, plot_losses
from libs.save_eval_state import save_analysis_state
from libs.log_wandb import merge_and_upload

# loading from the global libs for evaluation
sys.path.insert(0, "../../libs")
from utils import process_rec_raw
from metrics import evaluate_direct_match_truncate

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SFT training and outputs.")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--model_root", default="../results/Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dataset_path", default="../processed_datasets/sft_dataset")
    parser.add_argument("--catalog_path", default="../gt_catalog.pkl")
    parser.add_argument("--output_dir", default="figs", help="Directory to save analysis figures.")
    parser.add_argument("--wandb_project", default="sft_eval_val", help="Weights & Biases project name.")
    parser.add_argument("--use_multiprocessing", action="store_true", help="Use multiprocessing for evaluation.")
    parser.add_argument("--upload_wandb", action="store_true", help="Upload results to Weights & Biases.")
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = args.dataset_path

    print(f"📂 Loading validation dataset from {data_root} ...")
    val_dataset = load_from_disk(os.path.join(data_root, "validation"))

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
    latest_checkpoint_path = find_latest_checkpoint(model_root)
    print(f"Latest checkpoint: {latest_checkpoint_path}")
    trainer_state_path = os.path.join(latest_checkpoint_path, "trainer_state.json")

    print("Parsing training log history ...")
    train_steps, train_losses, eval_steps, eval_losses = parse_log_history(trainer_state_path)
    plot_losses(train_steps, train_losses, eval_steps, eval_losses, args.model_name, args.output_dir)

    last_step = int(latest_checkpoint_path.split("-")[-1])
    step_list = list(range(0, last_step, 200))
    llm_outputs = {}
    test_data_with_rec = [item for item in val_dataset]

    # --- Step 4: Model inference ---
    print("🧠 Generating recommendations across checkpoints ...")
    model_path_tmpl = os.path.join(model_root, "checkpoint-{}")

    for step in step_list:
        if step in llm_outputs:
            continue
        print(f"Processing step {step} ...")
        model_to_load = args.model_name if step == 0 else model_path_tmpl.format(step)
        llm = LLM(model=model_to_load,
                  tensor_parallel_size=1,
                  gpu_memory_utilization=0.8,
                  max_model_len=8192)
        sampling_params = SamplingParams(temperature=0, top_p=0.9, max_tokens=1024)
        llm_outputs[step] = llm.chat(all_input_texts, sampling_params)
        del llm

    # --- Step 5: Assign outputs to dataset ---
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
        test_data_with_rec = [process_rec_raw(item, field, f"rec_after_step_{step}")[1] for item in test_data_with_rec]
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

    # --- Step 7: Recall / NDCG evaluation ---
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
            {k: np.mean(ndcgs[k]) for k in k_list}
        )

    # --- Step 8: Save results ---
    print("💾 Saving evaluation results ...")
    analysis_json = save_analysis_state(model_root, ratios, avg_metrics)
    with open(os.path.join(model_root, "output.pkl"), "wb") as f:
        pickle.dump(test_data_with_rec, f)

    if args.upload_wandb:
        merge_and_upload(
            model_dir=model_root,
            project=args.wandb_project,
            run_name_suffix="sft_eval",
            merged_filename="trainer_plus_analysis.json",
            upload=True
        )
    print(f"✅ Evaluation complete. Results saved to {model_root}")

if __name__ == "__main__":
    main()
