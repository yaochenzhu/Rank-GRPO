
# Rank-GRPO: Training LLM-based Conversational Recommender Systems with Reinforcement Learning

  

This repository contains the full training, alignment, and evaluation pipeline for **ConvRec-R1**, including both **Supervised Fine-Tuning (SFT)** and **Rank-GRPO** stages.  

> Rank-GRPO: Training LLM-based Conversational Recommender Systems with Reinforcement Learning       
>**Yaochen Zhu**, Harald Steck, Dawen Liang, Yinhan He, Vito Ostuni, Jundong Li, Nathan Kallus     

### **Update!** ###

We have released the SFT checkpoint and on-policy Rank-GRPO (exp_inf) checkpoints for Qwen2.5-0.5B-Instruct and Llama3.2-3B-Instruct as well as the full trainer state!

## ⚙️ Environment Setup

```bash

# Create a dedicated conda environment
conda create -n rank-grpo python==3.10.0

# Activate the environment
conda  activate  rank-grpo

# Install core dependencies
uv pip install trl[vllm]==0.21.0
uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
uv pip install vllm==0.10.0
uv pip install deepspeed bitsandbytes

# Install Rank-GRPO to trl
bash INSTALL.SH
```


## 🚀 Training Pipeline Overview

### **SFT Stage**

This stage fine-tunes a pretrained model (e.g., `Qwen2.5-0.5B-Instruct`) on conversational recommendation data.

Adjust the `configs/qwen25_0.5b_sft.yaml` file to ensure the **effective batch size** matches your GPU configuration.

#### Example Command

```bash

accelerate launch --config_file configs/qwen25_0.5b_sft.yaml  train_sft.py \
--dataset_path ./processed_datasets/sft_dataset  \
--model_name qwen/Qwen2.5-0.5B-Instruct \
--num_train_epochs 10  \
--learning_rate 5e-5 \
--warmup_ratio 0.05  \
--per_device_train_batch_size 12 \
--per_device_eval_batch_size 12  \
--gradient_accumulation_steps 8 \
--save_strategy  steps  \
--save_steps 100 \
--eval_strategy steps  \
--eval_steps 10 \
--logging_steps 10  \
2>&1 | tee logs/outputs_qwen25_0.5b_sft.txt
```

### **Rank-GRPO RL Alignment**

This stage aligns the SFT model using **offline RL with verifiable rewards** (Rank-GRPO). You can modify the config (`configs/sft_test_qwen25_0.5b.yaml`) to match your available hardware.

#### Example Command
```bash
accelerate launch --config_file  configs/sft_test_qwen25_0.5b.yaml  \
grpo_test_qwen25_0.5b_with_kl.py \
--train_path processed_datasets/grpo/grpo_dataset  \
--model_name qwen/Qwen2.5-0.5B-Instruct \
--sft_checkpoint 1500 \
--reward_func exp_inf \
--mu 1 \
--lr 1e-6 \
--kl_beta 1e-3 \
--adam_beta1 0.9 \
--adam_beta2 0.99 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--num_train_epochs 2 \
--gradient_accumulation_steps 6 \
--save_strategy steps \
--save_steps 200 \
--logging_steps 10 \
--use_vllm \
--vllm_mode colocate \
--vllm_gpu_memory_utilization 0.5 \
--vllm_tensor_parallel_size 4 \
--max_prompt_length 2048 \
--max_completion_length 1024 \
--num_generations 8 \
--seed 3407 \
--wandb_project rank_grpo \
--catalog_path gt_catalog.pkl \
2>&1 | tee logs/outputs_qwen25_0.5b_zero.txt
```

## 📚 Datasets

We release a **processed Reddit-v2** dataset prepared for both **SFT** and **Rank-GRPO**:

- **Download:** [Reddit-v2 (processed)](https://drive.google.com/file/d/11tOfUMlVOylkkcnwPqGM_0IuiIeHjLle/view?usp=sharing)

### Time-based splits (following the original paper)
| Split       | Time window    |
|-------------|-----------------|
| **Train**   | ≤ **2022-10**   |
| **Val**     | **2022-11**     |
| **Test**    | **2022-12**     |

As in prior work, **recommendations from different Reddit users for the same query (which are used as groundtruths) are treated as different samples**, rather than aggregating them into a single sample.

## 📦 Checkpoints & Trainer State

We provide **SFT model weights**, **Rank-GRPO (exp_inf) policy weights**, and the **full trainer state** to reproduce or continue experiments.

### Downloads

**Supervised Fine-Tuning (SFT)**
- Qwen2.5-0.5B-Instruct — [checkpoint](https://drive.google.com/file/d/1Br-CMrzgUzJ7_fwFVahCqzJ8TyGZ5Zfz/view?usp=drive_link)
- Llama3.2-3B-Instruct — [checkpoint](https://drive.google.com/file/d/1GxVs51Ry78cQGGVrxwTWUinzQUrH7xuK/view?usp=sharing)

**Rank-GRPO (exp_inf)**
- Qwen2.5-0.5B-Instruct — [policy checkpoint](https://drive.google.com/file/d/1VNSkbXGpawfj1amRMhfoiPQHsOKcYJzA/view?usp=sharing)
- Llama3.2-3B-Instruct — [policy checkpoint](https://drive.google.com/file/d/1y2yqcsru-_jxbfWEyRBbkcrt5wZ8RFsD/view?usp=sharing)


## 📊 Evaluation

Evaluation is divided into **validation** (training dynamics) and **test** (final model quality).

### **SFT Evaluation**


Plot training loss and validation metrics (NDCG / Recall).

```bash
python eval_sft.py \
--model_name Qwen/Qwen2.5-0.5B-Instruct \
--model_root ../results/Qwen/Qwen2.5-0.5B-Instruct  \
--dataset_path ../processed_datasets/sft_dataset \
--catalog_path ../processed_datasets/gt_catalog.pkl  \
--wandb_project sft_eval_val \
--upload_wandb
```
  
### **GRPO Validation Evaluation**

Track reward curves and validation metrics over training checkpoints.

```bash
python eval_grpo_val.py \
--model_name Qwen/Qwen2.5-0.5B-Instruct \
--model_root ../results/grpo/Qwen/Qwen2.5-0.5B-Instruct_lr1e-06_kl0.001 \
--dataset_path ../processed_datasets/sft_dataset \
--catalog_path ../processed_datasets/gt_catalog.pkl  \
--wandb_project grpo_eval_val \
--upload_wandb \
--step_interval 200
```

### **Final Evaluation on Test Set**

Evaluate the final checkpoint on the **held-out test set** using vLLM inference and compute top-K Recall / NDCG.

```bash
python grpo_test.py \
--model_name Qwen/Qwen2.5-0.5B-Instruct \
--model_root ../results/grpo/Qwen/Qwen2.5-0.5B-Instruct_lr1e-06_kl0.001 \
--dataset_path ../processed_datasets/sft_dataset \
--catalog_path ../processed_datasets/gt_catalog.pkl  \
--checkpoint 15800 \
2>&1 | tee logs/eval_grpo_test_qwen25_0.5b.txt
```


## 🌟 Citation
If you find this work is helpful to your research, please consider citing our paper:
```
@inproceedings{zhu2025rankgrpo,
  title={Rank-GRPO: Training LLM-based Conversational Recommender Systems with Reinforcement Learning},
  author={Zhu, Yaochen and Steck, Harald and Liang, Dawen and He, Yinhan, Ostuni, Vito and Li, Jundong and Kallus, Nathan},
  booktitle={ArXiv},
  year={2025}
}
```
**Thanks for your interest in our work!**
