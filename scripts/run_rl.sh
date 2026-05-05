
# grpo_test_qwen25_0.5b_with_kl.py \
clear
export CUDA_VISIBLE_DEVICES=0,1
export RANK_GRPO_ENV=${RANK_GRPO_ENV:-/home/dyvm6xra/dyvm6xrauser45/miniconda3/envs/rank-grpo}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.2}
export PATH=${RANK_GRPO_ENV}/bin:${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}
export DS_IGNORE_CUDA_DETECTION=1
export VLLM_USE_FLASHINFER_SAMPLER=0

N_GPUS=2

ACCELERATE_BIN=${ACCELERATE_BIN:-${RANK_GRPO_ENV}/bin/accelerate}
DATA_DIR=/home/dyvm6xra/dyvm6xrauser45/fred/local_backup/rankgrpo_data_ckpts
SAVE_STEPS=${SAVE_STEPS:-200}
TOPK_CHECKPOINTS=${TOPK_CHECKPOINTS:-3}
TOPK_METRIC=${TOPK_METRIC:-eval_reward_total}
${ACCELERATE_BIN} launch --config_file configs/qwen25_0.5b_grpo.yaml --num_processes ${N_GPUS} \
/home/dyvm6xra/dyvm6xrauser45/fred/local_backup/Rank-GRPO/train_rank_grpo.py \
--train_path ${DATA_DIR}/processed_datasets/grpo/grpo_dataset  \
--val_path ${DATA_DIR}/processed_datasets/sft_dataset \
--model_name ${DATA_DIR}/Qwen2.5-0.5B-Instruct \
--sft_checkpoint 1500 \
--reward_func exp_inf \
--mu 1 \
--lr 1e-6 \
--kl_beta 1e-3 \
--adam_beta1 0.9 \
--adam_beta2 0.99 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 16 \
--val_max_samples 1600 \
--no-val_shuffle \
--num_train_epochs 1 \
--gradient_accumulation_steps 6 \
--save_strategy steps \
--save_steps ${SAVE_STEPS} \
--logging_steps 10 \
--topk_checkpoints ${TOPK_CHECKPOINTS} \
--topk_metric ${TOPK_METRIC} \
--gradient_checkpointing \
--use_vllm \
--vllm_mode colocate \
--vllm_gpu_memory_utilization 0.25 \
--vllm_tensor_parallel_size ${N_GPUS} \
--max_prompt_length 2048 \
--max_completion_length 1024 \
--num_generations 8 \
--seed 3407 \
--bf16 \
--report_to tensorboard \
--wandb_project rank_grpo \
--catalog_path ${DATA_DIR}/processed_datasets/gt_catalog.pkl \
2>&1 | tee logs/raw_qwen25_0.5b_zero_2gpus.log