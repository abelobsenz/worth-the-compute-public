#!/bin/bash
# Submits every experiment of the project as SLURM jobs.
#
#   bash slurm/submit_all.sh        (from the project folder, after python src/prepare_data.py)
#
# Every script skips the runs whose results already exist, so this file can be submitted again
# after jobs were interrupted. Jobs marked "needs the embeddings" must start after the first job
# has finished, because few-shot prompting and the latency measurement read the BGE embeddings
# that train_classic.py saves.
mkdir -p logs
job() { name=$1; shift; sbatch --parsable -J "$name" slurm/gpu_job.sbatch "$@"; }

# ---- 1. Majority class, TF-IDF + LR, BGE + LR: every budget, seeds 0-4 -----------------------
classic=$(job classic python train_classic.py)
after="--dependency=afterok:$classic"

# ---- 2. SetFit: 8 to 128 labels per class, seeds 0-4 ---------------------------------------
for seed in 0 1 2 3 4; do
  job setfit-s$seed python train_setfit.py --seeds $seed
done

# ---- 3. ModernBERT: every budget, seeds 0-4, three learning rates ---------------------------
for lr in 2e-5 5e-5 1e-4; do
  job modernbert-$lr python train_modernbert.py --lr $lr
done

# ---- 4. Zero-shot prompting, six model sizes ------------------------------------------------
job zs-small bash -c 'for size in 0.6B 1.7B 4B 8B; do python run_llm_prompting.py zeroshot --size $size --val; done'
sbatch --parsable -J zs-14B --gres=gpu:2 slurm/gpu_job.sbatch python run_llm_prompting.py zeroshot --size 14B --val
sbatch --parsable -J zs-32B --gres=gpu:4 --mem=96G slurm/gpu_job.sbatch python run_llm_prompting.py zeroshot --size 32B

# ---- 5. Few-shot prompting (needs the embeddings) -------------------------------------------
# With the full training set as the pool the retrieved examples do not depend on the seed, so
# "full" runs once (seed 0). Seeds 1-4 use the 8 / 32 / 128 budgets.
for size in 1.7B 8B; do
  for dataset in banking77 goemotions; do
    sbatch --parsable $after -J fs-$size-$dataset-s0 slurm/gpu_job.sbatch \
      python run_llm_prompting.py fewshot --size $size --val --datasets $dataset
    sbatch --parsable $after -J fs-$size-$dataset-s1-4 slurm/gpu_job.sbatch \
      python run_llm_prompting.py fewshot --size $size --ks 8 32 128 --seeds 1 2 3 4 --datasets $dataset
  done
done
sbatch --parsable $after -J fs-0.6B slurm/gpu_job.sbatch python run_llm_prompting.py fewshot --size 0.6B --ks 8 --val

# ---- 6. LoRA fine-tuning: 8 / 32 / 128 / all labels, seeds 0-4 -------------------------------
for size in 1.7B 8B; do
  for dataset in banking77 goemotions; do
    for seed in 0 1 2 3 4; do
      job lora-$size-$dataset-s$seed python train_lora.py --size $size --datasets $dataset --seeds $seed
    done
  done
done
job lora-0.6B python train_lora.py --size 0.6B --ks 8 --datasets banking77

# ---- 7. Single-request latency (needs the embeddings) ----------------------------------------
sbatch --parsable $after -J latency slurm/gpu_job.sbatch python measure_latency.py
