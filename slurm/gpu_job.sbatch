#!/bin/bash
# Runs one command on a 24 GB GPU of a SLURM cluster.
#
#   sbatch -J <job name> slurm/gpu_job.sbatch python train_classic.py
#
# Submit from the project folder. The command runs inside src/. Options given on the sbatch
# command line (for example --gres=gpu:4 for the largest model) replace the ones below.
# The partition and node names are those of the cluster the experiments ran on; change them
# for another cluster.
#
# Partitions with NVIDIA RTX A5000 / A5500 cards (24 GB):
#SBATCH --partition=a5500,a5000,gpu_preempt
# Nodes in those partitions that have other kinds of GPU:
#SBATCH --exclude=turing01,turing02,p01,p02,p03,p04,gv01,l40sx4,adanova01,h200a,h200b
# One GPU, two CPU cores, 48 GB of memory, at most 7 hours 55 minutes:
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=7:55:00
# If the job is pre-empted, start it again (finished runs are skipped):
#SBATCH --requeue
# Log file, named after the job name and job number:
#SBATCH --output=logs/%x.%j.out
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
echo "host $(hostname) job $SLURM_JOB_ID start $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$@"
echo "end $(date) exit $?"
