# Environment shared by every job. It is loaded by gpu_job.sbatch.
#
# WTC_ROOT is the project folder on the cluster; it defaults to the folder the job was
# submitted from. It must contain a Python virtual environment in venv/ with the packages of
# requirements.txt installed.
export WTC_ROOT=${WTC_ROOT:-$SLURM_SUBMIT_DIR}

# Where Hugging Face keeps downloaded models. The compute nodes have no internet access, so the
# models must be downloaded to this folder beforehand (on a login node), and the two OFFLINE
# settings stop the libraries from trying to reach the internet.
export HF_HOME=${HF_HOME:-$WTC_ROOT/hf_cache}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

export TOKENIZERS_PARALLELISM=false                       # avoids a warning when jobs fork
export PYTHONUNBUFFERED=1                                 # print() shows up in the log at once
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # less GPU memory fragmentation

source "$WTC_ROOT/venv/bin/activate"
cd "$WTC_ROOT/src"
