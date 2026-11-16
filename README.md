# Worth the Compute? When LLM Classifiers Pay for Themselves

Code, data and results for comparing cheap text classifiers with large language models (LLMs) used
as classifiers. Every method is trained with a growing number of labeled examples, scored on a
held-out test set, and priced by the labels, training time and prediction time it needs. The code
answers three questions:

1. **Accuracy.** As labeled examples grow from 8 per class to the full training set, when does an
   LLM classify better than a cheaper model?
2. **Cost.** Which method reaches a target score for the least money, given the price of a label
   and the number of predictions to make?
3. **Routing.** If a cheap model answers first and passes only its uncertain inputs to an LLM,
   how much of the LLM's accuracy is kept, and at what cost?

## Datasets

| Dataset | Task | Classes | Train / validation / test |
|---|---|---|---|
| Banking77 | intent of a customer message to a bank | 77 | 9,000 / 1,003 / 3,080 |
| GoEmotions (single-label subset) | emotion of a Reddit comment | 28 | 36,094 / 4,548 / 4,590 |

The original files are in `raw/` and the prepared splits in `data/`.

- Banking77: Casanueva et al., *Efficient Intent Detection with Dual Sentence Encoders* (2020),
  from [PolyAI-LDN/task-specific-datasets](https://github.com/PolyAI-LDN/task-specific-datasets),
  licensed CC BY 4.0.
- GoEmotions: Demszky et al., *GoEmotions: A Dataset of Fine-Grained Emotions* (2020), from
  [google-research/goemotions](https://github.com/google-research/google-research/tree/master/goemotions),
  licensed Apache 2.0.

## Methods

| Name in `results/` | Method | Script |
|---|---|---|
| `majority` | always predict the most common class | `train_classic.py` |
| `tfidf` | TF-IDF word features + logistic regression | `train_classic.py` |
| `bge` | frozen BGE sentence embeddings + logistic regression | `train_classic.py` |
| `setfit` | SetFit: contrastive fine-tuning of the BGE encoder, then logistic regression | `train_setfit.py` |
| `modernbert`, `modernbert-lr2e-05`, `modernbert-lr0.0001` | fine-tuned ModernBERT at three learning rates | `train_modernbert.py` |
| `zs-qwen3-<size>` | Qwen3 prompted with the label names only (zero-shot) | `run_llm_prompting.py` |
| `fs-qwen3-<size>` | Qwen3 prompted with the 10 most similar labeled examples (few-shot) | `run_llm_prompting.py` |
| `lora-qwen3-<size>` | Qwen3 fine-tuned with LoRA | `train_lora.py` |

## Words used in the code

- **Label budget.** The training examples a run may use. `k32_s1` is 32 labeled examples per
  class, random draw number 1; `full_s0` is the whole training set with seed 0. With the same
  seed, a larger budget contains every example of a smaller one.
- **Seed.** Every trained configuration is run five times (seeds 0 to 4) with different labeled
  examples and different training randomness. Results are reported as the mean and standard
  deviation over seeds.
- **Macro-F1.** The F1 score of every class, averaged. A rare class counts as much as a common
  one. This is the main metric.
- **Scores.** Each run saves a table with one row per text and one column per class; the largest
  number in a row is the predicted class. Saving the scores, not just the predictions, lets the
  analysis measure confidence, calibration and routing without running any model again.

## Repository layout

Read the files in this order; each begins with an explanation of what it does and why.

| File | What it does |
|---|---|
| `src/common.py` | Shared helpers: load data, compute metrics, time code, save a run |
| `src/prepare_data.py` | Step 1. Clean the raw files, make the splits and the label budgets |
| `src/logreg.py` | Logistic regression with settings chosen by cross-validation |
| `src/train_classic.py` | Step 2a. Majority class, TF-IDF + LR, BGE + LR |
| `src/train_setfit.py` | Step 2b. SetFit |
| `src/train_modernbert.py` | Step 2c. ModernBERT fine-tuning |
| `src/llm_common.py` | Prompts, and scoring every label with one pass of the LLM |
| `src/run_llm_prompting.py` | Step 2d. Zero-shot and few-shot prompting |
| `src/train_lora.py` | Step 2e. LoRA fine-tuning |
| `src/measure_latency.py` | Step 3. Time of a single prediction for every method |
| `src/analysis_runs.py` | Step 4. Read all runs, price them, average over seeds |
| `src/analysis_cost.py` | Step 4. Cost model: cheapest option, break-even volumes |
| `src/analysis_diagnostics.py` | Step 4. Bootstrap intervals, calibration, per-class scores, retrieval quality |
| `src/analysis_router.py` | Step 4. Routing between a cheap model and the LLM |
| `src/make_tables.py` | Step 5. Write every result table to `outputs/tables/` |
| `src/make_figures.py` | Step 6. Draw the figures in `outputs/figures/` |
| `slurm/` | Job files for running the experiments on a SLURM cluster |
| `raw/`, `data/` | Original dataset files; prepared splits and label budgets |
| `results/` | One folder per run: `results/<dataset>/<method>/<budget>/meta.json` |
| `outputs/` | The tables (CSV) and figures (PNG) produced by steps 5 and 6 |

## Running it

### Setup

```bash
python -m venv venv
```

```bash
source venv/bin/activate
```

```bash
pip install -r requirements.txt
```

Steps 1, 5 and 6 run on a laptop. Steps 2 and 3 need a GPU with 24 GB of memory (the two
largest zero-shot models need 2 and 4 such GPUs).

### Step 1. Prepare the data

```bash
python src/prepare_data.py
```

This rewrites `data/` from `raw/`. The files in `data/` are already included, so this step can
be skipped.

### Steps 2 and 3. Run the experiments

Each script runs one method. Without arguments it runs both datasets, all label budgets and
seeds 0 to 4 (the LLM scripts default to 8, 32, 128 and all labels per class, and to seed 0). The arguments `--datasets`, `--ks` (labels per
class, or `full`) and `--seeds` select a part. Runs whose results already exist are skipped.

```bash
python src/train_classic.py
```

```bash
python src/train_setfit.py
```

```bash
python src/train_modernbert.py --lr 5e-5
```

```bash
python src/run_llm_prompting.py zeroshot --size 8B --val
```

```bash
python src/run_llm_prompting.py fewshot --size 8B --ks 8 32 128 --seeds 0 --val
```

```bash
python src/train_lora.py --size 8B --ks 8 32 128 full --seeds 0
```

```bash
python src/measure_latency.py
```

`train_classic.py` must run before few-shot prompting and the latency measurement, because it
saves the text embeddings they use. A small example that finishes in a few minutes on one GPU:

```bash
python src/train_classic.py --datasets banking77 --ks 8 --seeds 0
```

On a SLURM cluster, `bash slurm/submit_all.sh` submits every experiment (599 runs). Edit the
partition names in `slurm/gpu_job.sbatch` for your cluster first.

### Steps 5 and 6. Tables and figures

`results/` contains the metrics of all 599 runs (`meta.json`). The analysis also needs the saved
score tables, which are too large for git (about 800 MB). Download `score-matrices.tar.gz` from
this repository's [Releases](../../releases) page into the project folder and unpack it:

```bash
tar xzf score-matrices.tar.gz
```

Then:

```bash
python src/make_tables.py
```

```bash
python src/make_figures.py
```

## Outputs

`outputs/tables/`

| File | Content |
|---|---|
| `results_by_method.csv` | macro-F1, accuracy, cost and time of every method at every budget |
| `significance.csv` | LoRA Qwen3-8B against every other method, with 95% bootstrap intervals |
| `per_class_f1.csv`, `rare_classes.csv` | F1 of every class; rare against common classes |
| `confusions.csv` | the most frequent mistakes of the strongest model |
| `calibration.csv` | calibration error before and after temperature scaling |
| `learning_rate_sensitivity.csv` | ModernBERT at each learning rate, and the one chosen on validation |
| `retrieval.csv` | quality of the examples retrieved for few-shot prompting |
| `cost.csv` | training time, cost per million predictions, latency, throughput |
| `cost_break_even.csv`, `zero_shot_break_even.csv` | up to how many predictions an LLM is the cheaper choice |
| `routing.csv`, `routing_by_signal.csv` | cheap model in front, LLM behind: score, share routed, cost |

`outputs/figures/`

| File | Content |
|---|---|
| `learning_curves.png` | test macro-F1 against labeled examples per class, every method |
| `zero_shot_and_per_class.png` | zero-shot score by model size; F1 of every GoEmotions class |
| `cost_maps.png` | the cheapest option for every number of predictions and price per label |
| `routing.png` | test macro-F1 against the share of inputs sent to the LLM |

## How costs are computed

GPU time is priced at $0.40 per hour and CPU time at $0.04 per core per hour. All GPU runs used
NVIDIA RTX A5000 or A5500 cards with 24 GB of memory.

```
total cost = labels x price per label + training cost + predictions x cost per prediction
```

The price of a label and the number of predictions depend on the application, so the analysis
computes the cheapest method for a range of both (`src/analysis_cost.py`).
