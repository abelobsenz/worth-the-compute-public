"""Step 4 (shared by the analysis scripts). Read the results of every run, put a price on each
run, and average over the five seeds.

Two tables are built here and used everywhere else:

  runs      one row per run (dataset, method, budget, seed) with its scores and costs
  summary   one row per (dataset, method, budget): the mean and standard deviation over seeds

Prices
------
A run records how many seconds of GPU and CPU time it used. We convert time to money with a
fixed hourly price, so that methods running on different hardware can be compared:

  GPU   $0.40 per hour   (an on-demand price for a 24 GB card of the kind we used)
  CPU   $0.04 per core per hour
"""
import json
from functools import lru_cache

import numpy as np
import pandas as pd

from common import DATA_DIR, PROJECT_DIR, RESULTS_DIR

GPU_PRICE_PER_HOUR = 0.40
CPU_PRICE_PER_HOUR = 0.04
OUTPUT_DIR = PROJECT_DIR / "outputs"
DATASET_TITLES = {"banking77": "Banking77", "goemotions": "GoEmotions"}


# --------------------------------------------------------------------------------------------
# Reading labels and score tables
# --------------------------------------------------------------------------------------------
def _read_split(dataset, split):
    return [json.loads(line) for line in open(DATA_DIR / dataset / f"{split}.jsonl")]


@lru_cache(maxsize=None)    # remember the answer, so each file is read only once
def validation_rows_to_keep(dataset):
    """True for every validation example whose text does NOT also appear in the test split.

    The official GoEmotions validation and test files share ten short comments (such as "Thank
    you!"). The analysis uses validation data to make choices (a learning rate, a temperature, a
    routing threshold), and those choices must not be influenced by test examples. So these ten
    rows are left out of every use of validation data. Banking77 has no such overlap.
    """
    test_texts = {row["text"] for row in _read_split(dataset, "test")}
    return np.array([row["text"] not in test_texts for row in _read_split(dataset, "val")])


@lru_cache(maxsize=None)
def load_labels(dataset, split="test"):
    """The correct class of every example of a split."""
    labels = np.array([row["label"] for row in _read_split(dataset, split)])
    if split == "val":
        labels = labels[validation_rows_to_keep(dataset)]
    return labels


@lru_cache(maxsize=None)
def label_names(dataset):
    return json.load(open(DATA_DIR / dataset / "labels.json"))["names"]


def n_classes(dataset):
    return len(label_names(dataset))


def load_scores(dataset, folder, run, split="test"):
    """The score table a run saved: one row per example, one column per class."""
    scores = np.load(RESULTS_DIR / dataset / folder / run / f"{split}_scores.npy")
    if split == "val":
        scores = scores[validation_rows_to_keep(dataset)]
    return scores


# --------------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------------
def per_class_f1(labels, predictions, k):
    """F1 score of each of the k classes.

    For one class:  F1 = 2 x (correct predictions of the class)
                         / (examples that truly have the class + examples predicted as the class)
    It is 1 when the class is always found and never predicted wrongly, and 0 when it is never
    predicted correctly.
    """
    correct = np.bincount(labels[labels == predictions], minlength=k)
    total = np.bincount(labels, minlength=k) + np.bincount(predictions, minlength=k)
    return np.where(total > 0, 2 * correct / np.maximum(total, 1), 0.0)


def macro_f1(labels, predictions, k):
    """The average of the per-class F1 scores: every class counts the same, however rare."""
    return float(per_class_f1(labels, predictions, k).mean())


# --------------------------------------------------------------------------------------------
# The table of runs
# --------------------------------------------------------------------------------------------
def load_all_runs():
    """One row per run, read from every results/<dataset>/<folder>/<run>/meta.json.

    Columns
      dataset, folder, run   where the run is stored (folder is the method's results folder)
      budget                 labeled examples per class as text: "8", "32", ..., "full", and "0"
                             for zero-shot runs, which use no labels
      seed                   0 to 4
      n_train                number of labeled training examples the run used
      acc, f1, ece           test accuracy, macro-F1 and calibration error
      train_usd              cost of training the model once, in dollars
      infer_usd              cost of ONE prediction, in dollars
      infer_ms               time per prediction in a large batch, in milliseconds
      train_min              training time in minutes
    """
    rows = []
    for meta_file in sorted(RESULTS_DIR.glob("*/*/*/meta.json")):
        meta = json.load(open(meta_file))
        run = meta["run"]
        if run == "zeroshot":
            budget, seed = "0", 0
        else:                                   # "k32_s1" or "full_s1"
            size, seed_text = run.split("_s")
            budget = "full" if size == "full" else size[1:]
            seed = int(seed_text)

        # Seconds of GPU and CPU time; a missing entry means the run used none.
        train_gpu_sec, train_cpu_sec = meta.get("train_gpu_sec", 0), meta.get("train_sec", 0)
        infer_gpu_sec, infer_cpu_sec = meta.get("infer_gpu_sec", 0), meta.get("infer_sec", 0)
        n_test = meta.get("n_test") or len(load_labels(meta["dataset"]))
        n_cpu_cores = meta.get("n_cpus", 8)    # training on CPU (cross-validation) uses every core

        train_usd = (train_gpu_sec * GPU_PRICE_PER_HOUR + train_cpu_sec * n_cpu_cores * CPU_PRICE_PER_HOUR) / 3600
        # The test split was predicted in one go, so cost per prediction = cost of that / n_test.
        infer_usd = (infer_gpu_sec * GPU_PRICE_PER_HOUR + infer_cpu_sec * CPU_PRICE_PER_HOUR) / 3600 / n_test

        rows.append(dict(dataset=meta["dataset"], folder=meta["method"], run=run, budget=budget, seed=seed,
                         n_train=meta.get("n_train", 0), acc=meta["acc"], f1=meta["macro_f1"], ece=meta["ece"],
                         train_usd=train_usd, infer_usd=infer_usd,
                         infer_ms=1e3 * (infer_gpu_sec + infer_cpu_sec) / n_test,
                         train_min=(train_gpu_sec + train_cpu_sec) / 60, gpu=meta.get("gpu")))
    runs = pd.DataFrame(rows)

    # Few-shot prompting must embed every incoming text with BGE to find similar examples, so a
    # few-shot prediction also pays for one BGE embedding.
    for dataset in runs.dataset.unique():
        bge_runs = runs[(runs.dataset == dataset) & (runs.folder == "bge")]
        if len(bge_runs):
            is_few_shot = (runs.dataset == dataset) & runs.folder.str.startswith("fs-")
            runs.loc[is_few_shot, "infer_usd"] += bge_runs.infer_usd.median()
    return runs


def choose_modernbert_learning_rate(all_runs):
    """Keep, for every dataset and budget, the ModernBERT learning rate with the best VALIDATION
    macro-F1 (averaged over seeds). Choosing on validation data means the test set plays no part.

    Returns the runs with a new column `method`. It equals `folder` for every method except
    ModernBERT, whose chosen runs all get method "modernbert" (their `folder` still tells which
    learning rate was chosen). The runs with the other two learning rates are dropped.
    """
    is_modernbert = all_runs.folder.str.startswith("modernbert")
    bert_runs = all_runs[is_modernbert].copy()
    bert_runs["val_f1"] = [
        macro_f1(load_labels(run.dataset, "val"), load_scores(run.dataset, run.folder, run.run, "val").argmax(1),
                 n_classes(run.dataset))
        for run in bert_runs.itertuples()]

    chosen_parts = []
    for (dataset, budget), group in bert_runs.groupby(["dataset", "budget"]):
        mean_val_f1 = group.groupby("folder").val_f1.mean()      # one number per learning rate
        best_folder = mean_val_f1.idxmax()
        chosen_parts.append(group[group.folder == best_folder])
    chosen = pd.concat(chosen_parts).drop(columns="val_f1")
    chosen["method"] = "modernbert"

    others = all_runs[~is_modernbert].copy()
    others["method"] = others.folder
    return pd.concat([others, chosen], ignore_index=True)


def summarize(runs):
    """Average the runs of each (dataset, method, budget) over their seeds.

    f1 and acc get a mean and a standard deviation (f1_sd, acc_sd); the standard deviation shows
    how much the result moves from one seed to the next. Costs and times are averaged.
    """
    return runs.groupby(["dataset", "method", "budget"], sort=False).agg(
        f1=("f1", "mean"), f1_sd=("f1", "std"), acc=("acc", "mean"), acc_sd=("acc", "std"), ece=("ece", "mean"),
        n_train=("n_train", "mean"), train_usd=("train_usd", "mean"), infer_usd=("infer_usd", "mean"),
        infer_ms=("infer_ms", "mean"), train_min=("train_min", "mean"), seeds=("seed", "nunique"),
        folder=("folder", "first")).reset_index()


def load_results():
    """The three tables the analysis works with: every run, the runs after choosing ModernBERT's
    learning rate, and their summary over seeds."""
    all_runs = load_all_runs()
    runs = choose_modernbert_learning_rate(all_runs)
    return all_runs, runs, summarize(runs)


# --------------------------------------------------------------------------------------------
# Looking things up
# --------------------------------------------------------------------------------------------
def summary_row(summary, dataset, method, budget):
    """The summary row of one (dataset, method, budget), or None if that combination was not run."""
    match = summary[(summary.dataset == dataset) & (summary.method == method) & (summary.budget == str(budget))]
    return match.iloc[0] if len(match) else None


def runs_of(runs, dataset, method, budget):
    """The runs (one per seed, ordered by seed) of one (dataset, method, budget)."""
    match = runs[(runs.dataset == dataset) & (runs.method == method) & (runs.budget == str(budget))]
    return match.sort_values("seed")


def predictions_by_seed(runs, dataset, method, budget, split="test"):
    """{seed: predicted class of every example} for one (dataset, method, budget)."""
    return {run.seed: load_scores(dataset, run.folder, run.run, split).argmax(1)
            for run in runs_of(runs, dataset, method, budget).itertuples()}
