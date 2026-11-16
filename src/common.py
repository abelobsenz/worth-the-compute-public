"""Helpers shared by every script: loading data, computing metrics, timing code, saving a run.

Words used throughout the project
---------------------------------
dataset   "banking77" or "goemotions".
split     "train", "val" (validation) or "test".
budget    Which training examples a run is allowed to use. "k32_s1" means 32 labeled examples
          per class, random draw number 1. "full_s0" means the whole training set, seed 0.
method    The name of a classifier, e.g. "tfidf", "bge", "setfit", "modernbert",
          "zs-qwen3-8B" (zero-shot), "fs-qwen3-8B" (few-shot) or "lora-qwen3-8B".
scores    A table of numbers with one row per text and one column per class. A higher number
          means the model finds that class more likely, so the predicted class of a text is
          the column holding the largest number in its row.

Every run writes its output to results/<dataset>/<method>/<budget>/:
    meta.json         accuracy, macro-F1, calibration error, timings, hardware
    test_scores.npy   the score table for the test split
    val_scores.npy    the score table for the validation split (when it was computed)
"""
import json
import os
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# The project folder is the parent of src/. Set the environment variable WTC_ROOT to use
# another location (for example a scratch disk on a cluster).
PROJECT_DIR = Path(os.environ.get("WTC_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
DATASETS = ["banking77", "goemotions"]


# --------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------
def load_dataset(dataset):
    """Read the three splits of a dataset.

    Returns a dictionary:
        data["train"], data["val"], data["test"]   each a pair (texts, labels), where texts is a
                                                   list of strings and labels a NumPy array of
                                                   class numbers (0, 1, 2, ...)
        data["label_names"]                        class number -> readable name
    """
    data = {}
    for split in ["train", "val", "test"]:
        # Each line of a .jsonl file is one example: {"text": "...", "label": 3}
        rows = [json.loads(line) for line in open(DATA_DIR / dataset / f"{split}.jsonl")]
        texts = [row["text"] for row in rows]
        labels = np.array([row["label"] for row in rows])
        data[split] = (texts, labels)
    data["label_names"] = json.load(open(DATA_DIR / dataset / "labels.json"))["names"]
    return data


def budget_indices(dataset, budget):
    """Positions in the training split that the budget (e.g. "k32_s1") is allowed to use."""
    all_budgets = json.load(open(DATA_DIR / dataset / "budgets.json"))
    return np.array(all_budgets[budget])


def budget_names(dataset, sizes, seeds):
    """Turn sizes like [8, 32, "full"] and seeds like [0, 1] into the budget names that exist.

    Some sizes do not exist for a dataset (Banking77 has no class with 512 examples), so they
    are skipped.
    """
    available = json.load(open(DATA_DIR / dataset / "budgets.json"))
    names = []
    for size in sizes:
        for seed in seeds:
            name = f"full_s{seed}" if size == "full" else f"k{size}_s{seed}"
            if name in available and name not in names:
                names.append(name)
    return names


def seed_of(budget):
    """The seed number at the end of a budget name: "k32_s1" -> 1."""
    return int(budget.split("_s")[1])


def parse_size(text):
    """Command-line helper: "full" stays text, anything else becomes a whole number."""
    return text if text == "full" else int(text)


# --------------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------------
def softmax(scores, axis=-1):
    """Turn each row of scores into probabilities that are positive and add up to 1."""
    scores = scores - scores.max(axis=axis, keepdims=True)  # subtracting the max avoids overflow
    exponentials = np.exp(scores)
    return exponentials / exponentials.sum(axis=axis, keepdims=True)


def expected_calibration_error(probabilities, labels, n_bins=15):
    """How far the model's confidence is from its actual accuracy (0 is perfect).

    A model is calibrated if, among the predictions it makes with 80% confidence, 80% are right.
    We sort predictions into 15 confidence bins (0-6.7%, 6.7-13.3%, ...). In each bin we compare
    the average confidence with the share of correct predictions, and we average those gaps,
    weighting each bin by how many predictions fall in it.
    """
    confidence = probabilities.max(1)        # probability of the predicted class
    prediction = probabilities.argmax(1)     # the predicted class
    bin_of = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    error = 0.0
    for b in range(n_bins):
        in_bin = bin_of == b
        if in_bin.any():
            accuracy_in_bin = (prediction[in_bin] == labels[in_bin]).mean()
            confidence_in_bin = confidence[in_bin].mean()
            error += in_bin.mean() * abs(accuracy_in_bin - confidence_in_bin)
    return float(error)


def evaluate(probabilities, labels):
    """Accuracy, macro-F1 and calibration error of a table of class probabilities.

    Macro-F1 computes the F1 score of every class separately and averages them, so a rare class
    counts as much as a common one. That is why it is the project's main metric.
    """
    prediction = probabilities.argmax(1)
    n_classes = probabilities.shape[1]
    return {
        "acc": float(accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", labels=np.arange(n_classes),
                                   zero_division=0)),
        "ece": expected_calibration_error(probabilities, labels),
    }


# --------------------------------------------------------------------------------------------
# Saving runs
# --------------------------------------------------------------------------------------------
def run_folder(dataset, method, run):
    folder = RESULTS_DIR / dataset / method / run
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def is_done(dataset, method, run):
    """A run is finished once its meta.json exists. Scripts skip finished runs, so a job that
    was interrupted can simply be started again."""
    return (RESULTS_DIR / dataset / method / run / "meta.json").exists()


def hardware_info():
    """Which hardware the run used; the cost analysis needs the CPU count and the GPU type."""
    if hasattr(os, "sched_getaffinity"):      # Linux: the CPUs this job is allowed to use
        n_cpus = len(os.sched_getaffinity(0))
    else:
        n_cpus = os.cpu_count()
    info = {"n_cpus": n_cpus}
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["n_gpus"] = torch.cuda.device_count()
    except ImportError:
        pass
    return info


def save_run(dataset, method, run, test_labels, test_scores, val_scores=None, **extra):
    """Store the score tables of a finished run together with its metrics.

    The scores can be any numbers where higher means more likely (log-probabilities or raw model
    outputs). The metrics are computed on their softmax. `extra` holds anything else worth
    recording, such as training time or the number of training examples.
    """
    folder = run_folder(dataset, method, run)
    np.save(folder / "test_scores.npy", test_scores.astype(np.float32))
    if val_scores is not None:
        np.save(folder / "val_scores.npy", val_scores.astype(np.float32))
    metrics = evaluate(softmax(test_scores), test_labels)
    meta = {"dataset": dataset, "method": method, "run": run, **metrics, **extra, **hardware_info()}
    json.dump(meta, open(folder / "meta.json", "w"), indent=1)
    timings = " ".join(f"{key}={value:.1f}" for key, value in meta.items() if key.endswith("_sec"))
    print(f"[{dataset} {method} {run}] acc={metrics['acc']:.4f} macroF1={metrics['macro_f1']:.4f} "
          f"ece={metrics['ece']:.3f} {timings}", flush=True)
    return meta


# --------------------------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------------------------
class Timer:
    """Measures how long a block of code takes:

        with Timer() as timer:
            do_work()
        print(timer.sec)

    A GPU works in the background: Python can move on before the GPU has finished. Before reading
    the clock we therefore wait for the GPU, otherwise GPU work would look faster than it is.
    """

    def __enter__(self):
        _wait_for_gpu()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *unused):
        _wait_for_gpu()
        self.sec = time.perf_counter() - self.start


def _wait_for_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass
