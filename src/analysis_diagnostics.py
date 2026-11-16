"""Step 4 (diagnostics). Analyses that explain WHY the methods score the way they do.

  bootstrap_difference   is the gap between two methods larger than chance?
  calibration            can the models' confidence be trusted? (with temperature scaling)
  mean_per_class_f1      which classes does a method get right?
  mean_confusion_matrix  which classes does it mix up?
  retrieval_quality      how good are the "similar examples" that few-shot prompting retrieves?
"""
import json

import numpy as np
from scipy.optimize import minimize_scalar

from analysis_runs import (load_labels, load_scores, macro_f1, n_classes, per_class_f1, predictions_by_seed,
                           runs_of)
from common import DATA_DIR, RESULTS_DIR, expected_calibration_error, softmax


# --------------------------------------------------------------------------------------------
# Is a difference real?
# --------------------------------------------------------------------------------------------
def bootstrap_difference(runs, dataset, method_a, budget_a, method_b, budget_b, n_resamples=2000, seed=0):
    """Macro-F1 of A minus macro-F1 of B (each averaged over its seeds), with a 95% interval.

    The test set is a sample of all the texts the classifiers could meet. With a different sample
    the gap between two methods would come out a little differently. The bootstrap estimates by
    how much: we draw a new "test set" of the same size by picking test examples at random WITH
    replacement, recompute the gap on it, and repeat 2,000 times. The middle 95% of those gaps is
    the interval. If it does not contain 0, the gap is unlikely to be chance.

    It is "paired" because both methods are scored on the same resampled examples each time.
    Also returns in how many seeds (of those both methods have) A beats B on the real test set.
    """
    labels, k = load_labels(dataset), n_classes(dataset)
    predictions_a = predictions_by_seed(runs, dataset, method_a, budget_a)
    predictions_b = predictions_by_seed(runs, dataset, method_b, budget_b)
    table_a = np.stack(list(predictions_a.values()))     # one row per seed
    table_b = np.stack(list(predictions_b.values()))

    rng = np.random.default_rng(seed)
    resampled_gaps = np.empty(n_resamples)
    for i in range(n_resamples):
        chosen = rng.integers(0, len(labels), len(labels))     # example numbers, with repeats
        f1_a = np.mean([macro_f1(labels[chosen], p[chosen], k) for p in table_a])
        f1_b = np.mean([macro_f1(labels[chosen], p[chosen], k) for p in table_b])
        resampled_gaps[i] = f1_a - f1_b

    shared_seeds = sorted(set(predictions_a) & set(predictions_b))
    wins = sum(macro_f1(labels, predictions_a[s], k) > macro_f1(labels, predictions_b[s], k) for s in shared_seeds)
    gap = np.mean([macro_f1(labels, p, k) for p in table_a]) - np.mean([macro_f1(labels, p, k) for p in table_b])
    return dict(difference=float(gap), low=float(np.percentile(resampled_gaps, 2.5)),
                high=float(np.percentile(resampled_gaps, 97.5)), seed_wins=int(wins), seeds=len(shared_seeds))


# --------------------------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------------------------
def fit_temperature(scores, labels):
    """Find the temperature T that makes softmax(scores / T) fit the correct labels best.

    Dividing the scores by T changes how confident the probabilities are without changing which
    class is predicted: T > 1 spreads probability over more classes (less confident), T < 1
    concentrates it (more confident). We pick the T that gives the correct class the highest
    probability on average (the lowest "negative log-likelihood"), searching between 0.05 and 20.
    This is "temperature scaling", a standard fix for over- or under-confident models. T is
    always fitted on validation data.
    """
    def negative_log_likelihood(temperature):
        probabilities = softmax(scores / temperature)
        probability_of_correct = probabilities[np.arange(len(labels)), labels]
        return -np.log(np.clip(probability_of_correct, 1e-12, None)).mean()
    return minimize_scalar(negative_log_likelihood, bounds=(0.05, 20), method="bounded").x


def calibration(runs, dataset, method, budget):
    """Test calibration error (ECE) before and after temperature scaling, averaged over seeds.

    Returns None if the method has no validation scores at this budget.
    """
    val_labels, test_labels = load_labels(dataset, "val"), load_labels(dataset)
    raw, scaled, temperatures = [], [], []
    for run in runs_of(runs, dataset, method, budget).itertuples():
        if not (RESULTS_DIR / dataset / run.folder / run.run / "val_scores.npy").exists():
            continue
        val_scores = load_scores(dataset, run.folder, run.run, "val")
        test_scores = load_scores(dataset, run.folder, run.run)
        temperature = fit_temperature(val_scores, val_labels)
        raw.append(expected_calibration_error(softmax(test_scores), test_labels))
        scaled.append(expected_calibration_error(softmax(test_scores / temperature), test_labels))
        temperatures.append(temperature)
    if not raw:
        return None
    return dict(raw=float(np.mean(raw)), scaled=float(np.mean(scaled)), temperature=float(np.mean(temperatures)),
                seeds=len(raw))


# --------------------------------------------------------------------------------------------
# Per-class behaviour
# --------------------------------------------------------------------------------------------
def mean_per_class_f1(runs, dataset, method, budget):
    """Test F1 of every class, averaged over seeds."""
    labels, k = load_labels(dataset), n_classes(dataset)
    predictions = predictions_by_seed(runs, dataset, method, budget)
    return np.mean([per_class_f1(labels, p, k) for p in predictions.values()], 0)


def mean_confusion_matrix(runs, dataset, method, budget):
    """Entry [i, j] = how many test examples of true class i were predicted as class j, averaged
    over seeds. The diagonal holds the correct predictions; everything else is an error."""
    labels, k = load_labels(dataset), n_classes(dataset)
    predictions = predictions_by_seed(runs, dataset, method, budget)
    matrix = np.zeros((k, k))
    for p in predictions.values():
        np.add.at(matrix, (labels, p), 1)      # add 1 at [true class, predicted class]
    return matrix / max(len(predictions), 1)


# --------------------------------------------------------------------------------------------
# Quality of the retrieved examples
# --------------------------------------------------------------------------------------------
def retrieval_quality(dataset, budget_name, few_shot_folder=None, n_examples=10):
    """For every test text, look at its 10 nearest labeled examples in BGE embedding space (the
    same ones few-shot prompting puts in the prompt) and measure:

      neighbour_share    share of those neighbours that have the test text's correct label
      label_in_examples  share of test texts with the correct label among their 10 neighbours
                         (if it is missing, the prompt never shows the model the right answer)
      nearest_correct    share of test texts whose single nearest neighbour has the correct label
      prediction_in_examples   share of test texts for which the few-shot model's prediction is
                               one of the labels shown in the prompt (how much it "copies")

    These numbers show whether the frozen embedding space is organized by the classes.
    """
    embeddings = {split: np.load(RESULTS_DIR / dataset / "_emb" / f"bge-base_{split}.npy")
                  for split in ["train", "test"]}
    positions = np.array(json.load(open(DATA_DIR / dataset / "budgets.json"))[budget_name])
    pool_labels = load_labels(dataset, "train")[positions]
    test_labels = load_labels(dataset)
    similarity = embeddings["test"] @ embeddings["train"][positions].T
    nearest = np.argsort(-similarity, axis=1)[:, :n_examples]
    neighbour_labels = pool_labels[nearest]                    # one row of 10 labels per test text
    is_correct_label = neighbour_labels == test_labels[:, None]
    quality = dict(neighbour_share=float(is_correct_label.mean()),
                   label_in_examples=float(is_correct_label.any(1).mean()),
                   nearest_correct=float((neighbour_labels[:, 0] == test_labels).mean()))
    if few_shot_folder and (RESULTS_DIR / dataset / few_shot_folder / budget_name / "test_scores.npy").exists():
        prediction = load_scores(dataset, few_shot_folder, budget_name).argmax(1)
        quality["prediction_in_examples"] = float((neighbour_labels == prediction[:, None]).any(1).mean())
    return quality
