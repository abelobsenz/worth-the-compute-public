"""Step 5. Turn the results of all runs into the project's result tables (CSV files).

Needs the score tables of the runs in results/ (see the README for how to get them).

Run it with:  python src/make_tables.py
Output:       outputs/tables/*.csv, and the main table printed on the screen

Tables
  results_by_method.csv          macro-F1, accuracy, cost and time of every method at every budget
  significance.csv               LoRA Qwen3-8B against every other method, with 95% intervals
  per_class_f1.csv               F1 of every class for the main methods
  rare_classes.csv               GoEmotions: rare classes against common ones
  confusions.csv                 the most frequent mistakes of the strongest model
  calibration.csv                calibration error before and after temperature scaling
  learning_rate_sensitivity.csv  ModernBERT at each of its three learning rates
  retrieval.csv                  quality of the examples retrieved for few-shot prompting
  cost.csv                       training time, cost per million predictions, latency
  cost_break_even.csv            LLM against non-LLM: up to how many predictions the LLM is cheaper
  zero_shot_break_even.csv       the same for zero-shot prompting
  routing.csv                    cheap model in front, LLM behind: score, share routed, cost
  routing_by_signal.csv          the routed score for each confidence signal
"""
import json

import numpy as np
import pandas as pd

from analysis_cost import TARGETS, break_even_volumes, zero_shot_break_even
from analysis_diagnostics import (bootstrap_difference, calibration, mean_confusion_matrix, mean_per_class_f1,
                                  retrieval_quality)
from analysis_router import SIGNALS, routing_experiments, summarize_routing
from analysis_runs import (OUTPUT_DIR, label_names, load_labels, load_results, n_classes, runs_of, summary_row)
from common import RESULTS_DIR

TABLE_DIR = OUTPUT_DIR / "tables"
DATASETS = ["banking77", "goemotions"]
STRONGEST = "lora-qwen3-8B"

METHOD_TITLES = {
    "majority": "Majority class", "tfidf": "TF-IDF + LR", "bge": "BGE + LR", "setfit": "SetFit",
    "modernbert": "ModernBERT", "fs-qwen3-1.7B": "Few-shot Qwen3-1.7B", "fs-qwen3-8B": "Few-shot Qwen3-8B",
    "lora-qwen3-1.7B": "LoRA Qwen3-1.7B", "lora-qwen3-8B": "LoRA Qwen3-8B",
    "zs-qwen3-8B": "Zero-shot Qwen3-8B", "zs-qwen3-32B": "Zero-shot Qwen3-32B",
}


def save(rows, name):
    """Write a list of dictionaries (or a DataFrame) to outputs/tables/<name>.csv."""
    table = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    table.to_csv(TABLE_DIR / f"{name}.csv", index=False)
    print(f"wrote outputs/tables/{name}.csv ({len(table)} rows)")
    return table


def budget_title(budget):
    return {"full": "all", "0": "none"}.get(str(budget), str(budget))


# --------------------------------------------------------------------------------------------
def results_table(summary):
    """Every method at every budget: mean and standard deviation over seeds."""
    table = summary.sort_values(["dataset", "method", "n_train"]).copy()
    table["usd_per_million_predictions"] = 1e6 * table.infer_usd
    table = table.rename(columns={"f1": "macro_f1", "f1_sd": "macro_f1_sd", "acc": "accuracy", "acc_sd": "accuracy_sd",
                                  "infer_ms": "ms_per_prediction", "train_min": "train_minutes",
                                  "folder": "results_folder"})
    columns = ["dataset", "method", "budget", "seeds", "n_train", "macro_f1", "macro_f1_sd", "accuracy", "accuracy_sd",
               "ece", "train_minutes", "train_usd", "usd_per_million_predictions", "ms_per_prediction", "results_folder"]
    return save(table[columns], "results_by_method")


def print_main_table(summary):
    """Macro-F1 (%) by method (rows) and labels per class (columns), like the main results table."""
    for dataset in DATASETS:
        zero_shot = summary[(summary.dataset == dataset) & summary.method.str.startswith("zs-")]
        print(f"\n{dataset}: test macro-F1 (%), mean over seeds")
        print("  zero-shot, no labels: " + ", ".join(
            f"{row.method.split('-')[-1]} {100 * row.f1:.1f}" for row in zero_shot.sort_values("infer_usd").itertuples()))
        for method in ["tfidf", "bge", "setfit", "modernbert", "fs-qwen3-1.7B", "fs-qwen3-8B", "lora-qwen3-1.7B",
                       "lora-qwen3-8B"]:
            cells = []
            for budget in [8, 16, 32, 64, 128, 512, "full"]:
                row = summary_row(summary, dataset, method, budget)
                cells.append("    -      " if row is None else f"{100 * row.f1:5.1f}±{100 * row.f1_sd:3.1f}  "
                             if row.f1_sd == row.f1_sd else f"{100 * row.f1:5.1f}      ")
            print(f"  {METHOD_TITLES[method]:21s}" + "".join(cells))
        print("  " + " " * 21 + "".join(f"{budget_title(b):^11s}" for b in [8, 16, 32, 64, 128, 512, "full"]))


def significance_table(runs):
    """The strongest model against every other method at the same budget."""
    rows = []
    for dataset in DATASETS:
        for budget in [8, 32, 128, "full"]:
            for other in ["tfidf", "bge", "setfit", "modernbert", "fs-qwen3-8B", "lora-qwen3-1.7B"]:
                if len(runs_of(runs, dataset, STRONGEST, budget)) and len(runs_of(runs, dataset, other, budget)):
                    result = bootstrap_difference(runs, dataset, STRONGEST, budget, other, budget)
                    rows.append(dict(dataset=dataset, budget=budget_title(budget), method_a=STRONGEST, method_b=other,
                                     f1_a_minus_b=result["difference"], interval_low=result["low"],
                                     interval_high=result["high"], seeds_a_wins=result["seed_wins"],
                                     seeds_compared=result["seeds"]))
    return save(rows, "significance")


def per_class_tables(runs):
    """F1 of every class, and for GoEmotions the gap between rare and common classes."""
    rows = []
    for dataset in DATASETS:
        class_sizes = np.bincount(load_labels(dataset, "train"), minlength=n_classes(dataset))
        for budget in [32, "full"]:
            f1 = {method: mean_per_class_f1(runs, dataset, method, budget)
                  for method in ["tfidf", "bge", "setfit", "modernbert", "lora-qwen3-1.7B", "lora-qwen3-8B"]
                  if len(runs_of(runs, dataset, method, budget))}
            for c, name in enumerate(label_names(dataset)):
                row = dict(dataset=dataset, budget=budget_title(budget), label=name, train_examples=int(class_sizes[c]))
                row.update({method: float(values[c]) for method, values in f1.items()})
                rows.append(row)
    save(rows, "per_class_f1")

    # GoEmotions with all labels: classes with fewer than 512 training examples are "rare".
    dataset, budget, cutoff = "goemotions", "full", 512
    is_rare = np.bincount(load_labels(dataset, "train"), minlength=n_classes(dataset)) < cutoff
    rows = []
    for method in ["tfidf", "bge", "modernbert", "lora-qwen3-1.7B", "lora-qwen3-8B"]:
        f1 = mean_per_class_f1(runs, dataset, method, budget)
        rows.append(dict(method=method, n_rare_classes=int(is_rare.sum()), f1_rare_classes=float(f1[is_rare].mean()),
                         f1_common_classes=float(f1[~is_rare].mean())))
    save(rows, "rare_classes")


def confusion_table(runs, top=4):
    """The pairs (true class -> predicted class) the strongest model confuses most often."""
    rows = []
    for dataset in DATASETS:
        names = label_names(dataset)
        matrix = mean_confusion_matrix(runs, dataset, STRONGEST, "full")
        examples_per_class = matrix.sum(1)
        errors = matrix.copy()
        np.fill_diagonal(errors, 0)                 # keep only the mistakes
        largest = np.argsort(-errors.ravel())[:top]    # positions of the largest entries
        for true_class, predicted_class in zip(*np.unravel_index(largest, errors.shape)):
            rows.append(dict(dataset=dataset, true_label=names[true_class], predicted_label=names[predicted_class],
                             mean_count=float(errors[true_class, predicted_class]),
                             test_examples_of_true_label=int(examples_per_class[true_class])))
        if "neutral" in names:       # GoEmotions: how many of the errors involve "neutral"?
            neutral = names.index("neutral")
            share = (errors[neutral].sum() + errors[:, neutral].sum()) / errors.sum()
            print(f"  {dataset}: {100 * share:.0f}% of the errors have 'neutral' as the true or the predicted label")
    return save(rows, "confusions")


def calibration_table(runs):
    rows = []
    for method in ["tfidf", "bge", "setfit", "modernbert", "lora-qwen3-1.7B", "lora-qwen3-8B"]:
        for dataset in DATASETS:
            for budget in [32, "full"]:
                result = calibration(runs, dataset, method, budget)
                if result is not None:
                    rows.append(dict(method=method, dataset=dataset, budget=budget_title(budget), ece_raw=result["raw"],
                                     ece_after_temperature_scaling=result["scaled"],
                                     temperature=result["temperature"], seeds=result["seeds"]))
    return save(rows, "calibration")


def learning_rate_table(all_runs, summary):
    """ModernBERT's test macro-F1 at each learning rate, and which one validation data chose."""
    learning_rates = {"modernbert-lr2e-05": "2e-5", "modernbert": "5e-5", "modernbert-lr0.0001": "1e-4"}
    rows = []
    for dataset in DATASETS:
        for budget in [8, 32, 128, "full"]:
            chosen_folder = summary_row(summary, dataset, "modernbert", budget).folder
            for folder, learning_rate in learning_rates.items():
                match = all_runs[(all_runs.dataset == dataset) & (all_runs.folder == folder)
                                 & (all_runs.budget == str(budget))]
                rows.append(dict(dataset=dataset, budget=budget_title(budget), learning_rate=learning_rate,
                                 macro_f1=float(match.f1.mean()), macro_f1_sd=float(match.f1.std()),
                                 chosen_on_validation=folder == chosen_folder))
    return save(rows, "learning_rate_sensitivity")


def retrieval_table(runs):
    """Averaged over seeds: every seed has its own pool of labeled examples."""
    rows = []
    for dataset in DATASETS:
        for budget in [8, 32, 128, "full"]:
            per_seed = [retrieval_quality(dataset, run.run, "fs-qwen3-8B")
                        for run in runs_of(runs, dataset, "bge", budget).itertuples()]
            row = dict(dataset=dataset, budget=budget_title(budget))
            for measure in per_seed[0]:
                row[measure] = float(np.mean([seed[measure] for seed in per_seed if measure in seed]))
            rows.append(row)
    return save(rows, "retrieval")


def cost_table(summary):
    """Training time, cost per million predictions, single-request latency and throughput."""
    latency_file = RESULTS_DIR / "latency.json"
    latency = json.load(open(latency_file))["banking77"] if latency_file.exists() else {}
    # (method, the budget whose model is priced). Zero-shot models have no budget ("0").
    priced = [("tfidf", 32), ("bge", 32), ("setfit", 32), ("modernbert", 32), ("zs-qwen3-8B", 0), ("zs-qwen3-32B", 0),
              ("fs-qwen3-8B", 32), ("lora-qwen3-1.7B", 32), ("lora-qwen3-8B", 32)]
    rows = []
    for method, budget in priced:
        banking = summary_row(summary, "banking77", method, budget)
        goemotions = summary_row(summary, "goemotions", method, budget)

        def train_minutes(dataset, train_budget):
            if method.startswith(("zs-", "fs-")):
                return 0.0                        # prompting needs no training
            row = summary_row(summary, dataset, method, train_budget)
            return None if row is None else float(row.train_min)

        rows.append(dict(
            method=METHOD_TITLES[method],
            train_minutes_banking77_32=train_minutes("banking77", 32),
            train_minutes_banking77_all=train_minutes("banking77", "full"),
            train_minutes_goemotions_all=train_minutes("goemotions", "full"),
            usd_per_million_banking77=float(1e6 * banking.infer_usd),
            usd_per_million_goemotions=float(1e6 * goemotions.infer_usd),
            latency_median_ms=latency.get(method, {}).get("p50_ms"),
            latency_95th_percentile_ms=latency.get(method, {}).get("p95_ms"),
            predictions_per_second=float(1e3 / banking.infer_ms)))
    return save(rows, "cost")


def routing_tables(runs, summary):
    experiments = routing_experiments(runs)
    routing = summarize_routing(experiments, summary)
    rows, signal_rows = [], []
    for key, result in routing.items():
        dataset, budget, front = key.split("/")
        rows.append(dict(
            dataset=dataset, budget=budget_title(budget), cheap_model=result["cheap"],
            front_model_rule="best on validation" if front == "best" else "ModernBERT", seeds=result["seeds"],
            cheap_f1=result["cheap_f1"][0], cheap_f1_sd=result["cheap_f1"][1],
            llm_f1=result["llm_f1"][0], llm_f1_sd=result["llm_f1"][1],
            routed_f1=result["routed_f1"][0], routed_f1_sd=result["routed_f1"][1],
            share_sent_to_llm=result["fraction"][0], share_sent_to_llm_sd=result["fraction"][1],
            seeds_routed_beats_both=result["seeds_beating_both"],
            share_only_cheap_right=result["only_cheap_right"][0], share_only_llm_right=result["only_llm_right"][0],
            oracle_best_f1=result["oracle_peak"][0],
            routed_usd_per_million=result["routed_usd_per_million"], llm_usd_per_million=result["llm_usd_per_million"]))
        for signal in SIGNALS:
            signal_rows.append(dict(dataset=dataset, budget=budget_title(budget), cheap_model=result["cheap"],
                                    front_model_rule=rows[-1]["front_model_rule"], signal=signal,
                                    routed_f1=result["f1_by_signal"][signal][0],
                                    share_sent_to_llm=result["fraction_by_signal"][signal][0]))
    save(rows, "routing")
    save(signal_rows, "routing_by_signal")


if __name__ == "__main__":
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    all_runs, runs, summary = load_results()
    print(f"{len(all_runs)} runs found")

    results_table(summary)
    significance_table(runs)
    per_class_tables(runs)
    confusion_table(runs)
    calibration_table(runs)
    learning_rate_table(all_runs, summary)
    retrieval_table(runs)
    cost_table(summary)
    save(break_even_volumes(summary, TARGETS), "cost_break_even")
    save(zero_shot_break_even(summary, TARGETS), "zero_shot_break_even")
    routing_tables(runs, summary)
    print_main_table(summary)
