"""Step 4 (routing). Let a cheap model answer first and pass only its uncertain inputs to the LLM.

The idea
--------
The cheap model predicts a class for every input and also tells us how confident it is. We sort
the inputs by confidence and send the LEAST confident ones to the LLM, whose answer replaces the
cheap model's. Sending 0% is the cheap model alone; sending 100% is the LLM alone; in between we
pay for only part of the LLM's work.

Two questions:
  * How do we measure confidence? We compare five "signals" (see confidence_signals).
  * How many inputs do we send? We pick the fraction that gives the best macro-F1 on the
    VALIDATION split, turn it into a confidence threshold, and apply that threshold to the test
    split. The test set is never used to choose anything.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

from analysis_diagnostics import fit_temperature
from analysis_runs import load_labels, load_scores, macro_f1, n_classes, runs_of, summary_row
from common import softmax

SIGNALS = ["msp", "msp_ts", "margin", "neg_entropy", "learned"]
LLM_BEHIND = "lora-qwen3-8B"     # the LLM that receives the routed inputs


def confidence_signals(dataset, cheap_folder, cheap_run, second=None):
    """Five ways of scoring how confident the cheap model is, on the validation and test splits.

    msp          "maximum softmax probability": the probability of the predicted class
    msp_ts       the same after temperature scaling (see analysis_diagnostics.fit_temperature)
    margin       probability of the best class minus probability of the second best
    neg_entropy  minus the entropy of the probabilities: high when one class dominates
    learned      a small logistic regression trained on the validation split to predict "is the
                 cheap model right?" from the four signals above (plus, if `second` is given,
                 whether another cheap model agrees with the prediction)

    For every signal, a higher value means more confident.
    second   optional (folder, run) of another cheap model, used only by the learned signal
    Returns ({"val": {...}, "test": {...}}, temperature).
    """
    val_scores = load_scores(dataset, cheap_folder, cheap_run, "val")
    test_scores = load_scores(dataset, cheap_folder, cheap_run, "test")
    val_labels = load_labels(dataset, "val")
    temperature = fit_temperature(val_scores, val_labels)

    signals = {}
    for split, scores in [("val", val_scores), ("test", test_scores)]:
        probabilities = softmax(scores)
        scaled_probabilities = softmax(scores / temperature)
        sorted_probabilities = np.sort(probabilities, 1)                 # each row from low to high
        margin = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
        neg_entropy = (probabilities * np.log(probabilities + 1e-12)).sum(1)
        features = np.stack([probabilities.max(1), margin, neg_entropy, scaled_probabilities.max(1)], 1)
        if second:
            second_prediction = load_scores(dataset, second[0], second[1], split).argmax(1)
            features = np.column_stack([features, second_prediction == scores.argmax(1)])
        signals[split] = dict(msp=probabilities.max(1), msp_ts=scaled_probabilities.max(1), margin=margin,
                              neg_entropy=neg_entropy, features=features, prediction=scores.argmax(1))

    # The learned signal: probability that the cheap model's prediction is correct.
    is_correct = signals["val"]["prediction"] == val_labels
    correctness_model = LogisticRegression(max_iter=1000).fit(signals["val"]["features"], is_correct)
    for split in signals:
        signals[split]["learned"] = correctness_model.predict_proba(signals[split]["features"])[:, 1]
    return signals, temperature


def routing_curve(confidence, cheap_prediction, llm_prediction, labels, k, fractions):
    """Macro-F1 when the given fractions of the inputs, least confident first, go to the LLM."""
    least_confident_first = np.argsort(confidence, kind="stable")
    f1 = []
    for fraction in fractions:
        prediction = cheap_prediction.copy()
        sent = least_confident_first[:int(round(fraction * len(labels)))]
        prediction[sent] = llm_prediction[sent]
        f1.append(macro_f1(labels, prediction, k))
    return np.array(f1)


def route_one_seed(dataset, cheap_folder, cheap_run, llm_folder, llm_run, second=None,
                   fractions=np.linspace(0, 1, 41)):
    """The routing experiment for one pair of runs (a cheap model and the LLM, same budget and seed).

    Returns a dictionary with
      curves             test macro-F1 for each signal as the routed fraction goes from 0 to 1,
                         plus two reference curves, "oracle" and "random"
      operating_points   for each signal: the fraction chosen on validation, the fraction of test
                         inputs actually sent, and the resulting test macro-F1
      agreement          how often each of the two models is right when the other is wrong
      cheap_f1, llm_f1   test macro-F1 of each model on its own
    """
    k = n_classes(dataset)
    val_labels, test_labels = load_labels(dataset, "val"), load_labels(dataset)
    signals, temperature = confidence_signals(dataset, cheap_folder, cheap_run, second)
    cheap_val, cheap_test = signals["val"]["prediction"], signals["test"]["prediction"]
    llm_val = load_scores(dataset, llm_folder, llm_run, "val").argmax(1)
    llm_test = load_scores(dataset, llm_folder, llm_run).argmax(1)

    curves = {signal: routing_curve(signals["test"][signal], cheap_test, llm_test, test_labels, k, fractions)
              for signal in SIGNALS}
    # Oracle: a perfect signal that knows the answers. It sends exactly the inputs the cheap
    # model gets wrong, starting with those the LLM gets right. No real signal can beat it, so
    # it shows how much room for improvement there is.
    cheap_wrong = cheap_test != test_labels
    oracle_confidence = -(cheap_wrong.astype(float) + (llm_test == test_labels) * cheap_wrong)
    curves["oracle"] = routing_curve(oracle_confidence, cheap_test, llm_test, test_labels, k, fractions)
    # Random: send inputs in a random order (average of 5 orders). A useful signal must beat it.
    rng = np.random.default_rng(0)
    curves["random"] = np.mean([routing_curve(rng.random(len(test_labels)), cheap_test, llm_test, test_labels, k,
                                              fractions) for _ in range(5)], 0)

    # Choose how much to route, using validation data only.
    operating_points = {}
    candidate_fractions = np.linspace(0, 1, 101)       # 0%, 1%, ..., 100%
    for signal in SIGNALS:
        val_f1 = routing_curve(signals["val"][signal], cheap_val, llm_val, val_labels, k, candidate_fractions)
        # the smallest fraction that reaches the best validation macro-F1
        best_fraction = candidate_fractions[np.argmax(val_f1 >= val_f1.max() - 1e-12)]
        # the confidence value below which that fraction of validation inputs falls
        threshold = np.quantile(signals["val"][signal], best_fraction) if best_fraction > 0 else -np.inf
        send_to_llm = signals["test"][signal] < threshold
        prediction = np.where(send_to_llm, llm_test, cheap_test)
        operating_points[signal] = dict(val_fraction=float(best_fraction), test_fraction=float(send_to_llm.mean()),
                                        f1=macro_f1(test_labels, prediction, k),
                                        acc=float((prediction == test_labels).mean()))

    cheap_right, llm_right = cheap_test == test_labels, llm_test == test_labels
    agreement = dict(both_right=float((cheap_right & llm_right).mean()),
                     only_cheap=float((cheap_right & ~llm_right).mean()),
                     only_llm=float((~cheap_right & llm_right).mean()),
                     both_wrong=float((~cheap_right & ~llm_right).mean()))
    return dict(curves=curves, operating_points=operating_points, agreement=agreement, temperature=float(temperature),
                cheap_f1=macro_f1(test_labels, cheap_test, k), llm_f1=macro_f1(test_labels, llm_test, k))


def route_all_seeds(runs, dataset, cheap_method, llm_method, budget, second_method=None):
    """Run route_one_seed for every seed that has both models. Returns {seed: result}."""
    cheap_runs = runs_of(runs, dataset, cheap_method, budget).set_index("seed")
    llm_runs = runs_of(runs, dataset, llm_method, budget).set_index("seed")
    second_runs = runs_of(runs, dataset, second_method, budget).set_index("seed") if second_method else None
    results = {}
    for seed in sorted(set(cheap_runs.index) & set(llm_runs.index)):
        second = None
        if second_runs is not None and seed in second_runs.index:
            second = (second_runs.loc[seed, "folder"], second_runs.loc[seed, "run"])
        results[seed] = route_one_seed(dataset, cheap_runs.loc[seed, "folder"], cheap_runs.loc[seed, "run"],
                                       llm_runs.loc[seed, "folder"], llm_runs.loc[seed, "run"], second)
    return results


def routing_experiments(runs, budgets=(32, "full")):
    """Every routing experiment of the project.

    For each dataset and budget, two cheap models are put in front of the LLM:
      "best"         the cheap model with the highest mean validation macro-F1
      "modernbert"   ModernBERT, the usual choice of small model
    Returns {"<dataset>/<budget>/<best or modernbert>": {"cheap": method, "val_f1": {...},
                                                         "seeds": {seed: result}}}
    """
    experiments = {}
    for dataset in ["banking77", "goemotions"]:
        val_labels, k = load_labels(dataset, "val"), n_classes(dataset)
        for budget in budgets:
            # Mean validation macro-F1 of each candidate cheap model at this budget.
            val_f1 = {}
            for method in ["bge", "modernbert", "setfit"]:
                predictions = predictions_on_validation(runs, dataset, method, budget)
                if predictions:
                    val_f1[method] = float(np.mean([macro_f1(val_labels, p, k) for p in predictions.values()]))
            best = max(val_f1, key=val_f1.get)
            for name, cheap in [("best", best), ("modernbert", "modernbert")]:
                # The learned signal may also look at whether a second cheap model agrees.
                second = "tfidf" if cheap == "bge" else "bge"
                seeds = route_all_seeds(runs, dataset, cheap, LLM_BEHIND, budget, second)
                if seeds:
                    experiments[f"{dataset}/{budget}/{name}"] = dict(cheap=cheap, val_f1=val_f1, seeds=seeds)
    return experiments


def predictions_on_validation(runs, dataset, method, budget):
    return {run.seed: load_scores(dataset, run.folder, run.run, "val").argmax(1)
            for run in runs_of(runs, dataset, method, budget).itertuples()}


def mean_and_sd(values):
    """Mean and sample standard deviation of a list of numbers (sd is NaN for a single number)."""
    values = np.array(values)
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else float("nan")


def summarize_routing(experiments, summary):
    """Average each routing experiment over its seeds. The headline numbers use the "msp_ts"
    signal (confidence after temperature scaling).

    Cost of the routed system per prediction = the cheap model on every input
                                               + the LLM on the routed fraction of inputs.
    """
    table = {}
    for key, experiment in experiments.items():
        dataset, budget, _ = key.split("/")
        seeds = list(experiment["seeds"].values())
        cheap_usd = summary_row(summary, dataset, experiment["cheap"], budget).infer_usd
        llm_usd = summary_row(summary, dataset, LLM_BEHIND, budget).infer_usd
        routed_f1 = [seed["operating_points"]["msp_ts"]["f1"] for seed in seeds]
        fraction = [seed["operating_points"]["msp_ts"]["test_fraction"] for seed in seeds]
        best_single = [max(seed["cheap_f1"], seed["llm_f1"]) for seed in seeds]
        row = dict(
            cheap=experiment["cheap"], seeds=len(seeds),
            cheap_f1=mean_and_sd([seed["cheap_f1"] for seed in seeds]),
            llm_f1=mean_and_sd([seed["llm_f1"] for seed in seeds]),
            routed_f1=mean_and_sd(routed_f1),
            fraction=mean_and_sd(fraction),
            gain_over_best_single=mean_and_sd([r - b for r, b in zip(routed_f1, best_single)]),
            seeds_beating_both=int(sum(r > b for r, b in zip(routed_f1, best_single))),
            only_cheap_right=mean_and_sd([seed["agreement"]["only_cheap"] for seed in seeds]),
            only_llm_right=mean_and_sd([seed["agreement"]["only_llm"] for seed in seeds]),
            both_wrong=mean_and_sd([seed["agreement"]["both_wrong"] for seed in seeds]),
            f1_by_signal={signal: mean_and_sd([seed["operating_points"][signal]["f1"] for seed in seeds])
                          for signal in SIGNALS},
            fraction_by_signal={signal: mean_and_sd([seed["operating_points"][signal]["test_fraction"]
                                                     for seed in seeds]) for signal in SIGNALS},
            oracle_peak=mean_and_sd([seed["curves"]["oracle"].max() for seed in seeds]),
            cheap_usd_per_million=float(1e6 * cheap_usd), llm_usd_per_million=float(1e6 * llm_usd))
        row["routed_usd_per_million"] = float(1e6 * (cheap_usd + row["fraction"][0] * llm_usd))
        table[key] = row
    return table
