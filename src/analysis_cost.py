"""Step 4 (cost). Which classifier reaches a target score for the least money?

The cost model
--------------
Building and running a classifier costs three things:

    total cost  =  (number of labels x price of one label)       paid once, to people who label
                +  cost of training                               paid once, for computer time
                +  (number of predictions x cost of one prediction)    paid for as long as it runs

An "option" is a method trained on a certain label budget, for example "BGE + LR with 32 labels
per class". Its score, training cost and cost per prediction come from our measurements (the
summary table). The price of a label and the number of predictions depend on the application, so
we do not fix them: we compute the cheapest option for a whole grid of both.

An LLM typically needs fewer labels but costs more per prediction. So it tends to win when
labels are expensive or predictions are few, and to lose once enough predictions have been made.
The number of predictions at which the two options cost the same is the break-even volume.
"""
import pandas as pd

LABEL_PRICES = [0.01, 0.1, 1.0]    # dollars per label used for the break-even tables
# The target macro-F1 scores we study: one for Banking77 and three for GoEmotions, from a score
# that zero-shot prompting already reaches (0.30) to one that needs a strong trained model (0.45).
TARGETS = {"banking77": [0.90], "goemotions": [0.30, 0.35, 0.45]}


def options_reaching(summary, dataset, target):
    """Every (method, budget) of a dataset whose mean macro-F1 reaches the target."""
    reached = (summary.dataset == dataset) & (summary.f1 >= target) & (summary.method != "majority")
    return summary[reached].copy()


def cheapest_options(summary, dataset, target, volumes, label_prices):
    """For every (number of predictions, price per label): the cheapest option reaching the target.

    Returns a table with columns volume, label_usd, method, budget, total_usd.
    """
    options = options_reaching(summary, dataset, target)
    rows = []
    for volume in volumes:
        for label_price in label_prices:
            total = options.n_train * label_price + options.train_usd + volume * options.infer_usd
            if len(options):
                best = total.idxmin()
                rows.append(dict(volume=volume, label_usd=label_price, method=options.loc[best, "method"],
                                 budget=options.loc[best, "budget"], total_usd=total[best]))
            else:    # nothing reaches the target
                rows.append(dict(volume=volume, label_usd=label_price, method=None, budget=None, total_usd=None))
    return pd.DataFrame(rows)


def break_even_volumes(summary, targets, llm_cost_scale=None):
    """LLM versus non-LLM: up to how many predictions is the LLM the cheaper choice?

    For each dataset, target and label price we take
      * the LLM option that needs labels and is cheapest to set up (labels + training), and
      * the non-LLM option that is cheapest to set up,
    both among the options that reach the target. The LLM option usually needs fewer labels
    (cheaper to set up) but costs more per prediction, so there is a volume where the totals meet:

        break-even volume = (set-up cost of non-LLM - set-up cost of LLM)
                            / (cost per prediction of LLM - cost per prediction of non-LLM)

    llm_cost_scale   multiply the LLM's cost per prediction by this number, to ask "what if the
                     LLM were served more efficiently?" (0.1 = ten times cheaper)

    targets          {dataset: [target macro-F1, ...]}
    Returns a list of dictionaries, one per (dataset, target, label price).
    """
    rows = []
    for dataset, dataset_targets in targets.items():
        for target in dataset_targets:
            options = options_reaching(summary, dataset, target)
            is_llm = options.method.str.contains("qwen3")
            if llm_cost_scale:
                options.loc[is_llm, "infer_usd"] *= llm_cost_scale
            for label_price in LABEL_PRICES:
                setup_cost = options.n_train * label_price + options.train_usd
                llm_options = options[is_llm & (options.n_train > 0)]   # zero-shot is handled below
                other_options = options[~is_llm]
                if llm_options.empty or other_options.empty:
                    continue
                llm = setup_cost[llm_options.index].idxmin()
                other = setup_cost[other_options.index].idxmin()
                extra_per_prediction = options.loc[llm, "infer_usd"] - options.loc[other, "infer_usd"]
                if extra_per_prediction > 0 and setup_cost[other] > setup_cost[llm]:
                    break_even = float((setup_cost[other] - setup_cost[llm]) / extra_per_prediction)
                else:
                    break_even = None    # the LLM is not cheaper to set up, so it never leads
                rows.append(dict(
                    dataset=dataset, target=target, label_usd=label_price,
                    llm_option=f"{options.loc[llm, 'method']} @ {options.loc[llm, 'budget']}",
                    other_option=f"{options.loc[other, 'method']} @ {options.loc[other, 'budget']}",
                    labels_saved=float(options.loc[other, "n_train"] - options.loc[llm, "n_train"]),
                    extra_usd_per_million=float(1e6 * extra_per_prediction),
                    break_even_volume=break_even))
    return rows


def zero_shot_break_even(summary, targets):
    """Zero-shot prompting needs no labels and no training, so it costs nothing to set up.

    Up to how many predictions is the cheapest zero-shot model that reaches the target cheaper
    than the non-LLM option that is cheapest to set up?

        break-even volume = set-up cost of non-LLM
                            / (cost per prediction of zero-shot - cost per prediction of non-LLM)
    """
    rows = []
    for dataset, dataset_targets in targets.items():
        for target in dataset_targets:
            options = options_reaching(summary, dataset, target)
            zero_shot_options = options[options.method.str.startswith("zs-")]
            other_options = options[~options.method.str.contains("qwen3")]
            if zero_shot_options.empty or other_options.empty:
                continue
            zero_shot = zero_shot_options.loc[zero_shot_options.infer_usd.idxmin()]
            for label_price in LABEL_PRICES:
                setup_cost = other_options.n_train * label_price + other_options.train_usd
                other = other_options.loc[setup_cost.idxmin()]
                rows.append(dict(
                    dataset=dataset, target=target, label_usd=label_price, zero_shot_model=zero_shot.method,
                    other_option=f"{other.method} @ {other.budget}", labels_needed=float(other.n_train),
                    break_even_volume=float(setup_cost.min() / (zero_shot.infer_usd - other.infer_usd))))
    return rows
