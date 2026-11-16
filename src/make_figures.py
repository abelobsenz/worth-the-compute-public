"""Step 6. Draw the project's four figures.

Needs the score tables of the runs in results/ (see the README for how to get them).

Run it with:  python src/make_figures.py
Output:       outputs/figures/learning_curves.png          score against number of labels
              outputs/figures/zero_shot_and_per_class.png  zero-shot score by model size; per-class F1
              outputs/figures/cost_maps.png                 the cheapest option for every situation
              outputs/figures/routing.png                   score against share of inputs sent to the LLM
"""
import matplotlib

matplotlib.use("Agg")    # draw to files; no window is opened
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from analysis_cost import TARGETS, cheapest_options  # noqa: E402
from analysis_diagnostics import mean_per_class_f1  # noqa: E402
from analysis_router import routing_experiments, summarize_routing  # noqa: E402
from analysis_runs import (DATASET_TITLES, OUTPUT_DIR, label_names, load_labels, load_results,  # noqa: E402
                           n_classes, summary_row)

FIGURE_DIR = OUTPUT_DIR / "figures"
DATASETS = ["banking77", "goemotions"]

# How each method is drawn: (name in the legend, colour, marker, line style).
STYLE = {
    "tfidf": ("TF-IDF + LR", "#008300", "s", "-"),
    "bge": ("BGE + LR", "#1baf7a", "D", "-"),
    "setfit": ("SetFit", "#e87ba4", "v", "-"),
    "modernbert": ("ModernBERT", "#2a78d6", "o", "-"),
    "fs-qwen3-1.7B": ("Few-shot 1.7B", "#eda100", "^", "--"),
    "fs-qwen3-8B": ("Few-shot 8B", "#e34948", "^", "--"),
    "lora-qwen3-1.7B": ("LoRA 1.7B", "#6250d6", "P", "-"),
    "lora-qwen3-8B": ("LoRA 8B", "#eb6834", "P", "-"),
}
GRAY = "#52514e"
ROUTED_COLOUR = "#2a78d6"
LIGHT_GRAY = "#e4e3df"
ZERO_SHOT_SIZES = ["0.6B", "1.7B", "4B", "8B", "14B", "32B"]
BUDGETS = {"banking77": [8, 16, 32, 64, 128, "full"], "goemotions": [8, 16, 32, 64, 128, 512, "full"]}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": LIGHT_GRAY, "grid.linewidth": 0.6, "legend.frameon": False, "lines.linewidth": 1.6,
                     "lines.markersize": 4.5})


# --------------------------------------------------------------------------------------------
# Figure 1: learning curves
# --------------------------------------------------------------------------------------------
def learning_curves(summary):
    """Test macro-F1 (mean over seeds, with a band of +-1 standard deviation) against the number
    of labeled examples per class. The dotted line is zero-shot Qwen3-32B, which uses no labels."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for axis, dataset in zip(axes, DATASETS):
        rows = summary[summary.dataset == dataset]
        k = n_classes(dataset)
        for method, (title, colour, marker, line) in STYLE.items():
            points = rows[rows.method == method].sort_values("n_train")
            if points.empty:
                continue
            labels_per_class = points.n_train / k
            spread = points.f1_sd.fillna(0)      # a single run has no standard deviation
            axis.plot(labels_per_class, points.f1, line, color=colour, marker=marker, label=title, zorder=3)
            axis.fill_between(labels_per_class, points.f1 - spread, points.f1 + spread, color=colour, alpha=0.16, lw=0)
        zero_shot = rows[rows.method == "zs-qwen3-32B"]
        if len(zero_shot):
            axis.axhline(zero_shot.f1.iloc[0], color=GRAY, ls=":", lw=1.4, label="Zero-shot 32B (no labels)")

        # A log scale spreads the small budgets out: each doubling of labels is one step.
        axis.set_xscale("log", base=2)
        all_labels = rows[rows.budget == "full"].n_train.max() / k      # average class size
        # Banking77's "128" budget is almost the full set, so its tick would sit on top of "all".
        ticks = [b for b in BUDGETS[dataset] if b != "full" and not 0.8 < b / all_labels < 1.25]
        axis.set_xticks(ticks + [all_labels])
        axis.set_xticklabels([str(tick) for tick in ticks] + ["all"])
        axis.minorticks_off()
        axis.set_xlabel("labeled examples per class (log scale)")
        axis.set_title(f"{DATASET_TITLES[dataset]} ({k} classes)")
    axes[0].set_ylabel("test macro-F1")
    handles, titles = axes[0].get_legend_handles_labels()
    figure.legend(handles, titles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02))
    figure.tight_layout(rect=(0, 0.12, 1, 1))
    figure.savefig(FIGURE_DIR / "learning_curves.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


# --------------------------------------------------------------------------------------------
# Figure 2: zero-shot score by model size, and F1 per class
# --------------------------------------------------------------------------------------------
def zero_shot_and_per_class(summary, runs, budget=32, methods=("modernbert", "setfit", "lora-qwen3-8B")):
    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 3.4), gridspec_kw=dict(width_ratios=[1, 2.9], wspace=0.18))

    # (a) Does a bigger model classify better without any labels?
    sizes = [float(size[:-1]) for size in ZERO_SHOT_SIZES]      # "8B" -> 8.0
    for dataset, colour, line, marker in [("banking77", "black", "-", "o"), ("goemotions", GRAY, "--", "s")]:
        f1 = [summary_row(summary, dataset, f"zs-qwen3-{size}", 0).f1 for size in ZERO_SHOT_SIZES]
        left.plot(sizes, f1, line, color=colour, marker=marker, label=DATASET_TITLES[dataset])
    left.set_xscale("log")
    left.set_xticks(sizes)
    left.set_xticklabels([size[:-1] for size in ZERO_SHOT_SIZES])
    left.minorticks_off()
    left.set_ylim(0, 0.85)
    left.set_xlabel("Qwen3 size (billions of weights)")
    left.set_ylabel("zero-shot macro-F1")
    left.set_title("(a) Zero-shot score by model size")
    left.legend(loc="upper left")

    # (b) F1 of every GoEmotions class, most common class on the left. The grey bar spans the
    # three methods, so a long bar means the methods differ a lot on that class.
    dataset = "goemotions"
    names = label_names(dataset)
    class_sizes = np.bincount(load_labels(dataset, "train"), minlength=len(names))
    order = np.argsort(-class_sizes)
    f1 = {method: mean_per_class_f1(runs, dataset, method, budget)[order] for method in methods}
    x = np.arange(len(names))
    lowest, highest = np.min(list(f1.values()), 0), np.max(list(f1.values()), 0)
    right.vlines(x, lowest, highest, color=LIGHT_GRAY, lw=3, zorder=1)
    for method in methods:
        title, colour, marker, _ = STYLE[method]
        right.plot(x, f1[method], linestyle="none", marker=marker, color=colour, label=title, zorder=3)
    right.set_xticks(x)
    right.set_xticklabels([names[i] for i in order], rotation=55, ha="right", rotation_mode="anchor", fontsize=8)
    right.set_xlim(-0.7, len(names) - 0.3)
    right.set_ylim(0, 1)
    right.grid(axis="x", visible=False)
    right.set_ylabel("test F1")
    right.set_title(f"(b) GoEmotions: F1 per class at {budget} labels per class (most common class first)")
    right.legend(loc="upper right", ncol=3)
    figure.savefig(FIGURE_DIR / "zero_shot_and_per_class.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


# --------------------------------------------------------------------------------------------
# Figure 3: cost maps
# --------------------------------------------------------------------------------------------
def option_title(method, budget):
    if method.startswith("zs-"):
        return f"Zero-shot {method.split('-')[-1]}\nno labels"
    return f"{STYLE[method][0]}\n{'all labels' if budget == 'full' else budget + '/class'}"


def draw_cost_map(axis, summary, dataset, target, volumes, label_prices):
    """Colour every (number of predictions, price per label) by its cheapest option."""
    cheapest = cheapest_options(summary, dataset, target, volumes, label_prices)
    cheapest["option"] = cheapest.method + "@" + cheapest.budget
    options = list(dict.fromkeys(cheapest.option))             # the distinct options, in order
    number_of = {option: i for i, option in enumerate(options)}
    # A grid of option numbers: one row per label price, one column per volume.
    grid = cheapest.option.map(number_of).to_numpy().reshape(len(volumes), len(label_prices)).T
    colours = [GRAY if option.startswith("zs-") else STYLE[option.split("@")[0]][1] for option in options]
    extent = (np.log10(volumes[0]), np.log10(volumes[-1]), np.log10(label_prices[0]), np.log10(label_prices[-1]))
    axis.imshow(grid, origin="lower", aspect="auto", cmap=ListedColormap(colours), vmin=-0.5, vmax=len(options) - 0.5,
                extent=extent, interpolation="nearest")
    # Write each option's name in the middle of its region (if the region is big enough). The
    # middle is kept away from the edges of the map so that the text is not cut off.
    x, y = np.meshgrid(np.log10(volumes), np.log10(label_prices))
    left, right, bottom, top = extent
    for option, i in number_of.items():
        region = grid == i
        if region.mean() >= 0.05:
            text_x = np.clip(x[region].mean(), left + 0.3 * (right - left), left + 0.7 * (right - left))
            text_y = np.clip(y[region].mean(), bottom + 0.12 * (top - bottom), bottom + 0.88 * (top - bottom))
            axis.text(text_x, text_y, option_title(*option.split("@")), ha="center", va="center", fontsize=7.5,
                      color="white", fontweight="bold")
    axis.grid(False)
    axis.set_xticks([3, 5, 7, 9])
    axis.set_xticklabels(["1 thousand", "100 thousand", "10 million", "1 billion"], fontsize=6.5)
    axis.set_yticks(np.log10([0.01, 0.1, 1, 10]))
    axis.set_yticklabels(["1¢", "10¢", "$1", "$10"])


def cost_maps(summary):
    """One map per target score. Both axes are on a log scale."""
    volumes = np.logspace(3, 10, 57)        # 1 thousand to 10 billion predictions
    label_prices = np.logspace(-2, 1, 25)   # 1 cent to 10 dollars per label
    panels = [(dataset, target) for dataset in DATASETS for target in TARGETS[dataset]]
    figure, axes = plt.subplots(1, len(panels), figsize=(17, 3.6), gridspec_kw=dict(wspace=0.3))
    for axis, (dataset, target) in zip(axes, panels):
        draw_cost_map(axis, summary, dataset, target, volumes, label_prices)
        axis.set_title(f"{DATASET_TITLES[dataset]}, macro-F1 of at least {target:.2f}", fontsize=9)
        axis.set_xlabel("predictions over the system's lifetime")
    axes[0].set_ylabel("price of one label")
    figure.savefig(FIGURE_DIR / "cost_maps.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


# --------------------------------------------------------------------------------------------
# Figure 4: routing
# --------------------------------------------------------------------------------------------
def routing(runs, summary):
    """Test macro-F1 as more inputs are sent to the LLM, for the best cheap model at two budgets."""
    experiments = routing_experiments(runs)
    results = summarize_routing(experiments, summary)
    keys = [f"{dataset}/{budget}/best" for dataset in DATASETS for budget in (32, "full")]
    percent_sent = 100 * np.linspace(0, 1, 41)
    figure, axes = plt.subplots(1, len(keys), figsize=(15, 3.4), gridspec_kw=dict(wspace=0.3))
    for axis, key in zip(axes, keys):
        dataset, budget, _ = key.split("/")
        cheap = experiments[key]["cheap"]
        seeds = list(experiments[key]["seeds"].values())
        curves_to_draw = [("oracle", "black", "--", "oracle (knows the answers)"),
                          ("random", GRAY, ":", "random order"),
                          ("msp_ts", ROUTED_COLOUR, "-", "routed, least confident first")]
        for signal, colour, line, title in curves_to_draw:
            curves = 100 * np.stack([seed["curves"][signal] for seed in seeds])     # one row per seed
            axis.plot(percent_sent, curves.mean(0), line, color=colour, label=title)
            if signal == "msp_ts" and len(seeds) > 1:
                spread = curves.std(0, ddof=1)
                axis.fill_between(percent_sent, curves.mean(0) - spread, curves.mean(0) + spread, color=colour,
                                  alpha=0.2, lw=0)
        # The point actually used: the threshold chosen on validation data, with its spread over seeds.
        result = results[key]
        axis.errorbar(100 * result["fraction"][0], 100 * result["routed_f1"][0], xerr=100 * result["fraction"][1],
                      yerr=100 * result["routed_f1"][1], fmt="D", color="black", markerfacecolor="white", capsize=2,
                      zorder=5, label="threshold chosen on validation")
        budget_text = "all labels" if budget == "full" else f"{budget} labels per class"
        axis.set_title(f"{DATASET_TITLES[dataset]}, {budget_text}\n{STYLE[cheap][0]} in front of LoRA 8B", fontsize=9)
        axis.set_xlabel("% of inputs sent to the LLM")
        axis.set_xlim(0, 100)
        axis.set_xticks([0, 25, 50, 75, 100])
    axes[0].set_ylabel("test macro-F1 (%)")
    handles, titles = axes[0].get_legend_handles_labels()
    figure.legend(handles, titles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    figure.savefig(FIGURE_DIR / "routing.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    all_runs, runs, summary = load_results()
    learning_curves(summary)
    zero_shot_and_per_class(summary, runs)
    cost_maps(summary)
    routing(runs, summary)
    print("figures written to", FIGURE_DIR)
