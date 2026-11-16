"""Step 1. Turn the raw dataset files into clean train / validation / test splits and choose the
"label budgets" (which training examples each experiment may use).

Input   raw/banking_train.csv, raw/banking_test.csv           Banking77
        raw/goemo_train.tsv, goemo_dev.tsv, goemo_test.tsv    GoEmotions
        raw/goemo_emotions.txt                                the 28 emotion names

Output  data/<dataset>/train.jsonl, val.jsonl, test.jsonl     one example per line
        data/<dataset>/labels.json                            class number -> name
        data/<dataset>/budgets.json                           budget name -> training positions

Run it with:  python src/prepare_data.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "raw"
DATA_DIR = PROJECT_DIR / "data"

SEEDS = [0, 1, 2, 3, 4]                # five random draws of every budget
BUDGET_SIZES = [8, 16, 32, 64, 128, 512]  # labeled examples per class ("full" is added below)


def readable(label):
    """"card_arrival" -> "card arrival". The LLM reads label names, so they should be words."""
    return label.lower().replace("_", " ")


def write_split(dataset, split, table):
    folder = DATA_DIR / dataset
    folder.mkdir(parents=True, exist_ok=True)
    table[["text", "label"]].to_json(folder / f"{split}.jsonl", orient="records", lines=True, force_ascii=False)


# --------------------------------------------------------------------------------------------
# Banking77: 77 customer-service intents. It comes with a train and a test file only.
# --------------------------------------------------------------------------------------------
def prepare_banking77():
    train = pd.read_csv(RAW_DIR / "banking_train.csv")   # columns: text, category
    test = pd.read_csv(RAW_DIR / "banking_test.csv")

    # Give every intent a number, in alphabetical order.
    categories = sorted(train["category"].unique())
    number_of = {category: i for i, category in enumerate(categories)}
    train["label"] = train["category"].map(number_of)
    test["label"] = test["category"].map(number_of)

    # There is no validation split, so we hold out 10% of the training examples of every intent.
    # ("Stratified": every intent keeps the same share.) The fixed seed makes this repeatable.
    rng = np.random.default_rng(1234)
    val_positions = []
    for c in range(len(categories)):
        positions = np.where(train["label"].to_numpy() == c)[0]
        chosen = rng.choice(positions, size=round(0.1 * len(positions)), replace=False)
        val_positions += chosen.tolist()
    val = train.iloc[sorted(val_positions)].reset_index(drop=True)
    train = train.drop(index=val_positions).reset_index(drop=True)

    for split, table in [("train", train), ("val", val), ("test", test)]:
        write_split("banking77", split, table)
    labels = {"names": [readable(c) for c in categories], "raw": categories}
    json.dump(labels, open(DATA_DIR / "banking77" / "labels.json", "w"), indent=1)
    return train, val, test, len(categories)


# --------------------------------------------------------------------------------------------
# GoEmotions: Reddit comments labeled with 27 emotions or "neutral". It comes with official
# train / dev / test files; the dev file is our validation split.
# --------------------------------------------------------------------------------------------
def prepare_goemotions():
    names = open(RAW_DIR / "goemo_emotions.txt").read().split()
    columns = ["text", "labels", "id"]
    splits = {}
    for split in ["train", "dev", "test"]:
        # quoting=3 tells pandas not to treat quote characters in the comments specially
        table = pd.read_csv(RAW_DIR / f"goemo_{split}.tsv", sep="\t", names=columns, quoting=3)
        # A comment can carry several emotions ("3,17"). We keep the comments with exactly one,
        # so that every text has a single correct answer, as in Banking77.
        table = table[~table["labels"].astype(str).str.contains(",")].copy()
        table["label"] = table["labels"].astype(int)
        splits[split] = table

    # Remove training comments whose text also appears in dev or test (the model would otherwise
    # be tested on something it was trained on), then remove repeated training comments.
    held_out_texts = set(splits["dev"]["text"]) | set(splits["test"]["text"])
    train = splits["train"]
    n_single_label = len(train)
    train = train[~train["text"].isin(held_out_texts)]
    n_without_overlap = len(train)
    train = train.drop_duplicates("text").reset_index(drop=True)
    print(f"goemotions train: {n_single_label} single-label, "
          f"-{n_single_label - n_without_overlap} overlapping dev/test, "
          f"-{n_without_overlap - len(train)} duplicates -> {len(train)}")

    for split, table in [("train", train), ("val", splits["dev"]), ("test", splits["test"])]:
        write_split("goemotions", split, table.reset_index(drop=True))
    json.dump({"names": names, "raw": names}, open(DATA_DIR / "goemotions" / "labels.json", "w"), indent=1)
    return train, splits["dev"], splits["test"], len(names)


# --------------------------------------------------------------------------------------------
# Label budgets
# --------------------------------------------------------------------------------------------
def shuffled_positions_per_class(labels, n_classes, seed):
    """For every class, its training positions in a random order that depends only on the seed."""
    rng = np.random.default_rng(seed)
    return {c: rng.permutation(np.where(labels == c)[0]).tolist() for c in range(n_classes)}


def make_budgets(dataset, train, n_classes):
    """Choose the training examples of every budget and write them to budgets.json.

    The budget "k32_s1" takes the first 32 examples of every class from the shuffle made with
    seed 1 (or all of them, if the class has fewer than 32). Because a budget always takes the
    first k of the same shuffle, the 8 examples of "k8_s1" are also among the 16 of "k16_s1",
    and so on: with the same seed, a larger budget only ever adds examples.
    """
    labels = train["label"].to_numpy()
    class_sizes = np.bincount(labels, minlength=n_classes)
    budgets = {}
    for seed in SEEDS:
        shuffled = shuffled_positions_per_class(labels, n_classes, seed)
        for k in BUDGET_SIZES:
            if k > class_sizes.max():
                continue    # no class has that many examples; the budget would equal "full"
            positions = sorted(i for c in range(n_classes) for i in shuffled[c][:k])
            budgets[f"k{k}_s{seed}"] = positions
    # "full" always means every training example. The seed then only changes the randomness
    # inside training (starting weights, order of the examples).
    for seed in SEEDS:
        budgets[f"full_s{seed}"] = list(range(len(train)))
    json.dump(budgets, open(DATA_DIR / dataset / "budgets.json", "w"))
    return {name: len(positions) for name, positions in budgets.items()}


if __name__ == "__main__":
    for dataset, prepare in [("banking77", prepare_banking77), ("goemotions", prepare_goemotions)]:
        train, val, test, n_classes = prepare()
        sizes = make_budgets(dataset, train, n_classes)
        class_sizes = np.bincount(train["label"], minlength=n_classes)
        print(f"{dataset}: train {len(train)} val {len(val)} test {len(test)} classes {n_classes} "
              f"per-class min {class_sizes.min()} median {int(np.median(class_sizes))} max {class_sizes.max()}")
        print("  budget sizes:", {name: n for name, n in sizes.items() if name.endswith("s0")})
