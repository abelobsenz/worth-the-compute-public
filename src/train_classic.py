"""Step 2a. The three simplest classifiers.

majority   Always predicts the most common class in the training examples. Any real method must
           beat this.
tfidf      TF-IDF + logistic regression. A text becomes a long list of numbers, one per word
           and word pair, that is large when the word is frequent in this text but rare
           overall. Runs on a CPU.
bge        BGE + logistic regression. A pretrained "sentence encoder" (BGE) turns each text into
           768 numbers (an "embedding") so that texts with similar meaning get similar numbers.
           The encoder is not trained here ("frozen"); only the logistic regression is.

Run it with:  python src/train_classic.py
              python src/train_classic.py --datasets banking77 --ks 8 32 --seeds 0
"""
import argparse

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from common import (RESULTS_DIR, Timer, budget_indices, budget_names, is_done, load_dataset, parse_size, save_run,
                    seed_of)
from logreg import fit_logistic_regression, log_probabilities

BGE_MODEL = "BAAI/bge-base-en-v1.5"


def embed_all_splits(dataset, data):
    """Turn every text of every split into its BGE embedding (on the GPU) and save them.

    The embeddings are saved to results/<dataset>/_emb/ because few-shot prompting and the
    analysis reuse them to find similar training examples.

    Returns the embeddings per split and the GPU seconds spent per text, which the cost analysis
    charges to this method.
    """
    from sentence_transformers import SentenceTransformer
    folder = RESULTS_DIR / dataset / "_emb"
    folder.mkdir(parents=True, exist_ok=True)
    encoder = SentenceTransformer(BGE_MODEL, device="cuda")
    embeddings, seconds_per_text = {}, {}
    for split in ["train", "val", "test"]:
        file = folder / f"bge-base_{split}.npy"
        texts = data[split][0]
        with Timer() as timer:
            # normalize_embeddings=True scales every embedding to length 1, so the dot product
            # of two embeddings is their cosine similarity.
            vectors = encoder.encode(texts, batch_size=256, normalize_embeddings=True, convert_to_numpy=True)
        seconds_per_text[split] = timer.sec / len(texts)
        if not file.exists():   # other jobs may be reading the file, so never overwrite it
            np.save(file, vectors.astype(np.float32))
        embeddings[split] = vectors
    print(f"[{dataset}] bge embed sec/example:", seconds_per_text, flush=True)
    return embeddings, seconds_per_text


def main(args):
    for dataset in args.datasets:
        data = load_dataset(dataset)
        train_texts, train_labels = data["train"]
        val_texts, val_labels = data["val"]
        test_texts, test_labels = data["test"]
        n_classes = len(data["label_names"])
        if "bge" in args.methods:
            embeddings, embed_seconds = embed_all_splits(dataset, data)

        for budget in budget_names(dataset, args.ks, args.seeds):
            positions = budget_indices(dataset, budget)   # the training examples of this budget
            labels = train_labels[positions]
            seed = seed_of(budget)

            # ---- majority class ----------------------------------------------------------
            if "majority" in args.methods and not is_done(dataset, "majority", budget):
                most_common = np.bincount(labels, minlength=n_classes).argmax()
                # A score table that always prefers the most common class.
                test_scores = np.full((len(test_labels), n_classes), -1e4, np.float32)
                test_scores[:, most_common] = 0
                val_scores = np.full((len(val_labels), n_classes), -1e4, np.float32)
                val_scores[:, most_common] = 0
                save_run(dataset, "majority", budget, test_labels, test_scores, val_scores,
                         n_train=len(positions), train_sec=0.0, infer_sec=0.0)

            # ---- TF-IDF + logistic regression --------------------------------------------
            if "tfidf" in args.methods and not is_done(dataset, "tfidf", budget):
                texts = [train_texts[i] for i in positions]
                with Timer() as train_timer:
                    # ngram_range=(1, 2): single words and pairs of neighbouring words.
                    # sublinear_tf=True: a word that appears 10 times counts less than 10x.
                    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
                    model, chosen = fit_logistic_regression(texts, labels, seed, steps_before=[vectorizer])
                with Timer() as test_timer:
                    test_scores = log_probabilities(model, test_texts, n_classes)
                val_scores = log_probabilities(model, val_texts, n_classes)
                save_run(dataset, "tfidf", budget, test_labels, test_scores, val_scores,
                         n_train=len(positions), best=chosen, device="cpu",
                         train_sec=train_timer.sec, infer_sec=test_timer.sec, n_test=len(test_labels))

            # ---- BGE embeddings + logistic regression ------------------------------------
            if "bge" in args.methods and not is_done(dataset, "bge", budget):
                with Timer() as train_timer:
                    model, chosen = fit_logistic_regression(embeddings["train"][positions], labels, seed)
                with Timer() as test_timer:
                    test_scores = log_probabilities(model, embeddings["test"], n_classes)
                val_scores = log_probabilities(model, embeddings["val"], n_classes)
                # The GPU time to embed the texts is part of the cost: the budget's examples for
                # training, the test texts for inference.
                save_run(dataset, "bge", budget, test_labels, test_scores, val_scores,
                         n_train=len(positions), best=chosen, device="gpu+cpu",
                         train_sec=train_timer.sec, train_gpu_sec=embed_seconds["train"] * len(positions),
                         infer_sec=test_timer.sec, infer_gpu_sec=embed_seconds["test"] * len(test_labels),
                         n_test=len(test_labels))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["banking77", "goemotions"])
    parser.add_argument("--methods", nargs="+", default=["majority", "tfidf", "bge"])
    parser.add_argument("--ks", nargs="+", type=parse_size, default=[8, 16, 32, 64, 128, 512, "full"],
                        help="labeled examples per class, or 'full'")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    main(parser.parse_args())
