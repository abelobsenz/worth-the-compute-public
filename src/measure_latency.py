"""Step 3. Measure latency: how long one prediction takes when texts arrive one at a time.

The training scripts time predictions in large batches, which measures throughput (predictions
per second). A live service often has to answer a single request, so here every classifier
handles 200 test texts one by one and we record the median and the 95th-percentile time.

How long a prediction takes depends on the size of the network and the length of the input, not
on what the weights have learned. So we do not need the trained models: ModernBERT gets a fresh
output layer, and the LoRA rows use the plain Qwen3 model with the short prompt that fine-tuned
models receive.

Needs train_classic.py to have run first (few-shot prompting uses its saved embeddings).
Output: results/latency.json

Run it with:  python src/measure_latency.py
"""
import json
import time

import numpy as np
import torch

from common import RESULTS_DIR, load_dataset
from llm_common import LabelScorer, build_user_message, load_model, tokenize_labels, tokenize_prompt

N_TIMED = 200     # predictions that are timed
N_WARMUP = 10     # predictions made first and not timed (the first calls are always slower)


def time_one_by_one(predict, texts):
    """Call predict(text) for each text and return the median, 95th-percentile and mean time in ms."""
    for text in texts[:N_WARMUP]:
        predict(text)
    seconds = []
    for text in texts[N_WARMUP:N_WARMUP + N_TIMED]:
        torch.cuda.synchronize()          # wait for the GPU before starting the clock ...
        start = time.perf_counter()
        predict(text)
        torch.cuda.synchronize()          # ... and before stopping it
        seconds.append(time.perf_counter() - start)
    return {"p50_ms": 1e3 * float(np.median(seconds)), "p95_ms": 1e3 * float(np.percentile(seconds, 95)),
            "mean_ms": 1e3 * float(np.mean(seconds))}


def main():
    results = {}
    for dataset in ["banking77", "goemotions"]:
        data = load_dataset(dataset)
        label_names = data["label_names"]
        train_texts, train_labels = data["train"]
        test_texts = data["test"][0]
        rng = np.random.default_rng(0)
        texts = [test_texts[i] for i in rng.choice(len(test_texts), N_TIMED + N_WARMUP, replace=False)]
        latency = {}

        # ---- TF-IDF + logistic regression (CPU) -----------------------------------------
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        tfidf_model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                                    LogisticRegression(C=10, max_iter=2000)).fit(train_texts, train_labels)
        latency["tfidf"] = time_one_by_one(lambda text: tfidf_model.predict_proba([text]), texts)

        # ---- BGE + logistic regression: embed the text, then classify ---------------------
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda")
        train_embeddings = encoder.encode(train_texts[:2000], normalize_embeddings=True)
        classifier = LogisticRegression(C=10, max_iter=2000).fit(train_embeddings, train_labels[:2000])
        latency["bge"] = time_one_by_one(
            lambda text: classifier.predict_proba(encoder.encode([text], normalize_embeddings=True)), texts)
        latency["setfit"] = latency["bge"]   # SetFit predicts with the same encoder + classifier
        del encoder

        # ---- ModernBERT ---------------------------------------------------------------------
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        bert_tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        bert = AutoModelForSequenceClassification.from_pretrained("answerdotai/ModernBERT-base",
                                                                  num_labels=len(label_names)).cuda().eval()

        @torch.no_grad()
        def bert_predict(text):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return bert(**bert_tokenizer([text], return_tensors="pt").to("cuda")).logits.float().cpu()
        latency["modernbert"] = time_one_by_one(bert_predict, texts)
        del bert

        # ---- Qwen3: zero-shot prompt, short (LoRA) prompt and few-shot prompt ---------------
        pool = np.load(RESULTS_DIR / dataset / "_emb" / "bge-base_train.npy")
        test_embeddings = np.load(RESULTS_DIR / dataset / "_emb" / "bge-base_test.npy")
        position_of = {text: i for i, text in enumerate(test_texts)}
        for size in ["0.6B", "1.7B", "4B", "8B"]:
            tokenizer, model = load_model(size)
            scorer = LabelScorer(model, tokenize_labels(tokenizer, label_names))

            def zero_shot(text):
                return scorer.score([tokenize_prompt(tokenizer, dataset,
                                                     build_user_message(dataset, label_names, text))])

            def short_prompt(text):
                message = build_user_message(dataset, label_names, text, include_label_list=False)
                return scorer.score([tokenize_prompt(tokenizer, dataset, message)])

            def few_shot(text):
                # find the 10 most similar training examples, most similar last, then score
                nearest = np.argsort(-(pool @ test_embeddings[position_of[text]]))[:10][::-1]
                examples = [(train_texts[j], int(train_labels[j])) for j in nearest]
                message = build_user_message(dataset, label_names, text, examples)
                return scorer.score([tokenize_prompt(tokenizer, dataset, message)])

            latency[f"zs-qwen3-{size}"] = time_one_by_one(zero_shot, texts)
            latency[f"lora-qwen3-{size}"] = time_one_by_one(short_prompt, texts)
            latency[f"fs-qwen3-{size}"] = time_one_by_one(few_shot, texts)
            print(dataset, size, {name: round(value["p50_ms"], 1) for name, value in latency.items()}, flush=True)
            del model, scorer
            torch.cuda.empty_cache()
        results[dataset] = latency

    results["gpu"] = torch.cuda.get_device_name(0)
    json.dump(results, open(RESULTS_DIR / "latency.json", "w"), indent=1)


if __name__ == "__main__":
    main()
