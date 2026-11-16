"""Step 2d. Classify by prompting a Qwen3 model, without changing its weights.

zeroshot   The prompt holds the task, the list of label names and the text. The model has seen
           no labeled example. Results go to results/<dataset>/zs-qwen3-<size>/zeroshot/.
fewshot    The same prompt plus 10 labeled examples. For every text we pick the 10 training
           examples (from the label budget) that are most similar to it, measured with the BGE
           embeddings that train_classic.py saved. Results go to
           results/<dataset>/fs-qwen3-<size>/<budget>/.

Few-shot prompting therefore needs train_classic.py to have run first.

Run it with:  python src/run_llm_prompting.py zeroshot --size 8B --val
              python src/run_llm_prompting.py fewshot --size 8B --ks 8 32 --seeds 0 --val
"""
import argparse

import numpy as np
import torch

from common import RESULTS_DIR, Timer, budget_indices, budget_names, is_done, load_dataset, parse_size, save_run
from llm_common import (MAX_TOKENS_PER_BATCH, LabelScorer, build_user_message, load_model, tokenize_labels,
                        tokenize_prompt)

N_EXAMPLES_IN_PROMPT = 10


def find_similar_examples(dataset, data, positions, n_examples=N_EXAMPLES_IN_PROMPT):
    """For every validation and test text, the most similar labeled examples of the budget.

    positions   the training examples this budget may use (the "pool")
    Returns     {"val": [...], "test": [...]}; each entry is the list of (text, label) examples
                for one input text.
    """
    embeddings = {split: np.load(RESULTS_DIR / dataset / "_emb" / f"bge-base_{split}.npy")
                  for split in ["train", "val", "test"]}
    pool = embeddings["train"][positions]
    train_texts, train_labels = data["train"]
    examples = {}
    for split in ["val", "test"]:
        # The embeddings have length 1, so this matrix product is the cosine similarity between
        # every input text (rows) and every pool example (columns).
        similarity = embeddings[split] @ pool.T
        # The n most similar pool examples per text. [:, ::-1] reverses their order so that the
        # MOST similar example comes last in the prompt, right before the text to classify.
        nearest = np.argsort(-similarity, axis=1)[:, :n_examples][:, ::-1]
        examples[split] = [[(train_texts[positions[j]], int(train_labels[positions[j]])) for j in row]
                           for row in nearest]
    return examples


def run_prompting(dataset, data, size, budget, tokenizer, model, splits, n_gpus=1):
    """Score the given splits with zero-shot (budget=None) or few-shot (budget="k8_s0") prompts."""
    label_names = data["label_names"]
    method = f"{'fs' if budget else 'zs'}-qwen3-{size}"
    run = budget or "zeroshot"
    if is_done(dataset, method, run):
        return

    examples = find_similar_examples(dataset, data, budget_indices(dataset, budget)) if budget else None
    scorer = LabelScorer(model, tokenize_labels(tokenizer, label_names))
    scores, seconds = {}, {}
    for split in splits:
        texts = data[split][0]
        with Timer() as timer:
            prompts = []
            for i, text in enumerate(texts):
                examples_for_text = examples[split][i] if examples else None
                message = build_user_message(dataset, label_names, text, examples_for_text)
                prompts.append(tokenize_prompt(tokenizer, dataset, message))
            scores[split] = scorer.score(prompts, max_tokens=MAX_TOKENS_PER_BATCH[size])
        seconds[split] = timer.sec
        mean_length = np.mean([len(prompt) for prompt in prompts])
        print(f"  {method} {run} {split}: {timer.sec:.0f}s, mean prompt {mean_length:.0f} tok", flush=True)

    # When the model is spread over several GPUs, all of them are busy, so the cost is the
    # elapsed time multiplied by the number of GPUs.
    save_run(dataset, method, run, data["test"][1], scores["test"], scores.get("val"),
             n_train=len(budget_indices(dataset, budget)) if budget else 0,
             shots=N_EXAMPLES_IN_PROMPT if budget else 0,
             infer_gpu_sec=seconds["test"] * n_gpus, n_test=len(data["test"][1]), gpus=n_gpus,
             val_gpu_sec=seconds.get("val", 0) * n_gpus)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["zeroshot", "fewshot"])
    parser.add_argument("--size", default="1.7B", help="Qwen3 size: 0.6B, 1.7B, 4B, 8B, 14B or 32B")
    parser.add_argument("--datasets", nargs="+", default=["banking77", "goemotions"])
    parser.add_argument("--ks", nargs="+", type=parse_size, default=[8, 32, 128, "full"],
                        help="few-shot only: labeled examples per class, or 'full'")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0], help="few-shot only")
    parser.add_argument("--val", action="store_true",
                        help="also score the validation split (needed for calibration and routing)")
    args = parser.parse_args()

    n_gpus = torch.cuda.device_count()
    print("GPUs:", n_gpus, torch.cuda.get_device_name(0), flush=True)
    tokenizer, model = load_model(args.size, n_gpus=n_gpus)
    splits = ["test", "val"] if args.val else ["test"]
    for dataset in args.datasets:
        data = load_dataset(dataset)
        if args.mode == "zeroshot":
            run_prompting(dataset, data, args.size, None, tokenizer, model, splits, n_gpus)
        else:
            for budget in budget_names(dataset, args.ks, args.seeds):
                run_prompting(dataset, data, args.size, budget, tokenizer, model, splits, n_gpus)
