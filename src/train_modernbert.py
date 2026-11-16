"""Step 2c. Fine-tune ModernBERT, a small pretrained "encoder" network (149 million weights).

Fine-tuning means: put a new output layer with one output per class on top of the pretrained
network, then keep training ALL of its weights on our labeled examples. This is the standard way
of building a text classifier.

The learning rate (how big each training step is) matters a lot with few examples, so we train
with three learning rates. The analysis later keeps, for every dataset and budget, the one with
the best validation score. Results go to the folders
    modernbert             learning rate 5e-5
    modernbert-lr2e-05     learning rate 2e-5
    modernbert-lr0.0001    learning rate 1e-4

Run it with:  python src/train_modernbert.py --lr 5e-5
              python src/train_modernbert.py --lr 1e-4 --datasets banking77 --ks 8 --seeds 0
"""
import argparse
import math

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import Timer, budget_indices, budget_names, is_done, load_dataset, parse_size, save_run, seed_of

MODERNBERT_MODEL = "answerdotai/ModernBERT-base"
MAX_TOKENS = 128    # longer texts are cut off here; only a handful of comments are that long


def number_of_epochs(n_examples, batch_size, target_steps=600, fewest=3, most=20):
    """How many times to go through the training examples.

    A small budget has few batches per pass, so it needs more passes to get enough training
    steps. We aim for about 600 steps, but never fewer than 3 passes or more than 20.
    """
    steps_per_epoch = math.ceil(n_examples / batch_size)
    return min(most, max(fewest, math.ceil(target_steps / steps_per_epoch)))


@torch.no_grad()    # we only predict here, so PyTorch does not need to track gradients
def predict_scores(model, tokenizer, texts, batch_size=256):
    """The model's score for every class, for every text (one row per text)."""
    model.eval()
    # Texts in a batch are padded to the longest one. Sorting by length puts texts of similar
    # length together, which wastes less computation on padding.
    order = np.argsort([len(text) for text in texts])
    batches = []
    for start in range(0, len(texts), batch_size):
        batch = [texts[i] for i in order[start:start + batch_size]]
        tokens = tokenizer(batch, padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt").to("cuda")
        # autocast runs the network in 16-bit numbers, which is faster and uses less memory
        with torch.autocast("cuda", dtype=torch.bfloat16):
            batches.append(model(**tokens).logits.float().cpu())
    sorted_scores = torch.cat(batches).numpy()
    scores = np.empty_like(sorted_scores)
    scores[order] = sorted_scores    # put the rows back in the original order
    return scores


def run_modernbert(dataset, data, budget, learning_rate, method, batch_size=32):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    seed = seed_of(budget)
    torch.manual_seed(seed)   # fixes the random starting weights of the new output layer
    positions = budget_indices(dataset, budget)
    train_texts = [data["train"][0][i] for i in positions]
    train_labels = data["train"][1][positions]
    n_classes = len(data["label_names"])

    tokenizer = AutoTokenizer.from_pretrained(MODERNBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODERNBERT_MODEL, num_labels=n_classes).cuda()

    epochs = number_of_epochs(len(train_texts), batch_size)
    shuffle_generator = torch.Generator().manual_seed(seed)   # fixes the order of the examples
    loader = DataLoader(list(zip(train_texts, train_labels)), batch_size=batch_size, shuffle=True,
                        generator=shuffle_generator,
                        collate_fn=lambda batch: ([text for text, _ in batch],
                                                  torch.tensor([label for _, label in batch])))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = epochs * len(loader)
    # The learning rate rises from 0 during the first 10% of the steps ("warm-up"), then falls
    # in a straight line back to 0.
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    with Timer() as train_timer:
        model.train()
        for _ in range(epochs):
            for texts, labels in loader:
                tokens = tokenizer(texts, padding=True, truncation=True, max_length=MAX_TOKENS,
                                   return_tensors="pt").to("cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    # Cross-entropy is small when the model gives the correct class a high score.
                    loss = torch.nn.functional.cross_entropy(model(**tokens).logits.float(), labels.cuda())
                loss.backward()                                            # how to change each weight
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # no single step may be huge
                optimizer.step()                                          # change the weights
                scheduler.step()                                          # update the learning rate
                optimizer.zero_grad(set_to_none=True)

    with Timer() as test_timer:
        test_scores = predict_scores(model, tokenizer, data["test"][0])
    val_scores = predict_scores(model, tokenizer, data["val"][0])
    save_run(dataset, method, budget, data["test"][1], test_scores, val_scores,
             n_train=len(positions), epochs=epochs, lr=learning_rate, bs=batch_size,
             train_gpu_sec=train_timer.sec, infer_gpu_sec=test_timer.sec, n_test=len(test_scores))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate: 2e-5, 5e-5 or 1e-4")
    parser.add_argument("--datasets", nargs="+", default=["banking77", "goemotions"])
    parser.add_argument("--ks", nargs="+", type=parse_size, default=[8, 16, 32, 64, 128, 512, "full"],
                        help="labeled examples per class, or 'full'")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    # Each learning rate has its own results folder (see the top of this file).
    method = "modernbert" if args.lr == 5e-5 else f"modernbert-lr{args.lr:g}"
    for dataset in args.datasets:
        data = load_dataset(dataset)
        for budget in budget_names(dataset, args.ks, args.seeds):
            if not is_done(dataset, method, budget):
                run_modernbert(dataset, data, budget, args.lr, method)
                torch.cuda.empty_cache()   # release GPU memory before the next run
