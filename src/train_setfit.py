"""Step 2b. SetFit: a method built for learning from very few labeled examples.

SetFit starts from the same pretrained sentence encoder as the "bge" method, but first adapts the
encoder to the task ("contrastive fine-tuning"):

  1. Make pairs of training texts. A pair with the same label is a positive pair; a pair with
     different labels is a negative pair.
  2. Train the encoder so that the embeddings of positive pairs move closer together and those of
     negative pairs move apart. After this, texts of the same class sit near each other.
  3. Embed the training texts with the adapted encoder and fit a logistic regression on top,
     exactly as the "bge" method does.

Because steps 1-2 are the only difference from "bge", comparing the two methods shows what the
contrastive fine-tuning is worth.

Run it with:  python src/train_setfit.py
              python src/train_setfit.py --datasets banking77 --ks 8 --seeds 0
"""
import argparse
import shutil
import tempfile

import numpy as np
import torch

from common import Timer, budget_indices, budget_names, is_done, load_dataset, parse_size, save_run, seed_of
from logreg import fit_logistic_regression, log_probabilities

BGE_MODEL = "BAAI/bge-base-en-v1.5"


def run_setfit(dataset, data, budget):
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments

    seed = seed_of(budget)
    positions = budget_indices(dataset, budget)
    n_classes = len(data["label_names"])
    train_texts, train_labels = data["train"]
    train_set = Dataset.from_dict({"text": [train_texts[i] for i in positions],
                                   "label": train_labels[positions].tolist()})

    model = SetFitModel.from_pretrained(BGE_MODEL, head_params={"max_iter": 2000})

    # SetFit writes checkpoints while it trains. Each run gets its own temporary folder so that
    # jobs running at the same time do not write to the same place.
    checkpoint_folder = tempfile.mkdtemp(prefix=f"setfit_{dataset}_{budget}_")
    # num_iterations=20: for every training text, 20 positive and 20 negative pairs (40 pairs).
    # num_epochs=1: go through those pairs once.
    training_args = TrainingArguments(batch_size=32, num_epochs=1, num_iterations=20, seed=seed,
                                      show_progress_bar=False, output_dir=checkpoint_folder)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_set)
    encoder = model.model_body     # the sentence encoder inside the SetFit model

    def embed(texts):
        return encoder.encode(texts, batch_size=256, normalize_embeddings=True, convert_to_numpy=True)

    with Timer() as train_timer:
        trainer.train()            # steps 1-2: contrastive fine-tuning of the encoder
        encoder.to("cuda")
        # step 3: the same cross-validated logistic regression that the "bge" method uses
        classifier, chosen = fit_logistic_regression(embed(train_set["text"]), np.array(train_set["label"]), seed)
    shutil.rmtree(checkpoint_folder, ignore_errors=True)

    with Timer() as test_timer:
        test_scores = log_probabilities(classifier, embed(data["test"][0]), n_classes)
    val_scores = log_probabilities(classifier, embed(data["val"][0]), n_classes)
    save_run(dataset, "setfit", budget, data["test"][1], test_scores, val_scores,
             n_train=len(positions), num_iterations=20, best=chosen,
             train_gpu_sec=train_timer.sec, infer_gpu_sec=test_timer.sec, n_test=len(test_scores))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["banking77", "goemotions"])
    parser.add_argument("--ks", nargs="+", type=parse_size, default=[8, 16, 32, 64, 128],
                        help="labeled examples per class")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    for dataset in args.datasets:
        data = load_dataset(dataset)
        for budget in budget_names(dataset, args.ks, args.seeds):
            if not is_done(dataset, "setfit", budget):
                run_setfit(dataset, data, budget)
                torch.cuda.empty_cache()   # release GPU memory before the next run
