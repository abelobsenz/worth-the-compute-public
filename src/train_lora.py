"""Step 2e. Fine-tune a Qwen3 model on our labeled examples with LoRA, then classify with it.

Why LoRA
--------
Qwen3-8B has 8 billion weights. Updating all of them needs far more memory than our GPUs have
and would give a separate 16 GB model for every experiment. LoRA ("low-rank adaptation") freezes
every original weight and adds a small trainable correction to each weight matrix W:

        W  is replaced by  W + B x A

A and B are two narrow matrices (16 rows / columns, the "rank"), so together they hold far fewer
numbers than W. Only A and B are trained: about 17 million numbers for Qwen3-1.7B and 44 million
for Qwen3-8B, roughly 1% and 0.5% of the model. The trained A and B are saved as an "adapter".

What the model is trained to do
-------------------------------
Each training example is a short prompt (a question and the text, without the label list)
followed by the correct label name and the end-of-reply token. The model is trained to predict
the label tokens. After training, classification works exactly as for the prompted models: every
label is scored with LabelScorer and the best one wins.

Results go to results/<dataset>/lora-qwen3-<size>/<budget>/.

Run it with:  python src/train_lora.py --size 8B
              python src/train_lora.py --size 1.7B --datasets banking77 --ks 8 --seeds 0
"""
import argparse
import json
import math

import numpy as np
import torch

from common import RESULTS_DIR, Timer, budget_indices, budget_names, is_done, load_dataset, parse_size, save_run, seed_of
from llm_common import (MAX_TOKENS_PER_BATCH, MODEL_NAMES, LabelScorer, build_user_message, load_model,
                        tokenize_labels, tokenize_prompt)

MAX_TEXT_TOKENS = 128    # training texts longer than this are cut off (a handful of comments)


def train_adapter(dataset, data, size, budget, adapter_folder, seed, learning_rate=2e-4, batch_size=16, rank=16):
    """Train a LoRA adapter on the budget's examples and save it. Returns facts about the training."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    torch.manual_seed(seed)   # fixes the random starting values of the adapter
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAMES[size])
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAMES[size], dtype=torch.bfloat16,
                                                 attn_implementation="sdpa", device_map={"": 0})
    # Gradient checkpointing saves GPU memory during training by recomputing some intermediate
    # values instead of storing them. The next line is needed for it to work with frozen inputs.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    # Add the trainable A and B matrices to every attention and feed-forward weight matrix.
    #   lora_alpha    scales the correction B x A (2 x rank is a common choice)
    #   lora_dropout  randomly switches off 5% of the adapter inputs in training, against overfitting
    config = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, config)   # from here on only A and B are trainable
    # The frozen weights stay in 16-bit numbers; the small trainable ones use 32 bits, which
    # keeps their tiny updates accurate.
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()

    label_names = data["label_names"]
    label_tokens = tokenize_labels(tokenizer, label_names)
    positions = budget_indices(dataset, budget)
    train_texts, train_labels = data["train"]

    def cut_off(text):
        tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
        return text if len(tokens) <= MAX_TEXT_TOKENS else tokenizer.decode(tokens[:MAX_TEXT_TOKENS])

    # Every training example: (prompt tokens, tokens of the correct label).
    examples = []
    for i in positions:
        message = build_user_message(dataset, label_names, cut_off(train_texts[i]), include_label_list=False)
        examples.append((tokenize_prompt(tokenizer, dataset, message), label_tokens[train_labels[i]]))

    # Go through the examples at least twice and at most 10 times, aiming for about 300 steps.
    steps_per_epoch = math.ceil(len(examples) / batch_size)
    epochs = min(10, max(2, math.ceil(300 / steps_per_epoch)))
    total_steps = epochs * steps_per_epoch
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.0)
    # The learning rate rises from 0 during the first 5% of the steps, then falls back to 0
    # along a cosine curve.
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.05 * total_steps), total_steps)
    rng = np.random.default_rng(seed)   # fixes the order of the examples

    model.train()
    losses = []
    with Timer() as train_timer:
        for epoch in range(epochs):
            order = rng.permutation(len(examples))
            for step in range(steps_per_epoch):
                batch = [examples[i] for i in order[step * batch_size:(step + 1) * batch_size]]

                # Put the batch into rectangular tables. Rows differ in length, so shorter rows
                # are padded ON THE LEFT: every row then ENDS with its label, which makes the
                # label tokens easy to find (they are the last tokens of each row).
                max_length = max(len(prompt) + len(label) for prompt, label in batch)
                max_label_length = max(len(label) for _, label in batch)
                token_ids = torch.full((len(batch), max_length), tokenizer.pad_token_id)
                is_real_token = torch.zeros((len(batch), max_length), dtype=torch.long)   # 0 = padding
                # The tokens the model should predict. -100 means "nothing to predict here".
                targets = torch.full((len(batch), max_label_length), -100)
                for row, (prompt, label) in enumerate(batch):
                    n = len(prompt) + len(label)
                    token_ids[row, max_length - n:] = torch.tensor(prompt + label)
                    is_real_token[row, max_length - n:] = 1
                    targets[row, max_label_length - len(label):] = torch.tensor(label)
                # Number the real tokens 0, 1, 2, ... so that left padding does not shift them.
                position_ids = (is_real_token.cumsum(1) - 1).clamp(min=0)

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    # The output at one place predicts the token at the NEXT place. To predict
                    # the last `max_label_length` tokens of a row we need the outputs at the
                    # `max_label_length` places just before them. logits_to_keep returns the
                    # last max_label_length + 1 outputs and [:, :-1] drops the final one, which
                    # would predict a token after the end. Computing only these outputs (and not
                    # one for every prompt token) is what lets training fit in GPU memory.
                    logits = model(input_ids=token_ids.cuda(), attention_mask=is_real_token.cuda(),
                                   position_ids=position_ids.cuda(),
                                   logits_to_keep=max_label_length + 1).logits[:, :-1]
                    # Cross-entropy is small when the correct label tokens get high probability.
                    loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                                             targets.cuda().reshape(-1), ignore_index=-100)
                loss.backward()                                    # how to change A and B
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)     # no single step may be huge
                optimizer.step()                                   # change A and B
                scheduler.step()                                   # update the learning rate
                optimizer.zero_grad(set_to_none=True)
                losses.append(loss.item())
            mean_loss = np.mean(losses[-steps_per_epoch:])
            print(f"  lora {size} {budget} epoch {epoch + 1}/{epochs} loss {mean_loss:.4f}", flush=True)

    model.save_pretrained(adapter_folder)   # saves only A and B
    del model, optimizer
    torch.cuda.empty_cache()
    return dict(train_gpu_sec=train_timer.sec, epochs=epochs, steps=total_steps, lr=learning_rate, bs=batch_size,
                r=rank, final_loss=float(np.mean(losses[-steps_per_epoch:])))


def run_lora(dataset, data, size, budget):
    method = f"lora-qwen3-{size}"
    if is_done(dataset, method, budget):
        return

    # 1. Train the adapter, unless an earlier (interrupted) job already saved it.
    adapter_folder = RESULTS_DIR / dataset / method / budget / "adapter"
    if not (adapter_folder / "adapter_config.json").exists():
        training_info = train_adapter(dataset, data, size, budget, adapter_folder, seed=seed_of(budget))
        json.dump(training_info, open(adapter_folder / "train_info.json", "w"))
    training_info = json.load(open(adapter_folder / "train_info.json"))

    # 2. Load the model with the adapter merged in and score every label for every text.
    tokenizer, model = load_model(size, adapter_folder=str(adapter_folder))
    label_names = data["label_names"]
    scorer = LabelScorer(model, tokenize_labels(tokenizer, label_names))
    scores, seconds = {}, {}
    for split in ["test", "val"]:
        with Timer() as timer:
            prompts = [tokenize_prompt(tokenizer, dataset,
                                       build_user_message(dataset, label_names, text, include_label_list=False))
                       for text in data[split][0]]
            scores[split] = scorer.score(prompts, max_tokens=MAX_TOKENS_PER_BATCH[size])
        seconds[split] = timer.sec
    save_run(dataset, method, budget, data["test"][1], scores["test"], scores["val"],
             n_train=len(budget_indices(dataset, budget)), infer_gpu_sec=seconds["test"],
             n_test=len(data["test"][1]), val_gpu_sec=seconds["val"], **training_info)
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default="1.7B", help="Qwen3 size: 0.6B, 1.7B, 4B or 8B")
    parser.add_argument("--datasets", nargs="+", default=["banking77", "goemotions"])
    parser.add_argument("--ks", nargs="+", type=parse_size, default=[8, 32, 128, "full"],
                        help="labeled examples per class, or 'full'")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    args = parser.parse_args()

    print("GPUs:", torch.cuda.device_count(), torch.cuda.get_device_name(0), flush=True)
    for dataset in args.datasets:
        data = load_dataset(dataset)
        for budget in budget_names(dataset, args.ks, args.seeds):
            run_lora(dataset, data, args.size, budget)
