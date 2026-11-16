"""Shared code for the three ways we use the Qwen3 language model (LLM) as a classifier:
zero-shot and few-shot prompting (run_llm_prompting.py) and LoRA fine-tuning (train_lora.py).

How an LLM classifies without writing text
------------------------------------------
An LLM reads a sequence of tokens (word pieces) and gives, for every possible next token, the
probability that it comes next. We never let the model write an answer. Instead we ask, for
every label: "how likely is it that the reply to this prompt is exactly this label name?"

    score(label) = log P(1st label token | prompt)
                 + log P(2nd label token | prompt, 1st label token)
                 + ...
                 + log P(end-of-reply token | prompt, all label tokens)

The label with the highest score is the prediction, and because every label gets a score we can
also turn the scores into a confidence, which the router and the calibration analysis need.
Including the end-of-reply token matters: without it, a short label that is the beginning of a
longer one ("card" / "card arrival") could not be told apart.

Scoring all labels in one pass
------------------------------
The obvious way is to run the model once per label: 77 runs for one Banking77 message. Each of
those runs repeats the same expensive work of reading the prompt. `LabelScorer` below instead
runs the model ONCE on

    [ prompt tokens ][ label 1 tokens ][ label 2 tokens ] ... [ label K tokens ]

and uses two tricks so that every label is still scored as if it stood alone right after the
prompt:

  * Attention mask. A token may only "look at" the prompt and at earlier tokens of its OWN
    label, never at the other labels.
  * Position numbers. The model is told that every label starts at the position right after the
    prompt, as it would if it were the only reply.

The result equals scoring every label separately, at the cost of roughly one run.
"""
import numpy as np
import torch

# The Qwen3 family comes in several sizes ("8B" = 8 billion weights). Names on Hugging Face:
MODEL_NAMES = {size: f"Qwen/Qwen3-{size}" for size in ["0.6B", "1.7B", "4B", "8B", "14B", "32B"]}

# Upper limit on the tokens put into one batch, so that every model size fits in GPU memory.
MAX_TOKENS_PER_BATCH = {"0.6B": 32768, "1.7B": 24576, "4B": 16384, "8B": 12288, "14B": 12288, "32B": 8192}

# The wording of the prompt for each dataset.
PROMPT_TEXT = {
    "banking77": dict(
        system="You route customer messages for an online bank to the right support intent.",
        ask="Classify the customer message into exactly one of these intents:",
        item="Message", answer="Intent",
        reply="Reply with the intent name only, exactly as written in the list.",
        short="What is the intent of this customer message?"),
    "goemotions": dict(
        system="You label the emotion expressed in Reddit comments.",
        ask="Which one of these emotions does the comment express? Options:",
        item="Comment", answer="Emotion",
        reply="Reply with the emotion name only, exactly as written in the list.",
        short="Which emotion does this comment express?"),
}


# --------------------------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------------------------
def build_user_message(dataset, label_names, text, examples=None, include_label_list=True):
    """The message we send to the model for one input text.

    Zero-shot (examples=None):
        Classify the customer message into exactly one of these intents:
        - activate my card
        - age limit
        ...
        Message: <the text>

        Reply with the intent name only, exactly as written in the list.

    Few-shot: the same, with "Labeled examples:" (a list of (text, label) pairs) inserted before
    the message.

    LoRA (include_label_list=False): a fine-tuned model has learned the label set during
    training, so the long label list is left out and the prompt is just a question and the text.
    A shorter prompt is cheaper to run.
    """
    wording = PROMPT_TEXT[dataset]
    if not include_label_list:
        return f"{wording['short']}\n{wording['item']}: {text}"
    parts = [wording["ask"], "\n".join(f"- {name}" for name in label_names), ""]
    if examples:
        parts.append("Labeled examples:")
        parts += [f"{wording['item']}: {example_text}\n{wording['answer']}: {label_names[example_label]}\n"
                  for example_text, example_label in examples]
    parts += [f"{wording['item']}: {text}", "", wording["reply"]]
    return "\n".join(parts)


def tokenize_prompt(tokenizer, dataset, user_message):
    """Wrap the message in the model's chat format and convert it to token numbers.

    The chat format marks who is speaking. add_generation_prompt=True ends the prompt at the
    point where the model's reply would start, which is exactly where the label is scored.
    enable_thinking=False switches off Qwen3's optional "thinking" text.
    """
    messages = [{"role": "system", "content": PROMPT_TEXT[dataset]["system"]},
                {"role": "user", "content": user_message}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
    return tokenizer(prompt, add_special_tokens=False)["input_ids"]


def tokenize_labels(tokenizer, label_names):
    """Token numbers of every label name followed by <|im_end|>, the end-of-reply token."""
    return [tokenizer(name + "<|im_end|>", add_special_tokens=False)["input_ids"] for name in label_names]


# --------------------------------------------------------------------------------------------
# Loading a model
# --------------------------------------------------------------------------------------------
def load_model(size, adapter_folder=None, n_gpus=1, dtype=torch.bfloat16):
    """Load a Qwen3 model and its tokenizer.

    adapter_folder  folder of a trained LoRA adapter (see train_lora.py). The adapter is merged
                    into the model's weights, so the merged model runs as fast as the original.
    n_gpus          the largest models do not fit on one GPU; "auto" spreads them over several.
    dtype           bfloat16 stores every weight in 16 bits instead of 32 (half the memory).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAMES[size])
    device_map = "auto" if n_gpus > 1 else {"": 0}
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAMES[size], dtype=dtype, attn_implementation="sdpa",
                                                 device_map=device_map)
    if adapter_folder:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_folder).merge_and_unload()
    model.eval()
    return tokenizer, model


# --------------------------------------------------------------------------------------------
# Scoring all labels in one pass (see the explanation at the top of this file)
# --------------------------------------------------------------------------------------------
class LabelScorer:
    """Computes score(label) for every label, for a list of prompts.

    A small example. Suppose there are two labels with these tokens (E = end-of-reply token):

        label 0:  [a, b, E]        label 1:  [c, E]

    To score a label, the model must predict each of its tokens from what comes before it:

        predict a  from  prompt               predict c  from  prompt
        predict b  from  prompt, a            predict E  from  prompt, c
        predict E  from  prompt, a, b

    So the model must READ every label token except the last one (a, b and c), and the sequence
    we feed it is

        [ prompt ... ] a b c          (a and b belong to label 0, c belongs to label 1)

    The model's output at the LAST PROMPT TOKEN predicts the first token of every label (a and
    c). Its output at `a` predicts b, at `b` predicts E (label 0), and at `c` predicts E
    (label 1). `read_from` below lists, for each of the 5 predictions, the place to read.
    """

    def __init__(self, model, label_tokens):
        self.model = model
        self.label_tokens = label_tokens
        self.n_labels = len(label_tokens)

        # The tokens the model has to read for each label: all but the last.
        self.tokens_to_read = [tokens[:-1] for tokens in label_tokens]
        # All of them in one flat list; this is what gets appended to every prompt.
        self.packed_tokens = [token for tokens in self.tokens_to_read for token in tokens]
        self.packed_length = sum(len(tokens) for tokens in self.tokens_to_read)

        # Where each label's block starts inside the packed list: [0, 2] in the example.
        block_starts = np.cumsum([0] + [len(tokens) for tokens in self.tokens_to_read])[:-1]

        # Position of each packed token inside its own label: [0, 1, 0] in the example. Added to
        # the prompt length this gives the position number the model is told (trick 2).
        self.position_in_label = np.concatenate(
            [np.arange(len(tokens)) for tokens in self.tokens_to_read]).astype(np.int64)

        # Trick 1: which packed tokens may look at which. Row = the token that looks, column =
        # the token looked at. A token may look at itself and at earlier tokens of its own label
        # (a lower triangle inside each label's block), and at nothing in other labels.
        may_look = np.zeros((self.packed_length, self.packed_length), dtype=bool)
        for start, tokens in zip(block_starts, self.tokens_to_read):
            n = len(tokens)
            may_look[start:start + n, start:start + n] = np.tril(np.ones((n, n), dtype=bool))
        self.may_look = torch.from_numpy(may_look)

        # One entry per label token to predict (5 in the example):
        #   read_from     where to read the model's output: -1 means "the last prompt token",
        #                 any other number is a place in the packed list
        #   target_token  the token that should be predicted there
        #   label_of      which label this prediction belongs to
        read_from, target_token, label_of = [], [], []
        for label, (start, tokens) in enumerate(zip(block_starts, label_tokens)):
            read_from += [-1] + [start + t for t in range(len(tokens) - 1)]
            target_token += tokens
            label_of += [label] * len(tokens)
        self.read_from = torch.tensor(read_from)
        self.target_token = torch.tensor(target_token)
        self.label_of = torch.tensor(label_of)

    @torch.no_grad()    # scoring only; PyTorch does not need to track gradients
    def score(self, prompts, max_tokens=16384):
        """prompts: a list of token-number lists. Returns a table (one row per prompt, one column
        per label) holding log P(label | prompt)."""
        device = self.model.get_input_embeddings().weight.device
        # Longest prompts first, so prompts of similar length share a batch (less padding).
        order = np.argsort([-len(prompt) for prompt in prompts])
        scores = np.zeros((len(prompts), self.n_labels), np.float32)
        start = 0
        while start < len(order):
            # As many prompts per batch as fit in the token limit (at least 1, at most 64).
            longest = len(prompts[order[start]]) + self.packed_length
            batch_size = max(1, min(64, max_tokens // longest))
            chosen = order[start:start + batch_size]
            scores[chosen] = self._score_batch([prompts[i] for i in chosen], device)
            start += batch_size
        return scores

    def _score_batch(self, prompts, device):
        batch_size = len(prompts)
        lengths = [len(prompt) + self.packed_length for prompt in prompts]
        max_length = max(lengths)

        # Build the model inputs. Shorter rows are padded at the end with token 0.
        token_ids = torch.full((batch_size, max_length), 0, dtype=torch.long)
        positions = torch.zeros((batch_size, max_length), dtype=torch.long)
        # allowed[b, i, j] is True if token i of row b may look at token j.
        allowed = torch.zeros((batch_size, max_length, max_length), dtype=torch.bool)
        read_positions = torch.zeros((batch_size, len(self.read_from)), dtype=torch.long)
        for b, prompt in enumerate(prompts):
            n_prompt = len(prompt)
            end = n_prompt + self.packed_length
            token_ids[b, :end] = torch.tensor(prompt + self.packed_tokens)
            # Prompt tokens are numbered 0, 1, 2, ...; every label continues from n_prompt.
            positions[b, :n_prompt] = torch.arange(n_prompt)
            positions[b, n_prompt:end] = torch.from_numpy(self.position_in_label) + n_prompt
            # A prompt token looks at itself and at earlier prompt tokens, like any text.
            allowed[b, :n_prompt, :n_prompt] = torch.tril(torch.ones(n_prompt, n_prompt, dtype=torch.bool))
            # A label token looks at the whole prompt ...
            allowed[b, n_prompt:end, :n_prompt] = True
            # ... and at its own label, as prepared in __init__.
            allowed[b, n_prompt:end, n_prompt:end] = self.may_look
            # Padding looks only at itself. (A token that may look at nothing produces invalid
            # numbers.) Nothing ever looks at padding, so it does not change any score.
            padding = torch.arange(end, max_length)
            allowed[b, padding, padding] = True
            # Turn "-1 = last prompt token" and packed places into places in this row.
            read_positions[b] = torch.where(self.read_from < 0, torch.full_like(self.read_from, n_prompt - 1),
                                            self.read_from + n_prompt)

        # The model wants the mask as numbers that are added before attention: 0 where looking is
        # allowed and a hugely negative number where it is not.
        dtype = self.model.dtype
        mask = torch.zeros((batch_size, 1, max_length, max_length), dtype=dtype)
        mask.masked_fill_(~allowed[:, None], torch.finfo(dtype).min)

        # Run the network once. `hidden` holds its output at every place of every row.
        hidden = self.model.model(input_ids=token_ids.to(device), attention_mask=mask.to(device),
                                  position_ids=positions.to(device)).last_hidden_state
        # Keep only the places we need to read.
        index = read_positions.to(hidden.device)[..., None].expand(-1, -1, hidden.shape[-1])
        hidden_at_reads = torch.gather(hidden, 1, index)

        # The output layer turns each of those outputs into a score for every token in the
        # vocabulary ("logits").
        output_layer = self.model.lm_head
        hidden_at_reads = hidden_at_reads.to(output_layer.weight.device)
        target_token = self.target_token.to(hidden_at_reads.device)
        label_of = self.label_of.to(hidden_at_reads.device)
        scores = torch.zeros((batch_size, self.n_labels), device=hidden_at_reads.device)
        for b in range(batch_size):   # one row at a time: the logits of a whole batch are large
            logits = output_layer(hidden_at_reads[b]).float()
            # log-probability of the target token = its logit - log(sum of exp of all logits)
            token_log_prob = logits.gather(1, target_token[:, None])[:, 0] - torch.logsumexp(logits, -1)
            # Add up the token log-probabilities that belong to the same label.
            scores[b].index_add_(0, label_of, token_log_prob)
        return scores.cpu().numpy()
