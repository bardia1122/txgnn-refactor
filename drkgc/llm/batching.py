"""Tokenise examples into batches the model can consume.

Padding side matters and differs by phase:

* **training** pads right, and labels are -100 everywhere except the answer
  tokens, so the loss is only on the answer;
* **generation** pads left, because `generate` continues from the final position
  and right padding would make it continue from padding.

The adapter batch is built in the same example order, so placeholder slots and
adapter vectors line up.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from drkgc.adapter.data import collate as collate_subgraphs
from drkgc.llm.dataset import Example

IGNORE_INDEX = -100


def make_batch(
    examples: Sequence[Example],
    tokenizer,
    global_embeddings,
    evidence: str = "embedding",
    for_training: bool = True,
    max_length: int = 2048,
    max_text_triples: int = 60,
    device=None,
) -> Dict:
    import torch

    prompts = [e.prompt(evidence=evidence, max_text_triples=max_text_triples) for e in examples]

    if for_training:
        tokenizer.padding_side = "right"
        sequences, label_sequences = [], []
        for example, prompt in zip(examples, prompts):
            prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
            answer_ids = tokenizer(
                " " + example.gold_name, add_special_tokens=False
            )["input_ids"] + [tokenizer.eos_token_id]
            ids = (prompt_ids + answer_ids)[:max_length]
            labels = ([IGNORE_INDEX] * len(prompt_ids) + answer_ids)[:max_length]
            sequences.append(ids)
            label_sequences.append(labels)

        width = max(len(s) for s in sequences)
        pad = tokenizer.pad_token_id
        input_ids = torch.tensor(
            [s + [pad] * (width - len(s)) for s in sequences], dtype=torch.long
        )
        labels = torch.tensor(
            [l + [IGNORE_INDEX] * (width - len(l)) for l in label_sequences],
            dtype=torch.long,
        )
        attention_mask = torch.tensor(
            [[1] * len(s) + [0] * (width - len(s)) for s in sequences], dtype=torch.long
        )
    else:
        tokenizer.padding_side = "left"
        encoded = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=max_length, add_special_tokens=True,
        )
        input_ids, attention_mask, labels = (
            encoded["input_ids"], encoded["attention_mask"], None
        )

    batch: Dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "examples": list(examples),
    }

    if evidence == "embedding":
        adapter_batch = collate_subgraphs([e.sample for e in examples])
        features = global_embeddings[adapter_batch["node_ids"]]
        batch["adapter_batch"] = adapter_batch
        batch["global_features"] = features

    if device is not None:
        batch["input_ids"] = batch["input_ids"].to(device)
        batch["attention_mask"] = batch["attention_mask"].to(device)
        if batch["labels"] is not None:
            batch["labels"] = batch["labels"].to(device)
        if "adapter_batch" in batch:
            batch["adapter_batch"] = {
                k: (v.to(device) if hasattr(v, "to") else v)
                for k, v in batch["adapter_batch"].items()
            }
            batch["global_features"] = batch["global_features"].to(device)
    return batch


def truncation_report(
    examples: Sequence[Example], tokenizer, evidence: str, max_length: int
) -> Dict:
    """Prompt-length statistics, to catch silent truncation before training.

    Truncation is dangerous here: cutting the prompt can drop placeholder tokens,
    which desynchronises them from the adapter vectors.
    """
    lengths = [
        len(tokenizer(e.prompt(evidence=evidence), add_special_tokens=True)["input_ids"])
        for e in examples
    ]
    over = sum(l > max_length for l in lengths)
    return {
        "num_examples": len(lengths),
        "min": min(lengths),
        "median": sorted(lengths)[len(lengths) // 2],
        "max": max(lengths),
        "over_max_length": over,
        "max_length": max_length,
    }
