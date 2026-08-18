"""Base LLM + LoRA + GCN adapter, with structural embeddings spliced into the prompt.

The one non-standard mechanic in DrKGC: the prompt contains `[Placeholder]`
tokens that are replaced, after tokenisation but before the forward pass, by the
adapter's vectors. That rules out any text-only trainer - we build the embedding
sequence ourselves and pass `inputs_embeds`.

Concretely, per example:

    input_ids   -> base.get_input_embeddings() -> [T, H]
    positions of PLACEHOLDER in input_ids      -> [1 + k] slots
    adapter(subgraph)                          -> [1 + k, H]
    scatter the adapter rows into those slots  -> inputs_embeds

Gradients flow back through those spliced positions into the adapter and the GCN,
which is what makes joint training work and what the step-4 smoke test verified.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from drkgc.adapter.model import GCNAdapter
from drkgc.llm.prompts import PLACEHOLDER


def load_base_model(
    model_id: str,
    quantise: bool = True,
    dtype: str = "bfloat16",
    device_map: str = "auto",
):
    """Load the base LLM, 4-bit NF4 by default (QLoRA)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"device_map": device_map, "dtype": torch_dtype}
    if quantise:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    return model, tokenizer


def attach_lora(
    model,
    r: int = 32,
    alpha: int = 32,
    dropout: float = 0.1,
    quantised: bool = True,
    gradient_checkpointing: bool = True,
):
    """LoRA with the paper's hyper-parameters (r=32, alpha=32, dropout=0.1)."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if quantised:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=gradient_checkpointing
        )
    elif gradient_checkpointing:
        model.gradient_checkpointing_enable()

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, config)
    # inputs_embeds must require grad for checkpointing to propagate into the adapter
    model.enable_input_require_grads()
    return model


class DrKGCModel(nn.Module):
    """LLM + LoRA + GCN adapter, joined by embedding splicing."""

    def __init__(
        self,
        llm,
        tokenizer,
        adapter: Optional[GCNAdapter],
        placeholder_token: str = PLACEHOLDER,
    ) -> None:
        super().__init__()
        self.llm = llm
        self.tokenizer = tokenizer
        self.adapter = adapter

        ids = tokenizer.encode(placeholder_token, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"placeholder {placeholder_token!r} encodes to {len(ids)} tokens; "
                "it must be exactly one so its positions can be located"
            )
        self.placeholder_id = ids[0]

    # -- embedding assembly -------------------------------------------------

    def _embedding_layer(self):
        return self.llm.get_input_embeddings()

    def build_inputs_embeds(
        self,
        input_ids: torch.Tensor,  # [B, T]
        adapter_batch: Optional[Dict] = None,
        global_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Token embeddings with adapter vectors spliced into placeholder slots."""
        embeddings = self._embedding_layer()(input_ids)
        if self.adapter is None or adapter_batch is None:
            return embeddings

        output = self.adapter(
            global_features,
            adapter_batch["edge_index"],
            adapter_batch["edge_type"],
            adapter_batch["query_index"],
            adapter_batch["candidate_index"],
        )
        # per example the slots are ordered: query first, then candidates
        vectors = torch.cat(
            [output["query"].unsqueeze(1), output["candidates"]], dim=1
        ).to(embeddings.dtype)  # [B, 1+k, H]

        mask = input_ids == self.placeholder_id
        counts = mask.sum(dim=1)
        expected = vectors.shape[1]
        if not bool((counts == expected).all()):
            raise ValueError(
                f"expected {expected} placeholders per example, found "
                f"{counts.tolist()} - prompt and candidate set are out of sync"
            )
        # index_put keeps the graph, so gradients reach the adapter
        embeddings = embeddings.masked_scatter(
            mask.unsqueeze(-1), vectors.reshape(-1, vectors.shape[-1])
        )
        return embeddings

    # -- forward / generate --------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        adapter_batch: Optional[Dict] = None,
        global_features: Optional[torch.Tensor] = None,
    ):
        inputs_embeds = self.build_inputs_embeds(
            input_ids, adapter_batch, global_features
        )
        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        adapter_batch: Optional[Dict] = None,
        global_features: Optional[torch.Tensor] = None,
        max_new_tokens: int = 24,
        **kwargs,
    ):
        inputs_embeds = self.build_inputs_embeds(
            input_ids, adapter_batch, global_features
        )
        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            **kwargs,
        )

    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def summary(self) -> str:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        adapter_params = self.adapter.num_parameters if self.adapter else 0
        return (
            f"trainable {trainable:,} / {total:,} "
            f"({trainable / max(total, 1):.2%}) | adapter {adapter_params:,}"
        )
