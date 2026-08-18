"""QLoRA fine-tuning of the LLM together with the GCN adapter.

Trainable parameters: the LoRA matrices and the whole GCN adapter. The base model
is frozen (and 4-bit quantised by default). Gradients reach the adapter through
the spliced placeholder positions, so the GCN learns what to encode - the paper
trains this way, and step 4's smoke test verifies the path is live.

Paper hyper-parameters (appendix A.5): LoRA r=32, alpha=32, dropout=0.1,
lr 2e-4, 15 epochs with early stopping.

Run::

    python -m drkgc.llm.train --out-dir drkgc/data_holdout \
        --model-dir drkgc/models/rgcn_holdout_v2 --run-dir drkgc/models/llm_holdout
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from drkgc.config import DEFAULT_OUT_DIR, DRKGC_ROOT, SEED, TARGET_RELATIONS

DEFAULT_RANKER_DIR = DRKGC_ROOT / "models" / "rgcn"
DEFAULT_RUN_DIR = DRKGC_ROOT / "models" / "llm"
DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"


def build_model(
    model_id: str,
    data,
    global_dim: int,
    llm_dim: int,
    evidence: str,
    hidden_dim: int = 128,
    lora_r: int = 32,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    quantise: bool = True,
    gradient_checkpointing: bool = True,
):
    from drkgc.adapter.model import GCNAdapter
    from drkgc.llm.model import DrKGCModel, attach_lora, load_base_model

    llm, tokenizer = load_base_model(model_id, quantise=quantise)
    llm = attach_lora(
        llm, r=lora_r, alpha=lora_alpha, dropout=lora_dropout,
        quantised=quantise, gradient_checkpointing=gradient_checkpointing,
    )
    adapter = None
    if evidence == "embedding":
        adapter = GCNAdapter(
            global_dim=global_dim,
            hidden_dim=hidden_dim,
            llm_dim=llm_dim,
            num_relations=data.num_relations,
        )
    return DrKGCModel(llm, tokenizer, adapter)


def load_run(
    run_dir: Path,
    ranker_dir: Path,
    model_id: str,
    data,
    quantise: bool = True,
    evidence: str = "embedding",
):
    """Reload a fine-tuned run for evaluation: LoRA weights + adapter state."""
    import torch
    from peft import PeftModel

    from drkgc.adapter.data import load_global_embeddings
    from drkgc.adapter.model import GCNAdapter
    from drkgc.llm.model import DrKGCModel, load_base_model

    run_dir = Path(run_dir)
    config = json.loads((run_dir / "run_config.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    llm, tokenizer = load_base_model(model_id, quantise=quantise)
    llm = PeftModel.from_pretrained(llm, str(run_dir / "lora"))
    llm.eval()

    global_embeddings = load_global_embeddings(ranker_dir).to(device)
    adapter = None
    if config.get("evidence", evidence) == "embedding":
        adapter = GCNAdapter(
            global_dim=config["global_dim"],
            hidden_dim=config["hidden_dim"],
            llm_dim=config["llm_dim"],
            num_relations=config["num_relations"],
        )
        state = torch.load(run_dir / "adapter.pt", map_location="cpu", weights_only=False)
        adapter.load_state_dict(state["state_dict"])
        adapter = adapter.to(device).eval()

    model = DrKGCModel(llm, tokenizer, adapter)
    return model, tokenizer, global_embeddings, device


def train(
    out_dir: Path = DEFAULT_OUT_DIR,
    ranker_dir: Path = DEFAULT_RANKER_DIR,
    run_dir: Path = DEFAULT_RUN_DIR,
    model_id: str = DEFAULT_MODEL_ID,
    relations: Sequence[str] = TARGET_RELATIONS,
    train_split: str = "valid",
    eval_split: str = "valid",
    evidence: str = "embedding",
    hidden_dim: int = 128,
    epochs: int = 5,
    batch_size: int = 1,
    grad_accum: int = 8,
    lr: float = 5e-5,
    adapter_lr: Optional[float] = None,
    max_length: int = 2048,
    limit: Optional[int] = None,
    dedup: bool = True,
    skip_unanswerable: bool = True,
    eval_every: int = 1,
    eval_steps: int = 25,
    patience: int = 4,
    eval_limit: int = 200,
    dev_frac: float = 0.15,
    quantise: bool = True,
    seed: int = SEED,
) -> Dict:
    import torch

    from drkgc.adapter.data import load_global_embeddings
    from drkgc.llm.batching import make_batch, truncation_report
    from drkgc.llm.dataset import (
        dedup_by_query, describe, load_examples, name_lookup, split_by_query,
    )
    from drkgc.llm.evaluate import predict, summarise
    from drkgc.ranker.data import build_dataset

    out_dir, ranker_dir, run_dir = Path(out_dir), Path(ranker_dir), Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = build_dataset(out_dir)
    names = name_lookup(data.entities.table)
    global_embeddings = load_global_embeddings(ranker_dir).to(device)
    global_dim = int(global_embeddings.shape[1])

    # ---- data -------------------------------------------------------------
    rng = np.random.default_rng(seed)

    # The paper fine-tunes on the validation split and tests on test, which leaves
    # no clean set for checkpoint selection. Carve a dev slice out of the training
    # pool, partitioned BY QUERY ENTITY so no disease appears on both sides.
    train_examples: List = []
    eval_examples: List = []
    for relation in relations:
        raw = load_examples(
            out_dir, relation, train_split, data.rel2id, names, limit=limit,
            skip_unanswerable=False, dedup=False,
        )
        pool, dev = split_by_query(raw, dev_frac, rng)

        # dev stays per-triple and keeps unanswerable rows, so it measures the
        # same thing the test evaluation will
        if eval_limit and len(dev) > eval_limit:
            picked = rng.choice(len(dev), size=eval_limit, replace=False)
            dev = [dev[i] for i in sorted(picked)]

        if skip_unanswerable:
            pool = [e for e in pool if e.gold_in_candidates]
        if dedup:
            pool = dedup_by_query(pool)

        print(f"train {relation}/{train_split}: {describe(pool)}")
        print(f"dev   {relation}/{train_split}: {len(dev):,} prompts over "
              f"{len({e.query_global_id for e in dev}):,} query entities")
        train_examples.extend(pool)
        eval_examples.extend(dev)

    if not train_examples:
        raise SystemExit("no trainable examples - check the split and retrieval outputs")
    if not eval_examples:
        raise SystemExit("no dev examples - raise --dev-frac")

    order = rng.permutation(len(train_examples))
    train_examples = [train_examples[i] for i in order]
    print(f"\n{len(train_examples):,} training prompts, {len(eval_examples):,} eval prompts")

    # ---- model ------------------------------------------------------------
    from transformers import AutoConfig

    llm_dim = int(AutoConfig.from_pretrained(model_id).hidden_size)
    print(f"llm hidden size: {llm_dim}")

    model = build_model(
        model_id, data, global_dim, llm_dim, evidence,
        hidden_dim=hidden_dim, quantise=quantise,
    )
    if model.adapter is not None:
        model.adapter = model.adapter.to(device=device, dtype=torch.float32)
    print(model.summary())

    stats = truncation_report(train_examples[:200], model.tokenizer, evidence, max_length)
    print(f"prompt tokens: {stats}")
    if stats["over_max_length"]:
        raise SystemExit(
            f"{stats['over_max_length']} of {stats['num_examples']} sampled prompts "
            f"exceed --max-length {max_length}. Truncation would drop placeholder "
            "tokens and desynchronise them from the adapter vectors. Raise "
            "--max-length or lower the candidate count."
        )

    # the adapter is small and randomly initialised, so it tolerates - and needs -
    # a larger step size than the LoRA matrices riding on a pretrained model
    groups = [{"params": [p for p in model.llm.parameters() if p.requires_grad], "lr": lr}]
    if model.adapter is not None:
        groups.append(
            {"params": list(model.adapter.parameters()), "lr": adapter_lr or lr * 5}
        )
    optimiser = torch.optim.AdamW(groups)

    steps_per_epoch = max(1, len(train_examples) // (batch_size * grad_accum))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=max(1, epochs * steps_per_epoch)
    )

    history: List[Dict] = []
    best = {"hits@1": -1.0, "epoch": -1, "step": -1}
    optimiser_steps = 0
    since_improved = 0
    stop = False

    def run_eval(tag: str) -> Dict:
        """Evaluate on dev, checkpoint if improved, and report against the ranker."""
        nonlocal best, since_improved, stop
        model.eval()
        rows = predict(
            model, eval_examples, model.tokenizer, global_embeddings,
            evidence, device, batch_size=max(1, batch_size * 2),
        )
        metrics = summarise(rows, seed=seed)
        print(
            f"  [{tag}] llm hits@1 {metrics['llm_hits@1']:.4f} "
            f"(ranker {metrics['ranker_hits@1']:.4f}, "
            f"delta {metrics['delta_hits@1']:+.4f}, "
            f"unmatched {metrics['unmatched_rate']:.1%})",
            flush=True,
        )
        if metrics["llm_hits@1"] > best["hits@1"]:
            best = {
                "hits@1": metrics["llm_hits@1"],
                "epoch": epoch,
                "step": optimiser_steps,
            }
            model.llm.save_pretrained(str(run_dir / "lora"))
            if model.adapter is not None:
                torch.save(
                    {"state_dict": model.adapter.state_dict()}, run_dir / "adapter.pt"
                )
            print("    * checkpoint saved", flush=True)
            since_improved = 0
        else:
            since_improved += 1
            if patience and since_improved >= patience:
                print(
                    f"    early stopping: no dev improvement in {patience} evaluations",
                    flush=True,
                )
                stop = True
        model.train()
        return metrics

    for epoch in range(1, epochs + 1):
        model.train()
        running, seen, started = 0.0, 0, time.time()
        optimiser.zero_grad()

        for index in range(0, len(train_examples), batch_size):
            chunk = train_examples[index : index + batch_size]
            batch = make_batch(
                chunk, model.tokenizer, global_embeddings, evidence=evidence,
                for_training=True, max_length=max_length, device=device,
            )
            output = model(
                batch["input_ids"], batch["attention_mask"], labels=batch["labels"],
                adapter_batch=batch.get("adapter_batch"),
                global_features=batch.get("global_features"),
            )
            (output.loss / grad_accum).backward()
            running += float(output.loss.item())
            seen += 1

            if seen % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
                optimiser.step()
                scheduler.step()
                optimiser.zero_grad()
                optimiser_steps += 1

                # Step-level evaluation. The useful signal can arrive well inside
                # the first epoch - epoch granularity misses the peak entirely.
                if eval_steps and optimiser_steps % eval_steps == 0:
                    print(
                        f"  epoch {epoch} step {optimiser_steps} "
                        f"loss {running / seen:.4f} ({time.time() - started:.0f}s)",
                        flush=True,
                    )
                    metrics = run_eval(f"step {optimiser_steps}")
                    history.append(
                        {"epoch": epoch, "step": optimiser_steps,
                         "loss": running / seen, "eval": metrics}
                    )
                    if stop:
                        break
            if not eval_steps and seen % (grad_accum * 25) == 0:
                print(
                    f"  epoch {epoch} step {seen // grad_accum} "
                    f"loss {running / seen:.4f} ({time.time() - started:.0f}s)",
                    flush=True,
                )

        epoch_loss = running / max(seen, 1)
        entry = {"epoch": epoch, "loss": epoch_loss}
        if stop:
            history.append(entry)
            break

        if not eval_steps and epoch % eval_every == 0:
            model.eval()
            rows = predict(
                model, eval_examples, model.tokenizer, global_embeddings,
                evidence, device, batch_size=max(1, batch_size * 2),
            )
            metrics = summarise(rows, seed=seed)
            entry["eval"] = metrics
            print(
                f"epoch {epoch}: loss {epoch_loss:.4f} | "
                f"llm hits@1 {metrics['llm_hits@1']:.4f} "
                f"(ranker {metrics['ranker_hits@1']:.4f}, "
                f"delta {metrics['delta_hits@1']:+.4f}, "
                f"unmatched {metrics['unmatched_rate']:.1%})"
            )
            if metrics["llm_hits@1"] > best["hits@1"]:
                best = {"hits@1": metrics["llm_hits@1"], "epoch": epoch,
                        "step": optimiser_steps}
                model.llm.save_pretrained(str(run_dir / "lora"))
                if model.adapter is not None:
                    torch.save(
                        {"state_dict": model.adapter.state_dict()},
                        run_dir / "adapter.pt",
                    )
                print("  * checkpoint saved")
        else:
            print(f"epoch {epoch}: loss {epoch_loss:.4f}")
        history.append(entry)

    config = {
        "model_id": model_id,
        "evidence": evidence,
        "global_dim": global_dim,
        "hidden_dim": hidden_dim,
        "llm_dim": llm_dim,
        "num_relations": data.num_relations,
        "out_dir": str(out_dir),
        "ranker_dir": str(ranker_dir),
        "train_split": train_split,
        "dedup": dedup,
        "skip_unanswerable": skip_unanswerable,
        "dev_frac": dev_frac,
        "eval_split_note": "dev slice carved from the training split by query entity",
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "lr": lr,
        "adapter_lr": adapter_lr or lr * 5,
        "seed": seed,
        "best_epoch": best["epoch"],
        "best_step": best.get("step"),
        "best_eval_hits@1": best["hits@1"],
        "num_train_prompts": len(train_examples),
    }
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2))
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"\nbest epoch {best['epoch']} (eval hits@1 {best['hits@1']:.4f}) -> {run_dir}")
    return config


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--model-dir", dest="ranker_dir", default=str(DEFAULT_RANKER_DIR))
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--relations", nargs="+", default=list(TARGET_RELATIONS))
    parser.add_argument("--train-split", default="valid",
                        help="the paper fine-tunes on the validation split (A.5)")
    parser.add_argument("--eval-split", default="valid")
    parser.add_argument("--evidence", default="embedding",
                        choices=["embedding", "text", "none"])
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--adapter-lr", type=float, default=None)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-dedup", action="store_true",
                        help="one prompt per triple (paper-faithful, ~8x more "
                             "prompts). Recommended here: dedup leaves only ~187.")
    parser.add_argument("--keep-unanswerable", action="store_true")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--eval-steps", type=int, default=25,
                        help="evaluate every N optimiser steps; 0 = per epoch")
    parser.add_argument("--patience", type=int, default=4,
                        help="stop after N evaluations without dev improvement")
    parser.add_argument("--eval-limit", type=int, default=200)
    parser.add_argument("--dev-frac", type=float, default=0.15,
                        help="fraction of training query entities held out for "
                             "checkpoint selection")
    parser.add_argument("--no-quantise", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(list(argv) if argv is not None else None)

    train(
        out_dir=Path(args.out_dir),
        ranker_dir=Path(args.ranker_dir),
        run_dir=Path(args.run_dir),
        model_id=args.model_id,
        relations=args.relations,
        train_split=args.train_split,
        eval_split=args.eval_split,
        evidence=args.evidence,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        adapter_lr=args.adapter_lr,
        max_length=args.max_length,
        limit=args.limit,
        dedup=not args.no_dedup,
        skip_unanswerable=not args.keep_unanswerable,
        eval_every=args.eval_every,
        eval_steps=args.eval_steps,
        patience=args.patience,
        eval_limit=args.eval_limit,
        dev_frac=args.dev_frac,
        quantise=not args.no_quantise,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
