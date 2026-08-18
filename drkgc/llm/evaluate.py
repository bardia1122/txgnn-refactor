"""Predict with the fine-tuned model and compare it against the ranker.

The comparison is the point: **same queries, same candidate sets**, so `recall@k`
is identical in both arms and only the ordering differs. Any change in Hits@1 is
attributable to the LLM, not to retrieval.

* `ranker_hits@1`  - the candidate list is already in the ranker's order, so its
  top-1 is `candidate_ids[0]`;
* `llm_hits@1`     - the LLM's single choice;
* `delta`          - with a **bootstrap CI resampled over query entities**, not
  over triples. Variance here is disease-level: a disease contributes several
  triples that succeed or fail together, so resampling triples would understate
  the interval badly.

A generated answer is matched back to a candidate by exact name, then normalised
name, then prefix. Unmatched output counts as wrong and is reported separately -
a high unmatched rate means the prompt is not constraining the model.

Run::

    python -m drkgc.llm.evaluate --out-dir drkgc/data_holdout \
        --model-dir drkgc/models/rgcn_holdout_v2 --run-dir drkgc/models/llm_holdout
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from drkgc.config import DEFAULT_OUT_DIR, DRKGC_ROOT, SEED, TARGET_RELATIONS
from drkgc.llm.dataset import Example, load_examples, name_lookup

DEFAULT_RANKER_DIR = DRKGC_ROOT / "models" / "rgcn"
DEFAULT_RUN_DIR = DRKGC_ROOT / "models" / "llm"

_STRIP_CHARS = "'\".,;: "


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def match_candidate(generated: str, candidate_names: Sequence[str]) -> int:
    """Index of the matched candidate, or -1. Exact, then normalised, then prefix."""
    answer = generated.strip().split("\n")[0].strip().strip(_STRIP_CHARS)
    if not answer:
        return -1
    for i, name in enumerate(candidate_names):
        if answer == name:
            return i
    target = normalise(answer)
    if not target:
        return -1
    for i, name in enumerate(candidate_names):
        if normalise(name) == target:
            return i
    for i, name in enumerate(candidate_names):
        norm = normalise(name)
        if norm and (norm.startswith(target) or target.startswith(norm)):
            return i
    return -1


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def bootstrap_delta(
    per_query: Dict[int, List[Dict]],
    num_samples: int = 2000,
    seed: int = SEED,
) -> Dict[str, float]:
    """Bootstrap the Hits@1 difference, resampling **query entities**.

    Clustering by query entity is what makes the interval honest: the unit of
    independent variation is the disease, not the triple.
    """
    keys = list(per_query)
    if not keys:
        return {}
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(num_samples):
        drawn = rng.choice(len(keys), size=len(keys), replace=True)
        llm, ranker, total = 0, 0, 0
        for index in drawn:
            for row in per_query[keys[index]]:
                llm += row["llm_correct"]
                ranker += row["ranker_correct"]
                total += 1
        if total:
            deltas.append(llm / total - ranker / total)
    if not deltas:
        return {}
    values = np.asarray(deltas)
    return {
        "delta_mean": float(values.mean()),
        "delta_ci_low": float(np.percentile(values, 2.5)),
        "delta_ci_high": float(np.percentile(values, 97.5)),
        "bootstrap_samples": int(num_samples),
    }


def summarise(rows: Sequence[Dict], seed: int = SEED) -> Dict:
    if not rows:
        return {"num_queries": 0}
    total = len(rows)
    llm = sum(r["llm_correct"] for r in rows)
    ranker = sum(r["ranker_correct"] for r in rows)
    unmatched = sum(r["matched_index"] < 0 for r in rows)
    answerable = sum(r["gold_in_candidates"] for r in rows)

    per_query: Dict[int, List[Dict]] = defaultdict(list)
    for row in rows:
        per_query[row["query_global_id"]].append(row)

    result = {
        "num_queries": total,
        "num_query_entities": len(per_query),
        "recall_at_k": round(answerable / total, 5),
        "ranker_hits@1": round(ranker / total, 5),
        "llm_hits@1": round(llm / total, 5),
        "delta_hits@1": round((llm - ranker) / total, 5),
        "unmatched_generations": unmatched,
        "unmatched_rate": round(unmatched / total, 5),
    }
    result.update(bootstrap_delta(per_query, seed=seed))
    significant = (
        "delta_ci_low" in result
        and (result["delta_ci_low"] > 0 or result["delta_ci_high"] < 0)
    )
    result["significant_at_95"] = bool(significant)
    return result


# ---------------------------------------------------------------------------
# prediction
# ---------------------------------------------------------------------------


def predict(
    model,
    examples: Sequence[Example],
    tokenizer,
    global_embeddings,
    evidence: str,
    device,
    batch_size: int = 4,
    max_new_tokens: int = 24,
    max_length: int = 2048,
) -> List[Dict]:
    from drkgc.llm.batching import make_batch

    rows: List[Dict] = []
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        batch = make_batch(
            chunk, tokenizer, global_embeddings, evidence=evidence,
            for_training=False, max_length=max_length, device=device,
        )
        generated = model.generate(
            batch["input_ids"], batch["attention_mask"],
            adapter_batch=batch.get("adapter_batch"),
            global_features=batch.get("global_features"),
            max_new_tokens=max_new_tokens,
        )
        texts = tokenizer.batch_decode(generated, skip_special_tokens=True)

        for example, text in zip(chunk, texts):
            index = match_candidate(text, example.candidate_names)
            predicted = example.candidate_ids[index] if index >= 0 else -1
            rows.append(
                {
                    "relation": example.relation,
                    "split": example.split,
                    "query_global_id": example.query_global_id,
                    "query_name": example.query_name,
                    "gold_global_id": example.gold_global_id,
                    "gold_name": example.gold_name,
                    "gold_in_candidates": bool(example.gold_in_candidates),
                    "generated": text.strip()[:200],
                    "matched_index": index,
                    "predicted_global_id": predicted,
                    "llm_correct": int(predicted == example.gold_global_id),
                    # the candidate list is in ranker order, so [0] is its top-1
                    "ranker_correct": int(
                        example.candidate_ids[0] == example.gold_global_id
                    ),
                }
            )
    return rows


def ranker_only(examples: Sequence[Example], seed: int = SEED) -> Dict:
    """Baseline metrics with no LLM involved - useful before any fine-tuning."""
    rows = [
        {
            "query_global_id": e.query_global_id,
            "gold_in_candidates": e.gold_in_candidates,
            "matched_index": 0,
            "llm_correct": int(e.candidate_ids[0] == e.gold_global_id),
            "ranker_correct": int(e.candidate_ids[0] == e.gold_global_id),
        }
        for e in examples
    ]
    summary = summarise(rows, seed=seed)
    for key in ("delta_mean", "delta_ci_low", "delta_ci_high", "delta_hits@1",
                "llm_hits@1", "significant_at_95", "unmatched_generations",
                "unmatched_rate", "bootstrap_samples"):
        summary.pop(key, None)
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--model-dir", default=str(DEFAULT_RANKER_DIR),
                        help="ranker dir, for global_embeddings.pt")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR),
                        help="fine-tuned LoRA + adapter; unused with --baseline-only")
    parser.add_argument("--relations", nargs="+", default=list(TARGET_RELATIONS))
    parser.add_argument("--split", default="test")
    parser.add_argument("--evidence", default="embedding",
                        choices=["embedding", "text", "none"])
    parser.add_argument("--model-id", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--baseline-only", action="store_true",
                        help="report ranker metrics without loading any LLM")
    parser.add_argument("--no-quantise", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    from drkgc.ranker.data import build_dataset

    out_dir = Path(args.out_dir)
    data = build_dataset(out_dir)
    names = name_lookup(data.entities.table)

    report: Dict[str, Dict] = {}
    model = tokenizer = global_embeddings = device = None

    for relation in args.relations:
        examples = load_examples(
            out_dir, relation, args.split, data.rel2id, names, limit=args.limit
        )
        if args.baseline_only:
            report[relation] = ranker_only(examples, seed=args.seed)
            print(f"\n{relation}/{args.split}")
            for key, value in report[relation].items():
                print(f"  {key:<24} {value}")
            continue

        if model is None:  # load once, reuse across relations
            from drkgc.llm.train import load_run

            model, tokenizer, global_embeddings, device = load_run(
                Path(args.run_dir), Path(args.model_dir), args.model_id,
                data, quantise=not args.no_quantise,
            )
        rows = predict(
            model, examples, tokenizer, global_embeddings, args.evidence,
            device, batch_size=args.batch_size,
        )
        report[relation] = summarise(rows, seed=args.seed)
        print(f"\n{relation}/{args.split}")
        for key, value in report[relation].items():
            print(f"  {key:<24} {value}")

        target = Path(args.run_dir) / f"predictions_{relation}_{args.split}.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        print(f"  predictions -> {target}")

    destination = Path(args.out_dir if args.baseline_only else args.run_dir)
    destination.mkdir(parents=True, exist_ok=True)
    suffix = "baseline" if args.baseline_only else "llm"
    path = destination / f"metrics_{suffix}_{args.split}.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"\nmetrics -> {path}")


if __name__ == "__main__":
    main()
