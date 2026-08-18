"""Assemble training and evaluation examples from the step 1-3 artifacts.

One example = one query: the question, its candidate set, the retrieved subgraph
(for the adapter), and the gold answer.

Two policies worth knowing:

**Unanswerable training examples are skipped by default.** When the retriever
missed the gold, no candidate is correct, so training on it teaches the model to
emit an entity outside the list - contradicting its own instruction. Evaluation
never skips: a missed gold counts as wrong, which is exactly why `recall@k` is
the ceiling.

**`dedup` collapses training prompts per query entity.** The splits are per
triple, so a disease with 8 indications yields 8 near-identical prompts with
different single answers - contradictory supervision and 8x the cost. Dedup keeps
one prompt per (relation, query) and picks the highest-ranked gold among them.
Evaluation stays per triple either way, so the reported metric is unaffected.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from drkgc.adapter.data import SubgraphSample, record_to_sample, subgraph_path
from drkgc.llm.prompts import build_prompt


@dataclass
class Example:
    relation: str
    split: str
    query_global_id: int
    query_name: str
    candidate_ids: List[int]
    candidate_names: List[str]
    gold_global_id: int
    gold_name: str
    gold_in_candidates: bool
    sample: SubgraphSample
    triples_named: List[List[str]] = field(default_factory=list)

    @property
    def gold_rank_in_candidates(self) -> int:
        """1-based position of the gold in the candidate list, or -1."""
        try:
            return self.candidate_ids.index(self.gold_global_id) + 1
        except ValueError:
            return -1

    def prompt(self, evidence: str = "embedding", max_text_triples: int = 60) -> str:
        return build_prompt(
            self.relation,
            self.query_name,
            self.candidate_names,
            evidence=evidence,
            triples=self.triples_named,
            max_text_triples=max_text_triples,
        )


def load_examples(
    out_dir: Path,
    relation: str,
    split: str,
    rel2id: Dict[str, int],
    name_of: Dict[int, str],
    limit: Optional[int] = None,
    skip_unanswerable: bool = False,
    dedup: bool = False,
) -> List[Example]:
    path = subgraph_path(out_dir, relation, split)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run `python -m drkgc.retrieval.subgraph` for this split."
        )

    examples: List[Example] = []
    with path.open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if limit and len(examples) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            candidates = [int(c) for c in record["candidates"]]
            example = Example(
                relation=record["relation"],
                split=record["split"],
                query_global_id=int(record["query_global_id"]),
                query_name=record.get("query_name", ""),
                candidate_ids=candidates,
                candidate_names=[name_of.get(c, f"entity_{c}") for c in candidates],
                gold_global_id=int(record["gold_global_id"]),
                gold_name=name_of.get(int(record["gold_global_id"]), ""),
                gold_in_candidates=bool(record.get("gold_in_candidates", False)),
                sample=record_to_sample(record, rel2id),
                triples_named=record.get("triples_named", []),
            )
            if skip_unanswerable and not example.gold_in_candidates:
                continue
            examples.append(example)

    if dedup:
        examples = dedup_by_query(examples)
    return examples


def dedup_by_query(examples: Sequence[Example]) -> List[Example]:
    """One example per query entity, keeping the one whose gold ranks highest."""
    best: Dict[int, Example] = {}
    for example in examples:
        rank = example.gold_rank_in_candidates
        current = best.get(example.query_global_id)
        if current is None or (0 < rank < current.gold_rank_in_candidates):
            best[example.query_global_id] = example
    return list(best.values())


def split_by_query(
    examples: Sequence[Example], frac: float, rng
) -> tuple:
    """Partition examples into (major, minor) by **query entity**.

    Splitting by entity rather than by row is essential: the same disease appears
    in several triples, so a row-wise split would put near-identical prompts on
    both sides and make the held-out score meaningless.
    """
    entities = sorted({e.query_global_id for e in examples})
    if not entities or frac <= 0:
        return list(examples), []
    picked = rng.choice(len(entities), size=max(1, int(round(frac * len(entities)))),
                        replace=False)
    minor_ids = {entities[i] for i in picked}
    major = [e for e in examples if e.query_global_id not in minor_ids]
    minor = [e for e in examples if e.query_global_id in minor_ids]
    return major, minor


def name_lookup(entity_table) -> Dict[int, str]:
    """global_id -> display name, from the ranker's entity table."""
    return {
        int(gid): (str(name) if str(name) else f"entity_{int(gid)}")
        for gid, name in zip(entity_table.global_id, entity_table.node_name)
    }


def describe(examples: Sequence[Example]) -> str:
    if not examples:
        return "no examples"
    answerable = sum(e.gold_in_candidates for e in examples)
    ranks = [e.gold_rank_in_candidates for e in examples if e.gold_rank_in_candidates > 0]
    top1 = sum(r == 1 for r in ranks)
    return (
        f"{len(examples):,} examples | gold in candidates {answerable:,} "
        f"({answerable / len(examples):.1%}) | ranker top-1 correct {top1:,} "
        f"({top1 / len(examples):.1%})"
    )


if __name__ == "__main__":
    import argparse

    from drkgc.config import DEFAULT_OUT_DIR, TARGET_RELATIONS
    from drkgc.ranker.data import build_dataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--relation", default=TARGET_RELATIONS[0])
    parser.add_argument("--split", default="valid")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dedup", action="store_true")
    parser.add_argument("--evidence", default="embedding",
                        choices=["embedding", "text", "none"])
    args = parser.parse_args()

    data = build_dataset(Path(args.out_dir))
    names = name_lookup(data.entities.table)
    examples = load_examples(
        Path(args.out_dir), args.relation, args.split, data.rel2id, names,
        limit=args.limit, dedup=args.dedup,
    )
    print(describe(examples))
    print("\n--- first prompt ---\n")
    print(examples[0].prompt(evidence=args.evidence))
    print(f"\ngold: {examples[0].gold_name!r} "
          f"(rank {examples[0].gold_rank_in_candidates} in candidates)")
