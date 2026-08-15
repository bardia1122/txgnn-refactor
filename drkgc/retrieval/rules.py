"""Mine confidence-scored relation-path rules for the target relations.

A rule is a Horn clause whose body is a path of relations::

    indication(x, y) <- drug_protein(x, z1) & disease_protein(z1, y)
    "x targets a gene that is associated with disease y"

DrKGC uses NCRL here. **We enumerate exhaustively instead**, because this KG has
6 relations (12 with inverses), so every type-consistent body up to length 3 is
a few dozen candidates — countable exactly, in minutes, with no model to train.
The paper's own ablation (Table 8) moves PrimeKG MRR by under 1% between NCRL,
RNNLogic and random rules, so a learned miner is not where the value lies.

Counting is done with sparse boolean matrix products over the global entity
index: the body's grounding set is the boolean product of its relation
adjacency matrices, and

    support(B)     = #{(x, y) : body B connects x to y}
    confidence(B)  = #{(x, y) : B(x, y) and r(x, y)} / support(B)
    head_coverage  = #{(x, y) : B(x, y) and r(x, y)} / #{(x, y) : r(x, y)}

Only **training** edges are used, so the rules carry no test information.

Run standalone::

    python -m drkgc.retrieval.rules --out-dir drkgc/data_holdout
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from drkgc.config import DEFAULT_OUT_DIR, TARGET_RELATIONS
from drkgc.ranker.data import RankerData, build_dataset

#: give up on a body whose grounding set explodes (protects against hub blowup)
MAX_NNZ = 25_000_000


@dataclass
class Rule:
    """`head(x, y) <- body[0](x, z1) & body[1](z1, z2) & ...`"""

    head: str
    #: (relation name, inverted) pairs, applied left to right
    body: List[Tuple[str, bool]]
    support: int
    correct: int
    confidence: float
    head_coverage: float

    @property
    def length(self) -> int:
        return len(self.body)

    def as_text(self) -> str:
        atoms = " & ".join(
            f"{rel}^-1" if inv else rel for rel, inv in self.body
        )
        return f"{self.head}(x, y) <- {atoms}"

    def to_dict(self) -> Dict:
        return {
            "head": self.head,
            "body": [[rel, bool(inv)] for rel, inv in self.body],
            "text": self.as_text(),
            "length": self.length,
            "support": self.support,
            "correct": self.correct,
            "confidence": round(self.confidence, 6),
            "head_coverage": round(self.head_coverage, 6),
        }


# ---------------------------------------------------------------------------
# adjacency matrices over the global entity index
# ---------------------------------------------------------------------------


def relation_matrices(data: RankerData):
    """{(relation, inverted): csr_matrix} over the global entity space.

    Built from the *training* graph only — `data.train_triples` plus, when the
    auxiliary edges are not used as supervision, the graph edges themselves.
    """
    from scipy import sparse

    n = data.entities.num_entities
    id2rel = {v: k for k, v in data.rel2id.items()}

    # data.edge_index already holds train edges + inverses; take the forward half
    forward = data.edge_index.shape[1] // 2
    src = data.edge_index[0, :forward]
    dst = data.edge_index[1, :forward]
    rel = data.edge_type[:forward]

    matrices = {}
    for rel_id, relation in id2rel.items():
        mask = rel == rel_id
        if not mask.any():
            continue
        rows, cols = src[mask], dst[mask]
        forward_m = sparse.csr_matrix(
            (np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n)
        )
        forward_m.data[:] = 1
        matrices[(relation, False)] = forward_m
        matrices[(relation, True)] = forward_m.T.tocsr()
    return matrices


def _binarise(matrix):
    matrix = matrix.tocsr()
    matrix.data[:] = 1
    matrix.eliminate_zeros()
    return matrix


# ---------------------------------------------------------------------------
# body enumeration
# ---------------------------------------------------------------------------


def enumerate_bodies(
    data: RankerData,
    head: str,
    max_length: int = 3,
) -> List[List[Tuple[str, bool]]]:
    """Every type-consistent relation path from the head's source to its target.

    Type consistency prunes the search hard: of 12 directed relations only a
    handful can follow any given one, so the candidate set stays small enough
    to score exhaustively.
    """
    src_type, dst_type = data.rel_types[head]

    # (relation, inverted) -> (from_type, to_type)
    steps: Dict[Tuple[str, bool], Tuple[str, str]] = {}
    for relation, (a, b) in data.rel_types.items():
        steps[(relation, False)] = (a, b)
        steps[(relation, True)] = (b, a)

    bodies: List[List[Tuple[str, bool]]] = []

    def walk(current_type: str, path: List[Tuple[str, bool]]) -> None:
        if path and current_type == dst_type:
            # a length-1 body identical to the head is the trivial rule
            if not (len(path) == 1 and path[0] == (head, False)):
                bodies.append(list(path))
        if len(path) >= max_length:
            return
        for step, (a, b) in steps.items():
            if a != current_type:
                continue
            if path and step[0] == path[-1][0] and step[1] != path[-1][1]:
                continue  # immediately undoing the previous step is a no-op
            path.append(step)
            walk(b, path)
            path.pop()

    walk(src_type, [])
    return bodies


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def score_bodies(
    data: RankerData,
    head: str,
    bodies: Sequence[Sequence[Tuple[str, bool]]],
    matrices: Dict,
    min_support: int = 10,
    min_confidence: float = 0.0,
    verbose: bool = True,
) -> List[Rule]:
    """Compute support / confidence / head coverage for each candidate body."""
    target = matrices.get((head, False))
    if target is None:
        raise KeyError(f"no training edges for the head relation {head!r}")
    num_head = int(target.nnz)

    rules: List[Rule] = []
    skipped = 0
    for body in bodies:
        product = None
        too_big = False
        for step in body:
            piece = matrices.get(step)
            if piece is None:
                too_big = True  # relation absent from the training graph
                break
            product = piece if product is None else _binarise(product @ piece)
            if product.nnz > MAX_NNZ:
                too_big = True
                break
        if too_big or product is None or product.nnz == 0:
            skipped += 1
            continue

        product = _binarise(product)
        support = int(product.nnz)
        if support < min_support:
            continue
        correct = int(product.multiply(target).nnz)
        confidence = correct / support if support else 0.0
        if confidence < min_confidence:
            continue
        rules.append(
            Rule(
                head=head,
                body=[tuple(s) for s in body],
                support=support,
                correct=correct,
                confidence=confidence,
                head_coverage=correct / num_head if num_head else 0.0,
            )
        )

    if verbose and skipped:
        print(f"  {skipped} bodies skipped (empty, missing relation, or over {MAX_NNZ:,} pairs)")
    return rules


# ---------------------------------------------------------------------------
# post-processing (DrKGC appendix A.4)
# ---------------------------------------------------------------------------


def resolve_conflicts(rules_by_head: Dict[str, List[Rule]]) -> Dict[str, List[Rule]]:
    """Group rules by identical body; when bodies predict several heads, keep
    only the highest-confidence head. Mirrors DrKGC's conflict resolution."""
    best: Dict[Tuple, Rule] = {}
    for head_rules in rules_by_head.values():
        for rule in head_rules:
            key = tuple(rule.body)
            if key not in best or rule.confidence > best[key].confidence:
                best[key] = rule

    out: Dict[str, List[Rule]] = {head: [] for head in rules_by_head}
    dropped = 0
    for head, head_rules in rules_by_head.items():
        for rule in head_rules:
            if best[tuple(rule.body)] is rule:
                out[head].append(rule)
            else:
                dropped += 1
    if dropped:
        print(f"  conflict resolution dropped {dropped} lower-confidence duplicates")
    return out


def eliminate_redundancy(rules: List[Rule]) -> List[Rule]:
    """Drop a rule when a strictly shorter rule that is a prefix of it has
    higher confidence — the extra hops are not earning anything.

    DrKGC states this over set-valued bodies (`A subset of B`); for path-shaped
    bodies the meaningful containment is the prefix, so that is what we use.
    """
    by_body = {tuple(r.body): r for r in rules}
    keep = []
    dropped = 0
    for rule in rules:
        redundant = False
        for cut in range(1, rule.length):
            shorter = by_body.get(tuple(rule.body[:cut]))
            if shorter is not None and shorter.confidence > rule.confidence:
                redundant = True
                break
        if redundant:
            dropped += 1
        else:
            keep.append(rule)
    if dropped:
        print(f"  redundancy elimination dropped {dropped} rules")
    return keep


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def mine(
    out_dir: Path = DEFAULT_OUT_DIR,
    relations: Sequence[str] = TARGET_RELATIONS,
    max_length: int = 3,
    min_support: int = 10,
    min_confidence: float = 0.0,
    top_k: Optional[int] = None,
    capped: bool = True,
    data: Optional[RankerData] = None,
) -> Dict[str, List[Rule]]:
    out_dir = Path(out_dir)
    if data is None:
        data = build_dataset(out_dir, capped=capped)

    print("Building relation matrices ...")
    matrices = relation_matrices(data)
    print(f"  {len(matrices)} directed relations (including inverses)")

    rules_by_head: Dict[str, List[Rule]] = {}
    for head in relations:
        print(f"\nMining rules for {head!r} (bodies up to length {max_length}) ...")
        bodies = enumerate_bodies(data, head, max_length)
        print(f"  {len(bodies)} type-consistent candidate bodies")
        rules = score_bodies(
            data, head, bodies, matrices, min_support, min_confidence
        )
        rules_by_head[head] = rules
        print(f"  {len(rules)} rules pass support >= {min_support}")

    rules_by_head = resolve_conflicts(rules_by_head)

    rules_dir = out_dir / "rules"
    rules_dir.mkdir(parents=True, exist_ok=True)

    for head, rules in rules_by_head.items():
        rules = eliminate_redundancy(rules)
        rules.sort(key=lambda r: (-r.confidence, -r.support))
        if top_k:
            rules = rules[:top_k]
        rules_by_head[head] = rules

        slug = head.replace(" ", "_").replace("/", "_")
        path = rules_dir / f"{slug}_rules.json"
        path.write_text(
            json.dumps(
                {
                    "head": head,
                    "max_length": max_length,
                    "min_support": min_support,
                    "num_rules": len(rules),
                    "rules": [r.to_dict() for r in rules],
                },
                indent=2,
            )
        )
        print(f"\n{head}: {len(rules)} rules -> {path}")
        for rule in rules[:10]:
            print(
                f"  conf={rule.confidence:.3f}  supp={rule.support:>8,}  "
                f"cov={rule.head_coverage:.3f}  {rule.as_text()}"
            )
    return rules_by_head


def load_rules(out_dir: Path, relation: str) -> List[Rule]:
    slug = relation.replace(" ", "_").replace("/", "_")
    payload = json.loads((Path(out_dir) / "rules" / f"{slug}_rules.json").read_text())
    return [
        Rule(
            head=item["head"],
            body=[(rel, bool(inv)) for rel, inv in item["body"]],
            support=item["support"],
            correct=item["correct"],
            confidence=item["confidence"],
            head_coverage=item["head_coverage"],
        )
        for item in payload["rules"]
    ]


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--relations", nargs="+", default=list(TARGET_RELATIONS))
    parser.add_argument("--max-length", type=int, default=3,
                        help="maximum rule body length (DrKGC uses 3)")
    parser.add_argument("--min-support", type=int, default=10)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None,
                        help="keep only the k highest-confidence rules per relation")
    parser.add_argument("--uncapped", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    mine(
        Path(args.out_dir),
        args.relations,
        args.max_length,
        args.min_support,
        args.min_confidence,
        args.top_k,
        capped=not args.uncapped,
    )


if __name__ == "__main__":
    main()
