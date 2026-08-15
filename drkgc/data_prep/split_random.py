"""Entity-safe random split of the target triples.

This mirrors what the DrKGC paper does for PrimeKG: shuffle the (drug,
indication, disease) triples, cut ~90/5/5, and make sure every drug and every
disease occurring in valid/test also occurs at least once in train.

The strategy is deliberately isolated behind :func:`get_random_split` so that a
future ``get_disease_area_split`` with the same signature (see
``split_base.SplitResult``) can be dropped in without touching anything else.

Run standalone::

    python -m drkgc.data_prep.split_random --out-dir drkgc/data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from drkgc.config import DEFAULT_OUT_DIR, SEED, SPLIT_FRACS, TARGET_RELATIONS
from drkgc.data_prep.extract_triples import load_triples
from drkgc.data_prep.split_base import (
    SplitResult,
    compute_split_stats,
    enforce_entity_safety,
    register_split,
    save_split,
)


@register_split("random")
def get_random_split(
    triples: pd.DataFrame,
    fracs: Sequence[float] = SPLIT_FRACS,
    seed: int = SEED,
    head_col: str = "head_idx",
    tail_col: str = "tail_idx",
    on_violation: str = "reassign",
    **_ignored,
) -> SplitResult:
    """Shuffle `triples` and cut them into an entity-safe train/valid/test split.

    Accepts and ignores the extra keyword arguments the other strategies take
    (`area`, `df`, ...) so `run_all` can call every strategy uniformly.

    Parameters
    ----------
    triples
        Triple table from `extract_triples` (one relation).
    fracs
        (train, valid, test) fractions; must sum to 1.
    seed
        Seed for the shuffle - the split is fully reproducible.
    on_violation
        ``'reassign'`` moves entity-unsafe valid/test triples into train (default,
        lossless); ``'drop'`` discards them.

    Returns
    -------
    SplitResult
        ``.train`` / ``.valid`` / ``.test`` frames with the same columns as the
        input, plus ``.stats``.
    """
    if len(fracs) != 3 or abs(sum(fracs) - 1.0) > 1e-9:
        raise ValueError(f"fracs must be three numbers summing to 1, got {fracs}")
    if len(triples) == 0:
        raise ValueError("no triples to split")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(triples))
    shuffled = triples.iloc[order].reset_index(drop=True)

    n = len(shuffled)
    n_train = int(round(fracs[0] * n))
    n_valid = int(round(fracs[1] * n))
    n_train = min(n_train, n)
    n_valid = min(n_valid, n - n_train)

    train = shuffled.iloc[:n_train]
    valid = shuffled.iloc[n_train : n_train + n_valid]
    test = shuffled.iloc[n_train + n_valid :]

    raw_sizes = {"train": len(train), "valid": len(valid), "test": len(test)}

    train, held_out, safety_info = enforce_entity_safety(
        train,
        {"valid": valid, "test": test},
        head_col=head_col,
        tail_col=tail_col,
        on_violation=on_violation,
    )

    result = SplitResult(
        train=train,
        valid=held_out["valid"],
        test=held_out["test"],
        strategy="random",
    )
    result.stats = compute_split_stats(result, head_col, tail_col)
    result.stats.update(
        {
            "seed": int(seed),
            "requested_fracs": {
                "train": fracs[0],
                "valid": fracs[1],
                "test": fracs[2],
            },
            "sizes_before_entity_safety": raw_sizes,
            "entity_safety": {
                "policy": on_violation,
                **safety_info,
            },
        }
    )
    return result


def split_all(
    out_dir: Path = DEFAULT_OUT_DIR,
    relations: Sequence[str] = TARGET_RELATIONS,
    fracs: Sequence[float] = SPLIT_FRACS,
    seed: int = SEED,
    on_violation: str = "reassign",
) -> dict:
    """Split every target relation independently and write the CSVs."""
    out = {}
    for relation in relations:
        triples = load_triples(out_dir, relation)
        print(f"\nSplitting {relation!r}: {len(triples):,} triples")
        result = get_random_split(
            triples, fracs=fracs, seed=seed, on_violation=on_violation
        )
        paths = save_split(result, relation, out_dir)
        out[relation] = result
        _print_stats(relation, result)
        print(f"  -> {paths['train'].parent}")
    return out


def _print_stats(relation: str, result: SplitResult) -> None:
    stats = result.stats
    for name in ("train", "valid", "test"):
        print(
            f"  {name:<5} {stats['sizes'][name]:>7,} triples "
            f"({stats['fractions'][name]:.3%})  "
            f"drugs={stats['unique_heads'][name]:,}  "
            f"diseases={stats['unique_tails'][name]:,}"
        )
    safety = stats["entity_safety"]
    for name in ("valid", "test"):
        info = safety[name]
        print(
            f"  entity-safety {name}: {info['num_violations']} violating triples "
            f"({info['num_reassigned_to_train']} reassigned to train, "
            f"{info['num_dropped']} dropped)"
        )
    viol = stats["entity_safety_violations"]
    print(f"  residual unseen entities: {viol}")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--relations", nargs="+", default=list(TARGET_RELATIONS))
    parser.add_argument("--fracs", nargs=3, type=float, default=list(SPLIT_FRACS))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--on-violation", choices=["reassign", "drop"], default="reassign"
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    split_all(
        Path(args.out_dir),
        args.relations,
        tuple(args.fracs),
        args.seed,
        args.on_violation,
    )


if __name__ == "__main__":
    main()
