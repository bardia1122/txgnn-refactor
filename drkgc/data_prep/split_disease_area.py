"""Zero-shot splits: hold out whole *diseases*, not random triples.

Two strategies, both returning a `split_base.SplitResult` exactly like
`split_random.get_random_split`, so `run_all.py --split <name>` swaps them in:

``disease_holdout``
    TxGNN's `complex_disease` semantics (`utils.py:194`): shuffle the unique
    diseases and give each split a disjoint set of them. Needs nothing beyond
    the triple table, so it runs on the KG you already prepared.

``disease_area``
    TxGNN's disease-*area* split: the held-out diseases are an
    ontology-defined area (`cardiovascular`, `diabetes`, ...). This one needs
    the area-specific KG, whose `split` column marks the edges TxGNN removes
    from training — see `kg_loader.load_area_kg_directed`.

**Entity safety differs from the random split, by design.** A zero-shot split is
*defined* by test diseases never appearing in train, so disease-side safety is
not enforced — it is asserted to be violated on purpose and reported as
`zero_shot_diseases`. Drug-side safety is still enforced: a drug the model has
never seen cannot be ranked sensibly.

Run standalone::

    python -m drkgc.data_prep.split_disease_area --strategy disease_area --area cardiovascular
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from drkgc.config import (
    AREA_VALID_FRAC,
    DEFAULT_KG_FOLDER,
    DEFAULT_OUT_DIR,
    DISEASE_AREAS,
    DISEASE_FILES_DIR,
    SEED,
    TARGET_RELATIONS,
    ZERO_SHOT_FRACS,
)
from drkgc.data_prep.extract_triples import load_triples
from drkgc.data_prep.kg_loader import area_disease_indices, load_kg_directed
from drkgc.data_prep.split_base import (
    SplitResult,
    compute_split_stats,
    enforce_entity_safety,
    register_split,
    save_split,
)


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _enforce_drug_safety(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    head_col: str = "head_idx",
) -> tuple:
    """Entity safety on the *head* (drug) side only; diseases stay zero-shot.

    Implemented by passing the head column as both key columns, so
    `enforce_entity_safety` never inspects the tail.

    Violations are **dropped, not reassigned**. Reassigning would move the
    offending triple into train — and with it the held-out disease, which would
    then appear in train and destroy the zero-shot property the whole split
    exists for. Dropping costs a handful of evaluation triples, and those are
    triples whose drug has no trained embedding anyway, so they could only add
    noise to the metrics.
    """
    train, held_out, info = enforce_entity_safety(
        train,
        {"valid": valid, "test": test},
        head_col=head_col,
        tail_col=head_col,
        on_violation="drop",
    )
    return train, held_out["valid"], held_out["test"], info


def _zero_shot_report(result: SplitResult) -> dict:
    """Confirm the held-out diseases really are unseen, and count them."""
    train_diseases = set(result.train.tail_idx)
    report = {}
    for name in ("valid", "test"):
        part = getattr(result, name)
        diseases = set(part.tail_idx)
        report[name] = {
            "num_diseases": len(diseases),
            "num_diseases_also_in_train": len(diseases & train_diseases),
            "num_drugs": int(part.head_idx.nunique()) if len(part) else 0,
        }
    return report


def _finalise(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    strategy: str,
    extra_stats: dict,
    safety_info: dict,
) -> SplitResult:
    result = SplitResult(
        train=train.reset_index(drop=True),
        valid=valid.reset_index(drop=True),
        test=test.reset_index(drop=True),
        strategy=strategy,
    )
    result.stats = compute_split_stats(result)
    result.stats.update(extra_stats)
    result.stats["entity_safety"] = {
        "policy": "drop_heads_only",
        "note": (
            "drug-side safety enforced by dropping violating eval triples "
            "(reassigning them would pull their held-out disease into train); "
            "disease-side safety deliberately not enforced - held-out diseases "
            "are unseen in train, that is the zero-shot setting"
        ),
        **safety_info,
    }
    result.stats["zero_shot_diseases"] = _zero_shot_report(result)
    return result


# ---------------------------------------------------------------------------
# strategy 1: random disease holdout (TxGNN 'complex_disease')
# ---------------------------------------------------------------------------


def compute_disease_partition(
    triple_tables: Sequence[pd.DataFrame],
    fracs: Sequence[float] = ZERO_SHOT_FRACS,
    seed: int = SEED,
) -> Dict[str, np.ndarray]:
    """Partition the diseases **once, across every target relation**.

    This has to be shared. `complex_disease_fold` (`utils.py:194-199`) takes
    `df_dd.y_idx.unique()` over *all* drug-disease relations at once, so a
    held-out disease loses its indication **and** contraindication edges
    together. Partitioning each relation separately would leave a disease that
    is held out for indication still connected through contraindication — the
    R-GCN would see it during training and the split would not be zero-shot.
    """
    diseases = pd.concat([t.tail_idx for t in triple_tables]).unique()
    np.random.seed(seed)
    np.random.shuffle(diseases)
    train_d, valid_d, test_d = np.split(
        diseases,
        [int(fracs[0] * len(diseases)), int((fracs[0] + fracs[1]) * len(diseases))],
    )
    return {"train": train_d, "valid": valid_d, "test": test_d}


@register_split("disease_holdout")
def get_disease_holdout_split(
    triples: pd.DataFrame,
    fracs: Sequence[float] = ZERO_SHOT_FRACS,
    seed: int = SEED,
    disease_partition: Optional[Dict[str, np.ndarray]] = None,
    **_ignored,
) -> SplitResult:
    """Split by disease: every disease belongs to exactly one of train/valid/test.

    Mirrors `complex_disease_fold` (`utils.py:194-206`) — same
    `np.random.seed` + `np.random.shuffle` + `np.split` over the unique
    diseases, so with the same seed the disease partition matches TxGNN's.

    `disease_partition` must be the partition shared by every target relation
    (see :func:`compute_disease_partition`). It is computed from this relation
    alone only when the caller passes nothing, which is correct just for a
    single-relation dataset.
    """
    if len(triples) == 0:
        raise ValueError("no triples to split")

    if disease_partition is None:
        disease_partition = compute_disease_partition([triples], fracs, seed)
    train_d = disease_partition["train"]
    valid_d = disease_partition["valid"]
    test_d = disease_partition["test"]

    train = triples[triples.tail_idx.isin(train_d)]
    valid = triples[triples.tail_idx.isin(valid_d)]
    test = triples[triples.tail_idx.isin(test_d)]

    raw_sizes = {"train": len(train), "valid": len(valid), "test": len(test)}
    train, valid, test, safety = _enforce_drug_safety(train, valid, test)

    return _finalise(
        train,
        valid,
        test,
        "disease_holdout",
        {
            "seed": int(seed),
            "requested_fracs": dict(zip(("train", "valid", "test"), map(float, fracs))),
            "fracs_are_over": "diseases, not triples",
            "num_diseases": {
                "train": len(train_d),
                "valid": len(valid_d),
                "test": len(test_d),
            },
            "sizes_before_entity_safety": raw_sizes,
        },
        safety,
    )


# ---------------------------------------------------------------------------
# strategy 2: ontology-defined disease area
# ---------------------------------------------------------------------------


@register_split("disease_area")
def get_disease_area_split(
    triples: pd.DataFrame,
    area: str = "cardiovascular",
    df: Optional[pd.DataFrame] = None,
    data_folder: Path = DEFAULT_KG_FOLDER,
    disease_files_dir: Path = DISEASE_FILES_DIR,
    valid_frac: float = AREA_VALID_FRAC,
    seed: int = SEED,
    **_ignored,
) -> SplitResult:
    """Hold out one ontology-defined disease area.

    Requires `triples` to carry the `split` column of the area KG (produced by
    `extract_triples` when the KG has one):

    * `split == 'test'` **and** disease in the area's node list  -> test
      (the area filter mirrors `process_disease_area_split`, `utils.py:1080-1081`)
    * everything else -> train, minus a random `valid_frac` slice for validation
      (TxGNN uses `random_fold(train_val, [0.875, 0.125, 0.0])`, `utils.py:395`)
    """
    if "split" not in triples.columns:
        raise ValueError(
            "the triple table has no 'split' column - the disease-area split needs "
            "the area KG. Re-run extract_triples with --area <area> (or run_all "
            "--split disease_area --area <area>)."
        )
    if df is None:
        df = load_kg_directed(data_folder, area=area, seed=seed)

    area_diseases = area_disease_indices(df, area, disease_files_dir)

    held = triples[triples.split == "test"]
    in_area = held[held.tail_idx.isin(area_diseases)]
    out_of_area = held[~held.tail_idx.isin(area_diseases)]

    train_pool = triples[triples.split != "test"]
    # a held-out triple whose disease is not in the area is not an evaluation
    # target, but it was removed from TxGNN's training KG - so drop it rather
    # than quietly training on it.
    test = in_area

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train_pool))
    shuffled = train_pool.iloc[order]
    n_valid = int(round(valid_frac * len(shuffled)))
    valid = shuffled.iloc[:n_valid]
    train = shuffled.iloc[n_valid:]

    raw_sizes = {"train": len(train), "valid": len(valid), "test": len(test)}
    train, valid, test, safety = _enforce_drug_safety(train, valid, test)

    result = _finalise(
        train,
        valid,
        test,
        "disease_area",
        {
            "seed": int(seed),
            "area": area,
            "valid_frac": float(valid_frac),
            "num_area_diseases_in_kg": len(area_diseases),
            "num_held_out_triples": int(len(held)),
            "num_held_out_in_area": int(len(in_area)),
            "num_held_out_dropped_out_of_area": int(len(out_of_area)),
            "sizes_before_entity_safety": raw_sizes,
        },
        safety,
    )
    if len(test) == 0:
        print(
            f"  ! no test triples for area {area!r} - this relation has no "
            "drug-disease edges into that area."
        )
    return result


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def split_all(
    strategy: str = "disease_area",
    out_dir: Path = DEFAULT_OUT_DIR,
    relations: Sequence[str] = TARGET_RELATIONS,
    area: Optional[str] = None,
    data_folder: Path = DEFAULT_KG_FOLDER,
    seed: int = SEED,
    **kwargs,
) -> dict:
    fn = {"disease_area": get_disease_area_split,
          "disease_holdout": get_disease_holdout_split}[strategy]
    df = load_kg_directed(data_folder, area=area, seed=seed) if strategy == "disease_area" else None

    tables = {relation: load_triples(out_dir, relation) for relation in relations}
    partition = (
        compute_disease_partition(list(tables.values()), seed=seed)
        if strategy == "disease_holdout"
        else None
    )

    out = {}
    for relation, triples in tables.items():
        print(f"\nSplitting {relation!r} ({strategy}): {len(triples):,} triples")
        result = (
            fn(triples, area=area, df=df, data_folder=data_folder, seed=seed, **kwargs)
            if strategy == "disease_area"
            else fn(triples, seed=seed, disease_partition=partition, **kwargs)
        )
        save_split(result, relation, out_dir)
        out[relation] = result
        for name in ("train", "valid", "test"):
            s = result.stats
            print(
                f"  {name:<5} {s['sizes'][name]:>7,} triples  "
                f"drugs={s['unique_heads'][name]:,}  diseases={s['unique_tails'][name]:,}"
            )
        print(f"  zero-shot check: {result.stats['zero_shot_diseases']}")
    return out


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=["disease_area", "disease_holdout"],
                        default="disease_area")
    parser.add_argument("--area", choices=list(DISEASE_AREAS), default=None)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--data-folder", default=str(DEFAULT_KG_FOLDER))
    parser.add_argument("--relations", nargs="+", default=list(TARGET_RELATIONS))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.strategy == "disease_area" and not args.area:
        parser.error("--area is required for the disease_area strategy")

    split_all(
        args.strategy,
        Path(args.out_dir),
        args.relations,
        args.area,
        Path(args.data_folder),
        args.seed,
    )


if __name__ == "__main__":
    main()
