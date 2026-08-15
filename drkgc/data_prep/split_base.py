"""The split contract shared by every DrKGC splitting strategy.

Implementations: `split_random.get_random_split` (`'random'`) and, in
`split_disease_area.py`, `get_disease_holdout_split` (`'disease_holdout'`) and
`get_disease_area_split` (`'disease_area'`). Every strategy must

* take a triple table as produced by `extract_triples` (plus a seed and
  strategy-specific kwargs), and
* return a :class:`SplitResult`,

so that swapping one for the other is a one-line change in `run_all.py`.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from drkgc.config import DEFAULT_OUT_DIR, SPLITS_DIR

SPLIT_NAMES = ("train", "valid", "test")


@dataclass
class SplitResult:
    """The output signature every split function must produce."""

    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    #: how the split was produced, e.g. "random" / "disease_area"
    strategy: str = "unknown"
    #: sizes, unique entity counts, number of triples moved/dropped, ...
    stats: Dict = field(default_factory=dict)

    def as_dict(self) -> Dict[str, pd.DataFrame]:
        return {"train": self.train, "valid": self.valid, "test": self.test}

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"SplitResult(strategy={self.strategy!r}, train={len(self.train)}, "
            f"valid={len(self.valid)}, test={len(self.test)})"
        )


def enforce_entity_safety(
    train: pd.DataFrame,
    held_out: Dict[str, pd.DataFrame],
    head_col: str = "head_idx",
    tail_col: str = "tail_idx",
    on_violation: str = "reassign",
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict]:
    """Guarantee every entity in a held-out split also occurs in `train`.

    Triples whose head or tail is unseen in train are either moved to train
    (``on_violation='reassign'``, the default - no data is thrown away) or
    dropped (``on_violation='drop'``).

    Rows are visited in their existing (already shuffled) order; because
    reassigning a triple only ever *adds* entities to train, a single pass is
    enough to establish the invariant.
    """
    if on_violation not in ("reassign", "drop"):
        raise ValueError(f"on_violation must be 'reassign' or 'drop', got {on_violation!r}")

    train_heads = set(train[head_col].tolist())
    train_tails = set(train[tail_col].tolist())

    moved: List[pd.DataFrame] = []
    kept: Dict[str, pd.DataFrame] = {}
    info: Dict[str, Dict[str, int]] = {}

    for name, part in held_out.items():
        keep_mask = []
        n_violation = 0
        for head, tail in zip(part[head_col].tolist(), part[tail_col].tolist()):
            ok = head in train_heads and tail in train_tails
            keep_mask.append(ok)
            if not ok:
                n_violation += 1
                if on_violation == "reassign":
                    # this triple joins train, so its entities become "seen"
                    train_heads.add(head)
                    train_tails.add(tail)
        keep_mask = pd.Series(keep_mask, index=part.index, dtype=bool)
        kept[name] = part[keep_mask]
        if on_violation == "reassign":
            moved.append(part[~keep_mask])
        info[name] = {
            "num_violations": n_violation,
            "num_reassigned_to_train": n_violation if on_violation == "reassign" else 0,
            "num_dropped": n_violation if on_violation == "drop" else 0,
        }

    if moved:
        train = pd.concat([train] + moved, ignore_index=True)
    else:
        train = train.reset_index(drop=True)
    kept = {k: v.reset_index(drop=True) for k, v in kept.items()}
    return train, kept, info


def check_entity_safety(
    result: SplitResult,
    head_col: str = "head_idx",
    tail_col: str = "tail_idx",
) -> Dict[str, Dict[str, int]]:
    """Count entities appearing in valid/test but never in train (should be 0)."""
    train_heads = set(result.train[head_col].tolist())
    train_tails = set(result.train[tail_col].tolist())
    report = {}
    for name in ("valid", "test"):
        part = getattr(result, name)
        report[name] = {
            "unseen_heads": len(set(part[head_col].tolist()) - train_heads),
            "unseen_tails": len(set(part[tail_col].tolist()) - train_tails),
        }
    return report


def compute_split_stats(
    result: SplitResult,
    head_col: str = "head_idx",
    tail_col: str = "tail_idx",
) -> Dict:
    """Sizes and unique-entity counts per split, plus the safety report."""
    stats: Dict = {"sizes": {}, "unique_heads": {}, "unique_tails": {}}
    total = sum(len(getattr(result, n)) for n in SPLIT_NAMES)
    for name in SPLIT_NAMES:
        part = getattr(result, name)
        stats["sizes"][name] = int(len(part))
        stats["unique_heads"][name] = int(part[head_col].nunique())
        stats["unique_tails"][name] = int(part[tail_col].nunique())
    stats["total"] = int(total)
    stats["fractions"] = {
        name: (stats["sizes"][name] / total if total else 0.0) for name in SPLIT_NAMES
    }
    stats["entity_safety_violations"] = check_entity_safety(result, head_col, tail_col)
    return stats


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------


def split_dir(out_dir: Path = DEFAULT_OUT_DIR) -> Path:
    return Path(out_dir) / SPLITS_DIR


def save_split(
    result: SplitResult,
    relation: str,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> Dict[str, Path]:
    """Write `<out>/splits/<relation>_{train,valid,test}.csv` (+ stats json)."""
    target = split_dir(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    slug = relation.replace(" ", "_").replace("/", "_")

    paths = {}
    for name in SPLIT_NAMES:
        path = target / f"{slug}_{name}.csv"
        getattr(result, name).to_csv(path, index=False)
        paths[name] = path

    stats_path = target / f"{slug}_split_stats.json"
    stats_path.write_text(
        json.dumps({"strategy": result.strategy, **result.stats}, indent=2)
    )
    paths["stats"] = stats_path
    return paths


def load_split(relation: str, out_dir: Path = DEFAULT_OUT_DIR) -> SplitResult:
    """Read back a split written by :func:`save_split`."""
    target = split_dir(out_dir)
    slug = relation.replace(" ", "_").replace("/", "_")
    frames = {
        name: pd.read_csv(
            target / f"{slug}_{name}.csv",
            dtype={"head_id": str, "tail_id": str},
            keep_default_na=False,
        )
        for name in SPLIT_NAMES
    }
    stats_path = target / f"{slug}_split_stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    return SplitResult(
        train=frames["train"],
        valid=frames["valid"],
        test=frames["test"],
        strategy=stats.get("strategy", "unknown"),
        stats=stats,
    )


# ---------------------------------------------------------------------------
# registry - the swap point for the future disease-area split
# ---------------------------------------------------------------------------

#: name -> split function with signature (triples, seed=..., **kwargs) -> SplitResult
SPLIT_REGISTRY: Dict[str, Callable[..., SplitResult]] = {}


def register_split(name: str) -> Callable[[Callable[..., SplitResult]], Callable[..., SplitResult]]:
    def decorator(fn: Callable[..., SplitResult]) -> Callable[..., SplitResult]:
        SPLIT_REGISTRY[name] = fn
        return fn

    return decorator


def get_split_fn(name: str) -> Callable[..., SplitResult]:
    # importing here keeps `split_base` free of concrete-strategy imports
    from drkgc.data_prep import split_disease_area, split_random  # noqa: F401

    if name not in SPLIT_REGISTRY:
        raise KeyError(
            f"unknown split strategy {name!r}; available: {sorted(SPLIT_REGISTRY)}"
        )
    return SPLIT_REGISTRY[name]


def available_splits() -> Iterable[str]:
    from drkgc.data_prep import split_disease_area, split_random  # noqa: F401

    return sorted(SPLIT_REGISTRY)
