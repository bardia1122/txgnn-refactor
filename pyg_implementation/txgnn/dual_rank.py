"""Dual-ranking ensemble (Option A) for TxGNN.

Scores a (disease, drug, relation) triple two independent ways from one trained model:

    p_A(i, j, r)  disease-pooled h_i^hat  x  plain h_j        -- original TxGNN
    p_B(i, j, r)  plain h_i               x  drug-pooled h_j^hat

and fuses the two ranked candidate lists per disease. Path A is the untouched baseline
code path (``DistMultPredictor.score_path == 'A'``); path B is a read-out of the *same*
trained encoder, so any difference between them is attributable to the read-out alone
and never to a divergent training trajectory.

Everything here operates on the raw prediction dicts returned by
``utils.disease_centric_evaluation(..., return_raw=True)``:

    preds[disease_id][drug_id] -> float score
    labels[disease_id][drug_id] -> 1 (test positive) / 0 (negative) / -1 (in train)

The candidate filtering below deliberately mirrors ``utils.calculate_metrics`` line for
line -- intersect with the drugs that participate in the relation, drop the ``-1``
train sentinels -- so that a fused AUPRC is comparable with the library's p_A AUPRC.
``assert_matches_library`` checks that equivalence and should be run once per sweep.
"""

import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.stats import rankdata, spearmanr

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# candidate filtering -- mirrors utils.calculate_metrics
# ---------------------------------------------------------------------------

def filter_candidates(pred, lab, rel_drug_ids):
    """Reduce one disease's candidate list to the set the library actually scores.

    ``rel_drug_ids`` is utils.disease_centric_evaluation's ``drug_ids_rels[rel]``: the
    drugs that participate in this relation anywhere in the full KG.
    """
    scored = [i for i, j in lab.items() if j != -1]
    fixed_keys = np.intersect1d(rel_drug_ids, scored)
    p = np.array([pred[i] for i in fixed_keys], dtype=float)
    l = np.array([lab[i] for i in fixed_keys], dtype=float)
    return fixed_keys, p, l


def auprc_one(pred_array, lab_array):
    """AUPRC for one disease, with the library's -1 sentinel convention."""
    if len(np.where(lab_array == 1)[0]) == 0:
        return -1.0
    try:
        return float(average_precision_score(lab_array, pred_array))
    except Exception:
        return -1.0


def auroc_one(pred_array, lab_array):
    if len(np.where(lab_array == 1)[0]) == 0:
        return -1.0
    try:
        return float(roc_auc_score(lab_array, pred_array))
    except Exception:
        return -1.0


# ---------------------------------------------------------------------------
# fusion rules
# ---------------------------------------------------------------------------

def _zscore(x):
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def _rank_pct(x):
    """Percentile rank, 1.0 = best. Ties averaged, matching rankdata's default."""
    return rankdata(x) / len(x) if len(x) else x


def fuse_rank_average(p_a, p_b):
    """Mean of the two ranks. Needs no training and is invariant to score scale.

    Returned as a score (higher = better) so it drops straight into auprc_one.
    """
    return -(rankdata(-p_a) + rankdata(-p_b)) / 2.0


def fuse_scalar(p_a, p_b, w):
    """w * z(p_A) + (1 - w) * z(p_B), per-disease standardised."""
    return w * _zscore(p_a) + (1.0 - w) * _zscore(p_b)


FUSION_FEATURES = ['z_a', 'z_b', 'rank_pct_a', 'rank_pct_b', 'dd_degree_train', 'kg_degree']


def build_fusion_features(p_a, p_b, dd_degree_train, kg_degree):
    """Per-candidate feature matrix for the learned fusion.

    The two degree columns are constant within a disease -- they let the MLP learn a
    *degree-conditional* mixing weight, which is the whole point of learning it rather
    than fixing one global w.
    """
    n = len(p_a)
    return np.column_stack([
        _zscore(p_a),
        _zscore(p_b),
        _rank_pct(p_a),
        _rank_pct(p_b),
        np.full(n, float(dd_degree_train)),
        np.full(n, float(kg_degree)),
    ])


class MLPFusion(nn.Module):
    """Two-layer late fusion. Trained post-hoc on saved scores, never inside TxGNN.

    Keeping this outside the model is what preserves p_A bit-for-bit: it adds no
    parameters to DistMultPredictor and therefore consumes none of the torch RNG that
    determines w_rels and the frozen node embeddings.
    """

    def __init__(self, n_feat=len(FUSION_FEATURES), n_hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1),
        )

    def forward(self, x):
        return self.net(x).reshape(-1)


def fit_scalar_weight(val_rows, grid=None):
    """Grid-search the scalar mixing weight on *validation* diseases.

    Fitting on the test diseases would leak; a disease-area split's validation diseases
    come from the complement area, so the degree distribution differs -- that is the
    honest cost of not leaking, and it is worth stating alongside any result.

    ``val_rows`` is a list of (p_a, p_b, lab_array) tuples, one per validation disease.
    """
    grid = np.linspace(0.0, 1.0, 101) if grid is None else grid
    best_w, best_score = 0.5, -np.inf
    for w in grid:
        scores = [auprc_one(fuse_scalar(pa, pb, w), l) for pa, pb, l in val_rows]
        scores = [s for s in scores if s >= 0]
        m = float(np.mean(scores)) if scores else -np.inf
        if m > best_score:
            best_w, best_score = float(w), m
    return best_w, best_score


def fit_mlp_fusion(val_feats, val_labels, n_epoch=200, lr=1e-2, seed=0, verbose=False):
    """Train MLPFusion with BCE on validation-disease candidates.

    Seeded locally so refitting is reproducible without touching global RNG state that
    the TxGNN training path depends on.
    """
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)

    X = torch.tensor(np.vstack(val_feats), dtype=torch.float32)
    y = torch.tensor(np.concatenate(val_labels), dtype=torch.float32)

    mu, sd = X.mean(0), X.std(0).clamp(min=1e-8)
    X = (X - mu) / sd

    model = MLPFusion()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    # positives are ~0.3% of candidates; without this the net predicts all-negative
    pos_weight = torch.tensor([(len(y) - y.sum()) / y.sum().clamp(min=1)])
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for ep in range(n_epoch):
        opt.zero_grad()
        loss = lossf(model(X), y)
        loss.backward()
        opt.step()
        if verbose and ep % 50 == 0:
            print('   fusion epoch %3d  loss %.4f' % (ep, loss.item()))

    model.eval()
    return model, (mu, sd)


def apply_mlp_fusion(model, norm, feats):
    mu, sd = norm
    X = torch.tensor(feats, dtype=torch.float32)
    X = (X - mu) / sd
    with torch.no_grad():
        return model(X).numpy()


# ---------------------------------------------------------------------------
# per-disease evaluation across variants
# ---------------------------------------------------------------------------

def evaluate_variants(preds_a, preds_b, labels, rel_drug_ids, degrees,
                      relation, seed, split, fusion=None):
    """Build the tidy per-disease table for every variant of one relation.

    ``degrees`` maps disease_id -> {'dd_degree_train': int, 'kg_degree': int}.
    ``fusion`` is None (rank-average only) or a dict with keys 'w' and/or
    ('mlp', 'norm') to add the learned variants.

    Returns a long DataFrame: one row per (disease, variant).
    """
    rows = []
    for disease_id in preds_a:
        _, p_a, lab = filter_candidates(preds_a[disease_id], labels[disease_id], rel_drug_ids)
        _, p_b, _ = filter_candidates(preds_b[disease_id], labels[disease_id], rel_drug_ids)

        deg = degrees.get(disease_id, {})
        dd_deg = deg.get('dd_degree_train', np.nan)
        kg_deg = deg.get('kg_degree', np.nan)

        variants = {
            'p_A': p_a,
            'p_B': p_b,
            'fuse_rankavg': fuse_rank_average(p_a, p_b),
        }

        if fusion is not None and 'w' in fusion:
            variants['fuse_scalar'] = fuse_scalar(p_a, p_b, fusion['w'])
        if fusion is not None and 'mlp' in fusion:
            feats = build_fusion_features(p_a, p_b, dd_deg, kg_deg)
            variants['fuse_mlp'] = apply_mlp_fusion(fusion['mlp'], fusion['norm'], feats)

        # rank agreement between the two paths -- the kill criterion. If p_B is a
        # monotone restatement of p_A there is nothing for a fusion to combine.
        rho = spearmanr(p_a, p_b).correlation if len(p_a) > 2 else np.nan

        for name, score in variants.items():
            rows.append({
                'split': split,
                'seed': seed,
                'relation': relation,
                'disease_id': disease_id,
                'variant': name,
                'auprc': auprc_one(score, lab),
                'auroc': auroc_one(score, lab),
                'n_pos': int((lab == 1).sum()),
                'n_candidates': len(lab),
                'dd_degree_train': dd_deg,
                'kg_degree': kg_deg,
                'spearman_A_B': rho,
            })
    return pd.DataFrame(rows)


def assert_matches_library(table, library_df, relation, tol=1e-9):
    """Guarantee the p_A column reproduces the library's AUPRC exactly.

    If this reimplementation of the candidate filtering ever drifts from
    utils.calculate_metrics, every fused number stops being comparable to the baseline
    and the whole experiment is void. Fail loudly and early instead.
    """
    ours = (table[(table.variant == 'p_A') & (table.relation == relation)]
            .set_index('disease_id')['auprc'])
    theirs = library_df['AUPRC']
    common = ours.index.intersection(theirs.index)
    if len(common) == 0:
        raise AssertionError('no overlapping diseases between reimplementation and library')
    delta = (ours.loc[common] - theirs.loc[common]).abs()
    worst = float(delta.max())
    if worst > tol:
        bad = delta.sort_values(ascending=False).head(5)
        raise AssertionError(
            'p_A AUPRC does not reproduce the library (max |delta| = %.3g).\n'
            'Fused metrics would not be comparable to the baseline. Worst:\n%s' % (worst, bad)
        )
    return worst
