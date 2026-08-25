"""Unit checks for the dual-ranking ensemble (Option A).

Run from the repo root:

    python -m pytest pyg_implementation/tests/test_dual_rank.py -q
    # or, without pytest:
    python pyg_implementation/tests/test_dual_rank.py

These cover the parts that fail *silently* if wrong -- a sign error in the rank fusion
or a misalignment between the p_A and p_B candidate lists produces plausible-looking
numbers rather than an exception. The end-to-end guarantee that p_A still reproduces the
library lives in dual_rank.assert_matches_library and runs inside the sweep itself.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from txgnn import dual_rank as dr
from txgnn.utils import exponential

import torch


LAB = np.array([1., 1., 0., 0., 0., 0., 0., 0.])
PERFECT = np.array([9., 8., 1., .9, .8, .7, .6, .5])


# ── rank-average fusion ────────────────────────────────────────────────────────

def test_rank_average_preserves_a_perfect_ranking():
    fused = dr.fuse_rank_average(PERFECT, PERFECT)
    assert abs(dr.auprc_one(fused, LAB) - 1.0) < 1e-12


def test_rank_average_is_scale_invariant():
    a = np.array([3., 1., 2., 5., 4.])
    b = np.array([1., 5., 2., 4., 3.])
    assert np.allclose(dr.fuse_rank_average(a, b),
                       dr.fuse_rank_average(a * 1000 + 7, b))


def test_rank_average_of_opposed_inputs_ties_everything():
    # the degenerate case: exactly anti-correlated inputs cancel. Worth pinning down
    # because it is the one situation where fused AUPRC is tie-break dependent.
    fused = dr.fuse_rank_average(PERFECT, -PERFECT)
    assert len(np.unique(fused)) == 1


def test_rank_average_does_not_collapse_to_one_input():
    a = np.array([3., 1., 2., 5., 4.])
    b = np.array([1., 5., 2., 4., 3.])
    fused = dr.fuse_rank_average(a, b)
    assert not np.allclose(np.argsort(-fused), np.argsort(-a))


# ── candidate filtering: must mirror utils.calculate_metrics ───────────────────

def _fixture():
    rel_ids = np.array(['d1', 'd2', 'd3', 'd4'])
    labels = {'d1': 1, 'd2': 0, 'd3': -1, 'd4': 0, 'd9': 1}  # d3 in train, d9 off-relation
    pa = {'d1': .9, 'd2': .1, 'd3': .5, 'd4': .2, 'd9': .8}
    pb = {'d1': .3, 'd2': .7, 'd3': .4, 'd4': .6, 'd9': .1}
    return rel_ids, labels, pa, pb


def test_filter_drops_train_sentinels_and_off_relation_drugs():
    rel_ids, labels, pa, _ = _fixture()
    keys, _, _ = dr.filter_candidates(pa, labels, rel_ids)
    assert list(keys) == ['d1', 'd2', 'd4']


def test_filter_gives_both_paths_the_same_candidate_order():
    # if this ever diverges, fuse_rank_average silently combines mismatched drugs
    rel_ids, labels, pa, pb = _fixture()
    ka, _, la = dr.filter_candidates(pa, labels, rel_ids)
    kb, _, lb = dr.filter_candidates(pb, labels, rel_ids)
    assert list(ka) == list(kb)
    assert np.array_equal(la, lb)


def test_auprc_sentinel_for_a_disease_with_no_positives():
    assert dr.auprc_one(np.random.rand(5), np.zeros(5)) == -1.0
    assert dr.auroc_one(np.random.rand(5), np.zeros(5)) == -1.0


# ── gate behaviour ────────────────────────────────────────────────────────────

def test_gate_is_saturated_at_zero_degree():
    # every held-out disease in a zero-shot split has dd_degree_train == 0, so the
    # disease-side gate contributes no variation at evaluation time
    assert abs(float(exponential(torch.tensor(0.0), 0.7)) - 0.9) < 1e-6


def test_gate_decays_with_degree():
    c = [float(exponential(torch.tensor(float(d)), 0.7)) for d in (0, 1, 5, 20)]
    assert c == sorted(c, reverse=True)
    assert c[-1] < 0.21          # floor is the hardcoded +0.2


def test_empty_signature_mask_preserves_the_drug_embedding():
    # DR6: without the valid-mask, a zero pooled vector shrinks h_j by (1 - c_j)
    h_j = np.array([1., 2., 3.])
    c_j, pooled = 0.9, np.zeros(3)
    assert np.allclose((1 - c_j) * h_j + c_j * pooled, 0.1 * h_j)   # the bug
    c_masked = c_j * 0.0
    assert np.array_equal((1 - c_masked) * h_j + c_masked * pooled, h_j)  # the fix


# ── fusion feature block ──────────────────────────────────────────────────────

def test_fusion_features_shape_and_constant_degree_columns():
    pa, pb = np.random.rand(50), np.random.rand(50)
    X = dr.build_fusion_features(pa, pb, dd_degree_train=0.0, kg_degree=17.0)
    assert X.shape == (50, len(dr.FUSION_FEATURES))
    assert len(np.unique(X[:, 4])) == 1 and len(np.unique(X[:, 5])) == 1


def test_scalar_fusion_endpoints_recover_each_input_ranking():
    pa, pb = np.random.rand(40), np.random.rand(40)
    assert np.array_equal(np.argsort(-dr.fuse_scalar(pa, pb, 1.0)), np.argsort(-pa))
    assert np.array_equal(np.argsort(-dr.fuse_scalar(pa, pb, 0.0)), np.argsort(-pb))


def test_scalar_weight_fit_recovers_the_informative_path():
    # p_B pure noise, p_A perfect -> the fitted weight must favour p_A
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(12):
        lab = np.zeros(60); lab[:3] = 1
        good = np.concatenate([rng.uniform(.8, 1., 3), rng.uniform(0., .5, 57)])
        noise = rng.random(60)
        rows.append((good, noise, lab))
    w, _ = dr.fit_scalar_weight(rows)
    assert w > 0.5, w


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    failed = []
    for fn in fns:
        try:
            fn()
            print('PASS  %s' % fn.__name__)
        except Exception as e:
            failed.append(fn.__name__)
            print('FAIL  %s -- %s' % (fn.__name__, e))
    print('\n%d/%d passed' % (len(fns) - len(failed), len(fns)))
    sys.exit(1 if failed else 0)
