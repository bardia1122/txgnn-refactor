"""Phase 3 verification for the per-block Gumbel disease-similarity gate.

Runs on a small synthetic knowledge graph -- no PrimeKG download, no training run. Checks the
things that are cheap to get wrong and expensive to notice later:

  1. gate parameters are actually registered (the exact failure that silently killed agg_measure='learn')
  2. gate parameters round-trip through state_dict() / load_state_dict()
  3. gradients reach the gate
  4. existing 'rarity' checkpoints still load into the new code (no new state_dict keys)
  5. new hyperparameters survive save_model -> load_pretrained via self.config
  6. temperature annealing advances once per optimizer step, and is pinned at tau_end for
     eval / Explainer runs
  7. the gate emits [N_q, hidden] -- the shape DistMult requires -- and falls back to the
     disease's own embedding when no block is selected

Usage (from pyg_implementation/):
    python tests/test_gumbel_gate.py
Exit code 0 = all passed.
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from txgnn.model import HeteroRGCN, N_SIM_BLOCKS, BLOCK_ETYPES, BLOCK_NODES
from txgnn.utils import initialize_node_embedding, obtain_disease_profile, obtain_disease_profile_blocks


N_DISEASE, N_DRUG, N_GENE, N_PHENO, N_EXPOSURE = 12, 9, 20, 15, 7
HID = 16

DD_RELATIONS = ['contraindication', 'indication', 'off-label use']

_failures = []


def check(name, condition, detail=''):
    if condition:
        print('  PASS  %s' % name)
    else:
        print('  FAIL  %s %s' % (name, detail))
        _failures.append(name)


def build_synthetic_graph(seed=0):
    """Small heterogeneous KG containing every relation the gate depends on."""
    g = torch.Generator().manual_seed(seed)
    G = HeteroData()
    counts = {'disease': N_DISEASE, 'drug': N_DRUG, 'gene/protein': N_GENE,
              'effect/phenotype': N_PHENO, 'exposure': N_EXPOSURE}
    for nt, n in counts.items():
        G[nt].num_nodes = n

    def edges(n_src, n_dst, n_edges):
        return torch.stack([torch.randint(0, n_src, (n_edges,), generator=g),
                            torch.randint(0, n_dst, (n_edges,), generator=g)])

    # Drug-disease relations: every disease must appear in ALL of them, otherwise the shared
    # disease-set check in _build_block_similarity correctly refuses to run.
    all_dis = torch.arange(N_DISEASE)
    for rel in DD_RELATIONS:
        drugs = torch.randint(0, N_DRUG, (N_DISEASE,), generator=g)
        G['drug', rel, 'disease'].edge_index = torch.stack([drugs, all_dis])
        G['disease', 'rev_' + rel, 'drug'].edge_index = torch.stack([all_dis, drugs])

    # The four signature blocks (source must be 'disease': obtain_disease_profile_blocks
    # matches on edge_index[0]).
    G['disease', 'disease_disease', 'disease'].edge_index = edges(N_DISEASE, N_DISEASE, 30)
    G['disease', 'rev_disease_protein', 'gene/protein'].edge_index = edges(N_DISEASE, N_GENE, 40)
    G['disease', 'disease_phenotype_positive', 'effect/phenotype'].edge_index = edges(N_DISEASE, N_PHENO, 35)
    G['disease', 'rev_exposure_disease', 'exposure'].edge_index = edges(N_DISEASE, N_EXPOSURE, 18)
    G['gene/protein', 'protein_protein', 'gene/protein'].edge_index = edges(N_GENE, N_GENE, 50)

    return initialize_node_embedding(G, HID)


def make_model(G, agg_measure, **kw):
    params = dict(attention=False, proto=True, proto_num=3, sim_measure='all_nodes_profile',
                  bert_measure='disease_name', num_walks=10, walk_mode='bit', path_length=2,
                  split='random', data_folder='.', exp_lambda=0.7, device=torch.device('cpu'))
    params.update(kw)
    return HeteroRGCN(G, in_size=HID, hidden_size=HID, out_size=HID,
                      agg_measure=agg_measure, **params)


def negative_graph(G, seed=1):
    """'fix_dst'-style corruption: keep sources, resample destinations."""
    g = torch.Generator().manual_seed(seed)
    neg = HeteroData()
    for nt in G.node_types:
        neg[nt].num_nodes = G[nt].num_nodes
        neg[nt].inp = G[nt].inp
    for et in G.edge_types:
        ei = G[et].edge_index
        neg[et].edge_index = torch.stack(
            [ei[0], torch.randint(0, G[et[2]].num_nodes, (ei.shape[1],), generator=g)])
    return neg


# ---------------------------------------------------------------------------------------------

def test_profile_blocks(G):
    print('\n[1] per-block signature construction')
    blocks = obtain_disease_profile_blocks(G, torch.tensor(0), BLOCK_ETYPES, BLOCK_NODES)
    check('returns one vector per block', len(blocks) == N_SIM_BLOCKS,
          '(got %d)' % len(blocks))
    widths = [b.shape[0] for b in blocks]
    check('block widths match node-type counts',
          widths == [G[nt].num_nodes for nt in BLOCK_NODES], '(got %s)' % widths)
    flat = obtain_disease_profile(G, torch.tensor(0), BLOCK_ETYPES, BLOCK_NODES)
    check('flat view == concat of blocks', torch.equal(flat, torch.cat(blocks)))


def test_registration(G):
    print('\n[2] parameter registration (the agg_measure="learn" failure mode)')
    m = make_model(G, 'gumbel_block')
    names = [n for n, _ in m.named_parameters()]
    gate_names = [n for n in names if 'gate_mlps' in n]
    check('gate params in model.parameters()', len(gate_names) > 0)
    sd = m.state_dict()
    check('gate params in state_dict()', any('gate_mlps' in k for k in sd))
    check('gumbel_step buffer in state_dict()', any('gumbel_step' in k for k in sd))

    # The bug this guards against: params reachable as attributes but absent from parameters().
    n_gate_params = sum(p.numel() for n, p in m.named_parameters() if 'gate_mlps' in n)
    direct = sum(p.numel() for p in m.pred.gate_mlps.parameters())
    check('no gate parameters lost from the registry', n_gate_params == direct,
          '(registry %d vs module %d)' % (n_gate_params, direct))

    m_learn = make_model(G, 'learn')
    check('agg_measure="learn" W_gate now registered',
          any('W_gate' in n for n, _ in m_learn.named_parameters()))


def test_rarity_checkpoint_compat(G):
    print('\n[3] backward compatibility of existing rarity checkpoints')
    m_rarity = make_model(G, 'rarity')
    keys = set(m_rarity.state_dict().keys())
    check('rarity has no gate_mlps keys', not any('gate_mlps' in k for k in keys))
    check('rarity has no W_gate keys', not any('W_gate' in k for k in keys))
    check('rarity has no gumbel_step key', not any('gumbel_step' in k for k in keys))
    # A checkpoint saved by rarity must load into a freshly built rarity model, strictly.
    try:
        make_model(G, 'rarity').load_state_dict(m_rarity.state_dict())
        check('rarity state_dict loads strictly', True)
    except Exception as exc:
        check('rarity state_dict loads strictly', False, '(%s)' % exc)


def test_forward_and_shapes(G):
    print('\n[4] forward pass, shapes, and gradient flow')
    neg = negative_graph(G)
    for agg in ('rarity', 'rarity_4block', 'gumbel_block'):
        m = make_model(G, agg)
        m.train()
        scores, scores_neg, _, _ = m(G, neg, pretrain_mode=False, mode='train')
        dd = [('drug', r, 'disease') for r in DD_RELATIONS]
        ok = all(e in scores for e in dd)
        check('%-14s produces scores for all dd etypes' % agg, ok)
        loss = torch.cat([scores[e] for e in dd]).sigmoid().mean()
        loss = loss - 0.01 * m.pop_gate_regularizer()
        loss.backward()
        if agg == 'gumbel_block':
            grads = [p.grad for n, p in m.named_parameters() if 'gate_mlps' in n]
            check('gumbel gate receives gradient',
                  any(gr is not None and gr.abs().sum() > 0 for gr in grads))

    m = make_model(G, 'gumbel_block')
    check('sim_blocks is [4, D, D]',
          m.pred.sim_blocks.shape[0] == N_SIM_BLOCKS
          and m.pred.sim_blocks.shape[1] == m.pred.sim_blocks.shape[2],
          '(got %s)' % (tuple(m.pred.sim_blocks.shape),))


def test_zero_shot_unseen_disease(G):
    """The eval path: query diseases that have no drug-disease edge in the training graph.

    Regression test -- the precomputed [4, D, D] tensor only covers diseases present at init, so
    a zero-shot disease is absent from the index map and must fall back to on-demand signatures.
    """
    print('\n[5] zero-shot / unseen-disease eval path')

    # Training graph: drug-disease edges for the FIRST 8 diseases only.
    G_train = build_synthetic_graph()
    held_out = list(range(8, N_DISEASE))
    seen = torch.arange(8)
    for rel in DD_RELATIONS:
        drugs = torch.randint(0, N_DRUG, (8,), generator=torch.Generator().manual_seed(3))
        G_train['drug', rel, 'disease'].edge_index = torch.stack([drugs, seen])
        G_train['disease', 'rev_' + rel, 'drug'].edge_index = torch.stack([seen, drugs])

    for agg in ('rarity_4block', 'gumbel_block'):
        m = make_model(G_train, agg)
        check('%-14s precomputes only the seen diseases' % agg,
              m.pred.sim_blocks.shape[1] == 8, '(got %d)' % m.pred.sim_blocks.shape[1])

        # Eval graph containing the held-out diseases, exactly as evaluate_fb builds it.
        g_eval = HeteroData()
        for nt in G_train.node_types:
            g_eval[nt].num_nodes = G_train[nt].num_nodes
            g_eval[nt].inp = G_train[nt].inp
        ho = torch.tensor(held_out)
        drugs = torch.zeros(len(held_out), dtype=torch.long)
        for rel in DD_RELATIONS:
            g_eval['drug', rel, 'disease'].edge_index = torch.stack([drugs, ho])
            g_eval['disease', 'rev_' + rel, 'drug'].edge_index = torch.stack([ho, drugs])

        m.eval()
        try:
            with torch.no_grad():
                scores, _, _, _ = m(G_train, g_eval, eval_pos_G=g_eval,
                                    pretrain_mode=False, mode='test')
            ok = all(torch.isfinite(v).all() for v in scores.values())
            check('%-14s scores unseen diseases without KeyError' % agg, ok)
        except Exception as exc:
            check('%-14s scores unseen diseases without KeyError' % agg, False,
                  '(%s: %s)' % (type(exc).__name__, exc))


def test_gate_semantics(G):
    print('\n[5] gate output shape and empty-gate fallback')
    m = make_model(G, 'gumbel_block')
    m.eval()
    n_q = 5
    h_query = torch.randn(n_q, HID)
    out_blocks = torch.randn(n_q, N_SIM_BLOCKS, HID)

    out = m.pred._gumbel_block_gate(h_query, out_blocks)
    check('gate output is [N_q, hidden]', out.shape == (n_q, HID), '(got %s)' % (tuple(out.shape),))

    # Force every gate closed -> must fall back to the disease's own embedding.
    with torch.no_grad():
        for mlp in m.pred.gate_mlps:
            last = [l for l in mlp if isinstance(l, torch.nn.Linear)][-1]
            last.weight.zero_()
            last.bias.copy_(torch.tensor([10.0, -10.0]))   # logit[0] >> logit[1] => "don't use"
    out_closed = m.pred._gumbel_block_gate(h_query, out_blocks)
    check('all gates closed -> own embedding', torch.allclose(out_closed, h_query, atol=1e-6))

    # Force every gate open -> must be the MEAN of the blocks, not the sum.
    with torch.no_grad():
        for mlp in m.pred.gate_mlps:
            last = [l for l in mlp if isinstance(l, torch.nn.Linear)][-1]
            last.bias.copy_(torch.tensor([-10.0, 10.0]))
    out_open = m.pred._gumbel_block_gate(h_query, out_blocks)
    check('all gates open -> mean over blocks (not sum)',
          torch.allclose(out_open, out_blocks.mean(dim=1), atol=1e-6))


def test_temperature_schedule(G):
    print('\n[6] temperature annealing')
    m = make_model(G, 'gumbel_block', gumbel_tau_start=1.0, gumbel_tau_end=0.3,
                   gumbel_anneal_steps=10)
    p = m.pred

    m.eval()
    check('eval mode pins tau to tau_end', abs(p._current_tau() - 0.3) < 1e-9)

    m.train()
    check('training starts at tau_start', abs(p._current_tau() - 1.0) < 1e-9)

    neg = negative_graph(G)
    for _ in range(4):
        m(G, neg, pretrain_mode=False, mode='train')
    check('step counter advances once per optimizer step (not per forward)',
          int(p.gumbel_step.item()) == 4, '(got %d)' % int(p.gumbel_step.item()))
    tau_mid = p._current_tau()
    check('tau anneals downward', 0.3 < tau_mid < 1.0, '(got %.4f)' % tau_mid)

    for _ in range(20):
        m(G, neg, pretrain_mode=False, mode='train')
    check('tau clamps at tau_end', abs(p._current_tau() - 0.3) < 1e-9,
          '(got %.4f)' % p._current_tau())

    p.gumbel_frozen = True
    check('gumbel_frozen pins tau to tau_end (Explainer)', abs(p._current_tau() - 0.3) < 1e-9)
    before = int(p.gumbel_step.item())
    m(G, neg, pretrain_mode=False, mode='train')
    check('gumbel_frozen halts the step counter', int(p.gumbel_step.item()) == before)
    p.gumbel_frozen = False

    # Pretraining must not advance the gate at all -- it never reaches the similarity block.
    m2 = make_model(G, 'gumbel_block')
    m2.train()
    m2.forward_minibatch(G, neg, G, mode='train', pretrain_mode=True)
    check('pretrain_mode does not advance the gate', int(m2.pred.gumbel_step.item()) == 0)


def test_config_persistence(G):
    print('\n[7] config persistence through save_model / load_pretrained')
    from txgnn import TxGNN

    class FakeData:
        pass

    rows = []
    for rel in DD_RELATIONS:
        for d in range(N_DISEASE):
            rows.append({'x_idx': d % N_DRUG, 'relation': rel, 'y_idx': d,
                         'x_type': 'drug', 'y_type': 'disease',
                         'x_id': str(d % N_DRUG), 'y_id': str(d)})
    df = pd.DataFrame(rows)

    data = FakeData()
    data.G = G
    data.df = data.df_train = data.df_valid = data.df_test = df
    data.data_folder = tempfile.mkdtemp()
    data.disease_eval_idx = None
    data.split = 'random'
    data.no_kg = False

    tmp = tempfile.mkdtemp()
    try:
        model = TxGNN(data=data, weight_bias_track=False, device='cpu')
        model.model_initialize(n_hid=HID, n_inp=HID, n_out=HID, proto=True, proto_num=3,
                              sim_measure='all_nodes_profile', agg_measure='gumbel_block',
                              exp_lambda=0.55, gumbel_tau_start=0.9, gumbel_tau_end=0.15,
                              gumbel_anneal_steps=777, gumbel_hidden=8,
                              gumbel_entropy_weight=0.02)
        for key, want in [('exp_lambda', 0.55), ('gumbel_tau_start', 0.9),
                          ('gumbel_tau_end', 0.15), ('gumbel_anneal_steps', 777),
                          ('gumbel_hidden', 8), ('gumbel_entropy_weight', 0.02)]:
            check('config carries %s' % key, model.config.get(key) == want,
                  '(got %r)' % model.config.get(key))

        out_dir = os.path.join(tmp, 'ckpt')
        model.save_model(out_dir)

        reloaded = TxGNN(data=data, weight_bias_track=False, device='cpu')
        reloaded.load_pretrained(out_dir)
        check('exp_lambda survives reload (was silently reset before)',
              abs(reloaded.model.pred.exp_lambda - 0.55) < 1e-9,
              '(got %r)' % reloaded.model.pred.exp_lambda)
        check('tau_end survives reload', abs(reloaded.model.pred.gumbel_tau_end - 0.15) < 1e-9)
        check('anneal_steps survives reload', reloaded.model.pred.gumbel_anneal_steps == 777)

        a = model.model.pred.gate_mlps[0][0].weight
        b = reloaded.model.pred.gate_mlps[0][0].weight
        check('gate weights survive the state_dict round-trip', torch.allclose(a, b))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(data.data_folder, ignore_errors=True)


def test_invalid_agg_measure(G):
    print('\n[8] loud failure on unknown agg_measure')
    try:
        make_model(G, 'not_a_real_measure')
        check('unknown agg_measure raises', False, '(no exception)')
    except ValueError:
        check('unknown agg_measure raises ValueError', True)


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    G = build_synthetic_graph()

    test_profile_blocks(G)
    test_registration(G)
    test_rarity_checkpoint_compat(G)
    test_forward_and_shapes(G)
    test_zero_shot_unseen_disease(G)
    test_gate_semantics(G)
    test_temperature_schedule(G)
    test_config_persistence(G)
    test_invalid_agg_measure(G)

    print('\n' + '=' * 60)
    if _failures:
        print('FAILED (%d): %s' % (len(_failures), ', '.join(_failures)))
        return 1
    print('All checks passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
