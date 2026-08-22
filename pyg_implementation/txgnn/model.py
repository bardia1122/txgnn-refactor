import math
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import pandas as pd
import copy
import os
import random

import warnings
warnings.filterwarnings("ignore")

from torch_geometric.data import HeteroData
from torch_geometric.utils import degree, softmax
from torch_scatter import scatter

from .utils import sim_matrix, exponential, obtain_disease_profile, obtain_disease_profile_blocks, obtain_protein_random_walk_profile, convert2str
from .graphmask.multiple_inputs_layernorm_linear import MultipleInputsLayernormLinear
from .graphmask.squeezer import Squeezer
from .graphmask.sigmoid_penalty import SoftConcrete


# ---------------------------------------------------------------------------------------------
# Disease-signature blocks.
#
# Single source of truth for the four similarity blocks. Previously these lists were duplicated
# in DistMultPredictor.__init__ and again in the unseen-disease fallback inside forward(); any
# edit had to be made in both places or the two would silently disagree. Order defines the block
# index used everywhere downstream (per-block similarity, per-block gates, logging).
#
#   block 0: disease  - disease          (disease_disease)
#   block 1: gene / protein              (rev_disease_protein)
#   block 2: effect / phenotype          (disease_phenotype_positive)
#   block 3: exposure                    (rev_exposure_disease)
#
# NOTE: this preserves the pre-existing `disease_etypes_all` / `disease_nodes_all` ordering, which
# is NOT the order the blocks are usually named in prose (gene/protein, phenotype, exposure,
# disease-disease). Index 0 is disease-disease, not gene/protein.
# ---------------------------------------------------------------------------------------------
DISEASE_SIM_BLOCKS = (
    ('disease_disease',            'disease'),
    ('rev_disease_protein',        'gene/protein'),
    ('disease_phenotype_positive', 'effect/phenotype'),
    ('rev_exposure_disease',       'exposure'),
)
BLOCK_ETYPES = [b[0] for b in DISEASE_SIM_BLOCKS]
BLOCK_NODES  = [b[1] for b in DISEASE_SIM_BLOCKS]
N_SIM_BLOCKS = len(DISEASE_SIM_BLOCKS)

# The two-block subset the legacy `all_nodes_profile` sim_measure uses.
LEGACY_ETYPES = ['disease_disease', 'rev_disease_protein']
LEGACY_NODES  = ['disease', 'gene/protein']

# agg_measure values that operate on the four separate blocks rather than one pre-blended vector.
BLOCK_AGG_MEASURES = ('rarity_4block', 'gumbel_block')

# agg_measure values that are valid at all -- used to fail loudly on typos instead of dying later
# with an undefined-variable NameError (the pre-existing failure mode of the if/elif chains).
VALID_AGG_MEASURES = ('learn', 'heuristics-0.8', 'avg', 'rarity', '100proto') + BLOCK_AGG_MEASURES

# sim_measure values whose profile is a concatenation of graph-neighbourhood blocks, and which can
# therefore be split per block. 'bert' / 'profile+bert' embeddings have no block structure.
BLOCK_COMPATIBLE_SIM_MEASURES = ('all_nodes_profile', 'all_nodes_profile_more')


class DistMultPredictor(nn.Module):
    def __init__(self, n_hid, w_rels, G, rel2idx, proto, proto_num, sim_measure, bert_measure, agg_measure, num_walks, walk_mode, path_length, split, data_folder, exp_lambda, device,
                 gumbel_tau_start = 1.0, gumbel_tau_end = 0.3, gumbel_anneal_steps = 1000,
                 gumbel_hidden = 64, gumbel_entropy_weight = 0.01):
        super().__init__()

        if agg_measure not in VALID_AGG_MEASURES:
            raise ValueError("Unknown agg_measure %r. Valid options: %s" % (agg_measure, ', '.join(VALID_AGG_MEASURES)))

        self.proto = proto
        self.sim_measure = sim_measure
        self.bert_measure = bert_measure
        self.agg_measure = agg_measure
        self.num_walks = num_walks
        self.walk_mode = walk_mode
        self.path_length = path_length
        self.exp_lambda = exp_lambda
        self.device = device
        self.W = w_rels
        self.rel2idx = rel2idx

        # --- per-block similarity mode -----------------------------------------------------
        # 'rarity_4block' and 'gumbel_block' compare the four signature blocks separately instead
        # of cosine-ing one concatenated vector, so they need the per-block similarity tensors.
        self.block_mode = agg_measure in BLOCK_AGG_MEASURES
        if self.block_mode and sim_measure not in BLOCK_COMPATIBLE_SIM_MEASURES:
            raise ValueError(
                "agg_measure=%r needs a block-structured signature, but sim_measure=%r has none. "
                "Use one of: %s" % (agg_measure, sim_measure, ', '.join(BLOCK_COMPATIBLE_SIM_MEASURES)))

        # Gumbel gate hyperparameters. Kept as plain attributes; the schedule is driven by
        # self.gumbel_step (a registered buffer, so it survives save/load).
        self.gumbel_tau_start = gumbel_tau_start
        self.gumbel_tau_end = gumbel_tau_end
        self.gumbel_anneal_steps = gumbel_anneal_steps
        self.gumbel_entropy_weight = gumbel_entropy_weight
        # When True: never anneal, never advance the step counter, always take the deterministic
        # (argmax) gate. Set by graphmask_forward so Explainer runs are not "mid-anneal".
        self.gumbel_frozen = False
        # Entropy terms accumulated during forward() and drained by pop_gate_regularizer().
        self._gate_reg_terms = []
        # Diagnostics: set record_gates=True to have forward() log every gate decision as
        # (etype, disease_ids, gate_matrix[N_q, N_SIM_BLOCKS]) into self.gate_log. Off by default
        # -- it holds one [N_q, 4] tensor per etype per forward.
        self.record_gates = False
        self.gate_log = []

        self.etypes_dd = [('drug', 'contraindication', 'disease'),
                           ('drug', 'indication', 'disease'),
                           ('drug', 'off-label use', 'disease'),
                           ('disease', 'rev_contraindication', 'drug'),
                           ('disease', 'rev_indication', 'drug'),
                           ('disease', 'rev_off-label use', 'drug')]

        self.node_types_dd = ['disease', 'drug']

        if proto:
            # Built ONLY for agg_measure='learn'. Previously this was constructed unconditionally
            # as a plain python dict, so its nn.Linear layers appeared in neither model.parameters()
            # nor state_dict() -- 'learn' could never actually train. It is now an nn.ModuleDict, so
            # it registers properly; scoping it to 'learn' keeps state_dict keys unchanged for the
            # other agg_measures, so existing 'rarity' checkpoints still load.
            if agg_measure == 'learn':
                w_gate = {}
                for i in self.node_types_dd:
                    temp_w = nn.Linear(n_hid * 2, 1)
                    nn.init.xavier_uniform_(temp_w.weight)
                    w_gate[i] = temp_w
                self.W_gate = nn.ModuleDict(w_gate)
            self.k = proto_num
            self.m = nn.Sigmoid()

            # Gumbel decision network: one small MLP per block, mapping
            # [h_disease_query || h_block_pooled] -> 2 logits (don't-use / use).
            # nn.ModuleList so the parameters are registered (see the W_gate note above).
            if agg_measure == 'gumbel_block':
                self.gate_mlps = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(n_hid * 2, gumbel_hidden),
                        nn.ReLU(),
                        nn.Linear(gumbel_hidden, 2),
                    ) for _ in range(N_SIM_BLOCKS)
                ])
                for mlp in self.gate_mlps:
                    for layer in mlp:
                        if isinstance(layer, nn.Linear):
                            nn.init.xavier_uniform_(layer.weight)
                            nn.init.zeros_(layer.bias)
                # Optimizer step counter driving the temperature anneal. Registered as a buffer so
                # it round-trips through save_model/load_pretrained and a resumed run does not
                # restart the anneal from tau_start.
                self.register_buffer('gumbel_step', torch.zeros(1, dtype=torch.long))

            if sim_measure in ['bert', 'profile+bert']:

                data_path = os.path.join(data_folder, 'kg.csv')

                if os.path.exists(data_path):
                    df = pd.read_csv(data_path)

                self.disease_dict = dict(df[df.x_type == 'disease'][['x_idx', 'x_id']].values)
                self.disease_dict.update(dict(df[df.y_type == 'disease'][['y_idx', 'y_id']].values))

                if bert_measure == 'disease_name':
                    self.bert_embed = np.load('/n/scratch3/users/k/kh278/bert_basic.npy')
                    df_nodes_bert = pd.read_csv('/n/scratch3/users/k/kh278/nodes.csv')

                elif bert_measure == 'v1':
                    self.bert_embed = np.load('/n/scratch3/users/k/kh278/disease_embeds_single_def.npy')
                    df_nodes_bert = pd.read_csv('/n/scratch3/users/k/kh278/disease_nodes_for_BERT_embeds.csv')

                df_nodes_bert['node_id'] = df_nodes_bert.node_id.apply(lambda x: convert2str(x))
                self.id2bertindex = dict(zip(df_nodes_bert.node_id.values, df_nodes_bert.index.values))

            self.diseases_profile = {}
            self.sim_all_etypes = {}
            self.diseaseid2id_etypes = {}
            self.diseases_profile_etypes = {}

            disease_etypes_all = BLOCK_ETYPES
            disease_nodes_all = BLOCK_NODES

            disease_etypes = LEGACY_ETYPES
            disease_nodes = LEGACY_NODES

            if self.block_mode:
                self._build_block_similarity(G)

            for etype in self.etypes_dd:
                if self.block_mode:
                    # Per-block mode uses a single shared similarity tensor (see
                    # _build_block_similarity); the per-etype flat matrices below are not built.
                    break
                if etype not in G.edge_types:
                    continue
                src, dst = etype[0], etype[2]
                if src == 'disease':
                    out_deg = degree(G[etype].edge_index[0], num_nodes=G[src].num_nodes)
                    all_disease_ids = torch.where(out_deg != 0)[0]
                elif dst == 'disease':
                    in_deg = degree(G[etype].edge_index[1], num_nodes=G[dst].num_nodes)
                    all_disease_ids = torch.where(in_deg != 0)[0]

                if sim_measure == 'all_nodes_profile':
                    diseases_profile = {i.item(): obtain_disease_profile(G, i, disease_etypes, disease_nodes) for i in all_disease_ids}
                elif sim_measure == 'all_nodes_profile_more':
                    diseases_profile = {i.item(): obtain_disease_profile(G, i, disease_etypes_all, disease_nodes_all) for i in all_disease_ids}
                elif sim_measure == 'protein_profile':
                    diseases_profile = {i.item(): obtain_disease_profile(G, i, ['rev_disease_protein'], ['gene/protein']) for i in all_disease_ids}
                elif sim_measure == 'protein_random_walk':
                    diseases_profile = {i.item(): obtain_protein_random_walk_profile(i, num_walks, path_length, G, disease_etypes, disease_nodes, walk_mode) for i in all_disease_ids}
                elif sim_measure == 'bert':
                    diseases_profile = {i.item(): torch.Tensor(self.bert_embed[self.id2bertindex[self.disease_dict[i.item()]]]) for i in all_disease_ids}
                elif sim_measure == 'profile+bert':
                    diseases_profile = {i.item(): torch.cat((obtain_disease_profile(G, i, disease_etypes, disease_nodes), torch.Tensor(self.bert_embed[self.id2bertindex[self.disease_dict[i.item()]]]))) for i in all_disease_ids}

                diseaseid2id = dict(zip(all_disease_ids.detach().cpu().numpy(), range(len(all_disease_ids))))
                disease_profile_tensor = torch.stack([diseases_profile[i.item()] for i in all_disease_ids])
                sim_all = sim_matrix(disease_profile_tensor, disease_profile_tensor)

                self.sim_all_etypes[etype] = sim_all
                self.diseaseid2id_etypes[etype] = diseaseid2id
                self.diseases_profile_etypes[etype] = diseases_profile

    def _build_block_similarity(self, G):
        """Precompute one [N_SIM_BLOCKS, D, D] cosine-similarity tensor shared by all dd-etypes.

        The six drug-disease etypes address the same disease node set, so a per-etype copy would
        store six near-identical [4, D, D] tensors. This verifies that assumption explicitly and
        raises if it does not hold -- a silent per-etype fallback would quietly change the memory
        and host<->device-sync characteristics partway through a long run.
        Result lives on self.device so the per-step CPU->GPU copy of the old path disappears.
        """
        disease_sets = {}
        for etype in self.etypes_dd:
            if etype not in G.edge_types:
                continue
            src, dst = etype[0], etype[2]
            if src == 'disease':
                deg = degree(G[etype].edge_index[0], num_nodes=G[src].num_nodes)
            elif dst == 'disease':
                deg = degree(G[etype].edge_index[1], num_nodes=G[dst].num_nodes)
            else:
                continue
            disease_sets[etype] = torch.where(deg != 0)[0]

        if not disease_sets:
            raise ValueError('No drug-disease etypes found in G; cannot build per-block similarity.')

        etypes_present = list(disease_sets.keys())
        reference_etype = etypes_present[0]
        all_disease_ids = disease_sets[reference_etype]
        for etype in etypes_present[1:]:
            other = disease_sets[etype]
            if other.shape != all_disease_ids.shape or not torch.equal(other, all_disease_ids):
                raise ValueError(
                    'Per-block similarity requires all drug-disease etypes to address an identical '
                    'disease node set, but %s covers %d diseases and %s covers %d (or the ids '
                    'differ). Sharing one similarity tensor would be wrong here.'
                    % (reference_etype, len(all_disease_ids), etype, len(other)))

        D = len(all_disease_ids)
        print('[block-sim] verified shared disease set across %d etypes, size D_e = %d'
              % (len(etypes_present), D))

        # Per-block binary signatures -> per-block cosine similarity.
        # The per-disease profiles are cached (CPU) so diseases unseen at init -- every zero-shot
        # test disease, which by construction has no drug-disease edge in the training graph --
        # can be scored lazily in _block_sim_fallback().
        self.block_profiles = {}
        per_block_profiles = [[] for _ in range(N_SIM_BLOCKS)]
        for i in all_disease_ids:
            blocks = obtain_disease_profile_blocks(G, i, BLOCK_ETYPES, BLOCK_NODES)
            self.block_profiles[i.item()] = blocks
            for b, vec in enumerate(blocks):
                per_block_profiles[b].append(vec)

        sims = []
        for b in range(N_SIM_BLOCKS):
            profile_tensor = torch.stack(per_block_profiles[b])          # [D, |V_block|]
            sims.append(sim_matrix(profile_tensor, profile_tensor))      # [D, D]
            print('[block-sim]   block %d (%s via %s): profile dim %d'
                  % (b, BLOCK_NODES[b], BLOCK_ETYPES[b], profile_tensor.shape[1]))

        sim_blocks = torch.stack(sims).to(self.device)                   # [4, D, D]
        self.register_buffer('sim_blocks', sim_blocks, persistent=False)
        self.block_disease_ids = all_disease_ids
        self.block_diseaseid2id = dict(zip(all_disease_ids.detach().cpu().numpy(), range(D)))
        print('[block-sim] similarity tensor %s on %s (%.1f MB)'
              % (tuple(sim_blocks.shape), self.device,
                 sim_blocks.numel() * sim_blocks.element_size() / 1e6))

    def _block_sim_fallback(self, G, query_ids, key_ids):
        """Per-block similarity for diseases absent from the precomputed tensor.

        Mirrors the legacy `except:` path: zero-shot test diseases have no drug-disease edge in
        the training graph, so they are not in block_diseaseid2id and their signature has to be
        built on demand. Profiles are cached; the similarity itself is recomputed per call.
        Returns [N_SIM_BLOCKS, len(query_ids), len(key_ids)] on self.device.
        """
        for i in list(query_ids) + list(key_ids):
            if i not in self.block_profiles:
                self.block_profiles[i] = obtain_disease_profile_blocks(
                    G, torch.tensor(i), BLOCK_ETYPES, BLOCK_NODES)

        sims = []
        for b in range(N_SIM_BLOCKS):
            q = torch.stack([self.block_profiles[i][b] for i in query_ids])
            k = torch.stack([self.block_profiles[i][b] for i in key_ids])
            sims.append(sim_matrix(q, k))
        return torch.stack(sims).to(self.device)

    def _current_tau(self):
        """Gumbel temperature for this forward pass.

        Linearly annealed tau_start -> tau_end over gumbel_anneal_steps optimizer steps while
        training. Eval and Explainer (gumbel_frozen) runs pin tau to tau_end so they are never
        observed mid-anneal.
        """
        if self.gumbel_frozen or not self.training:
            return self.gumbel_tau_end
        steps = max(int(self.gumbel_anneal_steps), 1)
        frac = min(float(self.gumbel_step.item()) / steps, 1.0)
        return self.gumbel_tau_start + (self.gumbel_tau_end - self.gumbel_tau_start) * frac

    def _gumbel_block_gate(self, h_query, out_blocks):
        """Independent binary Gumbel gate per similarity block.

        h_query    [N_q, d]     the disease's own encoded embedding
        out_blocks [N_q, 4, d]  top-k pooled neighbour embedding, one per block
        returns    [N_q, d]     mean over the blocks whose gate fired, falling back to h_query
                                for diseases where no gate fired.
        Four independent 2-way decisions, NOT one softmax over blocks, so blocks do not compete.
        """
        n_q = h_query.shape[0]
        tau = self._current_tau()
        deterministic = self.gumbel_frozen or not self.training

        logits = torch.stack([
            self.gate_mlps[b](torch.cat((h_query, out_blocks[:, b, :]), dim=1))
            for b in range(N_SIM_BLOCKS)
        ], dim=1)                                                    # [N_q, 4, 2]

        if deterministic:
            # argmax, no Gumbel noise -- eval must be reproducible.
            gate = (logits[..., 1] > logits[..., 0]).float()          # [N_q, 4]
        else:
            # hard=True gives a one-hot forward pass with straight-through gradients.
            gate = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)[..., 1]

            # Entropy of the (noise-free) gate distribution, averaged over diseases and blocks.
            # Maximising it discourages premature collapse; finetune() subtracts it from the loss.
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs.clamp(min=1e-9))).sum(dim=-1).mean()
            self._gate_reg_terms.append(entropy)

        self._last_gate = gate.detach()

        n_open = gate.sum(dim=1, keepdim=True)                        # [N_q, 1]
        # MEAN over open gates, not sum: a sum lets the embedding norm grow with the number of
        # active blocks, which destabilises the bilinear DistMult score in _apply_edges.
        pooled = (gate.unsqueeze(-1) * out_blocks).sum(dim=1) / n_open.clamp(min=1.0)
        # No block selected -> keep the disease's own embedding (the "none" choice).
        return torch.where(n_open > 0, pooled, h_query)

    def pop_gate_regularizer(self):
        """Drain the entropy terms accumulated during forward(); returns a scalar tensor.

        Returns 0.0 when the Gumbel gate is not in use, so the training loop can call this
        unconditionally for any agg_measure.
        """
        if not self._gate_reg_terms:
            return torch.zeros((), device=self.device)
        reg = torch.stack(self._gate_reg_terms).mean()
        self._gate_reg_terms = []
        return reg

    def _apply_edges(self, graph, etype, h):
        """Inline DistMult scoring: score = sum(h_u * h_r * h_v, dim=1)"""
        src, rel, dst = etype
        ei = graph[etype].edge_index
        src_nodes = ei[0]
        dst_nodes = ei[1]
        h_u = h[src][src_nodes]
        h_v = h[dst][dst_nodes]
        rel_idx = self.rel2idx[etype]
        h_r = self.W[rel_idx]
        score = torch.sum(h_u * h_r * h_v, dim=1)
        return score

    def _etype_has_sim(self, etype):
        """Whether disease-similarity augmentation is available for this etype.

        Block mode keeps one shared similarity tensor instead of the per-etype dict, so the old
        `etype in self.sim_all_etypes` test would be False for every etype and silently disable
        the whole mechanism.
        """
        if self.block_mode:
            return etype in self.etypes_dd and hasattr(self, 'sim_blocks')
        return etype in self.sim_all_etypes

    def forward(self, graph, G, h, pretrain_mode, mode, block=None, only_relation=None):
        scores = {}
        s_l = []

        # One optimizer step drives two pred() calls (positive graph, then negative). Advance the
        # anneal on the positive pass only, so gumbel_step counts optimizer steps, not forwards.
        if (self.agg_measure == 'gumbel_block' and self.training and not self.gumbel_frozen
                and not pretrain_mode and mode == 'train_pos'):
            self.gumbel_step += 1

        if len(graph.edge_types) == 1:
            etypes_train = graph.edge_types
        else:
            etypes_train = self.etypes_dd

        if only_relation is not None:
            if only_relation == 'indication':
                etypes_train = [('drug', 'indication', 'disease'),
                                ('disease', 'rev_indication', 'drug')]
            elif only_relation == 'contraindication':
                etypes_train = [('drug', 'contraindication', 'disease'),
                               ('disease', 'rev_contraindication', 'drug')]
            elif only_relation == 'off-label':
                etypes_train = [('drug', 'off-label use', 'disease'),
                               ('disease', 'rev_off-label use', 'drug')]
            else:
                return ValueError

        if pretrain_mode:
            # during pretraining....
            etypes_all = [et for et in graph.edge_types if graph[et].num_edges > 0]
            for etype in etypes_all:
                out = torch.sigmoid(self._apply_edges(graph, etype, h))
                s_l.append(out)
                scores[etype] = out
        else:
            # finetuning on drug disease only...

            for etype in etypes_train:

                if self.proto and self._etype_has_sim(etype):
                    src, dst = etype[0], etype[2]
                    graph_out_deg = degree(graph[etype].edge_index[0], num_nodes=graph[src].num_nodes)
                    graph_in_deg = degree(graph[etype].edge_index[1], num_nodes=graph[dst].num_nodes)
                    src_rel_idx = torch.where(graph_out_deg != 0)
                    dst_rel_idx = torch.where(graph_in_deg != 0)
                    src_h = h[src][src_rel_idx]
                    dst_h = h[dst][dst_rel_idx]

                    G_out_deg = degree(G[etype].edge_index[0], num_nodes=G[src].num_nodes)
                    G_in_deg = degree(G[etype].edge_index[1], num_nodes=G[dst].num_nodes)
                    src_rel_ids_keys = torch.where(G_out_deg != 0)
                    dst_rel_ids_keys = torch.where(G_in_deg != 0)
                    src_h_keys = h[src][src_rel_ids_keys]
                    dst_h_keys = h[dst][dst_rel_ids_keys]

                    h_disease = {}

                    if src == 'disease':
                        h_disease['disease_query'] = src_h
                        h_disease['disease_key'] = src_h_keys
                        h_disease['disease_query_id'] = src_rel_idx
                        h_disease['disease_key_id'] = src_rel_ids_keys
                    elif dst == 'disease':
                        h_disease['disease_query'] = dst_h
                        h_disease['disease_key'] = dst_h_keys
                        h_disease['disease_query_id'] = dst_rel_idx
                        h_disease['disease_key_id'] = dst_rel_ids_keys

                    if self.block_mode:
                        # ---- per-block similarity path -------------------------------------
                        # sim_blocks is [4, D, D]; columns are aligned with `disease_key` because
                        # the key set IS block_disease_ids (verified identical across etypes at
                        # init by _build_block_similarity).
                        query_ids = [i.item() for i in h_disease['disease_query_id'][0]]
                        if all(i in self.block_diseaseid2id for i in query_ids):
                            query_rows = torch.as_tensor(
                                [self.block_diseaseid2id[i] for i in query_ids],
                                dtype=torch.long, device=self.sim_blocks.device)
                            sim_b = self.sim_blocks[:, query_rows, :]          # [4, N_q, D]
                        else:
                            # Zero-shot diseases: absent from the precomputed tensor by
                            # construction, so build their block signatures on demand.
                            key_ids = [i.item() for i in h_disease['disease_key_id'][0]]
                            sim_b = self._block_sim_fallback(G, query_ids, key_ids)

                        # Same train/eval top-k convention as the legacy path: when the query set
                        # equals the key set every disease matches itself at rank 0, so drop it.
                        # (Deliberately left on the pre-existing shape-equality check -- the Gumbel
                        # hard/soft switch uses an explicit self.training flag instead.)
                        if src_h.shape[0] == src_h_keys.shape[0]:
                            topk_vals, topk_idx = torch.topk(sim_b, self.k + 1, dim=-1)
                            topk_vals, topk_idx = topk_vals[..., 1:], topk_idx[..., 1:]
                        else:
                            topk_vals, topk_idx = torch.topk(sim_b, self.k, dim=-1)

                        coef_b = F.normalize(topk_vals, p=1, dim=-1)           # [4, N_q, k]
                        embed_b = h_disease['disease_key'][topk_idx]           # [4, N_q, k, d]
                        # [4, N_q, d] -> [N_q, 4, d] so the gate is indexed per disease.
                        out_blocks = (embed_b * coef_b.unsqueeze(-1)).sum(dim=2).permute(1, 0, 2)

                        if self.agg_measure == 'gumbel_block':
                            out = self._gumbel_block_gate(h_disease['disease_query'], out_blocks)
                            if self.record_gates:
                                self.gate_log.append((etype, list(query_ids), self._last_gate.cpu()))
                        else:
                            # 'rarity_4block': fixed, unlearned recombination. Isolates "more
                            # signal" from "smarter gating" as separate contributions.
                            out = out_blocks.mean(dim=1)                       # [N_q, d]

                    elif self.sim_measure in ['protein_profile', 'all_nodes_profile', 'protein_random_walk', 'bert', 'profile+bert', 'all_nodes_profile_more']:

                        try:
                            sim = self.sim_all_etypes[etype][np.array([self.diseaseid2id_etypes[etype][i.item()] for i in h_disease['disease_query_id'][0]])]
                        except:

                            disease_etypes = LEGACY_ETYPES
                            disease_nodes = LEGACY_NODES
                            disease_etypes_all = BLOCK_ETYPES
                            disease_nodes_all = BLOCK_NODES
                            ## new disease not seen in the training set
                            for i in h_disease['disease_query_id'][0]:
                                if i.item() not in self.diseases_profile_etypes[etype]:
                                    if self.sim_measure == 'all_nodes_profile':
                                        self.diseases_profile_etypes[etype][i.item()] = obtain_disease_profile(G, i, disease_etypes, disease_nodes)
                                    elif self.sim_measure == 'all_nodes_profile_more':
                                        self.diseases_profile_etypes[etype][i.item()] = obtain_disease_profile(G, i, disease_etypes_all, disease_nodes_all)
                                    elif self.sim_measure == 'protein_profile':
                                        self.diseases_profile_etypes[etype][i.item()] = obtain_disease_profile(G, i, ['rev_disease_protein'], ['gene/protein'])
                                    elif self.sim_measure == 'protein_random_walk':
                                        self.diseases_profile_etypes[etype][i.item()] = obtain_protein_random_walk_profile(i, self.num_walks, self.path_length, G, disease_etypes, disease_nodes, self.walk_mode)
                                    elif self.sim_measure == 'bert':
                                        self.diseases_profile_etypes[etype][i.item()] = torch.Tensor(self.bert_embed[self.id2bertindex[self.disease_dict[i.item()]]])
                                    elif self.sim_measure == 'profile+bert':
                                        self.diseases_profile_etypes[etype][i.item()] = torch.cat((obtain_disease_profile(G, i, disease_etypes, disease_nodes), torch.Tensor(self.bert_embed[self.id2bertindex[self.disease_dict[i.item()]]])))

                            profile_query = [self.diseases_profile_etypes[etype][i.item()] for i in h_disease['disease_query_id'][0]]
                            profile_query = torch.cat(profile_query).view(len(profile_query), -1)

                            profile_keys = [self.diseases_profile_etypes[etype][i.item()] for i in h_disease['disease_key_id'][0]]
                            profile_keys = torch.cat(profile_keys).view(len(profile_keys), -1)

                            sim = sim_matrix(profile_query, profile_keys)

                        if src_h.shape[0] == src_h_keys.shape[0]:
                            ## during training...
                            coef = torch.topk(sim, self.k + 1).values[:, 1:]
                            coef = F.normalize(coef, p=1, dim=1)
                            embed = h_disease['disease_key'][torch.topk(sim, self.k + 1).indices[:, 1:]]
                        else:
                            ## during evaluation...
                            coef = torch.topk(sim, self.k).values[:, :]
                            coef = F.normalize(coef, p=1, dim=1)
                            embed = h_disease['disease_key'][torch.topk(sim, self.k).indices[:, :]]
                        out = torch.mul(embed, coef.unsqueeze(dim = 2).to(self.device)).sum(dim = 1)

                    if self.block_mode or self.sim_measure in ['protein_profile', 'all_nodes_profile', 'all_nodes_profile_more', 'protein_random_walk', 'bert', 'profile+bert']:
                        # for protein profile, we are only looking at diseases for now...
                        if self.agg_measure == 'learn':
                            coef_all = self.m(self.W_gate['disease'](torch.cat((h_disease['disease_query'], out), dim = 1)))
                            proto_emb = (1 - coef_all)*h_disease['disease_query'] + coef_all*out
                        elif self.agg_measure == 'heuristics-0.8':
                            proto_emb = 0.8*h_disease['disease_query'] + 0.2*out
                        elif self.agg_measure == 'avg':
                            proto_emb = 0.5*h_disease['disease_query'] + 0.5*out
                        elif self.agg_measure in ('rarity', 'rarity_4block', 'gumbel_block'):
                            # Same degree-based scalar c_i for all three. In the block measures the
                            # per-block selection has already happened inside `out`; c_i stays the
                            # unlearned rarity heuristic so "which blocks" and "how much prototype"
                            # remain separate ablation axes.
                            if src == 'disease':
                                G_out_deg_etype = degree(G[etype].edge_index[0], num_nodes=G[src].num_nodes)
                                graph_out_deg_etype = degree(graph[etype].edge_index[0], num_nodes=graph[src].num_nodes)
                                coef_all = exponential(G_out_deg_etype[torch.where(graph_out_deg_etype != 0)], self.exp_lambda).reshape(-1, 1)
                            elif dst == 'disease':
                                G_in_deg_etype = degree(G[etype].edge_index[1], num_nodes=G[dst].num_nodes)
                                graph_in_deg_etype = degree(graph[etype].edge_index[1], num_nodes=graph[dst].num_nodes)
                                coef_all = exponential(G_in_deg_etype[torch.where(graph_in_deg_etype != 0)], self.exp_lambda).reshape(-1, 1)
                            proto_emb = (1 - coef_all)*h_disease['disease_query'] + coef_all*out
                        elif self.agg_measure == '100proto':
                            proto_emb = out
                        else:
                            # Previously fell through leaving proto_emb undefined, which surfaced as
                            # a confusing NameError on the assignment below.
                            raise ValueError('Unknown agg_measure %r' % self.agg_measure)
                        h['disease'][h_disease['disease_query_id']] = proto_emb
                    else:
                        # NOTE: this branch is unreachable for every sim_measure the codebase
                        # supports, and it is broken as written -- sim_emb_src / sim_emb_dst are
                        # never assigned anywhere in forward(), so reaching it raises NameError.
                        # Left in place (rather than deleted) to keep this diff scoped to the
                        # gating work; the guard above now fails loudly on unknown agg_measure.
                        raise NotImplementedError(
                            'Symmetric src/dst prototype path is not implemented: sim_emb_src / '
                            'sim_emb_dst are never computed. Reached with sim_measure=%r.' % self.sim_measure)
                        if self.agg_measure == 'learn':
                            coef_src = self.m(self.W_gate[src](torch.cat((src_h, sim_emb_src), dim = 1)))
                            coef_dst = self.m(self.W_gate[dst](torch.cat((dst_h, sim_emb_dst), dim = 1)))
                        elif self.agg_measure == 'rarity':
                            # give high weights to proto embeddings for nodes that have low degrees
                            G_out_deg_etype = degree(G[etype].edge_index[0], num_nodes=G[src].num_nodes)
                            G_in_deg_etype = degree(G[etype].edge_index[1], num_nodes=G[dst].num_nodes)
                            graph_out_deg_etype = degree(graph[etype].edge_index[0], num_nodes=graph[src].num_nodes)
                            graph_in_deg_etype = degree(graph[etype].edge_index[1], num_nodes=graph[dst].num_nodes)
                            coef_src = exponential(G_out_deg_etype[torch.where(graph_out_deg_etype != 0)], self.exp_lambda).reshape(-1, 1)
                            coef_dst = exponential(G_in_deg_etype[torch.where(graph_in_deg_etype != 0)], self.exp_lambda).reshape(-1, 1)
                        elif self.agg_measure == 'heuristics-0.8':
                            coef_src = 0.2
                            coef_dst = 0.2
                        elif self.agg_measure == 'avg':
                            coef_src = 0.5
                            coef_dst = 0.5
                        elif self.agg_measure == '100proto':
                            coef_src = 1
                            coef_dst = 1

                        proto_emb_src = (1 - coef_src)*src_h + coef_src*sim_emb_src
                        proto_emb_dst = (1 - coef_dst)*dst_h + coef_dst*sim_emb_dst

                        h[src][src_rel_idx] = proto_emb_src
                        h[dst][dst_rel_idx] = proto_emb_dst

                out = self._apply_edges(graph, etype, h)
                s_l.append(out)
                scores[etype] = out

                if self.proto and self._etype_has_sim(etype):
                    # recover back to the original embeddings for other relations
                    h[src][src_rel_idx] = src_h
                    h[dst][dst_rel_idx] = dst_h


        if pretrain_mode:
            s_l = torch.cat(s_l)
        else:
            s_l = torch.cat(s_l).reshape(-1,).detach().cpu().numpy()
        return scores, s_l



class AttHeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super(AttHeteroRGCNLayer, self).__init__()
        self.weight = nn.ModuleDict({
                name : nn.Linear(in_size, out_size) for name in etypes
            })

        self.attn_fc = nn.ModuleDict({
                name : nn.Linear(out_size * 2, 1, bias = False) for name in etypes
            })
        self.out_size = out_size

    def forward(self, G, feat_dict, return_att=False):
        att = {}

        etypes_all = [(s, e, d) for s, e, d in G.edge_types if G[(s, e, d)].num_edges > 0]

        # Compute all Wh transformations (keyed by relation name string)
        Wh_dict = {}
        for srctype, etype, dsttype in etypes_all:
            Wh_dict[etype] = self.weight[etype](feat_dict[srctype])

        out = {ntype: None for ntype in G.node_types}

        for srctype, etype, dsttype in etypes_all:
            ei = G[(srctype, etype, dsttype)].edge_index
            src_nodes = ei[0]
            dst_nodes = ei[1]

            Wh_src = Wh_dict[etype][src_nodes]  # (num_edges, out_size)

            # Determine destination etype key for attention
            if srctype == dsttype:
                dst_etype_key = etype
            elif etype[:3] == 'rev':
                dst_etype_key = etype[4:]
            else:
                dst_etype_key = 'rev_' + etype

            try:
                Wh_dst = Wh_dict[dst_etype_key][dst_nodes]  # (num_edges, out_size)
                wh2 = torch.cat([Wh_src, Wh_dst], dim=1)  # (num_edges, 2*out_size)
            except:
                print('AttHeteroRGCNLayer forward error for etype: ', etype)
                raise ValueError

            e = F.leaky_relu(self.attn_fc[etype](wh2))  # (num_edges, 1)

            # Softmax over dst nodes
            alpha = softmax(e.squeeze(-1), dst_nodes, num_nodes=G[dsttype].num_nodes)

            if return_att:
                att[(srctype, etype, dsttype)] = e.detach().cpu().numpy()

            # Weighted aggregation
            msg = alpha.unsqueeze(-1) * Wh_src  # (num_edges, out_size)
            agg = scatter(msg, dst_nodes, dim=0, reduce='sum', dim_size=G[dsttype].num_nodes)

            if out[dsttype] is None:
                out[dsttype] = agg
            else:
                out[dsttype] = out[dsttype] + agg

        # Fill zeros for node types with no incoming edges
        device = feat_dict[list(feat_dict.keys())[0]].device
        for ntype in G.node_types:
            if out[ntype] is None:
                out[ntype] = torch.zeros(G[ntype].num_nodes, self.out_size, device=device)

        return out, att

class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super(HeteroRGCNLayer, self).__init__()
        self.weight = nn.ModuleDict({
                name : nn.Linear(in_size, out_size) for name in etypes
            })
        self.in_size = in_size
        self.out_size = out_size

        self.gate_storage = {}
        self.gate_score_storage = {}
        self.gate_penalty_storage = {}

    def add_graphmask_parameter(self, gate, baseline, layer):
        self.gate = gate
        self.baseline = baseline
        self.layer = layer

    def forward(self, G, feat_dict):
        out = {ntype: None for ntype in G.node_types}

        for srctype, etype, dsttype in G.edge_types:
            ei = G[(srctype, etype, dsttype)].edge_index
            if ei.shape[1] == 0:
                continue
            src_nodes = ei[0]
            dst_nodes = ei[1]

            Wh = self.weight[etype](feat_dict[srctype])
            msg = Wh[src_nodes]

            # scatter mean per etype, then sum across etypes (matches DGL multi_update_all 'sum')
            agg = scatter(msg, dst_nodes, dim=0, reduce='mean', dim_size=G[dsttype].num_nodes)

            if out[dsttype] is None:
                out[dsttype] = agg
            else:
                out[dsttype] = out[dsttype] + agg

        # Fill zeros for node types with no incoming edges
        device = feat_dict[list(feat_dict.keys())[0]].device
        for ntype in G.node_types:
            if out[ntype] is None:
                out[ntype] = torch.zeros(G[ntype].num_nodes, self.out_size, device=device)

        return out

    def graphmask_forward(self, G, feat_dict, graphmask_mode, return_gates, no_base):
        self.no_base = no_base
        self.return_gates = return_gates
        self.penalty = []
        self.num_masked = 0

        # Compute all Wh transformations (keyed by relation name string)
        Wh_dict = {}
        for srctype, etype, dsttype in G.edge_types:
            Wh_dict[etype] = self.weight[etype](feat_dict[srctype])

        out = {ntype: None for ntype in G.node_types}

        for srctype, etype, dsttype in G.edge_types:
            ei = G[(srctype, etype, dsttype)].edge_index
            if ei.shape[1] == 0:
                continue
            src_nodes = ei[0]
            dst_nodes = ei[1]

            src_feat = Wh_dict[etype][src_nodes]  # (num_edges, out_size)

            if graphmask_mode:
                # Determine destination etype key for gate (mirrors DGL gm_online logic)
                if srctype == dsttype:
                    dst_etype_key = etype
                elif etype[:3] == 'rev':
                    dst_etype_key = etype[4:]
                else:
                    dst_etype_key = 'rev_' + etype

                dst_feat = Wh_dict[dst_etype_key][dst_nodes]  # (num_edges, out_size)

                gate, penalty, gate_score, penalty_not_sum = self.gate[etype][self.layer]([src_feat, dst_feat])

                self.penalty.append(penalty)
                self.num_masked += len(torch.where(gate.reshape(-1) != 1)[0])

                if no_base:
                    message = gate.unsqueeze(-1) * src_feat
                else:
                    message = gate.unsqueeze(-1) * src_feat + (1 - gate.unsqueeze(-1)) * self.baseline[etype][self.layer].unsqueeze(0)

                if return_gates:
                    self.gate_storage[etype] = copy.deepcopy(gate.to('cpu').detach())
                    self.gate_penalty_storage[etype] = copy.deepcopy(penalty_not_sum.to('cpu').detach())
                    self.gate_score_storage[etype] = copy.deepcopy(gate_score.to('cpu').detach())
            else:
                message = src_feat

            # scatter mean to dst (matches DGL fn.mean reduce)
            agg = scatter(message, dst_nodes, dim=0, reduce='mean', dim_size=G[dsttype].num_nodes)

            if out[dsttype] is None:
                out[dsttype] = agg
            else:
                out[dsttype] = out[dsttype] + agg

        # Fill zeros for missing node types
        device = feat_dict[list(feat_dict.keys())[0]].device
        for ntype in G.node_types:
            if out[ntype] is None:
                out[ntype] = torch.zeros(G[ntype].num_nodes, self.out_size, device=device)

        if graphmask_mode:
            self.penalty = torch.stack(self.penalty).reshape(-1,)
            penalty = torch.mean(self.penalty)
        else:
            penalty = 0

        return out, penalty, self.num_masked

class HeteroRGCN(nn.Module):
    def __init__(self, G, in_size, hidden_size, out_size, attention, proto, proto_num, sim_measure, bert_measure, agg_measure, num_walks, walk_mode, path_length, split, data_folder, exp_lambda, device,
                 gumbel_tau_start = 1.0, gumbel_tau_end = 0.3, gumbel_anneal_steps = 1000,
                 gumbel_hidden = 64, gumbel_entropy_weight = 0.01):
        super(HeteroRGCN, self).__init__()

        etypes = [et[1] for et in G.edge_types]

        if attention:
            self.layer1 = AttHeteroRGCNLayer(in_size, hidden_size, etypes)
            self.layer2 = AttHeteroRGCNLayer(hidden_size, out_size, etypes)
        else:
            self.layer1 = HeteroRGCNLayer(in_size, hidden_size, etypes)
            self.layer2 = HeteroRGCNLayer(hidden_size, out_size, etypes)

        self.w_rels = nn.Parameter(torch.Tensor(len(G.edge_types), out_size))
        nn.init.xavier_uniform_(self.w_rels, gain=nn.init.calculate_gain('relu'))
        rel2idx = dict(zip(G.edge_types, list(range(len(G.edge_types)))))

        self.pred = DistMultPredictor(n_hid = hidden_size, w_rels = self.w_rels, G = G, rel2idx = rel2idx, proto = proto, proto_num = proto_num, sim_measure = sim_measure, bert_measure = bert_measure, agg_measure = agg_measure, num_walks = num_walks, walk_mode = walk_mode, path_length = path_length, split = split, data_folder = data_folder, exp_lambda = exp_lambda, device = device,
                                      gumbel_tau_start = gumbel_tau_start, gumbel_tau_end = gumbel_tau_end, gumbel_anneal_steps = gumbel_anneal_steps,
                                      gumbel_hidden = gumbel_hidden, gumbel_entropy_weight = gumbel_entropy_weight)
        self.attention = attention

        self.hidden_size = hidden_size
        self.out_size = out_size
        self.etypes = etypes
        self.device = device

    def forward_minibatch(self, pos_G, neg_G, G, mode='train', pretrain_mode=False):
        input_dict = {ntype: G[ntype].inp for ntype in G.node_types}
        h_dict = self.layer1(G, input_dict)
        h_dict = {k: F.leaky_relu(h) for k, h in h_dict.items()}
        h = self.layer2(G, h_dict)

        scores, out_pos = self.pred(pos_G, G, h, pretrain_mode, mode=mode + '_pos')
        scores_neg, out_neg = self.pred(neg_G, G, h, pretrain_mode, mode=mode + '_neg')
        return scores, scores_neg, out_pos, out_neg


    def forward(self, G, neg_G, eval_pos_G=None, return_h=False, return_att=False, mode='train', pretrain_mode=False):
        input_dict = {ntype: G[ntype].inp for ntype in G.node_types}

        if self.attention:
            h_dict, a_dict_l1 = self.layer1(G, input_dict, return_att)
            h_dict = {k: F.leaky_relu(h) for k, h in h_dict.items()}
            h, a_dict_l2 = self.layer2(G, h_dict, return_att)
        else:
            h_dict = self.layer1(G, input_dict)
            h_dict = {k: F.leaky_relu(h) for k, h in h_dict.items()}
            h = self.layer2(G, h_dict)

        if return_h:
            return h

        if return_att:
            return a_dict_l1, a_dict_l2

        # full batch
        if eval_pos_G is not None:
            # eval mode
            scores, out_pos = self.pred(eval_pos_G, G, h, pretrain_mode, mode=mode + '_pos')
            scores_neg, out_neg = self.pred(neg_G, G, h, pretrain_mode, mode=mode + '_neg')
            return scores, scores_neg, out_pos, out_neg
        else:
            scores, out_pos = self.pred(G, G, h, pretrain_mode, mode=mode + '_pos')
            scores_neg, out_neg = self.pred(neg_G, G, h, pretrain_mode, mode=mode + '_neg')
            return scores, scores_neg, out_pos, out_neg

    def graphmask_forward(self, G, pos_graph, neg_graph, graphmask_mode=False, return_gates=False, only_relation=None, no_base=False):
        # The Explainer calls .train() on its model copy, which would otherwise keep annealing the
        # Gumbel temperature and advancing gumbel_step. Pin the gate to its deterministic, tau_end
        # behaviour so explanations are never produced mid-anneal.
        self.pred.gumbel_frozen = True
        input_dict = {ntype: G[ntype].inp for ntype in G.node_types}
        h_dict_l1, penalty_l1, num_masked_l1 = self.layer1.graphmask_forward(G, input_dict, graphmask_mode, return_gates, no_base)
        h_dict = {k: F.leaky_relu(h) for k, h in h_dict_l1.items()}
        h, penalty_l2, num_masked_l2 = self.layer2.graphmask_forward(G, h_dict, graphmask_mode, return_gates, no_base)

        scores_pos, out_pos = self.pred(pos_graph, G, h, False, mode='train_pos', only_relation=only_relation)
        scores_neg, out_neg = self.pred(neg_graph, G, h, False, mode='train_neg', only_relation=only_relation)
        return scores_pos, scores_neg, penalty_l1 + penalty_l2, [num_masked_l1, num_masked_l2]


    def pop_gate_regularizer(self):
        """Gate entropy accumulated over the forward pass(es) since the last call.

        Returns a 0-d tensor; 0.0 unless agg_measure='gumbel_block' and the model is training, so
        the training loop can add it unconditionally.
        """
        return self.pred.pop_gate_regularizer()

    def enable_layer(self, layer, graphmask=True):
        print("Enabling layer "+str(layer))

        for name in self.etypes:
            if graphmask:
                for parameter in self.gates_all[name][layer].parameters():
                    parameter.requires_grad = True
                self.baselines_all[name][layer].requires_grad = True
            else:
                for parameter in self.gates_all[name].parameters():
                    parameter.requires_grad = True

    def count_layers(self):
        return 2

    def get_gates(self):
        return [self.layer1.gate_storage, self.layer2.gate_storage]

    def get_gates_scores(self):
        return [self.layer1.gate_score_storage, self.layer2.gate_score_storage]

    def get_gates_penalties(self):
        return [self.layer1.gate_penalty_storage, self.layer2.gate_penalty_storage]


    def add_graphmask_parameters(self, G, threshold=0.5, remove_key_parts=False, use_top_k=False, k=0.05, gate_hidden_size=32):
        gates_all, baselines_all = {}, {}
        hidden_size = self.hidden_size
        out_size = self.out_size
        print('gate_hidden_size: ', gate_hidden_size)
        for name in [et[1] for et in G.edge_types]:
            ## for each relation type

            gates = []
            baselines = []

            vertex_embedding_dims = [hidden_size, out_size]
            message_dims = [hidden_size, out_size]
            h_dims = message_dims

            for v_dim, m_dim, h_dim in zip(vertex_embedding_dims, message_dims, h_dims):
                gate_input_shape = [m_dim, m_dim]

                ### different layers have different gates
                gate = torch.nn.Sequential(
                    MultipleInputsLayernormLinear(gate_input_shape, gate_hidden_size),
                    nn.ReLU(),
                    nn.Linear(gate_hidden_size, 1),
                    Squeezer(),
                    SoftConcrete(threshold, remove_key_parts, use_top_k, k)
                )

                gates.append(gate)

                baseline = torch.FloatTensor(m_dim)
                stdv = 1. / math.sqrt(m_dim)
                baseline.uniform_(-stdv, stdv)
                baseline = torch.nn.Parameter(baseline, requires_grad=True)

                baselines.append(baseline)

            gates = torch.nn.ModuleList(gates)
            gates_all[name] = gates

            baselines = torch.nn.ParameterList(baselines)
            baselines_all[name] = baselines

        self.gates_all = nn.ModuleDict(gates_all)
        self.baselines_all = nn.ModuleDict(baselines_all)

        # Initially we cannot update any parameters. They should be enabled layerwise
        for parameter in self.parameters():
            parameter.requires_grad = False

        self.layer1.add_graphmask_parameter(self.gates_all, self.baselines_all, 0)
        self.layer2.add_graphmask_parameter(self.gates_all, self.baselines_all, 1)
