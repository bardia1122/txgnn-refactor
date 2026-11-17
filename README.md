# Disentangling curated knowledge from graph learning in zero-shot drug repurposing

Code, experiment notebooks and per-run results for the paper

> **Disentangling curated knowledge from graph learning in zero-shot drug repurposing**
> *Authors* — *Journal / preprint link to be added*

The paper asks where the zero-shot performance of [TxGNN](https://github.com/mims-harvard/TxGNN)
(Huang et al., *Nature Medicine* 2024) comes from: the curated biomedical knowledge in its knowledge graph,
or the graph-learning machinery built on top of it. We answer it with 725 models trained in controlled,
capacity-matched and split-paired ablations on PrimeKG.

**Main findings**

- A graph reduced to the drug–disease labels plus two curated relations (disease–protein and
  disease–disease annotations, 3.8% of training edges) retains **88%** of TxGNN's above-chance performance.
- Exact Shapley values assign **84%** of the performance to these two relations.
- Most of the disease–protein effect (**76–93%**) bypasses message passing entirely, through a
  parameter-free disease-similarity lookup.
- Pre-training contributes about one sixth of the performance. Message passing contributes about one sixth
  once node features are learnable, but little with TxGNN's frozen features.
- Making node features learnable, or halving the encoder, improves on TxGNN's configuration.

---

## Repository layout

```
.
├── pyg_implementation/        TxGNN re-implemented in PyTorch Geometric (used for every experiment)
│   └── txgnn/                 TxData, TxGNN, TxEval, model.py, utils.py, GraphMask explainer
├── dgl_implementation/        The original TxGNN code (DGL 0.5.2), kept for reference and port checks
├── ablation_*.ipynb           One notebook per experiment (see the table below)
├── results/                   Per-run outputs of every experiment, one folder per run
├── data/disease_files/        Disease lists used by TxGNN's disease-area splits
├── TxGNN_Reproduce.ipynb      End-to-end reproduction of baseline TxGNN with the PyG port
├── TxGNN_PyG_Train_Eval.ipynb Full training and evaluation walkthrough, including the explainer
├── dgl_benchmark.ipynb        Port check, DGL side
├── pyg_benchmark.ipynb        Port check, PyG side (includes a forward-pass parity test with DGL weights)
└── compare_results.ipynb      Side-by-side comparison of the two benchmark runs
```

## Why a PyTorch Geometric re-implementation

The original TxGNN code is built on DGL 0.5.2, an early release with old dependencies that was difficult
to install, run and reproduce. That release also lacks many components of current graph-learning
libraries, which made the changes our experiments needed (replacing the encoder, adding learnable node
features, controlling which relations the similarity module reads) hard to implement.

`pyg_implementation/` is a port of the released code to PyTorch Geometric. The model, similarity module,
data split, training procedure, evaluation and all hyperparameters are unchanged. The port was checked
against the original with `dgl_benchmark.ipynb`, `pyg_benchmark.ipynb` and `compare_results.ipynb`.
Absolute AUPRC runs about 0.05 below the values reported in the TxGNN paper, so every conclusion in our
paper rests on within-study comparisons in which only the manipulated factor differs.

The only change to the library made for the experiments is one gate option, `agg_measure='const-0.9'`, in
`pyg_implementation/txgnn/model.py`. It is used only by `ablation_gate_constant.ipynb`, and all default
behaviour is identical to the released code.

## Installation

Tested with Python 3.12.13, PyTorch 2.6.0 (CUDA 11.8) and PyTorch Geometric 2.8.0 on a single NVIDIA
RTX 4060 Ti (16 GB).

```bash
conda create -n txgnn-pyg python=3.12
conda activate txgnn-pyg

pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric==2.8.0
pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.6.0+cu118.html

pip install -e pyg_implementation
pip install jupyter
```

For a different CUDA version, change `cu118` in both URLs to match your PyTorch build.

## Data

The notebooks download PrimeKG automatically from
[Harvard Dataverse](https://doi.org/10.7910/DVN/IXA7BM) (`kg.csv`, `node.csv`, `edges.csv`) into
`data/kg/` on first use. Splits are built and cached there too. Allow about 4 GB of disk space.

Every experiment uses TxGNN's `complex_disease` zero-shot split: held-out diseases have no drug–disease
edge in training. A seed indexes one cached split, so `seed 7` is the same split in every notebook, and
every comparison is paired by seed.

## Experiments

Each notebook runs one experiment of the paper. Notebooks are resumable: a finished run is skipped, and
runs shared between experiments (for example, the reference model) are imported rather than retrained.
Set `STOP_AFTER_N_RUNS = 1` for a smoke test, then `None` for the full grid.

| Experiment in the paper | Notebook | Splits |
|---|---|---|
| Reference model (TxGNN, R-GCN) on the 20 shared splits | `ablation_encoder_swap.ipynb` (`rgcn` arm) | 20 |
| Relation screen (leave one relation out) | `ablation_stage_a.ipynb` | 5 |
| Hold-out replication of the two selected relations | `ablation_stage_a_seeds10_19.ipynb` | 20 |
| Annotation-only graph | `ablation_keep_disease_only.ipynb` | 20 |
| Identity versus count | `ablation_keep_only_controls.ipynb` | 20 |
| Capacity matching of every reduced graph | `ablation_capacity_matched_graph.ipynb` | 20 |
| Signature access (signature-only removals, pooling disabled) | `ablation_signature_only.ipynb`, `ablation_signature_baselines.ipynb` | 10 |
| Pre-training and gate (constant 0.5) | `ablation_pretrain_and_gate.ipynb` | 20 |
| Gate at constant 0.9 | `ablation_gate_constant.ipynb` | 20 |
| Encoder capacity and depth | `ablation_deep_mlp.ipynb` | 20 |
| Learnable node features | `ablation_learnable_embeddings.ipynb` | 20 |
| Untrained encoder weights | `ablation_frozen_encoder.ipynb` | 20 |
| Half-size encoders | `ablation_encoder_halfparam.ipynb` | 20 |
| Disease-centric ranking protocol | `ablation_full_ranking.ipynb` | 10 |
| Run-to-run variation (repeat trainings) | `ablation_repeat_baseline.ipynb` | 3 |

One training run (1 pre-training epoch plus 1,000 fine-tuning epochs) takes roughly 25–45 minutes on the
GPU above. The full study is 725 models and about 344 GPU-hours.

## Results

`results/<experiment>/<arm>_seed<n>/` holds one folder per trained model:

| File | Contents |
|---|---|
| `run_meta.json` | Full configuration: model settings, gate, edge types kept, relations read by the signature, encoder parameter count, training edges, timings, software versions |
| `edge_level_test_metrics.csv` | Test AUROC and AUPRC for indication, contraindication, off-label use and the micro-average |
| `disease_centric_*.csv` | Per-disease ranking metrics (`ablation_full_ranking` only) |

Each notebook also writes a per-experiment summary CSV with paired per-split deltas against its reference.

**Reading the numbers.** The edge-level evaluation samples one negative per positive, so **chance AUPRC is
0.5, not 0**. The paper reports effects as a share of the above-chance margin (reference AUPRC − 0.5).
Effects are means of per-split paired differences with 95% *t* intervals and an exact sign test.

## Citation

If you use this code or these results, please cite our paper and the original TxGNN and PrimeKG papers:

```bibtex
@article{ourpaper,
  title   = {Disentangling curated knowledge from graph learning in zero-shot drug repurposing},
  author  = {TO BE ADDED},
  journal = {TO BE ADDED},
  year    = {TO BE ADDED}
}

@article{huang2024txgnn,
  title   = {A foundation model for clinician-centered drug repurposing},
  author  = {Huang, Kexin and Chandak, Payal and Wang, Qianwen and Havaldar, Shreyas and Vaid, Akhil
             and Leskovec, Jure and Nadkarni, Girish N. and Glicksberg, Benjamin S. and Gehlenborg, Nils
             and Zitnik, Marinka},
  journal = {Nature Medicine},
  volume  = {30},
  pages   = {3601--3613},
  year    = {2024},
  doi     = {10.1038/s41591-024-03233-x}
}

@article{chandak2023primekg,
  title   = {Building a knowledge graph to enable precision medicine},
  author  = {Chandak, Payal and Huang, Kexin and Zitnik, Marinka},
  journal = {Scientific Data},
  volume  = {10},
  pages   = {67},
  year    = {2023},
  doi     = {10.1038/s41597-023-01960-3}
}
```

## License and acknowledgements

TxGNN was developed by the Zitnik Lab at Harvard ([mims-harvard/TxGNN](https://github.com/mims-harvard/TxGNN))
and is released under the MIT License. Both `dgl_implementation/` and the derived `pyg_implementation/` keep
that license (see the `LICENSE` file in each folder). PrimeKG is distributed by its authors through
Harvard Dataverse under its own terms.
