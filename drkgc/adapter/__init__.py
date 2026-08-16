"""Step 4 of the DrKGC pipeline: the GCN adapter.

Turns a retrieved subgraph into vectors the LLM can consume:

    global embedding (step 2)  ->  subgraph GCN  ->  local embedding
    [global ; local]           ->  adapter MLP   ->  LLM input dimension

The paper (section 3.5, "Structure-Aware Embedding Enhancement") runs the GCN in a
low-dimensional space to keep it cheap, then projects up to the LLM's hidden size,
and lets gradients flow through the whole thing during LoRA fine-tuning. So this
module is a *component of step 5*, not a standalone trainable model — it has no
loss of its own.

Modules
-------
data        subgraph JSONL -> per-query PyG graphs with global-embedding features
model       SubgraphGCN + StructureAdapter
precompute  shape/forward smoke test, and optional frozen local-embedding export
"""
