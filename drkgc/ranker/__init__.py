"""Step 2 of the DrKGC pipeline: the lightweight structure-based model.

An R-GCN + DistMult link predictor trained on the step-1 artifacts. It produces
the two things every later step needs:

* **global structural embeddings** for every entity (input to the GCN adapter);
* **top-k candidate drugs** per query, which constrain the LLM's output space.

Modules
-------
data    global entity index, message-passing graph, triple sets, filter sets
model   R-GCN encoder + DistMult decoder
train   training loop with filtered MRR / Hits@k evaluation
rank    export global embeddings and top-k candidates per query
"""
