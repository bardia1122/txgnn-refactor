"""Question templates and prompt assembly (DrKGC sections 3.3 and 3.6).

The prompt has four parts, following the paper's Table 5:

1. an instruction naming the role and constraining the answer to the candidate set;
2. the candidate list;
3. structural evidence - either `[Placeholder]` tokens later replaced by the GCN
   adapter's vectors, or the retrieved triples written out as text, or nothing;
4. the question, generated from a per-relation template.

`evidence` selects between those three, which gives the paper's ablations
("w/o local embedding", "w/o embedding") directly, and lets you probe a
frozen off-the-shelf model with text evidence before committing to fine-tuning.

The placeholder is `<|reserved_special_token_0|>`: it is already in the Llama
vocabulary, encodes to exactly one token, and never occurs in natural text - so
placeholder positions can be located unambiguously and swapped for adapter
vectors without resizing the embedding matrix.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

PLACEHOLDER = "<|reserved_special_token_0|>"

#: head prediction: given the disease, name the drug. One template per relation,
#: the paper's "question-template lexicon" (appendix A.3) for PrimeKG.
HEAD_TEMPLATES: Dict[str, str] = {
    "indication": "What drug is used to treat {}?",
    "contraindication": "What drug is contraindicated for a patient with {}?",
    "off-label use": "What drug is used off-label to treat {}?",
}

#: tail prediction, kept for completeness; unused while --direction is head
TAIL_TEMPLATES: Dict[str, str] = {
    "indication": "What disease is treated by {}?",
    "contraindication": "For what disease is {} contraindicated?",
    "off-label use": "What disease is {} used off-label to treat?",
}

ROLE = "biomedical scientist"

INSTRUCTION = (
    "You are an excellent {role}. The task is to predict the answer based on the "
    "given question, and you only need to answer one entity. The answer must be "
    "in ({candidates})."
)


def question_for(relation: str, entity_name: str, direction: str = "head") -> str:
    templates = HEAD_TEMPLATES if direction == "head" else TAIL_TEMPLATES
    template = templates.get(relation)
    if template is None:
        raise KeyError(f"no question template for relation {relation!r}")
    return template.format(entity_name)


def _candidate_clause(names: Sequence[str]) -> str:
    return ", ".join(f"'{name}'" for name in names)


def _embedding_clause(names: Sequence[str]) -> str:
    parts = [f"'query entity': {PLACEHOLDER}"]
    parts += [f"'{name}': {PLACEHOLDER}" for name in names]
    return "You can refer to the entity embeddings: " + ", ".join(parts) + "."


def _text_clause(triples: Sequence[Sequence[str]], max_triples: int = 60) -> str:
    if not triples:
        return "No supporting graph context was retrieved."
    lines = [
        f"({head}, {relation}, {tail})"
        for head, relation, tail in triples[:max_triples]
        if head and tail
    ]
    return "You can refer to the following facts from the knowledge graph:\n" + "\n".join(lines)


def build_prompt(
    relation: str,
    query_name: str,
    candidate_names: Sequence[str],
    evidence: str = "embedding",
    triples: Optional[Sequence[Sequence[str]]] = None,
    direction: str = "head",
    role: str = ROLE,
    max_text_triples: int = 60,
) -> str:
    """Assemble one prompt. `evidence` is 'embedding', 'text' or 'none'."""
    if evidence not in ("embedding", "text", "none"):
        raise ValueError(f"evidence must be embedding/text/none, got {evidence!r}")

    blocks = [INSTRUCTION.format(role=role, candidates=_candidate_clause(candidate_names))]
    if evidence == "embedding":
        blocks.append(_embedding_clause(candidate_names))
    elif evidence == "text":
        blocks.append(_text_clause(triples or [], max_text_triples))
    blocks.append(f"Question: {question_for(relation, query_name, direction)}")
    blocks.append("Answer:")
    return "\n\n".join(blocks)


def num_placeholders(candidate_names: Sequence[str]) -> int:
    """Query entity plus one per candidate - the count the adapter must supply."""
    return 1 + len(candidate_names)
