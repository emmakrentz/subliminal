# Subliminal Semantic Bible

Experimental program testing whether LLM-generated number sequences contain
semantic fingerprints of teacher concept biases. See the Bible document
(`subliminal_semantic_bible.pdf`) for the full 42-experiment specification.

## Layout

- `universal/` — single source of truth: concepts, prompts, models, paths,
  the filter rule, anchor-sentence corpus, concept facets. Every notebook
  imports from here. **Edit here once; never hardcode in notebooks.**
- `phase_{0..6}_notebooks/` — experiment notebooks, one per experiment.
- `data/` — generated artifacts (sequences, embeddings, concept vectors,
  divergence masks, LoRA checkpoints, NLA explanations, manifests).
- `results/` — paper-bound figures and summary JSONs.
- `env/requirements.txt` — pinned dependencies.

## Conventions

- Every manifest records the SHA-256 of `universal/concepts.yaml` at the
  time of generation, so cross-phase artifacts can be checked for
  config consistency.
- Sequence files: `data/sequences/{canonical,es}/{model_id}_{concept}.json`.
- Never write into `universal/` from a notebook (it's read-only config).
