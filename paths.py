"""paths.py — single source of truth for all project paths.

Every notebook does:
    from universal import paths
    
    # Access data directories
    canonical_dir = paths.DATA / "sequences" / "canonical"
    manifest_path = paths.DATA / "manifests" / "my_manifest.json"
    
    # Access results
    figure_path = paths.RESULTS / "my_figure.png"

Never hardcode '/home/...' or relative paths in notebooks.
Project root is auto-detected from this file's location.
"""
from pathlib import Path
from typing import Optional


# ==============================================================================
# Project Root Detection
# ==============================================================================

def _find_project_root() -> Path:
    """Find the project root by walking up from this file's location.
    
    Assumes this file is at: subliminal-semantic-bible/universal/paths.py
    Returns: Path to subliminal-semantic-bible/
    """
    # This file is at: PROJECT_ROOT/universal/paths.py
    universal_dir = Path(__file__).parent
    project_root = universal_dir.parent
    
    # Sanity check: verify we found the right directory
    expected_markers = ["universal", "data", "results", "env"]
    found = [marker for marker in expected_markers if (project_root / marker).exists()]
    
    if len(found) < 3:  # Allow some flexibility during setup
        raise RuntimeError(
            f"Project root detection failed. Expected to find subdirs "
            f"{expected_markers} in {project_root}, but only found {found}. "
            f"Is this file at the correct location (universal/paths.py)?"
        )
    
    return project_root


# ==============================================================================
# Core Directories
# ==============================================================================

PROJECT_ROOT = _find_project_root()

UNIVERSAL = PROJECT_ROOT / "universal"
DATA = PROJECT_ROOT / "data"
RESULTS = PROJECT_ROOT / "results"
ENV = PROJECT_ROOT / "env"

# Phase directories (notebooks)
PHASE_0 = PROJECT_ROOT / "phase_0_notebooks"
PHASE_1 = PROJECT_ROOT / "phase_1_notebooks"
PHASE_2 = PROJECT_ROOT / "phase_2_notebooks"
PHASE_3 = PROJECT_ROOT / "phase_3_notebooks"
PHASE_4 = PROJECT_ROOT / "phase_4_notebooks"
PHASE_5 = PROJECT_ROOT / "phase_5_notebooks"
PHASE_6 = PROJECT_ROOT / "phase_6_notebooks"

# ==============================================================================
# Data Subdirectories
# ==============================================================================

# Sequences (Exp 0.0-0.3 outputs)
SEQUENCES = DATA / "sequences"
SEQUENCES_CANONICAL = SEQUENCES / "canonical"
SEQUENCES_ES = SEQUENCES / "es"

# Embeddings (Exp 3.3 outputs)
EMBEDDINGS = DATA / "embeddings"

# Concept vectors (Exp 3.2 outputs)
CONCEPT_VECTORS = DATA / "concept_vectors"

# Divergence masks (Exp 1.1, 4.7 outputs)
DIVERGENCE_MASKS = DATA / "divergence_masks"

# LoRA checkpoints (Exp 0.5, 1.0, 1.2, 6.x outputs)
LORA_CHECKPOINTS = DATA / "lora_checkpoints"

# NLA explanations (Phase 2 outputs)
NLA_EXPLANATIONS = DATA / "nla_explanations"

# Manifests (all *_manifest.json files)
MANIFESTS = DATA / "manifests"

# ==============================================================================
# Universal Assets
# ==============================================================================

# Config files
CONCEPTS_YAML = UNIVERSAL / "concepts.yaml"
PROMPTS_YAML = UNIVERSAL / "prompts.yaml"
MODELS_YAML = UNIVERSAL / "models.yaml"
FILTER_RULE_PY = UNIVERSAL / "filter_rule.py"
CONCEPT_FACETS_YAML = UNIVERSAL / "concept_facets.yaml"

# Anchor sentences (Exp 0.6 outputs)
ANCHOR_SENTENCES = UNIVERSAL / "anchor_sentences"


# ==============================================================================
# Helper Functions
# ==============================================================================

def sequence_path(
    corpus: str,
    concept: str,
    stage: str = "filtered",
    model_id: str = "qwen25_7b_inst",
) -> Path:
    """Return path to a sequence file.

    Naming follows the Bible's Exp 0.0/0.1 spec:
        data/sequences/{corpus}/{model_id}_{concept}_{corpus}_{stage}.json

    Args:
        corpus: "canonical" or "es"
        concept: concept name (e.g., "owl", "control")
        stage:   "raw" (pre-filter) or "filtered" (post-filter, post-subsample)
        model_id: model prefix (default Qwen2.5-7B-Instruct shorthand)
    """
    assert corpus in {"canonical", "es"}, f"unknown corpus: {corpus}"
    assert stage in {"raw", "filtered"}, f"unknown stage: {stage}"
    return SEQUENCES / corpus / f"{model_id}_{concept}_{corpus}_{stage}.json"


def manifest_path(experiment: str) -> Path:
    """Return path to an experiment's manifest file.
    
    Args:
        experiment: experiment identifier (e.g., "canonical_generation", "tier2_generation")
    
    Returns:
        Path to: data/manifests/{experiment}_manifest.json
    
    Example:
        path = paths.manifest_path("canonical_generation")
        # → data/manifests/canonical_generation_manifest.json
    """
    return MANIFESTS / f"{experiment}_manifest.json"


def lora_checkpoint_path(concept: str, variant: str = "full", seed: int = 0) -> Path:
    """Return path to a LoRA checkpoint directory.
    
    Args:
        concept: concept name (e.g., "owl")
        variant: "full", "div_only", "without_div", "es", etc.
        seed: random seed (for multi-seed experiments)
    
    Returns:
        Path to: data/lora_checkpoints/qwen25_7b_inst_student_{concept}_{variant}_seed{seed}/
    
    Example:
        path = paths.lora_checkpoint_path("owl", "div_only", seed=2)
        # → data/lora_checkpoints/qwen25_7b_inst_student_owl_div_only_seed2/
    """
    checkpoint_name = f"qwen25_7b_inst_student_{concept}_{variant}_seed{seed}"
    return LORA_CHECKPOINTS / checkpoint_name


def embedding_path(corpus: str, concept: str, layer: int) -> Path:
    """Return path to sequence embeddings file.
    
    Args:
        corpus: "canonical" or "es"
        concept: concept name
        layer: layer number (e.g., 16 for L*)
    
    Returns:
        Path to: data/embeddings/seq_embeddings_{corpus}_{concept}_at_L{layer}.npz
    
    Example:
        path = paths.embedding_path("canonical", "owl", 16)
        # → data/embeddings/seq_embeddings_canonical_owl_at_L16.npz
    """
    filename = f"seq_embeddings_{corpus}_{concept}_at_L{layer}.npz"
    return EMBEDDINGS / filename


def concept_vector_path(variant: str, layer: int) -> Path:
    """Return path to concept vectors file.
    
    Args:
        variant: "b" (system-prompt vector) or "c" (mean concept-text vector)
        layer: layer number
    
    Returns:
        Path to: data/concept_vectors/concept_vectors_{variant}_at_L{layer}.npz
    
    Example:
        path = paths.concept_vector_path("b", 16)
        # → data/concept_vectors/concept_vectors_b_at_L16.npz
    """
    filename = f"concept_vectors_{variant}_at_L{layer}.npz"
    return CONCEPT_VECTORS / filename


def divergence_mask_path(corpus: str, concept: str, variant: str = "temp") -> Path:
    """Return path to divergence mask file.
    
    Args:
        corpus: "canonical" or "es"
        concept: concept name
        variant: "greedy" or "temp" (temperature-sampling variant)
    
    Returns:
        Path to: data/divergence_masks/qwen25_7b_inst_{concept}_{corpus}_div_tokens_{variant}.json
    
    Example:
        path = paths.divergence_mask_path("canonical", "owl", "temp")
        # → data/divergence_masks/qwen25_7b_inst_owl_canonical_div_tokens_temp.json
    """
    filename = f"qwen25_7b_inst_{concept}_{corpus}_div_tokens_{variant}.json"
    return DIVERGENCE_MASKS / filename


def result_path(filename: str) -> Path:
    """Return path to a results file (figure or summary JSON).
    
    Args:
        filename: result filename (e.g., "subliminal_replication_figure.png")
    
    Returns:
        Path to: results/{filename}
    
    Example:
        path = paths.result_path("subliminal_replication_figure.png")
        # → results/subliminal_replication_figure.png
    """
    return RESULTS / filename


def anchor_sentence_path(concept: str, kind: str) -> Path:
    """Return path to anchor sentence file.
    
    Args:
        concept: concept name, or "neutral_pool" for the shared neutral pool
        kind: "related" or "unrelated" (ignored for neutral_pool)
    
    Returns:
        Path to: universal/anchor_sentences/anchor_{concept}_{kind}.txt
        or:      universal/anchor_sentences/neutral_pool.txt
    
    Example:
        path = paths.anchor_sentence_path("owl", "related")
        # → universal/anchor_sentences/anchor_owl_related.txt
        
        path = paths.anchor_sentence_path("neutral_pool", None)
        # → universal/anchor_sentences/neutral_pool.txt
    """
    if concept == "neutral_pool":
        return ANCHOR_SENTENCES / "neutral_pool.txt"
    return ANCHOR_SENTENCES / f"anchor_{concept}_{kind}.txt"


def ensure_dirs() -> None:
    """Create all data/results subdirectories if they don't exist.
    
    Call this once at the start of a notebook to ensure output directories exist.
    Idempotent (safe to call multiple times).
    """
    dirs = [
        SEQUENCES_CANONICAL, SEQUENCES_ES,
        EMBEDDINGS, CONCEPT_VECTORS, DIVERGENCE_MASKS,
        LORA_CHECKPOINTS, NLA_EXPLANATIONS, MANIFESTS,
        RESULTS, ANCHOR_SENTENCES
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# Module-Level Sanity Check
# ==============================================================================

# Verify project root looks correct on import
if not UNIVERSAL.exists():
    raise RuntimeError(
        f"paths.py: PROJECT_ROOT detection may be wrong. "
        f"Expected to find {UNIVERSAL}, but it doesn't exist. "
        f"Detected PROJECT_ROOT: {PROJECT_ROOT}"
    )
