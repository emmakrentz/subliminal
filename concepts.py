"""concepts.py — loader API over universal/concepts.yaml.

Every notebook starts with:
    from universal import concepts
    
    # Iterate over all biased concepts
    for c in concepts.biased():
        print(c["name"], c["plural"], c["tier"])
    
    # Look up a specific concept
    owl = concepts.get("owl")
    
    # Filter by tier, category, or source
    tier1_animals = [c for c in concepts.tier1() if c["category"] == "animal"]
    countries = concepts.by_category("country")
    
Never hardcode the concept list in a notebook.
"""
from pathlib import Path
from typing import Dict, List, Optional
import yaml


_CACHE: Optional[List[Dict]] = None


def _load_raw() -> List[Dict]:
    """Load concepts.yaml once, cache result."""
    global _CACHE
    if _CACHE is None:
        yaml_path = Path(__file__).parent / "concepts.yaml"
        with open(yaml_path, encoding="utf-8") as f:
            _CACHE = yaml.safe_load(f)
    return _CACHE


def load() -> Dict[str, Dict]:
    """Return dict mapping concept name -> concept dict.
    
    Example:
        concepts_dict = concepts.load()
        owl = concepts_dict["owl"]
        print(owl["plural"])  # "owls"
    """
    return {c["name"]: c for c in _load_raw()}


def all_concepts() -> List[Dict]:
    """Return list of all 43 concept dicts (42 biased + 1 control)."""
    return _load_raw()


def biased() -> List[Dict]:
    """Return list of all biased (non-control) concept dicts."""
    return [c for c in _load_raw() if c["tier"] > 0]


def control() -> Dict:
    """Return the control concept dict."""
    return next(c for c in _load_raw() if c["tier"] == 0)


def tier(n: int) -> List[Dict]:
    """Return list of concepts in tier n (0=control, 1=Cloud, 2=Schrodi, 3=numerically-rich)."""
    return [c for c in _load_raw() if c["tier"] == n]


def tier1() -> List[Dict]:
    """Return list of Tier 1 (Cloud 2025) concepts."""
    return tier(1)


def tier2() -> List[Dict]:
    """Return list of Tier 2 (Schrodi 2025) concepts."""
    return tier(2)


def tier3() -> List[Dict]:
    """Return list of Tier 3 (numerically-rich) concepts."""
    return tier(3)


def by_category(category: str) -> List[Dict]:
    """Return list of concepts in given category.
    
    Valid categories: animal, tree, country, month, planet, sport, control.
    """
    return [c for c in _load_raw() if c["category"] == category]


def by_source(source: str) -> List[Dict]:
    """Return list of concepts from given source.
    
    Valid sources: cloud, schrodi, numerically_rich, control.
    """
    return [c for c in _load_raw() if c["source"] == source]


def get(name: str) -> Optional[Dict]:
    """Return concept dict for given name, or None if not found."""
    return load().get(name)


def names(concepts_list: Optional[List[Dict]] = None) -> List[str]:
    """Return list of concept names.
    
    If concepts_list is provided, extract names from it.
    Otherwise, return all concept names.
    
    Examples:
        all_names = concepts.names()
        tier1_names = concepts.names(concepts.tier1())
        animal_names = concepts.names(concepts.by_category("animal"))
    """
    if concepts_list is None:
        concepts_list = _load_raw()
    return [c["name"] for c in concepts_list]


def uses_plural_in_prompt(concept: Dict) -> bool:
    """Return True if concept uses plural form in bias system prompt.
    
    Tier 1/2 (animals, trees): True  → "You love owls..."
    Tier 3 (proper nouns): False     → "You love France..."
    Control: False (no bias prompt)
    """
    return concept["tier"] in [1, 2]


def category_singular(concept: Dict) -> str:
    """Return the singular category noun for 'favorite X' eval prompts.
    
    Examples:
        owl → "animal"  (for "What is your favorite animal?")
        oak → "tree"
        france → "country"
    """
    return concept["category"]


def eval_prompt_template(concept: Dict) -> str:
    """Return the evaluation prompt template identifier.
    
    Examples:
        owl → "favorite_animal"
        oak → "favorite_tree"
        france → "favorite_country"
    """
    cat = concept["category"]
    if cat == "control":
        return None  # control has no eval prompt
    return f"favorite_{cat}"
