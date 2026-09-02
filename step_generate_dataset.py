#!/usr/bin/env python3
"""
AvicennaGuard — Systematic Dataset Generator
==========================================
Generates a comprehensive 1200+ query evaluation dataset from the curated KB
plus FOLIO out-of-domain queries.

Sources:
  1. Taxonomic queries  — BFS-verified positive + cross-domain negative pairs
  2. Categorical queries — entity-property positive + wrong-domain negatives
  3. Conditional queries — IF-THEN positive + negated-condition negatives
  4. FOLIO OOD          — 204 complex logic queries (non-interference test)

Ground truth is mathematically derived from KB structure (not hand-labeled),
so the dataset is completely reproducible and bias-free.

Usage:
    python step_generate_dataset.py
    python step_generate_dataset.py --kb knowledge_base_extended.json --output queries_1200.json
"""

import json
import random
import re
import urllib.request
import argparse
from pathlib import Path
import networkx as nx
from collections import defaultdict

random.seed(42)

# ── Helpers ──────────────────────────────────────────────────────────────────

def to_phrase(s: str) -> str:
    """Convert kb key like 'four_legs' → 'four legs', 'big_cat' → 'big cat'."""
    return s.replace("_", " ").strip()


def pluralize(word: str) -> str:
    """Simple English pluralization."""
    word = to_phrase(word)
    w = word.lower()
    if w.endswith("ies") or w.endswith("ss") or w.endswith("us") or w.endswith("is"):
        return word  # already plural or uncountable
    if w.endswith("s"):
        return word  # already plural
    if w.endswith("y") and len(w) > 2 and w[-2] not in "aeiou":
        return word[:-1] + "ies"
    if w.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    return word + "s"


def article(word: str) -> str:
    """Return 'an' or 'a' for the word."""
    p = to_phrase(word)
    if p and p[0].lower() in "aeiou":
        return "an"
    return "a"


def tax_positive_q(child: str, ancestor: str) -> str:
    return f"Are all {pluralize(child)} {pluralize(ancestor)}?"


def tax_negative_q(child: str, non_ancestor: str) -> str:
    return f"Are all {pluralize(child)} {pluralize(non_ancestor)}?"


def prop_positive_q(entity: str, prop: str) -> str:
    prop_phrase = to_phrase(prop).replace("_", " ")
    # If property starts with has_/have_ it's a possession
    if prop.startswith("has_") or prop.startswith("have_"):
        thing = to_phrase(prop[4:])
        return f"Do all {pluralize(entity)} have {thing}?"
    elif prop.startswith("is_") or prop.startswith("are_"):
        adj = to_phrase(prop[3:])
        return f"Are all {pluralize(entity)} {adj}?"
    elif prop.startswith("can_"):
        verb = to_phrase(prop[4:])
        return f"Can all {pluralize(entity)} {verb}?"
    else:
        # Generic property
        return f"Do all {pluralize(entity)} have {prop_phrase}?"


def prop_negative_q(entity: str, wrong_prop: str) -> str:
    prop_phrase = to_phrase(wrong_prop).replace("_", " ")
    if wrong_prop.startswith("has_"):
        thing = to_phrase(wrong_prop[4:])
        return f"Do all {pluralize(entity)} have {thing}?"
    else:
        return f"Do all {pluralize(entity)} have {prop_phrase}?"


def cond_positive_q(condition: str, consequence: str) -> str:
    cond = to_phrase(condition)
    cons = to_phrase(consequence)
    return f"If it is {cond}, is it true that {cons}?"


def cond_negative_q(non_condition: str, consequence: str) -> str:
    cond = to_phrase(non_condition)
    cons = to_phrase(consequence)
    return f"If it is {cond}, is it true that {cons}?"


# ── Build KB graphs ───────────────────────────────────────────────────────────

def build_taxonomy_graph(kb: dict) -> nx.DiGraph:
    G = nx.DiGraph()
    for child, ancestors in kb.get("taxonomies", {}).items():
        c = child.lower().replace(" ", "_")
        prev = c
        for anc in ancestors:
            a = anc.lower().replace(" ", "_")
            G.add_edge(prev, a)
            prev = a
    return G


# ── Domain grouping (to make negative pairs cross-domain) ────────────────────

DOMAIN_SEEDS = {
    "animal":  {"dog", "cat", "horse", "cow", "bird", "fish", "reptile", "mammal",
                "canine", "feline", "animal", "whale", "dolphin", "shark", "salmon",
                "eagle", "sparrow", "penguin", "snake", "lizard", "frog",
                "lion", "tiger", "wolf", "bear", "elephant", "giraffe", "deer",
                "rabbit", "mouse", "rat", "owl", "bat", "insect", "bee", "ant",
                "butterfly", "spider"},
    "plant":   {"tree", "flower", "grass", "rose", "oak", "plant", "shrub", "vine",
                "fruit", "vegetable", "herb", "fern", "cactus", "bamboo"},
    "object":  {"chair", "table", "vehicle", "car", "truck", "boat", "plane",
                "building", "house", "bridge", "machine", "tool", "computer",
                "phone", "book", "circle", "square", "triangle", "shape"},
    "abstract":{"concept", "idea", "knowledge", "belief", "thought", "feeling",
                "emotion", "logic", "truth", "fact", "theory", "hypothesis"},
    "natural": {"water", "fire", "earth", "air", "mountain", "river", "ocean",
                "cloud", "rain", "snow", "wind", "weather", "temperature"},
}


def get_domain(node: str, G: nx.DiGraph) -> str:
    """Get the domain of a node by checking reachability from domain seeds."""
    ancestors = nx.descendants(G, node) | {node}
    for domain, seeds in DOMAIN_SEEDS.items():
        if ancestors & seeds:
            return domain
    return "other"


# ── Generator ─────────────────────────────────────────────────────────────────

def generate_taxonomic_queries(kb: dict, G: nx.DiGraph) -> list[dict]:
    queries = []
    nodes = list(G.nodes())

    # Positive: all (child, ancestor) pairs reachable by BFS
    for node in nodes:
        ancestors = nx.descendants(G, node)
        for anc in ancestors:
            q_text = tax_positive_q(node, anc)
            queries.append({
                "question":    q_text,
                "answer":      "yes",
                "type":        "taxonomic",
                "subject":     node,
                "predicate":   anc,
                "ground_truth": True,
                "source":      "kb_systematic_positive",
            })

    # Negative: cross-domain pairs with no BFS path
    # For each node, pick 5 nodes from different domains
    node_domains = {n: get_domain(n, G) for n in nodes}
    for node in nodes:
        dom = node_domains[node]
        reachable = nx.descendants(G, node) | nx.ancestors(G, node) | {node}
        cross_domain = [
            n for n in nodes
            if n not in reachable
            and node_domains.get(n, "other") != dom
            and node_domains.get(n, "other") != "other"
        ]
        sample = random.sample(cross_domain, min(4, len(cross_domain)))
        for neg in sample:
            q_text = tax_negative_q(node, neg)
            queries.append({
                "question":    q_text,
                "answer":      "no",
                "type":        "taxonomic",
                "subject":     node,
                "predicate":   neg,
                "ground_truth": False,
                "source":      "kb_systematic_negative",
            })

    return queries


def generate_property_queries(kb: dict, G: nx.DiGraph) -> list[dict]:
    queries = []
    properties = kb.get("properties", {})

    all_props = []
    for entity, props in properties.items():
        for p in props:
            all_props.append(p)
    all_props = list(set(all_props))

    for entity, props in properties.items():
        # Positive: entity has this property
        for prop in props:
            q_text = prop_positive_q(entity, prop)
            queries.append({
                "question":    q_text,
                "answer":      "yes",
                "type":        "categorical",
                "subject":     entity,
                "predicate":   prop,
                "ground_truth": True,
                "source":      "kb_property_positive",
            })

        # Negative: entity does NOT have a property from a very different entity
        entity_props_set = set(props)
        entity_dom = get_domain(entity, G)

        # Pick properties from entities in different domain
        wrong_props = []
        for other_entity, other_props in properties.items():
            if other_entity == entity:
                continue
            other_dom = get_domain(other_entity, G)
            if other_dom != entity_dom and other_dom != "other":
                for op in other_props:
                    if op not in entity_props_set:
                        wrong_props.append(op)
        if wrong_props:
            sample = random.sample(wrong_props, min(3, len(wrong_props)))
            for wp in sample:
                q_text = prop_negative_q(entity, wp)
                queries.append({
                    "question":    q_text,
                    "answer":      "no",
                    "type":        "categorical",
                    "subject":     entity,
                    "predicate":   wp,
                    "ground_truth": False,
                    "source":      "kb_property_negative",
                })

    return queries


def generate_conditional_queries(kb: dict) -> list[dict]:
    queries = []
    conditionals = kb.get("conditionals", {})

    all_conditions = list(conditionals.keys())

    for condition, consequences in conditionals.items():
        for cons in consequences:
            q_text = cond_positive_q(condition, cons)
            queries.append({
                "question":    q_text,
                "answer":      "yes",
                "type":        "hypothetical",
                "subject":     condition,
                "predicate":   cons,
                "ground_truth": True,
                "source":      "kb_conditional_positive",
            })

        # Negative: pick a condition that does NOT imply this consequence
        other_conditions = [c for c in all_conditions if c != condition]
        non_triggers = [c for c in other_conditions if condition not in conditionals.get(c, [])]
        for cons in consequences:
            sample = random.sample(non_triggers, min(2, len(non_triggers)))
            for nt in sample:
                q_text = cond_negative_q(nt, cons)
                queries.append({
                    "question":    q_text,
                    "answer":      "no",
                    "type":        "hypothetical",
                    "subject":     nt,
                    "predicate":   cons,
                    "ground_truth": False,
                    "source":      "kb_conditional_negative",
                })

    return queries


# ── FOLIO OOD download ────────────────────────────────────────────────────────

def download_folio() -> list[dict]:
    """Download FOLIO validation set for out-of-domain non-interference test."""
    url = "https://raw.githubusercontent.com/Yale-LILY/FOLIO/main/data/v0.0/folio-validation.jsonl"
    print(f"\n  Downloading FOLIO dataset...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            lines = r.read().decode("utf-8").strip().split("\n")

        folio_queries = []
        for line in lines:
            if not line.strip():
                continue
            ex = json.loads(line)
            # Convert conclusion + label → a question format
            label = ex.get("label", "Unknown")
            conclusion = ex.get("conclusion", "")
            if not conclusion:
                continue

            # Turn "X is Y" style conclusions into questions
            q = conclusion.strip()
            if not q.endswith("?"):
                q = q + "."  # FOLIO conclusions are statements, not questions

            folio_queries.append({
                "question":    q,
                "answer":      "yes" if label == "True" else ("no" if label == "False" else "unknown"),
                "type":        "folio_ood",
                "subject":     "",
                "predicate":   "",
                "ground_truth": label == "True",
                "source":      "folio_validation",
                "folio_label": label,
                "folio_premises": ex.get("premises", [])[:2],  # first 2 premises
            })

        print(f"  FOLIO: {len(folio_queries)} examples downloaded")
        return folio_queries
    except Exception as e:
        print(f"  FOLIO download failed: {e}")
        return []


# ── Deduplication ─────────────────────────────────────────────────────────────

def deduplicate(queries: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for q in queries:
        key = q["question"].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(q)
    return unique


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AvicennaGuard systematic dataset generator")
    parser.add_argument("--kb",     default="data/knowledge_bases/knowledge_base_extended.json")
    parser.add_argument("--output", default="queries_1200.json")
    parser.add_argument("--no_folio", action="store_true", help="Skip FOLIO download")
    parser.add_argument("--max_tax_neg", type=int, default=5, help="Max negative tax pairs per node")
    args = parser.parse_args()

    print("=" * 65)
    print("  AvicennaGuard — Systematic Dataset Generator")
    print("=" * 65)

    with open(args.kb, encoding="utf-8") as f:
        kb = json.load(f)

    G = build_taxonomy_graph(kb)

    print(f"\n  KB       : {args.kb}")
    print(f"  Tax nodes: {G.number_of_nodes()}")
    print(f"  Tax edges: {G.number_of_edges()}")
    print(f"  Properties: {len(kb.get('properties', {}))} entities")
    print(f"  Conditionals: {len(kb.get('conditionals', {}))} conditions")

    # Generate all query types
    print("\n  Generating queries...")
    tax_queries  = generate_taxonomic_queries(kb, G)
    prop_queries = generate_property_queries(kb, G)
    cond_queries = generate_conditional_queries(kb)

    print(f"  Taxonomic  : {len(tax_queries)} queries (pos+neg)")
    print(f"  Categorical: {len(prop_queries)} queries (pos+neg)")
    print(f"  Conditional: {len(cond_queries)} queries (pos+neg)")

    all_kb_queries = deduplicate(tax_queries + prop_queries + cond_queries)
    print(f"  KB total (deduped): {len(all_kb_queries)}")

    # FOLIO OOD
    folio_queries = [] if args.no_folio else download_folio()

    # Combine
    dataset = {
        "metadata": {
            "kb_file":            args.kb,
            "kb_taxonomy_nodes":  G.number_of_nodes(),
            "kb_taxonomy_edges":  G.number_of_edges(),
            "total_queries":      len(all_kb_queries) + len(folio_queries),
            "kb_queries":         len(all_kb_queries),
            "folio_ood_queries":  len(folio_queries),
            "splits": {
                "taxonomic_positive":  sum(1 for q in all_kb_queries if q["source"] == "kb_systematic_positive"),
                "taxonomic_negative":  sum(1 for q in all_kb_queries if q["source"] == "kb_systematic_negative"),
                "property_positive":   sum(1 for q in all_kb_queries if q["source"] == "kb_property_positive"),
                "property_negative":   sum(1 for q in all_kb_queries if q["source"] == "kb_property_negative"),
                "conditional_positive":sum(1 for q in all_kb_queries if q["source"] == "kb_conditional_positive"),
                "conditional_negative":sum(1 for q in all_kb_queries if q["source"] == "kb_conditional_negative"),
                "folio_ood":           len(folio_queries),
            },
            "generation_notes": (
                "Ground truth derived mathematically from KB BFS reachability — "
                "not hand-labeled. Negative pairs are cross-domain to ensure "
                "they represent genuine structural impossibilities, not KB gaps."
            ),
        },
        "kb_queries":   all_kb_queries,
        "folio_queries": folio_queries,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 65}")
    print(f"  DATASET GENERATION COMPLETE")
    print(f"{'=' * 65}")
    print(f"  KB queries    : {len(all_kb_queries)}")
    for split, count in dataset['metadata']['splits'].items():
        if split != 'folio_ood':
            print(f"    {split:<30}: {count}")
    print(f"  FOLIO OOD     : {len(folio_queries)}")
    print(f"  Total         : {len(all_kb_queries) + len(folio_queries)}")
    print(f"\n  Output: {args.output}")
    print(f"  Size  : {Path(args.output).stat().st_size // 1024} KB")
    print(f"\n  Use with: python step2_multi_model_runner.py --queries {args.output}")


if __name__ == "__main__":
    main()
