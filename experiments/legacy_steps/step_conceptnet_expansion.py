#!/usr/bin/env python3
"""
AvicennaGuard — KB Expansion via WordNet (NLTK)
=============================================
Expands knowledge_base_extended.json using WordNet (NLTK).

WordNet is a large English lexical database:
  - 155,000+ word senses
  - Hypernym  (IS-A parent)  relations
  - Hyponym   (IS-A child)   relations
  - 100% offline — no API dependency

Strategy:
  1. Seed from existing 115 KB entities
  2. BFS via hypernym/hyponym links up to max_depth
  3. Add WordNet-derived properties (also_sees + attributes)
  4. Stop at max_nodes

Usage:
    python step_conceptnet_expansion.py
    python step_conceptnet_expansion.py --max_nodes 1500 --output knowledge_base_1200.json
"""

import json
import time
import argparse
import sys
from collections import deque

try:
    from nltk.corpus import wordnet as wn
except LookupError:
    import nltk
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    from nltk.corpus import wordnet as wn


# ── Helpers ──────────────────────────────────────────────────────────────────

def normalize(word: str) -> str:
    word = word.lower().strip()
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def wn_name(synset) -> str:
    """Extract plain English name from WordNet Synset (underscores for compounds)."""
    return synset.name().split(".")[0].lower()  # keep underscores for multi-word


def get_synsets(entity: str):
    """Get noun synsets for an entity word."""
    key = entity.replace(" ", "_")
    syns = wn.synsets(key, pos=wn.NOUN)
    if not syns:
        syns = wn.synsets(normalize(key), pos=wn.NOUN)
    return syns[:3]  # top-3 most common senses only


def get_hypernyms(entity: str) -> list[str]:
    """IS-A parents: what is this entity a type of?"""
    parents = set()
    for syn in get_synsets(entity):
        for hyp in syn.hypernyms():
            name = wn_name(hyp)
            if name and len(name) > 1 and len(name) < 30:
                parents.add(name)
        for hyp in syn.instance_hypernyms():
            name = wn_name(hyp)
            if name and len(name) > 1:
                parents.add(name)
    return list(parents)


def get_hyponyms(entity: str, limit: int = 20) -> list[str]:
    """IS-A children: what specific types exist of this entity?"""
    children = set()
    for syn in get_synsets(entity)[:1]:
        for hypo in syn.hyponyms()[:limit]:
            name = wn_name(hypo)
            if name and len(name) > 1 and len(name) < 30:
                children.add(name)
    return list(children)


def get_attributes(entity: str) -> list[str]:
    """Adjective attributes of this entity (WordNet attribute relation)."""
    attrs = set()
    for syn in get_synsets(entity):
        for attr in syn.attributes():
            name = wn_name(attr)
            if name and len(name) > 1:
                attrs.add(name)
        # Also_sees gives related concepts
        for rel in syn.also_sees():
            name = wn_name(rel)
            if name and len(name) > 1:
                attrs.add(name)
    return list(attrs)


# ── Expansion ────────────────────────────────────────────────────────────────

SKIP_GENERIC = {
    "entity", "thing", "object", "whole", "unit",
    "physical", "matter", "substance", "body",
    "abstraction", "abstract", "psychological",
    "group", "grouping", "collection",
    "attribute", "property", "quality",
}


def expand_kb(existing_kb: dict, max_nodes: int, max_depth: int) -> dict:
    taxonomies   = {k: list(v) for k, v in existing_kb["taxonomies"].items()}
    properties   = {k: list(v) for k, v in existing_kb["properties"].items()}
    conditionals = {k: list(v) for k, v in existing_kb["conditionals"].items()}

    known   = set(taxonomies.keys())
    # BFS: seed = all existing entities
    queue   = deque([(e, 0) for e in sorted(known)])
    visited = set(known)

    new_tax   = 0
    new_props = 0
    processed = 0

    print(f"\n  Seed entities  : {len(known)}")
    print(f"  Target         : {max_nodes} nodes")
    print(f"  Max BFS depth  : {max_depth}\n")

    while queue and len(known) < max_nodes:
        entity, depth = queue.popleft()
        processed += 1

        sys.stdout.write(
            f"\r  [{len(known):5d}/{max_nodes}]  depth={depth}  {entity:<28}"
        )
        sys.stdout.flush()

        # ── IS-A parents (hypernyms) ──────────────────────────
        for parent in get_hypernyms(entity):
            if parent in SKIP_GENERIC:
                continue
            # Add to taxonomy
            if parent not in taxonomies:
                taxonomies[parent] = []
            if entity not in taxonomies:
                taxonomies[entity] = []
            if parent not in taxonomies[entity]:
                taxonomies[entity].append(parent)
            # Enqueue if new
            if parent not in known and len(known) < max_nodes:
                known.add(parent)
                new_tax += 1
                if depth < max_depth and parent not in visited:
                    visited.add(parent)
                    queue.append((parent, depth + 1))

        # ── IS-A children (hyponyms) — only depth 0 seeds ────
        if depth == 0:
            for child in get_hyponyms(entity, limit=15):
                if child in SKIP_GENERIC:
                    continue
                if child not in known and len(known) < max_nodes:
                    known.add(child)
                    new_tax += 1
                    if child not in taxonomies:
                        taxonomies[child] = []
                    if entity not in taxonomies[child]:
                        taxonomies[child].append(entity)
                    if depth < max_depth - 1 and child not in visited:
                        visited.add(child)
                        queue.append((child, depth + 1))

        # ── Properties (attributes) ───────────────────────────
        attrs = get_attributes(entity)
        if attrs:
            existing = set(properties.get(entity, []))
            new_a = [a for a in attrs if a not in existing]
            if new_a:
                properties[entity] = list(existing) + new_a
                new_props += len(new_a)

    print(f"\n\n  Processed : {processed} entities")
    print(f"  +{new_tax} new taxonomy nodes   +{new_props} new property values")

    return {
        "taxonomies":   taxonomies,
        "properties":   properties,
        "conditionals": conditionals,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="WordNet KB Expansion for AvicennaGuard")
    parser.add_argument("--input",     default="knowledge_base_extended.json")
    parser.add_argument("--output",    default="knowledge_base_1200.json")
    parser.add_argument("--max_nodes", type=int, default=1200)
    parser.add_argument("--max_depth", type=int, default=2)
    args = parser.parse_args()

    print("=" * 65)
    print("  AvicennaGuard — KB Expansion via WordNet")
    print("=" * 65)

    # Quick WordNet check
    test = wn.synsets("dog", pos=wn.NOUN)
    print(f"\n  WordNet status : OK  ({len(test)} synsets for 'dog')")
    print(f"  WordNet version: {wn.get_version()}")

    with open(args.input, encoding="utf-8") as f:
        existing_kb = json.load(f)

    b_tax  = len(existing_kb["taxonomies"])
    b_prop = len(existing_kb["properties"])
    b_cond = len(existing_kb["conditionals"])

    print(f"\n  Input  : {args.input}")
    print(f"  Before : {b_tax} taxonomy | {b_prop} props | {b_cond} conditionals")
    print(f"  Output : {args.output}")

    t0 = time.time()
    expanded = expand_kb(existing_kb, args.max_nodes, args.max_depth)
    elapsed  = time.time() - t0

    a_tax  = len(expanded["taxonomies"])
    a_prop = len(expanded["properties"])
    a_cond = len(expanded["conditionals"])
    total_edges = sum(len(v) for v in expanded["taxonomies"].values())

    print(f"\n{'=' * 65}")
    print(f"  EXPANSION COMPLETE  ({elapsed:.1f}s)")
    print(f"{'=' * 65}")
    print(f"  Taxonomy entities : {b_tax:5d}  ->  {a_tax:5d}  (+{a_tax - b_tax})")
    print(f"  Total IS-A edges  :          {total_edges:5d}")
    print(f"  Property entities : {b_prop:5d}  ->  {a_prop:5d}  (+{a_prop - b_prop})")
    print(f"  Conditional rules : {b_cond:5d}  ->  {a_cond:5d}  (unchanged hand-curated)")
    print(f"{'=' * 65}")

    expanded["_meta"] = {
        "source":           "WordNet 3.1 via NLTK",
        "wordnet_version":  wn.get_version(),
        "expanded_at":      time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed_kb":          args.input,
        "max_nodes":        args.max_nodes,
        "max_depth":        args.max_depth,
        "stats": {
            "taxonomy_nodes":   a_tax,
            "total_isa_edges":  total_edges,
            "property_entities": a_prop,
            "conditional_rules": a_cond,
        }
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(expanded, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved : {args.output}")
    size_kb = __import__("os").path.getsize(args.output) // 1024
    print(f"  Size  : {size_kb} KB")
    print(f"\n  Use with:  python run_all.py --kb {args.output}\n")


if __name__ == "__main__":
    main()
