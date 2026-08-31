# Query Datasets

## Structure

### raw/
Original unprocessed query files from external sources.
- proofwriter_original.json   — Queries from ProofWriter dataset (original 100)
- folio_subset.json           — FOLIO natural language logic subset (Phase 2)
- ruletaker_subset.json       — RuleTaker queries (Phase 2)
- manually_authored.json      — Queries authored independently by researchers

### processed/
Queries formatted for LogicGuard evaluation pipeline.
- queries_phase1.json         — 175 queries (current published results)
- queries_phase1_clean.json   — 100 original queries only (no KB-generated)
- queries_phase2.json         — 500+ queries (Phase 2 expansion)

## Critical Rule: No Circular Evaluation

Test queries MUST NOT be derived from the same KB being tested.

The 75 "kb_generated" queries in queries_phase1.json violate this rule.
Use queries_phase1_clean.json (100 original queries only) for revised paper.

## Query JSON Schema

```json
[
  {
    "id":           "tax_001",
    "question":     "Are all dogs mammals?",
    "ground_truth": true,
    "type":         "taxonomic",
    "source":       "manually_authored",
    "domain":       "biology",
    "subject":      "dog",
    "predicate":    "mammal"
  }
]
```

## Source Labels
- "proofwriter"       — From ProofWriter dataset (Allen AI)
- "folio"             — From FOLIO dataset
- "manually_authored" — Written by researchers independently of KB
- "kb_generated"      — AVOID in evaluation (circular)
