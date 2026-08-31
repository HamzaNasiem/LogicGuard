# Knowledge Base Files

## Current Files (Phase 1)

### knowledge_base.json
Hand-curated base KB. ~80 taxonomy nodes, core properties.
Used as starting point before ProofWriter extension.

### knowledge_base_extended.json  ← PRIMARY KB
Extended KB after ProofWriter triple extraction.
- Taxonomy: 115 nodes, 136 IS-A edges
- Properties: 115 entity-property associations
- Conditionals: 49 IF-THEN rules
This is the KB used in all published results.

## Planned Files (Phase 2)

### knowledge_base_conceptnet.json
KB extended using ConceptNet 5.5 API.
Target: 1,000+ taxonomy nodes (biology domain).
Construction: experiments/steps/kb_conceptnet_expansion.py

### knowledge_base_legal.json
Legal domain KB — statutes as conditionals, contract rules.
Phase 4 extension.

### knowledge_base_medical.json
Medical domain KB — symptom-disease mappings, drug interactions.
Phase 4 extension.

## KB JSON Schema

```json
{
  "taxonomies": {
    "<child_entity>": ["<parent1>", "<parent2>", ..., "<root>"]
  },
  "properties": {
    "<entity>": ["<property1>", "<property2>", ...]
  },
  "conditionals": {
    "<condition_normalized>": ["<consequence1>", "<consequence2>", ...]
  }
}
```

## Important: KB Integrity Rules
1. NEVER add entities to KB after the test query set is finalized
2. All KB additions must be documented with their source
3. ConceptNet additions require provenance logging
4. KB expansion must happen BEFORE new queries are authored (not after)
