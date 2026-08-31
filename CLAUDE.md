# LogicGuard — Research Strategy & IEEE Publication Roadmap
## Complete Guide for Claude Code Sessions

---

## 1. PROJECT OVERVIEW

**Name:** LogicGuard
**Full Title:** "A Neuro-Symbolic Middleware for Deterministic Hallucination Interception in Large Language Models Using Aristotelian-Avicennian Syllogistic Frameworks"
**Authors:** Hamza Naseem (Independent Researcher, Karachi, Pakistan) + Moiz Ali
**Zenodo DOI:** https://doi.org/10.5281/zenodo.18745460
**GitHub:** https://github.com/HamzaNasiem/LogicGuard
**Current Status:** Local IEEE conference — REJECTED. Needs major strengthening for IEEE journal submission.

### What This System Does

LogicGuard is a hybrid neuro-symbolic middleware that sits between LLMs and users. It intercepts structurally impossible LLM outputs (hallucinations) BEFORE they reach the user, using Ibn Sina's (Avicenna's) classical syllogistic logic framework (Qiyas / Mantiq) formalized as directed graph algorithms.

**Two-Stage Architecture:**
- **Stage 1 (Probabilistic):** LLM acts only as semantic parser. T=0.0, JSON-only output, 80 tokens max. NEVER answers — only extracts logical form.
- **Stage 2 (Deterministic):** BFS graph validator on a NetworkX knowledge base. Zero probability. Mathematical guarantees.

**Four Epistemic States (Ibn Sina's Framework):**
- `YAQEEN` (Certainty) — BFS path confirmed → Override LLM with correct answer
- `WAHM` (Illusion) — LLM contradicts BFS → Intercept the hallucination
- `SHAKK` (Doubt) — Entity not in KB → Defer to LLM, no intervention
- `ZANN` (Probability) — Semantic match but no formal structure → LLM with confidence flag

**Current KB Size:**
- Taxonomy graph: 115 nodes, 136 IS-A edges
- Property graph: 115 entity-property associations
- Conditional graph: 49 IF-THEN rules

**Current Results (175 queries × 3 models = 525 evaluations):**
- Precision = 100%, Specificity = 100%, FP = 0 (across all models)
- LLaMA2-7B: 60.0% → 94.3% (+34.3 pp)
- Mistral-7B: 94.9% → 97.7% (+2.8 pp)
- LLaMA3.2-3B: 84.6% → 96.6% (+12.0 pp)
- TruthfulQA (790 OOD questions): 99.5% non-interference rate

---

## 2. WHY THE LOCAL IEEE CONFERENCE REJECTED THIS PAPER

This is the honest diagnostic. Every point below is a likely reviewer objection. Fix these = publication.

### Rejection Reason #1 — DATASET IS TOO SMALL AND PARTIALLY CIRCULAR (CRITICAL)

**Problem:**
```json
"by_source": {
  "original": 100,
  "kb_generated": 75
}
```
75 of 175 test queries were generated FROM THE SAME KB being tested. This is circular evaluation. A reviewer sees this and immediately distrusts all results. Even though Section V.A of the paper addresses this ("queries were authored independently of the KB after its construction"), reviewers are skeptical. The 43% KB-generated ratio is too high.

**Why it matters:** Circular evaluation is a fundamental methodological flaw. It inflates coverage rates and makes the 91% KB coverage look better than it is.

**Fix required:** Either (a) remove all 75 KB-generated queries and report results on only 100 original queries, OR (b) use a fully external dataset (FOLIO, RuleTaker, ClimateNLI) that was never used in KB construction.

### Rejection Reason #2 — KB SCALE IS TOY-LEVEL

**Problem:** 115 nodes, 136 edges is trivially small. ConceptNet has 8 million+ concepts. YAGO has 17 million entities. WordNet has 155,000 word senses. Reviewers know these numbers.

**Why it matters:** A 115-node KB raises the question: "Does this system scale? Is this a real system or a demonstration prototype?" At current scale, LogicGuard is a proof-of-concept, not a deployable system.

**Fix required:** Expand KB to minimum 1,000+ nodes using ConceptNet API integration. Show that system architecture holds at larger scale.

### Rejection Reason #3 — NO COMPARISON TO EXISTING SYSTEMS

**Problem:** The paper positions against RAG and CoT but never benchmarks against actual neuro-symbolic systems. Reviewers will ask: "How does this compare to Logic-LM? To LARK? To NLProlog?"

**Systems that need comparison:**
- Logic-LM (Pan et al., EMNLP 2023) — directly comparable, similar architecture
- CRITIC (Gou et al., ICLR 2024) — mentioned in related work but not benchmarked
- SelfCheckGPT (Manakul et al., EMNLP 2023) — mentioned but not benchmarked
- A simple ProLog/OWL reasoner baseline on same queries

**Fix required:** Implement at least 2 baseline comparisons on the same 175 queries.

### Rejection Reason #4 — STAGE 1 PARSER ACCURACY NOT REPORTED

**Problem:** The paper claims the constrained LLM parser is robust but never reports:
- What % of queries produce valid JSON on first attempt?
- How often does the regex fallback trigger?
- What is parser accuracy across models?

If Stage 1 fails silently 20% of the time, the whole system's reliability is undermined. Reviewers will notice this gap.

**Fix required:** Add a Stage 1 Robustness section reporting parse success rates per model.

### Rejection Reason #5 — NO STATISTICAL SIGNIFICANCE TESTING

**Problem:** Comparing 60.0% vs 94.3% accuracy without p-values or confidence intervals is not acceptable for a journal paper. Is +34.3pp statistically significant? What is the confidence interval?

**Fix required:** Run McNemar's test or paired bootstrap significance testing on all accuracy comparisons.

### Rejection Reason #6 — "DETERMINISTIC" IN TITLE IS MISLEADING

**Problem:** Stage 1 is a probabilistic LLM. The title says "Deterministic Hallucination Interception" — which is only partially true (Stage 2 is deterministic). Reviewers who catch this may penalize the paper for overclaiming.

**Fix required:** Either change title to "Neuro-Symbolic Middleware for Hallucination Interception..." OR add explicit clarification in abstract that "deterministic" refers to Stage 2 only.

### Rejection Reason #7 — NO LATENCY / COMPUTATIONAL COST ANALYSIS

**Problem:** Real systems need to show overhead cost. How much latency does LogicGuard add to a standard LLM call? If it doubles response time, enterprise deployment is impractical.

**Fix required:** Report Stage 1 latency, Stage 2 latency, and end-to-end overhead per query.

---

## 3. TARGET IEEE JOURNAL — RECOMMENDATION

### Primary Target: IEEE Transactions on Artificial Intelligence (IEEE TAI)
- **Why:** Covers AI systems, neuro-symbolic methods, LLM safety — perfect scope fit
- **Impact Factor:** ~4.5
- **Acceptance Rate:** ~25-30%
- **Page Limit:** 14 pages (double column)
- **Review Time:** ~3-4 months
- **Required:** Strong related work, proper baselines, statistical testing

### Secondary Target: IEEE Access
- **Why:** Broad scope, faster review (~6-8 weeks), good for interdisciplinary work
- **Impact Factor:** ~3.9
- **Acceptance Rate:** ~35%
- **Advantage:** Open access — more citations potential
- **Required:** Technically sound, reproducible, adequate scope

### After Major Strengthening: IEEE TNNLS (Transactions on Neural Networks and Learning Systems)
- **Why:** Higher impact, covers neural+symbolic hybrid systems
- **Impact Factor:** ~10.4
- **Acceptance Rate:** ~15%
- **Required:** Significantly larger experiments, stronger baselines

### Do NOT target yet:
- Top AI/ML conferences (NeurIPS, ICLR, ACL, AAAI) — scale too small, competition too high
- IEEE TPAMI — not the right scope
- Local/national IEEE conferences — you already know this result

---

## 4. COMPLETE IMPROVEMENT ROADMAP

### PHASE 1 — Critical Fixes (Must complete before ANY resubmission)
**Timeline: 2-3 weeks**

#### Task 1.1 — Fix Circular Evaluation
- Remove KB-generated queries from evaluation set
- Report results on 100 original queries only
- OR: Collect 200+ fresh queries from FOLIO / RuleTaker datasets
- Goal: Zero overlap between KB construction data and test data
- Files to modify: `step1_proofwriter_extractor.py`, `all_model_results.json`

#### Task 1.2 — Add Stage 1 Parser Robustness Analysis
- Instrument `step2_multi_model_runner.py` to log:
  - `parse_success` (valid JSON returned)
  - `regex_fallback_triggered` (fallback activated)
  - `parse_failure` (complete failure)
- Run on all 175 queries × 3 models
- Report in new Table VII: "Stage 1 Parser Reliability"
- Files to modify: `step2_multi_model_runner.py`, `step5_generate_paper_tables.py`

#### Task 1.3 — Add Statistical Significance Tests
- Implement McNemar's test for all before/after accuracy comparisons
- Report p-values and 95% confidence intervals
- Goal: p < 0.001 for LLaMA2-7B (+34.3pp should easily pass this)
- Add to `step3_metrics.py`

#### Task 1.4 — Add Latency Measurements
- Measure and report: Stage 1 time (ms), Stage 2 time (ms), total overhead (ms)
- Compare to baseline LLM call time
- Report as percentage overhead

---

### PHASE 2 — Scale & Baselines (Core journal differentiation)
**Timeline: 3-4 weeks**

#### Task 2.1 — Expand KB Using ConceptNet API
- ConceptNet API: `https://api.conceptnet.io/query?node=/c/en/dog&rel=/r/IsA`
- Target: 1,000+ taxonomy nodes (biology domain minimum)
- Keep existing 115 as "core" — add ConceptNet as "extended" layer
- Test: Does accuracy hold / improve at 1,000 nodes?
- Report new KB stats in Table I
- Create `step_conceptnet_expansion.py`

#### Task 2.2 — Implement Logic-LM Baseline
- Replicate Logic-LM approach on same 175 queries
- Logic-LM uses Z3 solver for symbolic reasoning
- Compare: Precision, Recall, F1, latency
- If full replication too complex: use published numbers and cite directly
- Add to Section VI as Table VII: "Comparison to Related Systems"

#### Task 2.3 — Expand to 500+ Query Dataset
- Add 325+ new queries from:
  - FOLIO dataset (natural language formal logic)
  - RuleTaker (ProofWriter successor, deeper reasoning chains)
  - Manually authored cross-domain negatives
- New split: 300 taxonomy, 150 categorical, 100 hypothetical
- Goal: Remove "toy dataset" criticism

---

### PHASE 3 — Paper Strengthening (Writing and structure)
**Timeline: 1-2 weeks**

#### Task 3.1 — Fix Title
**Current:** "Deterministic Hallucination Interception in LLMs Using Aristotelian-Avicennian Syllogistic Frameworks"
**Proposed:** "LogicGuard: A Neuro-Symbolic Middleware for Structural Hallucination Interception in Large Language Models via Avicennian Epistemic Reasoning"
**Why:** Removes misleading "deterministic" from title, adds "structural hallucination" (more precise term), keeps Avicenna angle (unique differentiator).

#### Task 3.2 — Strengthen Related Work Section
Add comparison table (Table II in revised paper):

| System | Approach | Formal Guarantees | Scope | Epistemic Grading |
|--------|----------|-------------------|-------|-------------------|
| RAG | Retrieval | No | Factual | No |
| CoT | Prompting | No | General | No |
| Logic-LM | Symbolic solvers | Partial | Math/Logic | No |
| SelfCheckGPT | Sampling variance | No | Factual | No |
| CRITIC | Tool interaction | No | General | No |
| **LogicGuard** | **BFS + KB** | **Yes (KB-scoped)** | **Syllogistic** | **Yes (4-state)** |

#### Task 3.3 — Add Ablation Study Section
Test each component independently:
- Stage 2 only (no Stage 1, direct entity extraction): What accuracy?
- Stage 1 with different temperatures (T=0.0 vs T=0.3 vs T=0.7)
- KB with taxonomy only (no properties, no conditionals)
- KB with all three graph types (current setup)
This shows that each design choice is justified.

#### Task 3.4 — Add Error Analysis Section
For the 10 FN cases (LLaMA2-7B +LG still wrong):
- What type of queries are they?
- Are they KB coverage gaps or parser failures?
- Example: "Do all reptiles lay eggs?" — is 'reptile' in KB? Does it have 'lay_eggs' property?
This section shows intellectual honesty and research depth.

#### Task 3.5 — Strengthen Conclusion
Add one paragraph explicitly connecting LogicGuard to EU AI Act 2024 compliance requirements. The Act mandates explainability and audit trails for high-risk AI systems. LogicGuard's deterministic Stage 2 provides exactly this — a formal, auditable reasoning trace. This is a strong practical motivation reviewers will appreciate.

---

### PHASE 4 — Optional Enhancements (For TNNLS level)
**Timeline: 4-6 weeks (only if targeting higher venue)**

#### Task 4.1 — Fine-tuned BERT Stage 1 Parser
- Replace LLM-based parser with fine-tuned BERT classifier
- Train on 5,000 labeled logical form classification examples
- Output: {taxonomic, categorical, hypothetical, non-logical}
- Benefit: Eliminates LLM dependency from Stage 1, makes entire pipeline deterministic
- This would allow the title "Fully Deterministic..." to be accurate

#### Task 4.2 — Legal Domain Extension
- Model statutes as conditional rules (IF condition THEN obligation)
- Example: IF contract_signed AND consideration_paid THEN enforceable
- Test on CUAD (Contract Understanding Atticus Dataset)
- This dramatically increases real-world applicability

#### Task 4.3 — Medical Domain Extension
- Symptom-disease mappings as categorical relations
- Drug interaction rules as conditionals
- Test on MedQA or BioASQ subset
- Makes paper relevant to high-stakes AI safety

#### Task 4.4 — Real-time API (FastAPI)
- Wrap LogicGuard as REST endpoint
- Input: natural language question
- Output: JSON with epistemic_state, graph_answer, latency, KB_coverage
- Deploy on Railway/Render
- This demonstrates the system is production-ready, not just academic

---

## 5. FILE STRUCTURE (Current + Planned)

```
LogicGuard/
│
├── CURRENT FILES
│   ├── step1_proofwriter_extractor.py   # ProofWriter → KB builder (currently commented out)
│   ├── step2_multi_model_runner.py      # Multi-model evaluation engine + LogicGuardValidator
│   ├── step3_metrics.py                 # P/R/F1, confusion matrices
│   ├── step4_truthfulqa_validation.py   # Out-of-domain generalization test
│   ├── step5_generate_paper_tables.py   # IEEE paper tables generator
│   ├── run_all.py                       # Master pipeline runner
│   ├── knowledge_base.json              # Base KB (hand-curated)
│   ├── knowledge_base_extended.json     # KB after ProofWriter extension (115 nodes)
│   └── all_model_results.json           # Combined results from Step 2
│
├── FILES TO ADD (Phase 1)
│   ├── step2b_parser_robustness.py      # Stage 1 parse success/failure analysis
│   ├── step3b_statistical_tests.py      # McNemar's test, confidence intervals
│   └── step3c_latency_analysis.py       # Latency measurement per stage
│
├── FILES TO ADD (Phase 2)
│   ├── step_conceptnet_expansion.py     # ConceptNet API → KB expansion
│   ├── step_baseline_comparison.py      # Logic-LM / CRITIC baseline
│   ├── queries_folio.json               # FOLIO dataset queries
│   └── knowledge_base_1000.json         # Expanded 1000+ node KB
│
└── FILES TO ADD (Phase 4)
    ├── api/                             # FastAPI wrapper
    │   ├── main.py
    │   ├── routers/validate.py
    │   └── schemas.py
    └── domain_extensions/
        ├── legal_kb.json
        └── medical_kb.json
```

---

## 6. PAPER REVISION CHECKLIST

Before submitting to IEEE TAI or IEEE Access, verify ALL of these:

### Methodology
- [ ] No circular evaluation — test queries fully independent from KB construction data
- [ ] Stage 1 parser success rate reported (Table VII)
- [ ] Statistical significance for all accuracy comparisons (p-values reported)
- [ ] Latency overhead measured and reported
- [ ] KB coverage boundary explicitly stated and justified

### Experiments
- [ ] Minimum 300 test queries (preferably 500+)
- [ ] At least 1 external dataset comparison (FOLIO or RuleTaker)
- [ ] At least 1 comparable system baseline (Logic-LM or equivalent)
- [ ] Ablation study (each KB component tested independently)
- [ ] Error analysis on FN cases

### Writing
- [ ] Title does not overclaim "deterministic" for the full pipeline
- [ ] Abstract matches actual scope of claims
- [ ] Related work includes comparison table (not just prose)
- [ ] Limitations section is honest about KB coverage constraints
- [ ] Conclusion connects to EU AI Act / regulatory compliance motivation
- [ ] All tables labeled correctly as IEEE format (TABLE I, TABLE II, etc.)
- [ ] Figures have proper captions and are referenced in text

### Technical
- [ ] All experiments reproducible with public code
- [ ] Random seed documented (seed=42, T=0.0 already done — keep this)
- [ ] Dataset available or reproducible from public sources
- [ ] KB available in repository

### Submission
- [ ] Paper length matches journal limit (IEEE TAI: 14 pages double-column)
- [ ] Abstract: 150-250 words
- [ ] IEEE copyright notice at bottom of first page
- [ ] Conflict of interest statement if required
- [ ] ORCID IDs for both authors

---

## 7. THE CORE INTELLECTUAL CONTRIBUTION (Keep This Always)

The unique selling point of LogicGuard — what no reviewer can dismiss — is the combination of:

1. **Classical Islamic/Aristotelian epistemology** (Ibn Sina's Qiyas) formalized computationally
2. **Strict neuro-symbolic separation** (probabilistic parsing + deterministic reasoning)
3. **4-state epistemic grading** instead of binary True/False
4. **Precision = 100% as an architectural guarantee** (not an empirical observation)
5. **SHAKK state = formal expression of "I don't know"** — this is the safety guarantee

This combination does not exist anywhere in prior literature. The reviewers who rejected the local IEEE conference paper rejected it on experimental scale, NOT on idea quality. The idea is sound and original.

The core claim that must survive all revisions:
> "A false positive requires BFS to fail on a correct KB. BFS graph reachability is provably correct. Therefore FP=0 is architectural, not empirical — it does not depend on model size, dataset size, or domain."

This claim is mathematically true and must be preserved and strengthened, not weakened.

---

## 8. WHAT CLAUDE SHOULD ALWAYS REMEMBER IN THIS PROJECT

### DO:
- Treat this as a research-grade engineering project, not a software product
- All code changes must maintain reproducibility (seed=42, T=0.0)
- Any new experiment must avoid KB-test data overlap
- When adding to KB, document the source (manual curation vs ConceptNet vs ProofWriter)
- Statistical tests required for any accuracy comparison
- Keep the Ibn Sina / Avicenna framing — it IS the novelty

### DO NOT:
- Do not add any heuristic-based components to Stage 2 — it must remain deterministic
- Do not expand KB and re-run tests without documenting the expansion source
- Do not change the 4-state epistemic model (Yaqeen/Wahm/Shakk/Zann)
- Do not remove the TruthfulQA out-of-domain test from results
- Do not claim the full pipeline is deterministic — only Stage 2 is

### Key Files to Never Break:
- `step2_multi_model_runner.py` — contains LogicGuardValidator class (core logic)
- `knowledge_base_extended.json` — the validated KB (backup before any changes)
- `all_model_results.json` — the raw experimental results (source of truth)

---

## 9. QUICK REFERENCE — KEY NUMBERS

| Metric | Value | Source |
|--------|-------|--------|
| Test queries | 175 | step2 |
| Total evaluations | 525 (175 × 3 models) | step2 |
| KB taxonomy nodes | 115 | knowledge_base_extended.json |
| KB IS-A edges | 136 | knowledge_base_extended.json |
| KB property associations | 115 | knowledge_base_extended.json |
| KB conditional rules | 49 | knowledge_base_extended.json |
| TruthfulQA questions | 790 | step4 |
| Precision (all models) | 100% | step3 |
| False Positives (all models) | 0 | step3 |
| Best improvement | +34.3pp (LLaMA2-7B) | step3 |
| OOD non-interference | 99.5% | step4 |
| Runtime (full pipeline) | ~44 minutes | README |
| Hardware | CPU-only | README |

---

## 10. SUBMISSION HISTORY

| Date | Venue | Type | Outcome | Notes |
|------|-------|------|---------|-------|
| Early 2026 | Local IEEE Conference (Pakistan) | Conference paper | REJECTED | Reason not officially stated. Likely: small dataset, no baselines, circular evaluation |

**Next target:** IEEE Access or IEEE Transactions on Artificial Intelligence (after Phase 1+2 improvements)

---

*This file is the master reference for all future Claude Code sessions on the LogicGuard project.*
*Last updated: 2026-03-28*
*Maintained by: Hamza Naseem*
