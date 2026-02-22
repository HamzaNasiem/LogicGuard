# multi_model_runner.py

"""
STEP 2: Multi-Model Runner
===========================
Runs ALL THREE models (Llama2 7B, Mistral 7B, Llama3.2 3B)
+ LogicGuard on top of each — on the extended query set.

Saves:
  results_llama2.json
  results_mistral.json
  results_llama32.json
  results_logicguard.json  ← LogicGuard on Llama3.2 (primary)
  all_model_results.json   ← combined for Step 3

Usage:
    python step2_multi_model_runner.py \
        --queries extended_queries.json \
        --kb knowledge_base_extended.json
"""

import json
import time
import sys
import os
import argparse
import re
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# ── Try imports ───────────────────────────────────────────────────────
try:
    import ollama
except ImportError:
    print("ERROR: ollama not installed. Run: pip install ollama")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────
# MODELS TO EVALUATE
# ─────────────────────────────────────────────────────────────────────

MODELS = {
    'llama2_7b':    'llama2',        # llama2:7b or llama2:latest
    'mistral_7b':   'mistral',       # mistral:7b or mistral:latest
    'llama32_3b':   'llama3.2:3b',   # llama3.2:3b
}

# ─────────────────────────────────────────────────────────────────────
# LLM CALLER
# ─────────────────────────────────────────────────────────────────────

def get_llm_answer(question: str, model: str, retries: int = 3) -> str:
    """
    Ask LLM a yes/no question. Returns 'yes', 'no', or '[error]'.
    We use a direct prompt that forces YES/NO answer for logical questions.
    """
    prompt = (
        f"Answer this question with YES or NO only. "
        f"Do not explain. Just say YES or NO.\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    for attempt in range(retries):
        try:
            resp = ollama.chat(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.0, 'seed': 42, 'num_predict': 10}
            )
            raw = resp['message']['content'].strip().lower()
            # Parse yes/no from response
            if raw.startswith('yes'):
                return 'yes'
            elif raw.startswith('no'):
                return 'no'
            # Fallback: search in response
            if 'yes' in raw[:20]:
                return 'yes'
            elif 'no' in raw[:20]:
                return 'no'
            return raw[:50]  # return raw if can't parse
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)
            else:
                return '[llm_error]'
    return '[llm_error]'


def parse_llm_yn(raw_answer: str) -> Optional[bool]:
    """
    Parse LLM raw answer to True (yes) / False (no) / None (unclear).
    """
    if not raw_answer or raw_answer == '[llm_error]':
        return None
    t = raw_answer.lower().strip()
    # Strong yes signals
    if t.startswith('yes') or t in ('true', '1', 'correct', 'right'):
        return True
    # Strong no signals
    if t.startswith('no') or t in ('false', '0', 'incorrect', 'wrong'):
        return False
    # Sentence-level
    neg_words = ['not ', "isn't", "aren't", "doesn't", "don't",
                 "cannot", "never", "false", "wrong", "incorrect", "no,"]
    pos_words = ['yes,', 'yes.', 'correct', 'true', 'indeed',
                 'all are', 'always', 'absolutely', 'certainly']
    has_neg = any(w in t for w in neg_words)
    has_pos = any(w in t for w in pos_words)
    if has_pos and not has_neg:
        return True
    if has_neg and not has_pos:
        return False
    return None  # unclear


# ─────────────────────────────────────────────────────────────────────
# LOGICGUARD VALIDATOR (inline, reads from KB file)
# ─────────────────────────────────────────────────────────────────────

class LogicGuardValidator:
    """
    Self-contained LogicGuard validator.
    Reads knowledge_base_extended.json and validates via BFS graph traversal.
    """

    def __init__(self, kb_path: str):
        with open(kb_path, 'r') as f:
            self.kb = json.load(f)
        self._build_graph()

    def _build_graph(self):
        """Build adjacency dict for BFS."""
        self.graph = {}  # child → set of ancestors
        # Initialize
        for node in self.kb['taxonomies']:
            self.graph[node] = set(self.kb['taxonomies'][node])
        # Transitive closure via BFS
        changed = True
        while changed:
            changed = False
            for node in list(self.graph.keys()):
                for parent in list(self.graph[node]):
                    if parent in self.graph:
                        new_ancestors = self.graph[parent] - self.graph[node]
                        if new_ancestors:
                            self.graph[node].update(new_ancestors)
                            changed = True

    def normalize(self, word: str) -> str:
        """Normalize plural/singular with extended irregular support."""
        word = word.lower().strip()
        # Full irregular map including math/science terms
        irregulars = {
            'mice': 'mouse', 'geese': 'goose', 'feet': 'foot',
            'teeth': 'tooth', 'fish': 'fish', 'sheep': 'sheep',
            'deer': 'deer', 'children': 'child', 'people': 'person',
            # Math/geometry plurals
            'rhombuses': 'rhombus', 'buses': 'bus', 'foxes': 'fox',
            'boxes': 'box', 'taxes': 'tax', 'axes': 'axis',
            'vertices': 'vertex', 'indices': 'index', 'matrices': 'matrix',
            'appendices': 'appendix', 'radii': 'radius',
        }
        if word in irregulars:
            return irregulars[word]
        if word.endswith('ies') and len(word) > 3:
            return word[:-3] + 'y'
        if word.endswith('ves') and len(word) > 3:
            return word[:-3] + 'f'
        # -ses, -xes, -ches, -shes → strip -es (e.g. buses→bus, foxes→fox)
        if word.endswith('ses') and len(word) > 3:
            return word[:-2]
        if word.endswith('xes') and len(word) > 3:
            return word[:-2]
        if word.endswith('ches') and len(word) > 4:
            return word[:-2]
        if word.endswith('shes') and len(word) > 4:
            return word[:-2]
        if word.endswith('s') and not word.endswith(('ss', 'us', 'is')):
            return word[:-1]
        return word

    def is_ancestor(self, child: str, ancestor: str) -> bool:
        """BFS check: is ancestor reachable from child?"""
        c = self.normalize(child)
        a = self.normalize(ancestor)
        if c == a:
            return True
        return a in self.graph.get(c, set())

    def has_property(self, entity: str, prop: str) -> bool:
        """Check if entity or any ancestor has property."""
        e = self.normalize(entity)
        p = prop.lower().strip().replace(' ', '_')
        # Check entity directly
        if e in self.kb['properties']:
            if p in self.kb['properties'][e] or p.replace('_', ' ') in self.kb['properties'][e]:
                return True
        # Check via taxonomy ancestors
        ancestors = self.graph.get(e, set())
        for ancestor in ancestors:
            if ancestor in self.kb['properties']:
                props = self.kb['properties'][ancestor]
                if p in props or p.replace('_', ' ') in props:
                    return True
        return False

    def check_conditional(self, condition: str, consequence: str) -> tuple:
        """
        Check conditional KB.
        Returns (result: bool, covered: bool)
        covered=True ONLY if the condition node exists in KB.
        """
        cond = condition.lower().strip()
        cons = consequence.lower().strip()

        # Build consequence variants (handle "does it X", "is there X", etc.)
        cons_variants = {
            cons,
            cons.replace(' ', '_'),
            cons.split()[-1] if cons else '',
            cons.split()[0] if cons else '',
            # Drop helper verbs
            cons.replace('do we need an ', '').replace('do we need ', '').strip(),
            cons.replace('does it ', '').replace('is it ', '').strip(),
            cons.replace('there ', '').strip(),
            # Drop articles/copulas
            '_'.join(w for w in cons.split() if w not in ('is', 'are', 'the', 'a', 'an', 'do', 'does', 'we')),
        }
        # NEW: plural ↔ singular variants (e.g. "volume_decreases" ↔ "volume_decrease")
        plural_variants = set()
        for v in list(cons_variants):
            if v and v.endswith('s') and not v.endswith(('ss', 'us', 'is')):
                plural_variants.add(v[:-1])   # decreases → decrease
            elif v:
                plural_variants.add(v + 's')  # decrease → decreases
        cons_variants.update(plural_variants)
        cons_variants.discard('')

        # Build condition variants
        cond_variants = {
            cond,
            cond.replace(' ', '_'),
            re.sub(r'\b(is|are|the|it|there)\b', '', cond).strip(),
            cond.replace('it is ', '').strip(),
            cond.replace('there is ', '').strip(),
            cond.replace('there are ', '').strip(),
            cond.replace('if ', '').strip(),
        }
        cond_variants.discard('')

        for cv in cond_variants:
            if cv in self.kb['conditionals']:
                for consv in cons_variants:
                    if consv in self.kb['conditionals'][cv]:
                        return True, True   # found
                # Condition found but consequence not matching → WAHM
                return False, True

        # Condition not in KB at all → SHAKK (genuinely unknown)
        return False, False

    # ── Pattern extractors ────────────────────────────────────────────

    # Synonym map: multi-word predicates → KB node names
    _TAX_SYNONYMS = {
        'living things': 'living_thing',
        'living thing':  'living_thing',
        'living':        'living_thing',
        'warm blooded':  'warm_blooded',
        'cold blooded':  'cold_blooded',
    }

    def _normalize_predicate(self, pred: str) -> str:
        """Normalize a (possibly multi-word) predicate to KB node form."""
        p = pred.strip().lower()
        if p in self._TAX_SYNONYMS:
            return self._TAX_SYNONYMS[p]
        return self.normalize(p)

    def _extract_taxonomic(self, q: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract (subject, predicate) from taxonomic question."""
        t = q.lower().strip().rstrip('?')
        # "Are all X Y?" — capture multi-word predicate
        m = re.match(r'are all (\w+)\s+(?:a\s+|an\s+)?([\w ]+)', t)
        if m:
            pred = self._normalize_predicate(m.group(2).strip())
            return m.group(1), pred
        # "Is X a Y?"
        m = re.match(r'is\s+(\w+)\s+a[n]?\s+([\w ]+)', t)
        if m:
            pred = self._normalize_predicate(m.group(2).strip())
            return m.group(1), pred
        return None, None

    def _extract_categorical(self, q: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract (entity, property) from categorical question."""
        t = q.lower().strip().rstrip('?')
        m = re.match(r'do all (\w+)\s+(?:have|need)\s+(?:a\s+|an\s+|the\s+)?(.+)', t)
        if m:
            return m.group(1), m.group(2).strip()
        return None, None

    def _extract_hypothetical(self, q: str) -> Tuple[Optional[str], Optional[str], bool]:
        """Extract (condition, consequence, is_negation) from hypothetical question."""
        t = q.lower().strip().rstrip('?')
        if 'if' not in t:
            return None, None, False
        after_if = t.split('if', 1)[1].strip()
        if 'then' in after_if:
            parts = after_if.split('then', 1)
        elif ',' in after_if:
            parts = after_if.split(',', 1)
        else:
            return None, None, False
        cond = parts[0].strip()
        cons = parts[1].strip()
        # Detect negation in consequence
        is_neg = False
        for prefix in ['is there', 'is the', 'does it', 'is it', 'is', 'does', 'do we need']:
            if cons.startswith(prefix + ' '):
                rem = cons[len(prefix):].strip()
                if rem.startswith('no ') or rem.startswith('not ') or ' no ' in rem or ' not ' in rem:
                    is_neg = True
                    rem = (rem.replace('no ', '', 1)
                              .replace('not ', '', 1)
                              .replace(' no ', ' ', 1)
                              .replace(' not ', ' ', 1)
                              .strip())
                cons = rem
                break
        # Also catch "does X not Y" / "is X no Y" mid-string
        if not is_neg and (' no ' in cons or ' not ' in cons):
            is_neg = True
            cons = cons.replace(' no ', ' ', 1).replace(' not ', ' ', 1).strip()
        return cond, cons, is_neg

    def validate(self, question: str, qtype: str) -> Dict:
        """
        Validate a question. Returns:
        {
            'graph_answer': True/False/None,
            'epistemic_state': 'YAQEEN'/'WAHM'/'UNKNOWN',
            'proof': str,
            'covered': bool
        }
        covered=True  → KB has an answer (YAQEEN or WAHM)
        covered=False → KB cannot determine (SHAKK / genuine uncertainty)
        """
        q = question.strip()

        if qtype == 'taxonomic':
            subj, pred = self._extract_taxonomic(q)
            if not subj or not pred:
                return {'graph_answer': None, 'epistemic_state': 'UNKNOWN',
                        'proof': f'Parse failed: {q}', 'covered': False}
            subj_norm = self.normalize(subj)
            # Only mark covered if subject is in KB graph
            if subj_norm not in self.graph and subj not in self.graph:
                return {'graph_answer': None, 'epistemic_state': 'UNKNOWN',
                        'proof': f'Entity not in KB: {subj_norm}', 'covered': False}
            result = self.is_ancestor(subj, pred)
            return {
                'graph_answer': result,
                'epistemic_state': 'YAQEEN' if result else 'WAHM',
                'proof': f'BFS: {subj_norm} → {pred} = {result}',
                'covered': True
            }

        elif qtype == 'categorical':
            entity, prop = self._extract_categorical(q)
            if not entity or not prop:
                return {'graph_answer': None, 'epistemic_state': 'UNKNOWN',
                        'proof': f'Parse failed: {q}', 'covered': False}
            e_norm = self.normalize(entity)
            # Only mark covered if entity or its class is known
            has_direct = e_norm in self.kb.get('properties', {})
            has_ancestor = any(a in self.kb.get('properties', {}) for a in self.graph.get(e_norm, set()))
            if not has_direct and not has_ancestor and e_norm not in self.graph:
                return {'graph_answer': None, 'epistemic_state': 'UNKNOWN',
                        'proof': f'Entity not in KB: {e_norm}', 'covered': False}
            result = self.has_property(entity, prop)
            return {
                'graph_answer': result,
                'epistemic_state': 'YAQEEN' if result else 'WAHM',
                'proof': f'Property: {e_norm}.{prop} = {result}',
                'covered': True
            }

        elif qtype == 'hypothetical':
            cond, cons, is_neg = self._extract_hypothetical(q)
            if not cond or not cons:
                return {'graph_answer': None, 'epistemic_state': 'UNKNOWN',
                        'proof': f'Parse failed: {q}', 'covered': False}
            result, covered = self.check_conditional(cond, cons)
            if not covered:
                return {'graph_answer': None, 'epistemic_state': 'UNKNOWN',
                        'proof': f'Condition not in KB: {cond}', 'covered': False}
            # Apply negation
            if is_neg:
                result = not result
            return {
                'graph_answer': result,
                'epistemic_state': 'YAQEEN' if result else 'WAHM',
                'proof': f'Modus Ponens: {cond} → {cons} (neg={is_neg}) = {result}',
                'covered': True
            }

        return {'graph_answer': None, 'epistemic_state': 'UNKNOWN',
                'proof': 'Unknown type', 'covered': False}


# ─────────────────────────────────────────────────────────────────────
# EVALUATE ONE MODEL
# ─────────────────────────────────────────────────────────────────────

def evaluate_model(
    model_key: str,
    model_name: str,
    queries: List[Dict],
    validator: LogicGuardValidator,
    use_logicguard: bool = False,
    delay: float = 0.3
) -> List[Dict]:
    """
    Run model on all queries and return results list.
    If use_logicguard=True, override LLM answer with graph validator when possible.
    """
    results = []
    n = len(queries)
    correct = 0

    print(f"\n{'─' * 65}")
    print(f"  Model: {model_name}{'  [+ LogicGuard]' if use_logicguard else ''}")
    print(f"  Queries: {n}")
    print(f"{'─' * 65}")

    for i, query in enumerate(queries):
        q         = query['question']
        qtype     = query['type']
        gt        = query['ground_truth']  # True/False
        source    = query.get('source', 'unknown')

        # 1. Get LLM answer
        llm_raw    = get_llm_answer(q, model_name)
        llm_parsed = parse_llm_yn(llm_raw)

        # 2. Graph validation
        graph_result = validator.validate(q, qtype)
        graph_answer = graph_result['graph_answer']
        covered      = graph_result['covered']

        # 3. Final answer
        if use_logicguard and covered and graph_answer is not None:
            # LogicGuard overrides LLM
            final_answer       = graph_answer
            method             = 'logicguard_override' if llm_parsed != graph_answer else 'logicguard_agree'
            hallucination_caught = (llm_parsed != graph_answer) and (graph_answer == gt)
        else:
            final_answer       = llm_parsed
            method             = 'llm_only'
            hallucination_caught = False

        # 4. Correctness
        if final_answer is None:
            is_correct = False  # unclear → count as wrong
        else:
            is_correct = (final_answer == gt)

        if is_correct:
            correct += 1

        rec = {
            'question':           q,
            'type':               qtype,
            'source':             source,
            'ground_truth':       gt,
            'llm_raw':            llm_raw,
            'llm_parsed':         llm_parsed,
            'graph_answer':       graph_answer,
            'graph_covered':      covered,
            'final_answer':       final_answer,
            'is_correct':         is_correct,
            'method':             method,
            'hallucination_caught': hallucination_caught,
            'epistemic_state':    graph_result['epistemic_state'],
            'proof':              graph_result['proof'],
            'model':              model_key,
            'logicguard':         use_logicguard,
        }
        results.append(rec)

        # Progress
        icon = '✓' if is_correct else '✗'
        ep   = graph_result['epistemic_state']
        print(f"  [{i+1:03}/{n}] {icon} {ep:8} | {q[:55]}")

        time.sleep(delay)

    acc = correct / n * 100 if n > 0 else 0
    print(f"\n  ► Accuracy: {acc:.1f}%  ({correct}/{n})")
    return results


def compute_hallucination_rate(results: List[Dict], use_logicguard: bool) -> Tuple[int, int]:
    """Count how many LLM hallucinations were caught by LogicGuard."""
    if not use_logicguard:
        return 0, 0
    # Hallucinations = LLM was wrong on logical question, LogicGuard fixed it
    caught = sum(1 for r in results if r.get('hallucination_caught', False))
    # Total LLM errors on covered logical questions
    llm_errors = sum(1 for r in results
                     if r['graph_covered']
                     and r['llm_parsed'] is not None
                     and r['llm_parsed'] != r['ground_truth'])
    return caught, llm_errors


def compute_summary(results: List[Dict], model_key: str, use_logicguard: bool) -> Dict:
    """Compute accuracy stats from results list."""
    total = len(results)
    if total == 0:
        return {}

    # By type
    by_type = defaultdict(lambda: {'total': 0, 'correct': 0})
    for r in results:
        t = r['type']
        by_type[t]['total'] += 1
        if r['is_correct']:
            by_type[t]['correct'] += 1

    # Overall
    overall_correct = sum(1 for r in results if r['is_correct'])
    overall_acc = overall_correct / total * 100

    # Hallucination catching
    caught, llm_errors = compute_hallucination_rate(results, use_logicguard)

    summary = {
        'model':         model_key,
        'logicguard':    use_logicguard,
        'total':         total,
        'correct':       overall_correct,
        'accuracy':      round(overall_acc, 1),
        'by_type':       {t: {'total': v['total'],
                              'correct': v['correct'],
                              'accuracy': round(v['correct']/v['total']*100, 1) if v['total'] > 0 else 0}
                          for t, v in by_type.items()},
        'hallucinations_caught': caught,
        'llm_errors_on_logical': llm_errors,
    }
    return summary


# ─────────────────────────────────────────────────────────────────────
# CHECK OLLAMA AVAILABILITY
# ─────────────────────────────────────────────────────────────────────

def check_model_available(model_name: str) -> bool:
    """Check if model is available in Ollama."""
    try:
        models = ollama.list()
        available = [m['name'].split(':')[0] for m in models.get('models', [])]
        # Check base name
        base = model_name.split(':')[0]
        return base in available or model_name in [m['name'] for m in models.get('models', [])]
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queries', type=str, default='extended_queries.json')
    parser.add_argument('--kb',      type=str, default='knowledge_base_extended.json')
    parser.add_argument('--output',  type=str, default='all_model_results.json')
    parser.add_argument('--delay',   type=float, default=0.3,
                        help='Seconds delay between LLM calls (default 0.3)')
    parser.add_argument('--models',  type=str, default='all',
                        help='Comma-separated model keys or "all"')
    args = parser.parse_args()

    print("=" * 65)
    print("  LogicGuard — Step 2: Multi-Model Evaluation")
    print("=" * 65)

    # ── Load queries ─────────────────────────────────────────────────
    print(f"\nLoading queries from: {args.queries}")
    with open(args.queries, 'r') as f:
        data = json.load(f)
    queries = data['queries']
    print(f"  Total queries: {len(queries)}")

    # Break down by type
    for qtype in ['taxonomic', 'categorical', 'hypothetical']:
        n = sum(1 for q in queries if q['type'] == qtype)
        print(f"  {qtype:15}: {n}")

    # ── Load KB + validator ──────────────────────────────────────────
    print(f"\nLoading KB from: {args.kb}")
    if not os.path.exists(args.kb):
        print(f"  ⚠️  KB file not found. Run step1 first.")
        sys.exit(1)
    validator = LogicGuardValidator(args.kb)
    print(f"  KB nodes: {len(validator.graph)}")

    # ── Check models ─────────────────────────────────────────────────
    print(f"\nChecking Ollama models...")
    models_to_run = {}
    if args.models == 'all':
        requested = list(MODELS.keys())
    else:
        requested = [m.strip() for m in args.models.split(',')]

    for key in requested:
        if key not in MODELS:
            print(f"  ⚠️  Unknown model key: {key}")
            continue
        model_name = MODELS[key]
        if check_model_available(model_name):
            print(f"  ✅ {key:15} ({model_name}) — available")
            models_to_run[key] = model_name
        else:
            print(f"  ❌ {key:15} ({model_name}) — NOT FOUND (run: ollama pull {model_name})")

    if not models_to_run:
        print("\nERROR: No models available. Pull models first.")
        sys.exit(1)

    # ── Run evaluation ────────────────────────────────────────────────
    all_results = {}
    all_summaries = {}

    for model_key, model_name in models_to_run.items():
        print(f"\n{'=' * 65}")
        print(f"  EVALUATING: {model_key} ({model_name})")
        print(f"{'=' * 65}")

        # Run WITHOUT LogicGuard (pure LLM baseline)
        print(f"\n  Phase A: Pure LLM (no LogicGuard)")
        results_baseline = evaluate_model(
            model_key=f'{model_key}_baseline',
            model_name=model_name,
            queries=queries,
            validator=validator,
            use_logicguard=False,
            delay=args.delay
        )
        summary_baseline = compute_summary(results_baseline, f'{model_key}_baseline', False)
        all_results[f'{model_key}_baseline'] = results_baseline
        all_summaries[f'{model_key}_baseline'] = summary_baseline

        # Run WITH LogicGuard
        print(f"\n  Phase B: LogicGuard (graph override)")
        results_lg = evaluate_model(
            model_key=f'{model_key}_logicguard',
            model_name=model_name,
            queries=queries,
            validator=validator,
            use_logicguard=True,
            delay=args.delay
        )
        summary_lg = compute_summary(results_lg, f'{model_key}_logicguard', True)
        all_results[f'{model_key}_logicguard'] = results_lg
        all_summaries[f'{model_key}_logicguard'] = summary_lg

        # Save per-model results
        per_model_file = f'results_{model_key}.json'
        with open(per_model_file, 'w') as f:
            json.dump({
                'model': model_key,
                'model_name': model_name,
                'baseline': {'results': results_baseline, 'summary': summary_baseline},
                'logicguard': {'results': results_lg, 'summary': summary_lg},
            }, f, indent=2, default=str)
        print(f"\n  💾 Saved: {per_model_file}")

    # ── Print comparison table ────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("  CROSS-MODEL COMPARISON")
    print(f"{'=' * 65}")
    print(f"  {'Model':<22} {'Taxonomic':>10} {'Categorical':>12} {'Hypothetical':>13} {'Overall':>9} {'Hall.Caught':>12}")
    print(f"  {'─'*22} {'─'*10} {'─'*12} {'─'*13} {'─'*9} {'─'*12}")

    for run_key, summary in all_summaries.items():
        bt = summary.get('by_type', {})
        tax = bt.get('taxonomic', {}).get('accuracy', 0)
        cat = bt.get('categorical', {}).get('accuracy', 0)
        hyp = bt.get('hypothetical', {}).get('accuracy', 0)
        overall = summary.get('accuracy', 0)
        caught = summary.get('hallucinations_caught', 0)
        lg_tag = '[+LG]' if summary.get('logicguard') else '     '
        print(f"  {run_key:<22} {tax:>9.1f}% {cat:>11.1f}% {hyp:>12.1f}% {overall:>8.1f}% {caught:>10}")

    # ── Save combined ─────────────────────────────────────────────────
    combined = {
        'metadata': data.get('metadata', {}),
        'summaries': all_summaries,
        'results': {k: v for k, v in all_results.items()},
    }
    with open(args.output, 'w') as f:
        json.dump(combined, f, indent=2, default=str)

    print(f"\n  💾 Combined results saved: {args.output}")
    print(f"\n{'=' * 65}")
    print(f"  STEP 2 COMPLETE")
    print(f"  Next: python step3_metrics.py")
    print(f"{'=' * 65}\n")


if __name__ == '__main__':
    main()