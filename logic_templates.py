# logic_templates.py

"""
Ibn Sina's Logic Templates — NetworkX Graph Edition
=====================================================
Three Qiyas templates using the NetworkX-based KnowledgeGraph:
  1. Qiyas al-Haml (Taxonomic)    : "Are all X Y?"
  2. Qiyas al-Istithna (Hypothetical): "If X, then Y?"
  3. Categorical                  : "Do all X have/give/lay Y?"

Key upgrade from v1: Uses directed graph traversal (BFS) via NetworkX
instead of a hardcoded JSON dict. This enables full transitive inference.
"""

import re
from typing import Dict, List, Optional, Tuple
from knowledge_graph import KnowledgeGraph


IRREGULAR = {
    'mice': 'mouse', 'geese': 'goose', 'teeth': 'tooth',
    'feet': 'foot', 'men': 'man', 'women': 'woman',
    'children': 'child', 'people': 'person',
    'buses': 'bus', 'viruses': 'virus', 'campuses': 'campus',
    'oxen': 'ox', 'cacti': 'cactus', 'fungi': 'fungus',
    'fish': 'fish', 'sheep': 'sheep', 'deer': 'deer', 'moose': 'moose',
}

def normalize_plural(word: str) -> str:
    word = word.strip().lower()
    if len(word) <= 2:
        return word
    if word in IRREGULAR:
        return IRREGULAR[word]
    if word.endswith('ies') and len(word) > 3:
        return word[:-3] + 'y'
    if word.endswith(('shes', 'ches', 'xes', 'sses')):
        return word[:-2]
    if word.endswith('ves') and len(word) > 3:
        return word[:-3] + 'f'
    if word.endswith('s') and not word.endswith(('ss', 'us')):
        return word[:-1]
    return word


class TaxonomicTemplate:
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg

    def matches(self, question: str) -> bool:
        return 'are all' in question.lower()

    def extract_entities(self, question: str) -> Tuple[Optional[str], Optional[str]]:
        q = question.lower().replace('?', '').strip()
        if 'are all' not in q:
            return None, None
        after = q.split('are all', 1)[1].strip()
        for filler in [' a ', ' an ', ' the ']:
            after = after.replace(filler, ' ')
        parts = after.split()
        if len(parts) >= 2:
            return normalize_plural(parts[0]), normalize_plural(parts[1])
        return None, None

    def verify(self, question: str, llm_answer: str = '') -> Dict:
        subject, predicate = self.extract_entities(question)
        if not subject or not predicate:
            return {'logically_valid': None, 'certainty': 0,
                    'proof': f'Could not extract entities from: "{question}"',
                    'epistemic_state': 'UNKNOWN', 'method': 'Parse Failed'}

        is_subset, path = self.kg.is_subset(subject, predicate)
        if is_subset:
            return {'logically_valid': True, 'certainty': 100,
                    'proof': f'BFS proof: {" -> ".join(path)}',
                    'epistemic_state': 'YAQEEN',
                    'method': 'Qiyas al-Haml (Taxonomic)'}

        return {'logically_valid': False, 'certainty': 0,
                'proof': f'No BFS path: {subject} not-subset-of {predicate}',
                'epistemic_state': 'WAHM',
                'method': 'Qiyas al-Haml (Failed)'}


class HypotheticalTemplate:
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg

    def matches(self, question: str) -> bool:
        q = question.lower()
        return q.startswith('if ') or (' if ' in q)

    def extract_conditional(self, question: str) -> Tuple[Optional[str], Optional[str]]:
        q = question.lower().replace('?', '').strip()
        if 'if' not in q:
            return None, None
        after_if = q.split('if', 1)[1].strip()
        if 'then' in after_if:
            parts = after_if.split('then', 1)
        elif ',' in after_if:
            parts = after_if.split(',', 1)
        else:
            return None, None
        if len(parts) < 2:
            return None, None
        condition   = parts[0].strip()
        consequence = parts[1].strip()
        for filler in ["it is ", "it's ", "there is ", "there are ",
                       "there ", "you are ", "we are ", "they are "]:
            condition = condition.replace(filler, '')
        condition = condition.strip()
        for prefix in ["a ", "an ", "the "]:
            if condition.startswith(prefix):
                condition = condition[len(prefix):]
                break
        for pat in ["is there ", "is the ", "does it ", "are they ",
                    "are you ", "will it ", "is it ", "is ", "does ", "are "]:
            if consequence.startswith(pat):
                consequence = consequence[len(pat):]
                break
        return condition.strip(), consequence.strip()

    def verify(self, question: str, llm_answer: str = '') -> Dict:
        condition, consequence = self.extract_conditional(question)
        if not condition or not consequence:
            return {'logically_valid': None, 'certainty': 0,
                    'proof': f'Could not extract conditional from: "{question}"',
                    'epistemic_state': 'UNKNOWN', 'method': 'Parse Failed'}
        valid, proof = self.kg.check_conditional(condition, consequence)
        if valid:
            return {'logically_valid': True, 'certainty': 100,
                    'proof': proof,
                    'epistemic_state': 'YAQEEN',
                    'method': 'Qiyas al-Istithna (Hypothetical)'}
        return {'logically_valid': False, 'certainty': 0,
                'proof': f'Conditional unverified: {condition} -> {consequence}',
                'epistemic_state': 'WAHM',
                'method': 'Qiyas al-Istithna (Failed)'}


class CategoricalTemplate:
    VERBS = ['have', 'need', 'give', 'lay', 'produce', 'die',
             'grow', 'possess', 'contain', 'carry']

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg

    def matches(self, question: str) -> bool:
        q = question.lower()
        return 'do all' in q and any(v in q for v in self.VERBS)

    def extract_property_claim(self, question: str) -> Tuple[Optional[str], Optional[str]]:
        q = question.lower().replace('?', '').strip()
        if 'do all' not in q:
            return None, None
        after = q.split('do all', 1)[1]
        splitter = None
        split_pos = len(after) + 1
        for v in self.VERBS:
            idx = after.find(v)
            if idx != -1 and idx < split_pos:
                split_pos = idx
                splitter = v
        if not splitter:
            return None, None
        parts = after.split(splitter, 1)
        if len(parts) < 2:
            return None, None
        entity = normalize_plural(parts[0].strip())
        prop   = parts[1].strip()
        if not prop or prop in ['', '.']:
            prop = splitter
        else:
            for filler in ['a ', 'an ', 'the ']:
                if prop.startswith(filler):
                    prop = prop[len(filler):]
        return entity, prop.strip()

    def verify(self, question: str, llm_answer: str = '') -> Dict:
        entity, prop = self.extract_property_claim(question)
        if not entity or not prop:
            return {'logically_valid': None, 'certainty': 0,
                    'proof': f'Could not extract property claim from: "{question}"',
                    'epistemic_state': 'UNKNOWN', 'method': 'Parse Failed'}
        has_prop, proof = self.kg.has_property(entity, prop)
        if has_prop:
            return {'logically_valid': True, 'certainty': 100,
                    'proof': f'Graph inheritance: {proof}',
                    'epistemic_state': 'YAQEEN',
                    'method': 'Categorical Reasoning (Graph)'}
        return {'logically_valid': False, 'certainty': 0,
                'proof': f'Not in property graph: {entity} has no {prop}',
                'epistemic_state': 'WAHM',
                'method': 'Categorical Reasoning (Failed)'}