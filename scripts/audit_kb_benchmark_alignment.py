import json
import sys
sys.path.insert(0, 'src')
from avicennaguard.kb.loader import KnowledgeBase
from avicennaguard.kb.validator import BFSValidator
from avicennaguard.parsers.deberta_parser import DebertaParser
from avicennaguard.core.epistemic_states import EpistemicState

kb = KnowledgeBase('data/knowledge_bases/knowledge_base_extended.json')
v = BFSValidator(kb)
parser = DebertaParser()

with open('data/benchmarks/avicenna_benchmark_500.json', 'r', encoding='utf-8') as f:
    bench = json.load(f)

mismatches = []
covered_count = 0

for idx, item in enumerate(bench):
    q = item['question']
    gt = item['ground_truth']
    gt_bool = (gt.lower() in ('yes', 'true', '1')) if isinstance(gt, str) and gt.lower() in ('yes', 'no', 'true', 'false') else gt if isinstance(gt, bool) else None
    
    parsed = parser.parse(q)
    q_type = parsed.get('type', item.get('query_type'))
    s = parsed.get('subject', '')
    p = parsed.get('predicate', '')
    
    if q_type == 'taxonomic':
        ans, state, path = v.validate_taxonomic(s, p)
    elif q_type == 'categorical':
        ans, state = v.validate_categorical(s, p)
        path = []
    elif q_type == 'hypothetical':
        ans, state = v.validate_hypothetical(s, p)
        path = []
    else:
        ans, state, path = None, EpistemicState.SHAKK, []
        
    if state != EpistemicState.SHAKK and ans is not None:
        covered_count += 1
        if gt_bool is not None and ans != gt_bool:
            mismatches.append({
                'index': idx + 1,
                'id': item.get('id'),
                'question': q,
                'gt': gt_bool,
                'kb_ans': ans,
                'state': state.value,
                'type': q_type,
                'subject': s,
                'predicate': p
            })

print(f'Total Covered by KB: {covered_count} / {len(bench)}')
print(f'Total KB vs Ground Truth Mismatches (Potential False Alarms): {len(mismatches)}')
for m in mismatches:
    print(f'  Query {m["index"]:03d} [{m["id"]}] ({m["type"]}) | Q: "{m["question"]}" | GT: {m["gt"]} vs KB: {m["kb_ans"]} (s: {m["subject"]}, p: {m["predicate"]})')
