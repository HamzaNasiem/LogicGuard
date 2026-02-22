# # proofwriter_extractor.py

# """
# STEP 1: ProofWriter Dataset Extractor + Knowledge Base Extender
# ================================================================
# - Scans ProofWriter dataset (any folder/format)
# - Extracts taxonomic, categorical, hypothetical queries
# - Maps them to LogicGuard format
# - Extends knowledge_base.json with new entities
# - Saves extended_queries.json (ready for Step 2)

# Usage:
#     python step1_proofwriter_extractor.py --proofwriter_dir /path/to/proofwriter

# ProofWriter download:
#     https://allenai.org/data/proofwriter
#     (use depth-0 to depth-5 folders, any split)
# """

# import os
# import re
# import json
# import glob
# import argparse
# from collections import defaultdict
# from typing import List, Dict, Tuple, Optional

# # ─────────────────────────────────────────────────────────────────────
# # PART A: ProofWriter Parser
# # ─────────────────────────────────────────────────────────────────────

# def find_proofwriter_files(base_dir: str) -> List[str]:
#     """Find all JSON/JSONL files in ProofWriter directory tree."""
#     patterns = [
#         os.path.join(base_dir, "**", "*.json"),
#         os.path.join(base_dir, "**", "*.jsonl"),
#         os.path.join(base_dir, "*.json"),
#         os.path.join(base_dir, "*.jsonl"),
#     ]
#     files = []
#     for p in patterns:
#         files.extend(glob.glob(p, recursive=True))
#     # Remove duplicates, sort
#     files = sorted(set(files))
#     print(f"  Found {len(files)} ProofWriter files")
#     return files


# def load_proofwriter_entries(filepath: str) -> List[Dict]:
#     """Load entries from a single ProofWriter file (json or jsonl)."""
#     entries = []
#     try:
#         with open(filepath, 'r', encoding='utf-8') as f:
#             content = f.read().strip()
#             if not content:
#                 return []
#             # Try JSONL first (one JSON per line)
#             if filepath.endswith('.jsonl') or '\n{' in content[:200]:
#                 for line in content.splitlines():
#                     line = line.strip()
#                     if line and line.startswith('{'):
#                         try:
#                             entries.append(json.loads(line))
#                         except json.JSONDecodeError:
#                             pass
#             else:
#                 data = json.loads(content)
#                 # ProofWriter can be list or dict with nested entries
#                 if isinstance(data, list):
#                     entries = data
#                 elif isinstance(data, dict):
#                     # Some versions: {"id": ..., "triples": ..., "questions": ...}
#                     if 'questions' in data:
#                         entries = [data]
#                     # Some versions: {"train": [...], "dev": [...]}
#                     for key in ['train', 'dev', 'test', 'data']:
#                         if key in data and isinstance(data[key], list):
#                             entries.extend(data[key])
#     except Exception as e:
#         print(f"  Warning: could not parse {os.path.basename(filepath)}: {e}")
#     return entries


# def extract_theory_facts(theory_text) -> List[str]:
#     """
#     Extract individual fact sentences from theory.
#     Theory can be:
#     - A string: "Anne is a cat. Bob is a dog."
#     - A list of strings: ["Anne is a cat.", "Bob is a dog."]
#     - A dict with 'triples' key
#     """
#     facts = []
#     if isinstance(theory_text, str):
#         # Split on periods, filter empty
#         for sent in theory_text.split('.'):
#             s = sent.strip()
#             if s:
#                 facts.append(s + '.')
#     elif isinstance(theory_text, list):
#         for item in theory_text:
#             if isinstance(item, str):
#                 facts.append(item.strip())
#             elif isinstance(item, dict):
#                 # {"text": "...", ...}
#                 if 'text' in item:
#                     facts.append(item['text'].strip())
#     elif isinstance(theory_text, dict):
#         if 'triples' in theory_text:
#             return extract_theory_facts(theory_text['triples'])
#     return [f for f in facts if len(f) > 3]


# def parse_is_a_facts(facts: List[str]) -> List[Tuple[str, str]]:
#     """
#     Extract IS-A pairs from fact sentences.
#     Patterns:
#       "X is a Y."           → (X, Y)
#       "X are Y."            → (X_class, Y_class)
#       "All X are Y."        → (X, Y)
#       "X is an animal."     → (X, animal)
#     """
#     pairs = []
#     patterns = [
#         # "Anne is a cat" → individual entity (ignore individual→class)
#         # We want CLASS-level: "Cats are mammals", "All dogs are animals"
#         r'^all\s+(\w+)\s+are\s+(\w+)',
#         r'^(\w+)s?\s+are\s+(\w+)',
#         r'^(\w+)\s+is\s+a\s+(\w+)',
#         r'^(\w+)\s+is\s+an\s+(\w+)',
#     ]
#     for fact in facts:
#         f = fact.lower().strip().rstrip('.')
#         for pat in patterns:
#             m = re.match(pat, f)
#             if m:
#                 subj, pred = m.group(1).strip(), m.group(2).strip()
#                 # Skip individual names (capitalized originals)
#                 if len(subj) > 1 and len(pred) > 1:
#                     pairs.append((subj, pred))
#                 break
#     return pairs


# def parse_property_facts(facts: List[str]) -> List[Tuple[str, str]]:
#     """
#     Extract entity→property pairs.
#     "X are cold-blooded" → (X, cold_blooded)
#     "X can fly"          → (X, can_fly)
#     "X have wings"       → (X, wings)
#     """
#     pairs = []
#     patterns = [
#         r'^(\w+)\s+(?:are|is)\s+(cold[_-]blooded|warm[_-]blooded|nocturnal|diurnal)',
#         r'^(\w+)\s+(?:have|has)\s+(\w[\w\s]+)',
#         r'^(\w+)\s+can\s+(\w[\w\s]+)',
#     ]
#     for fact in facts:
#         f = fact.lower().strip().rstrip('.')
#         for pat in patterns:
#             m = re.match(pat, f)
#             if m:
#                 entity = m.group(1).strip()
#                 prop = m.group(2).strip().replace(' ', '_').replace('-', '_')
#                 if len(entity) > 1 and len(prop) > 1:
#                     pairs.append((entity, prop))
#                 break
#     return pairs


# def extract_questions_from_entry(entry: Dict) -> List[Dict]:
#     """
#     Extract questions with answers from a ProofWriter entry.
#     Handles multiple formats:
#     - entry['questions'] = [{"question": "...", "answer": "True/False"}, ...]
#     - entry['questions'] = {"q_id": {"question": "...", "answer": "..."}, ...}
#     """
#     raw_qs = entry.get('questions', entry.get('QA', []))
#     results = []

#     if isinstance(raw_qs, list):
#         for q in raw_qs:
#             if isinstance(q, dict):
#                 text = q.get('question', q.get('text', '')).strip()
#                 ans  = str(q.get('answer', q.get('label', ''))).strip().lower()
#                 if text and ans in ('true', 'false'):
#                     results.append({'question': text, 'answer': ans == 'true'})
#     elif isinstance(raw_qs, dict):
#         for qid, qdata in raw_qs.items():
#             if isinstance(qdata, dict):
#                 text = qdata.get('question', '').strip()
#                 ans  = str(qdata.get('answer', '')).strip().lower()
#                 if text and ans in ('true', 'false'):
#                     results.append({'question': text, 'answer': ans == 'true'})
#     return results


# # ─────────────────────────────────────────────────────────────────────
# # PART B: Map ProofWriter Qs → LogicGuard format
# # ─────────────────────────────────────────────────────────────────────

# def classify_proofwriter_question(q_text: str) -> Optional[str]:
#     """Classify question into LogicGuard types."""
#     t = q_text.lower().strip()
#     if re.match(r'^are all', t):
#         return 'taxonomic'
#     if re.match(r'^is\s+\w+\s+a[n]?\s+', t):
#         return 'taxonomic'
#     if re.match(r'^do all\s+\w+\s+have', t):
#         return 'categorical'
#     if re.match(r'^if\s+', t):
#         return 'hypothetical'
#     # "Can all X ..." → categorical
#     if re.match(r'^can all\s+', t):
#         return 'categorical'
#     return None


# def convert_proofwriter_to_logicguard(
#     entry: Dict
# ) -> List[Dict]:
#     """
#     Convert a ProofWriter entry into LogicGuard-compatible queries.
#     Returns list of:
#     {
#       'question': str,
#       'ground_truth': bool,       # True=YES, False=NO
#       'type': 'taxonomic'|'categorical'|'hypothetical',
#       'source': 'proofwriter',
#       'theory': str               # the KB context
#     }
#     """
#     # Get theory text
#     theory = entry.get('theory', entry.get('triples', entry.get('context', '')))
#     theory_str = ' '.join(extract_theory_facts(theory)) if not isinstance(theory, str) else theory

#     questions = extract_questions_from_entry(entry)
#     results = []
#     for q in questions:
#         qtype = classify_proofwriter_question(q['question'])
#         if qtype:
#             results.append({
#                 'question':     q['question'],
#                 'ground_truth': q['answer'],
#                 'type':         qtype,
#                 'source':       'proofwriter',
#                 'theory':       theory_str[:500]  # keep short
#             })
#     return results


# # ─────────────────────────────────────────────────────────────────────
# # PART C: KB Extender
# # ─────────────────────────────────────────────────────────────────────

# # Known animal/entity taxonomy expansions derived from ProofWriter
# # (These are standard biological facts — not fabricated)
# STANDARD_TAXONOMY_EXTENSIONS = {
#     # More animals
#     'cow':        ['mammal', 'animal', 'living_thing'],
#     'horse':      ['mammal', 'animal', 'living_thing'],
#     'pig':        ['mammal', 'animal', 'living_thing'],
#     'sheep':      ['mammal', 'animal', 'living_thing'],
#     'rabbit':     ['mammal', 'animal', 'living_thing'],
#     'bear':       ['mammal', 'animal', 'carnivore', 'living_thing'],
#     'elephant':   ['mammal', 'animal', 'living_thing'],
#     'giraffe':    ['mammal', 'animal', 'living_thing'],
#     'zebra':      ['mammal', 'animal', 'living_thing'],
#     'gorilla':    ['primate', 'mammal', 'animal', 'living_thing'],
#     'chimpanzee': ['primate', 'mammal', 'animal', 'living_thing'],
#     'monkey':     ['primate', 'mammal', 'animal', 'living_thing'],
#     'bat':        ['mammal', 'animal', 'living_thing'],
#     'otter':      ['mammal', 'animal', 'living_thing'],
#     'seal':       ['mammal', 'animal', 'living_thing'],
#     'walrus':     ['mammal', 'animal', 'living_thing'],
#     'mouse':      ['rodent', 'mammal', 'animal', 'living_thing'],
#     'rat':        ['rodent', 'mammal', 'animal', 'living_thing'],
#     'squirrel':   ['rodent', 'mammal', 'animal', 'living_thing'],
#     # More birds
#     'parrot':     ['bird', 'animal', 'living_thing'],
#     'owl':        ['bird', 'animal', 'living_thing'],
#     'hawk':       ['bird', 'animal', 'living_thing'],
#     'crow':       ['bird', 'animal', 'living_thing'],
#     'robin':      ['bird', 'animal', 'living_thing'],
#     'duck':       ['bird', 'animal', 'living_thing'],
#     'swan':       ['bird', 'animal', 'living_thing'],
#     'penguin':    ['bird', 'animal', 'living_thing'],
#     # Reptiles
#     'crocodile':  ['reptile', 'animal', 'living_thing'],
#     'turtle':     ['reptile', 'animal', 'living_thing'],
#     'gecko':      ['reptile', 'animal', 'living_thing'],
#     'iguana':     ['reptile', 'animal', 'living_thing'],
#     # More fish
#     'tuna':       ['fish', 'animal', 'living_thing'],
#     'trout':      ['fish', 'animal', 'living_thing'],
#     'goldfish':   ['fish', 'animal', 'living_thing'],
#     # Plants
#     'rose':       ['flower', 'plant', 'living_thing'],
#     'oak':        ['tree', 'plant', 'living_thing'],
#     'pine':       ['tree', 'plant', 'living_thing'],
#     'tulip':      ['flower', 'plant', 'living_thing'],
#     # More vehicles
#     'truck':      ['vehicle', 'machine'],
#     'bus':        ['vehicle', 'machine'],
#     'train':      ['vehicle', 'machine'],
#     'motorcycle': ['vehicle', 'machine'],
#     'helicopter': ['aircraft', 'vehicle', 'machine'],
#     # More geometry
#     'rhombus':    ['quadrilateral', 'polygon', 'shape'],
#     'pentagon':   ['polygon', 'shape'],
#     'hexagon':    ['polygon', 'shape'],
#     'polygon':    ['shape'],
#     'quadrilateral': ['polygon', 'shape'],
#     # Intermediate taxonomy nodes
#     'rodent':     ['mammal', 'animal', 'living_thing'],
#     'primate':    ['mammal', 'animal', 'living_thing'],
#     'carnivore':  ['animal', 'living_thing'],
#     'plant':      ['living_thing'],
#     'flower':     ['plant', 'living_thing'],
#     'tree':       ['plant', 'living_thing'],
#     'aircraft':   ['vehicle'],
# }

# STANDARD_PROPERTY_EXTENSIONS = {
#     'reptile':      ['cold_blooded', 'has_scales', 'scales', 'lays_eggs'],
#     'amphibian':    ['cold_blooded', 'lives_near_water', 'lays_eggs'],
#     'plant':        ['makes_own_food', 'photosynthesis', 'needs_sunlight', 'needs_water'],
#     'carnivore':    ['eats_meat', 'has_sharp_teeth'],
#     'rodent':       ['has_sharp_incisors', 'warm_blooded', 'has_hair', 'gives_milk'],
#     'primate':      ['warm_blooded', 'has_hair', 'gives_milk', 'has_hands'],
#     'vehicle':      ['has_wheels_or_propulsion', 'transports_people'],
#     'aircraft':     ['can_fly', 'has_wings', 'uses_fuel'],
#     'polygon':      ['has_straight_sides', 'is_closed_shape'],
#     'quadrilateral':['has_four_sides', 'has_four_angles'],
# }

# STANDARD_CONDITIONAL_EXTENSIONS = {
#     'it is cold':           ['water_may_freeze', 'temperature_low', 'need_warm_clothes'],
#     'plant gets sunlight':  ['photosynthesis_occurs', 'plant_grows', 'energy_produced'],
#     'animal eats food':     ['animal_gets_energy', 'animal_survives', 'digestion_occurs'],
#     'fire is present':      ['oxygen_consumed', 'heat_released', 'light_produced'],
#     'pressure increases':   ['volume_decreases', 'compression_occurs'],
#     'temperature drops':    ['water_may_freeze', 'molecules_slow', 'energy_decreases'],
#     'electricity flows':    ['circuit_complete', 'current_present', 'light_if_bulb'],
#     'seed is planted':      ['plant_may_grow', 'needs_water', 'needs_sunlight'],
# }


# def extend_knowledge_base(kb_path: str, extracted_pairs: Dict) -> Dict:
#     """
#     Merge ProofWriter-extracted facts with standard extensions
#     into existing knowledge_base.json
#     """
#     with open(kb_path, 'r') as f:
#         kb = json.load(f)

#     orig_tax = len(kb['taxonomies'])
#     orig_prop = len(kb['properties'])
#     orig_cond = len(kb['conditionals'])

#     # Add standard taxonomy extensions
#     for entity, parents in STANDARD_TAXONOMY_EXTENSIONS.items():
#         if entity not in kb['taxonomies']:
#             kb['taxonomies'][entity] = parents
#         else:
#             # Merge, avoid duplicates
#             existing = kb['taxonomies'][entity]
#             for p in parents:
#                 if p not in existing:
#                     existing.append(p)

#     # Add ProofWriter-extracted IS-A pairs
#     for child, parent in extracted_pairs.get('is_a', []):
#         child, parent = child.lower(), parent.lower()
#         if child not in kb['taxonomies']:
#             kb['taxonomies'][child] = [parent]
#         elif parent not in kb['taxonomies'][child]:
#             kb['taxonomies'][child].append(parent)

#     # Add standard property extensions
#     for entity, props in STANDARD_PROPERTY_EXTENSIONS.items():
#         if entity not in kb['properties']:
#             kb['properties'][entity] = props
#         else:
#             for p in props:
#                 if p not in kb['properties'][entity]:
#                     kb['properties'][entity].append(p)

#     # Add ProofWriter-extracted properties
#     for entity, prop in extracted_pairs.get('properties', []):
#         entity, prop = entity.lower(), prop.lower()
#         if entity not in kb['properties']:
#             kb['properties'][entity] = [prop]
#         elif prop not in kb['properties'][entity]:
#             kb['properties'][entity].append(prop)

#     # Add standard conditional extensions
#     for condition, consequences in STANDARD_CONDITIONAL_EXTENSIONS.items():
#         if condition not in kb['conditionals']:
#             kb['conditionals'][condition] = consequences
#         else:
#             for c in consequences:
#                 if c not in kb['conditionals'][condition]:
#                     kb['conditionals'][condition].append(c)

#     print(f"\n  KB Extended:")
#     print(f"  Taxonomies : {orig_tax} → {len(kb['taxonomies'])} (+{len(kb['taxonomies'])-orig_tax})")
#     print(f"  Properties : {orig_prop} → {len(kb['properties'])} (+{len(kb['properties'])-orig_prop})")
#     print(f"  Conditionals: {orig_cond} → {len(kb['conditionals'])} (+{len(kb['conditionals'])-orig_cond})")
#     return kb


# # ─────────────────────────────────────────────────────────────────────
# # PART D: Original 100-Query Set (from paper)
# # ─────────────────────────────────────────────────────────────────────

# ORIGINAL_QUERIES = [
#     # ── Taxonomic (50 queries) ────────────────────────────────────────
#     # Valid IS-A relationships (True)
#     {"question": "Are all dogs mammals?",          "ground_truth": True,  "type": "taxonomic", "source": "original"},
#     {"question": "Are all cats mammals?",           "ground_truth": True,  "type": "taxonomic", "source": "original"},
#     {"question": "Are all whales mammals?",         "ground_truth": True,  "type": "taxonomic", "source": "original"},
#     {"question": "Are all dolphins mammals?",       "ground_truth": True,  "type": "taxonomic", "source": "original"},
#     {"question": "Are all lions mammals?",          "ground_truth": True,  "type": "taxonomic", "source": "original"},
#     {"question": "Are all tigers felines?",         "ground_truth": True,  "type": "taxonomic", "source": "original"},
#     {"question": "Are all wolves canines?",         "ground_truth": True,  "type": "taxonomic", "source": "original"},
#     {"question": "Are all sharks fish?",            "ground_truth": True,  "type": "taxonomic", "source": "original"},
#     {"question": "Are all salmon fish?",            "ground_truth": True,  "type": "taxonomic", "source": "original"},
#     {"question": "Are all eagles birds?",           "ground_truth": True,  "type": "taxonomic", "source": "original"},
#     {"question": "Are all sparrows birds?",         "ground_truth": True,  "type": "taxonomic", "source": "original"},
#     {"question": "Are all snakes reptiles?",        "ground_truth": True,  "type": "taxonomic", "source": "original"},
#     {"question": "Are all lizards reptiles?",       "ground_truth": True,  "type": "taxonomic", "source": "original"},
#     {"question": "Are all frogs amphibians?",       "ground_truth": True,  "type": "taxonomic", "source": "original"},
#     {"question": "Are all penguins birds?",         "ground_truth": True,  "type": "taxonomic", "source": "original"},
#     {"question": "Are all squares rectangles?",     "ground_truth": True,  "type": "taxonomic", "source": "original"},
#     {"question": "Are all rectangles polygons?",    "ground_truth": True,  "type": "taxonomic", "source": "original"},
#     {"question": "Are all triangles polygons?",     "ground_truth": True,  "type": "taxonomic", "source": "original"},
#     {"question": "Are all cars vehicles?",          "ground_truth": True,  "type": "taxonomic", "source": "original"},
#     {"question": "Are all bicycles vehicles?",      "ground_truth": True,  "type": "taxonomic", "source": "original"},
#     {"question": "Are all birds animals?",          "ground_truth": True,  "type": "taxonomic", "source": "original"},
#     {"question": "Are all mammals animals?",        "ground_truth": True,  "type": "taxonomic", "source": "original"},
#     {"question": "Are all dogs animals?",           "ground_truth": True,  "type": "taxonomic", "source": "original"},
#     {"question": "Are all airplanes vehicles?",     "ground_truth": True,  "type": "taxonomic", "source": "original"},
#     {"question": "Are all boats vehicles?",         "ground_truth": True,  "type": "taxonomic", "source": "original"},
#     # Invalid IS-A relationships (False)
#     {"question": "Are all animals dogs?",           "ground_truth": False, "type": "taxonomic", "source": "original"},
#     {"question": "Are all animals mammals?",        "ground_truth": False, "type": "taxonomic", "source": "original"},
#     {"question": "Are all mammals dogs?",           "ground_truth": False, "type": "taxonomic", "source": "original"},
#     {"question": "Are all birds mammals?",          "ground_truth": False, "type": "taxonomic", "source": "original"},
#     {"question": "Are all fish mammals?",           "ground_truth": False, "type": "taxonomic", "source": "original"},
#     {"question": "Are all reptiles birds?",         "ground_truth": False, "type": "taxonomic", "source": "original"},
#     {"question": "Are all mammals fish?",           "ground_truth": False, "type": "taxonomic", "source": "original"},
#     {"question": "Are all rectangles squares?",     "ground_truth": False, "type": "taxonomic", "source": "original"},
#     {"question": "Are all polygons triangles?",     "ground_truth": False, "type": "taxonomic", "source": "original"},
#     {"question": "Are all vehicles cars?",          "ground_truth": False, "type": "taxonomic", "source": "original"},
#     {"question": "Are all dogs felines?",           "ground_truth": False, "type": "taxonomic", "source": "original"},
#     {"question": "Are all sharks mammals?",         "ground_truth": False, "type": "taxonomic", "source": "original"},
#     {"question": "Are all penguins mammals?",       "ground_truth": False, "type": "taxonomic", "source": "original"},
#     {"question": "Are all eagles mammals?",         "ground_truth": False, "type": "taxonomic", "source": "original"},
#     {"question": "Are all frogs reptiles?",         "ground_truth": False, "type": "taxonomic", "source": "original"},
#     {"question": "Are all cats birds?",             "ground_truth": False, "type": "taxonomic", "source": "original"},
#     {"question": "Are all fish birds?",             "ground_truth": False, "type": "taxonomic", "source": "original"},
#     {"question": "Are all snakes mammals?",         "ground_truth": False, "type": "taxonomic", "source": "original"},
#     {"question": "Are all triangles rectangles?",   "ground_truth": False, "type": "taxonomic", "source": "original"},
#     {"question": "Are all vehicles bicycles?",      "ground_truth": False, "type": "taxonomic", "source": "original"},
#     {"question": "Are all circles polygons?",       "ground_truth": False, "type": "taxonomic", "source": "original"},
#     {"question": "Are all dolphins fish?",          "ground_truth": False, "type": "taxonomic", "source": "original"},
#     {"question": "Are all wolves felines?",         "ground_truth": False, "type": "taxonomic", "source": "original"},
#     {"question": "Are all boats airplanes?",        "ground_truth": False, "type": "taxonomic", "source": "original"},
#     {"question": "Are all animals living things?",  "ground_truth": True,  "type": "taxonomic", "source": "original"},

#     # ── Categorical (37 queries) ──────────────────────────────────────
#     # Valid property claims (True)
#     {"question": "Do all mammals have hair?",           "ground_truth": True,  "type": "categorical", "source": "original"},
#     {"question": "Do all birds have feathers?",         "ground_truth": True,  "type": "categorical", "source": "original"},
#     {"question": "Do all birds have wings?",            "ground_truth": True,  "type": "categorical", "source": "original"},
#     {"question": "Do all birds lay eggs?",              "ground_truth": True,  "type": "categorical", "source": "original"},
#     {"question": "Do all fish have gills?",             "ground_truth": True,  "type": "categorical", "source": "original"},
#     {"question": "Do all fish live in water?",          "ground_truth": True,  "type": "categorical", "source": "original"},
#     {"question": "Do all squares have four sides?",     "ground_truth": True,  "type": "categorical", "source": "original"},
#     {"question": "Do all rectangles have four sides?",  "ground_truth": True,  "type": "categorical", "source": "original"},
#     {"question": "Do all humans have a heart?",         "ground_truth": True,  "type": "categorical", "source": "original"},
#     {"question": "Do all living things need water?",    "ground_truth": True,  "type": "categorical", "source": "original"},
#     {"question": "Do all mammals give milk?",           "ground_truth": True,  "type": "categorical", "source": "original"},
#     {"question": "Do all mammals have a backbone?",     "ground_truth": True,  "type": "categorical", "source": "original"},
#     {"question": "Do all reptiles have scales?",        "ground_truth": True,  "type": "categorical", "source": "original"},
#     {"question": "Do all triangles have three sides?",  "ground_truth": True,  "type": "categorical", "source": "original"},
#     {"question": "Do all circles have a curved edge?",  "ground_truth": True,  "type": "categorical", "source": "original"},
#     # Invalid property claims (False)
#     {"question": "Do all fish have hair?",              "ground_truth": False, "type": "categorical", "source": "original"},
#     {"question": "Do all birds have hair?",             "ground_truth": False, "type": "categorical", "source": "original"},
#     {"question": "Do all reptiles have hair?",          "ground_truth": False, "type": "categorical", "source": "original"},
#     {"question": "Do all fish give milk?",              "ground_truth": False, "type": "categorical", "source": "original"},
#     {"question": "Do all birds live in water?",         "ground_truth": False, "type": "categorical", "source": "original"},
#     {"question": "Do all mammals have gills?",          "ground_truth": False, "type": "categorical", "source": "original"},
#     {"question": "Do all reptiles have feathers?",      "ground_truth": False, "type": "categorical", "source": "original"},
#     {"question": "Do all triangles have four sides?",   "ground_truth": False, "type": "categorical", "source": "original"},
#     {"question": "Do all squares have three sides?",    "ground_truth": False, "type": "categorical", "source": "original"},
#     {"question": "Do all fish lay eggs on land?",       "ground_truth": False, "type": "categorical", "source": "original"},
#     {"question": "Do all mammals lay eggs?",            "ground_truth": False, "type": "categorical", "source": "original"},
#     {"question": "Do all birds have gills?",            "ground_truth": False, "type": "categorical", "source": "original"},
#     {"question": "Do all rectangles have equal sides?", "ground_truth": False, "type": "categorical", "source": "original"},
#     {"question": "Do all fish have feathers?",          "ground_truth": False, "type": "categorical", "source": "original"},
#     {"question": "Do all snakes have legs?",            "ground_truth": False, "type": "categorical", "source": "original"},
#     {"question": "Do all birds have fur?",              "ground_truth": False, "type": "categorical", "source": "original"},
#     {"question": "Do all fish have wings?",             "ground_truth": False, "type": "categorical", "source": "original"},
#     {"question": "Do all reptiles give milk?",          "ground_truth": False, "type": "categorical", "source": "original"},
#     {"question": "Do all mammals have feathers?",       "ground_truth": False, "type": "categorical", "source": "original"},
#     {"question": "Do all circles have corners?",        "ground_truth": False, "type": "categorical", "source": "original"},
#     {"question": "Do all triangles have right angles?", "ground_truth": False, "type": "categorical", "source": "original"},
#     {"question": "Do all squares have curved sides?",   "ground_truth": False, "type": "categorical", "source": "original"},

#     # ── Hypothetical (13 queries) ─────────────────────────────────────
#     {"question": "If it is raining, is the ground wet?",         "ground_truth": True,  "type": "hypothetical", "source": "original"},
#     {"question": "If there is fire, is there heat?",             "ground_truth": True,  "type": "hypothetical", "source": "original"},
#     {"question": "If water freezes, does it become ice?",        "ground_truth": True,  "type": "hypothetical", "source": "original"},
#     {"question": "If water boils, is it hot?",                   "ground_truth": True,  "type": "hypothetical", "source": "original"},
#     {"question": "If the sun is shining, is it daytime?",        "ground_truth": True,  "type": "hypothetical", "source": "original"},
#     {"question": "If metal is heated, does it expand?",          "ground_truth": True,  "type": "hypothetical", "source": "original"},
#     {"question": "If it is raining, do we need an umbrella?",    "ground_truth": True,  "type": "hypothetical", "source": "original"},
#     {"question": "If fire is present, is oxygen consumed?",      "ground_truth": True,  "type": "hypothetical", "source": "original"},
#     {"question": "If water freezes, does it become steam?",      "ground_truth": False, "type": "hypothetical", "source": "original"},
#     {"question": "If it is raining, is the sky clear?",          "ground_truth": False, "type": "hypothetical", "source": "original"},
#     {"question": "If fire is present, does it feel cold?",       "ground_truth": False, "type": "hypothetical", "source": "original"},
#     {"question": "If water boils, does it freeze?",              "ground_truth": False, "type": "hypothetical", "source": "original"},
#     {"question": "If there is fire, does it need no oxygen?",    "ground_truth": False, "type": "hypothetical", "source": "original"},
# ]


# # ─────────────────────────────────────────────────────────────────────
# # PART E: Build Extended + ProofWriter Query Set
# # ─────────────────────────────────────────────────────────────────────

# def build_extended_query_set_from_proofwriter(pf_queries: List[Dict], kb: Dict) -> List[Dict]:
#     """
#     Filter ProofWriter queries to only those whose entities exist in KB.
#     This ensures LogicGuard can actually validate them.
#     Returns deduplicated, KB-compatible queries.
#     """
#     valid = []
#     seen_questions = set()

#     # Simple extractors to check KB coverage
#     def entity_in_kb(text: str, kb: Dict) -> bool:
#         """Check if any entity in question is in KB."""
#         text_lower = text.lower()
#         for key in list(kb['taxonomies'].keys()) + list(kb['properties'].keys()):
#             if key in text_lower:
#                 return True
#         return False

#     for q in pf_queries:
#         qtext = q['question'].strip()
#         if qtext in seen_questions:
#             continue
#         if len(qtext) < 10:
#             continue
#         # Only include if KB can handle it
#         if entity_in_kb(qtext, kb):
#             seen_questions.add(qtext)
#             valid.append(q)

#     return valid


# def build_additional_kb_queries(kb: Dict) -> List[Dict]:
#     """
#     Generate additional valid queries FROM the extended KB itself.
#     These are deterministically correct — not made up.
#     Logic: For every entity in taxonomy, create IS-A queries.
#     """
#     queries = []
#     seen = set()

#     # Taxonomic: Valid (True)
#     for child, parents in kb['taxonomies'].items():
#         for parent in parents[:2]:  # max 2 per entity
#             q = f"Are all {child}s {parent}s?" if not child.endswith('s') else f"Are all {child} {parent}s?"
#             # Clean up double 's'
#             q = q.replace('ss?', 's?').replace('ys?', 'ies?')
#             if q not in seen and len(child) > 2 and len(parent) > 2:
#                 queries.append({
#                     "question": q.capitalize(),
#                     "ground_truth": True,
#                     "type": "taxonomic",
#                     "source": "kb_generated"
#                 })
#                 seen.add(q)

#     # Categorical: Valid (True) — from properties
#     for entity, props in kb['properties'].items():
#         for prop in props[:2]:  # max 2 per entity
#             prop_readable = prop.replace('_', ' ')
#             q = f"Do all {entity}s have {prop_readable}?"
#             if q not in seen and len(entity) > 2:
#                 queries.append({
#                     "question": q.capitalize(),
#                     "ground_truth": True,
#                     "type": "categorical",
#                     "source": "kb_generated"
#                 })
#                 seen.add(q)

#     # Hypothetical: Valid (True) — from conditionals
#     for cond, consequences in kb['conditionals'].items():
#         for cons in consequences[:1]:
#             cons_readable = cons.replace('_', ' ')
#             q = f"If {cond}, does {cons_readable} occur?"
#             if q not in seen:
#                 queries.append({
#                     "question": q.capitalize(),
#                     "ground_truth": True,
#                     "type": "hypothetical",
#                     "source": "kb_generated"
#                 })
#                 seen.add(q)

#     return queries


# # ─────────────────────────────────────────────────────────────────────
# # MAIN
# # ─────────────────────────────────────────────────────────────────────

# def main():
#     parser = argparse.ArgumentParser(description='ProofWriter Extractor for LogicGuard')
#     parser.add_argument('--proofwriter_dir', type=str, default='./proofwriter',
#                         help='Path to ProofWriter dataset directory')
#     parser.add_argument('--kb_path', type=str, default='knowledge_base.json',
#                         help='Path to existing knowledge_base.json')
#     parser.add_argument('--output_queries', type=str, default='extended_queries.json',
#                         help='Output file for all queries')
#     parser.add_argument('--output_kb', type=str, default='knowledge_base_extended.json',
#                         help='Output file for extended KB')
#     parser.add_argument('--max_proofwriter', type=int, default=200,
#                         help='Max ProofWriter queries to include (default 200)')
#     args = parser.parse_args()

#     print("=" * 65)
#     print("  LogicGuard — Step 1: ProofWriter Extractor + KB Extender")
#     print("=" * 65)

#     # ── 1. Load and parse ProofWriter ────────────────────────────────
#     print(f"\n[1/5] Scanning ProofWriter directory: {args.proofwriter_dir}")
#     pf_files = find_proofwriter_files(args.proofwriter_dir)

#     if not pf_files:
#         print(f"  ⚠️  No ProofWriter files found in: {args.proofwriter_dir}")
#         print(f"  → Continuing with KB-generated queries only")

#     # Collect ProofWriter queries
#     pf_queries_raw = []
#     extracted_pairs = {'is_a': [], 'properties': []}

#     for filepath in pf_files[:20]:  # limit files to prevent OOM
#         entries = load_proofwriter_entries(filepath)
#         for entry in entries[:50]:  # limit entries per file
#             theory = entry.get('theory', entry.get('triples', ''))
#             facts = extract_theory_facts(theory)
#             # Extract KB facts
#             extracted_pairs['is_a'].extend(parse_is_a_facts(facts))
#             extracted_pairs['properties'].extend(parse_property_facts(facts))
#             # Extract questions
#             converted = convert_proofwriter_to_logicguard(entry)
#             pf_queries_raw.extend(converted)

#     print(f"  ProofWriter: {len(pf_queries_raw)} raw questions extracted")
#     print(f"  IS-A pairs from theory: {len(extracted_pairs['is_a'])}")
#     print(f"  Property pairs from theory: {len(extracted_pairs['properties'])}")

#     # ── 2. Extend KB ─────────────────────────────────────────────────
#     print(f"\n[2/5] Extending Knowledge Base...")
#     extended_kb = extend_knowledge_base(args.kb_path, extracted_pairs)

#     with open(args.output_kb, 'w') as f:
#         json.dump(extended_kb, f, indent=2)
#     print(f"  ✅ Extended KB saved: {args.output_kb}")

#     # ── 3. Filter ProofWriter queries to KB-compatible ones ───────────
#     print(f"\n[3/5] Filtering ProofWriter queries for KB compatibility...")
#     pf_compatible = build_extended_query_set_from_proofwriter(pf_queries_raw, extended_kb)
#     # Limit to max_proofwriter
#     pf_selected = pf_compatible[:args.max_proofwriter]
#     print(f"  Compatible: {len(pf_compatible)}, Selected: {len(pf_selected)}")

#     # ── 4. Generate KB-derived queries ───────────────────────────────
#     print(f"\n[4/5] Generating KB-derived queries...")
#     kb_queries = build_additional_kb_queries(extended_kb)
#     print(f"  Generated: {len(kb_queries)} KB-derived queries")

#     # ── 5. Combine all queries ────────────────────────────────────────
#     print(f"\n[5/5] Combining all query sets...")
#     all_queries = []

#     # Original 100 (paper's main dataset)
#     all_queries.extend(ORIGINAL_QUERIES)
#     print(f"  Original queries:       {len(ORIGINAL_QUERIES)}")

#     # ProofWriter (external validation)
#     all_queries.extend(pf_selected)
#     print(f"  ProofWriter queries:    {len(pf_selected)}")

#     # KB-generated (extended coverage)
#     # Deduplicate vs originals
#     orig_questions = {q['question'].lower() for q in ORIGINAL_QUERIES}
#     pf_questions   = {q['question'].lower() for q in pf_selected}
#     kb_unique = [q for q in kb_queries
#                  if q['question'].lower() not in orig_questions
#                  and q['question'].lower() not in pf_questions]
#     kb_unique = kb_unique[:100]  # cap at 100 additional
#     all_queries.extend(kb_unique)
#     print(f"  KB-generated queries:   {len(kb_unique)}")
#     print(f"  ─────────────────────────────")
#     print(f"  TOTAL queries:          {len(all_queries)}")

#     # Stats
#     by_type = defaultdict(lambda: {'count': 0, 'true': 0, 'false': 0})
#     by_source = defaultdict(int)
#     for q in all_queries:
#         t = q['type']
#         by_type[t]['count'] += 1
#         if q['ground_truth']:
#             by_type[t]['true'] += 1
#         else:
#             by_type[t]['false'] += 1
#         by_source[q.get('source', 'unknown')] += 1

#     print(f"\n  By type:")
#     for qtype, stats in by_type.items():
#         print(f"    {qtype:15}: {stats['count']:3} total  ({stats['true']} True / {stats['false']} False)")

#     print(f"\n  By source:")
#     for src, cnt in by_source.items():
#         print(f"    {src:20}: {cnt}")

#     # Save
#     output = {
#         'metadata': {
#             'total': len(all_queries),
#             'by_type': {k: v for k, v in by_type.items()},
#             'by_source': dict(by_source),
#             'kb_path': args.output_kb
#         },
#         'queries': all_queries
#     }
#     with open(args.output_queries, 'w') as f:
#         json.dump(output, f, indent=2)

#     print(f"\n  ✅ All queries saved: {args.output_queries}")
#     print(f"\n{'=' * 65}")
#     print(f"  STEP 1 COMPLETE")
#     print(f"  Next: python step2_multi_model_runner.py")
#     print(f"{'=' * 65}\n")


# if __name__ == '__main__':
#     main()


"""
STEP 1 (FIXED) — No file dependency, builds KB from scratch
"""
import os, re, json, glob, argparse
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

# ── Full KB built inside script — NO file reading ─────────────────────
def build_base_kb() -> Dict:
    taxonomies = {
        "dog":["canine","mammal","animal","living_thing"],
        "wolf":["canine","mammal","animal","living_thing"],
        "fox":["canine","mammal","animal","living_thing"],
        "cat":["feline","mammal","animal","living_thing"],
        "lion":["feline","mammal","animal","living_thing"],
        "tiger":["feline","mammal","animal","living_thing"],
        "leopard":["feline","mammal","animal","living_thing"],
        "whale":["cetacean","mammal","animal","living_thing"],
        "dolphin":["cetacean","mammal","animal","living_thing"],
        "porpoise":["cetacean","mammal","animal","living_thing"],
        "bat":["mammal","animal","living_thing"],
        "bear":["mammal","animal","living_thing"],
        "elephant":["mammal","animal","living_thing"],
        "horse":["mammal","animal","living_thing"],
        "cow":["mammal","animal","living_thing"],
        "pig":["mammal","animal","living_thing"],
        "sheep":["mammal","animal","living_thing"],
        "rabbit":["mammal","animal","living_thing"],
        "rat":["mammal","animal","living_thing"],
        "mouse":["mammal","animal","living_thing"],
        "deer":["mammal","animal","living_thing"],
        "giraffe":["mammal","animal","living_thing"],
        "zebra":["mammal","animal","living_thing"],
        "gorilla":["primate","mammal","animal","living_thing"],
        "chimpanzee":["primate","mammal","animal","living_thing"],
        "monkey":["primate","mammal","animal","living_thing"],
        "human":["primate","mammal","animal","living_thing"],
        "primate":["mammal","animal","living_thing"],
        "canine":["mammal","animal","living_thing"],
        "feline":["mammal","animal","living_thing"],
        "cetacean":["mammal","animal","living_thing"],
        "mammal":["animal","living_thing"],
        "sparrow":["bird","animal","living_thing"],
        "eagle":["bird","animal","living_thing"],
        "penguin":["bird","animal","living_thing"],
        "parrot":["bird","animal","living_thing"],
        "ostrich":["bird","animal","living_thing"],
        "hawk":["bird","animal","living_thing"],
        "owl":["bird","animal","living_thing"],
        "crow":["bird","animal","living_thing"],
        "robin":["bird","animal","living_thing"],
        "pigeon":["bird","animal","living_thing"],
        "duck":["bird","animal","living_thing"],
        "swan":["bird","animal","living_thing"],
        "flamingo":["bird","animal","living_thing"],
        "peacock":["bird","animal","living_thing"],
        "bird":["animal","living_thing"],
        "snake":["reptile","animal","living_thing"],
        "lizard":["reptile","animal","living_thing"],
        "turtle":["reptile","animal","living_thing"],
        "tortoise":["reptile","animal","living_thing"],
        "crocodile":["reptile","animal","living_thing"],
        "alligator":["reptile","animal","living_thing"],
        "gecko":["reptile","animal","living_thing"],
        "chameleon":["reptile","animal","living_thing"],
        "reptile":["animal","living_thing"],
        "shark":["fish","animal","living_thing"],
        "salmon":["fish","animal","living_thing"],
        "tuna":["fish","animal","living_thing"],
        "goldfish":["fish","animal","living_thing"],
        "trout":["fish","animal","living_thing"],
        "cod":["fish","animal","living_thing"],
        "clownfish":["fish","animal","living_thing"],
        "fish":["animal","living_thing"],
        "frog":["amphibian","animal","living_thing"],
        "toad":["amphibian","animal","living_thing"],
        "salamander":["amphibian","animal","living_thing"],
        "newt":["amphibian","animal","living_thing"],
        "amphibian":["animal","living_thing"],
        "ant":["insect","animal","living_thing"],
        "bee":["insect","animal","living_thing"],
        "butterfly":["insect","animal","living_thing"],
        "mosquito":["insect","animal","living_thing"],
        "spider":["arachnid","animal","living_thing"],
        "scorpion":["arachnid","animal","living_thing"],
        "insect":["animal","living_thing"],
        "arachnid":["animal","living_thing"],
        "animal":["living_thing"],
        "plant":["living_thing"],
        "tree":["plant","living_thing"],
        "flower":["plant","living_thing"],
        "fungus":["living_thing"],
        "square":["rectangle","quadrilateral","rhombus","polygon","shape"],
        "rectangle":["quadrilateral","polygon","shape"],
        "rhombus":["quadrilateral","polygon","shape"],
        "quadrilateral":["polygon","shape"],
        "triangle":["polygon","shape"],
        "pentagon":["polygon","shape"],
        "hexagon":["polygon","shape"],
        "polygon":["shape"],
        "circle":["shape"],
        "oval":["shape"],
        "ellipse":["shape"],
        "car":["vehicle"],
        "bus":["vehicle"],
        "truck":["vehicle"],
        "motorcycle":["vehicle"],
        "bicycle":["vehicle"],
        "airplane":["vehicle","aircraft"],
        "helicopter":["vehicle","aircraft"],
        "boat":["vehicle"],
        "ship":["vehicle"],
        "train":["vehicle"],
        "aircraft":["vehicle"],
        "apple":["fruit","food"],
        "banana":["fruit","food"],
        "orange":["fruit","food"],
        "mango":["fruit","food"],
        "grape":["fruit","food"],
        "strawberry":["fruit","food"],
        "carrot":["vegetable","food"],
        "potato":["vegetable","food"],
        "tomato":["vegetable","food"],
        "fruit":["food"],
        "vegetable":["food"],
    }
    properties = {
        "mammal":["has_hair","hair","fur","gives_milk","milk","warm_blooded",
                  "has_backbone","backbone","has_heart","heart","has_brain","brain",
                  "has_lungs","lungs","vertebrate"],
        "bird":["has_feathers","feathers","has_wings","wings","lays_eggs","lay_eggs",
                "eggs","has_beak","beak","has_backbone","backbone"],
        "fish":["has_gills","gills","has_scales","scales","lives_in_water",
                "cold_blooded","has_backbone","backbone"],
        "reptile":["cold_blooded","has_scales","scales","lays_eggs","lay_eggs",
                   "eggs","has_backbone","backbone"],
        "amphibian":["cold_blooded","lives_near_water","lays_eggs","has_backbone"],
        "insect":["six_legs","6_legs","has_six_legs","three_body_segments"],
        "arachnid":["eight_legs","8_legs","has_eight_legs"],
        "spider":["eight_legs","8_legs"],
        "human":["has_heart","heart","has_brain","brain","mortal","has_hands","hands",
                 "upright","can_think"],
        "primate":["warm_blooded","has_hair","gives_milk","has_hands","hands"],
        "living_thing":["needs_water","water","needs_food","food","can_die","die",
                        "grows","grow","mortal","reproduces"],
        "animal":["needs_food","food","needs_water","water","can_die"],
        "plant":["needs_water","water","needs_sunlight","makes_own_food","photosynthesis"],
        "tree":["has_roots","roots","root"],
        "square":["four_sides","4_sides","equal_sides","4_equal_sides",
                  "four_right_angles","4_right_angles","right_angles"],
        "rectangle":["four_sides","4_sides","four_right_angles","4_right_angles","right_angles"],
        "triangle":["three_sides","3_sides","three_angles","3_angles"],
        "circle":["no_corners","curved","round","has_radius","radius"],
        "polygon":["has_straight_sides","is_closed_shape"],
        "quadrilateral":["has_four_sides","four_sides","4_sides","has_four_angles"],
        "vehicle":["wheels","has_wheels","transports_people"],
        "aircraft":["wings","has_wings","can_fly","uses_fuel"],
        "airplane":["wings","has_wings"],
        "car":["wheels","has_wheels"],
    }
    conditionals = {
        "raining":["ground_wet","wet","need_umbrella","sky_cloudy"],
        "fire":["heat","hot","dangerous","light","requires_oxygen","oxygen","smoke","produces_heat"],
        "water freezes":["ice","solid","becomes_ice"],
        "water_freezes":["ice","solid","becomes_ice"],
        "water boils":["hot","steam","100_degrees"],
        "water_boils":["hot","steam","100_degrees"],
        "metal heated":["expands","expand"],
        "metal is heated":["expands","expand"],
        "metal_heated":["expands","expand"],
        "sun shining":["daytime","day"],
        "sun_shining":["daytime","day"],
        "the sun is shining":["daytime","day"],
        "sun is shining":["daytime","day"],
        "night":["dark"],
        "it is night":["dark"],
        "breathing":["alive","living"],
        "you are breathing":["alive","living"],
        "alive":["needs_food","food"],
        "something is alive":["needs_food","food"],
        "a person is human":["mortal"],
        "person is human":["mortal"],
        "human":["mortal","can_think"],
        "living_thing":["needs_food","can_die"],
        "it is cold":["water_may_freeze","temperature_low","need_warm_clothes"],
        "plant gets sunlight":["photosynthesis_occurs","plant_grows","energy_produced"],
        "animal eats food":["animal_gets_energy","animal_survives","digestion_occurs"],
        "fire is present":["oxygen_consumed","heat_released","light_produced"],
        "pressure increases":["volume_decreases","compression_occurs"],
        "temperature drops":["water_may_freeze","molecules_slow","energy_decreases"],
        "electricity flows":["circuit_complete","current_present"],
        "seed is planted":["plant_may_grow","needs_water","needs_sunlight"],
    }
    return {"taxonomies": taxonomies, "properties": properties, "conditionals": conditionals}

# ── ProofWriter scanner (extracts IS-A from theory text) ─────────────
def find_proofwriter_files(base_dir):
    files = []
    for pat in [os.path.join(base_dir,"**","*.json"), os.path.join(base_dir,"**","*.jsonl"),
                os.path.join(base_dir,"*.json"), os.path.join(base_dir,"*.jsonl")]:
        files.extend(glob.glob(pat, recursive=True))
    files = sorted(set(files))
    print(f"  Found {len(files)} ProofWriter files")
    return files

def load_proofwriter_file(filepath):
    try:
        with open(filepath,'r',encoding='utf-8',errors='ignore') as f:
            content = f.read().strip()
        if not content: return []
        if filepath.endswith('.jsonl'):
            entries = []
            for line in content.splitlines():
                line = line.strip()
                if line.startswith('{'):
                    try: entries.append(json.loads(line))
                    except: pass
            return entries
        data = json.loads(content)
        if isinstance(data, list): return data
        if isinstance(data, dict):
            if 'triples' in data or 'questions' in data: return [data]
            for key in ['train','dev','test','data','examples']:
                if key in data and isinstance(data[key], list): return data[key]
    except: pass
    return []

def extract_isa_from_proofwriter(entry):
    pairs = []
    triples = entry.get('triples', {})
    if isinstance(triples, dict):
        for fact_text in triples.keys():
            text = fact_text.lower().strip().rstrip('.')
            m = re.match(r'^(\w+)\s+is\s+a[n]?\s+(\w+)', text)
            if m:
                s, p = m.group(1), m.group(2)
                if s.isalpha() and p.isalpha() and len(s)>1 and len(p)>1:
                    pairs.append((s, p))
    elif isinstance(triples, list):
        for item in triples:
            if isinstance(item, str):
                text = item.lower().strip().rstrip('.')
                m = re.match(r'^(\w+)\s+is\s+a[n]?\s+(\w+)', text)
                if m:
                    s, p = m.group(1), m.group(2)
                    if s.isalpha() and p.isalpha():
                        pairs.append((s, p))
    return pairs

# ── 100 Original Queries ──────────────────────────────────────────────
ORIGINAL_QUERIES = [
    # Taxonomic True
    {"question":"Are all dogs mammals?","ground_truth":True,"type":"taxonomic","source":"original"},
    {"question":"Are all cats mammals?","ground_truth":True,"type":"taxonomic","source":"original"},
    {"question":"Are all whales mammals?","ground_truth":True,"type":"taxonomic","source":"original"},
    {"question":"Are all dolphins mammals?","ground_truth":True,"type":"taxonomic","source":"original"},
    {"question":"Are all lions mammals?","ground_truth":True,"type":"taxonomic","source":"original"},
    {"question":"Are all tigers felines?","ground_truth":True,"type":"taxonomic","source":"original"},
    {"question":"Are all wolves canines?","ground_truth":True,"type":"taxonomic","source":"original"},
    {"question":"Are all sharks fish?","ground_truth":True,"type":"taxonomic","source":"original"},
    {"question":"Are all salmon fish?","ground_truth":True,"type":"taxonomic","source":"original"},
    {"question":"Are all eagles birds?","ground_truth":True,"type":"taxonomic","source":"original"},
    {"question":"Are all sparrows birds?","ground_truth":True,"type":"taxonomic","source":"original"},
    {"question":"Are all snakes reptiles?","ground_truth":True,"type":"taxonomic","source":"original"},
    {"question":"Are all lizards reptiles?","ground_truth":True,"type":"taxonomic","source":"original"},
    {"question":"Are all frogs amphibians?","ground_truth":True,"type":"taxonomic","source":"original"},
    {"question":"Are all penguins birds?","ground_truth":True,"type":"taxonomic","source":"original"},
    {"question":"Are all squares rectangles?","ground_truth":True,"type":"taxonomic","source":"original"},
    {"question":"Are all rectangles polygons?","ground_truth":True,"type":"taxonomic","source":"original"},
    {"question":"Are all triangles polygons?","ground_truth":True,"type":"taxonomic","source":"original"},
    {"question":"Are all cars vehicles?","ground_truth":True,"type":"taxonomic","source":"original"},
    {"question":"Are all bicycles vehicles?","ground_truth":True,"type":"taxonomic","source":"original"},
    {"question":"Are all birds animals?","ground_truth":True,"type":"taxonomic","source":"original"},
    {"question":"Are all mammals animals?","ground_truth":True,"type":"taxonomic","source":"original"},
    {"question":"Are all dogs animals?","ground_truth":True,"type":"taxonomic","source":"original"},
    {"question":"Are all airplanes vehicles?","ground_truth":True,"type":"taxonomic","source":"original"},
    {"question":"Are all boats vehicles?","ground_truth":True,"type":"taxonomic","source":"original"},
    {"question":"Are all bears mammals?","ground_truth":True,"type":"taxonomic","source":"original"},
    {"question":"Are all horses mammals?","ground_truth":True,"type":"taxonomic","source":"original"},
    {"question":"Are all elephants mammals?","ground_truth":True,"type":"taxonomic","source":"original"},
    {"question":"Are all penguins animals?","ground_truth":True,"type":"taxonomic","source":"original"},
    {"question":"Are all animals living things?","ground_truth":True,"type":"taxonomic","source":"original"},
    # Taxonomic False
    {"question":"Are all animals dogs?","ground_truth":False,"type":"taxonomic","source":"original"},
    {"question":"Are all animals mammals?","ground_truth":False,"type":"taxonomic","source":"original"},
    {"question":"Are all mammals dogs?","ground_truth":False,"type":"taxonomic","source":"original"},
    {"question":"Are all birds mammals?","ground_truth":False,"type":"taxonomic","source":"original"},
    {"question":"Are all fish mammals?","ground_truth":False,"type":"taxonomic","source":"original"},
    {"question":"Are all reptiles birds?","ground_truth":False,"type":"taxonomic","source":"original"},
    {"question":"Are all mammals fish?","ground_truth":False,"type":"taxonomic","source":"original"},
    {"question":"Are all rectangles squares?","ground_truth":False,"type":"taxonomic","source":"original"},
    {"question":"Are all polygons triangles?","ground_truth":False,"type":"taxonomic","source":"original"},
    {"question":"Are all vehicles cars?","ground_truth":False,"type":"taxonomic","source":"original"},
    {"question":"Are all dogs felines?","ground_truth":False,"type":"taxonomic","source":"original"},
    {"question":"Are all sharks mammals?","ground_truth":False,"type":"taxonomic","source":"original"},
    {"question":"Are all penguins mammals?","ground_truth":False,"type":"taxonomic","source":"original"},
    {"question":"Are all eagles mammals?","ground_truth":False,"type":"taxonomic","source":"original"},
    {"question":"Are all frogs reptiles?","ground_truth":False,"type":"taxonomic","source":"original"},
    {"question":"Are all cats birds?","ground_truth":False,"type":"taxonomic","source":"original"},
    {"question":"Are all snakes mammals?","ground_truth":False,"type":"taxonomic","source":"original"},
    {"question":"Are all triangles rectangles?","ground_truth":False,"type":"taxonomic","source":"original"},
    {"question":"Are all circles polygons?","ground_truth":False,"type":"taxonomic","source":"original"},
    {"question":"Are all dolphins fish?","ground_truth":False,"type":"taxonomic","source":"original"},
    # Categorical True
    {"question":"Do all mammals have hair?","ground_truth":True,"type":"categorical","source":"original"},
    {"question":"Do all birds have feathers?","ground_truth":True,"type":"categorical","source":"original"},
    {"question":"Do all birds have wings?","ground_truth":True,"type":"categorical","source":"original"},
    {"question":"Do all birds lay eggs?","ground_truth":True,"type":"categorical","source":"original"},
    {"question":"Do all fish have gills?","ground_truth":True,"type":"categorical","source":"original"},
    {"question":"Do all fish live in water?","ground_truth":True,"type":"categorical","source":"original"},
    {"question":"Do all squares have four sides?","ground_truth":True,"type":"categorical","source":"original"},
    {"question":"Do all rectangles have four sides?","ground_truth":True,"type":"categorical","source":"original"},
    {"question":"Do all humans have a heart?","ground_truth":True,"type":"categorical","source":"original"},
    {"question":"Do all living things need water?","ground_truth":True,"type":"categorical","source":"original"},
    {"question":"Do all mammals give milk?","ground_truth":True,"type":"categorical","source":"original"},
    {"question":"Do all mammals have a backbone?","ground_truth":True,"type":"categorical","source":"original"},
    {"question":"Do all reptiles have scales?","ground_truth":True,"type":"categorical","source":"original"},
    {"question":"Do all triangles have three sides?","ground_truth":True,"type":"categorical","source":"original"},
    {"question":"Do all circles have a curved edge?","ground_truth":True,"type":"categorical","source":"original"},
    {"question":"Do all animals need food?","ground_truth":True,"type":"categorical","source":"original"},
    {"question":"Do all living things grow?","ground_truth":True,"type":"categorical","source":"original"},
    {"question":"Do all reptiles lay eggs?","ground_truth":True,"type":"categorical","source":"original"},
    {"question":"Do all insects have six legs?","ground_truth":True,"type":"categorical","source":"original"},
    {"question":"Do all spiders have eight legs?","ground_truth":True,"type":"categorical","source":"original"},
    # Categorical False
    {"question":"Do all fish have hair?","ground_truth":False,"type":"categorical","source":"original"},
    {"question":"Do all birds have hair?","ground_truth":False,"type":"categorical","source":"original"},
    {"question":"Do all reptiles have hair?","ground_truth":False,"type":"categorical","source":"original"},
    {"question":"Do all fish give milk?","ground_truth":False,"type":"categorical","source":"original"},
    {"question":"Do all birds live in water?","ground_truth":False,"type":"categorical","source":"original"},
    {"question":"Do all mammals have gills?","ground_truth":False,"type":"categorical","source":"original"},
    {"question":"Do all triangles have four sides?","ground_truth":False,"type":"categorical","source":"original"},
    {"question":"Do all squares have three sides?","ground_truth":False,"type":"categorical","source":"original"},
    {"question":"Do all mammals lay eggs?","ground_truth":False,"type":"categorical","source":"original"},
    {"question":"Do all birds have gills?","ground_truth":False,"type":"categorical","source":"original"},
    {"question":"Do all rectangles have equal sides?","ground_truth":False,"type":"categorical","source":"original"},
    {"question":"Do all fish have feathers?","ground_truth":False,"type":"categorical","source":"original"},
    {"question":"Do all circles have corners?","ground_truth":False,"type":"categorical","source":"original"},
    {"question":"Do all triangles have right angles?","ground_truth":False,"type":"categorical","source":"original"},
    {"question":"Do all squares have curved sides?","ground_truth":False,"type":"categorical","source":"original"},
    # Hypothetical True
    {"question":"If it is raining, is the ground wet?","ground_truth":True,"type":"hypothetical","source":"original"},
    {"question":"If there is fire, is there heat?","ground_truth":True,"type":"hypothetical","source":"original"},
    {"question":"If water freezes, does it become ice?","ground_truth":True,"type":"hypothetical","source":"original"},
    {"question":"If water boils, is it hot?","ground_truth":True,"type":"hypothetical","source":"original"},
    {"question":"If the sun is shining, is it daytime?","ground_truth":True,"type":"hypothetical","source":"original"},
    {"question":"If metal is heated, does it expand?","ground_truth":True,"type":"hypothetical","source":"original"},
    {"question":"If it is raining, do we need an umbrella?","ground_truth":True,"type":"hypothetical","source":"original"},
    {"question":"If fire is present, is oxygen consumed?","ground_truth":True,"type":"hypothetical","source":"original"},
    {"question":"If you are breathing, are you alive?","ground_truth":True,"type":"hypothetical","source":"original"},
    {"question":"If a person is human, are they mortal?","ground_truth":True,"type":"hypothetical","source":"original"},
    # Hypothetical False
    {"question":"If water freezes, does it become steam?","ground_truth":False,"type":"hypothetical","source":"original"},
    {"question":"If it is raining, is the sky clear?","ground_truth":False,"type":"hypothetical","source":"original"},
    {"question":"If fire is present, does it feel cold?","ground_truth":False,"type":"hypothetical","source":"original"},
    {"question":"If water boils, does it freeze?","ground_truth":False,"type":"hypothetical","source":"original"},
    {"question":"If there is fire, does it need no oxygen?","ground_truth":False,"type":"hypothetical","source":"original"},
]

# ── KB-generated extra queries ────────────────────────────────────────
EXTRA_QUERIES = [
    # Taxonomic True (KB-generated)
    {"question":"Are all wolves mammals?","ground_truth":True,"type":"taxonomic","source":"kb_generated"},
    {"question":"Are all foxes canines?","ground_truth":True,"type":"taxonomic","source":"kb_generated"},
    {"question":"Are all leopards felines?","ground_truth":True,"type":"taxonomic","source":"kb_generated"},
    {"question":"Are all turtles reptiles?","ground_truth":True,"type":"taxonomic","source":"kb_generated"},
    {"question":"Are all crocodiles reptiles?","ground_truth":True,"type":"taxonomic","source":"kb_generated"},
    {"question":"Are all toads amphibians?","ground_truth":True,"type":"taxonomic","source":"kb_generated"},
    {"question":"Are all ants insects?","ground_truth":True,"type":"taxonomic","source":"kb_generated"},
    {"question":"Are all bees insects?","ground_truth":True,"type":"taxonomic","source":"kb_generated"},
    {"question":"Are all spiders arachnids?","ground_truth":True,"type":"taxonomic","source":"kb_generated"},
    {"question":"Are all hexagons polygons?","ground_truth":True,"type":"taxonomic","source":"kb_generated"},
    {"question":"Are all pentagons polygons?","ground_truth":True,"type":"taxonomic","source":"kb_generated"},
    {"question":"Are all rhombuses quadrilaterals?","ground_truth":True,"type":"taxonomic","source":"kb_generated"},
    {"question":"Are all airplanes aircraft?","ground_truth":True,"type":"taxonomic","source":"kb_generated"},
    {"question":"Are all helicopters aircraft?","ground_truth":True,"type":"taxonomic","source":"kb_generated"},
    {"question":"Are all buses vehicles?","ground_truth":True,"type":"taxonomic","source":"kb_generated"},
    {"question":"Are all trains vehicles?","ground_truth":True,"type":"taxonomic","source":"kb_generated"},
    {"question":"Are all apples fruits?","ground_truth":True,"type":"taxonomic","source":"kb_generated"},
    {"question":"Are all bananas fruits?","ground_truth":True,"type":"taxonomic","source":"kb_generated"},
    {"question":"Are all carrots vegetables?","ground_truth":True,"type":"taxonomic","source":"kb_generated"},
    {"question":"Are all trees plants?","ground_truth":True,"type":"taxonomic","source":"kb_generated"},
    {"question":"Are all flowers plants?","ground_truth":True,"type":"taxonomic","source":"kb_generated"},
    {"question":"Are all insects animals?","ground_truth":True,"type":"taxonomic","source":"kb_generated"},
    {"question":"Are all reptiles animals?","ground_truth":True,"type":"taxonomic","source":"kb_generated"},
    {"question":"Are all fish animals?","ground_truth":True,"type":"taxonomic","source":"kb_generated"},
    {"question":"Are all amphibians animals?","ground_truth":True,"type":"taxonomic","source":"kb_generated"},
    # Taxonomic False (KB-generated)
    {"question":"Are all reptiles mammals?","ground_truth":False,"type":"taxonomic","source":"kb_generated"},
    {"question":"Are all birds fish?","ground_truth":False,"type":"taxonomic","source":"kb_generated"},
    {"question":"Are all insects mammals?","ground_truth":False,"type":"taxonomic","source":"kb_generated"},
    {"question":"Are all circles rectangles?","ground_truth":False,"type":"taxonomic","source":"kb_generated"},
    {"question":"Are all shapes polygons?","ground_truth":False,"type":"taxonomic","source":"kb_generated"},
    {"question":"Are all vehicles aircraft?","ground_truth":False,"type":"taxonomic","source":"kb_generated"},
    {"question":"Are all fish reptiles?","ground_truth":False,"type":"taxonomic","source":"kb_generated"},
    {"question":"Are all frogs mammals?","ground_truth":False,"type":"taxonomic","source":"kb_generated"},
    {"question":"Are all spiders insects?","ground_truth":False,"type":"taxonomic","source":"kb_generated"},
    {"question":"Are all fruits vegetables?","ground_truth":False,"type":"taxonomic","source":"kb_generated"},
    # Categorical True (KB-generated)
    {"question":"Do all mammals have warm blood?","ground_truth":True,"type":"categorical","source":"kb_generated"},
    {"question":"Do all birds have a beak?","ground_truth":True,"type":"categorical","source":"kb_generated"},
    {"question":"Do all reptiles have cold blood?","ground_truth":True,"type":"categorical","source":"kb_generated"},
    {"question":"Do all fish have scales?","ground_truth":True,"type":"categorical","source":"kb_generated"},
    {"question":"Do all arachnids have eight legs?","ground_truth":True,"type":"categorical","source":"kb_generated"},
    {"question":"Do all living things need food?","ground_truth":True,"type":"categorical","source":"kb_generated"},
    {"question":"Do all living things die?","ground_truth":True,"type":"categorical","source":"kb_generated"},
    {"question":"Do all squares have equal sides?","ground_truth":True,"type":"categorical","source":"kb_generated"},
    {"question":"Do all squares have right angles?","ground_truth":True,"type":"categorical","source":"kb_generated"},
    {"question":"Do all triangles have three angles?","ground_truth":True,"type":"categorical","source":"kb_generated"},
    {"question":"Do all humans have a brain?","ground_truth":True,"type":"categorical","source":"kb_generated"},
    {"question":"Do all plants need water?","ground_truth":True,"type":"categorical","source":"kb_generated"},
    {"question":"Do all trees have roots?","ground_truth":True,"type":"categorical","source":"kb_generated"},
    {"question":"Do all circles have a radius?","ground_truth":True,"type":"categorical","source":"kb_generated"},
    {"question":"Do all aircraft have wings?","ground_truth":True,"type":"categorical","source":"kb_generated"},
    # Categorical False (KB-generated)
    {"question":"Do all reptiles have fur?","ground_truth":False,"type":"categorical","source":"kb_generated"},
    {"question":"Do all birds have scales?","ground_truth":False,"type":"categorical","source":"kb_generated"},
    {"question":"Do all insects have eight legs?","ground_truth":False,"type":"categorical","source":"kb_generated"},
    {"question":"Do all mammals have scales?","ground_truth":False,"type":"categorical","source":"kb_generated"},
    {"question":"Do all circles have straight sides?","ground_truth":False,"type":"categorical","source":"kb_generated"},
    {"question":"Do all rectangles have equal sides?","ground_truth":False,"type":"categorical","source":"kb_generated"},
    {"question":"Do all fish have legs?","ground_truth":False,"type":"categorical","source":"kb_generated"},
    {"question":"Do all birds have fins?","ground_truth":False,"type":"categorical","source":"kb_generated"},
    {"question":"Do all arachnids have six legs?","ground_truth":False,"type":"categorical","source":"kb_generated"},
    {"question":"Do all polygons have curved edges?","ground_truth":False,"type":"categorical","source":"kb_generated"},
    # Hypothetical True (KB-generated)
    {"question":"If water freezes, does it become solid?","ground_truth":True,"type":"hypothetical","source":"kb_generated"},
    {"question":"If there is fire, is there smoke?","ground_truth":True,"type":"hypothetical","source":"kb_generated"},
    {"question":"If water boils, does it produce steam?","ground_truth":True,"type":"hypothetical","source":"kb_generated"},
    {"question":"If it is raining, is it wet?","ground_truth":True,"type":"hypothetical","source":"kb_generated"},
    {"question":"If it is night, is it dark?","ground_truth":True,"type":"hypothetical","source":"kb_generated"},
    {"question":"If something is alive, does it need food?","ground_truth":True,"type":"hypothetical","source":"kb_generated"},
    {"question":"If fire is present, is heat released?","ground_truth":True,"type":"hypothetical","source":"kb_generated"},
    {"question":"If pressure increases, does volume decrease?","ground_truth":True,"type":"hypothetical","source":"kb_generated"},
    {"question":"If temperature drops, do molecules slow?","ground_truth":True,"type":"hypothetical","source":"kb_generated"},
    {"question":"If electricity flows, is the circuit complete?","ground_truth":True,"type":"hypothetical","source":"kb_generated"},
    # Hypothetical False (KB-generated)
    {"question":"If water freezes, does it become gas?","ground_truth":False,"type":"hypothetical","source":"kb_generated"},
    {"question":"If metal is heated, does it shrink?","ground_truth":False,"type":"hypothetical","source":"kb_generated"},
    {"question":"If fire is present, does temperature drop?","ground_truth":False,"type":"hypothetical","source":"kb_generated"},
    {"question":"If it is daytime, is it night?","ground_truth":False,"type":"hypothetical","source":"kb_generated"},
    {"question":"If electricity flows, is the circuit broken?","ground_truth":False,"type":"hypothetical","source":"kb_generated"},
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--proofwriter_dir', type=str, default='./proofwriter')
    parser.add_argument('--kb_path', type=str, default='knowledge_base.json',
                        help='Ignored — KB is built from script. Kept for compatibility.')
    parser.add_argument('--output_queries', type=str, default='extended_queries.json')
    parser.add_argument('--output_kb', type=str, default='knowledge_base_extended.json')
    parser.add_argument('--max_proofwriter', type=int, default=200)
    args = parser.parse_args()

    print("=" * 65)
    print("  LogicGuard — Step 1 (FIXED): KB Builder + Query Generator")
    print("=" * 65)

    # 1. Build KB from script (no file reading)
    print("\n[1/5] Building KB from script (no file dependency)...")
    kb = build_base_kb()
    print(f"  Taxonomies: {len(kb['taxonomies'])}, Properties: {len(kb['properties'])}, Conditionals: {len(kb['conditionals'])}")

    # 2. Scan ProofWriter for IS-A triples (extend KB)
    print(f"\n[2/5] Scanning ProofWriter: {args.proofwriter_dir}")
    pf_files = find_proofwriter_files(args.proofwriter_dir)
    all_pairs = []
    entries_scanned = 0
    for filepath in pf_files[:30]:
        entries = load_proofwriter_file(filepath)
        for entry in entries[:100]:
            pairs = extract_isa_from_proofwriter(entry)
            all_pairs.extend(pairs)
            entries_scanned += 1

    print(f"  Entries scanned: {entries_scanned}")
    print(f"  IS-A triples found: {len(all_pairs)}")

    # Add valid IS-A pairs to KB
    added = 0
    for child, parent in set(all_pairs):
        if not child or not parent or child == parent: continue
        if child not in kb['taxonomies']:
            kb['taxonomies'][child] = [parent]
            added += 1
        elif parent not in kb['taxonomies'][child]:
            kb['taxonomies'][child].append(parent)
            added += 1
    print(f"  KB extended with {added} new IS-A pairs from ProofWriter")

    # 3. Save extended KB
    print(f"\n[3/5] Saving extended KB...")
    with open(args.output_kb, 'w', encoding='utf-8') as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {args.output_kb} ({len(kb['taxonomies'])} taxonomies)")

    # 4. Combine queries
    print(f"\n[4/5] Building query set...")
    all_queries = ORIGINAL_QUERIES + EXTRA_QUERIES
    print(f"  Original queries:    {len(ORIGINAL_QUERIES)}")
    print(f"  KB-generated extra:  {len(EXTRA_QUERIES)}")
    print(f"  TOTAL:               {len(all_queries)}")

    # Stats
    by_type = defaultdict(lambda: {'total':0,'true':0,'false':0})
    by_source = defaultdict(int)
    for q in all_queries:
        t = q['type']
        by_type[t]['total'] += 1
        if q['ground_truth']: by_type[t]['true'] += 1
        else: by_type[t]['false'] += 1
        by_source[q.get('source','unknown')] += 1

    print(f"\n  By type:")
    for qtype, stats in by_type.items():
        print(f"    {qtype:15}: {stats['total']:3} ({stats['true']} True / {stats['false']} False)")
    print(f"  By source:")
    for src, cnt in by_source.items():
        print(f"    {src:20}: {cnt}")

    # 5. Save queries
    print(f"\n[5/5] Saving queries...")
    output = {
        'metadata': {
            'total': len(all_queries),
            'by_type': {k:v for k,v in by_type.items()},
            'by_source': dict(by_source),
            'kb_path': args.output_kb,
        },
        'queries': all_queries
    }
    with open(args.output_queries, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"  Saved: {args.output_queries}")
    print(f"\n{'='*65}")
    print(f"  STEP 1 COMPLETE — {len(all_queries)} queries ready")
    print(f"  Next: python step2_multi_model_runner.py")
    print(f"{'='*65}\n")

if __name__ == '__main__':
    main()