"""
Dataset Preparation Script for Stage 1 DeBERTa Fast Parser.

Generates 5,000 diverse synthetic and benchmark training pairs across 4 classes:
    0: taxonomic    (IS-A hierarchy / subtype relations)
    1: categorical  (entity-property attributions)
    2: hypothetical (conditional / modus ponens implications)
    3: non-logical  (open-domain QA, conversational, factual, math, non-syllogistic)

Formatted for HuggingFace `datasets` with JSONL outputs saved to:
    - data/training/stage1_train.jsonl
    - data/training/stage1_val.jsonl

Usage:
    python scripts/prepare_deberta_data.py [--total_samples 5000] [--train_ratio 0.8] [--seed 42]
"""

import argparse
import csv
import json
import logging
import os
import random
import re
from typing import List, Dict, Any, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Label mapping
LABEL_MAP = {
    0: "taxonomic",
    1: "categorical",
    2: "hypothetical",
    3: "non-logical",
}
NAME_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}


# ==============================================================================
# Domain Vocabularies & Templates
# ==============================================================================

TAXONOMIC_TEMPLATES = [
    "Are all {subject} {predicate}?",
    "Is a {subject_sing} a {predicate_sing}?",
    "Is an {subject_sing} an {predicate_sing}?",
    "Is a {subject_sing} an {predicate_sing}?",
    "Is an {subject_sing} a {predicate_sing}?",
    "Do all {subject} belong to {predicate}?",
    "Are {subject} classified as {predicate}?",
    "Every {subject_sing} is a {predicate_sing}?",
    "Every {subject_sing} is an {predicate_sing}?",
    "Is {subject_sing} a subtype of {predicate_sing}?",
    "Is {subject_sing} a subclass of {predicate_sing}?",
    "Is each {subject_sing} a type of {predicate_sing}?",
    "Can any {subject_sing} be considered a {predicate_sing}?",
    "Would an instance of {subject_sing} be considered a {predicate_sing}?",
    "Would any {subject_sing} fall under the category of {predicate}?",
    "Do {subject} fall into the category of {predicate}?",
    "Are {subject} considered to be {predicate}?",
    "Is every single {subject_sing} a member of {predicate}?",
]

CATEGORICAL_TEMPLATES = [
    "Do all {entity} have {property}?",
    "Does a {entity_sing} have {property}?",
    "Does an {entity_sing} have {property}?",
    "Do {entity} possess {property}?",
    "Does every {entity_sing} possess {property}?",
    "Do all {entity} exhibit {property}?",
    "Does a {entity_sing} feature {property}?",
    "Is {property} a property of all {entity}?",
    "Is {property} an inherent trait of {entity}?",
    "Are all {entity} characterized by {property}?",
    "Do all {entity} contain {property}?",
    "Does each {entity_sing} show signs of {property}?",
    "Are {entity} known to have {property}?",
    "Do {entity} produce {property}?",
    "Do all {entity} require {property}?",
]

HYPOTHETICAL_TEMPLATES = [
    "If {condition}, does {consequence}?",
    "If {condition}, then {consequence}?",
    "When {condition}, does {consequence}?",
    "When {condition}, will {consequence}?",
    "Assuming {condition}, does {consequence}?",
    "Assuming that {condition}, then {consequence}?",
    "Given that {condition}, will {consequence}?",
    "Given that {condition}, does it follow that {consequence}?",
    "If {condition}, is it true that {consequence}?",
    "If {condition}, will {consequence} happen?",
    "Provided that {condition}, does {consequence} occur?",
    "Suppose {condition}, does {consequence}?",
    "If {condition}, would {consequence} be expected?",
]

# Vocabulary pairs: (subject, predicate, subject_singular, predicate_singular)
TAXONOMIC_PAIRS = [
    # Biology / Zoology
    ("dogs", "canines", "dog", "canine"),
    ("cats", "felines", "cat", "feline"),
    ("lions", "carnivores", "lion", "carnivore"),
    ("tigers", "mammals", "tiger", "mammal"),
    ("whales", "mammals", "whale", "mammal"),
    ("dolphins", "cetaceans", "dolphin", "cetacean"),
    ("elephants", "vertebrates", "elephant", "vertebrate"),
    ("spiders", "arachnids", "spider", "arachnid"),
    ("scorpions", "arachnids", "scorpion", "arachnid"),
    ("ants", "insects", "ant", "insect"),
    ("bees", "insects", "bee", "insect"),
    ("butterflies", "arthropods", "butterfly", "arthropod"),
    ("eagles", "birds", "eagle", "bird"),
    ("penguins", "avians", "penguin", "avian"),
    ("frogs", "amphibians", "frog", "amphibian"),
    ("toads", "vertebrates", "toad", "vertebrate"),
    ("snakes", "reptiles", "snake", "reptile"),
    ("crocodiles", "reptiles", "crocodile", "reptile"),
    ("salmon", "fish", "salmon", "fish"),
    ("tuna", "chordates", "tuna", "chordate"),
    ("horses", "equines", "horse", "equine"),
    ("donkeys", "ungulates", "donkey", "ungulate"),
    ("wolves", "predators", "wolf", "predator"),
    ("bears", "mammals", "bear", "mammal"),
    ("monkeys", "primates", "monkey", "primate"),
    ("chimpanzees", "hominids", "chimpanzee", "hominid"),
    ("roses", "angiosperms", "rose", "angiosperm"),
    ("oaks", "deciduous_trees", "oak", "deciduous_tree"),
    ("pines", "gymnosperms", "pine", "gymnosperm"),
    ("ferns", "vascular_plants", "fern", "vascular_plant"),
    ("mosses", "bryophytes", "moss", "bryophyte"),
    ("mushrooms", "fungi", "mushroom", "fungus"),
    ("yeasts", "eukaryotes", "yeast", "eukaryote"),
    ("bacteria", "prokaryotes", "bacterium", "prokaryote"),
    ("corals", "cnidarians", "coral", "cnidarian"),
    ("jellyfish", "invertebrates", "jellyfish", "invertebrate"),
    ("octopuses", "cephalopods", "octopus", "cephalopod"),
    ("squids", "mollusks", "squid", "mollusk"),
    ("snails", "gastropods", "snail", "gastropod"),
    ("kangaroos", "marsupials", "kangaroo", "marsupial"),
    ("koalas", "mammals", "koala", "mammal"),
    ("platypuses", "monotremes", "platypus", "monotreme"),

    # Geometry & Mathematics
    ("squares", "rectangles", "square", "rectangle"),
    ("rectangles", "quadrilaterals", "rectangle", "quadrilateral"),
    ("rhombuses", "parallelograms", "rhombus", "parallelogram"),
    ("triangles", "polygons", "triangle", "polygon"),
    ("equilateral_triangles", "triangles", "equilateral_triangle", "triangle"),
    ("hexagons", "polygons", "hexagon", "polygon"),
    ("circles", "conic_sections", "circle", "conic_section"),
    ("ellipses", "geometric_curves", "ellipse", "geometric_curve"),
    ("cubes", "prisms", "cube", "prism"),
    ("spheres", "solids", "sphere", "solid"),
    ("tetrahedrons", "polyhedra", "tetrahedron", "polyhedron"),
    ("primes", "integers", "prime", "integer"),
    ("integers", "rational_numbers", "integer", "rational_number"),
    ("rationals", "real_numbers", "rational", "real_number"),
    ("matrices", "algebraic_structures", "matrix", "algebraic_structure"),

    # Medicine & Pharmacology
    ("aspirin", "analgesics", "aspirin", "analgesic"),
    ("ibuprofen", "nsaids", "ibuprofen", "nsaid"),
    ("paracetamol", "antipyretics", "paracetamol", "antipyretic"),
    ("amoxicillin", "antibiotics", "amoxicillin", "antibiotic"),
    ("penicillin", "beta_lactams", "penicillin", "beta_lactam"),
    ("atorvastatin", "statins", "atorvastatin", "statin"),
    ("metformin", "antidiabetics", "metformin", "antidiabetic"),
    ("lisinopril", "ace_inhibitors", "lisinopril", "ace_inhibitor"),
    ("metoprolol", "beta_blockers", "metoprolol", "beta_blocker"),
    ("morphine", "opioids", "morphine", "opioid"),
    ("influenza", "viral_infections", "influenza", "viral_infection"),
    ("tuberculosis", "bacterial_diseases", "tuberculosis", "bacterial_disease"),
    ("pneumonia", "respiratory_illnesses", "pneumonia", "respiratory_illness"),
    ("hypertension", "cardiovascular_disorders", "hypertension", "cardiovascular_disorder"),
    ("diabetes", "metabolic_syndromes", "diabetes", "metabolic_syndrome"),

    # Law & Jurisprudence
    ("manslaughter", "homicides", "manslaughter", "homicide"),
    ("murder", "felonies", "murder", "felony"),
    ("theft", "property_crimes", "theft", "property_crime"),
    ("burglary", "crimes", "burglary", "crime"),
    ("negligence", "torts", "negligence", "tort"),
    ("defamation", "civil_wrongs", "defamation", "civil_wrong"),
    ("copyright_infringement", "intellectual_property_violations", "copyright_infringement", "intellectual_property_violation"),
    ("leases", "contracts", "lease", "contract"),
    ("warranties", "covenants", "warranty", "covenant"),

    # Physics & Astronomy
    ("electrons", "leptons", "electron", "lepton"),
    ("quarks", "fermions", "quark", "fermion"),
    ("photons", "gauge_bosons", "photon", "gauge_boson"),
    ("neutrons", "baryons", "neutron", "baryon"),
    ("protons", "hadrons", "proton", "hadron"),
    ("planets", "celestial_bodies", "planet", "celestial_body"),
    ("supernovas", "astronomical_events", "supernova", "astronomical_event"),
    ("pulsars", "neutron_stars", "pulsar", "neutron_star"),
    ("quasars", "active_galactic_nuclei", "quasar", "active_galactic_nucleus"),

    # Computer Science
    ("binary_trees", "trees", "binary_tree", "tree"),
    ("trees", "acyclic_graphs", "tree", "acyclic_graph"),
    ("quicksort", "comparison_sorts", "quicksort", "comparison_sort"),
    ("heaps", "priority_queues", "heap", "priority_queue"),
    ("linked_lists", "linear_data_structures", "linked_list", "linear_data_structure"),
    ("compilers", "translators", "compiler", "translator"),
    ("linux", "operating_systems", "linux", "operating_system"),
]

CATEGORICAL_PAIRS = [
    # Biological Properties
    ("mammals", "hair", "mammal"),
    ("mammals", "mammary_glands", "mammal"),
    ("birds", "feathers", "bird"),
    ("birds", "beaks", "bird"),
    ("reptiles", "scales", "reptile"),
    ("fish", "gills", "fish"),
    ("fish", "fins", "fish"),
    ("amphibians", "permeable_skin", "amphibian"),
    ("insects", "six_legs", "insect"),
    ("insects", "exoskeletons", "insect"),
    ("arachnids", "eight_legs", "arachnid"),
    ("plants", "chlorophyll", "plant"),
    ("plants", "cell_walls", "plant"),
    ("vertebrates", "spinal_columns", "vertebrate"),
    ("eukaryotes", "membrane_bound_nuclei", "eukaryote"),
    ("carnivores", "sharp_canines", "carnivore"),
    ("herbivores", "flat_molars", "herbivore"),
    ("primates", "opposable_thumbs", "primate"),
    ("cetaceans", "blowholes", "cetacean"),
    ("dogs", "keen_sense_of_smell", "dog"),
    ("bats", "echolocation", "bat"),
    ("owls", "night_vision", "owl"),
    ("chameleons", "color_changing_cells", "chameleon"),
    ("electric_eels", "electrogenic_organs", "electric_eel"),

    # Material & Physical Properties
    ("metals", "electrical_conductivity", "metal"),
    ("metals", "thermal_conductivity", "metal"),
    ("metals", "metallic_luster", "metal"),
    ("liquids", "definite_volume", "liquid"),
    ("gases", "high_compressibility", "gas"),
    ("solids", "fixed_shape", "solid"),
    ("magnets", "magnetic_fields", "magnet"),
    ("crystals", "periodic_lattice_structures", "crystal"),
    ("acids", "sour_taste", "acid"),
    ("bases", "bitter_taste", "base"),

    # Medical & Chemical Attributes
    ("antibiotics", "antibacterial_action", "antibiotic"),
    ("vaccines", "antigenic_properties", "vaccine"),
    ("analgesics", "pain_relieving_properties", "analgesic"),
    ("antipyretics", "fever_reducing_action", "antipyretic"),
    ("statins", "cholesterol_lowering_effects", "statin"),
    ("nsaids", "anti_inflammatory_activity", "nsaid"),
    ("enzymes", "catalytic_activity", "enzyme"),
    ("hormones", "regulatory_functions", "hormone"),

    # Geometry & Mathematics
    ("triangles", "three_vertices", "triangle"),
    ("triangles", "interior_angles_summing_to_180", "triangle"),
    ("squares", "four_equal_sides", "square"),
    ("squares", "four_right_angles", "square"),
    ("circles", "constant_radius", "circle"),
    ("rectangles", "equal_opposite_sides", "rectangle"),
    ("rhombuses", "perpendicular_diagonals", "rhombus"),
    ("polygons", "closed_boundaries", "polygon"),

    # Legal & Computational
    ("contracts", "mutual_assent", "contract"),
    ("contracts", "legal_consideration", "contract"),
    ("statutes", "legislative_authority", "statute"),
    ("torts", "civil_liability", "tort"),
    ("algorithms", "deterministic_steps", "algorithm"),
    ("hash_tables", "constant_lookup_time", "hash_table"),
    ("binary_search_trees", "sorted_keys", "binary_search_tree"),
]

HYPOTHETICAL_PAIRS = [
    ("water is heated to 100 degrees Celsius", "it boils"),
    ("water is cooled below 0 degrees Celsius", "it freezes into ice"),
    ("an object is dropped in vacuum", "it accelerates due to gravity"),
    ("an electric circuit is closed", "current flows through the wire"),
    ("a light switch is flipped on", "the light bulb illuminates"),
    ("a plant is deprived of sunlight", "photosynthesis ceases"),
    ("chlorophyll absorbs red and blue light", "it reflects green light"),
    ("a metal rod is heated", "it expands in volume"),
    ("atmospheric pressure decreases", "the boiling point of water drops"),
    ("an acid reacts with a base", "salt and water are formed"),
    ("iron is exposed to oxygen and moisture", "rust develops on the surface"),
    ("a substance loses electrons", "oxidation occurs"),
    ("a substance gains electrons", "reduction occurs"),
    ("a patient takes a lethal dose of toxin", "organ failure results"),
    ("an antibiotic kills the bacterial infection", "the patient recovers"),
    ("a contract is breached by one party", "legal damages are enforceable"),
    ("a driver runs a red light", "a traffic violation is committed"),
    ("a triangle has three equal sides", "all its internal angles measure 60 degrees"),
    ("a number is divisible by both 2 and 3", "it is divisible by 6"),
    ("a prime number is greater than 2", "it is odd"),
    ("an algorithm runs in logarithmic time", "its execution time scales slowly"),
    ("a cache miss occurs", "data must be fetched from main memory"),
    ("the sun sets below the horizon", "darkness approaches"),
    ("heavy rain falls for hours", "local streams overflow"),
    ("ambient temperature rises significantly", "the glacier melts"),
    ("a balloon is filled with helium", "it floats in atmospheric air"),
    ("a resistor is placed in a circuit", "electrical resistance increases"),
    ("fuel is ignited inside the cylinder", "combustion drives the piston"),
    ("a magnetic field moves across a conductor", "an electric current is induced"),
    ("a sound wave enters a denser medium", "its speed increases"),
]

NON_LOGICAL_TEMPLATES = [
    "What is the capital of {place}?",
    "Who is the author of {book}?",
    "Can you summarize the main themes of {topic}?",
    "Write a short poem about {subject_word}.",
    "How do you cook a traditional {dish}?",
    "What year was the {event} founded?",
    "Explain the difference between {term1} and {term2} in simple terms.",
    "Tell me a fun joke about {subject_word}.",
    "What is the population of {city}?",
    "How does a {machine} work?",
    "What are the best tips for learning {skill}?",
    "Can you translate this sentence into {language}?",
    "Solve the equation {math_eq} for x.",
    "Write a Python function that {func_desc}.",
    "What did {person} achieve in {year}?",
    "Why is the sky blue during the daytime?",
    "What is the fastest land animal?",
    "How far is the Earth from the Moon?",
    "What is the best way to train for a marathon?",
    "Describe the scenery of {landscape}.",
]

NON_LOGICAL_FILLERS = {
    "place": ["France", "Japan", "Brazil", "Canada", "Germany", "Australia", "Egypt", "India", "Italy", "Norway"],
    "book": ["Hamlet", "1984", "Pride and Prejudice", "Moby Dick", "The Odyssey", "War and Peace", "The Great Gatsby"],
    "topic": ["existentialism", "the Renaissance", "macroeconomics", "machine learning", "impressionism", "space exploration"],
    "subject_word": ["autumn", "coffee", "mountains", "rain", "sunsets", "music", "friendship", "trees", "ocean"],
    "dish": ["lasagna", "sushi", "paella", "biryani", "apple pie", "tacos", "pad thai", "croissant"],
    "event": ["United Nations", "League of Nations", "Red Cross", "NASA", "European Union"],
    "term1": ["microeconomics", "classical physics", "supervised learning", "democracy", "prose"],
    "term2": ["macroeconomics", "quantum physics", "unsupervised learning", "republic", "poetry"],
    "city": ["Tokyo", "London", "Paris", "New York", "Cairo", "Sydney", "Berlin", "Toronto"],
    "machine": ["refrigerator", "jet engine", "microwave oven", "steam turbine", "bicycle"],
    "skill": ["guitar", "chess", "swimming", "public speaking", "baking", "programming"],
    "language": ["Spanish", "French", "German", "Japanese", "Arabic", "Italian"],
    "math_eq": ["3x + 12 = 27", "2x - 8 = 14", "5x + 10 = 50", "x^2 - 16 = 0", "4x + 7 = 31"],
    "func_desc": ["reverses a string", "calculates fibonacci numbers", "finds the maximum element in a list", "sorts an array with merge sort", "counts vowels in a string"],
    "person": ["Albert Einstein", "Marie Curie", "Isaac Newton", "Ada Lovelace", "Galileo Galilei"],
    "year": ["1905", "1911", "1687", "1843", "1610"],
    "landscape": ["the Grand Canyon", "the Swiss Alps", "the Sahara Desert", "the Amazon Rainforest"],
}


# ==============================================================================
# Helper Functions
# ==============================================================================

def load_kb_entities(kb_path: str) -> List[Tuple[str, str, str, str]]:
    """Extract taxonomic entity pairs from knowledge base JSON."""
    if not os.path.exists(kb_path):
        return []
    try:
        with open(kb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        pairs = []
        tax = data.get("taxonomies", {})
        for sub, preds in tax.items():
            for p in preds:
                s_sing = sub.replace("_", " ")
                p_sing = p.replace("_", " ")
                s_plur = s_sing + ("s" if not s_sing.endswith("s") else "")
                p_plur = p_sing + ("s" if not p_sing.endswith("s") else "")
                pairs.append((s_plur, p_plur, s_sing, p_sing))
        return pairs
    except Exception as e:
        logger.warning("Could not extract from KB at %s: %s", kb_path, e)
        return []


def load_kb_properties(kb_path: str) -> List[Tuple[str, str, str]]:
    """Extract categorical entity-property pairs from knowledge base JSON."""
    if not os.path.exists(kb_path):
        return []
    try:
        with open(kb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        pairs = []
        props = data.get("properties", {})
        for ent, plist in props.items():
            ent_clean = ent.replace("_", " ")
            ent_plur = ent_clean + ("s" if not ent_clean.endswith("s") else "")
            for pr in plist:
                pr_clean = pr.replace("_", " ")
                pairs.append((ent_plur, pr_clean, ent_clean))
        return pairs
    except Exception as e:
        logger.warning("Could not extract properties from KB at %s: %s", kb_path, e)
        return []


def load_truthfulqa(csv_path: str) -> List[str]:
    """Load open-domain / non-logical questions from TruthfulQA.csv."""
    if not os.path.exists(csv_path):
        return []
    questions = []
    try:
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                q = row.get("Question", "").strip()
                if q and len(q) > 8:
                    questions.append(q)
    except Exception as e:
        logger.warning("Could not load TruthfulQA: %s", e)
    return questions


def load_benchmark_queries(bench_path: str) -> List[Dict[str, Any]]:
    """Load benchmark queries if present."""
    if not os.path.exists(bench_path):
        return []
    try:
        with open(bench_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception as e:
        logger.warning("Could not load benchmark queries: %s", e)
    return []


# ==============================================================================
# Generator Functions
# ==============================================================================

def generate_taxonomic_samples(count: int, kb_pairs: List[Tuple[str, str, str, str]], rng: random.Random) -> List[Dict[str, Any]]:
    """Generate taxonomic training samples."""
    pool = list(TAXONOMIC_PAIRS) + kb_pairs
    samples = []
    seen = set()

    attempts = 0
    while len(samples) < count and attempts < count * 15:
        attempts += 1
        subj_pl, pred_pl, subj_sg, pred_sg = rng.choice(pool)
        tmpl = rng.choice(TAXONOMIC_TEMPLATES)

        # Singular/plural text formatting
        text = tmpl.format(
            subject=subj_pl,
            predicate=pred_pl,
            subject_sing=subj_sg,
            predicate_sing=pred_sg,
        )
        if text in seen:
            continue
        seen.add(text)

        norm_subj = subj_sg.lower().replace(" ", "_")
        norm_pred = pred_sg.lower().replace(" ", "_")

        samples.append({
            "text": text,
            "label": 0,
            "label_name": "taxonomic",
            "subject": norm_subj,
            "predicate": norm_pred,
            "condition": "",
            "consequence": "",
        })

    # If count not reached, cycle templates
    while len(samples) < count:
        subj_pl, pred_pl, subj_sg, pred_sg = rng.choice(pool)
        tmpl = rng.choice(TAXONOMIC_TEMPLATES)
        text = tmpl.format(
            subject=subj_pl,
            predicate=pred_pl,
            subject_sing=subj_sg,
            predicate_sing=pred_sg,
        )
        samples.append({
            "text": text,
            "label": 0,
            "label_name": "taxonomic",
            "subject": subj_sg.lower().replace(" ", "_"),
            "predicate": pred_sg.lower().replace(" ", "_"),
            "condition": "",
            "consequence": "",
        })

    return samples[:count]


def generate_categorical_samples(count: int, kb_props: List[Tuple[str, str, str]], rng: random.Random) -> List[Dict[str, Any]]:
    """Generate categorical training samples."""
    pool = list(CATEGORICAL_PAIRS) + kb_props
    samples = []
    seen = set()

    attempts = 0
    while len(samples) < count and attempts < count * 15:
        attempts += 1
        ent_pl, prop, ent_sg = rng.choice(pool)
        tmpl = rng.choice(CATEGORICAL_TEMPLATES)

        text = tmpl.format(
            entity=ent_pl,
            property=prop,
            entity_sing=ent_sg,
        )
        if text in seen:
            continue
        seen.add(text)

        norm_ent = ent_sg.lower().replace(" ", "_")
        norm_prop = prop.lower().replace(" ", "_")

        samples.append({
            "text": text,
            "label": 1,
            "label_name": "categorical",
            "subject": norm_ent,
            "predicate": norm_prop,
            "condition": "",
            "consequence": "",
        })

    while len(samples) < count:
        ent_pl, prop, ent_sg = rng.choice(pool)
        tmpl = rng.choice(CATEGORICAL_TEMPLATES)
        text = tmpl.format(
            entity=ent_pl,
            property=prop,
            entity_sing=ent_sg,
        )
        samples.append({
            "text": text,
            "label": 1,
            "label_name": "categorical",
            "subject": ent_sg.lower().replace(" ", "_"),
            "predicate": prop.lower().replace(" ", "_"),
            "condition": "",
            "consequence": "",
        })

    return samples[:count]


def generate_hypothetical_samples(count: int, rng: random.Random) -> List[Dict[str, Any]]:
    """Generate hypothetical training samples."""
    samples = []
    seen = set()

    attempts = 0
    while len(samples) < count and attempts < count * 15:
        attempts += 1
        cond, cons = rng.choice(HYPOTHETICAL_PAIRS)
        tmpl = rng.choice(HYPOTHETICAL_TEMPLATES)

        text = tmpl.format(condition=cond, consequence=cons)
        if text in seen:
            continue
        seen.add(text)

        norm_cond = cond.lower().replace(" ", "_")
        norm_cons = cons.lower().replace(" ", "_")

        samples.append({
            "text": text,
            "label": 2,
            "label_name": "hypothetical",
            "subject": "",
            "predicate": "",
            "condition": norm_cond,
            "consequence": norm_cons,
        })

    while len(samples) < count:
        cond, cons = rng.choice(HYPOTHETICAL_PAIRS)
        tmpl = rng.choice(HYPOTHETICAL_TEMPLATES)
        text = tmpl.format(condition=cond, consequence=cons)
        samples.append({
            "text": text,
            "label": 2,
            "label_name": "hypothetical",
            "subject": "",
            "predicate": "",
            "condition": cond.lower().replace(" ", "_"),
            "consequence": cons.lower().replace(" ", "_"),
        })

    return samples[:count]


def generate_non_logical_samples(count: int, truthful_qs: List[str], rng: random.Random) -> List[Dict[str, Any]]:
    """Generate non-logical training samples from TruthfulQA and synthesized non-syllogistic queries."""
    samples = []
    seen = set()

    # Add real TruthfulQA questions first
    for q in truthful_qs:
        if len(samples) >= count:
            break
        q_clean = q.strip()
        if q_clean not in seen:
            seen.add(q_clean)
            samples.append({
                "text": q_clean,
                "label": 3,
                "label_name": "non-logical",
                "subject": "",
                "predicate": "",
                "condition": "",
                "consequence": "",
            })

    # Fill remainder with synthesized open-domain / non-syllogistic queries
    attempts = 0
    while len(samples) < count and attempts < count * 20:
        attempts += 1
        tmpl = rng.choice(NON_LOGICAL_TEMPLATES)
        fmt_kwargs = {}
        for key, vals in NON_LOGICAL_FILLERS.items():
            if "{" + key + "}" in tmpl:
                fmt_kwargs[key] = rng.choice(vals)

        text = tmpl.format(**fmt_kwargs)
        if text not in seen:
            seen.add(text)
            samples.append({
                "text": text,
                "label": 3,
                "label_name": "non-logical",
                "subject": "",
                "predicate": "",
                "condition": "",
                "consequence": "",
            })

    while len(samples) < count:
        tmpl = rng.choice(NON_LOGICAL_TEMPLATES)
        fmt_kwargs = {k: rng.choice(v) for k, v in NON_LOGICAL_FILLERS.items() if "{" + k + "}" in tmpl}
        text = tmpl.format(**fmt_kwargs)
        samples.append({
            "text": text,
            "label": 3,
            "label_name": "non-logical",
            "subject": "",
            "predicate": "",
            "condition": "",
            "consequence": "",
        })

    return samples[:count]


# ==============================================================================
# Main Pipeline
# ==============================================================================

def prepare_stage1_dataset(
    output_dir: str = "data/training",
    total_samples: int = 5000,
    train_ratio: float = 0.8,
    seed: int = 42,
    kb_path: str = "data/knowledge_bases/knowledge_base_extended.json",
    truthfulqa_path: str = "TruthfulQA.csv",
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Generate balanced dataset across 4 classes and save to JSONL splits.

    Returns:
        (train_file_path, val_file_path, summary_statistics)
    """
    rng = random.Random(seed)
    os.makedirs(output_dir, exist_ok=True)

    samples_per_class = total_samples // 4
    logger.info("Generating %d samples per class across 4 classes (Total: %d)...", samples_per_class, total_samples)

    # Load resources
    kb_pairs = load_kb_entities(kb_path)
    kb_props = load_kb_properties(kb_path)
    truthful_qs = load_truthfulqa(truthfulqa_path)
    logger.info("Loaded %d KB taxonomy pairs, %d KB properties, %d TruthfulQA questions.", len(kb_pairs), len(kb_props), len(truthful_qs))

    # Generate samples per class
    tax_samples = generate_taxonomic_samples(samples_per_class, kb_pairs, rng)
    cat_samples = generate_categorical_samples(samples_per_class, kb_props, rng)
    hyp_samples = generate_hypothetical_samples(samples_per_class, rng)
    non_samples = generate_non_logical_samples(samples_per_class, truthful_qs, rng)

    # Perform stratified split per class
    train_samples: List[Dict[str, Any]] = []
    val_samples: List[Dict[str, Any]] = []

    for class_samples in [tax_samples, cat_samples, hyp_samples, non_samples]:
        rng.shuffle(class_samples)
        split_idx = int(len(class_samples) * train_ratio)
        train_samples.extend(class_samples[:split_idx])
        val_samples.extend(class_samples[split_idx:])

    # Shuffle the final splits
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)

    train_path = os.path.join(output_dir, "stage1_train.jsonl")
    val_path = os.path.join(output_dir, "stage1_val.jsonl")

    # Save to jsonl
    with open(train_path, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for s in val_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Calculate statistics
    def get_class_counts(data: List[Dict[str, Any]]) -> Dict[str, int]:
        counts = {name: 0 for name in LABEL_MAP.values()}
        for item in data:
            counts[item["label_name"]] += 1
        return counts

    train_counts = get_class_counts(train_samples)
    val_counts = get_class_counts(val_samples)

    stats = {
        "total_generated": len(train_samples) + len(val_samples),
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "train_distribution": train_counts,
        "val_distribution": val_counts,
        "train_file": train_path,
        "val_file": val_path,
        "train_size_kb": round(os.path.getsize(train_path) / 1024, 2),
        "val_size_kb": round(os.path.getsize(val_path) / 1024, 2),
    }

    logger.info("Dataset generated successfully:")
    logger.info("  Train samples: %d (%s)", stats["train_count"], stats["train_distribution"])
    logger.info("  Val samples:   %d (%s)", stats["val_count"], stats["val_distribution"])
    logger.info("  Files: %s (%s KB), %s (%s KB)", train_path, stats["train_size_kb"], val_path, stats["val_size_kb"])

    return train_path, val_path, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Stage 1 DeBERTa training dataset")
    parser.add_argument("--output_dir", type=str, default="data/training", help="Output directory for jsonl files")
    parser.add_argument("--total_samples", type=int, default=5000, help="Total number of dataset pairs")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train/val split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--kb_path", type=str, default="data/knowledge_bases/knowledge_base_extended.json")
    parser.add_argument("--truthfulqa_path", type=str, default="TruthfulQA.csv")
    args = parser.parse_args()

    prepare_stage1_dataset(
        output_dir=args.output_dir,
        total_samples=args.total_samples,
        train_ratio=args.train_ratio,
        seed=args.seed,
        kb_path=args.kb_path,
        truthfulqa_path=args.truthfulqa_path,
    )
