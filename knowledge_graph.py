# knowledge_graph.py

"""
LogicGuard — NetworkX Knowledge Graph
======================================
Replaces the hardcoded JSON KB with a proper directed semantic graph.
This enables full BFS/DFS transitive inference, making the system
academically robust for IEEE publication.

Graph Structure:
  - IS_A edges       : taxonomy (dog → mammal → animal → living_thing)
  - HAS_PROP edges   : properties (mammal → has_hair, bird → has_feathers)
  - IF_THEN edges    : conditionals (raining → ground_wet)

Paper claim: "We model our Knowledge Base as a directed semantic graph
using NetworkX, enabling polynomial-time transitive inference via
Breadth-First Search across biological, geometric, and causal domains."
"""

import networkx as nx
from typing import Optional, List, Tuple


class KnowledgeGraph:
    """
    Directed semantic graph for deterministic logical validation.
    Supports transitive IS-A inference and property inheritance.
    """

    def __init__(self):
        self.taxonomy   = nx.DiGraph()   # IS-A relationships
        self.properties = nx.DiGraph()   # entity → property
        self.conditionals = nx.DiGraph() # condition → consequence

        self._build_taxonomy()
        self._build_properties()
        self._build_conditionals()

    # ─────────────────────────────────────────────────────────────
    # TAXONOMY  (IS-A hierarchy)
    # ─────────────────────────────────────────────────────────────
    def _build_taxonomy(self):
        G = self.taxonomy
        edges = [
            # ── Mammals ──────────────────────────────────────────
            ("dog",       "canine"),
            ("dog",       "mammal"),
            ("wolf",      "canine"),
            ("wolf",      "mammal"),
            ("fox",       "canine"),
            ("fox",       "mammal"),
            ("cat",       "feline"),
            ("cat",       "mammal"),
            ("lion",      "feline"),
            ("lion",      "mammal"),
            ("tiger",     "feline"),
            ("tiger",     "mammal"),
            ("leopard",   "feline"),
            ("leopard",   "mammal"),
            ("whale",     "cetacean"),
            ("whale",     "mammal"),
            ("dolphin",   "cetacean"),
            ("dolphin",   "mammal"),
            ("porpoise",  "cetacean"),
            ("porpoise",  "mammal"),
            ("bat",       "mammal"),
            ("bear",      "mammal"),
            ("elephant",  "mammal"),
            ("horse",     "mammal"),
            ("cow",       "mammal"),
            ("pig",       "mammal"),
            ("rabbit",    "mammal"),
            ("rat",       "mammal"),
            ("mouse",     "mammal"),
            ("deer",      "mammal"),
            ("chimpanzee","mammal"),
            ("gorilla",   "mammal"),
            ("human",     "primate"),
            ("human",     "mammal"),
            ("primate",   "mammal"),
            ("canine",    "mammal"),
            ("feline",    "mammal"),
            ("cetacean",  "mammal"),
            ("mammal",    "animal"),
            # ── Birds ────────────────────────────────────────────
            ("sparrow",   "bird"),
            ("eagle",     "bird"),
            ("penguin",   "bird"),
            ("parrot",    "bird"),
            ("ostrich",   "bird"),
            ("hawk",      "bird"),
            ("owl",       "bird"),
            ("crow",      "bird"),
            ("robin",     "bird"),
            ("pigeon",    "bird"),
            ("duck",      "bird"),
            ("swan",      "bird"),
            ("flamingo",  "bird"),
            ("peacock",   "bird"),
            ("bird",      "animal"),
            # ── Reptiles ─────────────────────────────────────────
            ("snake",     "reptile"),
            ("lizard",    "reptile"),
            ("turtle",    "reptile"),
            ("tortoise",  "reptile"),
            ("crocodile", "reptile"),
            ("alligator", "reptile"),
            ("gecko",     "reptile"),
            ("chameleon", "reptile"),
            ("reptile",   "animal"),
            # ── Fish ─────────────────────────────────────────────
            ("shark",     "fish"),
            ("salmon",    "fish"),
            ("tuna",      "fish"),
            ("goldfish",  "fish"),
            ("trout",     "fish"),
            ("cod",       "fish"),
            ("clownfish", "fish"),
            ("fish",      "animal"),
            # ── Amphibians ───────────────────────────────────────
            ("frog",      "amphibian"),
            ("toad",      "amphibian"),
            ("salamander","amphibian"),
            ("newt",      "amphibian"),
            ("amphibian", "animal"),
            # ── Invertebrates ────────────────────────────────────
            ("ant",       "insect"),
            ("bee",       "insect"),
            ("butterfly", "insect"),
            ("mosquito",  "insect"),
            ("spider",    "arachnid"),
            ("scorpion",  "arachnid"),
            ("insect",    "animal"),
            ("arachnid",  "animal"),
            # ── Animal → Living thing ─────────────────────────────
            ("animal",    "living_thing"),
            ("plant",     "living_thing"),
            ("tree",      "plant"),
            ("flower",    "plant"),
            ("fungus",    "living_thing"),
            # ── Geometric shapes ─────────────────────────────────
            ("square",    "rectangle"),
            ("square",    "quadrilateral"),
            ("square",    "rhombus"),
            ("square",    "polygon"),
            ("square",    "shape"),
            ("rectangle", "quadrilateral"),
            ("rectangle", "polygon"),
            ("rectangle", "shape"),
            ("rhombus",   "quadrilateral"),
            ("rhombus",   "polygon"),
            ("rhombus",   "shape"),
            ("quadrilateral","polygon"),
            ("quadrilateral","shape"),
            ("triangle",  "polygon"),
            ("triangle",  "shape"),
            ("pentagon",  "polygon"),
            ("pentagon",  "shape"),
            ("hexagon",   "polygon"),
            ("hexagon",   "shape"),
            ("polygon",   "shape"),
            ("circle",    "shape"),
            ("oval",      "shape"),
            ("ellipse",   "shape"),
            # ── Vehicles ─────────────────────────────────────────
            ("car",       "vehicle"),
            ("bus",       "vehicle"),
            ("truck",     "vehicle"),
            ("motorcycle","vehicle"),
            ("bicycle",   "vehicle"),
            ("airplane",  "vehicle"),
            ("airplane",  "aircraft"),
            ("helicopter","vehicle"),
            ("helicopter","aircraft"),
            ("boat",      "vehicle"),
            ("ship",      "vehicle"),
            ("train",     "vehicle"),
            ("aircraft",  "vehicle"),
            # ── Food ─────────────────────────────────────────────
            ("apple",     "fruit"),
            ("banana",    "fruit"),
            ("orange",    "fruit"),
            ("mango",     "fruit"),
            ("grape",     "fruit"),
            ("strawberry","fruit"),
            ("carrot",    "vegetable"),
            ("potato",    "vegetable"),
            ("tomato",    "vegetable"),
            ("fruit",     "food"),
            ("vegetable", "food"),
        ]
        G.add_edges_from(edges)

    # ─────────────────────────────────────────────────────────────
    # PROPERTIES  (entity → property, inherited via taxonomy)
    # ─────────────────────────────────────────────────────────────
    def _build_properties(self):
        G = self.properties
        prop_edges = [
            # Mammal properties
            ("mammal",      "has_hair"),
            ("mammal",      "hair"),
            ("mammal",      "fur"),
            ("mammal",      "has_fur"),
            ("mammal",      "gives_milk"),
            ("mammal",      "give_milk"),
            ("mammal",      "milk"),
            ("mammal",      "warm_blooded"),
            ("mammal",      "has_backbone"),
            ("mammal",      "backbone"),
            ("mammal",      "spine"),
            ("mammal",      "has_spine"),
            ("mammal",      "vertebrate"),
            ("mammal",      "has_heart"),
            ("mammal",      "heart"),
            ("mammal",      "has_brain"),
            ("mammal",      "brain"),
            ("mammal",      "has_lungs"),
            ("mammal",      "lungs"),
            # Bird properties
            ("bird",        "has_feathers"),
            ("bird",        "feathers"),
            ("bird",        "has_wings"),
            ("bird",        "wings"),
            ("bird",        "lays_eggs"),
            ("bird",        "lay_eggs"),
            ("bird",        "eggs"),
            ("bird",        "egg"),
            ("bird",        "has_beak"),
            ("bird",        "beak"),
            ("bird",        "has_backbone"),
            ("bird",        "backbone"),
            # Fish properties
            ("fish",        "has_gills"),
            ("fish",        "gills"),
            ("fish",        "has_scales"),
            ("fish",        "scales"),
            ("fish",        "lives_in_water"),
            ("fish",        "cold_blooded"),
            ("fish",        "has_backbone"),
            ("fish",        "backbone"),
            # Reptile properties
            ("reptile",     "cold_blooded"),
            ("reptile",     "has_scales"),
            ("reptile",     "scales"),
            ("reptile",     "lays_eggs"),
            ("reptile",     "lay_eggs"),
            ("reptile",     "eggs"),
            ("reptile",     "egg"),
            ("reptile",     "has_backbone"),
            ("reptile",     "backbone"),
            # Insect properties
            ("insect",      "six_legs"),
            ("insect",      "6_legs"),
            ("insect",      "has_six_legs"),
            ("insect",      "three_body_segments"),
            # Arachnid properties
            ("arachnid",    "eight_legs"),
            ("arachnid",    "8_legs"),
            ("arachnid",    "has_eight_legs"),
            # Spider specifically
            ("spider",      "eight_legs"),
            ("spider",      "8_legs"),
            # Human properties (in addition to mammal inheritance)
            ("human",       "has_heart"),
            ("human",       "heart"),
            ("human",       "has_brain"),
            ("human",       "brain"),
            ("human",       "mortal"),
            ("human",       "can_think"),
            ("human",       "upright"),
            # Living thing properties
            ("living_thing","needs_water"),
            ("living_thing","water"),
            ("living_thing","needs_food"),
            ("living_thing","food"),
            ("living_thing","can_die"),
            ("living_thing","die"),
            ("living_thing","dies"),
            ("living_thing","mortal"),
            ("living_thing","grows"),
            ("living_thing","grow"),
            ("living_thing","reproduces"),
            # Animal properties (inherits from living_thing)
            ("animal",      "needs_food"),
            ("animal",      "food"),
            ("animal",      "needs_water"),
            ("animal",      "water"),
            ("animal",      "can_die"),
            # Plant properties
            ("plant",       "needs_water"),
            ("plant",       "water"),
            ("plant",       "needs_sunlight"),
            ("tree",        "has_roots"),
            ("tree",        "roots"),
            ("tree",        "root"),
            # Geometric shape properties
            ("square",      "four_sides"),
            ("square",      "4_sides"),
            ("square",      "equal_sides"),
            ("square",      "4_equal_sides"),
            ("square",      "four_right_angles"),
            ("square",      "4_right_angles"),
            ("square",      "right_angles"),
            ("rectangle",   "four_sides"),
            ("rectangle",   "4_sides"),
            ("rectangle",   "four_right_angles"),
            ("rectangle",   "4_right_angles"),
            ("rectangle",   "right_angles"),
            ("triangle",    "three_sides"),
            ("triangle",    "3_sides"),
            ("triangle",    "three_angles"),
            ("triangle",    "3_angles"),
            ("circle",      "no_corners"),
            ("circle",      "curved"),
            ("circle",      "round"),
            ("circle",      "has_radius"),
            ("circle",      "radius"),
            # Vehicle properties
            ("vehicle",     "wheels"),
            ("vehicle",     "has_wheels"),
            ("car",         "wheels"),
            ("car",         "has_wheels"),
            ("aircraft",    "wings"),
            ("aircraft",    "has_wings"),
            ("airplane",    "wings"),
            ("airplane",    "has_wings"),
        ]
        G.add_edges_from(prop_edges)

    # ─────────────────────────────────────────────────────────────
    # CONDITIONALS  (condition → consequence, Modus Ponens)
    # ─────────────────────────────────────────────────────────────
    def _build_conditionals(self):
        G = self.conditionals
        cond_edges = [
            # Weather / physical
            ("raining",                  "ground_wet"),
            ("raining",                  "wet"),
            ("raining",                  "need_umbrella"),
            ("raining",                  "sky_cloudy"),
            ("fire",                     "heat"),
            ("fire",                     "hot"),
            ("fire",                     "produces_heat"),
            ("fire",                     "dangerous"),
            ("fire",                     "light"),
            ("fire",                     "requires_oxygen"),
            ("fire",                     "smoke"),
            ("water freezes",            "ice"),
            ("water freezes",            "becomes_ice"),
            ("water freezes",            "solid"),
            ("water_freezes",            "ice"),
            ("water_freezes",            "solid"),
            ("water boils",              "hot"),
            ("water boils",              "steam"),
            ("water boils",              "100_degrees"),
            ("water_boils",              "hot"),
            ("water_boils",              "steam"),
            ("metal heated",             "expands"),
            ("metal is heated",          "expands"),
            ("metal_heated",             "expands"),
            ("sun shining",              "daytime"),
            ("sun shining",              "day"),
            ("sun_shining",              "daytime"),
            ("the sun is shining",       "daytime"),
            ("night",                    "dark"),
            ("it is night",              "dark"),
            ("darkness",                 "night"),
            # Biological / logical
            ("breathing",                "alive"),
            ("breathing",                "living"),
            ("you are breathing",        "alive"),
            ("alive",                    "needs_food"),
            ("alive",                    "food"),
            ("something is alive",       "needs_food"),
            ("something is alive",       "food"),
            ("person is human",          "mortal"),
            ("a person is human",        "mortal"),
            ("human",                    "mortal"),
            ("human",                    "can_think"),
            ("living_thing",             "needs_food"),
            ("living_thing",             "can_die"),
            # Metal / sun extras
            ("metal is heated",          "expand"),
            ("metal heated",             "expand"),
            ("metal_heated",             "expand"),
            ("sun is shining",           "daytime"),
            ("sun is shining",           "day"),
            ("sun shining",              "daytime"),
            ("sun shining",              "day"),
        ]
        G.add_edges_from(cond_edges)

    # ─────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────

    def is_subset(self, child: str, parent: str) -> Tuple[bool, List[str]]:
        """
        Check child IS-A parent via BFS graph traversal.
        Returns (result, proof_path).
        """
        child  = child.lower().strip()
        parent = parent.lower().strip()

        if child == parent:
            return False, []   # trivially same — not interesting

        if not self.taxonomy.has_node(child):
            return False, []

        try:
            path = nx.shortest_path(self.taxonomy, child, parent)
            return True, path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return False, []

    def has_property(self, entity: str, prop: str) -> Tuple[bool, str]:
        """
        Check if entity or any of its taxonomic ancestors have the property.
        Returns (result, proof_string).
        """
        entity = entity.lower().strip().replace(' ', '_')
        prop   = prop.lower().strip().replace(' ', '_')

        # Check entity itself
        if self.properties.has_edge(entity, prop):
            return True, f"{entity} → {prop}"

        # Check via taxonomy ancestors (property inheritance)
        if self.taxonomy.has_node(entity):
            ancestors = nx.descendants(self.taxonomy, entity)
            for ancestor in ancestors:
                if self.properties.has_edge(ancestor, prop):
                    return True, f"{entity} → ... → {ancestor} → {prop}"

        return False, f"{entity} does not have {prop}"

    def check_conditional(self, condition: str, consequence: str) -> Tuple[bool, str]:
        """
        Check IF condition THEN consequence via conditional graph.
        Returns (result, proof_string).
        """
        condition   = condition.lower().strip()
        consequence = consequence.lower().strip()

        # Build search variants
        cond_variants = {
            condition,
            condition.replace(' ', '_'),
            condition.replace('the ', '').strip(),
        }
        cons_variants = {
            consequence,
            consequence.replace(' ', '_'),
            consequence.split()[-1] if consequence else '',
            consequence.split()[0] if consequence else '',
        }
        cons_variants.discard('')

        for cv in cond_variants:
            if self.conditionals.has_node(cv):
                for cons in cons_variants:
                    if self.conditionals.has_edge(cv, cons):
                        return True, f"Modus Ponens: {cv} → {cons}"
        return False, f"Conditional unverified: {condition} → {consequence}"

    # ─────────────────────────────────────────────────────────────
    # GRAPH STATISTICS (for paper reporting)
    # ─────────────────────────────────────────────────────────────
    def stats(self) -> dict:
        return {
            "taxonomy_nodes":      self.taxonomy.number_of_nodes(),
            "taxonomy_edges":      self.taxonomy.number_of_edges(),
            "property_nodes":      self.properties.number_of_nodes(),
            "property_edges":      self.properties.number_of_edges(),
            "conditional_nodes":   self.conditionals.number_of_nodes(),
            "conditional_edges":   self.conditionals.number_of_edges(),
        }