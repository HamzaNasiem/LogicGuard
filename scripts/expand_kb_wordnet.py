"""
expand_kb_wordnet.py
====================
Expands knowledge_base_extended.json from ~115 nodes to 1000+ nodes.
All existing nodes are preserved. Schema maintained exactly.
"""

import json, os, sys

try:
    import nltk
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    from nltk.corpus import wordnet as wn
    WORDNET_AVAILABLE = True
    print("[INFO] WordNet loaded.")
except Exception as e:
    WORDNET_AVAILABLE = False
    print(f"[WARN] WordNet not available: {e}")

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
KB_PATH = os.path.join(PROJECT_ROOT, "data", "knowledge_bases", "knowledge_base_extended.json")

with open(KB_PATH, "r", encoding="utf-8") as f:
    kb = json.load(f)

taxonomies   = kb["taxonomies"]
properties   = kb["properties"]
conditionals = kb["conditionals"]

print(f"[INFO] Existing taxonomy  : {len(taxonomies)}")
print(f"[INFO] Existing properties: {len(properties)}")

NEW_ENTRIES = []

def add(name, parents, props):
    key = name.lower().replace("_", " ")
    if key not in taxonomies:
        NEW_ENTRIES.append((key, parents, props))

# ── MAMMALS ──────────────────────────────────────────────────────────────────
M = ["warm_blooded","has_hair","gives_milk","has_backbone","vertebrate"]
for a in ["coyote","jackal","dingo","hyena","african wild dog"]:
    add(a,["canine","mammal","animal","living_thing"],M+["carnivore","has_four_legs"])
for a in ["cheetah","jaguar","cougar","lynx","bobcat","ocelot","serval","snow leopard","clouded leopard","caracal"]:
    add(a,["feline","mammal","animal","living_thing"],M+["carnivore","has_four_legs","has_claws"])
add("feline",["mammal","animal","living_thing"],M+["carnivore"])
add("canine",["mammal","animal","living_thing"],M+["carnivore"])
for a in ["chimpanzee","gorilla","orangutan","bonobo","gibbon","baboon","macaque","lemur","marmoset","capuchin monkey","howler monkey","spider monkey","squirrel monkey","mandrill","gelada","colobus monkey"]:
    add(a,["primate","mammal","animal","living_thing"],M+["has_hands","intelligent"])
for a in ["horse","donkey","zebra","rhinoceros","hippopotamus","giraffe","okapi","camel","llama","alpaca","bison","buffalo","yak","wildebeest","antelope","gazelle","impala","springbok","kudu","eland","oryx","gemsbok","moose","elk","reindeer","caribou","deer","roe deer","fallow deer","mule deer","muntjac","pig","wild boar","warthog","peccary","tapir"]:
    add(a,["ungulate","mammal","animal","living_thing"],M+["herbivore","has_hooves"])
add("ungulate",["mammal","animal","living_thing"],M+["has_hooves"])
for a in ["mouse","rat","squirrel","chipmunk","beaver","porcupine","hamster","gerbil","guinea pig","vole","mole","lemming","marmot","prairie dog","capybara","chinchilla","degu","gopher"]:
    add(a,["rodent","mammal","animal","living_thing"],M+["gnaws_teeth"])
add("rodent",["mammal","animal","living_thing"],M+["gnaws_teeth"])
for a in ["rabbit","hare","pika"]:
    add(a,["lagomorph","mammal","animal","living_thing"],M+["has_long_ears"])
add("lagomorph",["mammal","animal","living_thing"],M)
for a in ["blue whale","humpback whale","sperm whale","orca","beluga whale","narwhal","bottlenose dolphin","common dolphin","spinner dolphin","pilot whale"]:
    add(a,["cetacean","mammal","animal","living_thing"],M+["lives_in_water","breathes_air"])
add("cetacean",["mammal","animal","living_thing"],M+["lives_in_water"])
for a in ["seal","sea lion","walrus","fur seal","elephant seal"]:
    add(a,["pinniped","mammal","animal","living_thing"],M+["semi_aquatic","has_flippers"])
add("pinniped",["mammal","animal","living_thing"],M+["semi_aquatic"])
for a in ["kangaroo","koala","wombat","wallaby","tasmanian devil","quokka","possum","opossum","bandicoot","numbat"]:
    add(a,["marsupial","mammal","animal","living_thing"],M+["has_pouch"])
add("marsupial",["mammal","animal","living_thing"],M+["has_pouch"])
for a in ["platypus","echidna"]:
    add(a,["monotreme","mammal","animal","living_thing"],M+["lays_eggs"])
add("monotreme",["mammal","animal","living_thing"],M+["lays_eggs"])
for a in ["grizzly bear","polar bear","black bear","panda bear","sun bear","spectacled bear","sloth bear"]:
    add(a,["bear","mammal","animal","living_thing"],M+["omnivore","has_claws"])
add("bear",["mammal","animal","living_thing"],M+["omnivore"])
for a in ["otter","weasel","ferret","mink","badger","skunk","wolverine","marten","stoat","polecat"]:
    add(a,["mustelid","mammal","animal","living_thing"],M+["carnivore"])
add("mustelid",["mammal","animal","living_thing"],M)
for a in ["vampire bat","fruit bat","bumblebee bat","flying fox","horseshoe bat","long-eared bat"]:
    add(a,["bat","mammal","animal","living_thing"],M+["can_fly","nocturnal","uses_echolocation"])
add("bat",["mammal","animal","living_thing"],M+["can_fly","nocturnal"])
for a in ["hedgehog","shrew","tenrec"]:
    add(a,["insectivore","mammal","animal","living_thing"],M+["eats_insects"])
add("insectivore",["mammal","animal","living_thing"],M)
for a in ["elephant","african elephant","asian elephant"]:
    add(a,["proboscidea","mammal","animal","living_thing"],M+["herbivore","has_trunk"])
add("proboscidea",["mammal","animal","living_thing"],M)

# ── BIRDS ─────────────────────────────────────────────────────────────────────
B = ["has_feathers","has_wings","lays_eggs","has_beak","has_backbone","warm_blooded"]
for a in ["eagle","hawk","falcon","osprey","kite","harrier","vulture","condor","secretary bird","bald eagle","golden eagle","peregrine falcon","red-tailed hawk"]:
    add(a,["raptor","bird","animal","living_thing"],B+["carnivore","can_fly","sharp_talons"])
add("raptor",["bird","animal","living_thing"],B+["carnivore","sharp_talons"])
for a in ["barn owl","great horned owl","snowy owl","screech owl","barred owl","spotted owl"]:
    add(a,["owl","bird","animal","living_thing"],B+["nocturnal","silent_flight","carnivore"])
add("owl",["raptor","bird","animal","living_thing"],B+["nocturnal"])
for a in ["sparrow","finch","robin","bluebird","cardinal","canary","nightingale","wren","thrush","warbler","swallow","swift","martin","starling","mynah","mockingbird","catbird","jay","magpie","crow","raven","rook","jackdaw"]:
    add(a,["passerine","bird","animal","living_thing"],B+["can_fly","sings"])
add("passerine",["bird","animal","living_thing"],B+["can_fly"])
for a in ["duck","goose","swan","mallard","teal","pintail","shoveler","wigeon","scaup","eider"]:
    add(a,["waterfowl","bird","animal","living_thing"],B+["can_swim","waterproof_feathers"])
add("waterfowl",["bird","animal","living_thing"],B+["can_swim"])
for a in ["heron","egret","flamingo","stork","ibis","spoonbill","crane","sandpiper","plover","curlew"]:
    add(a,["wader","bird","animal","living_thing"],B+["long_legs","lives_near_water"])
add("wader",["bird","animal","living_thing"],B+["long_legs"])
for a in ["ostrich","emu","kiwi","rhea","cassowary","penguin"]:
    add(a,["ratite","bird","animal","living_thing"],B+["cannot_fly"])
add("ratite",["bird","animal","living_thing"],B+["cannot_fly"])
for a in ["parrot","macaw","cockatoo","cockatiel","budgerigar","lorikeet","lovebird","conure"]:
    add(a,["psittacine","bird","animal","living_thing"],B+["can_mimic","colorful"])
add("psittacine",["bird","animal","living_thing"],B+["can_mimic"])
for a in ["pigeon","dove","wood pigeon","mourning dove","turtle dove"]:
    add(a,["columbidae","bird","animal","living_thing"],B+["can_fly"])
add("columbidae",["bird","animal","living_thing"],B)
for a in ["woodpecker","toucan","hornbill","kingfisher","hummingbird"]:
    add(a,["bird","animal","living_thing"],B+["can_fly"])
for a in ["chicken","turkey","peacock","pheasant","quail","guinea fowl"]:
    add(a,["galliform","bird","animal","living_thing"],B+["ground_dwelling"])
add("galliform",["bird","animal","living_thing"],B)

# ── REPTILES ─────────────────────────────────────────────────────────────────
R = ["cold_blooded","has_scales","lays_eggs","has_backbone","ectotherm"]
for a in ["python","boa constrictor","cobra","rattlesnake","mamba","viper","anaconda","king snake","corn snake","water moccasin","garter snake","sea snake"]:
    add(a,["snake","reptile","animal","living_thing"],R+["no_legs","flexible_jaw"])
add("snake",["reptile","animal","living_thing"],R+["no_legs"])
for a in ["iguana","monitor lizard","komodo dragon","skink","gila monster","bearded dragon","blue-tongued lizard"]:
    add(a,["lizard","reptile","animal","living_thing"],R+["has_four_legs"])
add("lizard",["reptile","animal","living_thing"],R)
for a in ["sea turtle","box turtle","snapping turtle","leatherback turtle","tortoise","red-eared slider"]:
    add(a,["chelonian","reptile","animal","living_thing"],R+["has_shell"])
add("chelonian",["reptile","animal","living_thing"],R+["has_shell"])
for a in ["crocodile","caiman","gharial"]:
    add(a,["crocodilian","reptile","animal","living_thing"],R+["semi_aquatic"])
add("crocodilian",["reptile","animal","living_thing"],R+["semi_aquatic"])

# ── FISH ─────────────────────────────────────────────────────────────────────
F = ["has_gills","has_scales","lives_in_water","cold_blooded","has_backbone","ectotherm"]
for a in ["bass","perch","pike","catfish","carp","tilapia","swordfish","mackerel","anchovy","herring","sardine","halibut","flounder","sole","haddock","pollock","eel","pufferfish","seahorse","angelfish","discus","guppy","molly","platy","betta","cichlid","barracuda","mahi-mahi","snapper","grouper","sturgeon","paddlefish"]:
    add(a,["fish","animal","living_thing"],F)
for a in ["hammerhead shark","great white shark","whale shark","tiger shark","bull shark","ray","manta ray","stingray","skate","chimaera"]:
    add(a,["cartilaginous fish","fish","animal","living_thing"],F+["no_bones"])
add("cartilaginous fish",["fish","animal","living_thing"],F+["no_bones"])

# ── INSECTS ───────────────────────────────────────────────────────────────────
I = ["six_legs","three_body_segments","has_exoskeleton","invertebrate"]
for a in ["dragonfly","damselfly","cockroach","termite","grasshopper","cricket","locust","mantis","stick insect","leaf insect","beetle","ladybug","firefly","weevil","click beetle","wasp","hornet","bumblebee","leafcutter ant","fire ant","moth","monarch butterfly","swallowtail","skipper","flea","louse","aphid","cicada","earwig","silverfish"]:
    add(a,["insect","animal","living_thing"],I)

# ── ARACHNIDS & INVERTS ───────────────────────────────────────────────────────
A = ["eight_legs","has_exoskeleton","invertebrate"]
for a in ["tarantula","black widow","brown recluse","orb weaver","wolf spider","jumping spider","daddy longlegs","tick","mite","harvestman"]:
    add(a,["arachnid","animal","living_thing"],A)
for a in ["crab","lobster","shrimp","crayfish","barnacle","krill","woodlouse"]:
    add(a,["crustacean","animal","living_thing"],["has_exoskeleton","invertebrate","has_jointed_legs"])
add("crustacean",["animal","living_thing"],["has_exoskeleton","invertebrate"])
for a in ["octopus","squid","cuttlefish","nautilus","clam","oyster","mussel","snail","slug","scallop"]:
    add(a,["mollusc","animal","living_thing"],["soft_bodied","invertebrate"])
add("mollusc",["animal","living_thing"],["soft_bodied","invertebrate"])
for a in ["earthworm","leech","polychaete worm"]:
    add(a,["annelid","animal","living_thing"],["segmented_body","invertebrate"])
add("annelid",["animal","living_thing"],["segmented_body","invertebrate"])
for a in ["jellyfish","sea anemone","coral","hydra"]:
    add(a,["cnidarian","animal","living_thing"],["stinging_cells","invertebrate"])
add("cnidarian",["animal","living_thing"],["stinging_cells","invertebrate"])
for a in ["sea urchin","starfish","sea cucumber","brittle star"]:
    add(a,["echinoderm","animal","living_thing"],["radial_symmetry","invertebrate"])
add("echinoderm",["animal","living_thing"],["radial_symmetry","invertebrate"])

# ── PLANTS — Trees ────────────────────────────────────────────────────────────
TP = ["has_roots","has_trunk","has_bark","produces_oxygen","photosynthesis"]
for p in ["oak tree","maple tree","pine tree","birch tree","willow tree","cherry tree","apple tree","pear tree","fig tree","olive tree","cedar tree","cypress tree","spruce tree","fir tree","larch tree","eucalyptus","sequoia","redwood","baobab tree","magnolia tree","beech tree","ash tree","elm tree","walnut tree","chestnut tree","sycamore tree","poplar tree","acacia tree","mahogany tree","teak tree","ebony tree","palm tree","coconut palm","date palm","mangrove tree"]:
    add(p,["tree","plant","living_thing"],TP)
add("conifer",["tree","plant","living_thing"],TP+["has_needles","evergreen"])
add("deciduous tree",["tree","plant","living_thing"],TP+["loses_leaves"])
add("evergreen tree",["tree","plant","living_thing"],TP+["keeps_leaves"])

# ── PLANTS — Flowers ──────────────────────────────────────────────────────────
FP = ["has_petals","has_pollen","photosynthesis","produces_oxygen","needs_sunlight"]
for p in ["rose","tulip","daisy","sunflower","lily","orchid","lavender","jasmine","carnation","peony","iris","chrysanthemum","dahlia","marigold","pansy","violet","poppy","hibiscus","geranium","begonia","zinnia","foxglove","snapdragon","amaranth","lotus"]:
    add(p,["flower","plant","living_thing"],FP)

# ── PLANTS — Vegetables ───────────────────────────────────────────────────────
VP = ["edible","nutritious","grows_in_soil","photosynthesis"]
for p in ["broccoli","cauliflower","cabbage","brussels sprout","spinach","lettuce","kale","arugula","celery","asparagus","onion","garlic","leek","shallot","beet","turnip","parsnip","radish","sweet potato","pumpkin","zucchini","cucumber","eggplant","bell pepper","chili pepper","corn","pea","green bean","artichoke","fennel"]:
    add(p,["vegetable","food","plant","living_thing"],VP)
add("vegetable",["food","plant","living_thing"],VP)

# ── PLANTS — Fruits ───────────────────────────────────────────────────────────
FRP = ["edible","contains_seeds","sweet","nutritious"]
for p in ["pineapple","watermelon","cantaloupe","honeydew","papaya","guava","kiwi fruit","lychee","durian","jackfruit","passion fruit","dragon fruit","star fruit","persimmon","pomegranate","fig","date","plum","peach","apricot","cherry","pear","lemon","lime","grapefruit","tangerine","coconut","avocado","blueberry","raspberry"]:
    add(p,["fruit","food","plant","living_thing"],FRP)
add("fruit",["food","plant","living_thing"],FRP)

# ── PLANTS — Other ───────────────────────────────────────────────────────────
for p in ["fern","moss","lichen","algae","seaweed","cactus","succulent","bamboo","grass","wheat grass","rye grass","kelp","duckweed"]:
    add(p,["plant","living_thing"],["photosynthesis","produces_oxygen"])
for p in ["basil","mint","parsley","rosemary","thyme","oregano","cilantro","dill","sage","chives","tarragon","bay leaf"]:
    add(p,["herb","plant","living_thing"],["edible","aromatic","photosynthesis"])
add("herb",["plant","living_thing"],["edible","aromatic"])
add("shrub",["plant","living_thing"],["has_roots","has_branches","photosynthesis"])
add("vine",["plant","living_thing"],["climbs","photosynthesis"])
for p in ["wheat","rice","oat","barley","rye","sorghum","millet","quinoa","buckwheat"]:
    add(p,["grain","food","plant","living_thing"],["edible","contains_seeds","grows_in_soil"])
add("grain",["food","plant","living_thing"],["edible","contains_seeds"])

# ── FUNGI ─────────────────────────────────────────────────────────────────────
for p in ["mushroom","truffle","yeast","mold","mildew","shiitake mushroom","oyster mushroom","portobello mushroom","chanterelle","porcini"]:
    add(p,["fungus","living_thing"],["decomposes_matter","produces_spores"])

# ── VEHICLES — Cars ───────────────────────────────────────────────────────────
VV = ["has_wheels","transports_people","uses_fuel","has_engine"]
for v in ["sedan","coupe","hatchback","station wagon","convertible","suv","pickup truck","minivan","sports car","electric car","hybrid car","limousine","taxicab","police car","ambulance","fire truck","garbage truck","delivery truck","semi truck","dump truck","cement mixer","tow truck","forklift","go-kart","dune buggy"]:
    add(v,["car","vehicle"],VV)

# ── VEHICLES — Aircraft ───────────────────────────────────────────────────────
AV = ["can_fly","has_wings","uses_fuel"]
for v in ["jet","fighter jet","bomber","cargo plane","seaplane","glider","hang glider","paraglider","blimp","zeppelin","hot air balloon","drone","uav","ultralight aircraft","biplane","turboprop","regional jet","wide body jet","concorde","space shuttle"]:
    add(v,["aircraft","vehicle"],AV)

# ── VEHICLES — Watercraft ────────────────────────────────────────────────────
BV = ["travels_on_water","has_hull","transports_people"]
for v in ["sailboat","canoe","kayak","rowboat","speedboat","yacht","catamaran","trimaran","ferry","cruise ship","tanker","container ship","submarine","aircraft carrier","destroyer","battleship","icebreaker","hovercraft","jet ski","paddleboat"]:
    add(v,["boat","vehicle"],BV)

# ── VEHICLES — Rail & Other ───────────────────────────────────────────────────
for v in ["steam locomotive","diesel locomotive","electric train","subway","metro","tram","monorail","bullet train","freight train","passenger train"]:
    add(v,["train","vehicle"],["has_wheels","travels_on_rails","transports_people"])
for v in ["scooter","moped","atv","snowmobile","tractor","segway","skateboard","wheelchair"]:
    add(v,["vehicle"],["has_wheels","transports_people"])

# ── TOOLS ─────────────────────────────────────────────────────────────────────
HT = ["used_for_manual_work","handheld","solid_material"]
for t in ["hammer","screwdriver","wrench","pliers","chisel","saw","handsaw","hacksaw","drill","level","tape measure","ruler","square tool","punch","awl","file","rasp","clamp","vice","jack","crowbar","bolt cutter","wire stripper","utility knife","box cutter"]:
    add(t,["hand tool","tool"],HT)
add("hand tool",["tool"],HT)
add("tool",[],["used_to_perform_task","made_by_humans"])
for t in ["power drill","circular saw","jigsaw","sander","grinder","router","nail gun","heat gun","angle grinder","band saw"]:
    add(t,["power tool","tool"],["uses_electricity","performs_cutting_or_shaping"])
add("power tool",["tool"],["uses_electricity","performs_cutting_or_shaping"])

# ── KITCHEN TOOLS ────────────────────────────────────────────────────────────
KP = ["used_in_kitchen","used_for_cooking"]
for t in ["knife","spatula","ladle","whisk","tongs","peeler","grater","colander","strainer","rolling pin","cutting board","mixing bowl","measuring cup","measuring spoon","can opener","bottle opener","corkscrew","mortar and pestle","garlic press","zester","pastry brush","sieve","slotted spoon","serving spoon","skimmer"]:
    add(t,["kitchen tool","tool"],KP)
add("kitchen tool",["tool"],KP)
for a in ["refrigerator","microwave","oven","stove","dishwasher","blender","toaster","coffee maker","electric kettle","food processor","stand mixer","slow cooker","rice cooker","pressure cooker","air fryer","waffle maker"]:
    add(a,["kitchen appliance","appliance","electronics"],["uses_electricity","used_in_kitchen"])
add("kitchen appliance",["appliance","electronics"],["uses_electricity","used_in_kitchen"])

# ── FOODS ────────────────────────────────────────────────────────────────────
FO = ["edible","nutritious","consumed_by_humans"]
for f in ["beef","pork","lamb","chicken meat","turkey meat","veal","venison","duck meat","rabbit meat","bison meat","goat meat","bacon","ham","sausage","salami","pepperoni"]:
    add(f,["meat","food"],FO+["protein_rich"])
add("meat",["food"],FO+["protein_rich"])
for f in ["milk","cheese","butter","yogurt","cream","ice cream","sour cream","cream cheese","whey","ghee"]:
    add(f,["dairy","food"],FO+["contains_calcium"])
add("dairy",["food"],FO+["contains_calcium"])
for f in ["water","coffee","tea","juice","soda","beer","wine","whiskey","vodka","rum","gin","tequila","champagne","smoothie","milkshake","lemonade","hot chocolate","energy drink","sports drink","kombucha"]:
    add(f,["beverage","food"],FO+["liquid"])
add("beverage",["food"],FO+["liquid"])
for f in ["bread","pasta","rice dish","soup","salad","sandwich","pizza","burger","sushi","steak","cake","cookie","pie","chocolate","candy","chips","popcorn","cereal","granola"]:
    add(f,["prepared food","food"],FO)
add("prepared food",["food"],FO)

# ── PROFESSIONS ───────────────────────────────────────────────────────────────
PP = ["human_occupation","requires_training","provides_service"]
MP = PP+["works_in_healthcare","helps_patients"]
for p in ["doctor","surgeon","physician","pediatrician","cardiologist","neurologist","oncologist","radiologist","dermatologist","psychiatrist","psychologist","dentist","orthodontist","nurse","registered nurse","nurse practitioner","pharmacist","physiotherapist","occupational therapist","speech therapist","dietitian","paramedic","midwife","anesthesiologist","pathologist"]:
    add(p,["medical professional","professional","person"],MP)
add("medical professional",["professional","person"],MP)
for p in ["lawyer","attorney","barrister","solicitor","judge","magistrate","prosecutor","public defender","notary","paralegal","legal assistant","mediator","arbitrator","law clerk","court reporter"]:
    add(p,["legal professional","professional","person"],PP+["works_in_law"])
add("legal professional",["professional","person"],PP+["works_in_law"])
for p in ["engineer","software engineer","electrical engineer","mechanical engineer","civil engineer","chemical engineer","aerospace engineer","biomedical engineer","data scientist","computer scientist","mathematician","statistician","physicist","chemist","biologist","geologist","astronomer","ecologist","microbiologist","geneticist"]:
    add(p,["stem professional","professional","person"],PP+["uses_science"])
add("stem professional",["professional","person"],PP+["uses_science"])
for p in ["teacher","professor","lecturer","tutor","librarian","historian","archaeologist","linguist","philosopher","economist","sociologist","anthropologist","geographer","political scientist","journalist"]:
    add(p,["professional","person"],PP)
for p in ["artist","painter","sculptor","musician","singer","actor","dancer","writer","author","architect","chef","barber","plumber","electrician","carpenter"]:
    add(p,["professional","person"],PP)
add("professional",["person"],PP)
add("person",[],["is_human","has_consciousness","mortal"])

# ── ELECTRONICS ───────────────────────────────────────────────────────────────
EP = ["uses_electricity","electronic_device","made_by_humans"]
CP = EP+["processes_data","has_processor"]
for e in ["laptop","desktop computer","tablet computer","server","mainframe","supercomputer","workstation","chromebook","raspberry pi","minicomputer"]:
    add(e,["computer","electronics"],CP)
add("computer",["electronics"],CP)
for e in ["smartphone","mobile phone","feature phone","satellite phone","walkie talkie","pager"]:
    add(e,["phone","electronics"],EP+["enables_communication"])
add("phone",["electronics"],EP+["enables_communication"])
for e in ["television","radio","camera","digital camera","video camera","projector","monitor","headphones","earphones","speaker","amplifier","turntable","cd player","dvd player","blu-ray player","gaming console","game controller","smartwatch","e-reader","gps device","router","modem","printer","scanner","external hard drive"]:
    add(e,["electronics"],EP)
add("electronics",[],EP)
add("appliance",["electronics"],EP+["performs_household_task"])

# ── BUILDINGS ────────────────────────────────────────────────────────────────
BP = ["has_walls","has_roof","provides_shelter","made_by_humans"]
for b in ["house","apartment","mansion","bungalow","cottage","townhouse","condominium","villa","farmhouse","log cabin"]:
    add(b,["residential building","building","structure"],BP+["used_for_living"])
add("residential building",["building","structure"],BP+["used_for_living"])
for b in ["office building","skyscraper","factory","warehouse","hospital","school","university","library","museum","theater","cinema","stadium","arena","shopping mall","supermarket","restaurant","hotel","church","mosque","synagogue","temple","courthouse","prison","police station","fire station","airport","train station","bus terminal","power plant"]:
    add(b,["building","structure"],BP)
add("building",["structure"],BP)
add("structure",[],["made_by_humans","occupies_space"])

# ── ROOMS ────────────────────────────────────────────────────────────────────
RP = ["has_walls","enclosed_space","inside_building"]
for r in ["bedroom","kitchen room","bathroom","living room","dining room","study","office room","garage","attic","basement","hallway","corridor","lobby","balcony","porch"]:
    add(r,["room","space"],RP)
add("room",["space","structure"],RP)
add("space",[],["occupies_volume"])

# ── SHAPES — 3D & additional ──────────────────────────────────────────────────
for s in ["cube","sphere","cylinder","cone","pyramid","prism","torus","hemisphere","cuboid","tetrahedron","dodecahedron","icosahedron","octahedron"]:
    add(s,["3d shape","shape"],["three_dimensional"])
add("3d shape",["shape"],["three_dimensional"])
for s in ["parallelogram","trapezoid","kite shape","heptagon","octagon","nonagon","decagon"]:
    add(s,["polygon","shape"],["has_straight_sides","is_closed_shape"])
for s in ["arc","sector","segment shape"]:
    add(s,["curved shape","shape"],["curved"])
add("curved shape",["shape"],["curved"])

# ── GEOGRAPHY & NATURAL PHENOMENA ────────────────────────────────────────────
for n in ["mountain","hill","valley","canyon","cliff","plateau","plain","desert","forest","jungle","rainforest","savanna","tundra","wetland","marsh","swamp","lake","river","ocean","sea"]:
    add(n,["geographical feature","natural feature"],["exists_in_nature"])
add("geographical feature",["natural feature"],["exists_in_nature"])
add("natural feature",[],["exists_in_nature"])
for n in ["volcano","earthquake","tsunami","tornado","hurricane","blizzard","avalanche","flood","drought","lightning"]:
    add(n,["natural phenomenon"],[])
add("natural phenomenon",[],["occurs_naturally"])

# ── MATERIALS ────────────────────────────────────────────────────────────────
for m in ["iron","steel","aluminum","copper","silver","gold","platinum","titanium","bronze","brass"]:
    add(m,["metal","material"],["conducts_electricity","solid_at_room_temperature","metallic"])
add("metal",["material"],["conducts_electricity","metallic"])
for m in ["wood","glass","plastic","rubber","ceramic","concrete","brick","stone","marble","granite","paper","fabric","leather","silicon","carbon fiber"]:
    add(m,["material"],["solid","used_in_construction"])
add("material",[],["has_physical_properties","made_of_matter"])

# ── ABSTRACT / SCIENTIFIC CONCEPTS ───────────────────────────────────────────
for c in ["hypothesis","theory","law","principle","axiom","theorem","lemma","corollary","conjecture","proof"]:
    add(c,["scientific concept"],["abstract","knowledge_based"])
add("scientific concept",["concept"],["abstract"])
add("concept",[],["abstract"])

# ══════════════════════════════════════════════════════════════════════════════
#  WRITE INTO KB
# ══════════════════════════════════════════════════════════════════════════════
added = skipped = 0
for (name, parents, props) in NEW_ENTRIES:
    if name not in taxonomies:
        taxonomies[name] = parents if parents else []
        if props and name not in properties:
            properties[name] = props
        added += 1
    else:
        skipped += 1

print(f"[INFO] Entries added   : {added}")
print(f"[INFO] Entries skipped : {skipped}")

kb["taxonomies"]  = taxonomies
kb["properties"]  = properties

with open(KB_PATH, "w", encoding="utf-8") as f:
    json.dump(kb, f, indent=2, ensure_ascii=False)

print(f"[INFO] Saved: {KB_PATH}")

all_entities = set(taxonomies.keys()) | set(properties.keys())
print(f"\n{'='*60}")
print(f"  FINAL NODE COUNT : {len(all_entities)}")
print(f"  Taxonomy entries : {len(taxonomies)}")
print(f"  Property entries : {len(properties)}")
print(f"{'='*60}")
if len(all_entities) >= 1000:
    print("SUCCESS — 1000+ nodes achieved!")
else:
    print(f"WARNING — only {len(all_entities)} nodes, target not met")
