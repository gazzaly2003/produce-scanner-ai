"""Produce knowledge base.

Nutrition: USDA FoodData Central reference values, RAW edible portion, per 100 g (rounded).
Real values vary by variety, growing conditions and ripeness - treat as good estimates,
not lab measurements.  Daily Values (DV) are the FDA adult reference values.
"""

STAGES = ["unripe", "ripe", "overripe", "rotten"]

# key -> (label, unit, FDA daily value)
DAILY_VALUES = {
    "vitamin_c":  ("Vitamin C", "mg", 90),
    "vitamin_a":  ("Vitamin A", "µg RAE", 900),
    "vitamin_k":  ("Vitamin K", "µg", 120),
    "vitamin_e":  ("Vitamin E", "mg", 15),
    "vitamin_b1": ("Vitamin B1 (thiamin)", "mg", 1.2),
    "vitamin_b2": ("Vitamin B2 (riboflavin)", "mg", 1.3),
    "vitamin_b3": ("Vitamin B3 (niacin)", "mg", 16),
    "vitamin_b5": ("Vitamin B5 (pantothenic)", "mg", 5),
    "vitamin_b6": ("Vitamin B6", "mg", 1.7),
    "folate":     ("Folate (B9)", "µg DFE", 400),
    "potassium":  ("Potassium", "mg", 4700),
    "magnesium":  ("Magnesium", "mg", 420),
    "calcium":    ("Calcium", "mg", 1300),
    "iron":       ("Iron", "mg", 18),
    "manganese":  ("Manganese", "mg", 2.3),
}
VITAMIN_KEYS = [k for k in DAILY_VALUES if k.startswith("vitamin_") or k == "folate"]
MINERAL_KEYS = [k for k in DAILY_VALUES if k not in VITAMIN_KEYS]

PRODUCE = {}


# ---- color-analysis profiles (see vision_cv.assess) ----
def ripener(fresh, unripe_green, ripe_min, rot_at, brown_max=14):
    return dict(type="ripener", fresh=fresh, unripe_green=unripe_green, ripe_min=ripe_min, rot_at=rot_at, brown_max=brown_max)

def freshness(fresh, rot_at, pale_at=0):
    return dict(type="freshness", fresh=fresh, rot_at=rot_at, pale_at=pale_at)

def spot(weights, rot_at, aging_at, green_at=0, min_green=0):
    return dict(type="spot", weights=weights, rot_at=rot_at, aging_at=aging_at, green_at=green_at, min_green=min_green)

def yellowing(weights, rot_at, yellow_at):
    return dict(type="yellowing", weights=weights, rot_at=rot_at, yellow_at=yellow_at)

def unjudgeable(weights, rot_at):
    return dict(type="unjudgeable", weights=weights, rot_at=rot_at)


def add(key, name, category, ripens, has_unripe, serving_g, serving_label, cv, n, tips, note=""):
    PRODUCE[key] = dict(key=key, name=name, category=category, ripens=ripens, has_unripe=has_unripe,
                        serving_g=serving_g, serving_label=serving_label, cv=cv, n=n, tips=tips, note=note)


# ============================ FRUIT ============================
add("banana", "Banana", "fruit", True, True, 118, "1 medium",
    ripener(["yellow", "orange"], 45, 40, 38),
    dict(kcal=89, carbs=22.8, fiber=2.6, sugar=12.2, protein=1.1, fat=0.3, vitamin_c=8.7, vitamin_b6=0.367, folate=20,
         vitamin_a=3, vitamin_k=0.5, vitamin_e=0.1, potassium=358, magnesium=27, manganese=0.27, calcium=5, iron=0.26),
    ["Firm, unblemished skin; dull grayish skin can mean cold damage.", "A few brown flecks on yellow skin = peak sweetness, not spoilage.", "Avoid bunches with a fermented smell."])
add("tomato", "Tomato", "fruit", True, True, 123, "1 medium",
    ripener(["red"], 50, 52, 28, 12),
    dict(kcal=18, carbs=3.9, fiber=1.2, sugar=2.6, protein=0.9, fat=0.2, vitamin_c=13.7, vitamin_a=42, vitamin_k=7.9, vitamin_e=0.54,
         vitamin_b6=0.08, folate=15, potassium=237, magnesium=11, calcium=10, iron=0.27),
    ["Heavy for its size = juicy.", "Smell the stem end; fragrant means flavorful.", "Skip wrinkled shoulders or cracked skin."])
add("mango", "Mango", "fruit", True, True, 165, "1 cup sliced",
    ripener(["yellow", "orange", "red"], 50, 38, 32, 16),
    dict(kcal=60, carbs=15, fiber=1.6, sugar=13.7, protein=0.8, fat=0.4, vitamin_c=36.4, vitamin_a=54, vitamin_k=4.2, vitamin_e=0.9,
         vitamin_b6=0.119, folate=43, potassium=168, magnesium=10, calcium=11, iron=0.16),
    ["Judge by smell and gentle squeeze, not just color - some varieties stay green when ripe.", "Ripe mango smells sweet at the stem.", "Avoid a sour or alcoholic smell."])
add("avocado", "Avocado", "fruit", True, True, 68, "½ avocado",
    dict(type="avocado"),
    dict(kcal=160, carbs=8.5, fiber=6.7, sugar=0.7, protein=2, fat=14.7, vitamin_c=10, vitamin_a=7, vitamin_k=21, vitamin_e=2.07,
         vitamin_b2=0.13, vitamin_b3=1.738, vitamin_b5=1.389, vitamin_b6=0.257, folate=81, potassium=485, magnesium=29, calcium=12, iron=0.55),
    ["Squeeze gently in your palm, not with fingertips (they bruise it).", "Flick off the small stem cap: green underneath = ripe, brown = overripe.", "Ripe gives slightly all over without feeling mushy."])
add("peach", "Peach / Nectarine", "fruit", True, True, 150, "1 medium",
    ripener(["yellow", "orange", "red"], 45, 42, 30, 14),
    dict(kcal=39, carbs=9.5, fiber=1.5, sugar=8.4, protein=0.9, fat=0.3, vitamin_c=6.6, vitamin_a=16, vitamin_k=2.6, vitamin_e=0.73,
         vitamin_b3=0.806, vitamin_b6=0.025, folate=4, potassium=190, magnesium=9, calcium=6, iron=0.25),
    ["Smell the stem end - sweet aroma is the best ripeness signal.", "Ripe peaches give slightly under gentle pressure.", "A green tint under the blush means picked too early."])
add("pear", "Pear", "fruit", True, True, 178, "1 medium",
    ripener(["yellow", "orange"], 55, 35, 26, 14),
    dict(kcal=57, carbs=15.2, fiber=3.1, sugar=9.8, protein=0.4, fat=0.1, vitamin_c=4.3, vitamin_a=1, vitamin_k=4.4, vitamin_e=0.12,
         vitamin_b6=0.029, folate=7, potassium=116, magnesium=7, calcium=9, iron=0.18),
    ["Press gently near the stem - that's where pears ripen first.", "Most are sold hard on purpose; ripen at home.", "Skip pears soft at the base but hard at the neck."])
add("papaya", "Papaya", "fruit", True, True, 140, "1 cup cubed",
    ripener(["yellow", "orange"], 55, 40, 30, 16),
    dict(kcal=43, carbs=10.8, fiber=1.7, sugar=7.8, protein=0.5, fat=0.3, vitamin_c=60.9, vitamin_a=47, vitamin_k=2.6, vitamin_e=0.3,
         vitamin_b6=0.038, folate=37, potassium=182, magnesium=21, calcium=20, iron=0.25),
    ["Mostly yellow-orange skin, not green.", "Should yield to gentle pressure like a ripe avocado.", "A sweet smell at the stem end is a good sign."])
add("pineapple", "Pineapple", "fruit", True, True, 165, "1 cup chunks",
    ripener(["yellow", "orange"], 60, 35, 30, 18),
    dict(kcal=50, carbs=13.1, fiber=1.4, sugar=9.9, protein=0.5, fat=0.1, vitamin_c=47.8, vitamin_a=3, vitamin_k=0.7, vitamin_b1=0.079,
         vitamin_b6=0.112, folate=18, potassium=109, magnesium=12, calcium=13, iron=0.29, manganese=0.927),
    ["Smell the base - sweet means ripe; no smell means it won't sweeten.", "A leaf should pull from the crown with light resistance.", "Smell and leaf-pull matter more than color."],
    "Pineapples barely sweeten after picking - a green one stays tart.")
add("kiwi", "Kiwi", "fruit", True, True, 69, "1 kiwi",
    unjudgeable(dict(dark=0.7, pale=0.8), 24),
    dict(kcal=61, carbs=14.7, fiber=3.0, sugar=9.0, protein=1.1, fat=0.5, vitamin_c=92.7, vitamin_a=4, vitamin_k=40.3, vitamin_e=1.46,
         vitamin_b6=0.063, folate=25, potassium=312, magnesium=17, calcium=34, iron=0.31),
    ["Gentle squeeze - ripe kiwi yields slightly like a ripe plum.", "Wrinkled skin means overripe or drying.", "Firm kiwi ripens at room temperature in days."],
    "Kiwi skin is naturally brown and fuzzy, so color says little - use a gentle squeeze.")
add("apple", "Apple", "fruit", False, False, 182, "1 medium",
    spot(None, 26, 10),
    dict(kcal=52, carbs=13.8, fiber=2.4, sugar=10.4, protein=0.3, fat=0.2, vitamin_c=4.6, vitamin_a=3, vitamin_k=2.2, vitamin_e=0.18,
         vitamin_b6=0.041, folate=3, potassium=107, magnesium=5, calcium=6, iron=0.12),
    ["Firmness matters more than color for most varieties.", "Avoid soft spots or a wine-like smell.", "A faint waxy bloom is natural."],
    "Apple skin color varies hugely by variety - firmness is the best final check.")
add("citrus", "Orange", "fruit", False, True, 131, "1 medium",
    freshness(["orange", "yellow"], 24, 20),
    dict(kcal=47, carbs=11.8, fiber=2.4, sugar=9.4, protein=0.9, fat=0.1, vitamin_c=53.2, vitamin_a=11, vitamin_b1=0.087,
         vitamin_b6=0.06, folate=30, potassium=181, magnesium=10, calcium=40, iron=0.1),
    ["Pick the heaviest for its size - more juice.", "Firm, fine-textured skin usually means thinner rind.", "Green patches on ripe citrus are normal."],
    "Citrus doesn't color-ripen off the tree; greenish skin isn't necessarily unripe.")
add("lemon", "Lemon", "fruit", False, True, 58, "1 lemon",
    freshness(["yellow"], 24, 20),
    dict(kcal=29, carbs=9.3, fiber=2.8, sugar=2.5, protein=1.1, fat=0.3, vitamin_c=53, vitamin_a=1, vitamin_e=0.15, vitamin_b6=0.08,
         folate=11, potassium=138, magnesium=8, calcium=26, iron=0.6),
    ["Heavy for its size, thin smooth skin = juicier.", "Firm with a little give; hard and shiny may be dry inside.", "Avoid dull, shriveled or soft spots."])
add("strawberry", "Strawberry", "fruit", False, True, 152, "1 cup",
    freshness(["red"], 20, 35),
    dict(kcal=32, carbs=7.7, fiber=2.0, sugar=4.9, protein=0.7, fat=0.3, vitamin_c=58.8, vitamin_a=1, vitamin_k=2.2, vitamin_e=0.29,
         vitamin_b6=0.047, folate=24, potassium=153, magnesium=13, calcium=16, iron=0.41, manganese=0.386),
    ["Choose fully red - they won't ripen after picking.", "Check the container bottom for juice stains (crushed/moldy berries).", "A bright green cap means recently picked."])
add("grape", "Grapes", "fruit", False, False, 151, "1 cup",
    freshness(["purple", "green", "red"], 18, 0),
    dict(kcal=69, carbs=18.1, fiber=0.9, sugar=15.5, protein=0.7, fat=0.2, vitamin_c=3.2, vitamin_a=3, vitamin_k=14.6, vitamin_e=0.19,
         vitamin_b1=0.069, vitamin_b6=0.086, folate=2, potassium=191, magnesium=7, calcium=10, iron=0.36),
    ["Plump grapes firmly attached to green, flexible stems.", "A light white bloom is natural.", "Lots of loose grapes at the bag bottom = aging."])
add("blueberry", "Blueberries", "fruit", False, False, 148, "1 cup",
    spot(dict(dark=0.9, brown=0.6, pale=0.3), 20, 8),
    dict(kcal=57, carbs=14.5, fiber=2.4, sugar=10.0, protein=0.7, fat=0.3, vitamin_c=9.7, vitamin_a=3, vitamin_k=19.3, vitamin_e=0.57,
         vitamin_b6=0.052, folate=6, potassium=77, magnesium=6, calcium=6, iron=0.28, manganese=0.336),
    ["Deep blue/purple with a dusty white bloom.", "Avoid shriveled, leaking or moldy berries in the pack.", "Shake the container - berries should move freely."])
add("cherry", "Cherries", "fruit", False, False, 138, "1 cup",
    freshness(["red", "purple"], 20, 0),
    dict(kcal=63, carbs=16.0, fiber=2.1, sugar=12.8, protein=1.1, fat=0.2, vitamin_c=7.0, vitamin_a=3, vitamin_k=2.1, vitamin_e=0.07,
         vitamin_b6=0.049, folate=4, potassium=222, magnesium=11, calcium=13, iron=0.36),
    ["Glossy, plump, deeply colored with green stems.", "Skip cherries with brown, dry stems.", "Avoid soft, leaking or split fruit."])
add("watermelon", "Watermelon", "fruit", False, False, 152, "1 cup diced",
    unjudgeable(dict(dark=0.9, pale=0.7), 22),
    dict(kcal=30, carbs=7.6, fiber=0.4, sugar=6.2, protein=0.6, fat=0.2, vitamin_c=8.1, vitamin_a=28, vitamin_k=0.1, vitamin_e=0.05,
         vitamin_b6=0.045, folate=3, potassium=112, magnesium=10, calcium=7, iron=0.24),
    ["Look for a creamy-yellow 'field spot' where it sat on the ground.", "Thump it: deep and hollow = ripe.", "Choose one heavy for its size."],
    "Rind color can't reveal watermelon ripeness in a photo - use the field spot and thump test.")

# ============================ VEGETABLES ============================
add("potato", "Potato", "vegetable", False, False, 173, "1 medium",
    dict(type="solanine"),
    dict(kcal=77, carbs=17.5, fiber=2.1, sugar=0.8, protein=2.0, fat=0.1, vitamin_c=19.7, vitamin_k=2.0, vitamin_b1=0.081, vitamin_b3=1.061,
         vitamin_b6=0.298, folate=16, potassium=421, magnesium=23, calcium=12, iron=0.81, manganese=0.153),
    ["Avoid green-tinted skin - that's solanine (a toxin), not ripeness.", "Skip soft spots, sprouted eyes or a musty smell.", "Store dark and cool."])
add("sweet_potato", "Sweet Potato", "vegetable", False, False, 130, "1 medium",
    spot(dict(dark=0.9, pale=0.6), 26, 10),
    dict(kcal=86, carbs=20.1, fiber=3.0, sugar=4.2, protein=1.6, fat=0.1, vitamin_c=2.4, vitamin_a=709, vitamin_k=1.8, vitamin_e=0.26,
         vitamin_b3=0.557, vitamin_b5=0.8, vitamin_b6=0.209, folate=11, potassium=337, magnesium=25, calcium=30, iron=0.61, manganese=0.258),
    ["Firm with smooth, unbruised skin.", "Avoid soft dark patches or wrinkling.", "Never refrigerate raw - it hardens the flesh."])
add("onion", "Onion", "vegetable", False, False, 110, "1 medium",
    spot(None, 30, 12, green_at=15),
    dict(kcal=40, carbs=9.3, fiber=1.7, sugar=4.2, protein=1.1, fat=0.1, vitamin_c=7.4, vitamin_b1=0.046, vitamin_b6=0.12, folate=19,
         potassium=146, magnesium=10, calcium=23, iron=0.21),
    ["Firm and heavy with dry, papery skin.", "Avoid soft spots near the neck.", "A small green sprout can be cut away."])
add("carrot", "Carrot", "vegetable", False, False, 61, "1 medium",
    spot(dict(dark=0.9, pale=0.7), 22, 10),
    dict(kcal=41, carbs=9.6, fiber=2.8, sugar=4.7, protein=0.9, fat=0.2, vitamin_c=5.9, vitamin_a=835, vitamin_k=13.2, vitamin_e=0.66,
         vitamin_b3=0.983, vitamin_b6=0.138, folate=19, potassium=320, magnesium=12, calcium=33, iron=0.3),
    ["Firm, snappy and deeply orange.", "White, dry 'blush' means dehydration.", "Skip bendy or slimy carrots; fresh tops should be green, not wilted."])
add("cucumber", "Cucumber", "vegetable", False, False, 104, "1 cup sliced",
    spot(dict(dark=0.9, brown=0.6, pale=0.6), 22, 10),
    dict(kcal=15, carbs=3.6, fiber=0.5, sugar=1.7, protein=0.7, fat=0.1, vitamin_c=2.8, vitamin_a=5, vitamin_k=16.4, vitamin_b5=0.259,
         vitamin_b6=0.04, folate=7, potassium=147, magnesium=13, calcium=16, iron=0.28),
    ["Firm end to end - any give means it's turning.", "Skip yellowing skin or wrinkled ends.", "Smaller ones are often less bitter."])
add("pepper", "Bell Pepper", "vegetable", False, False, 119, "1 medium",
    spot(dict(dark=0.8, brown=0.6, pale=0.6), 22, 9),
    dict(kcal=31, carbs=6.0, fiber=2.1, sugar=4.2, protein=1.0, fat=0.3, vitamin_c=127.7, vitamin_a=157, vitamin_k=4.9, vitamin_e=1.58,
         vitamin_b2=0.085, vitamin_b3=0.979, vitamin_b6=0.291, folate=46, potassium=211, magnesium=12, calcium=7, iron=0.43),
    ["Taut, glossy skin and a firm stem.", "Heavier peppers have thicker walls.", "Color is mostly variety/ripeness stage, not freshness."],
    "Nutrition shown is for red sweet pepper; green has less vitamin C/A.")
add("broccoli", "Broccoli", "vegetable", False, False, 91, "1 cup chopped",
    yellowing(dict(dark=0.9, brown=0.6), 20, 14),
    dict(kcal=34, carbs=6.6, fiber=2.6, sugar=1.7, protein=2.8, fat=0.4, vitamin_c=89.2, vitamin_a=31, vitamin_k=101.6, vitamin_e=0.78,
         vitamin_b2=0.117, vitamin_b3=0.639, vitamin_b5=0.573, vitamin_b6=0.175, folate=63, potassium=316, magnesium=21, calcium=47, iron=0.73, manganese=0.21),
    ["Tight deep-green buds; avoid yellow flowering buds.", "Cut stem end should look moist.", "A strong sulfur smell means past its best."])
add("cauliflower", "Cauliflower", "vegetable", False, False, 107, "1 cup",
    yellowing(dict(brown=0.7, dark=0.9), 20, 18),
    dict(kcal=25, carbs=5.0, fiber=2.0, sugar=1.9, protein=1.9, fat=0.3, vitamin_c=48.2, vitamin_k=15.5, vitamin_e=0.08, vitamin_b5=0.667,
         vitamin_b6=0.184, folate=57, potassium=299, magnesium=15, calcium=22, iron=0.42, manganese=0.155),
    ["Tight, creamy-white head with no brown spots.", "Leaves should be fresh green, not wilted.", "Avoid soft or fuzzy patches."])
add("lettuce", "Lettuce", "vegetable", False, False, 36, "1 cup shredded",
    yellowing(dict(dark=0.7, brown=0.5, pale=0.4), 18, 16),
    dict(kcal=15, carbs=2.9, fiber=1.3, sugar=0.8, protein=1.4, fat=0.2, vitamin_c=18, vitamin_a=370, vitamin_k=126.3, vitamin_e=0.29,
         vitamin_b6=0.09, folate=38, potassium=194, magnesium=13, calcium=36, iron=0.86, manganese=0.25),
    ["Crisp, vividly colored leaves.", "Avoid slimy patches or a wet, heavy feel.", "Store in a breathable bag in the fridge."])
add("spinach", "Spinach", "vegetable", False, False, 30, "1 cup",
    yellowing(dict(dark=0.8, brown=0.5, pale=0.3), 18, 16),
    dict(kcal=23, carbs=3.6, fiber=2.2, sugar=0.4, protein=2.9, fat=0.4, vitamin_c=28.1, vitamin_a=469, vitamin_k=482.9, vitamin_e=2.03,
         vitamin_b2=0.189, vitamin_b6=0.195, folate=194, potassium=558, magnesium=79, calcium=99, iron=2.71, manganese=0.897),
    ["Deep green, perky leaves - no yellowing or slime.", "Avoid bags with liquid pooled at the bottom.", "Use within a few days."])
add("cabbage", "Cabbage", "vegetable", False, False, 89, "1 cup chopped",
    yellowing(dict(dark=0.8, brown=0.6), 20, 18),
    dict(kcal=25, carbs=5.8, fiber=2.5, sugar=3.2, protein=1.3, fat=0.1, vitamin_c=36.6, vitamin_a=5, vitamin_k=76, vitamin_e=0.15,
         vitamin_b6=0.124, folate=43, potassium=170, magnesium=12, calcium=40, iron=0.47, manganese=0.16),
    ["Dense and heavy for its size.", "Outer leaves fresh and tight, not cracked.", "Avoid black spots or a strong sulfur smell."])
add("corn", "Corn (on the cob)", "vegetable", False, False, 90, "1 medium ear",
    spot(dict(dark=0.9, pale=0.4), 22, 100, min_green=30),
    dict(kcal=86, carbs=18.7, fiber=2.0, sugar=6.3, protein=3.3, fat=1.4, vitamin_c=6.8, vitamin_a=9, vitamin_k=0.3, vitamin_e=0.07,
         vitamin_b1=0.155, vitamin_b2=0.055, vitamin_b3=1.77, vitamin_b5=0.717, vitamin_b6=0.093, folate=42, potassium=270, magnesium=37, calcium=2, iron=0.52),
    ["Bright green, tightly wrapped husks.", "Golden-brown silk is normal; black, slimy silk is not.", "Feel for plump, evenly filled kernels."])
add("squash", "Pumpkin / Winter Squash", "vegetable", False, False, 116, "1 cup cubed",
    spot(dict(dark=0.9, pale=0.7), 20, 8),
    dict(kcal=26, carbs=6.5, fiber=0.5, sugar=2.8, protein=1.0, fat=0.1, vitamin_c=9.0, vitamin_a=426, vitamin_k=1.1, vitamin_e=0.44,
         vitamin_b2=0.11, vitamin_b3=0.6, vitamin_b6=0.061, folate=16, potassium=340, magnesium=12, calcium=21, iron=0.8),
    ["A hard rind that resists a fingernail = good storage life.", "Matte (not glossy) skin usually means fully mature.", "Heavy for its size = dense flesh."])
add("mushroom", "Mushroom", "vegetable", False, False, 70, "1 cup sliced",
    spot(dict(dark=0.9), 15, 6),
    dict(kcal=22, carbs=3.3, fiber=1.0, sugar=2.0, protein=3.1, fat=0.3, vitamin_c=2.1, vitamin_b1=0.081, vitamin_b2=0.402, vitamin_b3=3.607,
         vitamin_b5=1.497, vitamin_b6=0.104, folate=17, potassium=318, magnesium=9, calcium=3, iron=0.5),
    ["Firm, dry and plump with intact caps.", "Avoid slimy, sunken or dark-spotted mushrooms.", "Store in a paper bag, not plastic."])

# ============================ MORE VEGETABLES ============================
add("green_beans", "Green Beans", "vegetable", False, False, 100, "1 cup",
    spot(None, 20, 8),
    dict(kcal=31, carbs=7.0, fiber=3.4, sugar=3.3, protein=1.8, fat=0.2, vitamin_c=12.2, vitamin_a=35, vitamin_k=43,
         vitamin_b6=0.141, folate=33, potassium=211, magnesium=25, calcium=37, iron=1.03, manganese=0.216),
    ["Snap crisply in half - if it bends instead, it's past its best.", "Bright, uniform green with no brown or slimy patches.", "Avoid bulging pods (means tough, overgrown beans inside)."])
add("peas", "Peas (in pod)", "vegetable", False, False, 145, "1 cup",
    spot(None, 20, 8),
    dict(kcal=81, carbs=14.5, fiber=5.7, sugar=5.7, protein=5.4, fat=0.4, vitamin_c=40, vitamin_a=38, vitamin_k=24.8,
         vitamin_b1=0.266, vitamin_b6=0.169, folate=65, potassium=244, magnesium=33, calcium=25, iron=1.47, manganese=0.41),
    ["Plump, firm, bright green pods.", "Squeaky when rubbed together = fresh.", "Avoid yellowing or spotted pods."])
add("garlic", "Garlic", "vegetable", False, False, 3, "1 clove",
    spot(None, 25, 10),
    dict(kcal=149, carbs=33.1, fiber=2.1, sugar=1.0, protein=6.4, fat=0.5, vitamin_c=31.2, vitamin_b6=1.235, folate=3,
         potassium=401, magnesium=25, calcium=181, iron=1.7, manganese=1.672),
    ["Firm bulb, papery skin, no give when squeezed.", "Avoid soft, shriveled or sprouting cloves.", "A musty smell means mold has set in."])
add("ginger", "Ginger", "vegetable", False, False, 96, "1 piece (2\")",
    spot(None, 22, 10),
    dict(kcal=80, carbs=17.8, fiber=2.0, sugar=1.7, protein=1.8, fat=0.8, vitamin_c=5.0, vitamin_b6=0.16, folate=11,
         potassium=415, magnesium=43, calcium=16, iron=0.6, manganese=0.229),
    ["Firm, smooth skin with a spicy smell when scratched.", "Avoid shriveled, moldy or overly fibrous pieces.", "Should snap, not bend."])
add("eggplant", "Eggplant", "vegetable", False, False, 82, "1 cup cubed",
    spot(dict(dark=0.8, pale=0.6), 20, 8),
    dict(kcal=25, carbs=5.9, fiber=3.0, sugar=3.5, protein=1.0, fat=0.2, vitamin_c=2.2, vitamin_k=3.5, vitamin_b6=0.084,
         folate=22, potassium=229, magnesium=14, calcium=9, iron=0.23, manganese=0.232),
    ["Glossy, taut skin that springs back when pressed.", "Heavy for its size means fewer seeds.", "Avoid wrinkled skin or brown patches."])
add("zucchini", "Zucchini", "vegetable", False, False, 124, "1 cup sliced",
    spot(dict(dark=0.9, brown=0.6, pale=0.6), 22, 10),
    dict(kcal=17, carbs=3.1, fiber=1.0, sugar=2.5, protein=1.2, fat=0.3, vitamin_c=17.9, vitamin_a=10, vitamin_k=4.3,
         vitamin_b6=0.163, folate=24, potassium=261, magnesium=18, calcium=16, iron=0.37, manganese=0.177),
    ["Firm, glossy skin; smaller ones are more tender.", "Avoid soft spots or a rubbery feel.", "Skip overly large ones - can be watery and seedy."])
add("radish", "Radish", "vegetable", False, False, 116, "1 cup sliced",
    spot(dict(dark=0.9, pale=0.7), 20, 9),
    dict(kcal=16, carbs=3.4, fiber=1.6, sugar=1.9, protein=0.7, fat=0.1, vitamin_c=14.8, folate=25, potassium=233,
         magnesium=10, calcium=25, iron=0.34),
    ["Firm and crisp with smooth, unblemished skin.", "Fresh green leafy tops (if attached) mean recently picked.", "Avoid spongy or cracked radishes."])
add("beet", "Beet", "vegetable", False, False, 136, "1 cup",
    spot(dict(dark=0.9, pale=0.7), 22, 10),
    dict(kcal=43, carbs=9.6, fiber=2.8, sugar=6.8, protein=1.6, fat=0.2, vitamin_c=4.9, folate=109, potassium=325,
         magnesium=23, calcium=16, iron=0.8, manganese=0.329),
    ["Firm, smooth skin, heavy for its size.", "Avoid soft spots or wrinkled skin.", "Fresh, unwilted greens (if attached) are a good sign."])
add("celery", "Celery", "vegetable", False, False, 101, "1 cup chopped",
    yellowing(dict(dark=0.7, brown=0.5), 18, 15),
    dict(kcal=16, carbs=3.0, fiber=1.6, sugar=1.3, protein=0.7, fat=0.2, vitamin_c=3.1, vitamin_a=22, vitamin_k=29.3,
         folate=36, potassium=260, magnesium=11, calcium=40, iron=0.2),
    ["Firm stalks that snap, not bend.", "Deep green, not yellowing.", "Avoid limp stalks or browning at the base."])
add("asparagus", "Asparagus", "vegetable", False, False, 134, "1 cup",
    yellowing(dict(dark=0.7, brown=0.5), 18, 16),
    dict(kcal=20, carbs=3.9, fiber=2.1, sugar=1.9, protein=2.2, fat=0.1, vitamin_c=5.6, vitamin_a=38, vitamin_k=41.6,
         vitamin_b6=0.091, folate=52, potassium=202, magnesium=14, calcium=24, iron=2.14),
    ["Firm stalks with tightly closed, compact tips.", "Avoid limp stalks or spreading, mushy tips.", "Cut ends should look moist, not dried or woody."])
add("kale", "Kale", "vegetable", False, False, 67, "1 cup chopped",
    yellowing(dict(dark=0.7, brown=0.5, pale=0.4), 18, 16),
    dict(kcal=49, carbs=8.8, fiber=3.6, sugar=2.3, protein=4.3, fat=0.9, vitamin_c=120, vitamin_a=500, vitamin_k=704.8,
         vitamin_b6=0.271, folate=141, potassium=348, magnesium=33, calcium=150, iron=1.47, manganese=0.66),
    ["Deep green, firm leaves with no yellowing.", "Smaller leaves are more tender.", "Avoid wilted, slimy or heavily browned edges."])

# ============================ MORE FRUIT ============================
add("plum", "Plum", "fruit", True, True, 66, "1 medium",
    ripener(["purple", "red"], 50, 40, 28, 14),
    dict(kcal=46, carbs=11.4, fiber=1.4, sugar=9.9, protein=0.7, fat=0.3, vitamin_c=9.5, vitamin_a=17, vitamin_k=6.4,
         vitamin_b6=0.029, folate=5, potassium=157, magnesium=7, calcium=6, iron=0.17),
    ["Gentle squeeze near the stem - a slight give means ripe.", "Ripens further at room temperature if firm.", "Avoid wrinkled skin or a fermented smell."])
add("apricot", "Apricot", "fruit", True, True, 35, "1 medium",
    ripener(["yellow", "orange"], 50, 40, 28, 14),
    dict(kcal=48, carbs=11.1, fiber=2.0, sugar=9.2, protein=1.4, fat=0.4, vitamin_c=10, vitamin_a=96, vitamin_k=3.3,
         vitamin_b6=0.054, folate=9, potassium=259, magnesium=10, calcium=13, iron=0.39),
    ["Deep gold-orange color with a slight give when pressed.", "Avoid greenish, hard fruit - it won't ripen much further.", "Bruises easily, so check for soft spots."])
add("fig", "Fig", "fruit", False, True, 50, "1 medium",
    freshness(["purple", "red"], 22, 0),
    dict(kcal=74, carbs=19.2, fiber=2.9, sugar=16.3, protein=0.8, fat=0.3, vitamin_c=2.0, vitamin_a=7, vitamin_k=4.7,
         vitamin_b6=0.113, folate=6, potassium=232, magnesium=17, calcium=35, iron=0.37),
    ["Slightly soft to the touch with unblemished skin.", "A sweet smell at the base is a good sign.", "Very perishable - use within a day or two of buying."],
    "Fig color varies hugely by variety (green, brown or purple), so softness matters more than color.")
add("pomegranate", "Pomegranate", "fruit", False, True, 174, "1 whole",
    unjudgeable(dict(dark=0.8, pale=0.6), 26),
    dict(kcal=83, carbs=18.7, fiber=4.0, sugar=13.7, protein=1.7, fat=1.2, vitamin_c=10.2, vitamin_k=16.4,
         vitamin_b6=0.075, folate=38, potassium=236, magnesium=12, calcium=10, iron=0.3),
    ["Heavy for its size means juicier arils inside.", "Skin should be taut, not soft or wrinkled.", "A few flat facets from a squarish shape is normal."],
    "The rind hides the arils, so ripeness can't be judged by photo - weight and firmness are the real test.")
add("grapefruit", "Grapefruit", "fruit", False, True, 123, "½ grapefruit",
    freshness(["orange", "yellow"], 24, 20),
    dict(kcal=42, carbs=10.7, fiber=1.6, sugar=6.9, protein=0.8, fat=0.1, vitamin_c=31.2, vitamin_a=58, folate=10,
         potassium=135, magnesium=9, calcium=12, iron=0.06),
    ["Heavy for its size = juicier.", "Firm, finely-textured skin.", "Slight green tinge is normal and doesn't mean unripe."])
add("lime", "Lime", "fruit", False, True, 67, "1 lime",
    freshness(["green"], 22, 20),
    dict(kcal=30, carbs=10.5, fiber=2.8, sugar=1.7, protein=0.7, fat=0.2, vitamin_c=29.1, vitamin_b6=0.043, folate=8,
         potassium=102, magnesium=6, calcium=33, iron=0.6),
    ["Bright green, glossy, heavy for its size.", "Yellowing means it's aging (still usable, less tart).", "Avoid hard, dry or wrinkled limes."])
add("raspberry", "Raspberries", "fruit", False, False, 123, "1 cup",
    spot(dict(dark=0.9, pale=0.8, brown=0.5), 18, 8),
    dict(kcal=52, carbs=11.9, fiber=6.5, sugar=4.4, protein=1.2, fat=0.7, vitamin_c=26.2, vitamin_a=2, vitamin_k=7.8,
         vitamin_e=0.87, vitamin_b6=0.055, folate=21, potassium=151, magnesium=22, calcium=25, iron=0.69, manganese=0.67),
    ["Deep, uniform color with no wet or fuzzy patches.", "Check the container base for juice stains (crushed/moldy berries).", "Very perishable - eat within a day or two."])
add("blackberry", "Blackberries", "fruit", False, False, 144, "1 cup",
    spot(dict(dark=0.6, pale=0.8, brown=0.5), 20, 9),
    dict(kcal=43, carbs=9.6, fiber=5.3, sugar=4.9, protein=1.4, fat=0.5, vitamin_c=21, vitamin_a=11, vitamin_k=19.8,
         vitamin_e=1.17, vitamin_b6=0.03, folate=25, potassium=162, magnesium=20, calcium=29, iron=0.62, manganese=0.646),
    ["Plump, deep black-purple with no red patches (red = underripe).", "Avoid any fuzzy white/gray mold.", "Should be firmly attached, not mushy or leaking."])
add("coconut", "Coconut", "fruit", False, False, 45, "1 piece",
    unjudgeable(dict(dark=0.5, pale=0.7), 30),
    dict(kcal=354, carbs=15.2, fiber=9.0, sugar=6.2, protein=3.3, fat=33.5, vitamin_c=3.3, vitamin_b6=0.05, folate=26,
         potassium=356, magnesium=32, calcium=14, iron=2.43, manganese=1.5),
    ["Shake it - you should hear liquid sloshing inside.", "Check the three 'eyes' for mold or wetness.", "Heavy for its size means more coconut water inside."],
    "Husk color says little about ripeness in a photo - the shake test and checking the eyes work better.")
