import json
import random
from pathlib import Path

random.seed(42)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

N_TRAIN = 800
N_DEV = 160
N_TEST = 200

NAMES = ["ramesh", "sita", "rohan", "mehta", "rahul", "kumar", "sakshi", "arjun", "neha", "vivek"]
SURNAMES = ["kumar", "sharma", "reddy", "singh", "iyer", "rao", "patel", "verma"]
CITIES = ["chennai", "delhi", "mumbai", "bangalore", "kolkata", "pune", "hyderabad"]
LOCATIONS = ["central park", "marine drive", "mg road", "brigade road", "express avenue", "phoenix mall"]
DOMAINS = ["gmail dot com", "yahoo dot com", "outlook dot com", "hotmail dot com", "gamil dot com", "outlok dot com"]

DIGIT_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

FILLERS = ["uh", "umm", "like", "you know", "actually", "so", "ya", "basically", "right"]

# ---------- PII generators (already STT-like) ----------

def gen_person_name():
    return random.choice(NAMES) + " " + random.choice(SURNAMES)

def gen_city():
    return random.choice(CITIES)

def gen_location():
    return random.choice(LOCATIONS)

def gen_phone(n_digits: int = 10) -> str:
    """Return either wordy phone (with doubles) or digit phone with optional spacing."""
    if random.random() < 0.5:
        return gen_phone_words(n_digits)
    return gen_phone_digits(n_digits)

def gen_phone_words(n_digits: int = 10) -> str:
    nums = [random.randint(0, 9) for _ in range(n_digits)]
    words = []
    i = 0
    while i < n_digits:
        if i + 1 < n_digits and random.random() < 0.2 and nums[i] == nums[i + 1]:
            words.append("double " + DIGIT_WORDS[nums[i]])
            i += 2
        else:
            words.append(DIGIT_WORDS[nums[i]])
            i += 1
    # occasionally mix digits
    if random.random() < 0.3:
        words = [str(random.randint(0, 9)) if random.random() < 0.5 else w for w in words]
    return " ".join(words)

def gen_phone_digits(n_digits: int = 10) -> str:
    nums = [str(random.randint(0, 9)) for _ in range(n_digits)]
    if random.random() < 0.5:
        # spaced groups
        grouped = ["".join(nums[i:i+5]) for i in range(0, n_digits, 5)]
        return " ".join(grouped)
    return "".join(nums)

def gen_credit_card():
    nums = [str(random.randint(0, 9)) for _ in range(16)]
    # sometimes keep contiguous digits, sometimes spaced
    if random.random() < 0.6:
        groups = ["".join(nums[i:i+4]) for i in range(0, 16, 4)]
        return " ".join(groups)
    return "".join(nums)

def gen_email():
    first = random.choice(NAMES)
    last = random.choice(SURNAMES)
    dom = random.choice(DOMAINS)
    # sometimes add digits
    if random.random() < 0.4:
        first += str(random.randint(1, 99))
    return f"{first} dot {last} at {dom}"

def gen_date_phrase():
    days = ["1", "2", "5", "10", "15", "20", "25", "30"]
    months = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december"
    ]
    year = random.choice(["2022", "2023", "2024", "2025"])
    form = random.random()
    if form < 0.33:
        return f"{random.choice(days)} {random.choice(months)} {year}"
    elif form < 0.66:
        return f"{random.choice(days)} of {random.choice(months)} {year}"
    else:
        # numeric style
        day = random.choice(["01", "02", "05", "10", "15", "20", "25"])
        month = random.choice(["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"])
        return f"{day} {month} {year}"

# ---------- STT-style noise for CONTEXT ONLY ----------

def noise_context_chunk(chunk: str) -> str:
    """Apply simple STT-like noise to non-PII text."""
    words = chunk.split()

    # drop some small words
    new_words = []
    for w in words:
        if w in {"i", "please", "can", "you", "to"} and random.random() < 0.25:
            continue
        # simple contractions
        if (w, random.random()) == ("want", 0):  # rarely
            new_words.append("wanna")
            continue
        if w == "going" and random.random() < 0.3:
            new_words.append("gonna")
            continue
        new_words.append(w)

    # insert fillers
    i = 0
    out = []
    while i < len(new_words):
        if random.random() < 0.15:
            out.append(random.choice(FILLERS))
        # occasionally inject decoy digits to test precision
        if random.random() < 0.05:
            decoy_len = random.choice([6, 8, 10, 12])
            decoy = "".join(str(random.randint(0, 9)) for _ in range(decoy_len))
            out.append(decoy)
        out.append(new_words[i])
        i += 1

    return " ".join(out)

# ---------- Scenario templates ----------

SCENARIOS = [
    # (template, entities in order)
    ("hi this is {PERSON_NAME} i want to change my phone number to {PHONE}", ["PERSON_NAME", "PHONE"]),
    ("uh can you please update my email to {EMAIL} today", ["EMAIL"]),
    ("my credit card number is {CREDIT_CARD} and it got blocked yesterday", ["CREDIT_CARD"]),
    ("i live in {CITY} and my phone number is {PHONE}", ["CITY", "PHONE"]),
    ("i booked the ticket on {DATE} and i want a refund", ["DATE"]),
    ("this is {PERSON_NAME} from {CITY} my email is {EMAIL}", ["PERSON_NAME", "CITY", "EMAIL"]),
    ("please reset my password the phone linked is {PHONE}", ["PHONE"]),
    ("deliver to {LOCATION} in {CITY} and my phone is {PHONE}", ["LOCATION", "CITY", "PHONE"]),
    ("meeting on {DATE} at {LOCATION} call {PHONE}", ["DATE", "LOCATION", "PHONE"]),
    ("call {PERSON_NAME} at {PHONE} i stay near {LOCATION}", ["PERSON_NAME", "PHONE", "LOCATION"]),
    ("email is {EMAIL} but order id is 9876543210 and ref 4242 4242 4242 4242", ["EMAIL"]),  # decoy phone/cc
    ("i need to travel on {DATE} to {CITY} please call me at {PHONE}", ["DATE", "CITY", "PHONE"]),
    ("here is transaction number 4242 4242 4242 4242 and my card is {CREDIT_CARD}", ["CREDIT_CARD"]),  # decoy cc
    ("ticket number 98765 43210 but my phone is {PHONE}", ["PHONE"]),
    # calls with NO PII
    ("hi i am just calling to ask about delivery status", []),
    ("uh ya i wanted to know your return policy for damaged items", []),
    ("just checking store timings near {CITY}", ["CITY"]),
]

ENTITY_GENERATORS = {
    "PERSON_NAME": gen_person_name,
    "CITY": gen_city,
    "LOCATION": gen_location,
    "PHONE": gen_phone,
    "CREDIT_CARD": gen_credit_card,
    "EMAIL": gen_email,
    "DATE": gen_date_phrase,
}

def build_example(idx: int):
    template, ent_order = random.choice(SCENARIOS)

    # First, replace placeholders with markers so we can split safely
    text = template
    for ent in ent_order:
        text = text.replace("{" + ent + "}", f"__{ent}__", 1)

    parts = text.split("__")
    # parts will look like:  ["hi this is ", "PERSON_NAME", " i want to ... ", "PHONE", " ..."]

    out_text_parts = []
    entities = []
    offset = 0

    def maybe_add_space():
        nonlocal offset
        if out_text_parts:
            last = out_text_parts[-1]
            if last and last[-1] != " ":
                out_text_parts.append(" ")
                offset += 1

    i = 0
    while i < len(parts):
        segment = parts[i]
        if segment in ENTITY_GENERATORS:
            # entity island (no extra noise inside)
            value = ENTITY_GENERATORS[segment]()
            maybe_add_space()
            start = offset
            out_text_parts.append(value)
            offset += len(value)
            end = offset
            entities.append({"start": start, "end": end, "label": segment})
        else:
            # context chunk -> add noise, but keep spaces consistent
            chunk = segment.strip()
            if chunk:
                maybe_add_space()
                noisy = noise_context_chunk(chunk)
                if noisy:
                    out_text_parts.append(noisy)
                    offset += len(noisy)
        i += 1

    full_text = "".join(out_text_parts)
    return {
        "id": f"utt_{idx:04d}",
        "text": full_text,
        "entities": entities,
    }

def main():
    examples = [build_example(i+1) for i in range(N_TRAIN + N_DEV + N_TEST)]
    random.shuffle(examples)

    train = examples[:N_TRAIN]
    dev = examples[N_TRAIN:N_TRAIN+N_DEV]
    test = examples[N_TRAIN+N_DEV:]

    with open(DATA_DIR / "train.jsonl", "w", encoding="utf-8") as f:
        for ex in train:
            f.write(json.dumps(ex) + "\n")

    with open(DATA_DIR / "dev.jsonl", "w", encoding="utf-8") as f:
        for ex in dev:
            f.write(json.dumps(ex) + "\n")

    # test: drop entities
    with open(DATA_DIR / "test.jsonl", "w", encoding="utf-8") as f:
        for ex in test:
            f.write(json.dumps({"id": ex["id"], "text": ex["text"]}) + "\n")

    print(f"wrote {len(train)} train, {len(dev)} dev, {len(test)} test")

if __name__ == "__main__":
    main()
