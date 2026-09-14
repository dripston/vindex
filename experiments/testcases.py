"""
Test cases: 10 factual-QA tasks x 3 language variants.

Each task provides:
  - the question in English / Hindi (Devanagari) / Hinglish (Romanized)
  - a short gold answer (bare entity, e.g. "Mumbai") in English and in the
    same language as the question -- the ORIGINAL fields, kept as-is.
  - a full-sentence gold answer (e.g. "The capital of Maharashtra is
    Mumbai.") in English and in the same language -- added for
    discrimination.py, so gold references match the register (full
    sentences) of real model answers instead of bare entities.
  - a `human_correct` checker: given the model's raw answer text, does a
    human consider it correct? (Language-agnostic, accepts Hindi/Hinglish
    spellings of the same fact.)
  - hand-authored WRONG answers in all three variants, at two severities:
      wrong_hard   -- different entity/fact entirely
      wrong_subtle -- right topic and entity, one detail wrong

Nothing below is machine-translated: each variant's wrong answers are
written directly in that language/register, not translated word-for-word
from the English version.
"""
import os

# Each entry: id, base(en/hi/hinglish) question, gold (english short answer),
# accept: list of lowercase substrings ANY of which => human marks correct.
# gold_hi / gold_hinglish: short gold in that language (existing, unchanged).
# gold_en_full / gold_hi_full / gold_hinglish_full: full-sentence gold, new.
# wrong_hard / wrong_subtle: dict of variant -> hand-written wrong answer.
TASKS = [
    {
        "id": "capital_maharashtra",
        "en": "What is the capital of the Indian state of Maharashtra?",
        "hi": "भारत के महाराष्ट्र राज्य की राजधानी क्या है?",
        "hinglish": "Bharat ke Maharashtra rajya ki rajdhani kya hai?",
        "gold": "Mumbai",
        "accept": ["mumbai", "bombay", "मुंबई", "बंबई"],
        "gold_hi": "मुंबई",
        "gold_hinglish": "Mumbai",
        "gold_en_full": "The capital of Maharashtra is Mumbai.",
        "gold_hi_full": "महाराष्ट्र की राजधानी मुंबई है।",
        "gold_hinglish_full": "Maharashtra ki rajdhani Mumbai hai.",
        "wrong_hard": {
            "en": "The capital of Maharashtra is Chennai.",
            "hi": "महाराष्ट्र की राजधानी चेन्नई है।",
            "hinglish": "Maharashtra ki rajdhani Chennai hai.",
        },
        "wrong_subtle": {
            # right city, wrong claim about it (Mumbai is not the national capital)
            "en": "Mumbai is the capital of India.",
            "hi": "मुंबई भारत की राजधानी है।",
            "hinglish": "Mumbai Bharat ki rajdhani hai.",
        },
    },
    {
        "id": "planets_count",
        "en": "How many planets are there in our solar system?",
        "hi": "हमारे सौर मंडल में कितने ग्रह हैं?",
        "hinglish": "Hamare saur mandal mein kitne grah hain?",
        "gold": "Eight",
        "accept": ["eight", "8", "आठ"],
        "gold_hi": "आठ",
        "gold_hinglish": "Aath",
        "gold_en_full": "There are eight planets in our solar system.",
        "gold_hi_full": "हमारे सौर मंडल में आठ ग्रह हैं।",
        "gold_hinglish_full": "Hamare saur mandal mein aath grah hain.",
        "wrong_hard": {
            "en": "There are twelve planets in our solar system.",
            "hi": "हमारे सौर मंडल में बारह ग्रह हैं।",
            "hinglish": "Hamare saur mandal mein barah grah hain.",
        },
        "wrong_subtle": {
            # right count used to be true before 2006; states it as still current
            "en": "There are nine planets in our solar system, including Pluto.",
            "hi": "हमारे सौर मंडल में नौ ग्रह हैं, जिसमें प्लूटो भी शामिल है।",
            "hinglish": "Hamare saur mandal mein nau grah hain, jismein Pluto bhi shamil hai.",
        },
    },
    {
        "id": "freedom_year",
        "en": "In which year did India gain independence?",
        "hi": "भारत को किस वर्ष स्वतंत्रता मिली?",
        "hinglish": "Bharat ko kis saal aazadi mili?",
        "gold": "1947",
        "accept": ["1947", "१९४७"],
        "gold_hi": "१९४७",
        "gold_hinglish": "1947",
        "gold_en_full": "India gained independence in 1947.",
        "gold_hi_full": "भारत को 1947 में स्वतंत्रता मिली।",
        "gold_hinglish_full": "Bharat ko 1947 mein aazadi mili.",
        "wrong_hard": {
            "en": "India gained independence in 1857.",
            "hi": "भारत को 1857 में स्वतंत्रता मिली।",
            "hinglish": "Bharat ko 1857 mein aazadi mili.",
        },
        "wrong_subtle": {
            # off by one year
            "en": "India gained independence in 1948.",
            "hi": "भारत को 1948 में स्वतंत्रता मिली।",
            "hinglish": "Bharat ko 1948 mein aazadi mili.",
        },
    },
    {
        "id": "water_formula",
        "en": "What is the chemical formula of water?",
        "hi": "पानी का रासायनिक सूत्र क्या है?",
        "hinglish": "Pani ka rasayanik sutra kya hai?",
        "gold": "H2O",
        "accept": ["h2o", "h₂o", "h 2 o", "h-2-o", "एच2ओ"],
        "gold_hi": "H2O",
        "gold_hinglish": "H2O",
        "gold_en_full": "The chemical formula of water is H2O.",
        "gold_hi_full": "पानी का रासायनिक सूत्र H2O है।",
        "gold_hinglish_full": "Pani ka rasayanik sutra H2O hai.",
        "wrong_hard": {
            "en": "The chemical formula of water is CO2.",
            "hi": "पानी का रासायनिक सूत्र CO2 है।",
            "hinglish": "Pani ka rasayanik sutra CO2 hai.",
        },
        "wrong_subtle": {
            # right elements, wrong ratio
            "en": "The chemical formula of water is H2O2.",
            "hi": "पानी का रासायनिक सूत्र H2O2 है।",
            "hinglish": "Pani ka rasayanik sutra H2O2 hai.",
        },
    },
    {
        "id": "national_animal_india",
        "en": "What is the national animal of India?",
        "hi": "भारत का राष्ट्रीय पशु कौन सा है?",
        "hinglish": "Bharat ka rashtriya pashu kaun sa hai?",
        "gold": "Bengal tiger",
        "accept": ["tiger", "bengal tiger", "बाघ", "टाइगर", "sher"],
        "gold_hi": "बंगाल टाइगर",
        "gold_hinglish": "Bengal tiger",
        "gold_en_full": "The national animal of India is the Bengal tiger.",
        "gold_hi_full": "भारत का राष्ट्रीय पशु बंगाल टाइगर है।",
        "gold_hinglish_full": "Bharat ka rashtriya pashu Bengal tiger hai.",
        "wrong_hard": {
            "en": "The national animal of India is the lion.",
            "hi": "भारत का राष्ट्रीय पशु शेर है।",
            "hinglish": "Bharat ka rashtriya pashu sher hai.",
        },
        "wrong_subtle": {
            # right species, wrong specific subspecies name
            "en": "The national animal of India is the Siberian tiger.",
            "hi": "भारत का राष्ट्रीय पशु साइबेरियाई टाइगर है।",
            "hinglish": "Bharat ka rashtriya pashu Siberian tiger hai.",
        },
    },
    {
        "id": "largest_ocean",
        "en": "Which is the largest ocean on Earth?",
        "hi": "पृथ्वी का सबसे बड़ा महासागर कौन सा है?",
        "hinglish": "Prithvi ka sabse bada mahasagar kaun sa hai?",
        "gold": "Pacific Ocean",
        "accept": ["pacific", "प्रशांत", "prashant"],
        "gold_hi": "प्रशांत महासागर",
        "gold_hinglish": "Prashant Mahasagar",
        "gold_en_full": "The largest ocean on Earth is the Pacific Ocean.",
        "gold_hi_full": "पृथ्वी का सबसे बड़ा महासागर प्रशांत महासागर है।",
        "gold_hinglish_full": "Prithvi ka sabse bada mahasagar Prashant Mahasagar hai.",
        "wrong_hard": {
            "en": "The largest ocean on Earth is the Atlantic Ocean.",
            "hi": "पृथ्वी का सबसे बड़ा महासागर अटलांटिक महासागर है।",
            "hinglish": "Prithvi ka sabse bada mahasagar Atlantic Mahasagar hai.",
        },
        "wrong_subtle": {
            # right ocean, wrong superlative claim (it's not the deepest by that margin/definition mix-up)
            "en": "The Pacific Ocean is the smallest ocean on Earth.",
            "hi": "प्रशांत महासागर पृथ्वी का सबसे छोटा महासागर है।",
            "hinglish": "Prashant Mahasagar Prithvi ka sabse chhota mahasagar hai.",
        },
    },
    {
        "id": "sun_rise_direction",
        "en": "In which direction does the sun rise?",
        "hi": "सूर्य किस दिशा में उगता है?",
        "hinglish": "Suraj kis disha mein ugta hai?",
        "gold": "East",
        "accept": ["east", "पूर्व", "poorab", "purab", "purv"],
        "gold_hi": "पूर्व",
        "gold_hinglish": "Purab",
        "gold_en_full": "The sun rises in the east.",
        "gold_hi_full": "सूर्य पूर्व दिशा में उगता है।",
        "gold_hinglish_full": "Suraj purab disha mein ugta hai.",
        "wrong_hard": {
            "en": "The sun rises in the west.",
            "hi": "सूर्य पश्चिम दिशा में उगता है।",
            "hinglish": "Suraj paschim disha mein ugta hai.",
        },
        "wrong_subtle": {
            # adjacent direction, not opposite
            "en": "The sun rises in the north-east.",
            "hi": "सूर्य उत्तर-पूर्व दिशा में उगता है।",
            "hinglish": "Suraj uttar-purab disha mein ugta hai.",
        },
    },
    {
        "id": "days_in_leap_year",
        "en": "How many days are there in a leap year?",
        "hi": "एक लीप वर्ष में कितने दिन होते हैं?",
        "hinglish": "Leap year mein kitne din hote hain?",
        "gold": "366",
        "accept": ["366", "३६६", "three hundred sixty-six", "three hundred and sixty-six"],
        "gold_hi": "३६६",
        "gold_hinglish": "366",
        "gold_en_full": "A leap year has 366 days.",
        "gold_hi_full": "एक लीप वर्ष में 366 दिन होते हैं।",
        "gold_hinglish_full": "Ek leap year mein 366 din hote hain.",
        "wrong_hard": {
            "en": "A leap year has 400 days.",
            "hi": "एक लीप वर्ष में 400 दिन होते हैं।",
            "hinglish": "Ek leap year mein 400 din hote hain.",
        },
        "wrong_subtle": {
            # off by one day
            "en": "A leap year has 365 days.",
            "hi": "एक लीप वर्ष में 365 दिन होते हैं।",
            "hinglish": "Ek leap year mein 365 din hote hain.",
        },
    },
    {
        "id": "father_of_nation_india",
        "en": "Who is known as the Father of the Nation in India?",
        "hi": "भारत में राष्ट्रपिता के रूप में किसे जाना जाता है?",
        "hinglish": "Bharat mein rashtrapita ke roop mein kise jana jata hai?",
        "gold": "Mahatma Gandhi",
        "accept": ["gandhi", "mohandas", "बापू", "bapu", "महात्मा गांधी"],
        "gold_hi": "महात्मा गांधी",
        "gold_hinglish": "Mahatma Gandhi",
        "gold_en_full": "Mahatma Gandhi is known as the Father of the Nation in India.",
        "gold_hi_full": "भारत में महात्मा गांधी को राष्ट्रपिता के रूप में जाना जाता है।",
        "gold_hinglish_full": "Bharat mein Mahatma Gandhi ko rashtrapita ke roop mein jana jata hai.",
        "wrong_hard": {
            "en": "Jawaharlal Nehru is known as the Father of the Nation in India.",
            "hi": "भारत में जवाहरलाल नेहरू को राष्ट्रपिता के रूप में जाना जाता है।",
            "hinglish": "Bharat mein Jawaharlal Nehru ko rashtrapita ke roop mein jana jata hai.",
        },
        "wrong_subtle": {
            # right person, wrong title (that title belongs to someone else, Bose)
            "en": "Mahatma Gandhi is known as the Netaji of India.",
            "hi": "महात्मा गांधी को भारत का नेताजी कहा जाता है।",
            "hinglish": "Mahatma Gandhi ko Bharat ka Netaji kaha jata hai.",
        },
    },
    {
        "id": "boiling_point_water_c",
        "en": "At what temperature does water boil at sea level, in degrees Celsius?",
        "hi": "समुद्र तल पर पानी किस तापमान पर उबलता है, सेल्सियस में?",
        "hinglish": "Samudra tal par pani kis taapman par ubalta hai, Celsius mein?",
        "gold": "100 degrees Celsius",
        "accept": ["100", "१००", "hundred"],
        "gold_hi": "१०० डिग्री सेल्सियस",
        "gold_hinglish": "100 degree Celsius",
        "gold_en_full": "Water boils at 100 degrees Celsius at sea level.",
        "gold_hi_full": "समुद्र तल पर पानी 100 डिग्री सेल्सियस पर उबलता है।",
        "gold_hinglish_full": "Samudra tal par pani 100 degree Celsius par ubalta hai.",
        "wrong_hard": {
            "en": "Water boils at 50 degrees Celsius at sea level.",
            "hi": "समुद्र तल पर पानी 50 डिग्री सेल्सियस पर उबलता है।",
            "hinglish": "Samudra tal par pani 50 degree Celsius par ubalta hai.",
        },
        "wrong_subtle": {
            # confuses boiling point with freezing point
            "en": "Water boils at 0 degrees Celsius at sea level.",
            "hi": "समुद्र तल पर पानी 0 डिग्री सेल्सियस पर उबलता है।",
            "hinglish": "Samudra tal par pani 0 degree Celsius par ubalta hai.",
        },
    },
]

VARIANTS = ["en", "hi", "hinglish"]
LABELS = ["correct", "wrong_subtle", "wrong_hard"]


def human_is_correct(task, answer_text):
    """Language-agnostic human judgment: is the fact right?"""
    t = (answer_text or "").lower()
    return any(sub.lower() in t for sub in task["accept"])


def build_cases():
    """Original 30 cases: one per task x variant, used to score the real
    model answers stored in results.json. Unchanged by this module's new
    additions -- gold_same_lang here still points at the SHORT gold, so any
    script depending on the original field keeps working."""
    cases = []
    for task in TASKS:
        for v in VARIANTS:
            gold_same_lang_key = "gold" if v == "en" else f"gold_{v}"
            cases.append({
                "case_id": f"{task['id']}__{v}",
                "task_id": task["id"],
                "variant": v,
                "question": task[v],
                "gold": task["gold"],                          # English gold (short, existing)
                "gold_same_lang": task[gold_same_lang_key],     # same-language gold (short, existing)
                "task": task,
            })
    return cases


DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "results_clean.json",
)


def build_discrimination_cases(data_path=DEFAULT_DATA_PATH):
    """90 cases: 10 tasks x 3 variants x 3 labels (correct, wrong_subtle,
    wrong_hard). 'correct' reuses the real model answer loaded from
    data_path (joined in by the caller via case_id); wrong_subtle/wrong_hard
    are the hand-authored answers above.

    Each case carries all four gold references so the caller can pick
    gold_mode (english/same_language) x gold_length (short/full_sentence)
    without rebuilding cases per config.

    data_path defaults to data/results_clean.json. Pass the contaminated
    experiments/results.json explicitly if that's what you actually want
    (e.g. for a before/after comparison) -- there is no implicit
    redirection here.
    """
    import json

    with open(data_path, encoding="utf-8") as f:
        rows = json.load(f)
    # results_clean.json keeps rows that failed script adherence after every
    # retry attempt (for the record) but they must be excluded from scoring
    # -- results.json has no such field, so this is a no-op there.
    answer_rows = {
        r["case_id"]: r["answer"]
        for r in rows
        if not r.get("script_adherence_failure", False)
    }

    cases = []
    for task in TASKS:
        for v in VARIANTS:
            gold_short_same_key = "gold" if v == "en" else f"gold_{v}"
            gold_full_same_key = "gold_en_full" if v == "en" else f"gold_{v}_full"

            golds = {
                ("english_gold", "short"): task["gold"],
                ("english_gold", "full_sentence"): task["gold_en_full"],
                ("same_language_gold", "short"): task[gold_short_same_key],
                ("same_language_gold", "full_sentence"): task[gold_full_same_key],
            }

            for label in LABELS:
                if label == "correct":
                    case_id = f"{task['id']}__{v}"
                    answer = answer_rows.get(case_id)
                else:
                    case_id = f"{task['id']}__{v}__{label}"
                    answer = task[label][v]

                cases.append({
                    "case_id": case_id,
                    "task_id": task["id"],
                    "variant": v,
                    "label": label,
                    "answer": answer,
                    "golds": golds,   # dict[(gold_mode, gold_length)] -> gold text
                })
    return cases


if __name__ == "__main__":
    cases = build_cases()
    print(f"{len(cases)} positive cases")
    disc = build_discrimination_cases()
    print(f"{len(disc)} discrimination cases")
    from collections import Counter
    print(Counter(c["label"] for c in disc))
