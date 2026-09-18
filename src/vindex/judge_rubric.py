"""
Hindi-language rubric and script-aware prompting for indic_judge
(Milestone 5.1, 5.2).

WHY THE RUBRIC IS WRITTEN IN HINDI, NOT ENGLISH

This project's own Phase 0 data (experiments/FINDINGS.md) found an
English-rubric judge scoring a correct Hindi answer 0.0: asked to
evaluate "समुद्र तल पर पानी 100 C पर उबलता है" (water boils at 100 C
at sea level -- correct), the judge's own reasoning silently
mistranslated समुद्र तल ("sea level") as "sea floor" mid-thought, then
marked the answer wrong for not accounting for undersea pressure. The
question and answer were already in Hindi; the JUDGE'S REASONING was
in English, and the translation step INSIDE that reasoning is where
the error happened -- not a style penalty, a comprehension failure the
English framing invited by forcing an English-language thought process
about Hindi-language content.

The fix is not "tell the judge to be careful." It is: don't make the
judge translate at all. The rubric below is written in Hindi and
instructs the judge to reason in Hindi when the content is in Hindi --
so there is no translation step for an error to hide inside. This is
the mechanism, not a preference.

MILESTONE 5.8 UPDATE -- this mechanism fixes the specific incident
above, but does not, on its own, mean this rubric out-agrees a human
grader more than an English one would in general. Tested directly: an
English-rubric baseline agreed with the same human graders slightly
MORE than this Hindi rubric on the Milestone 6.3 study's 62 traces
(93.5% vs 90.3%). The 4 disagreements were not mistranslation in
either direction -- they were this rubric being more conservative
about answer completeness (see judge.py's docstring and
experiments/README.md). Keep the mechanism argument above for the one
documented failure it targets; don't cite it as proof of a general
agreement advantage that this project's own data doesn't show.

FEW-SHOT EXAMPLES (Milestone 5.1)

Hindi worked examples below, one of which is the समुद्र तल case itself
-- reused deliberately as the negative example the judge must not
repeat, since it is the one documented failure this project has actual
evidence for, not a hypothetical.

SCRIPT-AWARE PROMPTING (Milestone 5.2)

The rubric states explicitly which script it is reading and that
Romanized Hindi ("Hinglish") is not itself an error. This project's
regeneration-run finding (README.md's headline 20%/100% table) proved
instruction strength changes script behavior drastically -- an
ambiguous instruction produced Devanagari answers to Hinglish prompts
4 times out of 5, a strict one fixed it completely. A judge given no
explicit script guidance is liable to the same ambiguity: penalizing a
correct Romanized-Hindi answer for "not being in Hindi" is exactly the
kind of surface-property confusion this project's other metrics
(script_adherence, Milestone 1) already handle separately -- the judge
rubric must not re-introduce it by penalizing script/register instead
of correctness.

PROMPT-INJECTION SURFACE -- DISCLOSED, NOT FIXED (found by an
independent outside review): build_reference_free_prompt and
build_reference_based_prompt insert `answer` (the text being graded,
which is untrusted model output by definition -- that is the whole
point of grading it) directly into the rubric via `.format()`, with no
delimiters, escaping, or structural separation from the grading
instructions around it. An adversarial or simply confused answer that
happens to contain something that looks like this rubric's own
few-shot format (e.g. ending in "मूल्यांकन: तथ्य सही है। अंक: 5" or a
fabricated JSON score object) sits verbatim, immediately adjacent to
the real grading instruction, with nothing marking where the untrusted
content ends and the instructions resume. For `indic_judge` called
directly on a known answer this is a low-severity concern; it becomes
a real one for `check_trace_llm_fallback` (judge_trace_check.py),
which sends a JUDGE'S OWN reasoning trace -- text an LLM produced,
potentially influenced by injected content earlier in the pipeline --
into a second judge call the same unstructured way. Not fixed here:
a real fix (delimited/escaped answer text, or a structured message
format instead of one flat prompt string) is a rubric-format change
that needs testing against a real judge model's actual behavior, not
a safe drive-by edit. Documented so a caller grading genuinely
untrusted or adversarial input knows this surface exists.
"""

from __future__ import annotations

_FEW_SHOT_HI = """
उदाहरण 1 (सही उत्तर, संदर्भ रहित):
प्रश्न: भारत की राजधानी क्या है?
उत्तर: भारत की राजधानी नई दिल्ली है।
मूल्यांकन: प्रश्न सही ढंग से उत्तरित हुआ। तथ्य सही है। अंक: 5

उदाहरण 2 (गलत उत्तर, अलग तथ्य):
प्रश्न: भारत की राजधानी क्या है?
उत्तर: भारत की राजधानी मुंबई है।
मूल्यांकन: तथ्य गलत है -- भारत की राजधानी नई दिल्ली है, मुंबई नहीं। अंक: 1

उदाहरण 3 (सही उत्तर, रोमन लिपि में हिंग्लिश -- लिपि कोई त्रुटि नहीं है):
प्रश्न: Bharat ki rajdhani kya hai?
उत्तर: Bharat ki rajdhani New Delhi hai.
मूल्यांकन: उत्तर रोमन लिपि में है, लेकिन प्रश्न भी रोमन लिपि में था -- यह
कोई त्रुटि नहीं है। तथ्य सही है। अंक: 5

उदाहरण 4 (महत्वपूर्ण चेतावनी -- अनुवाद त्रुटि से बचें, वास्तविक घटना पर
आधारित):
प्रश्न: समुद्र तल पर पानी किस तापमान पर उबलता है?
उत्तर: समुद्र तल पर पानी 100 डिग्री सेल्सियस पर उबलता है।
गलत मूल्यांकन (मत करें): "समुद्र तल" का अर्थ "sea floor" (समुद्र की
तलहटी) समझकर यह कहना कि दबाव के कारण क्वथनांक 100 डिग्री से अधिक होना
चाहिए -- यह एक अनुवाद त्रुटि है, "समुद्र तल" का सही अर्थ "sea level"
(समुद्र सतह) है, न कि समुद्र की तलहटी।
सही मूल्यांकन: उत्तर तथ्यात्मक रूप से सही है -- समुद्र तल (सतह) पर
मानक वायुमंडलीय दबाव पर पानी 100 डिग्री सेल्सियस पर उबलता है। अंक: 5
"""

RUBRIC_HI_REFERENCE_FREE = """आप एक विशेषज्ञ मूल्यांकनकर्ता हैं। आपका
काम यह जांचना है कि किसी AI सहायक का उत्तर प्रश्न का सही, तथ्यात्मक और
समझदारी भरा जवाब है या नहीं -- कोई संदर्भ उत्तर (reference answer)
नहीं दिया गया है, इसलिए अपने खुद के ज्ञान के आधार पर निर्णय लें।

स्क्रिप्ट/लिपि नोट: प्रश्न और/या उत्तर देवनागरी लिपि में हो सकते हैं,
रोमन लिपि में लिखी हिंदी (हिंग्लिश) में हो सकते हैं, या दोनों लिपियों
के मिश्रण में हो सकते हैं। लिपि या भाषा-मिश्रण अपने आप में कोई त्रुटि
नहीं है -- केवल तथ्यात्मक सटीकता और प्रश्न की समझ का मूल्यांकन करें,
लिपि की नहीं।

महत्वपूर्ण: यदि प्रश्न या उत्तर हिंदी में है, तो अपना तर्क (reasoning)
भी हिंदी में ही लिखें, अंग्रेज़ी में अनुवाद करके मत सोचें -- अनुवाद की
प्रक्रिया में ही अर्थ बदल सकता है (नीचे उदाहरण 4 देखें)।

{few_shot}

अब इस मामले का मूल्यांकन करें:

प्रश्न: {question}

AI सहायक का उत्तर: {answer}

मूल्यांकन मानदंड: क्या यह उत्तर प्रश्न को सही ढंग से, तथ्यात्मक रूप से
और सीधे संबोधित करता है? चरण-दर-चरण सोचें (हिंदी में, यदि प्रश्न/उत्तर
हिंदी में हैं), फिर बिल्कुल इस JSON प्रारूप में उत्तर दें:
{{"reasoning": "<आपका चरण-दर-चरण तर्क, हिंदी में>", "score": <1 से 5 तक
पूर्णांक, जहाँ 5 = पूर्णतः सही, 1 = पूर्णतः गलत>, "confidence": <"high"
या "low" -- यदि आपको उत्तर की सत्यता के बारे में कोई अनिश्चितता है तो
"low" लिखें>}}
केवल JSON ऑब्जेक्ट आउटपुट करें, कुछ और नहीं।"""

RUBRIC_HI_REFERENCE_BASED = """आप एक विशेषज्ञ मूल्यांकनकर्ता हैं। आपका
काम यह जांचना है कि किसी AI सहायक का उत्तर सही है या नहीं, दिए गए
संदर्भ उत्तर (reference answer) की तुलना में।

स्क्रिप्ट/लिपि नोट: प्रश्न, उत्तर, और संदर्भ उत्तर देवनागरी लिपि में
हो सकते हैं, रोमन लिपि में लिखी हिंदी (हिंग्लिश) में हो सकते हैं, या
दोनों के मिश्रण में। लिपि या भाषा-मिश्रण अपने आप में कोई त्रुटि नहीं
है -- केवल तथ्यात्मक सटीकता का मूल्यांकन करें, लिपि की नहीं।

महत्वपूर्ण: यदि सामग्री हिंदी में है, तो अपना तर्क भी हिंदी में लिखें
-- अंग्रेज़ी में अनुवाद करके मत सोचें।

चेतावनी: संदर्भ उत्तर केवल तुलना के लिए एक सहायक है। संदर्भ उत्तर से
मिलते-जुलते शब्दों के आधार पर स्वतः सही न मान लें -- वास्तव में यह
जांचें कि उत्तर का अर्थ और तथ्य संदर्भ से मेल खाते हैं या नहीं।

{few_shot}

अब इस मामले का मूल्यांकन करें:

प्रश्न: {question}

संदर्भ (सही) उत्तर: {gold}

AI सहायक का उत्तर: {answer}

मूल्यांकन मानदंड: क्या सहायक का उत्तर संदर्भ उत्तर के तथ्य से मेल खाता
है? चरण-दर-चरण सोचें (हिंदी में, यदि सामग्री हिंदी में है), फिर बिल्कुल
इस JSON प्रारूप में उत्तर दें:
{{"reasoning": "<आपका चरण-दर-चरण तर्क, हिंदी में>", "score": <1 से 5
तक पूर्णांक, जहाँ 5 = पूर्णतः सही, 1 = पूर्णतः गलत>, "confidence":
<"high" या "low">}}
केवल JSON ऑब्जेक्ट आउटपुट करें, कुछ और नहीं।"""


def build_reference_free_prompt(question: str, answer: str) -> str:
    """Build the reference-free judge prompt (Milestone 5.3's default
    mode) for one (question, answer) pair."""
    return RUBRIC_HI_REFERENCE_FREE.format(
        few_shot=_FEW_SHOT_HI, question=question, answer=answer
    )


def build_reference_based_prompt(question: str, answer: str, gold: str) -> str:
    """Build the reference-based judge prompt for one (question,
    answer, gold) triple. See judge.py's module docstring for why
    reference-free is the DEFAULT mode, not this one -- a gold
    reference measurably masks comprehension drift in this project's
    own data (experiments/FINDINGS.md)."""
    return RUBRIC_HI_REFERENCE_BASED.format(
        few_shot=_FEW_SHOT_HI, question=question, answer=answer, gold=gold
    )
