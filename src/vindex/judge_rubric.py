"""Hindi-language rubric and script-aware prompts for indic_judge.

The rubric is written in Hindi and instructs the judge to reason in
Hindi when the content is Hindi, so there is no internal translation
step for errors to hide in. This targets a specific observed failure: an
English-reasoning judge mistranslated समुद्र तल ("sea level") as "sea
floor" mid-reasoning and marked a correct answer about water boiling at
100 C wrong. That case is included as a negative few-shot example. The
rubric does not, on its own, agree with human graders more than an
English rubric does (see :mod:`vindex.judge`).

The rubric also states that Romanized Hindi (Hinglish) and script mixing
are not errors, so the judge scores correctness rather than script;
script checking is handled separately by :func:`vindex.script_adherence`.

Security note: the answer text is inserted into the prompt verbatim with
``str.format()``, without delimiters or escaping. An answer that imitates
the rubric's few-shot format or contains a fabricated JSON score could
influence the judge. This matters most when grading untrusted or
adversarial output, and for :func:`vindex.check_trace_llm_fallback`,
which feeds an LLM-generated reasoning trace into a second judge call.
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
    """Build the reference-free (default) judge prompt.

    Args:
        question: The question asked.
        answer: The answer to grade.

    Returns:
        The complete prompt string.
    """
    return RUBRIC_HI_REFERENCE_FREE.format(
        few_shot=_FEW_SHOT_HI, question=question, answer=answer
    )


def build_reference_based_prompt(question: str, answer: str, gold: str) -> str:
    """Build the reference-based judge prompt.

    Reference-free is the default in :func:`vindex.indic_judge` because a
    gold reference can mask comprehension drift in the judge's reasoning.

    Args:
        question: The question asked.
        answer: The answer (or its mismatched spans) to grade.
        gold: The reference answer (or its mismatched spans).

    Returns:
        The complete prompt string.
    """
    return RUBRIC_HI_REFERENCE_BASED.format(
        few_shot=_FEW_SHOT_HI, question=question, answer=answer, gold=gold
    )
