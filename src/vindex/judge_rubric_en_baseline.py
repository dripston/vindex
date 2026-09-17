"""
English-rubric baseline judge prompt, for Milestone 5.8 ONLY.

This is not a shipped metric. `indic_judge` (vindex.judge) uses the
Hindi rubric in judge_rubric.py as its only rubric -- that choice is
deliberate and documented there. This module exists solely to answer
Milestone 5.8's question: does the Hindi rubric actually agree with
human graders more than an English rubric pointed at the same Hindi
content would, on the same 62 pinned traces from the 6.3 study?

Structurally identical to judge_rubric.py -- same instructions, same
four few-shot examples (including the समुद्र तल "sea level"/"sea
floor" case, translated), same JSON output contract, same conservative
scoring -- with exactly one variable changed: the rubric text and the
few-shot reasoning are in English instead of Hindi, and there is no
instruction to reason in Hindi. That is the single difference this
baseline is meant to isolate.
"""

from __future__ import annotations

_FEW_SHOT_EN = """
Example 1 (correct answer, reference-free):
Question: What is the capital of India?
Answer: The capital of India is New Delhi.
Evaluation: The question is answered correctly. The fact is correct. Score: 5

Example 2 (wrong answer, different fact):
Question: What is the capital of India?
Answer: The capital of India is Mumbai.
Evaluation: The fact is wrong -- the capital of India is New Delhi, not Mumbai. Score: 1

Example 3 (correct answer, Romanized Hindi -- script is not an error):
Question: Bharat ki rajdhani kya hai?
Answer: Bharat ki rajdhani New Delhi hai.
Evaluation: The answer is in Roman script, but so was the question -- this is
not an error. The fact is correct. Score: 5

Example 4 (important warning -- avoid a translation error, based on a real
documented incident):
Question: समुद्र तल पर पानी किस तापमान पर उबलता है?
Answer: समुद्र तल पर पानी 100 डिग्री सेल्सियस पर उबलता है।
Incorrect evaluation (do not do this): interpreting "समुद्र तल" as "sea
floor" and concluding the boiling point should be higher than 100C due to
pressure -- this is a translation error. "समुद्र तल" correctly means "sea
level," not the ocean floor.
Correct evaluation: the answer is factually correct -- at sea level, under
standard atmospheric pressure, water boils at 100 degrees Celsius. Score: 5
"""

RUBRIC_EN_REFERENCE_FREE = """You are an expert evaluator. Your job is to
check whether an AI assistant's answer is a correct, factual, and sensible
response to the question -- no reference answer is given, so judge based on
your own knowledge.

Script note: the question and/or answer may be in Devanagari script,
Romanized Hindi (Hinglish), or a mix of both. Script or language-mixing is
not itself an error -- evaluate only factual accuracy and comprehension of
the question, not script.

{few_shot}

Now evaluate this case:

Question: {question}

AI assistant's answer: {answer}

Evaluation criteria: does this answer correctly, factually, and directly
address the question? Think step by step, then respond in exactly this JSON
format:
{{"reasoning": "<your step-by-step reasoning>", "score": <integer 1 to 5,
where 5 = fully correct, 1 = fully wrong>, "confidence": <"high" or "low" --
write "low" if you have any uncertainty about the truth of the answer>}}
Output only the JSON object, nothing else."""


def build_reference_free_prompt_en_baseline(question: str, answer: str) -> str:
    """Build the English-rubric reference-free judge prompt (Milestone
    5.8's baseline ONLY -- not used by indic_judge). Structurally
    identical to judge_rubric.build_reference_free_prompt except the
    rubric language is English, isolating rubric language as the one
    variable under test."""
    return RUBRIC_EN_REFERENCE_FREE.format(
        few_shot=_FEW_SHOT_EN, question=question, answer=answer
    )
