"""English-rubric baseline prompt for comparison with the Hindi rubric.

Not used by :func:`vindex.indic_judge`, which uses only the Hindi rubric
in :mod:`vindex.judge_rubric`. This baseline is structurally identical
(same instructions, the same four few-shot examples including the
समुद्र तल case, the same JSON output contract) except that the rubric and
few-shot reasoning are in English and there is no instruction to reason
in Hindi. Use it to measure the effect of rubric language on your own
data, e.g. with :func:`vindex.datasets.score_judge_benchmark`.
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
    """Build the English-rubric reference-free baseline prompt.

    Args:
        question: The question asked.
        answer: The answer to grade.

    Returns:
        The complete prompt string.
    """
    return RUBRIC_EN_REFERENCE_FREE.format(
        few_shot=_FEW_SHOT_EN, question=question, answer=answer
    )
