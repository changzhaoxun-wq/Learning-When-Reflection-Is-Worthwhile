"""Prompt templates used by all current experiments.

The math direct prompt is frozen for comparable Base traces. Reflection
prompts are independently versioned so they can be improved without changing
the Base generation protocol. QA has a separate short-answer template.
"""

MATH_PROMPT_VERSION = "gsm8k_policy_compat_v1"
MATH_REFLECTION_PROMPT_VERSION = "gsm8k_semantic_trap_repair_v4"
QA_PROMPT_VERSION = "qa_direct_first_v2"
QA_REFLECTION_PROMPT_VERSION = "qa_target_aware_answer_repair_v7"
QA_FORMAT_REPAIR_PROMPT_VERSION = "qa_minimal_answer_format_v6"
QA_TRUNCATION_RECOVERY_PROMPT_VERSION = "qa_truncation_recovery_v2"

MATH_DIRECT_PROMPT = """You are solving a grade school math problem.
Question:
{question}
Show concise reasoning and end with: Final answer: <number>"""

_MATH_DIRECT_FORBIDDEN_CUES = (
    "/no_think",
    "reflect",
    "verify",
    "check the",
    "double-check",
    "wrong",
    "incorrect",
    "correct the",
    "improve",
    "revise",
    "critique",
    "previous answer",
    "previous trial",
    "current answer",
)


def validate_math_direct_prompt() -> None:
    template = MATH_DIRECT_PROMPT.lower()
    forbidden = [cue for cue in _MATH_DIRECT_FORBIDDEN_CUES if cue in template]
    if forbidden:
        raise ValueError(
            "Math direct prompt must not contain reflection or correction cues: "
            + ", ".join(forbidden)
        )
    for required in ("{question}", "final answer: <number>"):
        if required not in template:
            raise ValueError(f"Math direct prompt is missing required content: {required}")


validate_math_direct_prompt()

MATH_REFLECTION_PROMPT_ROUND_1 = """
Solve this grade-school math problem independently. No earlier answer is shown.
Accuracy is more important than agreeing with an obvious-looking interpretation.

Question:
{question}

Before calculating, translate every sentence into an exact algebraic relation.
1. Name the requested quantity and its unit.
2. Bind each number to its entity, unit, time interval, and operation.
3. Resolve language before arithmetic: distinguish a bundle price from a per-item
   price, elapsed periods from endpoints, a round trip from one-way distance,
   paid items from free items, a fixed expense from a changing income, and a
   daily or monthly target from a cumulative target.
4. Write one equation chain that uses every relevant condition exactly once.
5. Reject the equation if it produces an impossible sign, unit, or magnitude.

Use exactly this plain-text format, with no Markdown or LaTeX:
Asked: <requested quantity and unit>
Relations: <compact sentence-to-equation translation>
Equation: <one equation chain>
Check: <independent substitution, reverse operation, or sanity check>
Final answer: <bare number>

The final line must be last and must literally begin `Final answer:`. After the
colon write only an integer, decimal, or negative number: no unit, words,
commas, asterisks, dollar signs, boxes, or extra line."""

MATH_REFLECTION_PROMPT_ROUND_2 = """
Solve this grade-school math problem again from scratch as a semantic stress test.
The earlier repair is deliberately hidden so that it cannot anchor this attempt.
Assume a fluent but wrong solver chose the most tempting interpretation.

Question:
{question}

Test competing interpretations before choosing an equation:
- `N items for $X` normally gives one bundle total; `N items at $X each` gives
  a per-item price.
- A fixed bill that becomes a different percentage after an income change has
  the same dollar amount before and after.
- `every nth customer gets one free` separates paying items from free items;
  revenue counts only paid items, while the nth position counts all customers.
- A yearly or monthly target plus earlier shortfalls requires the current target
  and every unpaid shortfall; saving during off-season months uses 12 minus the
  season length.
- The gap from anniversary a to anniversary b is `(b-a)*12` months. A stated
  future offset must also be placed on the timeline.
- `D miles back and forth` means D outbound and D return unless the question
  explicitly says D is the total round-trip distance.
- For nested phrases such as `k less than m times as many as A`, compute A
  first, then `m*A-k`.
- For percentages, include the complement group and distinguish final amount,
  increase, and difference.

Choose the interpretation that uses every condition once and survives units,
positivity, substitution, and magnitude checks.

Use exactly this plain-text format, with no Markdown or LaTeX:
Asked: <requested quantity and unit>
Semantic trap: <which tempting interpretation was rejected>
Equation: <complete alternative equation chain>
Check: <independent substitution, reverse operation, or sanity check>
Final answer: <bare number>

The final line must be last and must literally begin `Final answer:`. After the
colon write only an integer, decimal, or negative number: no unit, words,
commas, asterisks, dollar signs, boxes, or extra line."""

QA_DIRECT_PROMPT = """/no_think
Answer the question using only the provided context.

{question}

Return exactly one final line in this format:
Final answer: <minimal answer span>

After the colon, write only the short answer: no explanation, evidence, Markdown,
quotation marks, or extra sentence."""

QA_REFLECTION_PROMPT_ROUND_1 = """/no_think
Audit an existing answer to this multi-hop question using only the provided
context.

{question}

Current final answer:
{current_answer}

Silently perform this comparison before deciding:
1. Parse the final question clause and determine the exact requested answer type
   and granularity.
2. Follow every bridge relation and independently derive the shortest
   context-supported candidate.
3. Compare that candidate with the current answer.

REVISE when the current answer is a bridge entity instead of the final target,
has the wrong answer type, chooses the wrong comparison entity, or is less
specific than the explicitly named answer requested by the question. KEEP when
it is semantically equivalent to the independently derived candidate. If the
context cannot establish a different candidate, KEEP.

Target rules:
- For `which of A or B`, return exactly A or B.
- For `who`, return the requested person or organization, never a date or a
  sentence about it.
- For `where`, return the most specific named place or venue directly attached
  to the event.
- For `when`, return only the requested date or time.
- For a property, role, reason, or commonality question, return that property,
  role, reason, or shared fact rather than the entity used to identify it.
- Do not copy an answer merely because it is mentioned in the context.

Complete exactly these two fields. The first field is already started by the
assistant prefix:
Decision: <KEEP or REVISE>
Final answer: <minimal answer span>

After `Final answer:`, write only the short answer, with no explanation,
Markdown, quotation marks, alternatives, or trailing sentence."""

QA_REFLECTION_PROMPT_ROUND_2 = """/no_think
Perform a second target-aware audit of the current final answer using only the
provided context.

{question}

Current final answer:
{current_answer}

Independently reconstruct the answer from the final interrogative clause. Check
that every bridge keeps the same entity and that the answer has the requested
type and specificity. Explicitly test whether the current span is merely a
person, work, place, date, title, or relative-clause clue used to identify the
actual target. For comparisons, evaluate both named alternatives and return only
the selected alternative. For a venue nested inside a city, use the most
specific named location supported for the event.

REVISE only to one complete context-supported replacement. KEEP when the current
span is semantically equivalent to the independently reconstructed answer or no
specific replacement is established.

Complete exactly these two fields. The first field is already started by the
assistant prefix:
Decision: <KEEP or REVISE>
Final answer: <minimal answer span>

After `Final answer:`, write only the shortest answer span, with no explanation,
Markdown, quotation marks, alternatives, or trailing sentence."""

QA_FORMAT_REPAIR_PROMPT = """/no_think
Convert a reflection draft into one minimal answer span. Do not solve the
question again and do not explain the result.

Question and context:
{question}

Draft answer:
{draft_answer}

Current answer before reflection:
{current_answer}

If the draft says KEEP, correct, supported, or no revision, copy the current
answer span. If the draft explicitly revises the answer, extract the replacement
named by the draft. Do not extract a date, person, place, or intermediate entity
that has the wrong type for the final question.

Return only the answer span itself, normally one to eight words. Do not write
`Final answer:`, a sentence, an explanation, evidence, Markdown, quotation
marks, angle-bracket placeholders, or a second line. The assistant response is
already prefixed with `Answer span:`; continue only that field. Match the
question's requested type: for `who` or `which person/player`, return a name,
not a title count; for `where`, return a place; for `when`, return a date."""

QA_TRUNCATION_RECOVERY_PROMPT = """/no_think
Answer this multi-hop question independently using only the provided context.
The earlier generation became repetitive and is not a usable answer, so ignore it.

{question}

Silently identify the main interrogative target, then use descriptive or relative
clauses only as bridge evidence. In particular, if asked what an abbreviation
stands for and a person is given to identify the company, return the expanded
name, not the person or company. Return only the shortest supported answer span,
normally one to eight words. Do not explain, list alternatives, or repeat text. Do not write `Final answer:` because the assistant response is
already prefixed with `Answer span:`; continue only that field."""

MATH_VERIFY_PROMPT = """/no_think
You are a strict verifier for a math problem.
Evaluate whether the proposed answer correctly solves the question.

Question:
{question}

Proposed answer:
{answer}

Return ONLY a JSON object with this schema:
{{"score": <float from 0.0 to 1.0>, "verdict": "correct|incorrect|uncertain", "reason": "brief reason"}}

Scoring guide:
- 0.90 to 1.00: clearly correct.
- 0.60 to 0.89: likely correct but with minor uncertainty.
- 0.30 to 0.59: uncertain or incomplete.
- 0.00 to 0.29: likely incorrect or no valid final answer.
Do not reveal a new solution; only verify the proposed answer."""

QA_VERIFY_PROMPT = """/no_think
You are a strict verifier for a question answering task.
Evaluate whether the proposed short answer is supported by the question and context.

Question:
{question}

Proposed answer:
{answer}

Return ONLY a JSON object with this schema:
{{"score": <float from 0.0 to 1.0>, "verdict": "correct|incorrect|uncertain", "reason": "brief reason"}}

Scoring guide:
- 0.90 to 1.00: clearly correct and supported.
- 0.60 to 0.89: likely correct but evidence is incomplete.
- 0.30 to 0.59: uncertain, incomplete, or ambiguous.
- 0.00 to 0.29: likely incorrect, unsupported, or no valid final answer.
Do not reveal a new answer; only verify the proposed answer."""


def prompt_version(task_type: str | None = None) -> str:
    return QA_PROMPT_VERSION if str(task_type or "math").lower() == "qa" else MATH_PROMPT_VERSION


def reflection_prompt_version(task_type: str | None = None) -> str:
    if str(task_type or "math").lower() == "qa":
        return QA_REFLECTION_PROMPT_VERSION
    return MATH_REFLECTION_PROMPT_VERSION


def direct_prompt(question: str, task_type: str | None = None) -> str:
    if str(task_type or "math").lower() == "qa":
        return QA_DIRECT_PROMPT.format(question=question)
    return MATH_DIRECT_PROMPT.format(question=question)


def reflection_prompt(question: str, current_answer: str, round_id: int, task_type: str | None = None) -> str:
    if str(task_type or "math").lower() == "qa":
        template = QA_REFLECTION_PROMPT_ROUND_2 if int(round_id) >= 2 else QA_REFLECTION_PROMPT_ROUND_1
        return template.format(
            question=question,
            current_answer=current_answer,
        )
    if int(round_id) >= 2:
        return MATH_REFLECTION_PROMPT_ROUND_2.format(question=question)
    return MATH_REFLECTION_PROMPT_ROUND_1.format(question=question)


def qa_truncation_recovery_prompt(question: str) -> str:
    return QA_TRUNCATION_RECOVERY_PROMPT.format(question=question)


def qa_format_repair_prompt(
    question: str,
    draft_answer: str,
    current_answer: str,
) -> str:
    return QA_FORMAT_REPAIR_PROMPT.format(
        question=question,
        draft_answer=draft_answer,
        current_answer=current_answer,
    )


def verify_prompt(question: str, answer: str, task_type: str | None = None) -> str:
    template = QA_VERIFY_PROMPT if str(task_type or "math").lower() == "qa" else MATH_VERIFY_PROMPT
    return template.format(question=question, answer=answer)
