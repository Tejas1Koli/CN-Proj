"""Benchmark tasks: dataset loading, prompt construction and scoring.

Every task exposes the same interface so the runner can treat them uniformly:

    Item -> {item_id, prompt, gold, meta}
    Task -> {name, items, scorer, max_output_tokens}

Prompts follow the standard lm-evaluation-harness conventions (5-shot MMLU,
5-shot GSM8K, 0-shot HumanEval, 3-shot MBPP, 3-shot BoolQ, 5-shot ARC-Challenge)
so the numbers are comparable with published results.

Few-shot exemplars are always drawn from the *train* split and are fixed for
every model, so all three systems see byte-identical prompts.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from .config import HF_CACHE, SAMPLES, SEED
from .exec_sandbox import run_python

LETTERS = "ABCDEFGHIJ"


def choice_directive(n_choices: int) -> str:
    """Instruction line that stops small models from preambling instead of answering.

    Without it, Gemma 4 E2B answers a 5-shot multiple-choice prompt with
    "Here are the solutions and explanations for the provided multiple-choice
    questions: ..." and never reaches a letter inside the token budget. With
    it, the model replies with the bare letter.
    """
    options = ", ".join(LETTERS[i] for i in range(n_choices))
    return (
        f"Answer the following multiple-choice question with a single letter "
        f"({options}). Do not explain.\n\n"
    )


@dataclass
class Item:
    item_id: str
    prompt: str
    gold: str
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_hash(self) -> str:
        """Fingerprint of the exact prompt text.

        Recorded with every result so that changing a prompt invalidates the
        old records instead of silently mixing two different prompts under the
        same item id.
        """
        return hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()[:12]


@dataclass
class Task:
    name: str
    items: list[Item]
    scorer: Callable[[str, Item], bool]
    max_output_tokens: int
    description: str = ""


# --------------------------------------------------------------------- utils
def _parquet(pattern: str) -> pd.DataFrame:
    """Load the first parquet file matching a pattern relative to the HF cache.

    The pattern may contain a `*` for the snapshot-hash directory, e.g.
    "datasets--cais--mmlu/snapshots/*/all/dev-*.parquet".
    """
    matches = sorted(HF_CACHE.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no parquet matching {pattern} under {HF_CACHE}")
    return pd.read_parquet(matches[0])


def _sample(df: pd.DataFrame, n: int, seed: int = SEED) -> pd.DataFrame:
    if n >= len(df):
        return df.reset_index(drop=True)
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def _normalize_number(text: str) -> float | None:
    text = text.replace(",", "").replace("$", "").strip()
    match = re.search(r"-?\d+\.?\d*", text)
    return float(match.group()) if match else None


# -------------------------------------------------------------------- scoring
def score_letter(text: str, item: Item) -> bool:
    """Extract a single A-J choice from a completion.

    Only an explicit answer marker counts. A bare "first standalone letter"
    fallback is deliberately avoided: prose such as "I think it is C"
    contains the standalone pronoun "I", which is inside the A-J range and
    would be scored as a choice.
    """
    upper = text.strip().upper()
    patterns = [
        r"ANSWER\s*(?:IS|:|-|=)\s*[\(\[\*]*\s*([A-J])\b",
        r"OPTION\s*\(?([A-J])\b",
        r"^\s*[\(\[\*]*\s*([A-J])\s*[\)\].\*]*\s*$",
        r"^\s*[\(\[\*]*\s*([A-J])\s*[\)\].:]",
    ]
    for pattern in patterns:
        match = re.search(pattern, upper)
        if match:
            return match.group(1) == item.gold
    return False


def score_yesno(text: str, item: Item) -> bool:
    upper = text.strip().upper()
    match = re.search(r"ANSWER\s*[:\-]?\s*\**\s*(YES|NO)\b", upper)
    if not match:
        match = re.search(r"\b(YES|NO)\b", upper)
    if not match:
        return False
    return match.group(1).lower() == item.gold.lower()


def score_number(text: str, item: Item) -> bool:
    gold = _normalize_number(item.gold)
    if gold is None:
        return False
    tail = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if not tail:
        return False
    predicted = _normalize_number(tail[-1])
    if predicted is None:
        return False
    return abs(predicted - gold) < 1e-4


def extract_code(text: str) -> str:
    """Pull the most plausible code block out of a completion."""
    # The MBPP prompt format uses [BEGIN]/[END] to delimit the answer, which
    # is not valid Python - strip the markers before anything else.
    text = text.replace("[BEGIN]", "").replace("[END]", "")
    blocks = re.findall(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", text, re.S)
    if not blocks:
        blocks = re.findall(r"```(?:[a-zA-Z0-9_+-]*)\s*\n(.*)", text, re.S)
    if not blocks:
        return text
    with_def = [b for b in blocks if re.search(r"^\s*def\s", b, re.M)]
    if with_def:
        return max(with_def, key=len)
    return max(blocks, key=len)


def score_code(text: str, item: Item) -> bool:
    """Run the extracted program against the task's unit tests.

    `meta["prefix"]` is prepended and `meta["suffix"]` appended to the
    candidate code, so the same scorer works for HumanEval (which needs a
    `check(entry_point)` call) and MBPP (which needs bare asserts).
    """
    code = extract_code(text)
    prefix = item.meta.get("prefix", "") or ""
    suffix = item.meta.get("suffix", "") or ""
    entry_point = item.meta.get("entry_point")

    candidates: list[str] = []

    if entry_point:
        has_def = re.search(rf"^\s*def\s+{re.escape(entry_point)}\b", code, re.M)
        if has_def:
            candidates.append(code)
        # Treat the completion as the function body appended to the signature.
        candidates.append(item.meta["base_prompt"] + code)
        candidates.append(
            item.meta["base_prompt"]
            + "\n"
            + "\n".join("    " + ln if ln.strip() else ln for ln in code.splitlines())
        )
    else:
        candidates.append(code)

    # Join the candidate to its test harness with a newline. Without this, a
    # completion that does not end in a newline is concatenated directly onto
    # the first assertion ("return x" + "assert ..." -> SyntaxError), which
    # silently marks correct programs wrong. HumanEval's suffix happens to
    # start with a newline, so only MBPP was affected -- but the bug is in the
    # join, not in the data.
    if suffix and not suffix.startswith("\n"):
        suffix = "\n" + suffix

    programs = [prefix + c.rstrip() + "\n" + suffix for c in candidates]
    programs.append(prefix + code.rstrip() + "\n" + suffix)

    for program in programs:
        try:
            passed, _ = run_python(program)
        except Exception:
            passed = False
        if passed:
            return True
    return False


# ---------------------------------------------------------------------- tasks
def build_mmlu(n: int | None = None) -> Task:
    n = n or SAMPLES["mmlu"]
    dev = _parquet("datasets--cais--mmlu/snapshots/*/all/dev-*.parquet")
    test = _parquet("datasets--cais--mmlu/snapshots/*/all/test-*.parquet")

    def block(row, with_answer: bool = False) -> str:
        choices = "\n".join(
            f"{LETTERS[i]}. {c}" for i, c in enumerate(row["choices"])
        )
        answer = f" {LETTERS[int(row['answer'])]}" if with_answer else ""
        return f"{row['question'].strip()}\n{choices}\nAnswer:{answer}"

    # The few-shot exemplars MUST carry their gold answers, otherwise the
    # model imitates the "Answer:" pattern without ever producing a letter.
    shots: dict[str, str] = {}
    for subject, group in dev.groupby("subject"):
        shots[subject] = "\n\n".join(
            block(r, with_answer=True) for _, r in group.head(5).iterrows()
        )

    subjects = sorted(test["subject"].unique())
    picked: list[pd.Series] = []
    per_subject = max(1, n // len(subjects))
    for subject in subjects:
        rows = test[test["subject"] == subject]
        picked.append(rows.sample(n=min(per_subject, len(rows)), random_state=SEED))
    frame = pd.concat(picked)
    if len(frame) < n:
        extra = test.drop(frame.index).sample(n=n - len(frame), random_state=SEED)
        frame = pd.concat([frame, extra])
    frame = frame.head(n)

    items = []
    for idx, row in frame.iterrows():
        header = (
            "The following are multiple choice questions (with answers) about "
            f"{row['subject'].replace('_', ' ')}.\n\n"
        )
        body = shots.get(row["subject"], "") + "\n\n" + block(row)
        prompt = choice_directive(len(row["choices"])) + header + body
        items.append(
            Item(
                item_id=f"mmlu/{row['subject']}/{idx}",
                prompt=prompt,
                gold=LETTERS[int(row["answer"])],
                meta={"subject": row["subject"]},
            )
        )
    return Task(
        "mmlu",
        items,
        score_letter,
        max_output_tokens=24,
        description="MMLU (5-shot, subject-stratified) - broad academic knowledge",
    )


def build_gsm8k(n: int | None = None) -> Task:
    n = n or SAMPLES["gsm8k"]
    train = _parquet("datasets--openai--gsm8k/snapshots/*/main/train-*.parquet")
    test = _parquet("datasets--openai--gsm8k/snapshots/*/main/test-*.parquet")

    def block(row) -> str:
        rationale = re.sub(r"<<[^>]*>>", "", row["answer"])
        rationale, _, final = rationale.partition("####")
        return (
            f"Question: {row['question'].strip()}\n"
            f"Answer: {rationale.strip()} The answer is {final.strip()}."
        )

    shots = "\n\n".join(block(r) for _, r in train.head(5).iterrows())
    frame = _sample(test, n)

    items = []
    for idx, row in frame.iterrows():
        prompt = f"Question: {row['question'].strip()}\nAnswer:"
        items.append(
            Item(
                item_id=f"gsm8k/{idx}",
                prompt=shots + "\n\n" + prompt,
                gold=row["answer"].split("####")[-1].strip(),
                meta={},
            )
        )
    return Task(
        "gsm8k",
        items,
        score_number,
        max_output_tokens=256,
        description="GSM8K (5-shot) - grade-school math word problems",
    )


def build_humaneval(n: int | None = None) -> Task:
    n = n or SAMPLES["humaneval"]
    frame = _parquet("datasets--openai--openai_humaneval/snapshots/*/openai_humaneval/*.parquet")
    frame = frame.head(n)

    items = []
    for _, row in frame.iterrows():
        prompt = (
            "Complete the following Python function. Reply with the complete "
            "function inside one ```python code block.\n\n```python\n"
            f"{row['prompt']}```"
        )
        items.append(
            Item(
                item_id=row["task_id"],
                prompt=prompt,
                gold=row["task_id"],
                meta={
                    "entry_point": row["entry_point"],
                    "base_prompt": row["prompt"],
                    "prefix": "",
                    "suffix": f"\n\n{row['test']}\ncheck({row['entry_point']})\n",
                },
            )
        )
    return Task(
        "humaneval",
        items,
        score_code,
        max_output_tokens=384,
        description="HumanEval (0-shot, pass@1) - Python code generation",
    )


def build_mbpp(n: int | None = None) -> Task:
    n = n or SAMPLES["mbpp"]
    train = _parquet("datasets--google-research-datasets--mbpp/snapshots/*/full/train-*.parquet")
    test = _parquet("datasets--google-research-datasets--mbpp/snapshots/*/full/test-*.parquet")

    def block(row) -> str:
        tests = "\n".join(row["test_list"])
        return (
            "You are an expert Python programmer, and here is your task: "
            f"{row['text'].strip()} Your code should pass these tests:\n\n"
            f"{tests}\n"
            f"[BEGIN]\n{row['code'].strip()}\n[END]"
        )

    # The [BEGIN]/[END] delimiters are what makes the few-shot block
    # unambiguous: without them the model continues the pattern by answering
    # one of the *exemplar* tasks instead of the query (measured pass@1 on the
    # local model: 0/25 with a prose format, 10/25 with this one).
    shots = "\n\n".join(block(r) for _, r in train.head(3).iterrows())
    frame = _sample(test, n)

    items = []
    for _, row in frame.iterrows():
        tests = "\n".join(row["test_list"])
        prompt = (
            "You are an expert Python programmer, and here is your task: "
            f"{row['text'].strip()} Your code should pass these tests:\n\n"
            f"{tests}\n"
        )
        items.append(
            Item(
                item_id=f"mbpp/{row['task_id']}",
                prompt=shots + "\n\n" + prompt,
                gold=row["task_id"],
                meta={
                    "prefix": (row.get("test_setup_code", "") or "") + "\n",
                    "suffix": "\n".join(row["test_list"]) + "\n",
                    "entry_point": None,
                },
            )
        )
    return Task(
        "mbpp",
        items,
        score_code,
        max_output_tokens=384,
        description="MBPP (3-shot, pass@1) - mostly basic Python problems",
    )


def build_boolq(n: int | None = None) -> Task:
    n = n or SAMPLES["boolq"]
    train = _parquet("datasets--google--boolq/snapshots/*/data/train-*.parquet")
    valid = _parquet("datasets--google--boolq/snapshots/*/data/validation-*.parquet")

    def block(row) -> str:
        return (
            f"Passage: {row['passage'].strip()}\n"
            f"Question: {row['question'].strip()}?\n"
            f"Answer: {'yes' if row['answer'] else 'no'}"
        )

    shots = "\n\n".join(block(r) for _, r in train.head(3).iterrows())
    frame = _sample(valid, n)

    items = []
    for idx, row in frame.iterrows():
        prompt = (
            f"Passage: {row['passage'].strip()}\n"
            f"Question: {row['question'].strip()}?\nAnswer:"
        )
        items.append(
            Item(
                item_id=f"boolq/{idx}",
                prompt=shots + "\n\n" + prompt,
                gold="yes" if row["answer"] else "no",
                meta={},
            )
        )
    return Task(
        "boolq",
        items,
        score_yesno,
        max_output_tokens=8,
        description="BoolQ (3-shot) - passage reading comprehension, yes/no",
    )


def build_arc_challenge(n: int | None = None) -> Task:
    n = n or SAMPLES["arc_challenge"]
    train = _parquet("datasets--allenai--ai2_arc/snapshots/*/ARC-Challenge/train-*.parquet")
    test = _parquet("datasets--allenai--ai2_arc/snapshots/*/ARC-Challenge/test-*.parquet")

    def block(row) -> str:
        labels = list(row["choices"]["label"])
        texts = list(row["choices"]["text"])
        choices = "\n".join(f"{la}. {t}" for la, t in zip(labels, texts))
        return f"Question: {row['question'].strip()}\n{choices}\nAnswer: {row['answerKey']}"

    shots = "\n\n".join(block(r) for _, r in train.head(5).iterrows())
    frame = _sample(test, n)

    items = []
    for _, row in frame.iterrows():
        labels = list(row["choices"]["label"])
        texts = list(row["choices"]["text"])
        choices = "\n".join(f"{la}. {t}" for la, t in zip(labels, texts))
        prompt = f"Question: {row['question'].strip()}\n{choices}\nAnswer:"
        gold = str(row["answerKey"]).strip()
        # Some rows use numeric labels; map them onto letters.
        if gold.isdigit():
            gold = LETTERS[int(gold) - 1]
        items.append(
            Item(
                item_id=f"arc_c/{row['id']}",
                prompt=choice_directive(len(labels)) + shots + "\n\n" + prompt,
                gold=gold,
                meta={},
            )
        )
    return Task(
        "arc_challenge",
        items,
        score_letter,
        max_output_tokens=24,
        description="ARC-Challenge (5-shot) - grade-school science reasoning",
    )


TASK_BUILDERS: dict[str, Callable[[int | None], Task]] = {
    "mmlu": build_mmlu,
    "gsm8k": build_gsm8k,
    "humaneval": build_humaneval,
    "mbpp": build_mbpp,
    "boolq": build_boolq,
    "arc_challenge": build_arc_challenge,
}


def build_all(names: list[str] | None = None) -> list[Task]:
    names = names or list(TASK_BUILDERS)
    return [TASK_BUILDERS[name]() for name in names]
