"""Explicit, frozen QA prompt profiles for historical candidate replication."""

import hashlib
import json
from pathlib import Path
from string import Formatter

from .answer_extractor import extract_final_text_answer


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def profile_hash(profile: dict) -> str:
    return text_hash(json.dumps(profile, sort_keys=True, ensure_ascii=False))


def validate_profile(profile: dict) -> dict:
    for name in ("version", "round_1", "round_2", "format_repair", "completion_suffix"):
        if not isinstance(profile.get(name), str) or not profile[name]:
            raise ValueError(f"QA profile is missing {name}")
    digest = text_hash(profile["round_1"] + "\n" + profile["round_2"])
    if digest != profile.get("reflection_sha256"):
        raise ValueError("QA reflection profile hash mismatch")
    if text_hash(profile["format_repair"]) != profile.get("format_repair_sha256"):
        raise ValueError("QA format repair profile hash mismatch")
    for name in ("round_1", "round_2", "format_repair"):
        fields = {field for _, field, _, _ in Formatter().parse(profile[name]) if field}
        if not fields <= {"question", "current_answer", "draft_answer"}:
            raise ValueError(f"Unsupported profile fields: {fields}")
        if "question" not in fields:
            raise ValueError(f"QA profile {name} has no question field")
    return profile


def load_profile(value: str | Path | dict | None) -> dict | None:
    if value is None:
        return None
    profile = value if isinstance(value, dict) else json.loads(Path(value).read_text())
    return validate_profile(profile)


def profile_prompt(profile: dict, question: str, current_answer: str, round_id: int) -> str:
    span = (extract_final_text_answer(current_answer) or current_answer).strip()
    return profile[f"round_{2 if round_id >= 2 else 1}"].format(
        question=question, current_answer=f"Final answer: {span}"
    )
