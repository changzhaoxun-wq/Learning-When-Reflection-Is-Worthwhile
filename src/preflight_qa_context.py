from __future__ import annotations

import argparse
import heapq
import json
import random
from pathlib import Path
from typing import Any

from .gsm8k_utils import normalize_example_item
from .prompts import direct_prompt, reflection_prompt


def _token_count(tokenizer: Any, prompt: str, api_mode: str, prefix: str, suffix: str) -> int:
    if api_mode == "completions":
        return len(tokenizer.encode(f"{prefix}{prompt}{suffix}", add_special_tokens=False))
    messages = [{"role": "user", "content": prompt}]
    try:
        tokens = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        tokens = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    if hasattr(tokens, "get") and tokens.get("input_ids") is not None:
        tokens = tokens["input_ids"]
    if hasattr(tokens, "shape") and len(tokens.shape) > 1:
        return int(tokens.shape[-1])
    if tokens and isinstance(tokens[0], list):
        return len(tokens[0])
    return len(tokens)


def _load_tokenizer(name: str) -> Any:
    from transformers import AutoTokenizer

    path = Path(name)
    return AutoTokenizer.from_pretrained(
        str(path if path.exists() else name),
        local_files_only=path.exists(),
        trust_remote_code=True,
    )


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    base_tokenizer = _load_tokenizer(args.base_tokenizer)
    reflector_tokenizer = _load_tokenizer(args.reflector_tokenizer)
    rng = random.Random(args.seed)
    longest: list[tuple[int, int, str]] = []
    sample: list[str] = []
    rows = 0
    max_words = 0
    full_max_direct = 0
    full_max_reflection = 0

    with Path(args.data).open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            item = json.loads(line)
            row = normalize_example_item(item, index, args.split)
            if row.get("task_type") != "qa":
                raise ValueError(f"Expected task_type=qa, got {row.get('task_type')} at row {index + 1}")
            question = row["question"]
            rows += 1
            max_words = max(max_words, len(question.split()))
            if args.full_scan:
                full_max_direct = max(
                    full_max_direct,
                    _token_count(base_tokenizer, direct_prompt(question, "qa"), "chat_completions", "", ""),
                )
                for round_id in (1, 2):
                    full_max_reflection = max(
                        full_max_reflection,
                        _token_count(
                            reflector_tokenizer,
                            reflection_prompt(question, "", round_id, "qa"),
                            args.reflector_api_mode,
                            args.reflector_prefix,
                            args.reflector_suffix,
                        ),
                    )
                continue
            entry = (len(question), index, question)
            if len(longest) < args.longest_count:
                heapq.heappush(longest, entry)
            elif entry > longest[0]:
                heapq.heapreplace(longest, entry)
            if len(sample) < args.sample_size:
                sample.append(question)
            else:
                replacement = rng.randrange(rows)
                if replacement < args.sample_size:
                    sample[replacement] = question

    if not args.full_scan:
        sample.extend(entry[2] for entry in longest)
    if rows == 0:
        raise ValueError(f"No QA examples found in {args.data}")

    direct_lengths = [
        _token_count(base_tokenizer, direct_prompt(q, "qa"), "chat_completions", "", "")
        for q in sample
    ]
    reflection_lengths = []
    for question in sample:
        for round_id in (1, 2):
            reflection_lengths.append(
                _token_count(
                    reflector_tokenizer,
                    reflection_prompt(question, "", round_id, "qa"),
                    args.reflector_api_mode,
                    args.reflector_prefix,
                    args.reflector_suffix,
                )
            )

    max_direct = full_max_direct if args.full_scan else max(direct_lengths)
    max_reflection = full_max_reflection if args.full_scan else max(reflection_lengths)
    direct_required = max_direct + args.base_max_tokens_cap + args.safety_tokens
    reflection_required = max_reflection + args.reflector_max_tokens_cap + args.safety_tokens
    result = {
        "data": str(Path(args.data).resolve()),
        "rows": rows,
        "scan_mode": "full" if args.full_scan else "deterministic_sample_plus_longest",
        "prompts_tokenized": rows if args.full_scan else len(sample),
        "max_words": max_words,
        "max_model_len": args.max_model_len,
        "safety_tokens": args.safety_tokens,
        "direct": {
            "max_input_tokens": max_direct,
            "max_output_tokens_cap": args.base_max_tokens_cap,
            "required_with_safety": direct_required,
            "fits": direct_required <= args.max_model_len,
        },
        "reflection": {
            "max_input_tokens": max_reflection,
            "max_output_tokens_cap": args.reflector_max_tokens_cap,
            "required_with_safety": reflection_required,
            "fits": reflection_required <= args.max_model_len,
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight QA prompt lengths against the vLLM context window.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--base_tokenizer", required=True)
    parser.add_argument("--reflector_tokenizer", required=True)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--base_max_tokens_cap", type=int, default=2048)
    parser.add_argument("--reflector_max_tokens_cap", type=int, default=4096)
    parser.add_argument("--safety_tokens", type=int, default=256)
    parser.add_argument("--reflector_api_mode", choices=["chat_completions", "completions"], default="completions")
    parser.add_argument("--reflector_prefix", default="<｜User｜>")
    parser.add_argument("--reflector_suffix", default="<｜Assistant｜><think>\n</think>\n")
    parser.add_argument("--sample_size", type=int, default=1000)
    parser.add_argument("--longest_count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full_scan", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    result = analyze(args)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    if not result["direct"]["fits"] or not result["reflection"]["fits"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
