import argparse
from pathlib import Path

import torch

try:
    from peft import PeftModel
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency peft. Run: ./scripts/deprecated/install_sft_deps.sh") from exc

from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    print(f"Saved merged model to {output}")


if __name__ == "__main__":
    main()
