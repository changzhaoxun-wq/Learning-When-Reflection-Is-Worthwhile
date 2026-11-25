import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader, Dataset

try:
    from peft import LoraConfig, get_peft_model
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency peft. Run: ./scripts/deprecated/install_sft_deps.sh"
    ) from exc

from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


class ChatSFTDataset(Dataset):
    def __init__(
        self,
        rows: List[Dict[str, Any]],
        tokenizer: Any,
        max_length: int,
        inject_no_think: bool,
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inject_no_think = inject_no_think

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.rows[index]
        messages = _messages(row, inject_no_think=self.inject_no_think)
        if len(messages) < 2 or messages[-1].get("role") != "assistant":
            raise ValueError(f"Row {index} does not end with an assistant message.")

        prompt_messages = messages[:-1]
        prompt_text = _apply_chat_template(self.tokenizer, prompt_messages, add_generation_prompt=True)
        full_text = _apply_chat_template(self.tokenizer, messages, add_generation_prompt=False)

        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]

        labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[len(prompt_ids):]
        labels = labels[: len(full_ids)]
        if all(label == -100 for label in labels):
            labels[-1] = full_ids[-1]

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_batch(batch: List[Dict[str, torch.Tensor]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].numel() for item in batch)
    input_ids, labels, attention_mask = [], [], []
    for item in batch:
        length = item["input_ids"].numel()
        pad = max_len - length
        input_ids.append(torch.cat([item["input_ids"], torch.full((pad,), pad_token_id, dtype=torch.long)]))
        labels.append(torch.cat([item["labels"], torch.full((pad,), -100, dtype=torch.long)]))
        attention_mask.append(torch.cat([torch.ones(length, dtype=torch.long), torch.zeros(pad, dtype=torch.long)]))
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask),
    }


def train(config: Dict[str, Any]) -> None:
    seed = int(config.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    rows = _read_jsonl(config["train_data_path"])
    limit = config.get("limit")
    if limit is not None:
        rows = rows[: int(limit)]
    random.shuffle(rows)
    print(f"Loaded {len(rows)} training rows from {config['train_data_path']}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(config["base_model_path"], trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = ChatSFTDataset(
        rows=rows,
        tokenizer=tokenizer,
        max_length=int(config.get("max_length", 2048)),
        inject_no_think=bool(config.get("inject_no_think", True)),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(config.get("per_device_train_batch_size", 1)),
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, tokenizer.pad_token_id),
    )

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        config["base_model_path"],
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if bool(config.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lora_cfg = config.get("lora", {})
    peft_config = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("lora_alpha", 32)),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
        target_modules=list(lora_cfg.get("target_modules", ["q_proj", "v_proj"])),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    grad_acc = int(config.get("gradient_accumulation_steps", 16))
    epochs = int(config.get("num_train_epochs", 2))
    total_update_steps = math.ceil(len(dataloader) / grad_acc) * epochs
    warmup_steps = int(total_update_steps * float(config.get("warmup_ratio", 0.03)))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 2e-4)),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    save_steps = int(config.get("save_steps", 250))
    logging_steps = int(config.get("logging_steps", 10))
    max_grad_norm = float(config.get("max_grad_norm", 1.0))
    global_step = 0
    update_step = 0
    running_loss = 0.0
    started = time.time()

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(epochs):
        for batch in dataloader:
            global_step += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / grad_acc
            loss.backward()
            running_loss += float(loss.detach().cpu()) * grad_acc

            if global_step % grad_acc == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update_step += 1

                if update_step % logging_steps == 0:
                    avg_loss = running_loss / logging_steps
                    running_loss = 0.0
                    elapsed = time.time() - started
                    lr = scheduler.get_last_lr()[0]
                    print(
                        f"[train] epoch={epoch + 1}/{epochs} update={update_step}/{total_update_steps} "
                        f"loss={avg_loss:.4f} lr={lr:.2e} elapsed={_fmt(elapsed)}",
                        flush=True,
                    )

                if save_steps > 0 and update_step % save_steps == 0:
                    checkpoint = output_dir / f"checkpoint-{update_step}"
                    model.save_pretrained(checkpoint)
                    tokenizer.save_pretrained(checkpoint)
                    print(f"Saved checkpoint to {checkpoint}", flush=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved final LoRA adapter to {output_dir}", flush=True)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _messages(row: Dict[str, Any], inject_no_think: bool) -> List[Dict[str, str]]:
    messages = [{"role": m["role"], "content": str(m["content"])} for m in row.get("messages", [])]
    if not messages and row.get("question") and row.get("answer"):
        messages = [
            {"role": "user", "content": f"Solve the math problem step by step and provide the final answer.\n\nProblem:\n{row['question']}"},
            {"role": "assistant", "content": str(row["answer"])},
        ]
    if inject_no_think and messages and messages[0].get("role") == "user":
        content = messages[0]["content"]
        if not content.lstrip().startswith("/no_think"):
            messages[0]["content"] = "/no_think\n" + content
    return messages


def _apply_chat_template(tokenizer: Any, messages: List[Dict[str, str]], add_generation_prompt: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def _fmt(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.limit is not None:
        config["limit"] = args.limit
    if args.output_dir:
        config["output_dir"] = args.output_dir
    train(config)


if __name__ == "__main__":
    main()
