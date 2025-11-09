#!/usr/bin/env python3
"""
Minimal supervised fine-tuning entry point for the auditor agent.

The script expects a JSONL dataset under ./dataset where each line contains a
`messages` list with dicts shaped like {"role": "...", "content": "..."}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


ROLE_MAP = {
    "auditor": "system",
    "system": "system",
    "assistant": "assistant",
    "chat": "assistant",
    "verifier": "assistant",
    "user": "user",
    "customer": "user",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen 1B on booking conversations.")
    parser.add_argument(
        "--dataset_path",
        type=Path,
        default=Path("dataset/conversations.jsonl"),
        help="Path to the JSONL dataset with booking transcripts.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("SFT/checkpoints/auditor"),
        help="Directory where checkpoints and tokenizer artifacts are saved.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2-1.5B-Instruct",
        help="Base model to fine tune (any chat-tuned Qwen 1B checkpoint).",
    )
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--max_checkpoints", type=int, default=2)
    parser.add_argument("--val_split", type=float, default=0.05, help="Fraction reserved for validation.")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", help="Enable bf16 mixed precision when supported.")
    parser.add_argument("--tf32", action="store_true", help="Enable TF32 compute on Ampere+ GPUs.")
    return parser.parse_args()


def normalize_role(raw_role: str) -> str:
    return ROLE_MAP.get(raw_role.lower(), "user") if raw_role else "user"


def format_messages(messages: Iterable[Dict[str, Any]], tokenizer) -> str:
    chat_turns: List[Dict[str, str]] = []
    for msg in messages or []:
        content = msg.get("content")
        if not content:
            continue
        chat_turns.append(
            {
                "role": normalize_role(str(msg.get("role", "user"))),
                "content": str(content),
            }
        )

    if not chat_turns:
        return ""

    try:
        return tokenizer.apply_chat_template(
            chat_turns,
            tokenize=False,
            add_generation_prompt=False,
        )
    except ValueError:
        # If a tokenizer lacks a chat template we fall back to a simple format.
        segments = [f"{turn['role'].upper()}: {turn['content']}" for turn in chat_turns]
        return "\n\n".join(segments)


def prepare_dataset(
    dataset_path: Path,
    tokenizer,
    val_split: float,
    seed: int,
) -> Tuple[Dataset, Dataset | None]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    raw_ds = load_dataset("json", data_files={"train": str(dataset_path)})["train"]
    if val_split and 0.0 < val_split < 1.0:
        split_ds = raw_ds.train_test_split(test_size=val_split, seed=seed)
        train_ds, eval_ds = split_ds["train"], split_ds["test"]
    else:
        train_ds, eval_ds = raw_ds, None

    def add_text(example: Dict[str, Any]) -> Dict[str, str]:
        text = format_messages(example.get("messages", []), tokenizer)
        if not text:
            text = json.dumps(example, ensure_ascii=False)
        return {"text": text}

    train_ds = train_ds.map(add_text, desc="Formatting train conversations")
    train_ds = train_ds.filter(lambda ex: isinstance(ex["text"], str) and ex["text"].strip())

    eval_formatted = None
    if eval_ds is not None:
        eval_formatted = eval_ds.map(add_text, desc="Formatting eval conversations")
        eval_formatted = eval_formatted.filter(lambda ex: isinstance(ex["text"], str) and ex["text"].strip())

    return train_ds, eval_formatted


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset, eval_dataset = prepare_dataset(
        args.dataset_path,
        tokenizer,
        args.val_split,
        args.seed,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )

    eval_strategy = "steps" if eval_dataset is not None and args.eval_steps > 0 else "no"

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.max_checkpoints,
        evaluation_strategy=eval_strategy,
        eval_steps=args.eval_steps if eval_strategy == "steps" else None,
        bf16=args.bf16,
        tf32=args.tf32,
        report_to="none",
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
    )

    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))


if __name__ == "__main__":
    main()
