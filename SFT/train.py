#!/usr/bin/env python3
"""
SFT to train Qwen3-4B-Instruct to predict the *last user turn* in each row,
while preserving full conversation context.

Usage:
  python train.py \
      --data dataset/cleaned_data/final_data_os_3.jsonl \
      --model Qwen/Qwen3-4B-Instruct-2507 \
      --seq 4096 --batch 1 --accum 16 --epochs 2.0
"""

import argparse
import torch
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings("ignore")

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

import wandb
from dotenv import load_dotenv

load_dotenv(".env", override=True)


def has_user_turn(msgs: List[Dict[str, Any]]) -> bool:
    """Check if messages contain at least one user turn with content."""
    return any(m.get("role") == "user" and (m.get("content") or "").strip() for m in msgs)


def _find_last_subseq(hay: List[int], needle: List[int]) -> Optional[int]:
    """Return start index of the LAST occurrence of `needle` subsequence in `hay`."""
    if not needle or len(needle) > len(hay):
        return None
    Lh, Ln = len(hay), len(needle)
    for i in range(Lh - Ln, -1, -1):
        if hay[i:i+Ln] == needle:
            return i
    return None


def _find_next_token(hay: List[int], token_id: int, start: int) -> Optional[int]:
    """Return index of the FIRST occurrence of token_id in hay[start:], else None."""
    for i in range(start, len(hay)):
        if hay[i] == token_id:
            return i
    return None


def infer_user_prefix_and_end_token(tokenizer):
    """
    Infer the token pattern for user messages in the chat template.
    Returns (user_prefix_ids, im_end_id)
    """
    placeholder = "<<<PLACEHOLDER>>>"
    sample_messages = [{"role": "user", "content": placeholder}]

    rendered = tokenizer.apply_chat_template(
        sample_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    prefix = rendered.split(placeholder)[0]
    user_prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is None or im_end_id == tokenizer.unk_token_id:
        ids = tokenizer("<|im_end|>", add_special_tokens=False)["input_ids"]
        im_end_id = ids[0] if ids else tokenizer.eos_token_id

    return user_prefix_ids, im_end_id


class LastUserSpanCollator:
    """
    Tokenizes each example, then sets labels = -100 everywhere
    EXCEPT on the final user span: from the last user-prefix to the next <|im_end|>.
    """

    def __init__(self, tokenizer, user_prefix_ids: List[int], im_end_id: int, max_length: int):
        self.tok = tokenizer
        self.user_prefix_ids = user_prefix_ids
        self.im_end_id = im_end_id
        self.max_length = max_length
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_list, attn_list, labels_list = [], [], []

        for ex in features:
            # Apply chat template
            text = self.tok.apply_chat_template(
                ex["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )

            enc = self.tok(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length,
                return_attention_mask=True,
            )
            ids = enc["input_ids"]
            attn = enc["attention_mask"]

            # init labels as ignore
            labels = [-100] * len(ids)

            # find LAST user-prefix occurrence
            start = _find_last_subseq(ids, self.user_prefix_ids)
            if start is not None:
                content_start = start + len(self.user_prefix_ids)
                content_end_tok = _find_next_token(ids, self.im_end_id, content_start)
                content_end = content_end_tok if content_end_tok is not None else len(ids)
                # unmask labels ONLY for content tokens
                for j in range(content_start, content_end):
                    labels[j] = ids[j]

            input_ids_list.append(ids)
            attn_list.append(attn)
            labels_list.append(labels)

        # Padding
        target_len = max(len(x) for x in input_ids_list)
        target_len = min(target_len, self.max_length)

        def pad_to(seq, val, L):
            if len(seq) >= L:
                return seq[:L]
            return seq + [val] * (L - len(seq))

        input_ids_list = [pad_to(x, self.pad_id, target_len) for x in input_ids_list]
        attn_list = [pad_to(x, 0, target_len) for x in attn_list]
        labels_list = [pad_to(x, -100, target_len) for x in labels_list]

        return {
            "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
            "attention_mask": torch.tensor(attn_list, dtype=torch.long),
            "labels": torch.tensor(labels_list, dtype=torch.long),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/cleaned_data/final_data_os_1.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--out", default="SFT/trained_models/qwen3-4b-os1-epoch")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--wandb_entity", default="ashar_wandb-mbzuai")
    ap.add_argument("--wandb_project", default="sft-user-sim")
    ap.add_argument("--wandb_name", default="qwen4b_user_lastspan")
    args = ap.parse_args()

    # BF16/FP16 selection
    use_bf16 = torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float16

    # W&B
    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.wandb_name,
        config=vars(args),
    )

    print(f"\n{'='*60}")
    print(f"Training Configuration (epoch-based):")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Output: {args.out}")
    print(f"{'='*60}\n")

    # Tokenizer
    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    print(f"Tokenizer loaded: {tok.__class__.__name__}\n")

    # Dataset
    print("Loading dataset...")
    ds = load_dataset("json", data_files={"train": args.data})["train"]
    print(f"Loaded {len(ds)} examples")

    ds = ds.filter(lambda ex: has_user_turn(ex["messages"]) and ex["messages"][-1].get("role") == "user")
    print(f"After filtering: {len(ds)} examples\n")

    # Train/eval split
    split = ds.train_test_split(test_size=0.05, shuffle=True, seed=42)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}\n")

    # Model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )

    model.config.use_cache = False
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tok.pad_token_id
    print("Model loaded\n")

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Apply LoRA
    print("Applying LoRA...")
    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],  
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    # Collator
    print("\nSetting up data collator...")
    user_prefix_ids, im_end_id = infer_user_prefix_and_end_token(tok)
    print(f"User prefix token IDs: {user_prefix_ids}")
    print(f"<|im_end|> token ID: {im_end_id}\n")

    collator = LastUserSpanCollator(
        tokenizer=tok,
        user_prefix_ids=user_prefix_ids,
        im_end_id=im_end_id,
        max_length=args.seq,
    )

    # Training args (epoch-based)
    training_args = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        bf16=use_bf16,
        fp16=not use_bf16,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.06,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        report_to=["wandb"],
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")

    trainer.train()

    # Save
    print("\n" + "="*60)
    print("Saving model...")
    print("="*60 + "\n")

    trainer.save_model(args.out)
    tok.save_pretrained(args.out)

    print(f"Model saved to: {args.out}")
    print("\nTraining finished successfully!")

    run.finish()


if __name__ == "__main__":
    main()