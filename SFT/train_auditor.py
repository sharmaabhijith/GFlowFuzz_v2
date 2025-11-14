#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to JSONL with per-user-turn 'messages' rows.")
    ap.add_argument("--model", required=True, help="HF model name or local path (e.g., meta-llama/Meta-Llama-3-8B-Instruct).")
    ap.add_argument("--out",   required=True, help="Output dir for the fine-tuned model.")
    ap.add_argument("--seq", type=int, default=2048, help="Max sequence length.")
    ap.add_argument("--batch", type=int, default=1, help="Per-device train batch size.")
    ap.add_argument("--accum", type=int, default=16, help="Gradient accumulation steps.")
    ap.add_argument("--epochs", type=float, default=2.0, help="Number of epochs.")
    ap.add_argument("--lr", type=float, default=2e-4, help="Learning rate for LoRA params.")
    ap.add_argument("--lora_r", type=int, default=32, help="LoRA rank.")
    ap.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha.")
    ap.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout.")
    ap.add_argument("--bnb_4bit", action="store_true", help="Use 4-bit quantization (QLoRA).")
    args = ap.parse_args()

    # -------- Dataset --------
    ds = load_dataset("json", data_files={"train": args.data})["train"]
    # (Optional) quick 95/5 split for sanity eval
    split = ds.train_test_split(test_size=0.05, shuffle=True, seed=42)
    train_ds, eval_ds = split["train"], split["test"]

    # -------- Tokenizer --------
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # This is the exact prefix your model uses right before any USER content.
    user_header = tok.apply_chat_template(
        [{"role": "user", "content": ""}],
        add_generation_prompt=False,
        tokenize=False,
    )

    # Formatting function: turn each row's messages into a single chat-formatted string
    def formatting_func(example):
        return tok.apply_chat_template(
            example["messages"],
            add_generation_prompt=False,   # we are *training on* the last user turn
            tokenize=False,
        )

    # -------- Model (QLoRA) --------
    quant_config = None
    if args.bnb_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant_config,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )

    # PEFT LoRA
    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[  # safe defaults for most decoder-only models
            "q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"
        ],
    )

    # -------- TRL SFT Trainer --------
    cfg = SFTConfig(
        output_dir=args.out,
        max_seq_length=args.seq,
        packing=True,                               # pack multiple short samples efficiently
        dataset_num_proc=4,                         # speed up formatting
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        report_to=[],
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,                                  # set fp16=True instead if your GPU prefers fp16
        completion_only_loss=True,                  # <— mask loss to "completion"
        response_template=user_header,              # <— the mask anchor (user header)
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=cfg,
        peft_config=peft_cfg,
        formatting_func=formatting_func,            # <— format messages → text on-the-fly
    )

    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)

if __name__ == "__main__":
    main()
