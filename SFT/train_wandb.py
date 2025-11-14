import wandb
import argparse, os, torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainerCallback
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


def guess_lora_targets(model_name: str):
    name = model_name.lower()
    if any(k in name for k in ["llama", "mistral", "gemma", "phi"]):
        return ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    if "qwen" in name:
        return ["W_pack","o_proj","gate_proj","up_proj","down_proj"]
    return ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","W_pack"]


def main():
    ap = argparse.ArgumentParser()
    # data/model/io
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    # train knobs
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    # lora/quant
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--bnb_4bit", action="store_true")
    ap.add_argument("--attention_only", action="store_true")
    ap.add_argument("--flash_attn", action="store_true")
    # wandb
    ap.add_argument("--wandb", action="store_true", help="Enable wandb logging.")
    ap.add_argument("--wandb_project", default="sft-user-sim")
    ap.add_argument("--wandb_name", default=None)
    ap.add_argument("--wandb_tags", nargs="*", default=None)
    ap.add_argument("--num_eval_samples", type=int, default=4)
    args = ap.parse_args()

    # --- Dataset ---
    ds = load_dataset("json", data_files={"train": args.data})["train"]
    split = ds.train_test_split(test_size=0.05, shuffle=True, seed=42)
    train_ds, eval_ds = split["train"], split["test"]

    # --- Tokenizer ---
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Mask anchor = exact prefix before any USER content
    user_header = tok.apply_chat_template(
        [{"role":"user","content":""}],
        add_generation_prompt=False,
        tokenize=False,
    )
    # --- Model & quant ---
    quant_config = None
    if args.bnb_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )

    attn_impl = "flash_attention_2" if args.flash_attn else None
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant_config,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()

    # --- LoRA ---
    targets = guess_lora_targets(args.model)
    if args.attention_only:
        targets = [t for t in targets if t in {"q_proj","k_proj","v_proj","o_proj","W_pack"}]

    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )

    # --- wandb env ---
    report_to = []
    run_name = args.wandb_name
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    if args.wandb_name:
        os.environ["WANDB_NAME"] = args.wandb_name
    if args.wandb_tags:
        os.environ["WANDB_TAGS"] = ",".join(args.wandb_tags)
    report_to = ["wandb"]
    os.environ["WANDB_MODE"] = "offline"

    # --- Trainer config ---
    use_bf16 = torch.cuda.is_bf16_supported()
    cfg = SFTConfig(
        output_dir=args.out,
        run_name=run_name,
        max_seq_length=args.seq,
        packing=True,
        group_by_length=True,
        dataset_num_proc=4,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=20,
        save_steps=1000,
        eval_strategy="steps",
        eval_steps=1000,
        report_to=report_to,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=use_bf16,
        fp16=not use_bf16,
        completion_only_loss=True,
        response_template=user_header,
        max_grad_norm=1.0,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=cfg,
        peft_config=peft_cfg,
        formatting_func=tok.apply_chat_template(
            example["messages"],
            add_generation_prompt=False,
            tokenize=False,
        ),
    )


    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.config.update({
        "total_params": int(total),
        "trainable_params": int(trainable),
        "trainable_ratio": float(trainable) / float(total)
    }, allow_val_change=True)


    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)

if __name__ == "__main__":
    main()
