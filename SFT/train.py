import argparse, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

def guess_lora_targets(model_name: str):
    name = model_name.lower()
    if any(k in name for k in ["llama", "mistral", "gemma", "phi"]):
        return ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    if "qwen" in name:
        return ["W_pack","o_proj","gate_proj","up_proj","down_proj"]
    # Safe fallbacks
    return ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","W_pack"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/cleaned_data/final_data_os_3.jsonl" , help="JSONL with per-user-turn 'messages' rows.")
    ap.add_argument("--model", required=True, help="HF model (e.g., meta-llama/Meta-Llama-3-8B-Instruct).")
    ap.add_argument("--out", default = "SFT/trained_models", help="Output dir.")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    args = ap.parse_args()

    # ---- Dataset ----
    ds = load_dataset("json", data_files={"train": args.data})["train"]
    split = ds.train_test_split(test_size=0.05, shuffle=True, seed=42)
    train_ds, eval_ds = split["train"], split["test"]

    # ---- Tokenizer ----
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Prefix that immediately precedes ANY user content in this model’s chat template
    user_header = tok.apply_chat_template(
        [{"role": "user", "content": ""}],
        add_generation_prompt=False,
        tokenize=False,
    )

    # ---- Model (QLoRA) ----
    quant_config = None
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant_config,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    # Important for checkpointing with some models
    model.config.use_cache = False
    # Save VRAM
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()

    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=guess_lora_targets(args.model),
    )

    # ---- Trainer ----
    use_bf16 = torch.cuda.is_bf16_supported()
    use_fp16 = not use_bf16

    cfg = SFTConfig(
        output_dir=args.out,
        max_seq_length=args.seq,
        packing=True,
        dataset_num_proc=4,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=20,
        save_steps=1000,
        eval_strategy="steps",
        eval_steps=1000,
        report_to=[],
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=use_bf16,
        fp16=use_fp16,
        completion_only_loss=True,
        response_template=user_header,
        group_by_length=True,  # better packing & fewer OOM spikes
        gradient_checkpointing=True,
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

    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)

if __name__ == "__main__":
    main()
