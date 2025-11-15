#!/usr/bin/env python3
import argparse, os, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
import wandb
from dotenv import load_dotenv

load_dotenv(".env", override=True)

# ---------------- utilities ----------------
def has_user_turn(msgs):
    return any(m.get("role") == "user" and (m.get("content") or "").strip() for m in msgs)

def render_text_with_last_user_target(example, tok):
    """
    From an example like:
      {"messages":[ ... , {"role":"user","content":"<TARGET>"} ]}
    1) Find the *last* user turn (should be last by your spec).
    2) Blank earlier user contents so only the final user span has tokens.
    3) Render with chat template -> single 'text' string.
    """
    msgs = list(example["messages"])
    # locate last user index (robust even if it's not literally last)
    last_user_idx = -1
    for i in range(len(msgs)-1, -1, -1):
        if msgs[i].get("role") == "user":
            last_user_idx = i
            break
    if last_user_idx == -1:
        return {"text": None}  # will be filtered out later

    msgs_out = []
    for i, m in enumerate(msgs):
        if m.get("role") == "user" and i != last_user_idx:
            msgs_out.append({"role": "user", "content": ""})  # keep role, blank content
        else:
            msgs_out.append(m)

    text = tok.apply_chat_template(
        msgs_out,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

def infer_prefix_ids_for_role(tok, role: str):
    """
    Compute token IDs for the chat-template prefix that immediately precedes
    the content of a given role (e.g., 'user'). Works with Qwen chat templates.
    """
    placeholder = "<<<P>>>"
    rendered = tok.apply_chat_template(
        [{"role": role, "content": placeholder}],
        tokenize=False,
        add_generation_prompt=False,
    )
    prefix = rendered.split(placeholder)[0]
    return tok(prefix, add_special_tokens=False)["input_ids"]

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/cleaned_data/final_data_os_3.jsonl",
                    help="JSONL with rows like {'messages': [..., {'role':'user','content':'<TARGET>'}]} — last turn is the user.")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507", help="HF model id or local path")
    ap.add_argument("--out",   default="SFT/trained_models", help="Output dir")
    ap.add_argument("--seq",   type=int, default=2048, help="Sequence length (consider 4096+ if VRAM allows)")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--epochs",type=float, default=2.0)
    ap.add_argument("--lr",    type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    # W&B
    ap.add_argument("--wandb_entity",  default="ashar_wandb-mbzuai")
    ap.add_argument("--wandb_project", default="sft-user-sim")
    ap.add_argument("--wandb_name",    default="qwen4b_user_sft_run")
    ap.add_argument("--wandb_tags", nargs="*", default=None)
    args = ap.parse_args()

    # ---------------- W&B ----------------
    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.wandb_name,
        config=vars(args),
    )

    # ---------------- Tokenizer ----------------
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # ---------------- Dataset ----------------
    ds = load_dataset("json", data_files={"train": args.data})["train"]

    # keep examples with at least one user turn
    ds = ds.filter(lambda ex: has_user_turn(ex["messages"]))

    # render to one training string per example, masking earlier user spans by blanking them
    ds = ds.map(lambda ex: render_text_with_last_user_target(ex, tok),
                num_proc=4)
    # drop any rows that failed rendering
    ds = ds.filter(lambda ex: isinstance(ex["text"], str) and len(ex["text"]) > 0)

    split = ds.train_test_split(test_size=0.05, shuffle=True, seed=42)
    train_ds, eval_ds = split["train"], split["test"]

    # ---------------- Model (QLoRA) ----------------
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant_config,
        dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if getattr(model.config, "pad_token_id", None) is None and tok.pad_token_id is not None:
        model.config.pad_token_id = tok.pad_token_id

    # save VRAM
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
        target_modules=["W_pack","o_proj","gate_proj","up_proj","down_proj"],
    )

    # ---------------- USER-only loss masking ----------------
    user_prefix_ids = infer_prefix_ids_for_role(tok, "user")
    collator = DataCollatorForCompletionOnlyLM(
        response_template_ids=user_prefix_ids,
        tokenizer=tok,
        response_template_with_eos=True,   # stop masking at <|im_end|>
    )

    # ---------------- Trainer config (current TRL API) ----------------
    use_bf16 = torch.cuda.is_bf16_supported()
    use_fp16 = not use_bf16

    cfg = SFTConfig(
        output_dir=args.out,
        max_length=args.seq,
        packing=True,
        dataset_num_proc=4,

        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,

        logging_steps=20,
        save_steps=1000,
        evaluation_strategy="steps",
        eval_steps=1000,
        report_to=[],                       # keep manual W&B logging; set ["wandb"] to auto-log

        lr_scheduler_type="cosine",
        warmup_ratio=0.05,

        bf16=use_bf16,
        fp16=use_fp16,

        assistant_only_loss=False,          # we mask via the collator (USER spans)
        group_by_length=True,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        peft_config=peft_cfg,
        processing_class=tok,               # modern replacement for tokenizer=
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",          # the field we created above
        data_collator=collator,             # masks only USER spans
    )

    # ---------------- Train ----------------
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.log({"params/total": total, "params/trainable": trainable})

    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    run.finish()

if __name__ == "__main__":
    main()
