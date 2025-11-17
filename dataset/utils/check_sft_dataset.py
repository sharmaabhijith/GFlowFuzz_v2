#!/usr/bin/env python3
"""
Quick sanity checks for TRL SFT datasets.

- Detects the messages field (messages / conversations / prompt+response).
- Validates each example has at least one 'assistant' turn with non-empty text.
- Tokenizes with the model's chat template to compute sequence length.
- Summarizes counts and shows a few problematic rows.

Usage:
  python dataset/check_sft_dataset.py --data dataset/cleaned_data/final_data_os_3.jsonl \
                              --model Qwen/Qwen3-4B-Instruct-2507 \
                              --max-seq-length 2048 --limit 5000
"""
import argparse, json, random, sys, os
from typing import List, Dict, Any
from datasets import load_dataset
from transformers import AutoTokenizer

from dotenv import load_dotenv
load_dotenv(".env", override=True)

VALID_ROLES = {"system", "user", "assistant", "tool"}

def detect_messages_field(row: Dict[str, Any]):
    if "messages" in row and isinstance(row["messages"], list):
        return "messages"
    if "conversations" in row and isinstance(row["conversations"], list):
        return "conversations"
    # common two-column format
    if {"prompt", "response"} <= row.keys():
        # synthesize messages for checking
        msgs = [{"role": "user", "content": str(row["prompt"])},
                {"role": "assistant", "content": str(row["response"])}]
        return ("__synthetic__", msgs)
    return None

def normalize_messages(row, field):
    if isinstance(field, tuple) and field[0] == "__synthetic__":
        return field[1]
    msgs = row[field]
    # normalize keys like "role", "content"
    out = []
    for m in msgs:
        role = m.get("role") or m.get("from") or m.get("speaker")
        content = m.get("content") or m.get("value") or m.get("text")
        out.append({"role": str(role) if role is not None else None,
                    "content": "" if content is None else str(content)})
    return out

def has_assistant_turn(msgs: List[Dict[str, str]]) -> bool:
    return any((m.get("role") == "assistant") and (m.get("content", "").strip()) for m in msgs)

def roles_ok(msgs) -> bool:
    return all((m.get("role") in VALID_ROLES) for m in msgs if m.get("role") is not None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to JSONL or HF dataset (json)")
    ap.add_argument("--model", required=True, help="HF model or local path")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=2000, help="Max rows to check (-1 = all)")
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    ds = load_dataset("json", data_files=args.data, split=args.split, cache_dir=os.environ.get("HF_DATASETS_CACHE"))

    # choose a manageable subset for speed
    if args.limit > 0 and len(ds) > args.limit:
        ds = ds.select(range(args.limit))

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # basic template check
    tmpl = getattr(tok, "chat_template", None)
    has_gen_tags = isinstance(tmpl, str) and ("{% generation" in tmpl) and ("{% endgeneration %}" in tmpl)

    bad_schema, no_assistant, bad_roles, too_long, empty_text = [], [], [], [], []
    lengths = []
    show_examples = {"no_assistant": [], "bad_roles": [], "too_long": [], "empty_text": []}

    for i, row in enumerate(ds):
        field = detect_messages_field(row)
        if field is None:
            bad_schema.append(i)
            if len(show_examples.setdefault("bad_schema", [])) < 3:
                show_examples["bad_schema"].append(row)
            continue

        msgs = normalize_messages(row, field)
        # empty content check
        if any((m.get("content", "").strip() == "") for m in msgs):
            empty_text.append(i)
            if len(show_examples["empty_text"]) < 3:
                show_examples["empty_text"].append(msgs)

        if not has_assistant_turn(msgs):
            no_assistant.append(i)
            if len(show_examples["no_assistant"]) < 3:
                show_examples["no_assistant"].append(msgs)

        if not roles_ok(msgs):
            bad_roles.append(i)
            if len(show_examples["bad_roles"]) < 3:
                show_examples["bad_roles"].append(msgs)

        # token length via chat template (falls back to simple join if no template)
        try:
            ids = tok.apply_chat_template(
                msgs,
                tokenize=True,
                return_tensors=None,
                add_generation_prompt=False,
                truncation=False,
            )
            seq_len = len(ids["input_ids"]) if isinstance(ids, dict) else len(ids)
        except Exception:
            # fallback: naive join (not ideal, but gives a signal)
            text = ""
            for m in msgs:
                text += f"{m.get('role','user').upper()}: {m.get('content','')}\n"
            seq_len = len(tok(text, add_special_tokens=True)["input_ids"])
        lengths.append(seq_len)
        if seq_len > args.max_seq_length:
            too_long.append(i)
            if len(show_examples["too_long"]) < 3:
                show_examples["too_long"].append({"len": seq_len, "msgs": msgs})

    n = len(ds)
    def pct(x): return f"{(100.0*len(x)/n):.2f}% ({len(x)}/{n})"

    print("\n=== DATASET SANITY REPORT ===")
    print(f"Rows checked: {n}")
    print(f"Chat template present: {bool(tmpl)}")
    print(f"Template has generation tags: {has_gen_tags}")
    print(f"Missing/unknown schema rows: {pct(bad_schema)}")
    print(f"Rows with NO assistant turn: {pct(no_assistant)}")
    print(f"Rows with invalid roles: {pct(bad_roles)}   (valid={sorted(VALID_ROLES)})")
    print(f"Rows with empty texts: {pct(empty_text)}")
    if lengths:
        print(f"Token length: mean={sum(lengths)//len(lengths)}  "
              f"min={min(lengths)} max={max(lengths)}  >{args.max_seq_length}: {pct(too_long)}")

    # sample problematic examples
    for k in ["bad_schema", "no_assistant", "bad_roles", "empty_text", "too_long"]:
        exs = show_examples.get(k, [])
        if exs:
            print(f"\n--- Examples: {k} ---")
            for e in exs:
                print(json.dumps(e, ensure_ascii=False)[:1200], "\n")

    # exit non-zero if problems that break assistant_only_loss
    if no_assistant or not has_gen_tags:
        print("\n[FAIL] Either some rows lack assistant messages OR your template lacks generation tags.")
        sys.exit(2)
    print("\n[OK] Basic sanity checks passed.")
    sys.exit(0)

if __name__ == "__main__":
    main()
