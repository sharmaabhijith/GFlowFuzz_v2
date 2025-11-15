import json, itertools

jsonl_path = "dataset/data_user-qwen-qwen3-235b-a22b-instruct-2507_chat-qwen-qwen3-235b-a22b-instruct-2507_temp-1.jsonl"
pretty_path = "dataset/conversations_pretty.json"
limit = 10

items = []
with open(jsonl_path, "r", encoding="utf-8") as f:
    lines = f if limit is None else itertools.islice(f, limit)
    for line in lines:
        if line.strip():
            items.append(json.loads(line))
with open(pretty_path, "w", encoding="utf-8") as out:
    json.dump(items, out, indent=2, ensure_ascii=False, sort_keys=True)

    