"""
Convert episodic chat logs into per-user-turn rows for SFT (messages format),
and oversample episodes where reward==1.

Each OUTPUT line (JSONL) has:
{
  "episode_id": "...",
  "turn_index": <int>,                 # which user turn in the episode
  "episode_reward": 0|1,               # success flag (see _episode_success)
  "messages": [                        # last message is the TARGET user turn
    {"role":"system","content":"<r:0|1>"},
    ... context so far ...,
    {"role":"user","content":"..."}    # <-- the user turn to learn
  ]
}
"""

import argparse, json, re
from typing import Any, Dict, List



def write_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def _load_episodes(path: str) -> List[Dict[str, Any]]:
    """Load JSONL or JSON array."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        return []
    if raw[0] == '[':
        return json.loads(raw)
    return [json.loads(line) for line in raw.splitlines() if line.strip()]

def _episode_success(ep: dict) -> int:
    r = ep.get("reward", {}) or {}
    try:
        nc = int(r.get("non_compliance", 0))
        cf = int(r.get("confabulation", 0))
    except Exception:
        nc, cf = 0, 0
    return 1 if (nc == 1 or cf == 1) else 0


def explode_episode(
    ep: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Create one training row per USER turn (last message is the target user turn)."""
    episode_id = ep.get("episode_id", "")
    reward_flag = _episode_success(ep)
    msgs = ep.get("messages", [])
    obj = ep.get("objective", "")
    rows = []

    # optional reward-conditioning as a tiny system message
    sys_prefix = [{"role": "system", "content": obj}, {"role": "assistant", "content": "Hello, how can I help you?"}]

    # Walk through messages; whenever we hit a user turn, emit a row with context up to that turn (inclusive)
    turn_index = 0
    for i, m in enumerate(msgs):
        if m.get("role") != "user":
            continue

        context = msgs[:i]  # everything BEFORE this user message
        target_user = {"role": "user", "content": m.get("content", "")}

        sample_messages = sys_prefix + context + [target_user]

        rows.append({
            "episode_id": episode_id,
            "turn_index": turn_index,
            "episode_reward": reward_flag,
            "messages": sample_messages
        })
        turn_index += 1

    return rows

def oversample_positive_episodes(rows: List[Dict[str, Any]], factor: float) -> List[Dict[str, Any]]:
    """
    Oversample all rows that belong to episodes with episode_reward==1 by 'factor'.
    factor=1.0 -> no change; 3.0 -> add ~2 extra copies (integer part).
    """
    if factor <= 1.0:
        return rows
    pos = [r for r in rows if r.get("episode_reward", 0) == 1]
    base = rows[:]
    dup = int(factor) - 1
    frac = factor - int(factor)
    for _ in range(max(0, dup)):
        base.extend(pos)
    if frac > 1e-6 and pos:
        k = max(1, int(len(pos) * frac))
        base.extend(pos[:k])
    return base

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="raw_data/final_data.jsonl" , help="Input path (JSONL or JSON array) of episodes.")
    ap.add_argument("--out", default="cleaned_data/final_data", help="Output JSONL path of per-user-turn rows.")
    ap.add_argument("--oversample-positive", type=float, default=1, help="Oversample factor for episodes with reward=1 (e.g., 3.0).")
    ap.add_argument("--no-reward-token", action="store_true",
                    help="Do NOT add <r:0|1> system message.")
    args = ap.parse_args()

    episodes = _load_episodes(args.inp)
    all_rows: List[Dict[str, Any]] = []
    for ep in episodes:
        all_rows.extend(explode_episode(ep))

    pos_rows = [r for r in all_rows if r.get("episode_reward", 0) == 1]
    neg_rows = [r for r in all_rows if r.get("episode_reward", 0) == 0]

    over_rows = oversample_positive_episodes(all_rows, args.oversample_positive)

    write_jsonl(over_rows, f"{args.out}_os_{args.oversample_positive}.jsonl")
    write_jsonl(pos_rows, f"{args.out}_reward_1.jsonl")
    write_jsonl(neg_rows, f"{args.out}_reward_0.jsonl")

    print(f"Wrote {len(all_rows)} samples to {args.out}")

if __name__ == "__main__":
    main()