# knapsack_select.py
# Step 5: choose segments under a time budget using 0/1 knapsack.
import argparse, json
from pathlib import Path
import math

def load_json(p: Path):
    return json.load(open(p, "r", encoding="utf-8"))

def save_json(obj, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(p, "w", encoding="utf-8"), indent=2)

def main():
    ap = argparse.ArgumentParser(description="Select segments under a time budget (knapsack).")
    ap.add_argument("--segments_scored", required=True, help="segments_scored.json from Step 4")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--use", default="generic", choices=["generic","query","mix"],
                    help="which score to use: generic, query, or mix=0.7*generic+0.3*query (if query present)")
    ap.add_argument("--budget_frac", type=float, default=0.15, help="fraction of full video duration to keep (e.g., 0.15 = 15%)")
    ap.add_argument("--time_unit", type=float, default=1.0, help="discretization in seconds (1.0 is fine)")
    args = ap.parse_args()

    segs = load_json(Path(args.segments_scored))
    # compute video length
    Tstart = min(s["t_start"] for s in segs)
    Tend   = max(s["t_end"]   for s in segs)
    full_dur = Tend - Tstart

    budget = args.budget_frac * full_dur
    print(f"[info] Video length = {full_dur:.1f}s, budget = {budget:.1f}s ({args.budget_frac*100:.0f}%).")

    # value & cost per segment
    values, costs = [], []
    for s in segs:
        dur = max(0.0, float(s["t_end"] - s["t_start"]))
        if args.use == "generic":
            val = float(s.get("score_generic", 0.0))
        elif args.use == "query":
            val = float(s.get("score_query", 0.0))
        else:  # mix
            g = float(s.get("score_generic", 0.0))
            q = float(s.get("score_query", 0.0))
            val = 0.7*g + 0.3*q
        values.append(val)
        costs.append(dur)

    # discretize time to units for DP
    unit = max(0.1, float(args.time_unit))  # dont let it be too tiny
    W = int(math.floor(budget / unit))
    weights = [int(math.ceil(c / unit)) for c in costs]

    n = len(segs)
    # DP tables: well also keep a take table for backtracking
    dp = [[0.0]*(W+1) for _ in range(n+1)]
    take = [[False]*(W+1) for _ in range(n+1)]

    for i in range(1, n+1):
        w_i = weights[i-1]
        v_i = values[i-1]
        for w in range(W+1):
            # skip
            best = dp[i-1][w]
            did_take = False
            # take if fits
            if w_i <= w:
                cand = dp[i-1][w - w_i] + v_i
                if cand > best:
                    best = cand
                    did_take = True
            dp[i][w] = best
            take[i][w] = did_take

    # backtrack
    w = W
    picked_idxs = []
    for i in range(n, 0, -1):
        if take[i][w]:
            picked_idxs.append(i-1)
            w -= weights[i-1]
    picked_idxs.reverse()

    # sort selected segments by their time (to play in order)
    selected = [segs[i] for i in picked_idxs]
    selected.sort(key=lambda s: s["t_start"])

    kept = sum(s["t_end"]-s["t_start"] for s in selected)
    print(f"[OK] Selected {len(selected)} segments, total {kept:.1f}s.")

    # write summary JSON (timeline)
    out = []
    for s in selected:
        out.append({
            "t_start": float(s["t_start"]),
            "t_end": float(s["t_end"]),
            "segment_id": s["segment_id"],
            "key_thumb": s.get("key_thumb"),
            "score": float(s.get("score_query" if args.use=="query" else "score_generic", 0.0)),
            "best_prompt": s.get("best_prompt")
        })
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out, out_dir / "summary.json")

    # quick preview
    print("\nSummary timeline (first 10 rows):")
    for row in out[:10]:
        print(f" {row['t_start']:.1f}-{row['t_end']:.1f}s | {row['segment_id']} | score={row['score']:.3f} | {row.get('best_prompt','')}")

if __name__ == "__main__":
    main()
