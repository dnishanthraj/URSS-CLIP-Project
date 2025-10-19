#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
from collections import Counter

def load_json(p): return json.load(open(p,"r"))

def norm_rows(E):
    n = np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
    return E / n

def cosine(A,B):  # expects L2-normalized rows
    return (A*B).sum(-1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", default="analytics.json")
    args = ap.parse_args()

    work = Path(args.work)
    segs = load_json(work/"segments_captions.json")
    summary = load_json(work/"summary_captions.json") if (work/"summary_captions.json").exists() else load_json(work/"summary.json")
    npz = np.load(work/"shots_embeddings.npz", allow_pickle=True)
    E = norm_rows(npz["emb"])
    shot_ids = npz["ids"].tolist()
    id2i = {sid:i for i,sid in enumerate(shot_ids)}

    # durations
    video_len = float(max([float(s["t_end"]) for s in segs] + [0.0]))
    sel_len = sum(float(s["t_end"])-float(s["t_start"]) for s in summary)
    coverage = sel_len / max(video_len, 1e-6)

    # redundancy proxy: mean of max pairwise cosine within each selected segment
    red_scores = []
    for s in segs:
        ts, te = float(s["t_start"]), float(s["t_end"])
        if not any( (abs(ts-float(u["t_start"]))<1e-6 and abs(te-float(u["t_end"]))<1e-6) for u in summary ):
            continue
        idxs = [id2i[x] for x in s["shot_ids"] if x in id2i]
        if len(idxs) >= 2:
            X = E[idxs]
            sim = X @ X.T
            # upper triangle without diag
            tri = sim[np.triu_indices_from(sim, k=1)]
            red_scores.append(float(tri.mean()))
    redundancy = float(np.mean(red_scores)) if red_scores else 0.0

    # prompt mix
    prompts = [s.get("best_prompt","") for s in segs]
    top_prompts = Counter(prompts).most_common(10)

    out = {
        "video_length_sec": round(video_len,3),
        "summary_length_sec": round(sel_len,3),
        "coverage_frac": round(coverage,4),
        "num_segments_all": len(segs),
        "num_segments_selected": len(summary),
        "avg_segment_len_sec": round(np.mean([float(s["t_end"])-float(s["t_start"]) for s in summary]) if summary else 0.0,3),
        "redundancy_cosine_mean": round(redundancy,4),
        "top_prompts": top_prompts,
    }
    json.dump(out, open(work/args.out,"w"), indent=2)
    print(f"[OK] wrote {work/args.out}")
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
