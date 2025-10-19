#!/usr/bin/env python
import argparse, json
from pathlib import Path

from transformers import pipeline

def make_paragraph_ml(summary_json: Path, out_json: Path, min_score: float = 0.22):
    # Load BLIP caption outputs
    data = json.load(open(summary_json, "r"))

    # Collect captions with decent confidence
    caps = []
    for seg in data:
        cap = seg.get("desc", "").strip()
        score = float(seg.get("score", seg.get("score_generic", 0.0)))
        if cap and score >= min_score:
            caps.append(cap)

    if not caps:
        print("[WARN] No captions passed the threshold.")
        out = {"paragraph": "", "num_sentences": 0}
        json.dump(out, open(out_json, "w"), indent=2)
        return

    # Concatenate into pseudo-document
    raw_text = " ".join(caps)

    # Use pretrained summarisation model
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)  # set device=0 for GPU
    summary = summarizer(
        raw_text,
        max_length=120,
        min_length=30,
        do_sample=False
    )[0]["summary_text"]

    out = {"paragraph": summary.strip(), "num_sentences": summary.count(".")}
    json.dump(out, open(out_json, "w"), indent=2)
    print(f"[OK] Wrote {out_json}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_json", required=True, help="Path to summary_captions.json")
    ap.add_argument("--out", default="summary_text.json")
    ap.add_argument("--min_score", type=float, default=0.22)
    args = ap.parse_args()
    make_paragraph_ml(Path(args.summary_json), Path(args.out), args.min_score)
