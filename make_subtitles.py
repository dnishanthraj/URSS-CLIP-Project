#!/usr/bin/env python3
import argparse, json
from pathlib import Path

def sec_to_srt_time(t: float) -> str:
    """Convert seconds -> SRT timestamp"""
    h = int(t // 3600); t -= h*3600
    m = int(t // 60); s = t - m*60
    ms = int((s - int(s)) * 1000)
    return f"{h:02}:{m:02}:{int(s):02},{ms:03}"

def make_srt(summary_json: Path, out_path: Path, min_score: float = 0.0):
    data = json.load(open(summary_json, "r"))
    lines = []
    idx = 1
    for seg in data:
        ts, te = float(seg["t_start"]), float(seg["t_end"])
        cap = seg.get("desc") or seg.get("title") or ""
        score = float(seg.get("score", seg.get("score_generic", 0.0)))
        if not cap or score < min_score:
            continue
        start = sec_to_srt_time(ts)
        end = sec_to_srt_time(te)
        lines.append(f"{idx}\n{start} --> {end}\n{cap.strip()}\n")
        idx += 1

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Wrote subtitles to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_json", required=True, help="summary_captions.json or summary.json")
    ap.add_argument("--out", default="highlight.srt", help="output SRT file")
    ap.add_argument("--min_score", type=float, default=0.0, help="skip captions below this score")
    args = ap.parse_args()

    make_srt(Path(args.summary_json), Path(args.out), args.min_score)
