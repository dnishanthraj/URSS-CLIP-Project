#!/usr/bin/env python3
import argparse, json, os
from pathlib import Path
from moviepy.editor import VideoFileClip, concatenate_videoclips
from tqdm import tqdm

def load_summary(path: Path):
    data = json.load(open(path, "r"))
    # Accept either format: [{t_start, t_end, ...}] or [{segment_id, t_start, t_end, ...}]
    return [{"t_start": float(r["t_start"]), "t_end": float(r["t_end"]), **r} for r in data]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="work dir containing summary*.json and (linked) video.mp4")
    ap.add_argument("--summary_json", help="defaults to summary_captions.json if exists, else summary.json")
    ap.add_argument("--video", help="defaults to work/video.mp4")
    ap.add_argument("--fade", type=float, default=0.0, help="optional per-cut crossfade (sec)")
    ap.add_argument("--out", default="highlight.mp4")
    args = ap.parse_args()

    work = Path(args.work).resolve()
    video_file = Path(args.video) if args.video else (work / "video.mp4")
    if not video_file.exists():
        raise SystemExit(f"[err] Missing video file: {video_file}\nRun link_video.py first.")

    # Choose best summary file automatically
    if args.summary_json:
        summary_file = Path(args.summary_json)
    else:
        s1 = work / "summary_captions.json"
        s2 = work / "summary.json"
        summary_file = s1 if s1.exists() else s2
    if not summary_file.exists():
        raise SystemExit(f"[err] summary file not found ({summary_file})")

    segments = load_summary(summary_file)
    if not segments:
        raise SystemExit("[err] empty summary  nothing to cut")

    print(f"[info] Loading video: {video_file}")
    base = VideoFileClip(str(video_file))
    clips = []
    # clamp to video duration
    VLEN = base.duration
    for s in tqdm(segments, desc="Cutting segments"):
        ts, te = max(0.0, float(s["t_start"])), min(VLEN, float(s["t_end"]))
        if te <= ts:
            continue
        sub = base.subclip(ts, te)
        if args.fade > 0:
            sub = sub.crossfadein(args.fade).crossfadeout(args.fade)
        clips.append(sub)

    if not clips:
        raise SystemExit("[err] no valid clips to concatenate")

    print(f"[info] Concatenating {len(clips)} clips")
    final = concatenate_videoclips(clips, method="compose", padding=-args.fade if args.fade>0 else 0)
    out_path = work / args.out
    # Fast but good-enough encoding settings
    final.write_videofile(
        str(out_path),
        codec="libx264", audio_codec="aac",
        preset="medium", bitrate="3000k",
        threads=os.cpu_count() or 4,
        temp_audiofile=str(work/"_tmp_audio.m4a"),
        remove_temp=True,
        verbose=False, logger=None
    )
    print(f"[OK] highlight written: {out_path}")

if __name__ == "__main__":
    main()
