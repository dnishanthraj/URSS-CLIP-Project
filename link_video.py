#!/usr/bin/env python3
import argparse, os
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="work directory (e.g., ../work/XkqCExn6_Us_fixed2s)")
    ap.add_argument("--dataset_video", help="absolute path to the original .mp4 (TVSum)")
    ap.add_argument("--guess_root", default="../tvsum50_ver_1_1/ydata-tvsum50-v1_1/ydata-tvsum50-video/video",
                    help="if --dataset_video not given, try to guess here")
    args = ap.parse_args()

    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    target = work / "video.mp4"
    if target.exists():
        print(f"[ok] video already present: {target}")
        return

    if args.dataset_video:
        src = Path(args.dataset_video).resolve()
        if not src.exists():
            raise SystemExit(f"[err] dataset video not found: {src}")
    else:
        # guess from work dir name: strip _fixed2s suffix
        vid_id = work.name.replace("_fixed2s", "")
        src = Path(args.guess_root).resolve() / f"{vid_id}.mp4"
        if not src.exists():
            raise SystemExit(f"[err] could not guess dataset video: {src}\n"
                             f"Tip: pass --dataset_video /absolute/path/to/<ID>.mp4")

    os.symlink(src, target)
    print(f"[ok] linked {src} -> {target}")

if __name__ == "__main__":
    main()
