# shot_sampler.py
import os, json, math
from pathlib import Path
import cv2
from typing import List, Dict, Tuple, Literal

# Optional (only needed for pyscenedetect mode)
try:
    from scenedetect import VideoManager, SceneManager
    from scenedetect.detectors import ContentDetector
    HAVE_SCENEDETECT = True
except Exception:
    HAVE_SCENEDETECT = False


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def get_video_meta(video_path: str) -> Tuple[float, int]:
    """Return (fps, frame_count)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return fps, frame_count


def timestamp_to_frame_idx(ts: float, fps: float) -> int:
    # Robust rounding  avoids drifting with float fps
    return max(0, int(round(ts * fps)))


def extract_frame_at_time(
    cap: cv2.VideoCapture, t_sec: float, out_path: Path, resize_to: Tuple[int,int] | None = None
) -> bool:
    """Seek by timestamp and save a JPG."""
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
    ok, frame = cap.read()
    if not ok or frame is None:
        return False
    if resize_to:
        frame = cv2.resize(frame, resize_to, interpolation=cv2.INTER_AREA)
    ok = cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return bool(ok)


def fixed_2s_scenes(duration_sec: float, window: float = 2.0) -> List[Tuple[float,float]]:
    """[(t_start, t_end)] in 2-second bins (last bin clamped to duration)."""
    scenes = []
    t = 0.0
    while t < duration_sec:
        t_start = t
        t_end = min(t + window, duration_sec)
        if t_end > t_start:  # guard
            scenes.append((t_start, t_end))
        t += window
    return scenes


def pyscenedetect_scenes(video_path: str, threshold: float = 27.0, min_scene_len_sec: float = 0.6) -> List[Tuple[float,float]]:
    """Return scenes using PySceneDetect content detector."""
    if not HAVE_SCENEDETECT:
        raise RuntimeError("PySceneDetect not installed. Install `pyscenedetect`.")
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=int(min_scene_len_sec * 1000)))
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    # Convert timecodes to seconds
    scenes = []
    for start, end in scene_list:
        t_start = start.get_seconds()
        t_end = end.get_seconds()
        if t_end > t_start:
            scenes.append((t_start, t_end))
    video_manager.release()
    return scenes


def sample_times_for_shot(t_start: float, t_end: float, max_samples: int = 3) -> List[float]:
    """Center ± small offsets if long enough."""
    mid = 0.5 * (t_start + t_end)
    dur = max(1e-6, t_end - t_start)

    if max_samples <= 1 or dur < 2.5:
        return [mid]

    # 3 samples: ~20% and 80% within the shot plus center
    t1 = t_start + 0.2 * dur
    t2 = mid
    t3 = t_start + 0.8 * dur
    return [t1, t2, t3]


def build_manifest(
    video_path: str,
    out_dir: str,
    mode: Literal["fixed2s","pyscenedetect"] = "fixed2s",
    max_samples_per_shot: int = 3,
    resize_to: Tuple[int,int] | None = (640, 360),
    threshold: float = 27.0,
) -> List[Dict]:
    """
    Returns manifest list and writes frames + manifest.json to out_dir.
    """
    video_path = str(video_path)
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    fps, frame_count = get_video_meta(video_path)
    duration_sec = frame_count / max(fps, 1e-6)

    if mode == "fixed2s":
        scenes = fixed_2s_scenes(duration_sec, window=2.0)
    elif mode == "pyscenedetect":
        scenes = pyscenedetect_scenes(video_path, threshold=threshold)
    else:
        raise ValueError("mode must be 'fixed2s' or 'pyscenedetect'.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    manifest = []
    for i, (t_start, t_end) in enumerate(scenes):
        shot_id = f"shot_{i:05d}"
        shot_dir = out_dir / shot_id
        ensure_dir(shot_dir)

        sample_times = sample_times_for_shot(t_start, t_end, max_samples=max_samples_per_shot)

        frame_paths = []
        for j, t in enumerate(sample_times):
            frame_path = shot_dir / f"frame_{j:02d}.jpg"
            ok = extract_frame_at_time(cap, t, frame_path, resize_to=resize_to)
            if ok:
                frame_paths.append(str(frame_path))
        if not frame_paths:
            # Fallback: try mid-frame by frame index seeking if timestamp failed
            mid_t = 0.5 * (t_start + t_end)
            fallback_path = shot_dir / "frame_00.jpg"
            cap.set(cv2.CAP_PROP_POS_FRAMES, timestamp_to_frame_idx(mid_t, fps))
            ok, frame = cap.read()
            if ok and frame is not None:
                if resize_to:
                    frame = cv2.resize(frame, resize_to, interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(fallback_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                frame_paths.append(str(fallback_path))

        manifest.append({
            "shot_id": shot_id,
            "t_start": round(t_start, 3),
            "t_end": round(t_end, 3),
            "frame_paths": frame_paths,
        })

    cap.release()

    # Write manifest
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[OK] Wrote {manifest_path} with {len(manifest)} shots.")
    return manifest
