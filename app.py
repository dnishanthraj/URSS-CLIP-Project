import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import streamlit as st
import pandas as pd

# ---------- CONFIG ----------
st.set_page_config(page_title="Zero-shot Video Summariser", layout="wide")
DEFAULT_WORK_DIR = "../work/XkqCExn6_Us_fixed2s"
# ---------------------------

def load_json(path: Path) -> List[Dict[str, Any]]:
    return json.load(open(path, "r"))

def sec_to_hms(t: float) -> str:
    t = float(t)
    h = int(t // 3600); t -= h*3600
    m = int(t // 60); s = t - m*60
    if h>0: return f"{h:d}:{m:02d}:{s:04.1f}"
    return f"{m:d}:{s:04.1f}"

def try_load_summary(work: Path):
    p = work / "summary_captions.json"
    if p.exists(): return load_json(p)
    return []

def try_load_segments(work: Path):
    p = work / "segments_captions.json"
    if p.exists(): return load_json(p)
    return []

def load_analytics(work: Path):
    segs = try_load_segments(work)
    summary = try_load_summary(work)
    video_len = max([float(s["t_end"]) for s in segs] + [0.0])
    sel_len = sum(float(s["t_end"]) - float(s["t_start"]) for s in summary)
    return {
        "video_length_sec": round(video_len, 3),
        "summary_length_sec": round(sel_len, 3),
        "coverage_frac": round(sel_len / max(video_len,1e-6), 4) if video_len else 0.0,
        "num_segments_all": len(segs),
        "num_segments_selected": len(summary),
        "avg_segment_len_sec": round(np.mean([float(s["t_end"])-float(s["t_start"]) for s in summary]) if summary else 0.0,3),
    }

def segment_duration(s): return float(s["t_end"]) - float(s["t_start"])

def try_load_text_summary(work: Path):
    p = work / "summary_text.json"
    if p.exists():
        return json.load(open(p))["paragraph"]
    return ""


# ---------------- UI ----------------
st.title("Zero-shot Video Summariser")
st.caption("Automatic summarisation with CLIP + clustering + language-guided scoring + BLIP captions.")

# Sidebar
st.sidebar.header("Inputs")
work_dir = st.sidebar.text_input("Work directory", value=DEFAULT_WORK_DIR)
work = Path(work_dir).resolve()

video_file = work / "video.mp4"
highlight_file = work / "highlight.mp4"

segments = try_load_segments(work)
summary = try_load_summary(work)
analytics = load_analytics(work)

if not segments:
    st.error("No segments found. Expected `segments_captions.json` in the work dir.")
    st.stop()

# Tabs
tab_overview, tab_segments, tab_gallery, tab_metrics = st.tabs(
    ["Overview", "Segments", "Gallery", "Analytics"]
)

# --- Overview ---
with tab_overview:
    st.subheader("Video Comparison")
    c1, c2 = st.columns([1,1])
    with c1:
        st.markdown("**Original Video**")
        if video_file.exists(): st.video(str(video_file))
        else: st.info("Place `video.mp4` in the work dir to preview here.")
    with c2:
        st.markdown("**Highlight Reel**")
        if highlight_file.exists():
            st.video(str(highlight_file))
            st.download_button("Download highlight.mp4", data=open(highlight_file,"rb"), file_name="highlight.mp4")

            srt_file = work / "highlight.srt"
            if srt_file.exists():
                st.download_button("Download subtitles (.srt)", data=open(srt_file,"rb"), file_name="highlight.srt")
            else:
                st.info("No subtitles yet. Run `make_subtitles.py` after highlight generation.")

    st.divider()
    st.subheader("Automatic Text Summary")
    para = try_load_text_summary(work)
    if para:
        st.write(para)
    else:
        st.info("Run `python make_text_summary.py --summary_json summary_captions.json --out summary_text.json` first.")


    st.divider()
    st.subheader("Project Stats")
    m = analytics
    cA, cB, cC, cD, cE = st.columns(5)
    cA.metric("Video length", f"{m['video_length_sec']:.1f}s")
    cB.metric("Summary length", f"{m['summary_length_sec']:.1f}s")
    cC.metric("Coverage", f"{100*m['coverage_frac']:.1f}%")
    cD.metric("Segments (selected/all)", f"{m['num_segments_selected']}/{m['num_segments_all']}")
    cE.metric("Avg seg len", f"{m['avg_segment_len_sec']:.1f}s")

# --- Segments ---
with tab_segments:
    st.subheader("Explore Segments")
    q = st.text_input("Search captions/prompts")
    filtered = [s for s in segments if (not q or q.lower() in (s.get("title","")+s.get("desc","")+s.get("best_prompt","")).lower())]
    st.caption(f"Showing {len(filtered)} / {len(segments)} segments")

    for seg in filtered:
        with st.container():
            cols = st.columns([0.9,2.5,1])
            with cols[0]:
                thumb = seg.get("key_thumb") or seg.get("thumb")
                if thumb and Path(thumb).exists():
                    st.image(thumb, use_column_width=True)
            with cols[1]:
                # st.markdown(f"**{seg.get('title','')}**  \n{seg.get('desc','')}")
                # st.caption(f"{sec_to_hms(seg['t_start'])}  {sec_to_hms(seg['t_end'])}")
                title = seg.get("title","").strip() or "(no title)"
                st.markdown(f"**{title}**")
                st.caption(f"{sec_to_hms(seg['t_start'])} to {sec_to_hms(seg['t_end'])}")
            with cols[2]:
                score = seg.get("score_query", seg.get("score_generic",0.0))
                st.metric("Score", f"{float(score):.3f}")

# --- Gallery ---
with tab_gallery:
    st.subheader("Shot Gallery")
    root = work
    shot_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("shot_")])
    n_show = st.slider("Max shots to show", 6, 60, 18, 6)
    cols = st.columns(6)
    for i, sd in enumerate(shot_dirs[:n_show]):
        f = next(sd.glob("*.jpg"), None)
        if not f: continue
        with cols[i % 6]:
            st.image(str(f), use_column_width=True)
            st.caption(sd.name)

# --- Analytics ---
with tab_metrics:
    st.subheader("Analytics")
    st.json(analytics)
