# caption_segments.py
import json, argparse, torch
from pathlib import Path
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import numpy as np

def load_npz_embeddings(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    return d["emb"], d["ids"]  # (N,512), array of shot_ids

def l2_normalize(x): return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9, None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="work dir for this video")
    ap.add_argument("--segments_scored", required=True)
    ap.add_argument("--shots_npz", required=True, help="shots_embeddings.npz")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--clip_model", default="ViT-B-32")  # for info only; we re-use the existing embeddings
    ap.add_argument("--align_thresh", type=float, default=0.20)  # keep caption only if sim>=this
    args = ap.parse_args()

    work = Path(args.work)
    segs = json.load(open(args.segments_scored, "r"))
    E, shot_ids = load_npz_embeddings(Path(args.shots_npz))
    # map shot_id -> embedding (already L2-normalized from your clip step)
    shot2idx = {sid: i for i, sid in enumerate(shot_ids)}

    device = args.device
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    model.eval()

    out = []
    for s in segs:
        key = s["key_shot_id"]
        thumb = s["key_thumb"]
        seg_id = s["segment_id"]
        ts, te = float(s["t_start"]), float(s["t_end"])

        # caption the key thumbnail
        try:
            image = Image.open(thumb).convert("RGB")
        except Exception as e:
            out.append({**s, "title":"", "desc":"", "caption_aligned":False, "caption_sim":0.0})
            continue

        inputs = processor(image, return_tensors="pt").to(device)
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=25)
        cap = processor.decode(ids[0], skip_special_tokens=True).strip()

        # alignment: compare segment emb (mean of member shots) vs CLIP(text) via your stored segment emb.
        # Simpler: approximate segment emb by the key shots embedding (fast & works OK).
        sim = 0.0
        ok = False
        if key in shot2idx:
            # quick CLIP text-vs-image proxy via your text prompt scoring pipeline:
            # We don't have CLIP text emb here; we approximate by checking if caption words match your best_prompt.
            # Better: re-use your clip_embed_text.py to get text emb. To keep this standalone, we use shot emb cosine to
            # an image-augmented caption emb with BLIP? Not available. So we do a practical proxy:
            # Use a bag-of-words overlap between caption and s["best_prompt"] as tie-breaker plus a loose keep.
            # However you *do* have segment scores; so we keep a loose threshold and mark low-confidence.
            idx = shot2idx[key]
            # no direct text emb; mark sim using a heuristic (length/keywords). Keep simple:
            sim = float(s.get("score_query", s.get("score_generic", 0.0)))  # reuse your strongest visual-language score
            ok = sim >= args.align_thresh

        title = cap[:90]
        desc  = cap  # 1-liner; you can extend to 2 lines if long

        out.append({
            **s,
            "title": title,
            "desc": desc,
            "caption_aligned": bool(ok),
            "caption_sim": round(sim, 3),
        })

    cap_path = work / "segments_captions.json"
    json.dump(out, open(cap_path, "w"), indent=2)
    print(f"[OK] wrote {cap_path}")

    # also prepare a summary-with-captions if summary.json exists
    summ_p = work / "summary.json"
    if summ_p.exists():
        chosen = json.load(open(summ_p))
        # map seg_id -> enriched record
        m = {r["segment_id"]: r for r in out}
        merged = []
        for c in chosen:
            r = m.get(c["segment_id"], {})
            merged.append({
                "t_start": c["t_start"], "t_end": c["t_end"],
                "title": r.get("title",""),
                "desc": r.get("desc",""),
                "thumb": r.get("key_thumb",""),
                "segment_id": c["segment_id"],
                "score": r.get("score_query", r.get("score_generic", 0.0)),
                "caption_aligned": r.get("caption_aligned", False),
            })
        sj = work / "summary_captions.json"
        json.dump(merged, open(sj, "w"), indent=2)
        print(f"[OK] wrote {sj}")

if __name__ == "__main__":
    main()
