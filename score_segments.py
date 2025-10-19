# score_segments.py
# Step 4: language-guided scoring using CLIP text embeddings (TVSum or News preset, + optional query, + optional OCR).
import argparse, json, re
from pathlib import Path
import numpy as np
import torch

# Prefer open_clip (same as your image embeddings). Fallback to OpenAI clip if needed.
try:
    import open_clip
    HAVE_OPENCLIP = True
except Exception:
    HAVE_OPENCLIP = False
    import clip as clip_orig

# ---- Prompt banks ----

TVSUM_PROMPTS = [
    # VT: Changing Vehicle Tire
    "changing a car tire",
    "a person using a car jack",
    "loosen lug nuts on a wheel",
    "replacing a flat tire at the roadside",

    # VU: Getting Vehicle Unstuck
    "a car stuck in mud",
    "pushing a car out of snow",
    "a tow strap pulling a stuck vehicle",
    "a winch pulling a vehicle",

    # GA: Grooming an Animal
    "brushing a dog",
    "washing a dog in a bath",
    "pet grooming with clippers",
    "drying a dog with a towel",

    # MS: Making Sandwich
    "hands making a sandwich",
    "spreading butter on bread",
    "slicing bread on a cutting board",
    "stacking cheese and ham on bread",

    # PK: Parkour
    "people doing parkour",
    "free running jump over obstacles",
    "vaulting a fence",
    "leaping between rooftops",

    # PR: Parade
    "parade on a city street",
    "marching band in a parade",
    "colorful parade floats",
    "crowd watching a parade",

    # FM: Flash Mob
    "flash mob dance in a public place",
    "group dancing in a mall",
    "surprised crowd watching dancers",
    "coordinated crowd performance",

    # BK: Bee Keeping
    "beekeeper wearing a protective suit",
    "beehives with frames of honeycomb",
    "smoking the beehive",
    "honeycomb frame covered with bees",

    # BT: Attempting Bike Tricks
    "bmx bike tricks",
    "bicycle jumping off a ramp",
    "doing a wheelie on a bike",
    "stunt riding a bicycle",

    # DS: Dog Show
    "dog show competition ring",
    "handler running with a dog in a show",
    "judges inspecting a purebred dog",
    "dogs competing at a dog show",
]

NEWS_PROMPTS = [
    "a news anchor in a studio",
    "a reporter on location",
    "a breaking news graphic",
    "a government press conference",
    "a protest crowd",
    "a hospital ward",
    "a wildfire",
    "flooded streets",
    "a crime scene with police tape",
    "a stock market chart on screen",
    "a sports highlight",
    "a courtroom",
    "a warzone with soldiers",
    "a classroom",
    "an airplane at an airport",
]

def load_shot_npz(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    E = d["emb"].astype(np.float32)      # (N,512) normalized
    ids = d["ids"]
    t_start = d["t_start"].astype(float)
    t_end = d["t_end"].astype(float)
    sid2idx = {str(sid): i for i, sid in enumerate(ids)}
    return E, ids, sid2idx, t_start, t_end

def load_json(p: Path):
    return json.load(open(p, "r", encoding="utf-8"))

def save_json(obj, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(p, "w", encoding="utf-8"), indent=2)

def l2norm(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + 1e-12)

def build_segment_embeddings(E: np.ndarray, segs: list, sid2idx: dict) -> np.ndarray:
    """Mean-pool shot embeddings per segment, then L2-normalize (cosine-ready)."""
    seg_vecs = []
    for seg in segs:
        idxs = [sid2idx[sid] for sid in seg["shot_ids"] if sid in sid2idx]
        if not idxs:
            seg_vecs.append(np.zeros(E.shape[1], dtype=np.float32))
            continue
        v = E[idxs].mean(axis=0)
        n = np.linalg.norm(v) + 1e-12
        seg_vecs.append((v / n).astype(np.float32))
    return torch.from_numpy(np.stack(seg_vecs, axis=0))  # (S,512)

def encode_texts(texts, device, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k"):
    """Encode a list of strings into normalized CLIP text vectors."""
    if HAVE_OPENCLIP:
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        model = model.to(device)
        model.eval()
        tokenizer = open_clip.get_tokenizer(model_name)
        with torch.no_grad():
            toks = tokenizer(texts).to(device)
            feats = model.encode_text(toks)
            feats = l2norm(feats.float())
        return feats  # (T,512)
    else:
        model, _ = clip_orig.load(model_name, device=device)
        model.eval()
        with torch.no_grad():
            toks = clip_orig.tokenize(texts).to(device)
            feats = model.encode_text(toks)
            feats = l2norm(feats.float())
        return feats

# ---- Optional OCR enrichment (Step 4C) ----
def ocr_words_from_image(img_path: str, max_words=10):
    try:
        import pytesseract
        import cv2
    except Exception:
        return []  # OCR unavailable
    img = cv2.imread(img_path)
    if img is None:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # light threshold to improve OCR on lower-thirds
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    text = pytesseract.image_to_string(gray)
    # Clean: keep words with letters/numbers, length >= 3
    toks = re.findall(r"[A-Za-z0-9\-]{3,}", text)
    # Deduplicate while preserving order
    seen, out = set(), []
    for w in toks:
        wlow = w.lower()
        if wlow not in seen:
            seen.add(wlow)
            out.append(w)
        if len(out) >= max_words:
            break
    return out

def build_prompts(preset: str, segments: list, enable_ocr: bool):
    if preset == "tvsum":
        prompts = list(TVSUM_PROMPTS)
    elif preset == "news":
        prompts = list(NEWS_PROMPTS)
    else:
        raise ValueError("preset must be 'tvsum' or 'news'.")

    # Optional OCR: add text on screen: {word} for each segments key thumbnail
    if enable_ocr:
        for seg in segments:
            thumb = seg.get("key_thumb")
            if not thumb: 
                continue
            words = ocr_words_from_image(thumb)
            for w in words:
                prompts.append(f"text on screen: {w}")
    # Deduplicate prompts
    prompts = list(dict.fromkeys(prompts))
    return prompts

def main():
    ap = argparse.ArgumentParser(description="Step 4: score segments with CLIP text (TVSum/News preset, + query, + OCR)")
    ap.add_argument("--emb", required=True, help="shots_embeddings.npz from Step 2")
    ap.add_argument("--segments", required=True, help="segments.json from Step 3")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--preset", default="tvsum", choices=["tvsum","news"], help="prompt bank to use")
    ap.add_argument("--device", default="cpu", choices=["cpu","cuda"], help="run text encoder on this device")
    ap.add_argument("--model", default="ViT-B-32", help="CLIP model (open_clip)")
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k", help="open_clip pretrained tag")
    ap.add_argument("--query", default=None, help="optional user query string to score as well")
    ap.add_argument("--ocr", action="store_true", help="enable OCR enrichment from key thumbnails")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device=="cuda" and torch.cuda.is_available()) else "cpu")

    # Load data
    E, ids, sid2idx, ts, te = load_shot_npz(Path(args.emb))
    segments = load_json(Path(args.segments))

    # Segment embeddings (CPU tensor -> move later)
    seg_vecs = build_segment_embeddings(E, segments, sid2idx)  # (S,512)

    # Build prompts (preset + optional OCR words)
    prompts = build_prompts(args.preset, segments, enable_ocr=args.ocr)
    print(f"[info] Using {len(prompts)} prompts ({args.preset}{' + OCR' if args.ocr else ''}).")

    # Encode prompts
    text_feats = encode_texts(prompts, device, model_name=args.model, pretrained=args.pretrained)  # (T,512)

    # Score: max cosine over prompts
    seg_vecs_dev = seg_vecs.to(device)
    with torch.no_grad():
        sims = seg_vecs_dev @ text_feats.T  # (S,T)
        best_scores, best_idx = sims.max(dim=1)  # per-segment
    best_scores = best_scores.cpu().numpy()
    best_idx = best_idx.cpu().numpy()

    # Optional query
    query_scores = None
    if args.query:
        q_feats = encode_texts([args.query], device, model_name=args.model, pretrained=args.pretrained)  # (1,512)
        with torch.no_grad():
            q_sims = (seg_vecs_dev @ q_feats.T).squeeze(1)  # (S,)
        query_scores = q_sims.cpu().numpy()

    # Attach scores to segments
    for i, seg in enumerate(segments):
        seg["score_generic"] = float(best_scores[i])
        seg["best_prompt"] = prompts[int(best_idx[i])]
        if query_scores is not None:
            seg["score_query"] = float(query_scores[i])

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "segments_scored.json"
    save_json(segments, out_json)
    print(f"[OK] Wrote {out_json} with scores.")

    # Print a quick preview
    top = sorted(segments, key=lambda s: s["score_generic"], reverse=True)[:8]
    print("\nTop by generic score:")
    for s in top:
        print(f" {s['segment_id']}: {s['score_generic']:.3f} | {s['best_prompt']} | {s['t_start']:.1f}-{s['t_end']:.1f}s | size={s['size']}")
    if query_scores is not None:
        topq = sorted(segments, key=lambda s: s["score_query"], reverse=True)[:8]
        print("\nTop by query score:")
        for s in topq:
            print(f" {s['segment_id']}: {s['score_query']:.3f} | {s['t_start']:.1f}-{s['t_end']:.1f}s | size={s['size']}")

if __name__ == "__main__":
    main()
