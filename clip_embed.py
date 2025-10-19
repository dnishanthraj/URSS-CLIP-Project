# clip_embed.py
# Step 2: Turn sampled frames (Step 1) into one CLIP embedding per shot.
# - Reads Step 1's manifest.json
# - Encodes each frame with CLIP ViT-B/32
# - Mean-pools per shot and L2-normalizes
# - Saves shots_embeddings.npz for later steps

import argparse, json
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from PIL import Image
import open_clip


def load_manifest(manifest_path: str) -> List[Dict]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


@torch.no_grad()
def encode_images(model, preprocess, device: str, img_paths: List[str], batch_size: int = 32) -> torch.Tensor:
    """
    Returns a (N, D) tensor of L2-normalized CLIP embeddings for the given image paths.
    We batch to avoid RAM spikes.
    """
    embs = []
    for i in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[i:i+batch_size]
        imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            imgs.append(preprocess(img))
        ims = torch.stack(imgs).to(device)
        feats = model.encode_image(ims)
        feats = feats / feats.norm(dim=-1, keepdim=True)  # per-frame L2 norm
        embs.append(feats)
    return torch.cat(embs, dim=0)


def main():
    ap = argparse.ArgumentParser(description="Step 2: CLIP embeddings per shot")
    ap.add_argument("--manifest", required=True, help="Path to Step-1 manifest.json")
    ap.add_argument("--out", required=True, help="Output folder (same folder as manifest is typical)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Use 'cuda' if you have a GPU")
    ap.add_argument("--batch", type=int, default=32, help="Batch size for image encoding")
    # open_clip model config (sensible defaults)
    ap.add_argument("--model", default="ViT-B-32", help="open_clip model name (e.g., ViT-B-32)")
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k", help="open_clip pretrained tag")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load shot manifest from Step 1
    shots = load_manifest(str(manifest_path))
    if not shots:
        raise RuntimeError(f"No shots found in {manifest_path}")

    # 2) Load CLIP model + preprocess
    device = args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=device
    )
    model.eval()

    shot_ids, t_starts, t_ends, shot_embs = [], [], [], []

    # 3) For each shot, encode its 13 frames, mean-pool, normalize
    for sh in shots:
        frame_paths = [p for p in sh.get("frame_paths", []) if Path(p).exists()]
        if not frame_paths:
            continue  # rare; can happen if extraction failed for a shot

        frame_feats = encode_images(model, preprocess, device, frame_paths, batch_size=args.batch)
        pooled = frame_feats.mean(dim=0, keepdim=False)
        pooled = pooled / pooled.norm()  # L2-normalize the shot vector

        shot_ids.append(sh["shot_id"])
        t_starts.append(sh["t_start"])
        t_ends.append(sh["t_end"])
        shot_embs.append(pooled.cpu().numpy())

    if not shot_embs:
        raise RuntimeError("No embeddings computed. Check manifest frame paths exist on disk.")

    E = np.vstack(shot_embs).astype(np.float32)  # (num_shots, 512)
    ids = np.array(shot_ids, dtype=object)
    t_starts = np.array(t_starts, dtype=np.float32)
    t_ends = np.array(t_ends, dtype=np.float32)

    # 4) Save for later steps
    np.savez_compressed(out_dir / "shots_embeddings.npz",
                        ids=ids, t_start=t_starts, t_end=t_ends, emb=E)

    # tiny JSON index for quick peeks (optional)
    meta = [{"id": i, "t_start": float(s), "t_end": float(e)} for i, s, e in zip(shot_ids, t_starts, t_ends)]
    with open(out_dir / "shots_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] Saved {out_dir/'shots_embeddings.npz'}")
    print(f"     emb shape = {E.shape} (num_shots, dim); dim should be 512 for ViT-B/32")
    print(f"     ids example: {ids[:3] if len(ids) >= 3 else ids}")


if __name__ == "__main__":
    main()
