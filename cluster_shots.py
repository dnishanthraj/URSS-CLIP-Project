# cluster_shots.py
# Step 3: Group visually similar shots into segments with cosine distance.
# Output: segments.json listing segments and their key shot/thumbnail.

import argparse, json
from pathlib import Path
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import sklearn


def load_npz(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    E = d["emb"].astype(np.float32)       # (N, 512), already L2-normalized
    ids = d["ids"]                        # (N,)
    t_start = d["t_start"].astype(float)  # (N,)
    t_end = d["t_end"].astype(float)      # (N,)
    return E, ids, t_start, t_end


def load_manifest(manifest_path: Path):
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_thumb_lookup(manifest):
    """Return {shot_id: first_frame_path_or_None} for thumbnails."""
    thumbs = {}
    for entry in manifest:
        sid = entry["shot_id"]
        fps = entry.get("frame_paths", [])
        thumbs[sid] = fps[0] if fps else None
    return thumbs


def pick_key_shot(E_cluster: np.ndarray, idxs: list[int]) -> int:
    """
    Choose the shot closest to the cluster centroid (cosine).
    E_cluster: (k, D) normalized vectors of this cluster
    idxs: indices into the FULL embedding array for these rows
    Returns the *global* index of the chosen key shot.
    """
    centroid = E_cluster.mean(axis=0)
    norm = np.linalg.norm(centroid) + 1e-12
    centroid = centroid / norm
    sims = E_cluster @ centroid  # since all are L2-normalized
    j = int(np.argmax(sims))
    return idxs[j]


def make_clustering_model(cosine_threshold: float):
    """
    Build an AgglomerativeClustering that cuts by distance threshold.
    sklearn changed API around affinity/metric, so we handle both.
    """
    # Try new API first (metric="cosine")
    ver = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
    if ver >= (1, 4):
        return AgglomerativeClustering(
            metric="cosine",
            linkage="average",
            distance_threshold=cosine_threshold,
            n_clusters=None,
        )
    else:
        # Older sklearn expects affinity="cosine" and metric=None
        return AgglomerativeClustering(
            affinity="cosine",
            linkage="average",
            distance_threshold=cosine_threshold,
            n_clusters=None,
        )


def main():
    ap = argparse.ArgumentParser(description="Step 3: segment discovery via agglomerative clustering")
    ap.add_argument("--emb", required=True, help="Path to shots_embeddings.npz (from Step 2)")
    ap.add_argument("--manifest", required=True, help="Path to Step-1 manifest.json")
    ap.add_argument("--out", required=True, help="Output folder (e.g., same work dir)")
    ap.add_argument("--cosine_threshold", type=float, default=0.25,
                    help="Cosine distance cut (smaller = more clusters). Try 0.220.30")
    ap.add_argument("--min_cluster_size", type=int, default=1,
                    help="Drop clusters smaller than this (1 keeps everything)")
    args = ap.parse_args()

    emb_path = Path(args.emb)
    manifest_path = Path(args.manifest)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load data
    E, ids, t_start, t_end = load_npz(emb_path)
    N, D = E.shape
    if D != 512:
        print(f"[warn] embedding dim is {D}, expected 512 (ViT-B/32)")

    # 2) Cluster shots by cosine distance
    clustering = make_clustering_model(args.cosine_threshold)
    labels = clustering.fit_predict(E)  # int labels 0..(K-1)
    n_clusters = int(labels.max()) + 1
    print(f"[OK] Clustering produced {n_clusters} clusters from {N} shots at threshold={args.cosine_threshold}.")

    # 3) Build clusters -> list of idxs per label
    clusters = {}
    for i, lab in enumerate(labels):
        clusters.setdefault(int(lab), []).append(i)

    # 4) Thumbnails & time lookup
    manifest = load_manifest(manifest_path)
    thumb_of = build_thumb_lookup(manifest)
    ts_of = {sid: float(s) for sid, s in zip(ids, t_start)}
    te_of = {sid: float(e) for sid, e in zip(ids, t_end)}

    def split_into_contiguous_runs(sorted_idxs):
        runs = []
        if not sorted_idxs:
            return runs
        run = [sorted_idxs[0]]
        for a, b in zip(sorted_idxs, sorted_idxs[1:]):
            if b == a + 1:
                run.append(b)
            else:
                runs.append(run)
                run = [b]
        runs.append(run)
        return runs

    segments = []
    seg_id = 0
    for lab, idxs in clusters.items():
        if len(idxs) < args.min_cluster_size:
            continue
        # sort by time/shot order, then split where shots are non-consecutive
        idxs_sorted = sorted(idxs)
        for run in split_into_contiguous_runs(idxs_sorted):
            key_idx = pick_key_shot(E[run], run)
            key_sid = str(ids[key_idx])

            sids = [str(ids[i]) for i in run]
            seg = {
                "segment_id": f"seg_{seg_id:04d}",
                "shot_ids": sids,
                "t_start": float(min(ts_of[s] for s in sids)),
                "t_end": float(max(te_of[s] for s in sids)),
                "key_shot_id": key_sid,
                "key_thumb": thumb_of.get(key_sid),
                "size": len(sids),
            }
            segments.append(seg)
            seg_id += 1


    segments.sort(key=lambda x: x["t_start"])

    # 6) Save
    out_json = out_dir / "segments.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2)
    print(f"[OK] Wrote {out_json} with {len(segments)} segments.")
    if segments:
        ex = segments[min(3, len(segments)-1)]
        print("[example] id:", ex["segment_id"], "| shots:", ex["size"], "| range:", (ex["t_start"], ex["t_end"]), "| thumb:", ex["key_thumb"])


if __name__ == "__main__":
    main()
