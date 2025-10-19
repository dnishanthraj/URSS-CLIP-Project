#!/bin/bash
#SBATCH --job-name=tvsum-pipeline
#SBATCH --partition=gecko        # GPU partition
#SBATCH --account=ug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00        # 2 days
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ---------------- SETUP ----------------
mkdir -p "$HOME/urss/code/logs"
source /etc/profile.d/modules.sh
module load CUDA
source "$HOME/urss/code/.venv/bin/activate"

export HF_HOME="$HOME/.cache/huggingface"

cd "$HOME/urss/code"

# ---------------- INPUT ----------------
# Pass video base name (without path) as arg, e.g. sbatch run_pipeline.sbatch XkqCExn6_Us.mp4
if [ -z "$1" ]; then
    echo "[ERR] Must provide a video file name from TVSum dataset, e.g.:"
    echo "      sbatch run_pipeline.sbatch XkqCExn6_Us.mp4"
    exit 1
fi

VIDNAME="$1"
VIDPATH="../tvsum50_ver_1_1/ydata-tvsum50-v1_1/ydata-tvsum50-video/video/$VIDNAME"
BASENAME="${VIDNAME%.mp4}"
WORK="../work/${BASENAME}_fixed2s"

echo "[INFO] Processing video: $VIDNAME"
echo "[INFO] Work dir: $WORK"

# ---------------- PIPELINE ----------------

echo "[STEP 1] Shot sampling..."
python -c "from shot_sampler import build_manifest; build_manifest('$VIDPATH', '$WORK', mode='fixed2s', max_samples_per_shot=3)"

echo "[STEP 2] Embedding shots..."
MAN="$WORK/manifest.json"
python clip_embed.py --manifest "$MAN" --out "$WORK" --device cpu

echo "[STEP 3] Clustering shots..."
python cluster_shots.py \
  --emb "$WORK/shots_embeddings.npz" \
  --manifest "$WORK/manifest.json" \
  --out "$WORK" \
  --cosine_threshold 0.28

echo "[STEP 4] Scoring segments..."
python score_segments.py \
  --emb "$WORK/shots_embeddings.npz" \
  --segments "$WORK/segments.json" \
  --out "$WORK" \
  --preset tvsum \
  --device cpu

echo "[STEP 5] Knapsack selection..."
python knapsack_select.py \
  --segments_scored "$WORK/segments_scored.json" \
  --out "$WORK" \
  --use generic \
  --budget_frac 0.15

echo "[STEP 6] Captioning (BLIP on GPU)..."
python caption_segments.py \
  --work "$WORK" \
  --segments_scored "$WORK/segments_scored.json" \
  --shots_npz "$WORK/shots_embeddings.npz" \
  --device cuda \
  --align_thresh 0.20

echo "[STEP 7] Linking video..."
python link_video.py --work "$WORK" --dataset_video "$VIDPATH"

echo "[STEP 8] Analytics..."
python analytics.py --work "$WORK"

echo "[STEP 9] Make highlight video..."
python make_highlight.py --work "$WORK" --fade 0.1

echo "[STEP 10] Make text summary..."
python make_text_summary.py \
  --summary_json "$WORK/summary_captions.json" \
  --out "$WORK/summary_text.json" \
  --min_score 0.22

echo "[STEP 11] Make subtitles..."
python make_subtitles.py \
  --summary_json "$WORK/summary_captions.json" \
  --out "$WORK/highlight.srt" \
  --min_score 0.22

python make_subtitles.py \
  --summary_json "$WORK/summary_captions.json" \
  --out "$WORK/highlight.vtt" \
  --min_score 0.22

echo "[DONE] Pipeline finished for $VIDNAME"
