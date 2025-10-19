# tvsum_utils.py
import csv
from pathlib import Path
from typing import Dict, List

def load_tvsum_info(info_tsv_path: str) -> List[Dict]:
    """
    Returns list of dicts with fields:
      category, video_id, title, url, length
    """
    rows = []
    with open(info_tsv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for r in reader:
            rows.append({
                "category": r[0],
                "video_id": r[1],
                "title": r[2],
                "url": r[3],
                "length": r[4],
            })
    return rows


def load_tvsum_annos(anno_tsv_path: str) -> Dict[str, List[List[int]]]:
    """
    Returns: { video_id: [ [scores for shot 0..N-1] by worker, ... ] }
    Each video has 20 lines (20 workers). Each score is int 1..5.
    """
    by_vid = {}
    with open(anno_tsv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for r in reader:
            video_id = r[0]
            scores = [int(x) for x in r[2].split(",")]
            by_vid.setdefault(video_id, []).append(scores)
    return by_vid
