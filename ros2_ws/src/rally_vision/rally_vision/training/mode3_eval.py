#!/usr/bin/env python3
"""모드 3 velcro_ball 모델 실사용 조건 평가 — Roboflow 점수 밖의 지표.
담당: 박성현

    python3 mode3_eval.py <run 이름> [<run 이름> ...] [--conf 0.25]

평가 세트
A. 라벨 있음 (Roboflow valid + test): 카메라별(손목·측면·근접 100장)·공 크기별 재현율·정밀도 (IoU ≥ 0.5, 상자 기준)
B. 라벨 없는 배경 프레임 (bg_1010_1649 중 라벨링 안 한 것): 공이 없으니 검출 1개라도 있으면 오검출
C. 라벨 없는 평가 녹화 프레임 (throw_1010_1646, 움직이는 공 구간 'f' 중 라벨링 안 한 것): 색·움직임으로 찾은 공 위치(약한 정답)
   근처 20 px 안에 검출이 있는 비율 = '일치율'. 정답이 약하므로 재현율의 근사로만 본다.
결과: <ROOT>/eval_results.csv 에 run 마다 한 줄.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from ultralytics import YOLO

ROOT = Path("/media/gt/5448E08048E061EA/rally_data")
DS = ROOT / "yolo" / "rally_v1_yolov8"
RUNS = ROOT / "yolo" / "runs"
FRAMES = ROOT / "frames" / "mode3_1010"
OUT = ROOT / "yolo" / "eval_results.csv"
W, H = 848, 480


def gt_boxes(label: Path) -> list[np.ndarray]:
    boxes = []
    if label.exists():
        for line in label.read_text().split("\n"):
            v = line.split()
            if len(v) < 7:
                continue
            xy = np.array(v[1:], float).reshape(-1, 2) * [W, H]
            boxes.append(np.r_[xy.min(0), xy.max(0)])
    return boxes


def iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def size_bin(w: float) -> str:
    return "<15px" if w < 15 else ("15-30px" if w < 30 else ">=30px")


def evaluate(run: str, conf: float) -> dict:
    imgsz = int(run.split("_")[1])
    model = YOLO(str(RUNS / run / "weights" / "best.pt"))
    smap = {r["image"]: r for r in csv.DictReader(open(DS / "source_map.csv"))}
    labeled = {r["orig_file"] for r in smap.values() if r["orig_file"]}

    # A
    stat = defaultdict(lambda: [0, 0, 0])  # tp, fn, fp
    for split in ("valid", "test"):
        for img in sorted((DS / split / "images").glob("*.jpg")):
            src = smap[img.name]
            group = "closeup" if src["bag"] == "closeup100" else src["cam"]
            gts = gt_boxes(DS / split / "labels" / (img.stem + ".txt"))
            res = model.predict(str(img), imgsz=imgsz, conf=conf, half=True, verbose=False)[0]
            preds = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
            used = set()
            for g in gts:
                j = max(range(len(preds)), key=lambda k: iou(g, preds[k]), default=None)
                hit = j is not None and j not in used and iou(g, preds[j]) >= 0.5
                if hit:
                    used.add(j)
                for key in (group, "all", size_bin(g[2] - g[0])):
                    stat[key][0 if hit else 1] += 1
            fp = len(preds) - len(used)
            stat[group][2] += fp
            stat["all"][2] += fp

    # B
    bg = [p for p in sorted(glob.glob(str(FRAMES / "bg" / "*.png"))) if os.path.relpath(p, FRAMES) not in labeled]
    bg_fp_imgs = 0
    for p in bg:
        r = model.predict(p, imgsz=imgsz, conf=conf, half=True, verbose=False)[0]
        bg_fp_imgs += int(len(r.boxes) > 0)

    # C
    man = list(csv.DictReader(open(FRAMES / "manifest.csv")))
    weak = [r for r in man if r["split"] == "test" and r["kind"] == "f" and r["blob_x"] and r["file"] not in labeled]
    agree = defaultdict(lambda: [0, 0])
    for r in weak:
        res = model.predict(str(FRAMES / r["file"]), imgsz=imgsz, conf=conf, half=True, verbose=False)[0]
        c = res.boxes.xywh.cpu().numpy()[:, :2] if len(res.boxes) else np.zeros((0, 2))
        ok = bool(len(c)) and np.min(np.hypot(c[:, 0] - float(r["blob_x"]), c[:, 1] - float(r["blob_y"]))) < 20
        for key in (r["cam"], "all"):
            agree[key][0] += int(ok); agree[key][1] += 1

    def rec(k):
        tp, fn, _ = stat[k]
        return round(tp / (tp + fn), 3) if tp + fn else None

    def prec(k):
        tp, _, fp = stat[k]
        return round(tp / (tp + fp), 3) if tp + fp else None

    return {
        "run": run, "conf": conf,
        "rec_all": rec("all"), "prec_all": prec("all"),
        "rec_wrist": rec("wrist"), "rec_side": rec("side"), "rec_closeup": rec("closeup"),
        "prec_wrist": prec("wrist"), "prec_side": prec("side"),
        "rec_lt15": rec("<15px"), "rec_15_30": rec("15-30px"), "rec_ge30": rec(">=30px"),
        "n_lt15": sum(stat["<15px"][:2]), "n_15_30": sum(stat["15-30px"][:2]), "n_ge30": sum(stat[">=30px"][:2]),
        "bg_fp_rate": round(bg_fp_imgs / len(bg), 3) if bg else None, "bg_n": len(bg),
        "weak_agree_all": round(agree["all"][0] / agree["all"][1], 3) if agree["all"][1] else None,
        "weak_agree_wrist": round(agree["wrist"][0] / agree["wrist"][1], 3) if agree["wrist"][1] else None,
        "weak_agree_side": round(agree["side"][0] / agree["side"][1], 3) if agree["side"][1] else None,
        "weak_n": agree["all"][1],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()
    rows = []
    for run in a.runs:
        r = evaluate(run, a.conf)
        rows.append(r)
        print(r, flush=True)
    new = not OUT.exists()
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
