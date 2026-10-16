#!/usr/bin/env python3
"""모드 3 velcro_ball YOLO-Seg 비교 실험 — 모델·입력 크기·증강을 바꿔 학습하고 결과를 CSV 한 줄씩 남긴다.
담당: 박성현

    python3 mode3_sweep.py --stage 1          # 모델 6종 × 입력 640/864/1024
    python3 mode3_sweep.py --stage 2 --base yolo11s-seg:864 --base yolov8s-seg:1024   # 증강 비교
    python3 mode3_sweep.py --stage 3 --base ...                                       # 시드 반복

이미 결과가 있는 실험 이름은 건너뛴다 (중단 후 다시 실행 가능).
지표: 학습 중 최고 epoch 의 valid 지표 + best.pt 로 test 분할 재평가 + 추론 시간(배치 1, FP16, 이 GPU).
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path("/media/gt/5448E08048E061EA/rally_data/yolo")
DATA = ROOT / "rally_v1_yolov8" / "data.yaml"
RUNS = ROOT / "runs"
WEIGHTS = ROOT / "weights"
CSV = ROOT / "sweep_results.csv"
MODELS = ["yolov8n-seg", "yolov8s-seg", "yolo11n-seg", "yolo11s-seg", "yolo26n-seg", "yolo26s-seg"]
SIZES = [640, 864, 1024]
AUG = {
    "base": {},
    "rot": {"flipud": 0.5, "degrees": 15.0},                       # 손목 카메라 90° 회전·자세 변화
    "scale": {"scale": 0.9, "copy_paste": 0.3},                     # 크기 변화 크게 + 공 붙여넣기
    "long": {"epochs": 300, "patience": 80, "close_mosaic": 30},    # 오래, 마지막 30 epoch 모자이크 끔
    "lowaug": {"mosaic": 0.5, "hsv_h": 0.005, "hsv_s": 0.4},        # 공 색 보존
}
FIELDS = ["name", "model", "imgsz", "aug", "seed", "epochs_run", "best_epoch", "train_min",
          "val_mask_map50", "val_mask_map", "val_box_map50", "val_box_map", "val_p", "val_r",
          "test_mask_map50", "test_mask_map", "test_box_map50", "test_box_map", "test_p", "test_r",
          "ms_fp16", "params_m"]


def done_names() -> set[str]:
    if not CSV.exists():
        return set()
    return {r["name"] for r in csv.DictReader(open(CSV))}


def best_row(run: Path) -> tuple[dict, int, int]:
    rows = list(csv.DictReader(open(run / "results.csv")))
    rows = [{k.strip(): v for k, v in r.items()} for r in rows]
    key = "metrics/mAP50-95(M)"
    best = max(rows, key=lambda r: float(r[key]))
    return best, int(float(best["epoch"])), len(rows)


def speed_ms(model: YOLO, imgsz: int) -> float:
    img = torch.zeros(1, 3, 480, 848, dtype=torch.uint8).numpy()[0].transpose(1, 2, 0)
    for _ in range(10):
        model.predict(img, imgsz=imgsz, half=True, verbose=False)
    torch.cuda.synchronize()
    t = time.perf_counter()
    n = 50
    for _ in range(n):
        model.predict(img, imgsz=imgsz, half=True, verbose=False)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


def run_one(model_name: str, imgsz: int, aug: str, seed: int) -> None:
    name = f"{model_name}_{imgsz}_{aug}_s{seed}"
    if name in done_names():
        print(f"[skip] {name}", flush=True)
        return
    params = {"epochs": 150, "patience": 40}
    params.update(AUG[aug])
    batch = 8 if (imgsz >= 1024 and model_name.endswith("s-seg")) else 16
    print(f"[run] {name} batch {batch} {params}", flush=True)
    run = RUNS / name
    t0 = time.time()
    if (run / "weights" / "best.pt").exists() and (run / "results.csv").exists():
        print(f"[reuse] {name} 학습 결과 있음 — 평가만", flush=True)   # 중단·오류 뒤 다시 실행할 때
        train_min = float("nan")
    else:
        m = YOLO(str(WEIGHTS / f"{model_name}.pt"))
        m.train(data=str(DATA), imgsz=imgsz, batch=batch, seed=seed, deterministic=True, cache="ram", workers=4,
                project=str(RUNS), name=name, exist_ok=True, verbose=False, plots=True, **params)
        train_min = (time.time() - t0) / 60
        del m
    best, best_ep, n_ep = best_row(run)
    bm = YOLO(str(run / "weights" / "best.pt"))
    tv = bm.val(data=str(DATA), split="test", imgsz=imgsz, batch=8, verbose=False, plots=False,
                project=str(RUNS), name=name + "_test", exist_ok=True)
    row = {
        "name": name, "model": model_name, "imgsz": imgsz, "aug": aug, "seed": seed, "epochs_run": n_ep,
        "best_epoch": best_ep, "train_min": round(train_min, 1),
        "val_mask_map50": best["metrics/mAP50(M)"], "val_mask_map": best["metrics/mAP50-95(M)"],
        "val_box_map50": best["metrics/mAP50(B)"], "val_box_map": best["metrics/mAP50-95(B)"],
        "val_p": best["metrics/precision(M)"], "val_r": best["metrics/recall(M)"],
        "test_mask_map50": tv.seg.map50, "test_mask_map": tv.seg.map, "test_box_map50": tv.box.map50,
        "test_box_map": tv.box.map, "test_p": tv.seg.mp, "test_r": tv.seg.mr,
        "ms_fp16": round(speed_ms(YOLO(str(run / "weights" / "best.pt")), imgsz), 2),  # val 뒤 같은 객체로 predict 하면 inference tensor 오류
        "params_m": round(sum(p.numel() for p in bm.model.parameters()) / 1e6, 2),
    }
    new = not CSV.exists()
    with open(CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"[done] {name} val mask mAP50-95 {float(row['val_mask_map']):.3f} test {row['test_mask_map']:.3f} "
          f"{row['ms_fp16']} ms {row['train_min']} min", flush=True)
    del bm
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--base", action="append", default=[], help="model:imgsz (stage 2·3)")
    ap.add_argument("--augs", default="rot,scale,long,lowaug")
    ap.add_argument("--aug", default="base", help="stage 3 증강")
    ap.add_argument("--seeds", default="1,2")
    a = ap.parse_args()
    if a.stage == 1:
        for imgsz in SIZES:
            for m in MODELS:
                run_one(m, imgsz, "base", 0)
    elif a.stage == 2:
        for b in a.base:
            m, s = b.split(":")
            for aug in a.augs.split(","):
                run_one(m, int(s), aug, 0)
    elif a.stage == 3:
        for b in a.base:
            m, s = b.split(":")
            for seed in a.seeds.split(","):
                run_one(m, int(s), a.aug, int(seed))


if __name__ == "__main__":
    main()
