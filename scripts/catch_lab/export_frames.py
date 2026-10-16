#!/usr/bin/env python3
"""모드 3 YOLO 학습용 사진 추출 — 투구 bag(mcap)에서 '움직이는 공' 구간 프레임을 PNG 로.
담당: 박성현

ROS 없이 mcap 파일을 직접 읽는다 (rosbag2_py 가 없는 PC 용). 필요: pip install mcap mcap-ros2-support

    python3 export_frames.py <bag 폴더> [<bag 폴더> ...] --split train --out <저장 루트>
    python3 export_frames.py .../bg_1010_1649 --split bg --mode every --every 40 --out <루트>

동작 (카메라마다 따로, 2번 읽음)
1. 1차: 프레임을 1/2 로 줄여 공 색(HSV, config.yaml ball.hsv) ∧ 움직임(2프레임 전과 차이) 마스크 → 공 크기·원형도
   조건을 만족하는 덩어리가 있으면 '움직이는 공 프레임'. 연속 구간(틈 ≤ max_gap)으로 묶어 길이 ≥ min_len 인 것만 구간으로.
   투구와 돌려주기 모두 공중 공이라 둘 다 쓴다.
2. 구간마다 프레임을 고르게 per_seg 장 + 구간 밖 프레임(손에 든 공·패드 위·공 없음)을 extra_ratio 만큼 무작위로.
3. 2차: 고른 프레임만 디코드해 저장. 파일명 {cam}_{bag}_{kind}{seg:03d}_{frame:05d}.png (cam = wrist | side).
   manifest.csv 에 한 줄씩 (bag, cam, split, frame, stamp_ns, kind, seg, 검출 원 x·y·r).
영상 방향은 바꾸지 않는다 (손목 카메라는 90° 돌아간 채 — 실사용 입력과 같게).
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np
import yaml
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

HERE = Path(__file__).resolve().parent
CAMS = {"wrist": "/camera/camera/color/image_raw", "side": "/side/side/color/image_raw"}


def load_hsv(cfg_path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    cfg = yaml.safe_load(cfg_path.read_text())
    return [(np.array(r["lo"], np.uint8), np.array(r["hi"], np.uint8)) for r in cfg["ball"]["hsv"]]


def to_bgr(msg) -> np.ndarray:
    img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
    enc = msg.encoding.lower()
    if enc == "rgb8":
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if enc in ("bgr8",):
        return img.copy()
    if enc in ("mono8", "8uc1"):
        return cv2.cvtColor(img[..., 0], cv2.COLOR_GRAY2BGR)
    raise ValueError(f"지원하지 않는 인코딩 {msg.encoding}")


def mcap_file(bag: Path) -> Path:
    files = sorted(bag.glob("*.mcap"))
    if not files:
        raise FileNotFoundError(f"{bag} 에 .mcap 없음")
    return files[0]


def iter_images(bag: Path, topic: str):
    """(index, stamp_ns, msg) — bag 기록 순서."""
    with open(mcap_file(bag), "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for i, (_, _, _, msg) in enumerate(reader.iter_decoded_messages(topics=[topic])):
            stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
            yield i, stamp, msg


def scan(bag: Path, topic: str, hsv, args) -> tuple[list[int], list[tuple | None]]:
    """1차: 프레임별 '움직이는 공' 여부와 원(x, y, r, 원본 픽셀)."""
    flags, blobs, hist = [], [], []
    k = 2
    min_a, max_a = args.min_area / 4, args.max_area / 4  # 1/2 축소라 면적은 1/4
    for _, _, msg in iter_images(bag, topic):
        small = cv2.resize(to_bgr(msg), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hist.append(gray)
        if len(hist) > k + 1:
            hist.pop(0)
        hit = None
        if len(hist) > k:
            moving = cv2.absdiff(gray, hist[0]) > args.diff
            hsvimg = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            color = np.zeros(gray.shape, bool)
            for lo, hi in hsv:
                color |= cv2.inRange(hsvimg, lo, hi) > 0
            mask = cv2.morphologyEx((color & moving).astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
            best = 0.0
            for j in range(1, n):
                a = stats[j, cv2.CC_STAT_AREA]
                w, h = stats[j, cv2.CC_STAT_WIDTH], stats[j, cv2.CC_STAT_HEIGHT]
                if not (min_a <= a <= max_a) or max(w, h) > 3 * max(1, min(w, h)):
                    continue
                fill = a / float(w * h)
                if fill >= 0.35 and a > best:
                    best = a
                    hit = (float(cents[j][0] * 2), float(cents[j][1] * 2), float(max(w, h)))  # 원본 좌표
        flags.append(hit is not None)
        blobs.append(hit)
    return flags, blobs


def segments(flags: list[bool], max_gap: int, min_len: int) -> list[tuple[int, int]]:
    segs, start, last = [], None, None
    for i, f in enumerate(flags):
        if not f:
            continue
        if start is None:
            start = last = i
        elif i - last <= max_gap:
            last = i
        else:
            segs.append((start, last))
            start = last = i
    if start is not None:
        segs.append((start, last))
    return [(a, b) for a, b in segs if b - a + 1 >= min_len]


def choose(n_frames: int, segs, per_seg: int, extra_ratio: float, rng) -> dict[int, tuple[str, int]]:
    """{frame: (kind, seg)}. kind: f = 움직이는 공 구간, x = 구간 밖."""
    pick: dict[int, tuple[str, int]] = {}
    inside = set()
    for s, (a, b) in enumerate(segs):
        inside.update(range(a, b + 1))
        for idx in np.unique(np.linspace(a, b, num=min(per_seg, b - a + 1)).round().astype(int)):
            pick[int(idx)] = ("f", s)
    outside = [i for i in range(n_frames) if i not in inside]
    n_extra = int(round(len(pick) * extra_ratio))
    for idx in rng.sample(outside, min(n_extra, len(outside))):
        pick[idx] = ("x", 0)
    return pick


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bags", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True, help="저장 루트. <out>/<split>/ 에 PNG, <out>/manifest.csv 누적")
    ap.add_argument("--split", required=True, help="train | val | test | bg")
    ap.add_argument("--mode", choices=["flight", "every"], default="flight")
    ap.add_argument("--every", type=int, default=40, help="mode=every: N 프레임마다 1장")
    ap.add_argument("--per-seg", type=int, default=5, help="움직이는 공 구간마다 장수")
    ap.add_argument("--extra-ratio", type=float, default=0.2, help="구간 밖 프레임 비율 (구간 프레임 수 대비)")
    ap.add_argument("--max-gap", type=int, default=6)
    ap.add_argument("--min-len", type=int, default=8)
    ap.add_argument("--diff", type=int, default=18, help="움직임 판정 밝기 차이")
    ap.add_argument("--min-area", type=float, default=40, help="원본 픽셀 면적 하한 (측면 공 ≈ 12 px → 113)")
    ap.add_argument("--max-area", type=float, default=6000)
    ap.add_argument("--cams", default="wrist,side")
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    hsv = load_hsv(args.config)
    rng = random.Random(args.seed)
    out = args.out / args.split
    out.mkdir(parents=True, exist_ok=True)
    man_path = args.out / "manifest.csv"
    new_man = not man_path.exists()
    with open(man_path, "a", newline="") as mf:
        w = csv.writer(mf)
        if new_man:
            w.writerow(["file", "bag", "cam", "split", "frame", "stamp_ns", "kind", "seg", "blob_x", "blob_y", "blob_d"])
        for bag in args.bags:
            bag = bag.resolve()
            for cam in args.cams.split(","):
                topic = CAMS[cam]
                if args.mode == "every":
                    n = sum(1 for _ in iter_images(bag, topic))
                    pick = {i: ("b", 0) for i in range(0, n, args.every)}
                    blobs = [None] * n
                    segs = []
                else:
                    flags, blobs = scan(bag, topic, hsv, args)
                    segs = segments(flags, args.max_gap, args.min_len)
                    pick = choose(len(flags), segs, args.per_seg, args.extra_ratio, rng)
                saved = 0
                for i, stamp, msg in iter_images(bag, topic):
                    if i not in pick:
                        continue
                    kind, seg = pick[i]
                    name = f"{cam}_{bag.name}_{kind}{seg:03d}_{i:05d}.png"
                    cv2.imwrite(str(out / name), to_bgr(msg))
                    b = blobs[i] if i < len(blobs) else None
                    w.writerow([f"{args.split}/{name}", bag.name, cam, args.split, i, stamp, kind, seg,
                                *(("%.1f" % v) for v in b)] if b else
                               [f"{args.split}/{name}", bag.name, cam, args.split, i, stamp, kind, seg, "", "", ""])
                    saved += 1
                mf.flush()
                print(f"[export] {bag.name} {cam}: 구간 {len(segs)}개, 저장 {saved}장", flush=True)


if __name__ == "__main__":
    main()
