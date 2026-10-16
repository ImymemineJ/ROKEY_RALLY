#!/usr/bin/env python3
"""모드 3 공 3D 궤적 — YOLO 검출(두 카메라) → 손목 깊이 3D → 측면 카메라 캘리브레이션 → 삼각측량 → 투구 표.
담당: 박성현.  ROS 없이 mcap 직접 읽음 (pip: mcap mcap-ros2-support ultralytics).

    python3 mode3_traj.py detect --model best.pt --imgsz 864 <bag> [<bag> ...]   # 프레임별 검출 CSV
    python3 mode3_traj.py calib  <calib bag>                                       # 측면 카메라 위치 (base_link)
    python3 mode3_traj.py traj   <throw bag> [...]                                 # 3D 궤적·투구 표

좌표: base_link [m]. 손목 카메라 = tcp_pose(base←TCP) × hand_eye.yaml(TCP←컬러 광학). 측면 = calib 결과.
깊이: 원본 깊이(mm) × depth_scale(hand_eye.yaml, 0.971), 컬러→깊이는 tf_static(같은 회전, x −0.015 m).
출력 폴더: --out (기본 /media/gt/5448E08048E061EA/rally_data/traj)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from catch_lab.camera import Camera  # noqa: E402
from catch_lab.detect import Blob, ball_center_depth_raw  # noqa: E402

REPO = HERE.parent.parent
HAND_EYE = REPO / "ros2_ws/src/rally_bringup/config/hand_eye.yaml"
OUT_DEFAULT = Path("/media/gt/5448E08048E061EA/rally_data/traj")
T_WRIST, T_SIDE = "/camera/camera/color/image_raw", "/side/side/color/image_raw"
I_WRIST, I_SIDE = "/camera/camera/color/camera_info", "/side/side/color/camera_info"
T_DEPTH, I_DEPTH = "/camera/camera/depth/image_rect_raw", "/camera/camera/depth/camera_info"
T_TCP = "/rally/robot/tcp_pose"
BALL_R = 0.0634 / 2
G = np.array([0.0, 0.0, -9.81])


# ---------------------------------------------------------------- 공통
def mcap_path(bag: Path) -> Path:
    return sorted(Path(bag).glob("*.mcap"))[0]


def iter_msgs(bag: Path, topics: list[str]):
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory
    with open(mcap_path(bag), "rb") as f:
        r = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, ch, _, m in r.iter_decoded_messages(topics=topics):
            yield ch.topic, m


def stamp(m) -> float:
    return m.header.stamp.sec + m.header.stamp.nanosec * 1e-9


def to_bgr(m) -> np.ndarray:
    img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if m.encoding.lower() == "rgb8" else img.copy()


def cam_from_info(m) -> Camera:
    return Camera(K=np.array(m.k, float).reshape(3, 3), D=np.zeros(5), width=int(m.width), height=int(m.height))


def quat_to_R(x, y, z, w) -> np.ndarray:
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def hand_eye() -> tuple[np.ndarray, float]:
    """TCP ← 컬러 광학 4x4, depth_scale."""
    he = yaml.safe_load(HAND_EYE.read_text())
    T = np.eye(4)
    T[:3, :3] = quat_to_R(*he["tcp_T_camera_optical"]["quat_xyzw"])
    T[:3, 3] = he["tcp_T_camera_optical"]["xyz_m"]
    return T, float(he.get("depth_scale", 1.0))


def pose_to_T(p) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_to_R(p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)
    T[:3, 3] = [p.position.x, p.position.y, p.position.z]
    return T


# ---------------------------------------------------------------- detect
def cmd_detect(a) -> None:
    from ultralytics import YOLO
    model = YOLO(a.model)
    T_tc, dscale = hand_eye()
    for bag in a.bags:
        bag = Path(bag)
        out = a.out / f"{bag.name}_det.csv"
        meta = {"bag": bag.name, "model": str(a.model), "imgsz": a.imgsz, "conf": a.conf}
        cams: dict[str, Camera] = {}
        depth_cam = None
        last_depth = None   # (stamp, depth_m)
        tcp = []            # (stamp, T base←TCP)
        rows = []
        frames: dict[str, list] = {"wrist": [], "side": []}   # (idx, t, [(u, v, d, conf, z), ...]) — 후보 전부 (정지 오검출 제거는 bag 끝에서)
        batch: list[tuple] = []

        def flush():
            if not batch:
                return
            res = model.predict([b[3] for b in batch], imgsz=a.imgsz, conf=a.conf, half=True, verbose=False)
            for (cam, idx, t, _, dep), r in zip(batch, res):
                cands = []
                for k in range(len(r.boxes)):
                    x1, y1, x2, y2 = r.boxes.xyxy[k].tolist()
                    u, v, d = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1)
                    z = np.nan
                    if cam == "wrist" and dep is not None and depth_cam is not None:
                        z = ball_center_depth_raw(dep, Blob(u=u, v=v, r=d / 2, area=0.0, circularity=1.0), cams["wrist"], depth_cam, BALL_R)
                    cands.append((u, v, d, float(r.boxes.conf[k]), float(z)))
                frames[cam].append((idx, t, cands))
            batch.clear()

        def select_rows():
            """프레임마다 후보 중 conf 최고 하나. 단 같은 자리에 오래 머무는 검출(초록 브래킷·테이블 장식, 10/12 발견)은 뺀다:
            격자 static_px 로 묶어 그 칸에 검출이 프레임의 static_frac 넘게 나오면 '정지 물체'."""
            for cam, fr in frames.items():
                n_frames = max(1, len(fr))
                cell = {}
                for _, _, cands in fr:
                    for u, v, *_ in cands:
                        key = (int(u // a.static_px), int(v // a.static_px))
                        cell[key] = cell.get(key, 0) + 1
                static = {k for k, n in cell.items() if n > a.static_frac * n_frames}
                if static:
                    print(f"[detect] {bag.name} {cam}: 정지 오검출 칸 {len(static)}개 제외 "
                          f"{[tuple(int(c * a.static_px + a.static_px / 2) for c in k) for k in sorted(static)][:6]}", flush=True)
                for idx, t, cands in fr:
                    live = [c for c in cands if (int(c[0] // a.static_px), int(c[1] // a.static_px)) not in static]
                    if not live:
                        rows.append([cam, idx, f"{t:.6f}", "", "", "", "", len(cands), ""])
                        continue
                    u, v, d, cf, z = max(live, key=lambda c: c[3])
                    rows.append([cam, idx, f"{t:.6f}", f"{u:.2f}", f"{v:.2f}", f"{d:.2f}", f"{cf:.3f}", len(cands),
                                 f"{z:.4f}" if np.isfinite(z) else ""])

        idx = {"wrist": 0, "side": 0}
        for topic, m in iter_msgs(bag, [T_WRIST, T_SIDE, I_WRIST, I_SIDE, T_DEPTH, I_DEPTH, T_TCP]):
            if topic == I_WRIST and "wrist" not in cams:
                cams["wrist"] = cam_from_info(m)
            elif topic == I_SIDE and "side" not in cams:
                cams["side"] = cam_from_info(m)
            elif topic == I_DEPTH and depth_cam is None:
                depth_cam = cam_from_info(m)
                depth_cam.t = np.array([-0.015, 0.0, 0.0])   # 컬러 광학 → 깊이 광학 (tf_static)
            elif topic == T_DEPTH:
                last_depth = (stamp(m), np.frombuffer(m.data, np.uint16).reshape(m.height, m.width) * (0.001 * dscale))
            elif topic == T_TCP:
                tcp.append((stamp(m), pose_to_T(m.pose)))
            elif topic in (T_WRIST, T_SIDE):
                cam = "wrist" if topic == T_WRIST else "side"
                t = stamp(m)
                dep = last_depth[1] if (cam == "wrist" and last_depth and abs(last_depth[0] - t) < 0.02) else None
                batch.append((cam, idx[cam], t, to_bgr(m), dep))
                idx[cam] += 1
                if len(batch) >= 32:
                    flush()
        flush()
        select_rows()
        rows.sort(key=lambda r: (r[0], r[1]))
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["cam", "frame", "stamp", "u", "v", "d_px", "conf", "n_det", "z_wrist"])
            w.writerows(rows)
        meta["cams"] = {k: {"K": c.K.tolist(), "w": c.width, "h": c.height} for k, c in cams.items()}
        meta["tcp"] = [[t, T.tolist()] for t, T in tcp[:: max(1, len(tcp) // 50)]]
        meta["T_tcp_cam"] = T_tc.tolist()
        (a.out / f"{bag.name}_meta.json").write_text(json.dumps(meta))
        n = {c: sum(1 for r in rows if r[0] == c and r[3]) for c in ("wrist", "side")}
        print(f"[detect] {bag.name}: 손목 {n['wrist']}/{idx['wrist']} · 측면 {n['side']}/{idx['side']} 프레임에 공 → {out}", flush=True)


# ---------------------------------------------------------------- 불러오기
def load_det(out: Path, bag: str):
    rows = list(csv.DictReader(open(out / f"{bag}_det.csv")))
    meta = json.loads((out / f"{bag}_meta.json").read_text())
    det = {}
    for cam in ("wrist", "side"):
        rs = [r for r in rows if r["cam"] == cam]
        t = np.array([float(r["stamp"]) for r in rs])
        uv = np.array([[float(r["u"]), float(r["v"])] if r["u"] else [np.nan, np.nan] for r in rs])
        d = np.array([float(r["d_px"]) if r["d_px"] else np.nan for r in rs])
        z = np.array([float(r["z_wrist"]) if r.get("z_wrist") else np.nan for r in rs])
        det[cam] = {"t": t, "uv": uv, "d": d, "z": z}
    cams = {k: Camera(K=np.array(v["K"]), D=np.zeros(5), width=v["w"], height=v["h"]) for k, v in meta["cams"].items()}
    # 투구 중 로봇 정지 가정 → tcp 중앙값 자세. (움직였으면 경고)
    Ts = np.array([np.array(T) for _, T in meta["tcp"]])
    spread = np.ptp(Ts[:, :3, 3], axis=0).max() if len(Ts) else 0
    T_bt = Ts[len(Ts) // 2] if len(Ts) else np.eye(4)
    T_bc = T_bt @ np.array(meta["T_tcp_cam"])    # base ← 손목 컬러 광학
    return det, cams, T_bc, spread


def wrist_points_base(det, cam: Camera, T_bc: np.ndarray) -> np.ndarray:
    """손목 검출 + 깊이 → base 3D (N,3), 없으면 nan."""
    uv, z = det["wrist"]["uv"], det["wrist"]["z"]
    P = np.full((len(uv), 3), np.nan)
    ok = np.isfinite(uv[:, 0]) & np.isfinite(z)
    if ok.any():
        Pc = cam.backproject(uv[ok], z[ok])
        P[ok] = Pc @ T_bc[:3, :3].T + T_bc[:3, 3]
    return P


def interp_uv(t_src, uv_src, t_q, max_gap=0.025):
    """시각 t_q 에서 픽셀 선형 보간 (양옆 검출이 max_gap 안에 있을 때만)."""
    out = np.full((len(t_q), 2), np.nan)
    ok = np.isfinite(uv_src[:, 0])
    ts, us = t_src[ok], uv_src[ok]
    if len(ts) < 2:
        return out
    j = np.searchsorted(ts, t_q)
    for i, (q, k) in enumerate(zip(t_q, j)):
        if 0 < k < len(ts) and ts[k] - ts[k - 1] <= max_gap:
            a = (q - ts[k - 1]) / (ts[k] - ts[k - 1])
            out[i] = us[k - 1] * (1 - a) + us[k] * a
    return out


# ---------------------------------------------------------------- calib
def cmd_calib(a) -> None:
    bag = Path(a.bag).name
    det, cams, T_bc, spread = load_det(a.out, bag)
    P = wrist_points_base(det, cams["wrist"], T_bc)
    tw = det["wrist"]["t"]
    uvw = det["wrist"]["uv"]
    # 정지: 앞뒤 ±8 프레임(약 0.27 s) 동안 손목 픽셀 이동 < 3 px, 3D(깊이 포함) 이동 < 6 cm.
    # 손목 카메라가 공을 2.5~3.3 m 에서 보므로 깊이 흔들림이 수 cm → 창 안 중앙값을 대표값으로 쓴다.
    uvs_all = interp_uv(det["side"]["t"], det["side"]["uv"], tw)
    h = 8
    cand = []
    for i in range(h, len(tw) - h):
        sl = slice(i - h, i + h + 1)
        if not (np.isfinite(uvw[sl]).all() and np.isfinite(P[sl]).all() and np.isfinite(uvs_all[sl]).all()):
            continue
        if np.ptp(uvw[sl], axis=0).max() < 3.0 and np.ptp(uvs_all[sl], axis=0).max() < 3.0 \
                and np.ptp(P[sl], axis=0).max() < 0.06:
            cand.append((i, np.median(P[sl], axis=0), np.median(uvs_all[sl], axis=0)))
    reps = []
    for i, Pm, um in cand:
        if all(np.linalg.norm(Pm - r[1]) > a.sep for r in reps):
            reps.append((i, Pm, um))
    ok = np.zeros(len(tw), bool)
    ok[[c[0] for c in cand]] = True
    obj = np.array([r[1] for r in reps], np.float64)
    img = np.array([r[2] for r in reps], np.float64)
    print(f"[calib] 정지 프레임 {ok.sum()}개 → 대표 위치 {len(reps)}곳 (손목 자세 흔들림 {spread*1000:.1f} mm)")
    np.savez(a.out / f"{bag}_static.npz", obj=obj, img=img)   # 번들 조정(calib3 --static)의 거리 기준
    if len(obj) < 6:
        print("[calib] 대표 위치가 6곳 미만 — PnP 생략 (calib3 --static 으로만 사용)")
        return
    K = cams["side"].K
    okp, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, K, None, reprojectionError=6.0, iterationsCount=2000,
                                             flags=cv2.SOLVEPNP_ITERATIVE)
    inl = inl.ravel() if inl is not None else np.arange(len(obj))
    rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], K, None, rvec, tvec)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
    err = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)
    R, _ = cv2.Rodrigues(rvec)
    cam_pos = (-R.T @ tvec).ravel()
    res = {"bag": bag, "R_side_base": R.tolist(), "t_side_base": tvec.ravel().tolist(),
           "side_pos_base": cam_pos.tolist(), "n_points": len(obj), "n_inliers": int(len(inl)),
           "reproj_px_median": float(np.median(err[inl])), "reproj_px_p90": float(np.percentile(err[inl], 90)),
           "depth_m_range": [float(np.min(np.linalg.norm(obj - cam_pos, axis=1))),
                             float(np.max(np.linalg.norm(obj - cam_pos, axis=1)))]}
    (a.out / "side_calib.json").write_text(json.dumps(res, indent=1))
    print(f"[calib] 측면 카메라 위치 base = ({cam_pos[0]:.3f}, {cam_pos[1]:.3f}, {cam_pos[2]:.3f}) m · "
          f"inlier {len(inl)}/{len(obj)} · 재투영 중앙 {res['reproj_px_median']:.2f} px, 90% {res['reproj_px_p90']:.2f} px")


def interp_P(t_src, P_src, t_q, max_gap=0.025, max_jump=0.15):
    """시각 t_q 에서 3D 선형 보간 (양옆 점이 max_gap 안·이동 max_jump 이하일 때)."""
    out = np.full((len(t_q), 3), np.nan)
    ok = np.isfinite(P_src[:, 0])
    ts, Ps = t_src[ok], P_src[ok]
    if len(ts) < 2:
        return out
    j = np.searchsorted(ts, t_q)
    for i, (q, k) in enumerate(zip(t_q, j)):
        if 0 < k < len(ts) and ts[k] - ts[k - 1] <= max_gap and np.linalg.norm(Ps[k] - Ps[k - 1]) <= max_jump:
            a = (q - ts[k - 1]) / (ts[k] - ts[k - 1])
            out[i] = Ps[k - 1] * (1 - a) + Ps[k] * a
    return out


def cmd_calib2(a) -> None:
    """여러 bag 의 (손목 깊이 3D, 측면 픽셀) 대응점을 모아 측면 카메라 자세 + 시각 차이를 함께 구한다."""
    data = []
    K = None
    for bag in a.bags:
        bag = Path(bag).name
        det, cams, T_bc, _ = load_det(a.out, bag)
        K = cams["side"].K
        data.append((bag, det["wrist"]["t"], wrist_points_base(det, cams["wrist"], T_bc), det["side"]["t"], det["side"]["uv"]))
    best = None
    for dt in np.arange(a.dt_min, a.dt_max + 1e-9, a.dt_step):
        obj, img = [], []
        for _, tw, P, tside, uvs in data:
            Pq = interp_P(tw, P, tside + dt)
            ok = np.isfinite(Pq[:, 0]) & np.isfinite(uvs[:, 0])
            obj.append(Pq[ok]); img.append(uvs[ok])
        obj, img = np.vstack(obj).astype(np.float64), np.vstack(img).astype(np.float64)
        okp, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, K, None, reprojectionError=4.0, iterationsCount=3000,
                                                 confidence=0.999)
        if not okp or inl is None:
            continue
        inl = inl.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], K, None, rvec, tvec)
        err = np.linalg.norm(cv2.projectPoints(obj[inl], rvec, tvec, K, None)[0].reshape(-1, 2) - img[inl], axis=1)
        score = (len(inl) / len(obj), -np.median(err))
        print(f"[calib2] dt {dt*1000:+5.1f} ms: 대응점 {len(obj)} · inlier {len(inl)} ({100*len(inl)/len(obj):.0f} %) · "
              f"재투영 중앙 {np.median(err):.2f} px", flush=True)
        if best is None or score > best[0]:
            best = (score, dt, rvec, tvec, len(obj), len(inl), float(np.median(err)), float(np.percentile(err, 90)))
    _, dt, rvec, tvec, n, ni, med, p90 = best
    R, _ = cv2.Rodrigues(rvec)
    pos = (-R.T @ tvec).ravel()
    res = {"bags": [d[0] for d in data], "dt_side_to_wrist_s": float(dt), "R_side_base": R.tolist(),
           "t_side_base": tvec.ravel().tolist(), "side_pos_base": pos.tolist(), "n_points": n, "n_inliers": ni,
           "reproj_px_median": med, "reproj_px_p90": p90,
           "tape_measure_pos": [1.00, -2.45, 0.865]}
    (a.out / "side_calib.json").write_text(json.dumps(res, indent=1))
    print(f"[calib2] 선택 dt {dt*1000:+.1f} ms · 측면 카메라 base = ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) m "
          f"(줄자 1.00, −2.45, 0.865) · inlier {ni}/{n} · 재투영 중앙 {med:.2f} px, 90 % {p90:.2f} px")


def sim_times(p0, v0, k, tt, h=0.004):
    """tt(정렬 안 돼도 됨, t0 기준 초) 위치 (N,3). 고정 간격 RK4 + 선형 보간."""
    T = max(float(np.max(tt)), 0.0) + h
    n = int(np.ceil(T / h)) + 1
    ps = np.zeros((n, 3)); p, v = np.array(p0, float), np.array(v0, float)
    for i in range(n):
        ps[i] = p
        p, v = _rk4(p, v, k, h)
    x = np.clip(np.asarray(tt) / h, 0, n - 1.001)
    i0 = np.floor(x).astype(int); f = (x - i0)[:, None]
    return ps[i0] * (1 - f) + ps[i0 + 1] * f


def cmd_calib3(a) -> None:
    """번들 조정: 측면 카메라 자세(6) + 시각 차(1) + 공기 저항 k(1) + 비행마다 (p0, v0).
    관측 = 두 카메라 픽셀. 손목 카메라 자세는 핸드아이(고정). 중력 9.81 이 길이 기준."""
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix
    cal = json.loads((a.out / "side_calib.json").read_text())
    R0, t0 = np.array(cal["R_side_base"]), np.array(cal["t_side_base"])
    dt0 = cal.get("dt_side_to_wrist_s", 0.0) if a.fix_dt is None else a.fix_dt
    fl = []   # (Kw, Rw, tw_, Ks, tw_obs, uvw_obs, ts_obs, uvs_obs, p0, v0)
    for bag in a.bags:
        bag = Path(bag).name
        det, cams, T_bc, _ = load_det(a.out, bag)
        T_cb = np.linalg.inv(T_bc)
        A = np.load(a.out / f"{bag}_P.npy"); tP, P = A[:, 0], A[:, 1:]
        for s_, e_ in flights(tP, P):
            sel = np.arange(s_, e_ + 1); sel = sel[np.isfinite(P[sel, 0])]
            if len(sel) < a.min_pts:
                continue
            ta, tb = tP[sel][0] - 0.03, tP[sel][-1] + 0.03
            w = det["wrist"]; sd = det["side"]
            if a.full_side:
                # 측면 카메라가 본 같은 비행 전체로 창을 넓힌다: 연속 검출(틈 ≤ 0.04 s) + 프레임당 3 px 넘게 움직이는 동안
                tsd, usd = sd["t"] + dt0, sd["uv"]
                okd = np.where(np.isfinite(usd[:, 0]))[0]
                if len(okd):
                    j = okd[np.argmin(np.abs(tsd[okd] - 0.5 * (ta + tb)))]
                    pos = int(np.where(okd == j)[0][0])
                    lo = hi = pos
                    while lo > 0 and tsd[okd[lo]] - tsd[okd[lo - 1]] <= 0.04 and \
                            np.linalg.norm(usd[okd[lo]] - usd[okd[lo - 1]]) > 3.0:
                        lo -= 1
                    while hi < len(okd) - 1 and tsd[okd[hi + 1]] - tsd[okd[hi]] <= 0.04 and \
                            np.linalg.norm(usd[okd[hi + 1]] - usd[okd[hi]]) > 3.0:
                        hi += 1
                    ta, tb = min(ta, tsd[okd[lo]]), max(tb, tsd[okd[hi]])
            mw = (w["t"] >= ta) & (w["t"] <= tb) & np.isfinite(w["uv"][:, 0])
            ms = (sd["t"] + dt0 >= ta) & (sd["t"] + dt0 <= tb) & np.isfinite(sd["uv"][:, 0])
            p0, v0, _, _ = fit_flight(tP[sel], P[sel], k=0.015)
            if tb - ta < a.min_dur:
                continue
            fl.append(dict(Kw=cams["wrist"].K, rw=cv2.Rodrigues(T_cb[:3, :3])[0].ravel(), tw=T_cb[:3, 3], Ks=cams["side"].K,
                           tref=tP[sel][0], tw_obs=w["t"][mw], uvw=w["uv"][mw], ts_obs=sd["t"][ms], uvs=sd["uv"][ms],
                           p0=simulate(p0, v0, 0.015, np.array([max(0.0, ta - tP[sel][0])]))[0] if False else p0,
                           v0=v0, bag=bag, dur=tb - ta))
    if a.max_flights and len(fl) > a.max_flights:
        fl = sorted(fl, key=lambda f: -(len(f["uvw"]) + len(f["uvs"])))[:a.max_flights]
    nF = len(fl)
    rs0 = cv2.Rodrigues(R0)[0].ravel()
    x0 = np.r_[rs0, t0, dt0, 0.015, np.concatenate([np.r_[f["p0"], f["v0"]] for f in fl])]
    st_obj, st_img = [], []
    for sf in a.static:
        z = np.load(sf); st_obj.append(z["obj"]); st_img.append(z["img"])
    st_obj = np.vstack(st_obj) if st_obj else np.zeros((0, 3))
    st_img = np.vstack(st_img) if st_img else np.zeros((0, 2))
    Ks_st = fl[0]["Ks"]
    sizes = [len(f["uvw"]) * 2 + len(f["uvs"]) * 2 for f in fl]
    offs = np.cumsum([0] + sizes)

    def resid(x):
        rs, ts, dt, k = x[:3], x[3:6], (x[6] if a.fix_dt is None else a.fix_dt), (x[7] if a.fix_k is None else a.fix_k)
        out = []
        for i, f in enumerate(fl):
            p0, v0 = x[8 + 6 * i:11 + 6 * i], x[11 + 6 * i:14 + 6 * i]
            Pw = sim_times(p0, v0, k, f["tw_obs"] - f["tref"])
            uw = cv2.projectPoints(Pw, f["rw"], f["tw"], f["Kw"], None)[0].reshape(-1, 2)
            Ps = sim_times(p0, v0, k, f["ts_obs"] + dt - f["tref"])
            us = cv2.projectPoints(Ps, rs, ts, f["Ks"], None)[0].reshape(-1, 2)
            out += [(uw - f["uvw"]).ravel(), (us - f["uvs"]).ravel()]
        if len(st_obj):
            ust = cv2.projectPoints(st_obj, rs, ts, Ks_st, None)[0].reshape(-1, 2)
            out.append(a.w_static * (ust - st_img).ravel())
        return np.concatenate(out)

    J = lil_matrix((offs[-1] + 2 * len(st_obj), len(x0)), dtype=int)
    if len(st_obj):
        J[offs[-1]:, 0:6] = 1
    for i, f in enumerate(fl):
        r0, r1 = offs[i], offs[i + 1]
        J[r0:r1, 8 + 6 * i:14 + 6 * i] = 1
        J[r0 + 2 * len(f["uvw"]):r1, 0:7] = 1     # 측면 관측만 측면 자세·dt 에 의존
        J[r0:r1, 7] = 1                           # k
    r_init = resid(x0)[:offs[-1]]
    sol = least_squares(resid, x0, jac_sparsity=J, loss="soft_l1", f_scale=a.f_scale, x_scale="jac", max_nfev=200)
    r = resid(sol.x)[:offs[-1]].reshape(-1, 2)
    if len(st_obj):
        est = np.linalg.norm(resid(sol.x)[offs[-1]:].reshape(-1, 2), axis=1) / a.w_static
        print(f"[calib3] 정지점 {len(st_obj)}개 재투영 중앙 {np.median(est):.2f} px, 90 % {np.percentile(est, 90):.2f} px")
    e = np.linalg.norm(r, axis=1)
    R, _ = cv2.Rodrigues(sol.x[:3]); tv = sol.x[3:6]; pos = (-R.T @ tv).ravel()
    res = dict(cal)
    res.update({"method": "bundle", "bags_ba": [Path(b).name for b in a.bags], "n_flights": nF,
                "R_side_base": R.tolist(), "t_side_base": tv.tolist(), "side_pos_base": pos.tolist(),
                "dt_side_to_wrist_s": float(sol.x[6]), "k_drag": float(sol.x[7]),
                "reproj_px_median": float(np.median(e)), "reproj_px_p90": float(np.percentile(e, 90)),
                "reproj_px_median_init": float(np.median(np.linalg.norm(r_init.reshape(-1, 2), axis=1)))})
    res["cost"] = float(sol.cost)
    if not a.dry:
        (a.out / "side_calib.json").write_text(json.dumps(res, indent=1))
    else:
        tag = f"dt{(a.fix_dt if a.fix_dt is not None else sol.x[6])*1000:+.0f}_k{'free' if a.fix_k is None else a.fix_k}"
        (a.out / f"side_calib_{tag}.json").write_text(json.dumps(res, indent=1))
    print(f"[calib3] 비행 길이 중앙 {np.median([f['dur'] for f in fl]):.2f} s · cost {sol.cost:.1f}  비행 {nF}개 · 관측 {len(e)} · 재투영 중앙 {res['reproj_px_median_init']:.2f} → {np.median(e):.2f} px "
          f"(90 % {np.percentile(e, 90):.2f}) · dt {sol.x[6]*1000:+.1f} ms · k {sol.x[7]:.4f} /m · "
          f"측면 카메라 ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) m (줄자 1.00, −2.45, 0.865)")


# ---------------------------------------------------------------- 삼각측량·궤적
def triangulate(Pw: np.ndarray, Ps: np.ndarray, uvw: np.ndarray, uvs: np.ndarray) -> np.ndarray:
    """DLT 두 시점. Pw/Ps = 3x4 투영행렬 (base → 픽셀)."""
    X = cv2.triangulatePoints(Pw, Ps, uvw.T.astype(np.float64), uvs.T.astype(np.float64))
    return (X[:3] / X[3]).T


def proj_mat(K, R, t):
    return K @ np.hstack([R, np.asarray(t).reshape(3, 1)])


def simulate(p0, v0, k, ts):
    """중력 + 2차 공기 저항, RK4. ts 는 0 부터 오름차순."""
    out = np.zeros((len(ts), 3))
    p, v, tc = np.array(p0, float), np.array(v0, float), 0.0
    j = 0
    dt = 0.002
    while j < len(ts):
        while tc + dt <= ts[j] + 1e-12:
            p, v = _rk4(p, v, k, dt)
            tc += dt
        h = ts[j] - tc
        pp, _ = _rk4(p, v, k, h) if h > 1e-9 else (p, v)
        out[j] = pp
        j += 1
    return out


def _acc(v, k):
    return G - k * np.linalg.norm(v) * v


def _rk4(p, v, k, h):
    a1 = _acc(v, k); k1p, k1v = v, a1
    a2 = _acc(v + 0.5 * h * k1v, k); k2p, k2v = v + 0.5 * h * k1v, a2
    a3 = _acc(v + 0.5 * h * k2v, k); k3p, k3v = v + 0.5 * h * k2v, a3
    a4 = _acc(v + h * k3v, k); k4p, k4v = v + h * k3v, a4
    return p + h / 6 * (k1p + 2 * k2p + 2 * k3p + k4p), v + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)


def fit_flight(t, P, k=None):
    """3D 점에 (p0, v0[, k]) 맞춤. k=None 이면 k 도 추정."""
    from scipy.optimize import least_squares
    tt = t - t[0]
    A = np.column_stack([np.ones_like(tt), tt])
    lin = np.linalg.lstsq(A, P - 0.5 * np.outer(tt ** 2, G), rcond=None)[0]
    x0 = np.r_[lin[0], lin[1], 0.015 if k is None else []]

    def res(x):
        kk = x[6] if k is None else k
        return (simulate(x[:3], x[3:6], kk, tt) - P).ravel()

    r = least_squares(res, x0, loss="soft_l1", f_scale=0.02)
    kk = r.x[6] if k is None else k
    rms = float(np.sqrt(np.mean(np.sum((simulate(r.x[:3], r.x[3:6], kk, tt) - P) ** 2, axis=1))))
    return r.x[:3], r.x[3:6], float(kk), rms


def plane_cross(p0, v0, k, plane_p, plane_n, t_max=2.0):
    """궤적이 평면을 (n 의 반대 방향으로 들어오며) 지나는 시각·점."""
    p, v, tc, dt = np.array(p0, float), np.array(v0, float), 0.0, 0.001
    s_prev = np.dot(p - plane_p, plane_n)
    while tc < t_max:
        p2, v2 = _rk4(p, v, k, dt)
        s = np.dot(p2 - plane_p, plane_n)
        if s_prev > 0 >= s:
            a = s_prev / (s_prev - s)
            return tc + a * dt, p + a * (p2 - p)
        p, v, tc, s_prev = p2, v2, tc + dt, s
    return None


def flights(t, P, min_pts=10, max_gap=0.05, min_speed=1.5):
    """3D 시계열에서 비행 구간 (start, end) 인덱스. 속도·연속성으로."""
    ok = np.isfinite(P[:, 0])
    idx = np.where(ok)[0]
    segs, cur = [], [idx[0]] if len(idx) else []
    for a_, b_ in zip(idx[:-1], idx[1:]):
        if t[b_] - t[a_] <= max_gap and np.linalg.norm(P[b_] - P[a_]) / (t[b_] - t[a_]) > min_speed:
            cur.append(b_)
        else:
            if len(cur) >= min_pts:
                segs.append((cur[0], cur[-1]))
            cur = [b_]
    if len(cur) >= min_pts:
        segs.append((cur[0], cur[-1]))
    return segs


def cmd_traj(a) -> None:
    cal = json.loads((a.out / "side_calib.json").read_text())
    Rs, ts_ = np.array(cal["R_side_base"]), np.array(cal["t_side_base"])
    plane_p, plane_n = np.array(a.plane_p), np.array(a.plane_n) / np.linalg.norm(a.plane_n)
    table = []
    for bag in a.bags:
        bag = Path(bag).name
        det, cams, T_bc, spread = load_det(a.out, bag)
        T_cb = np.linalg.inv(T_bc)
        Pw = proj_mat(cams["wrist"].K, T_cb[:3, :3], T_cb[:3, 3])
        Ps = proj_mat(cams["side"].K, Rs, ts_)
        t_side = det["side"]["t"] + a.dt          # 측면 시각 보정 (측면 + dt = 손목 시계)
        uvs = det["side"]["uv"]
        uvw = interp_uv(det["wrist"]["t"], det["wrist"]["uv"], t_side)
        ok = np.isfinite(uvs[:, 0]) & np.isfinite(uvw[:, 0])
        P = np.full((len(t_side), 3), np.nan)
        if ok.any():
            P[ok] = triangulate(Pw, Ps, uvw[ok], uvs[ok])
        # 재투영 오차로 이상치 제거
        for i in np.where(ok)[0]:
            ew = np.linalg.norm(cv2.projectPoints(P[i:i + 1], cv2.Rodrigues(T_cb[:3, :3])[0], T_cb[:3, 3],
                                                  cams["wrist"].K, None)[0].ravel() - uvw[i])
            es = np.linalg.norm(cv2.projectPoints(P[i:i + 1], cv2.Rodrigues(Rs)[0], ts_, cams["side"].K, None)[0].ravel() - uvs[i])
            if max(ew, es) > a.max_reproj:
                P[i] = np.nan
        np.save(a.out / f"{bag}_P.npy", np.column_stack([t_side, P]))
        for n, (s, e) in enumerate(flights(t_side, P)):
            sel = np.arange(s, e + 1)
            sel = sel[np.isfinite(P[sel, 0])]
            t, X = t_side[sel], P[sel]
            vx = (X[-1, 0] - X[0, 0]) / (t[-1] - t[0])
            kind = "throw" if vx < 0 else "return"
            p0, v0, k, rms = fit_flight(t, X)
            cr = plane_cross(p0, v0, k, plane_p, plane_n) if kind == "throw" else None
            row = {"bag": bag, "flight": n, "kind": kind, "t0": round(t[0], 4), "n_pts": len(t),
                   "dur_s": round(t[-1] - t[0], 3), "speed0": round(float(np.linalg.norm(v0)), 2),
                   "k": round(k, 4), "rms_mm": round(rms * 1000, 1),
                   "apex_z": round(float(X[:, 2].max()), 3),
                   "x0": round(float(X[0, 0]), 3), "y0": round(float(X[0, 1]), 3), "z0": round(float(X[0, 2]), 3),
                   "cross_t": "", "cross_x": "", "cross_y": "", "cross_z": ""}
            if cr:
                row.update(cross_t=round(cr[0], 3), cross_x=round(float(cr[1][0]), 3),
                           cross_y=round(float(cr[1][1]), 3), cross_z=round(float(cr[1][2]), 3))
            table.append(row)
        print(f"[traj] {bag}: 3D 점 {np.isfinite(P[:, 0]).sum()}개, 비행 {sum(1 for r in table if r['bag'] == bag)}개 "
              f"(투구 {sum(1 for r in table if r['bag'] == bag and r['kind'] == 'throw')})", flush=True)
    out = a.out / "throws.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    print(f"[traj] → {out}")


def side_window(sd_t, sd_uv, ta, tb):
    """측면 카메라가 본 같은 비행 전체 창 (연속 검출 틈 ≤ 0.04 s, 프레임당 3 px 넘게 이동)."""
    okd = np.where(np.isfinite(sd_uv[:, 0]))[0]
    if not len(okd):
        return ta, tb
    j = okd[np.argmin(np.abs(sd_t[okd] - 0.5 * (ta + tb)))]
    pos = int(np.where(okd == j)[0][0]); lo = hi = pos
    while lo > 0 and sd_t[okd[lo]] - sd_t[okd[lo - 1]] <= 0.04 and np.linalg.norm(sd_uv[okd[lo]] - sd_uv[okd[lo - 1]]) > 3.0:
        lo -= 1
    while hi < len(okd) - 1 and sd_t[okd[hi + 1]] - sd_t[okd[hi]] <= 0.04 and np.linalg.norm(sd_uv[okd[hi + 1]] - sd_uv[okd[hi]]) > 3.0:
        hi += 1
    return min(ta, sd_t[okd[lo]]), max(tb, sd_t[okd[hi]])


def cmd_fit(a) -> None:
    """투구마다 두 카메라 픽셀 전체로 (p0, v0) 맞춤 = '정답 궤적'. k 는 calib3 값(전역) + 투구별 자유 k 도 같이."""
    from scipy.optimize import least_squares
    cal = json.loads((a.out / "side_calib.json").read_text())
    Rs, tsv, dt, kg = np.array(cal["R_side_base"]), np.array(cal["t_side_base"]), cal["dt_side_to_wrist_s"], cal["k_drag"]
    rs = cv2.Rodrigues(Rs)[0].ravel()
    plane_p, plane_n = np.array(a.plane_p), np.array(a.plane_n) / np.linalg.norm(a.plane_n)
    rows, store = [], {}
    for bag in a.bags:
        bag = Path(bag).name
        det, cams, T_bc, _ = load_det(a.out, bag)
        T_cb = np.linalg.inv(T_bc); rw = cv2.Rodrigues(T_cb[:3, :3])[0].ravel()
        A = np.load(a.out / f"{bag}_P.npy"); tP, P = A[:, 0], A[:, 1:]
        w, sd = det["wrist"], det["side"]
        for n, (s_, e_) in enumerate(flights(tP, P)):
            sel = np.arange(s_, e_ + 1); sel = sel[np.isfinite(P[sel, 0])]
            if len(sel) < 8 or P[sel[0], 0] - P[sel[-1], 0] < 0.3:      # 로봇 쪽(−x)으로 오는 투구만
                continue
            ta, tb = side_window(sd["t"] + dt, sd["uv"], tP[sel][0] - 0.03, tP[sel][-1] + 0.03)
            if tb - ta < a.min_dur:
                continue
            mw = (w["t"] >= ta) & (w["t"] <= tb) & np.isfinite(w["uv"][:, 0])
            ms = (sd["t"] + dt >= ta) & (sd["t"] + dt <= tb) & np.isfinite(sd["uv"][:, 0])
            p0, v0, _, _ = fit_flight(tP[sel], P[sel], k=kg)
            p0 = simulate(p0, v0, kg, np.array([0.0]))[0]
            # 기준 시각을 창 시작(ta)으로: tP[sel][0] 에서 ta 까지 거꾸로 적분 대신 맞춤에서 같이 풀린다
            tref = tP[sel][0]

            def resid(x, kk):
                Pw_ = sim_back(x[:3], x[3:6], kk, w["t"][mw] - tref)
                Ps_ = sim_back(x[:3], x[3:6], kk, sd["t"][ms] + dt - tref)
                uw = cv2.projectPoints(Pw_, rw, T_cb[:3, 3], cams["wrist"].K, None)[0].reshape(-1, 2) - w["uv"][mw]
                us = cv2.projectPoints(Ps_, rs, tsv, cams["side"].K, None)[0].reshape(-1, 2) - sd["uv"][ms]
                return np.r_[uw.ravel(), us.ravel()]

            r1 = least_squares(lambda x: resid(x, kg), np.r_[p0, v0], loss="soft_l1", f_scale=2.0)
            r2 = least_squares(lambda x: resid(x[:6], x[6]), np.r_[r1.x, kg], loss="soft_l1", f_scale=2.0,
                               bounds=([-np.inf] * 6 + [0.0], [np.inf] * 6 + [0.5]))
            e1 = np.linalg.norm(resid(r1.x, kg).reshape(-1, 2), axis=1)
            # 릴리스 = 창 시작. 그 시각의 상태
            x_rel = sim_back(r1.x[:3], r1.x[3:6], kg, np.array([ta - tref]), return_v=True)
            p_rel, v_rel = x_rel[0][0], x_rel[1][0]
            cr = plane_cross(p_rel, v_rel, kg, plane_p, plane_n)
            ts_full = np.linspace(0, (tb - ta) + 0.4, 200)
            apex = simulate(p_rel, v_rel, kg, ts_full)[:, 2].max()
            key = f"{bag}#{n}"
            store[key] = dict(bag=bag, n=n, t_release=ta, t_end=tb, p_rel=p_rel, v_rel=v_rel, k=kg)
            row = dict(key=key, bag=bag, flight=n, t_release=round(ta, 4), window_s=round(tb - ta, 3),
                       n_wrist=int(mw.sum()), n_side=int(ms.sum()), reproj_px_med=round(float(np.median(e1)), 2),
                       speed0=round(float(np.linalg.norm(v_rel)), 2),
                       launch_deg=round(float(np.degrees(np.arctan2(v_rel[2], np.linalg.norm(v_rel[:2])))), 1),
                       x0=round(float(p_rel[0]), 3), y0=round(float(p_rel[1]), 3), z0=round(float(p_rel[2]), 3),
                       apex_z=round(float(apex), 3), k_free=round(float(r2.x[6]), 4),
                       cross_dt=round(cr[0], 3) if cr else "", cross_x=round(float(cr[1][0]), 3) if cr else "",
                       cross_y=round(float(cr[1][1]), 3) if cr else "", cross_z=round(float(cr[1][2]), 3) if cr else "")
            rows.append(row)
    # 목표 안: 교차점 평균(= 실제 노린 곳)에서 15 cm 안 + 고정 목표(plane_p)에서 15 cm 안
    C = np.array([[r["cross_x"], r["cross_y"], r["cross_z"]] for r in rows if r["cross_x"] != ""], float)
    mean = C.mean(axis=0) if len(C) else np.zeros(3)
    for r in rows:
        if r["cross_x"] == "":
            r.update(err_mean_cm="", in_mean15="", err_tcp_cm="", in_tcp15="")
            continue
        c = np.array([r["cross_x"], r["cross_y"], r["cross_z"]], float)
        r.update(err_mean_cm=round(float(np.linalg.norm(c - mean)) * 100, 1), in_mean15=bool(np.linalg.norm(c - mean) <= 0.15),
                 err_tcp_cm=round(float(np.linalg.norm(c - plane_p)) * 100, 1), in_tcp15=bool(np.linalg.norm(c - plane_p) <= 0.15))
    with open(a.out / "throws_fit.csv", "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wri.writeheader(); wri.writerows(rows)
    np.save(a.out / "throws_fit_store.npy", store, allow_pickle=True)
    kf = np.array([r["k_free"] for r in rows], float)
    print(f"[fit] 투구 {len(rows)}개 · 교차 {len(C)}개 · 재투영 중앙 {np.median([r['reproj_px_med'] for r in rows]):.2f} px · "
          f"k 투구별 중앙 {np.median(kf):.4f} (IQR {np.percentile(kf, 25):.4f}~{np.percentile(kf, 75):.4f}) /m · "
          f"교차점 평균 {np.round(mean, 3)}")


def sim_back(p0, v0, k, tt, return_v=False, h=0.002):
    """기준 시각 기준 tt(음수 허용) 위치 — 앞뒤로 적분."""
    tt = np.asarray(tt, float)
    outp = np.zeros((len(tt), 3)); outv = np.zeros((len(tt), 3))
    for sign in (1, -1):
        m = (tt >= 0) if sign > 0 else (tt < 0)
        if not m.any():
            continue
        q = np.abs(tt[m]); order = np.argsort(q)
        p, v, tc = np.array(p0, float), np.array(v0, float), 0.0
        res_p = np.zeros((m.sum(), 3)); res_v = np.zeros((m.sum(), 3))
        for idx in order:
            while tc + h <= q[idx]:
                p, v = _rk4_dir(p, v, k, h, sign); tc += h
            r = q[idx] - tc
            pp, vv = _rk4_dir(p, v, k, r, sign) if r > 1e-9 else (p, v)
            res_p[idx], res_v[idx] = pp, vv
        outp[m], outv[m] = res_p, res_v
    return (outp, outv) if return_v else outp


def _rk4_dir(p, v, k, h, sign):
    if sign > 0:
        return _rk4(p, v, k, h)
    # 시간 거꾸로: dp/ds = -v, dv/ds = -a(v)
    def f(pp, vv):
        return -vv, -_acc(vv, k)
    a1p, a1v = f(p, v); a2p, a2v = f(p + 0.5 * h * a1p, v + 0.5 * h * a1v)
    a3p, a3v = f(p + 0.5 * h * a2p, v + 0.5 * h * a2v); a4p, a4v = f(p + h * a3p, v + h * a3v)
    return p + h / 6 * (a1p + 2 * a2p + 2 * a3p + a4p), v + h / 6 * (a1v + 2 * a2v + 2 * a3v + a4v)


def cmd_predict(a) -> None:
    """예측기 평가: 투구마다 삼각측량 3D 점을 시간 순으로 FlightPredictor 에 넣고, 릴리스 후 시각별 요격점 오차."""
    sys.path.insert(0, str(REPO / "ros2_ws/src/rally_vision"))
    from rally_vision.flight_predictor import FlightPredictor
    store = np.load(a.out / "throws_fit_store.npy", allow_pickle=True).item()
    cal = json.loads((a.out / "side_calib.json").read_text())
    kg = cal["k_drag"]
    plane_p, plane_n = np.array(a.plane_p), np.array(a.plane_n) / np.linalg.norm(a.plane_n)
    grid = np.array([0.08, 0.10, 0.12, 0.16, 0.20, 0.25, 0.30, 0.40])
    recs = []
    by_bag = {}
    for key, s in store.items():
        by_bag.setdefault(s["bag"], []).append(s)
    Rs_, tsv_, dts_ = np.array(cal["R_side_base"]), np.array(cal["t_side_base"]), cal["dt_side_to_wrist_s"]
    rs_ = cv2.Rodrigues(Rs_)[0].ravel()
    for bag, items in by_bag.items():
        A = np.load(a.out / f"{bag}_P.npy"); tP, P = A[:, 0], A[:, 1:]
        det_, cams_, _, _ = load_det(a.out, bag)
        st_, su_ = det_["side"]["t"] + dts_, det_["side"]["uv"]
        Ks_ = cams_["side"].K
        for s in items:
            truth = plane_cross(s["p_rel"], s["v_rel"], kg, plane_p, plane_n)
            if truth is None:
                continue
            t_true = s["t_release"] + truth[0]; p_true = truth[1]
            for model, kk in (("ekf_drag", kg), ("ekf_nodrag", 0.0), ("ekf_k015", 0.015)):
                fp = FlightPredictor(k=kk, meas_std_m=a.meas_std)
                m = (tP >= s["t_release"]) & (tP <= s["t_end"]) & np.isfinite(P[:, 0])
                hist = []
                for t, p in zip(tP[m], P[m]):
                    fp.update(t, p)
                    c = fp.predict_crossing(plane_p, plane_n) if fp.ready else None
                    if c is not None:
                        # 모델과 무관한 정답: 예측 교차 시각에 측면 카메라가 실제로 찍은 공 픽셀
                        uo = interp_uv(st_, su_, np.array([c.t]), max_gap=0.04)[0]
                        up = cv2.projectPoints(c.point.reshape(1, 3), rs_, tsv_, Ks_, None)[0].ravel()
                        dist = np.linalg.norm(Rs_ @ c.point + tsv_)
                        e_obs = np.linalg.norm(up - uo) * dist / Ks_[0, 0] if np.isfinite(uo[0]) else np.nan
                        hist.append((t - s["t_release"], np.linalg.norm(c.point - p_true), abs(c.t - t_true),
                                     float(np.max(c.std_m)), e_obs))
                if not hist:
                    continue
                H = np.array(hist)
                for g in grid:
                    i = np.searchsorted(H[:, 0], g, side="right") - 1
                    if i >= 0:
                        recs.append(dict(key=f"{bag}#{s['n']}", model=model, t_since=g, err_cm=H[i, 1] * 100,
                                         t_err_ms=H[i, 2] * 1000, std_cm=H[i, 3] * 100, flight_s=truth[0],
                                         err_obs_cm=H[i, 4] * 100))
    with open(a.out / "predict_eval.csv", "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(recs[0].keys())); wri.writeheader(); wri.writerows(recs)
    print(f"[predict] 투구 {len(store)}개 · 평면 교차까지 비행 중앙 {np.median([r['flight_s'] for r in recs]):.2f} s")
    for model in ("ekf_drag", "ekf_k015", "ekf_nodrag"):
        line = []
        for g in grid:
            e = [r["err_cm"] for r in recs if r["model"] == model and abs(r["t_since"] - g) < 1e-9]
            eo = [r["err_obs_cm"] for r in recs if r["model"] == model and abs(r["t_since"] - g) < 1e-9 and np.isfinite(r["err_obs_cm"])]
            if e:
                line.append(f"{g:.2f}s 정답궤적 {np.median(e):.1f}/{np.percentile(e, 90):.1f} · 관측 {np.median(eo):.1f}/{np.percentile(eo, 90):.1f}cm(n{len(eo)})")
        print(f"[predict] {model}: " + " · ".join(line))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    sp = ap.add_subparsers(dest="cmd", required=True)
    d = sp.add_parser("detect"); d.add_argument("bags", nargs="+"); d.add_argument("--model", required=True)
    d.add_argument("--imgsz", type=int, default=864); d.add_argument("--conf", type=float, default=0.25)
    d.add_argument("--static-px", type=int, default=24, help="정지 오검출 격자 칸 [px]")
    d.add_argument("--static-frac", type=float, default=0.25, help="한 칸에 이 비율 넘는 프레임에서 검출되면 정지 물체로 제외 (1 이상이면 끔)")
    c = sp.add_parser("calib"); c.add_argument("bag"); c.add_argument("--sep", type=float, default=0.03)
    c2 = sp.add_parser("calib2"); c2.add_argument("bags", nargs="+")
    c2.add_argument("--dt-min", type=float, default=-0.04); c2.add_argument("--dt-max", type=float, default=0.04)
    c2.add_argument("--dt-step", type=float, default=0.005)
    c3 = sp.add_parser("calib3"); c3.add_argument("bags", nargs="+")
    c3.add_argument("--min-pts", type=int, default=12); c3.add_argument("--f-scale", type=float, default=2.0)
    c3.add_argument("--fix-dt", type=float, default=None); c3.add_argument("--max-flights", type=int, default=0)
    c3.add_argument("--dry", action="store_true", help="side_calib.json 덮어쓰지 않음")
    c3.add_argument("--static", nargs="*", default=[], help="calib 가 저장한 *_static.npz (손목 깊이 정지점)")
    c3.add_argument("--w-static", type=float, default=1.0)
    c3.add_argument("--fix-k", type=float, default=None)
    c3.add_argument("--full-side", action="store_true", help="측면 카메라가 본 비행 전체를 창으로")
    c3.add_argument("--min-dur", type=float, default=0.0)
    t = sp.add_parser("traj"); t.add_argument("bags", nargs="+")
    t.add_argument("--dt", type=float, default=0.0, help="측면 시각 보정 [s]")
    t.add_argument("--max-reproj", type=float, default=6.0)
    t.add_argument("--plane-p", type=float, nargs=3, default=[0.336, -0.222, 0.481], help="패드 평면 점 (기본 대기 자세 TCP, 가정)")
    t.add_argument("--plane-n", type=float, nargs=3, default=[0.7071, 0.0, 0.7071], help="패드 평면 법선 (투구자 쪽 45° 위, 가정)")
    fp_ = sp.add_parser("fit"); fp_.add_argument("bags", nargs="+"); fp_.add_argument("--min-dur", type=float, default=0.3)
    pr = sp.add_parser("predict"); pr.add_argument("--meas-std", type=float, default=0.015)
    for p_ in (fp_, pr):
        p_.add_argument("--plane-p", type=float, nargs=3, default=[0.336, -0.222, 0.481])
        p_.add_argument("--plane-n", type=float, nargs=3, default=[0.7071, 0.0, 0.7071])
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    {"detect": cmd_detect, "calib": cmd_calib, "calib2": cmd_calib2, "calib3": cmd_calib3, "traj": cmd_traj, "fit": cmd_fit, "predict": cmd_predict}[a.cmd](a)


if __name__ == "__main__":
    main()
