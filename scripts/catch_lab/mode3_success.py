#!/usr/bin/env python3
"""모드 3 성공률 상한 재계산 — 실측 흩어짐(throws_fit.csv) + 예측기 오차(predict_eval.csv) + 로봇 실측 도달.
담당: 박성현

    python3 mode3_success.py --out <traj 폴더>

모델: 대기 위치 = 실측 교차점 평균. 예측 시각 t_pred 에 예측점(실제 + 예측 오차, 방향 무작위)으로 출발,
출발 지연 t0 뒤 정지→정지 이동 거리 a_eff·(t − t0)²/4 (로봇 실측 2점으로 a_eff 맞춤) 안에서만 이동,
대기 자세 이동 한계(왼쪽 +y 15 cm = 어깨 특이점, 오른쪽 20 cm, 패드 면 위·아래 20 cm) 적용.
성공 = 실제 교차점과 패드 중심 거리 ≤ 7.5 cm (벨크로 부착·검출 실패·힘 STOP 은 미포함 = 상한).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROBOTS = {  # 10/10 실측 (mode3_reach_1010_*.csv): 출발 지연 t0, 20 cm 도착 시각 → a_eff = 4·0.20/(t − t0)²
    "acc1000": (0.148, 4 * 0.20 / (0.848 - 0.148) ** 2),
    "acc2000": (0.115, 4 * 0.20 / (0.610 - 0.115) ** 2),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pad-r", type=float, default=0.075)
    ap.add_argument("--plane-n", type=float, nargs=3, default=[0.7071, 0.0, 0.7071])
    a = ap.parse_args()
    R = [r for r in csv.DictReader(open(a.out / "throws_fit.csv")) if r["cross_x"] != ""]
    C = np.array([[float(r["cross_x"]), float(r["cross_y"]), float(r["cross_z"])] for r in R])
    Tf = np.array([float(r["cross_dt"]) for r in R])
    n = np.array(a.plane_n) / np.linalg.norm(a.plane_n)
    u = np.cross(n, [0, 0, 1]); u /= np.linalg.norm(u); w = np.cross(n, u)
    D = C - C.mean(0); du0, dw0 = D @ u, D @ w
    P = list(csv.DictReader(open(a.out / "predict_eval.csv")))

    def perr(t):
        v = np.array([float(r["err_obs_cm"]) for r in P if r["model"] == "ekf_drag" and abs(float(r["t_since"]) - t) < 1e-9])
        return v[np.isfinite(v)] / 100

    rng = np.random.default_rng(0)
    out = {"n_throws": len(C), "mean_cross": C.mean(0).tolist(), "sigma_lateral_m": float(du0.std()),
           "sigma_inplane_m": float(dw0.std()), "radial_median_m": float(np.median(np.hypot(du0, dw0))),
           "radial_p90_m": float(np.percentile(np.hypot(du0, dw0), 90)), "flight_to_plane_median_s": float(np.median(Tf)),
           "robots": {k: {"t0": v[0], "a_eff": v[1]} for k, v in ROBOTS.items()}, "rates": {}}
    for scale, label in ((1.0, "measured"), (0.5, "half_dispersion")):
        du, dw = du0 * scale, dw0 * scale
        out["rates"][f"{label}|static"] = float((np.hypot(du, dw) <= a.pad_r).mean())
        for tp in (0.16, 0.20):
            pe = perr(tp)
            for name, (t0, aeff) in ROBOTS.items():
                N = 40000
                idx = rng.integers(0, len(C), N); e = rng.choice(pe, N); ang = rng.uniform(0, 2 * np.pi, N)
                tu, tw = du[idx], dw[idx]
                reach = aeff * np.maximum(Tf[idx] - tp - t0, 0) ** 2 / 4
                gu, gw = tu + e * np.cos(ang), tw + e * np.sin(ang)
                s = np.minimum(1, reach / np.maximum(np.hypot(gu, gw), 1e-9))
                mu = np.clip(gu * s, -0.15, 0.20)    # u = 오른쪽(−y). 왼쪽 15 cm(특이점), 오른쪽 20 cm
                mw = np.clip(gw * s, -0.20, 0.20)
                out["rates"][f"{label}|pred{tp:.2f}|{name}"] = float((np.hypot(tu - mu, tw - mw) <= a.pad_r).mean())
    (a.out / "success_recalc.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for k, v in out["rates"].items():
        print(f"{k:32s} {v * 100:5.1f} %")


if __name__ == "__main__":
    main()
