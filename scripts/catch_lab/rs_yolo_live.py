#!/usr/bin/env python3
"""RealSense 컬러 영상에 YOLO(벨크로 공) 검출을 겹쳐 실시간으로 보여주는 확인용 스크립트 (로봇 없음, ROS 없음).

    ~/venvs/mode3/bin/python scripts/catch_lab/rs_yolo_live.py                 # 창에 표시 (q 종료, s 스냅샷)
    ~/venvs/mode3/bin/python scripts/catch_lab/rs_yolo_live.py --headless --seconds 10   # 창 없이 10초, 스냅샷만

RealSense 가 없으면 --webcam 0 으로 일반 UVC 카메라. 스냅샷은 --out 폴더에 live_*.jpg.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


def open_realsense(serial: str | None, w: int, h: int, fps: int, exposure: int | None, gain: int | None):
    import pyrealsense2 as rs
    ctx = rs.context()
    devs = [(d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.serial_number)) for d in ctx.query_devices()]
    if not devs:
        raise SystemExit("RealSense 를 못 찾음 — lsusb 에 Intel 장치가 있는지, 케이블(USB 3)·포트를 확인")
    print("[rs] 장치:", devs)
    p, c = rs.pipeline(), rs.config()
    if serial:
        c.enable_device(serial)
    c.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
    prof = p.start(c)
    s = prof.get_device().first_color_sensor()
    if exposure is not None:
        s.set_option(rs.option.enable_auto_exposure, 0); s.set_option(rs.option.exposure, exposure)
    if gain is not None:
        s.set_option(rs.option.gain, gain)
    def read():
        f = p.wait_for_frames().get_color_frame()
        return np.asanyarray(f.get_data()), f.get_timestamp()
    return read, p.stop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(Path.home() / "rally/models/velcro_ball_yolo11n-seg_1024_rot_final.pt"))
    ap.add_argument("--serial", default=None); ap.add_argument("--webcam", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=1024); ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--exposure", type=int, default=None, help="RealSense 수동 노출 (실사용 20)"); ap.add_argument("--gain", type=int, default=None)
    ap.add_argument("--headless", action="store_true"); ap.add_argument("--seconds", type=float, default=0)
    ap.add_argument("--out", type=Path, default=Path.home() / "rally_isaac/out/yolo_live")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO
    model = YOLO(a.model)
    if a.webcam is not None:
        cap = cv2.VideoCapture(a.webcam, cv2.CAP_V4L2)
        read = lambda: (cap.read()[1], time.time() * 1000); stop = cap.release
    else:
        read, stop = open_realsense(a.serial, 848, 480, 60, a.exposure, a.gain)
    t0 = time.time(); n = 0; shots = 0; lat = []; det_frames = 0; last_auto = 0
    try:
        while True:
            img, ts = read()
            if img is None:
                break
            t1 = time.perf_counter(); r = model.predict(img, imgsz=a.imgsz, conf=a.conf, verbose=False)[0]; dt = (time.perf_counter() - t1) * 1000
            lat.append(dt); n += 1; det_frames += int(len(r.boxes) > 0)
            vis = r.plot()
            txt = f"{dt:.1f} ms  det {len(r.boxes)}  " + "  ".join(f"{model.names[int(c)]} {float(s):.2f}" for c, s in zip(r.boxes.cls, r.boxes.conf))
            cv2.putText(vis, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            if a.headless:
                if time.time() - last_auto > 2.0:
                    cv2.imwrite(str(a.out / f"live_{shots:02d}.jpg"), vis); shots += 1; last_auto = time.time()
                if a.seconds and time.time() - t0 > a.seconds:
                    break
            else:
                cv2.imshow("RealSense + YOLO (q 종료, s 스냅샷)", vis)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
                if k == ord("s"):
                    cv2.imwrite(str(a.out / f"live_{shots:02d}.jpg"), vis); shots += 1; print("저장", a.out / f"live_{shots-1:02d}.jpg")
    finally:
        stop()
        if not a.headless:
            cv2.destroyAllWindows()
    if n:
        print(f"프레임 {n}, 추론 평균 {np.mean(lat[5:]) if n > 5 else np.mean(lat):.1f} ms, 검출 있는 프레임 {det_frames}/{n}, 스냅샷 {shots}장 → {a.out}")


if __name__ == "__main__":
    main()
