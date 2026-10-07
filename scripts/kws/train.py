#!/usr/bin/env python3
"""비상어("그만"·"멈춰") + 호출어("랠리야") 키워드 판별기 학습·평가 (소규모 시험용).

구조: openWakeWord 의 음성 특징 추출기(멜 스펙트로그램 → 96 차원 임베딩, 80 ms 마다)는 그대로 쓰고,
      마지막 16 프레임(1.28 s)을 보고 "그만"/"멈춰"/"랠리야"/아님 을 고르는 작은 판별기만 우리 녹음으로 학습한다.
      출력은 둘로 묶어 쓴다: 비상 = P(그만)+P(멈춰) → 정지, 호출 = P(랠리야) → 녹음·STT 시작.
      랠리야 녹음이 8개 미만이면 비상어만 학습한다.
      openWakeWord 자체 학습도 같은 구조 (특징 추출기는 고정, 끝단만 학습).
평가:
  - 키워드 인식률: 일부를 떼어 두고(학습에 안 씀) 맞히는 비율. "랠리야" 가 정지로, "그만" 이 호출로 잘못 잡히는지도
  - 오탐: 주변 소리(ambient)를 80 ms 씩 밀며 흘려 봤을 때 비상·호출 각각 시간당 몇 번 울리나 (학습에 안 쓴 구간)
  - 지연: 비상어가 끝난 뒤 판별까지 걸리는 시간 (프레임 단위 추정)

사용:
    python3 scripts/kws/train.py                  # 학습 + 평가 + models/kws/emergency_kws.joblib 저장
    python3 scripts/kws/train.py --threshold 0.8
데이터: data/datasets/kws/{geuman,meomchwo,rally,other,ambient}/*.wav (scripts/kws/record.py)
"""
import argparse
import random
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "datasets" / "kws"
OUT = ROOT / "models" / "kws"
RATE = 16000
WIN = 16                      # 판별에 쓰는 임베딩 프레임 수 (16 × 80 ms = 1.28 s)
CLIP = int(2.0 * RATE)        # 특징을 뽑을 때 쓰는 길이 (앞에 0.5 s 여유를 붙여 1.28 s 창이 꽉 차게)
KEYWORDS = ["geuman", "meomchwo", "rally"]   # 라벨 번호 1, 2, 3 (0 = 아님)
EMERGENCY = ("geuman", "meomchwo")
WAKE = ("rally",)


def read_wav(p: Path) -> np.ndarray:
    with wave.open(str(p)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)


def fit_len(x: np.ndarray, n: int, offset: int = 0) -> np.ndarray:
    """길이 n 으로 맞춤. offset 만큼 뒤로 밀고(앞을 0 으로), 넘치면 앞을 자름."""
    y = np.concatenate([np.zeros(max(0, offset), np.float32), x])
    if len(y) >= n:
        return y[-n:]
    return np.concatenate([np.zeros(n - len(y), np.float32), y])


def augment(x: np.ndarray, noises: list[np.ndarray], rng: random.Random) -> np.ndarray:
    """위치 이동 · 크기 · 주변 소음 섞기 (SNR 5~25 dB). 실제 환경 흔들림을 흉내."""
    y = fit_len(x, CLIP, offset=rng.randint(0, int(0.35 * RATE)))
    y = y * rng.uniform(0.5, 1.5)
    if noises and rng.random() < 0.8:
        n = rng.choice(noises)
        s = rng.randint(0, max(0, len(n) - CLIP))
        n = n[s:s + CLIP]
        if len(n) == CLIP:
            ps, pn = np.mean(y ** 2) + 1e-6, np.mean(n ** 2) + 1e-6
            snr = rng.uniform(5, 25)
            y = y + n * np.sqrt(ps / (pn * 10 ** (snr / 10)))
    return np.clip(y, -32768, 32767)


def embed(feat, clips: list[np.ndarray]) -> np.ndarray:
    """(N, CLIP) → (N, WIN*96). 마지막 WIN 프레임."""
    X = np.stack(clips).astype(np.int16)
    e = feat.embed_clips(X, batch_size=64)              # (N, frames, 96)
    return e[:, -WIN:, :].reshape(len(clips), -1)


def group_score(proba: np.ndarray, labels: list[str], group: tuple[str, ...]) -> np.ndarray:
    """(N, 클래스) 확률 → 묶음(비상·호출) 확률 합. 묶음 라벨이 학습에 없으면 0."""
    idx = [labels.index(k) for k in group if k in labels]
    return proba[:, idx].sum(axis=1) if idx else np.zeros(len(proba))


def stream_proba(feat, clf, audio: np.ndarray) -> np.ndarray:
    """긴 소리를 80 ms(1280 샘플)씩 흘려 넣으며 매 프레임 클래스 확률 (프레임, 클래스)."""
    feat.reset()
    out = []
    for i in range(0, len(audio) - 1280, 1280):
        feat(audio[i:i + 1280].astype(np.int16))
        f = feat.get_features(WIN)                       # (1, WIN, 96)
        out.append(clf.predict_proba(f.reshape(1, -1))[0])
    return np.array(out)


def count_hits(score: np.ndarray, threshold: float) -> int:
    """연속 프레임 감지는 한 번으로, 1 s 안 재발은 무시."""
    hits, last = 0, -99
    for i, s in enumerate(score):
        if s >= threshold and i - last > 12:
            hits += 1
            last = i
    return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--aug", type=int, default=15, help="키워드 녹음 하나당 만드는 변형 수")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    from openwakeword.utils import AudioFeatures
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    files = {k: sorted((DATA / k).glob("*.wav")) for k in KEYWORDS + ["other", "ambient"]}
    print("녹음:", {k: len(v) for k, v in files.items()})
    if min(len(files["geuman"]), len(files["meomchwo"])) < 8 or not files["ambient"]:
        raise SystemExit("녹음이 모자람 (그만·멈춰 각 8개 이상 + 주변 소리 1개 이상). scripts/kws/record.py")
    kws = [k for k in KEYWORDS if len(files[k]) >= 8]
    if "rally" not in kws:
        print("랠리야 녹음 8개 미만 → 비상어만 학습")
    labels = ["none"] + kws

    # 사람·순서 기준으로 20 % 를 평가용으로 떼어 둠
    def split(fs):
        fs = fs[:]
        rng.shuffle(fs)
        k = max(2, len(fs) // 5)
        return fs[k:], fs[:k]
    tr, te = {}, {}
    for k in kws + ["other"]:
        tr[k], te[k] = split(files[k])
    amb = [read_wav(p) for p in files["ambient"]]
    amb_all = np.concatenate(amb)
    cut = int(len(amb_all) * 0.7)                       # 주변 소리 앞 70 % 학습(소음·부정 예시), 뒤 30 % 오탐 평가
    amb_tr, amb_te = amb_all[:cut], amb_all[cut:]
    noises = [amb_tr]

    feat = AudioFeatures()
    X, y = [], []
    for li, k in enumerate(kws, start=1):
        for p in tr[k]:
            x = read_wav(p)
            for _ in range(a.aug):
                X.append(augment(x, noises, rng))
                y.append(li)
    for p in tr["other"]:
        x = read_wav(p)
        for _ in range(max(3, a.aug // 3)):
            X.append(augment(x, noises, rng))
            y.append(0)
    n_amb = len(X)                                       # 주변 소리 조각도 '아님' 으로 같은 수만큼
    for _ in range(n_amb // 2):
        s = rng.randint(0, len(amb_tr) - CLIP)
        X.append(amb_tr[s:s + CLIP] * rng.uniform(0.5, 1.5))
        y.append(0)
    print(f"학습 예시 {len(X)}개 (키워드 {sum(1 for v in y if v)} · 아님 {sum(1 for v in y if not v)}) → 특징 추출 중")
    Xe = embed(feat, X)
    clf = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, max_iter=300,
                                                       early_stopping=True, random_state=a.seed))
    clf.fit(Xe, np.array(y))

    # 1) 떼어 둔 녹음으로 인식률. 각 녹음이 비상·호출 어느 쪽으로 잡히는지 둘 다 센다
    print(f"\n[떼어 둔 녹음]  임계 {a.threshold}    비상 감지 / 호출 감지")
    for k in kws + ["other"]:
        Xt = embed(feat, [fit_len(read_wav(p), CLIP, offset=int(0.2 * RATE)) for p in te[k]])
        p = clf.predict_proba(Xt)
        em = group_score(p, labels, EMERGENCY) >= a.threshold
        wk = group_score(p, labels, WAKE) >= a.threshold
        want = "비상" if k in EMERGENCY else "호출" if k in WAKE else "없음"
        print(f"  {k:9s} {int(em.sum()):2d} / {int(wk.sum()):2d}   (총 {len(p)}, 맞는 쪽: {want})")

    # 2) 주변 소리 흘려서 오탐
    pr = stream_proba(feat, clf, amb_te)
    hours = len(amb_te) / RATE / 3600
    print(f"\n[주변 소리 {len(amb_te) / RATE:.0f} s] 잘못 감지 (목표: 10분에 1회 이하 = 시간당 6회)")
    for name, group in [("비상", EMERGENCY), ("호출", WAKE)]:
        if any(k in kws for k in group):
            hits = count_hits(group_score(pr, labels, group), a.threshold)
            print(f"  {name} {hits}회 → 시간당 {hits / hours:.0f}회")

    # 3) 지연: 떼어 둔 키워드를 주변 소리 사이에 넣고 흘려 봄 → 녹음 끝에서 몇 프레임 뒤에 넘는지
    delays = []
    for k in ["geuman", "meomchwo"]:
        for p in te[k]:
            x = read_wav(p)
            s = rng.randint(0, len(amb_te) - 3 * RATE)
            bg = amb_te[s:s + 3 * RATE] * 0.3
            audio = np.concatenate([bg[:RATE], x, bg[RATE:2 * RATE]])
            sc = group_score(stream_proba(feat, clf, audio), labels, EMERGENCY)
            end_frame = (RATE + len(x)) // 1280
            over = np.where(sc >= a.threshold)[0]
            if len(over):
                delays.append((over[0] - end_frame) * 0.08)
    if delays:
        print(f"\n[비상어 지연] 녹음 클립 끝 기준 {np.median(delays):+.2f} s (중간값, 음수 = 클립 안에서 이미 감지). "
              f"감지 {len(delays)}/{len(te['geuman']) + len(te['meomchwo'])}")

    OUT.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump({"clf": clf, "win": WIN, "labels": labels, "emergency": list(EMERGENCY), "wake": list(WAKE),
                 "threshold": a.threshold}, OUT / "emergency_kws.joblib")
    print(f"\n저장: {OUT / 'emergency_kws.joblib'}")


if __name__ == "__main__":
    main()
