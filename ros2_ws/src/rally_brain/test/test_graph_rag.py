"""지식 그래프 + Graph RAG — 온톨로지 → 그래프 행, 집계, LLM 입력 익명성, 리포트 연결 (DB 없이 가짜 저장소)."""
import json
import re
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from rally_brain import graph_rag as gr
from rally_brain import session_report as r
from rally_interfaces.msg import TrialResult

SID = "11111111-2222-3333-4444-555555555555"
T0 = datetime(2026, 10, 11, 10, 0, tzinfo=timezone.utc)


def row(tid, mode, success, observed=(), notes=""):
    fatigue = ["rally:fatigue"] if observed else []      # 온톨로지: 두 보상 동작 모두 —indicatesFatigue→ fatigue
    return {"trial_id": tid, "mode": mode, "success": success, "notes": notes,
            "observed": [f"rally:{c}" for c in observed], "signals": fatigue}


CURRENT = [row(101, 1, True, ["trunk_lean"]), row(102, 1, False, ["trunk_lean", "shoulder_hike"]),
           row(103, 1, True), row(104, 1, False, ["trunk_lean"], notes="no_hand"), row(105, 5, True)]
SIMILAR = ([row(i, 1, i % 5 != 0, ["trunk_lean"]) for i in range(1, 6)]          # 기울임 5회 중 4회 성공
           + [row(i, 1, True) for i in range(6, 9)] + [row(9, 1, False)])        # 없음 4회 중 3회 성공


def test_seed_rows_from_ontology():
    nodes, edges = gr.seed_rows()
    ids = {n[0]: n for n in nodes}
    assert ids["rally:trunk_lean"][1] == "Compensation" and ids["rally:trunk_lean"][2]       # 한국어 이름표
    assert ids["rally:mode1"][1] == "Mode" and ids["rally:fatigue"][1] == "FatigueSignal"
    assert ("rally:trunk_lean", "indicatesFatigue", "rally:fatigue", None) in edges
    assert any(e[:2] == ("rally:mode1", "trains") for e in edges)
    assert all(e[0] in ids and e[2] in ids for e in edges)                     # 외래 키: 끝점이 모두 노드
    assert not any(n[1] in ("Class", "ObjectProperty", "Ontology") for n in nodes)
    assert not any(n[1] == "Patient" for n in nodes)


def test_recorder_graph_ids_match_ontology():
    """recorder(rally_db.repository.trial_graph)가 쓰는 개념 id 가 온톨로지 노드와 같아야 그래프가 이어진다."""
    pytest.importorskip("redis")
    pytest.importorskip("psycopg")
    from rally_db.repository import COMPENSATION_KEYS, trial_graph
    ids = {n[0] for n in gr.seed_rows()[0]}
    trial = {"session_id": SID, "mode": 1}
    nodes, edges = trial_graph(trial, 7, ["trunk_lean", "bogus"])
    assert {f"rally:{c}" for c in COMPENSATION_KEYS} <= ids
    assert all(n[0] in ids for n in nodes if n[0].startswith("rally:"))
    assert ("trial:7", "observed", "rally:trunk_lean", SID) in edges
    assert not any("bogus" in e[2] for e in edges)
    for mode in (1, 3, 5, 7):
        assert f"rally:mode{mode}" in ids


def test_summarize_counts():
    rows = {g.mode: g for g in gr.summarize(CURRENT, SIMILAR)}
    m1 = rows[1]
    assert m1.this_trials == 3                                     # no_hand 제외
    assert m1.this_comp == {"shoulder_hike": 1, "trunk_lean": 2}
    assert (m1.similar.trials, m1.similar.successes) == (9, 7)
    assert (m1.similar_comp["trunk_lean"].trials, m1.similar_comp["trunk_lean"].successes) == (5, 4)
    assert (m1.similar_clean.trials, m1.similar_clean.successes) == (4, 3)
    assert m1.fatigue_linked == 5
    assert rows[5].similar.trials == 0 and rows[5].similar.rate is None


def test_llm_payload_has_no_ids():
    payload = gr.llm_payload(gr.summarize(CURRENT, SIMILAR))
    text = json.dumps(payload, ensure_ascii=False)
    assert "trial" not in text.replace("trials", "") and "session_id" not in text
    assert "rally:" not in text and "no_hand" not in text and SID[:8] not in text
    assert payload[0]["similar_past"]["with_compensation"]["trunk_lean"] == {"trials": 5, "success_rate": 0.8}
    assert payload[0]["similar_past"]["without_compensation"] == {"trials": 4, "success_rate": 0.75}


def test_markdown_table():
    md = gr.markdown(gr.summarize(CURRENT, SIMILAR), r.MODE_NAMES)
    assert "### 지식 그래프 근거" in md
    assert "| 1 공 굴리기 |" in md and "80% (4/5)" in md and "75% (3/4)" in md
    assert "없음 (기록이 쌓이면 나옵니다)" in md                    # 모드 5: 비슷한 과거 시도 없음
    assert "trunk_lean" not in md                                    # 한국어 이름으로


class FakeRepo:
    def __init__(self, ready=True, arrive_after=0):
        self.ready, self.arrive_after, self.calls, self.seeded = ready, arrive_after, 0, None

    def graph_ready(self):
        return self.ready

    def upsert_graph(self, nodes, edges):
        self.seeded = (len(nodes), len(edges))

    def graph_trials(self, sid, k):
        self.calls += 1
        cur = CURRENT if self.calls > self.arrive_after else CURRENT[:2]
        return cur, SIMILAR


def test_fetch_waits_for_last_trial():
    clock = [0.0]
    repo = FakeRepo(arrive_after=2)
    got = gr.fetch(repo, SID, expected=5, wait_s=3.0, sleep=lambda s: clock.__setitem__(0, clock[0] + s),
                   clock=lambda: clock[0])
    assert repo.seeded and repo.calls == 3 and len(got[0]) == 5


def test_fetch_gives_up_after_wait():
    clock = [0.0]
    repo = FakeRepo(arrive_after=100)
    got = gr.fetch(repo, SID, expected=5, wait_s=1.0, sleep=lambda s: clock.__setitem__(0, clock[0] + s),
                   clock=lambda: clock[0])
    assert len(got[0]) == 2 and repo.calls == 3


def test_fetch_without_tables():
    assert gr.fetch(FakeRepo(ready=False), SID, expected=1) is None


def trial(mode, success, idx):
    return TrialResult(mode=mode, success=success, error_mm=30.0, duration_s=5.0, trial_index=idx, session_id=SID)


def test_report_with_graph():
    calls = []

    def llm(prompt, request, schema):
        calls.append(request)
        return {"summary": "요약", "observations": ["관찰"], "next_session": "다음"}

    trials = [trial(1, True, i) for i in range(3)]
    graph = gr.summarize(CURRENT, SIMILAR)
    md, source = r.build_report(trials, {}, T0, T0 + timedelta(minutes=5), "succeeded", llm, "SYS", 5.0, graph=graph)
    assert source.startswith("llm:") and "### 지식 그래프 근거" in md
    assert set(calls[0]) == {"session_minutes", "outcome", "modes", "graph"}
    assert not re.search(r"[0-9a-f]{8}-[0-9a-f]{4}", json.dumps(calls[0]))


def test_report_without_graph_notes_reason():
    trials = [trial(1, True, 1)]
    md, _ = r.build_report(trials, {}, T0, T0, "succeeded", None, "SYS", 5.0, graph_note="DB 조회 실패 OperationalError")
    assert "지식 그래프 근거 없음 (DB 조회 실패" in md and "### 지식 그래프 근거" not in md


def test_prompt_explains_graph():
    from rally_brain.llm_advisor import load_prompt
    text = load_prompt("prompts/session_report.md")
    assert "graph" in text and "trunk_lean" in text and "단정하지 않는다" in text


# ---------- 휴식 판단용 (PR ②) ----------
def test_rest_summary_thresholds_and_mode1():
    rows = {g.mode: g for g in gr.summarize(CURRENT, SIMILAR)}
    m1 = gr.rest_summary(rows[1])
    assert m1 == {"similar_trials": 9, "fatigue_linked_share": 0.556}          # 모드 1: 성공률 안 보냄
    m5 = gr.GraphMode(mode=5, this_trials=3, this_comp={}, similar=gr.CompStat(5, 3),
                      similar_any=gr.CompStat(3, 1), similar_clean=gr.CompStat(2, 2), fatigue_linked=3)
    assert gr.rest_summary(m5) == {"similar_trials": 5, "fatigue_linked_share": 0.6, "with_compensation_rate": 0.333}
    assert gr.rest_summary(gr.GraphMode(mode=5, this_trials=1, this_comp={}, similar=gr.CompStat(2, 1))) is None


def test_rest_summary_fits_rest_whitelist():
    from rally_brain.rest_advisor import GRAPH_FIELDS, build_graph
    m5 = gr.GraphMode(mode=5, this_trials=3, this_comp={}, similar=gr.CompStat(6, 3),
                      similar_any=gr.CompStat(3, 1), similar_clean=gr.CompStat(3, 2), fatigue_linked=3)
    summ = gr.rest_summary(m5)
    assert set(summ) == GRAPH_FIELDS and build_graph(summ, GRAPH_FIELDS) == summ


class CacheRepo:
    def __init__(self, fail=False):
        self.fail, self.queries, self.seeded = fail, [], 0

    def graph_ready(self):
        return True

    def upsert_graph(self, nodes, edges):
        self.seeded += 1

    def graph_trials(self, sid, k):
        self.queries.append(sid)
        if self.fail:
            raise RuntimeError("db down")
        return [dict(r, mode=5) for r in CURRENT], [dict(r, mode=5) for r in SIMILAR]


def wait_idle(cache):
    for _ in range(200):
        if cache._worker is None:
            return
        time.sleep(0.01)
    raise AssertionError("worker did not finish")


def test_rest_cache_nonblocking_and_session_scoped():
    repo = CacheRepo()
    cache = gr.RestGraphCache(lambda: repo, delay_s=0.0)
    assert cache.get(SID, 5) is None                     # 아직 없음 → 그래프 없이
    cache.refresh(SID)
    wait_idle(cache)
    got = cache.get(SID, 5)
    assert got["similar_trials"] == 9 and "with_compensation_rate" in got
    assert cache.get("other", 5) is None and cache.get(SID, 7) is None
    cache.refresh(SID)
    wait_idle(cache)
    assert repo.seeded == 1                              # 개념 그래프는 한 번만


def test_rest_cache_coalesces_refresh_while_running():
    gate = threading.Event()
    repo = CacheRepo()
    cache = gr.RestGraphCache(lambda: repo, delay_s=0.0, sleep=lambda _s: gate.wait(1.0))
    cache.refresh(SID)
    cache.refresh(SID)
    cache.refresh(SID)                                   # 돌고 있는 동안 여러 번 → 한 번만 더
    assert threading.active_count() >= 2
    gate.set()
    wait_idle(cache)
    assert len(repo.queries) == 2


def test_rest_cache_failure_keeps_going():
    logs = []
    repo = CacheRepo(fail=True)
    cache = gr.RestGraphCache(lambda: repo, delay_s=0.0, log=logs.append)
    cache.refresh(SID)
    wait_idle(cache)
    assert cache.get(SID, 5) is None and logs and "그래프 없이" in logs[0]
    cache.close()
    cache.refresh(SID)
    assert cache._worker is None
