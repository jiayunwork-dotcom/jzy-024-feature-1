"""HTTP 路由与对象档持久化的端到端测试。"""

import importlib
import math
import os

import pytest
from fastapi.testclient import TestClient

from tests.conftest import loggrid

GRID = loggrid()


def sweep(client, plant, grid=None, **kw):
    return client.post("/api/sweep", json={"plant": plant, "freqs": grid or GRID, **kw})


def test_startup_plant_loaded_and_sweepable(client):
    r = sweep(client, "startup_plant")
    assert r.status_code == 200, r.text
    body = r.json()
    assert 30 < body["phase_margin_deg"] < 60
    assert body["gain_margin"] == pytest.approx(11.0, rel=1e-6)
    assert body["stable"] is True
    assert body["gain_crossover"]["exists"] and body["phase_crossover"]["exists"]
    # 网格幅相齐全
    n = len(GRID)
    for key in ("omega", "magnitude", "magnitude_db", "phase_deg"):
        assert len(body["bode"][key]) == n


def test_sweep_inline_poly(client):
    r = sweep(client, {"type": "poly", "num": [10.0],
                       "den": [1.0, 11.0, 10.0, 0.0], "K": 1.0, "L": 0.0})
    assert r.status_code == 200
    assert 30 < r.json()["phase_margin_deg"] < 60


def test_sweep_inline_nonproper_rejected_with_no_reading(client):
    r = sweep(client, {"type": "poly", "num": [1.0, 2.0, 3.0],
                       "den": [1.0, 1.0], "K": 1.0})
    assert r.status_code == 400
    body = r.json()
    assert "非真有理" in body["error"]
    assert "phase_margin_deg" not in body  # 绝不给假读数


def test_sweep_empty_grid_rejected(client):
    r = client.post("/api/sweep", json={"plant": "startup_plant", "freqs": []})
    assert r.status_code == 400
    assert "非空" in r.json()["error"]


def test_unknown_plant_404(client):
    r = sweep(client, "no_such_thing")
    assert r.status_code == 404
    assert "不存在" in r.json()["error"]


def test_monotonic_K_overrides_via_http(client):
    r1 = sweep(client, "startup_plant", K=1.0).json()
    r2 = sweep(client, "startup_plant", K=3.0).json()
    assert r2["gain_crossover"]["omega_c"] > r1["gain_crossover"]["omega_c"]
    assert r2["phase_margin_deg"] < r1["phase_margin_deg"]


def test_delay_overrides_magnitude_unchanged(client):
    r0 = sweep(client, "startup_plant", L=0.0).json()
    r1 = sweep(client, "startup_plant", L=0.5).json()
    assert r1["phase_margin_deg"] < r0["phase_margin_deg"]
    for a, b in zip(r0["bode"]["magnitude"], r1["bode"]["magnitude"]):
        assert a == pytest.approx(b, rel=1e-12, abs=0.0)
    # 当次覆盖不改动档本身
    listed = client.get("/api/plants/startup_plant").json()
    assert listed["L"] == 0.0


def test_no_crossover_marked_explicitly(client):
    body = sweep(client, "startup_plant",
                 grid=loggrid(0.01, 0.1, 200), K=0.01).json()
    assert body["gain_crossover"]["exists"] is False
    assert body["gain_crossover"]["omega_c"] is None
    assert body["gain_crossover"]["note"] == "无有限幅值穿越"
    assert body["phase_margin_deg"] is None


def test_plant_crud_and_listing(client, startup_spec):
    # 新增
    r = client.post("/api/plants/motor", json={"spec": startup_spec})
    assert r.status_code == 201
    # 重复新增
    assert client.post("/api/plants/motor", json={"spec": startup_spec}).status_code == 409
    # 列出：名字、分子分母、K、L 都在
    names = [p["name"] for p in client.get("/api/plants").json()["plants"]]
    assert "motor" in names and "startup_plant" in names
    motor = [p for p in client.get("/api/plants").json()["plants"] if p["name"] == "motor"][0]
    assert motor["K"] == 1.0 and motor["L"] == 0.0
    assert motor["num"] == [10.0]

    # 改完系数再扫：把 K 调到 3
    changed = dict(startup_spec, K=3.0)
    assert client.put("/api/plants/motor", json={"spec": changed}).status_code == 200
    assert client.get("/api/plants/motor").json()["K"] == 3.0
    body = sweep(client, "motor").json()
    assert body["plant"]["K"] == 3.0

    # 非法修改被拒绝，旧档保持不变
    bad_spec = {"type": "poly", "num": [1.0, 2.0, 3.0], "den": [1.0], "K": 1.0}
    assert client.put("/api/plants/motor", json={"spec": bad_spec}).status_code == 400
    assert client.get("/api/plants/motor").json()["K"] == 3.0

    # 删除
    assert client.delete("/api/plants/motor").status_code == 200
    assert sweep(client, "motor").status_code == 404


def test_create_invalid_plant_never_persisted(client):
    bad = {"type": "poly", "num": [1.0, 2.0, 3.0], "den": [1.0], "K": 1.0}
    r = client.post("/api/plants/bad", json={"spec": bad})
    assert r.status_code == 400
    assert sweep(client, "bad").status_code == 404


def test_persistence_survives_process_restart(client, tmp_path, monkeypatch, startup_spec):
    client.post("/api/plants/motor", json={"spec": startup_spec})
    assert os.path.isfile(os.path.join(str(tmp_path / "plants"), "motor.json"))

    # 模拟进程重启：重新载入 app，指向同一个档目录
    monkeypatch.setenv("PLANT_DIR", str(tmp_path / "plants"))
    import app.main as main
    importlib.reload(main)
    with TestClient(main.app) as c2:
        names = [p["name"] for p in c2.get("/api/plants").json()["plants"]]
        assert "motor" in names and "startup_plant" in names
        r = c2.post("/api/sweep", json={"plant": "motor", "freqs": GRID})
        assert r.status_code == 200
        assert 30 < r.json()["phase_margin_deg"] < 60


def test_zpk_listing_kept(client):
    spec = {"type": "zpk", "zeros": [], "poles": [[0.0, 0.0], [-1.0, 0.0], [-10.0, 0.0]],
            "gain": 10.0, "K": 1.0, "L": 0.0}
    client.post("/api/plants/z", json={"spec": spec})
    entry = [p for p in client.get("/api/plants").json()["plants"] if p["name"] == "z"][0]
    assert entry["zpk"]["gain"] == 10.0
    assert len(entry["zpk"]["poles"]) == 3
    assert entry["den"] == [1.0, 11.0, 10.0, 0.0]


def test_db_consistency_via_http(client):
    body = sweep(client, "startup_plant").json()
    for m, db in zip(body["bode"]["magnitude"], body["bode"]["magnitude_db"]):
        assert db == 20.0 * math.log10(m)


# ---------------------------------------------------------------------------
# 鲁棒最坏情况裕度：POST /api/robust-margins
# ---------------------------------------------------------------------------

def robust(client, plant, unc=None, grid=None, **kw):
    return client.post("/api/robust-margins",
                       json={"plant": plant, "freqs": grid or GRID,
                             "uncertainty": unc, **kw})


def test_robust_named_plant_degenerate_matches_sweep(client):
    r = robust(client, "startup_plant").json()
    s = sweep(client, "startup_plant").json()
    assert r["worst_phase_margin"]["phase_margin_deg"] == s["phase_margin_deg"]
    assert r["worst_gain_margin"]["gain_margin"] == s["gain_margin"]
    assert r["worst_phase_margin"]["gain_crossover"]["omega_c"] == s["gain_crossover"]["omega_c"]
    assert r["robust_stable"] is True and r["critical_point"]["found"] is False
    assert r["family"] == {"K": [1.0, 1.0], "L": [0.0, 0.0], "pole": None}


def test_robust_inline_plant_and_delay_family(client):
    spec = {"type": "poly", "num": [10.0], "den": [1.0, 11.0, 10.0, 0.0],
            "K": 1.0, "L": 0.0}
    r = robust(client, spec, {"L": [0.0, 0.4]}).json()
    s = sweep(client, spec, L=0.4).json()
    assert r["worst_phase_margin"]["phase_margin_deg"] == s["phase_margin_deg"]
    assert r["worst_phase_margin"]["at"] == {"K": 1.0, "L": 0.4, "pole": None}


def test_robust_K_override_and_pole_family_via_http(client):
    unc = {"pole": {"nominal": -1.0, "range": [-2.0, -0.5]}}
    r = robust(client, "startup_plant", unc, K=2.0).json()
    assert r["worst_phase_margin"]["at"]["K"] == 2.0
    assert r["worst_phase_margin"]["at"]["pole"] == pytest.approx(-0.5, abs=1e-8)
    # 当次覆盖不改动对象档
    assert client.get("/api/plants/startup_plant").json()["K"] == 1.0


def test_robust_unknown_plant_404(client):
    assert robust(client, "ghost", {"K": [1, 2]}).status_code == 404


def test_robust_bad_interval_rejected_without_reading(client):
    r = robust(client, "startup_plant", {"K": [3, 1]})
    assert r.status_code == 400
    assert "worst_phase_margin" not in r.json()
    assert "上界" in r.json()["error"]
    r = robust(client, "startup_plant", {"L": [-1, 1]})
    assert r.status_code == 400
    r = robust(client, "startup_plant",
               {"pole": {"nominal": -1, "range": [-1, 1]}})
    assert r.status_code == 400 and "右半平面" in r.json()["error"]


def test_robust_bad_grid_rejected(client):
    r = client.post("/api/robust-margins",
                    json={"plant": "startup_plant", "freqs": [],
                          "uncertainty": {"K": [1, 2]}})
    assert r.status_code == 400 and "非空" in r.json()["error"]


def test_robust_no_crossover_family_explicit_mark(client):
    r = robust(client, "startup_plant", {"K": [1e-7, 1e-5]},
               grid=loggrid(0.01, 10, 401)).json()
    blk = r["worst_phase_margin"]
    assert blk["exists"] is False and blk["phase_margin_deg"] is None
    assert blk["gain_crossover"]["note"] == "全族无有限幅值穿越"
    assert blk["at"] is None
    assert r["robust_stable"] is False


def test_robust_unstable_family_critical_point(client):
    r = robust(client, "startup_plant", {"K": [1.0, 12.0]}).json()
    assert r["robust_stable"] is False
    c = r["critical_point"]
    assert c["found"] and c["at"]["K"] == pytest.approx(11.0, abs=1e-6)
    assert abs(c["phase_margin_deg"]) < 1e-5


def test_robust_analysis_is_not_persisted(client):
    robust(client, "startup_plant", {"K": [1, 3], "L": [0, 0.2]})
    # 档目录里仍只有对象档本身，没有任何鲁棒分析流水文件
    files = client.get("/api/plants").json()["plants"]
    assert [p["name"] for p in files] == ["startup_plant"]


def test_robust_listed_in_root(client):
    assert "POST /api/robust-margins" in client.get("/").json()["endpoints"]
