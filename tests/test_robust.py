"""鲁棒裕度分析：四条不变量 + 非法区间拒绝 + 全族无穿越标记。

四条不变量（正确性锚点）：
  1) 退化一致性：所有区间收成单点，最坏读数与直接扫频逐项相等；
  2) 区间单调包含：区间放大，最坏裕度只会变差或持平；
  3) 延迟最坏端锁定：纯延迟族的最坏相位裕度 == 延迟上界的单点读数；
  4) 逐点口径对齐：最坏参数点回代单点扫频，两边数值完全一致。
"""

import math

import pytest

from app.errors import ServiceError
from app.frf import TransferFunction
from app.margins import analyze
from app.robust import _minimize_1d, robust_analyze
from app.uncertainty import PlantFamily, PoleInterval, PoleMover, validate_uncertainty
from app.validation import validate_grid, validate_plant
from tests.conftest import loggrid

GRID = loggrid()

STARTUP_ZPK = {
    "type": "zpk",
    "zeros": [],
    "poles": [[0.0, 0.0], [-1.0, 0.0], [-10.0, 0.0]],
    "gain": 10.0,
    "K": 1.0,
    "L": 0.0,
}

STARTUP_POLY = {"type": "poly", "num": [10.0], "den": [1.0, 11.0, 10.0, 0.0],
                "K": 1.0, "L": 0.0}


def family(spec, unc_raw):
    return PlantFamily(spec, validate_uncertainty(unc_raw))


def robust(spec, unc_raw, grid=None):
    return robust_analyze(family(spec, unc_raw), validate_grid(grid or GRID))


def direct(spec, K=None, L=None, grid=None):
    """把参数直接送进现有单点扫频（对照口径）。"""
    plant = validate_plant(spec, K_override=K, L_override=L)
    return analyze(TransferFunction(plant), validate_grid(grid or GRID))


def moved_pole_spec(p):
    """把启动对象的 -1 极点搬到 p（zpk 形式，其余零极点不动）。"""
    return dict(STARTUP_ZPK, poles=[[0.0, 0.0], [p, 0.0], [-10.0, 0.0]])


# ---------------------------------------------------------------------------
# 不变量 1：退化一致性
# ---------------------------------------------------------------------------

def test_degenerate_no_uncertainty_matches_plain_sweep():
    res = robust(STARTUP_ZPK, None)
    ref = direct(STARTUP_ZPK)
    assert res.worst_phase_margin.value == ref.phase_margin_deg
    assert res.worst_phase_margin.omega == ref.omega_c
    assert res.worst_gain_margin.value == ref.gain_margin
    assert res.worst_gain_margin.value_db == ref.gain_margin_db
    assert res.worst_gain_margin.omega == ref.omega_pi
    assert res.worst_phase_margin.params == {"K": 1.0, "L": 0.0, "pole": None}
    assert res.worst_phase_margin.coverage == "all"
    assert res.robustly_stable is True
    assert res.first_violation is None


def test_degenerate_empty_uncertainty_object_matches_plain_sweep():
    res = robust(STARTUP_ZPK, {})
    ref = direct(STARTUP_ZPK)
    assert res.worst_phase_margin.value == ref.phase_margin_deg
    assert res.worst_gain_margin.value_db == ref.gain_margin_db


def test_degenerate_single_point_intervals_match_direct_sweep():
    unc = {"K": {"min": 2.5, "max": 2.5},
           "L": {"min": 0.4, "max": 0.4},
           "pole": {"nominal": -1.0, "min": -1.0, "max": -1.0}}
    res = robust(STARTUP_ZPK, unc)
    ref = direct(STARTUP_ZPK, K=2.5, L=0.4)
    assert res.worst_phase_margin.value == ref.phase_margin_deg
    assert res.worst_phase_margin.omega == ref.omega_c
    assert res.worst_gain_margin.value == ref.gain_margin
    assert res.worst_gain_margin.value_db == ref.gain_margin_db
    assert res.worst_gain_margin.omega == ref.omega_pi
    assert res.worst_phase_margin.params == {"K": 2.5, "L": 0.4, "pole": -1.0}
    assert res.worst_gain_margin.params == {"K": 2.5, "L": 0.4, "pole": -1.0}


def test_degenerate_moved_pole_single_point_matches_direct_zpk():
    # 极点区间收成非名义单点：等价于把改动后的对象直接送进扫频
    res = robust(STARTUP_ZPK, {"pole": {"nominal": -1.0, "min": -0.6, "max": -0.6}})
    ref = direct(moved_pole_spec(-0.6))
    assert res.worst_phase_margin.value == ref.phase_margin_deg
    assert res.worst_phase_margin.omega == ref.omega_c
    assert res.worst_gain_margin.value == ref.gain_margin
    assert res.worst_gain_margin.omega == ref.omega_pi


# ---------------------------------------------------------------------------
# 不变量 2：区间单调包含（放大区间，最坏裕度只会变差或持平）
# ---------------------------------------------------------------------------

def test_enlarging_delay_interval_never_improves_worst_pm():
    r1 = robust(STARTUP_ZPK, {"L": {"min": 0.0, "max": 0.2}})
    r2 = robust(STARTUP_ZPK, {"L": {"min": 0.0, "max": 0.5}})   # 上界抬高
    r3 = robust(STARTUP_ZPK, {"L": {"min": 0.1, "max": 0.5}})   # 下界抬高（缩小）
    assert r2.worst_phase_margin.value <= r1.worst_phase_margin.value
    assert r3.worst_phase_margin.value >= r2.worst_phase_margin.value


def test_enlarging_gain_interval_never_improves_worst_margins():
    r1 = robust(STARTUP_ZPK, {"K": {"min": 1.0, "max": 2.0}})
    r2 = robust(STARTUP_ZPK, {"K": {"min": 1.0, "max": 4.0}})   # 上界抬高
    assert r2.worst_phase_margin.value <= r1.worst_phase_margin.value
    assert r2.worst_gain_margin.value_db <= r1.worst_gain_margin.value_db
    # 向下界方向放大（温和端）不影响最坏值：单调维度坍缩到上界，逐项相等
    r3 = robust(STARTUP_ZPK, {"K": {"min": 0.5, "max": 2.0}})
    assert r3.worst_phase_margin.value == r1.worst_phase_margin.value
    assert r3.worst_gain_margin.value_db == r1.worst_gain_margin.value_db


def test_enlarging_pole_interval_never_improves_worst_margins():
    r1 = robust(STARTUP_ZPK, {"pole": {"nominal": -1.0, "min": -1.2, "max": -0.8}})
    # 向不利方向（朝原点）拓宽：最坏值严格变差
    r2 = robust(STARTUP_ZPK, {"pole": {"nominal": -1.0, "min": -1.2, "max": -0.3}})
    assert r2.worst_phase_margin.value <= r1.worst_phase_margin.value
    assert r2.worst_gain_margin.value_db <= r1.worst_gain_margin.value_db
    # 向温和方向拓宽：最坏值持平（容差内）
    r3 = robust(STARTUP_ZPK, {"pole": {"nominal": -1.0, "min": -2.0, "max": -0.8}})
    assert r3.worst_phase_margin.value <= r1.worst_phase_margin.value + 1e-9
    assert r3.worst_phase_margin.value == pytest.approx(r1.worst_phase_margin.value, abs=1e-4)
    assert r3.worst_gain_margin.value_db == pytest.approx(r1.worst_gain_margin.value_db, abs=1e-4)


# ---------------------------------------------------------------------------
# 不变量 3：延迟最坏端锁定
# ---------------------------------------------------------------------------

def test_delay_only_family_worst_pm_exactly_at_upper_bound():
    res = robust(STARTUP_ZPK, {"L": {"min": 0.0, "max": 0.35}})
    ref = direct(STARTUP_ZPK, L=0.35)
    assert res.worst_phase_margin.params["L"] == 0.35
    assert res.worst_phase_margin.value == ref.phase_margin_deg
    assert res.worst_phase_margin.omega == ref.omega_c


def test_delay_only_family_worst_gm_exactly_at_upper_bound():
    res = robust(STARTUP_ZPK, {"L": {"min": 0.0, "max": 0.3}})
    ref = direct(STARTUP_ZPK, L=0.3)
    assert res.worst_gain_margin.params["L"] == 0.3
    assert res.worst_gain_margin.value == ref.gain_margin
    assert res.worst_gain_margin.value_db == ref.gain_margin_db
    assert res.worst_gain_margin.omega == ref.omega_pi


# ---------------------------------------------------------------------------
# 不变量 4：最坏参数点回代单点扫频，口径一致
# ---------------------------------------------------------------------------

def test_worst_params_replayed_through_single_point_sweep():
    unc = {"K": {"min": 1.0, "max": 3.0},
           "L": {"min": 0.0, "max": 0.2},
           "pole": {"nominal": -1.0, "min": -1.5, "max": -0.5}}
    res = robust(STARTUP_ZPK, unc)

    pm = res.worst_phase_margin
    ref_pm = direct(moved_pole_spec(pm.params["pole"]), K=pm.params["K"], L=pm.params["L"])
    assert pm.value == ref_pm.phase_margin_deg
    assert pm.omega == ref_pm.omega_c

    gm = res.worst_gain_margin
    ref_gm = direct(moved_pole_spec(gm.params["pole"]), K=gm.params["K"], L=gm.params["L"])
    assert gm.value == ref_gm.gain_margin
    assert gm.value_db == ref_gm.gain_margin_db
    assert gm.omega == ref_gm.omega_pi


# ---------------------------------------------------------------------------
# 单调维度的坍缩与极点搜索的正确性
# ---------------------------------------------------------------------------

def test_monotone_dims_collapse_to_upper_bounds():
    res = robust(STARTUP_ZPK, {"K": {"min": 0.5, "max": 3.0}, "L": {"min": 0.0, "max": 0.2}})
    assert res.worst_phase_margin.params["K"] == 3.0
    assert res.worst_phase_margin.params["L"] == 0.2
    assert res.worst_gain_margin.params["K"] == 3.0
    assert res.worst_gain_margin.params["L"] == 0.2


def test_pole_only_family_worst_margins_hand_checkable():
    # G(s) = 10/[s(s+a)(s+10)]，a ∈ [0.5, 1.5]：PM、GM 都随 a 单调，
    # 最坏点在离原点最近的 a=0.5；GM = a(10+a) 可手算核对。
    res = robust(STARTUP_ZPK, {"pole": {"nominal": -1.0, "min": -1.5, "max": -0.5}})
    assert res.worst_phase_margin.params["pole"] == pytest.approx(-0.5, abs=1e-6)
    assert res.worst_gain_margin.params["pole"] == pytest.approx(-0.5, abs=1e-6)
    assert res.worst_gain_margin.value == pytest.approx(0.5 * 10.5, rel=1e-6)
    # 最坏 PM 与 a=0.5 的单点读数一致
    ref = direct(moved_pole_spec(-0.5))
    assert res.worst_phase_margin.value == pytest.approx(ref.phase_margin_deg, abs=1e-6)


def test_minimize_1d_pins_interior_minimum_not_coarse_point():
    # 内部极小点在 x=-0.7；25 点粗扫网格上并不存在这个点
    x, v = _minimize_1d(lambda t: (t + 0.7) ** 2, -2.0, 0.0)
    assert x == pytest.approx(-0.7, abs=1e-6)
    assert v == pytest.approx(0.0, abs=1e-12)
    coarse = [-2.0 + 2.0 * i / 24 for i in range(25)]
    assert all(abs(x - p) > 1e-3 for p in coarse)  # 不是拿粗扫点冒充的


def test_minimize_1d_all_none_returns_none():
    x, v = _minimize_1d(lambda t: None, -1.0, -0.5)
    assert x is None and v is None


def test_minimize_1d_single_point_interval():
    x, v = _minimize_1d(lambda t: t * t, -0.5, -0.5)
    assert x == -0.5 and v == 0.25


# ---------------------------------------------------------------------------
# 非法不确定区间：当场拒绝，不给读数
# ---------------------------------------------------------------------------

def test_interval_upper_less_than_lower_rejected():
    with pytest.raises(ServiceError) as ei:
        validate_uncertainty({"K": {"min": 2.0, "max": 1.0}})
    assert "上界" in ei.value.reason
    with pytest.raises(ServiceError):
        validate_uncertainty({"L": {"min": 0.5, "max": 0.1}})
    with pytest.raises(ServiceError):
        validate_uncertainty({"pole": {"nominal": -1.0, "min": -0.5, "max": -1.5}})


def test_delay_interval_with_negative_rejected():
    with pytest.raises(ServiceError) as ei:
        validate_uncertainty({"L": {"min": -0.1, "max": 0.2}})
    assert "负" in ei.value.reason


def test_gain_interval_must_be_positive():
    with pytest.raises(ServiceError) as ei:
        validate_uncertainty({"K": {"min": 0.0, "max": 1.0}})
    assert "正" in ei.value.reason
    with pytest.raises(ServiceError):
        validate_uncertainty({"K": {"min": -1.0, "max": 2.0}})


def test_pole_interval_crossing_imaginary_axis_rejected():
    with pytest.raises(ServiceError) as ei:
        validate_uncertainty({"pole": {"nominal": -1.0, "min": -1.5, "max": 0.5}})
    assert "右半平面" in ei.value.reason


def test_pole_nominal_identifies_pole_regardless_of_interval():
    # nominal 只负责标识「哪个极点在飘」，区间才是活动范围；
    # 区间不含名义值也合法（例如把 -1 极点收成 -0.6 单点族）。
    unc = validate_uncertainty({"pole": {"nominal": -1.0, "min": -0.6, "max": -0.6}})
    assert unc.pole.nominal == -1.0
    res = robust(STARTUP_ZPK, {"pole": {"nominal": -1.0, "min": -0.6, "max": -0.6}})
    ref = direct(moved_pole_spec(-0.6))
    assert res.worst_phase_margin.value == ref.phase_margin_deg


def test_pole_nominal_must_be_actual_pole():
    unc = validate_uncertainty({"pole": {"nominal": -3.0, "min": -4.0, "max": -2.0}})
    with pytest.raises(ServiceError) as ei:
        PlantFamily(STARTUP_ZPK, unc)
    assert "找不到" in ei.value.reason
    # poly 形式同样要能找到分母的根
    with pytest.raises(ServiceError):
        PlantFamily(STARTUP_POLY, unc)


def test_uncertainty_endpoints_must_be_finite():
    with pytest.raises(ServiceError):
        validate_uncertainty({"K": {"min": float("nan"), "max": 1.0}})
    with pytest.raises(ServiceError):
        validate_uncertainty({"L": {"min": 0.0, "max": float("inf")}})
    with pytest.raises(ServiceError):
        validate_uncertainty({"K": {"min": 1.0}})  # 缺上界
    with pytest.raises(ServiceError):
        validate_uncertainty({"K": True})


# ---------------------------------------------------------------------------
# 全族 / 部分无有限穿越：显式标记，绝不填 0
# ---------------------------------------------------------------------------

def test_family_without_gain_crossover_marked_not_zero():
    # 增益压到极小：全族 |G|<1，没有任何幅值穿越
    res = robust(STARTUP_ZPK, {"K": {"min": 1e-4, "max": 1e-3}})
    pm = res.worst_phase_margin
    assert pm.exists is False
    assert pm.value is None
    assert pm.omega is None
    assert pm.coverage == "none"
    # 幅值裕度仍在（相位穿越与 K 无关），且最坏点在 K 上界
    assert res.worst_gain_margin.exists is True
    assert res.worst_gain_margin.params["K"] == 1e-3
    assert res.worst_gain_margin.value == pytest.approx(11.0 / 1e-3, rel=1e-6)
    # 最坏 PM 不存在 → 不能判鲁棒稳定，但也没有「越界点」
    assert res.robustly_stable is False
    assert res.first_violation is None


def test_partial_coverage_flagged_not_dropped():
    # K 从「全网格 |G|<1」跨到「有穿越」：族内一部分成员无幅值穿越
    res = robust(STARTUP_ZPK, {"K": {"min": 5e-3, "max": 1.0}})
    pm = res.worst_phase_margin
    assert pm.coverage == "partial"
    assert pm.exists is True
    assert pm.value is not None  # 最坏值落在有穿越的子族上，不是 0


# ---------------------------------------------------------------------------
# 鲁棒稳定判定
# ---------------------------------------------------------------------------

def test_robustly_stable_when_worst_margins_strictly_positive():
    res = robust(STARTUP_ZPK, {"K": {"min": 0.8, "max": 1.2}, "L": {"min": 0.0, "max": 0.05}})
    assert res.worst_phase_margin.value > 0.0
    assert res.worst_gain_margin.value_db > 0.0
    assert res.robustly_stable is True
    assert res.first_violation is None


def test_not_robust_reports_first_violation_params():
    # K 拉到 12 > GM=11 的临界：最坏 PM、GM 都翻负
    res = robust(STARTUP_ZPK, {"K": {"min": 1.0, "max": 12.0}})
    assert res.robustly_stable is False
    v = res.first_violation
    assert v is not None
    assert v["margin"] == "phase_margin"
    assert v["params"]["K"] == 12.0
    assert v["value_deg"] <= 0.0


def test_not_robust_when_worst_gain_margin_dips_below_zero_db():
    # PM 仍为正但 GM(dB) 先触零的族：K 上限略超 11
    res = robust(STARTUP_ZPK, {"K": {"min": 10.5, "max": 11.5}})
    assert res.worst_gain_margin.value_db < 0.0
    assert res.robustly_stable is False
    assert res.first_violation is not None


# ---------------------------------------------------------------------------
# PoleMover：poly 路径与退化捷径
# ---------------------------------------------------------------------------

def test_poly_pole_move_matches_manual_polynomial():
    mover = PoleMover(STARTUP_POLY, PoleInterval(nominal=-1.0, lo=-0.5, hi=-0.5))
    plant = validate_plant(mover.spec_at(-0.5))
    # s(s+0.5)(s+10) = s^3 + 10.5 s^2 + 5 s
    assert plant.den == pytest.approx((1.0, 10.5, 5.0, 0.0), rel=1e-12)
    assert plant.num == (10.0,)


def test_poly_pole_at_nominal_returns_original_coefficients():
    mover = PoleMover(STARTUP_POLY, PoleInterval(nominal=-1.0, lo=-1.0, hi=-1.0))
    plant = validate_plant(mover.spec_at(-1.0))
    assert plant.den == (1.0, 11.0, 10.0, 0.0)  # 逐项相等，不是近似


def test_poly_family_degenerate_matches_plain_sweep():
    res = robust(STARTUP_POLY, {"pole": {"nominal": -1.0, "min": -1.0, "max": -1.0}})
    ref = direct(STARTUP_POLY)
    assert res.worst_phase_margin.value == ref.phase_margin_deg
    assert res.worst_gain_margin.value_db == ref.gain_margin_db


# ---------------------------------------------------------------------------
# HTTP 端到端
# ---------------------------------------------------------------------------

def robust_post(client, plant, unc=None, grid=None):
    body = {"plant": plant, "freqs": grid or GRID}
    if unc is not None:
        body["uncertainty"] = unc
    return client.post("/api/robust", json=body)


def test_robust_endpoint_named_plant_matches_sweep_at_corner(client):
    unc = {"K": {"min": 1.0, "max": 2.0}, "L": {"min": 0.0, "max": 0.1}}
    r = robust_post(client, "startup_plant", unc)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["worst_phase_margin"]["params"] == {"K": 2.0, "L": 0.1, "pole": None}
    # 与单点扫频同一口径
    ref = client.post("/api/sweep",
                      json={"plant": "startup_plant", "freqs": GRID, "K": 2.0, "L": 0.1}).json()
    assert body["worst_phase_margin"]["value_deg"] == ref["phase_margin_deg"]
    assert body["worst_phase_margin"]["omega_c"] == ref["gain_crossover"]["omega_c"]
    assert body["worst_gain_margin"]["value_db"] == ref["gain_margin_db"]
    assert body["robustly_stable"] is True


def test_robust_endpoint_inline_plant_with_pole_uncertainty(client):
    unc = {"pole": {"nominal": -1.0, "min": -1.5, "max": -0.5}}
    r = robust_post(client, STARTUP_ZPK, unc)
    assert r.status_code == 200, r.text
    pm = r.json()["worst_phase_margin"]
    assert -1.5 <= pm["params"]["pole"] <= -0.5
    # 回代单点扫频复算，口径一致
    moved = dict(STARTUP_ZPK, poles=[[0.0, 0.0], [pm["params"]["pole"], 0.0], [-10.0, 0.0]])
    ref = client.post("/api/sweep", json={"plant": moved, "freqs": GRID}).json()
    assert pm["value_deg"] == ref["phase_margin_deg"]


def test_robust_endpoint_degenerate_matches_sweep(client):
    body = robust_post(client, "startup_plant").json()
    ref = client.post("/api/sweep", json={"plant": "startup_plant", "freqs": GRID}).json()
    assert body["worst_phase_margin"]["value_deg"] == ref["phase_margin_deg"]
    assert body["worst_gain_margin"]["value"] == ref["gain_margin"]
    assert body["worst_gain_margin"]["value_db"] == ref["gain_margin_db"]
    assert body["uncertainty"] == {"K": None, "L": None, "pole": None}


def test_robust_endpoint_illegal_uncertainty_rejected_without_reading(client):
    bad = [
        {"L": {"min": -0.5, "max": 0.1}},                                  # 延迟含负
        {"K": {"min": 2.0, "max": 1.0}},                                   # 上界 < 下界
        {"K": {"min": 0.0, "max": 1.0}},                                   # 增益非正
        {"pole": {"nominal": -1.0, "min": -1.0, "max": 0.5}},              # 越过虚轴
        {"pole": {"nominal": -3.0, "min": -4.0, "max": -2.0}},             # 不是对象的极点
    ]
    for unc in bad:
        r = robust_post(client, "startup_plant", unc)
        assert r.status_code == 400, r.text
        assert "error" in r.json()
        assert "worst_phase_margin" not in r.json()  # 绝不给假读数


def test_robust_endpoint_family_without_crossover_marked(client):
    r = robust_post(client, "startup_plant", {"K": {"min": 1e-4, "max": 1e-3}})
    assert r.status_code == 200
    pm = r.json()["worst_phase_margin"]
    assert pm["exists"] is False
    assert pm["value_deg"] is None
    assert pm["crossover_coverage"] == "none"
    assert "无有限幅值穿越" in pm["note"]
    assert r.json()["robustly_stable"] is False


def test_robust_endpoint_unknown_plant_404(client):
    r = robust_post(client, "no_such_thing")
    assert r.status_code == 404
    assert "不存在" in r.json()["error"]


def test_robust_endpoint_named_zpk_plant_pole_uncertainty(client):
    # 具名档按 zpk 存入：极点搬迁走 source_zpk 还原路径，与内联口径一致
    client.post("/api/plants/motor", json={"spec": STARTUP_ZPK})
    unc = {"pole": {"nominal": -1.0, "min": -1.5, "max": -0.5}}
    r_named = robust_post(client, "motor", unc)
    assert r_named.status_code == 200, r_named.text
    r_inline = robust_post(client, STARTUP_ZPK, unc)
    assert r_named.json()["worst_phase_margin"] == r_inline.json()["worst_phase_margin"]
    assert r_named.json()["worst_gain_margin"] == r_inline.json()["worst_gain_margin"]


def test_robust_endpoint_does_not_touch_archive(client):
    before = client.get("/api/plants/startup_plant").json()
    robust_post(client, "startup_plant", {"K": {"min": 1.0, "max": 5.0}})
    after = client.get("/api/plants/startup_plant").json()
    assert before == after  # 鲁棒分析是一次性读数，不改档
