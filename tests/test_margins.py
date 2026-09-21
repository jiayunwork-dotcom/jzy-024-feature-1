"""锁住题目点名的核心数值行为（直接打计算层，不绕 HTTP）。"""

import math

import pytest

from app.frf import TransferFunction, mag_to_db
from app.margins import analyze
from app.validation import validate_plant, validate_grid
from tests.conftest import loggrid

STARTUP_ZPK = {
    "type": "zpk",
    "zeros": [],
    "poles": [[0.0, 0.0], [-1.0, 0.0], [-10.0, 0.0]],
    "gain": 10.0,
    "K": 1.0,
    "L": 0.0,
}


def analyze_with(spec, grid=None, K=None, L=None):
    plant = validate_plant(spec, K_override=K, L_override=L)
    return analyze(TransferFunction(plant), validate_grid(grid or loggrid()))


# 1) 启动对象：相位裕度为正，且落在几十度量级
def test_startup_phase_margin_positive_tens_of_degrees():
    r = analyze_with(STARTUP_ZPK)
    assert r.has_finite_gain_crossover and r.has_finite_phase_crossover
    assert r.phase_margin_deg == pytest.approx(47.4, abs=1.0)
    assert 30.0 < r.phase_margin_deg < 60.0
    assert r.stable


def test_startup_margins_hand_checkable():
    # 手算核对：ω_c 处 |G|=1；ω_π = √10；GM = 11（≈20.83 dB）
    r = analyze_with(STARTUP_ZPK)
    wc = r.omega_c
    mag = 10.0 / (wc * math.hypot(1.0, wc) * math.hypot(10.0, wc))
    assert mag == pytest.approx(1.0, abs=1e-6)
    assert r.omega_pi == pytest.approx(math.sqrt(10.0), rel=1e-8)
    assert r.gain_margin == pytest.approx(11.0, rel=1e-6)
    assert r.gain_margin_db == pytest.approx(20.0 * math.log10(11.0), abs=1e-9)


# 2) K 只加大、零极点不动：ω_c 往高频挪、相位裕度变差
def test_increasing_K_moves_crossover_up_and_lowers_PM():
    r1 = analyze_with(STARTUP_ZPK, K=1.0)
    r2 = analyze_with(STARTUP_ZPK, K=2.0)
    r3 = analyze_with(STARTUP_ZPK, K=5.0)
    assert r1.omega_c < r2.omega_c < r3.omega_c
    assert r1.phase_margin_deg > r2.phase_margin_deg > r3.phase_margin_deg


# 3) 稳定极点朝原点搬、K 不变：低频相位更负、相位裕度下降
def test_moving_pole_toward_origin_lowers_PM():
    # 原系统 (s+1)(s+10)；把极点 -1 搬到 -0.5（分子、K 不变）
    moved = {
        "type": "poly",
        "num": [10.0],
        "den": [1.0, 10.5, 5.0, 0.0],  # s(s+0.5)(s+10)
        "K": 1.0,
        "L": 0.0,
    }
    r0 = analyze_with(STARTUP_ZPK)
    r1 = analyze_with(moved)
    # 同一低频点上相位更负
    w = 0.1
    tf0 = TransferFunction(validate_plant(STARTUP_ZPK))
    tf1 = TransferFunction(validate_plant(moved))
    p0 = tf0.bode_grid([w]).phase_deg[0]
    p1 = tf1.bode_grid([w]).phase_deg[0]
    assert p1 < p0
    assert r1.phase_margin_deg < r0.phase_margin_deg


# 4) 只加长延迟：相位裕度下降、幅值曲线形状不变
def test_delay_only_changes_phase_not_magnitude():
    grid = loggrid()
    r0 = analyze_with(STARTUP_ZPK, grid=grid, L=0.0)
    rL = analyze_with(STARTUP_ZPK, grid=grid, L=0.3)
    assert rL.phase_margin_deg < r0.phase_margin_deg
    assert rL.bode["magnitude"] == pytest.approx(r0.bode["magnitude"], rel=1e-12)
    assert rL.bode["magnitude_db"] == pytest.approx(r0.bode["magnitude_db"], abs=1e-9)
    # ω_c 不变（延迟不动幅值），相位差应等于 ω_c*L（度）
    assert rL.omega_c == pytest.approx(r0.omega_c, rel=1e-8)
    assert rL.phase_margin_deg == pytest.approx(
        r0.phase_margin_deg - r0.omega_c * 0.3 * 180.0 / math.pi, abs=1e-6
    )


# 5) K 收到接近 0：幅值裕度变大，或相位穿越消失报「无有限穿越」
def test_small_K_grows_gain_margin_or_no_crossover():
    r1 = analyze_with(STARTUP_ZPK, K=1.0)
    rs = analyze_with(STARTUP_ZPK, K=1e-3)
    if rs.has_finite_phase_crossover:
        assert rs.gain_margin > r1.gain_margin
    else:
        assert rs.gain_margin is None and rs.gain_margin_db is None


# 6) 无有限穿越：字段为 None，绝不填 0
def test_no_gain_crossover_returns_null_not_zero():
    # K=0.01 时全网格 |G|<1：没有幅值穿越；但 −180° 穿越仍在
    r = analyze_with(STARTUP_ZPK, K=0.01)
    assert not r.has_finite_gain_crossover
    assert r.omega_c is None
    assert r.phase_margin_deg is None
    assert r.has_finite_phase_crossover


def test_no_phase_crossover_returns_null_not_zero():
    # 一阶惯性 + 积分：相位在 (-180°, -90°) 之间，永不穿过 −180°
    spec = {"type": "poly", "num": [1.0], "den": [1.0, 1.0, 0.0], "K": 1.0, "L": 0.0}
    r = analyze_with(spec)
    assert r.has_finite_gain_crossover
    assert not r.has_finite_phase_crossover
    assert r.omega_pi is None
    assert r.gain_margin is None and r.gain_margin_db is None
    assert r.phase_margin_deg > 0.0 and r.stable


def test_neither_crossover_inside_grid_nulls_all_margins():
    # 低频网格只覆盖 0.01~0.1：|G| 全程远大于 1，相位也没到 −180°
    r = analyze_with(STARTUP_ZPK, grid=loggrid(0.01, 0.1, n=200))
    assert not r.has_finite_gain_crossover
    assert not r.has_finite_phase_crossover
    assert r.phase_margin_deg is None
    assert r.gain_margin is None and r.gain_margin_db is None


# 7) 分贝必须等于 20*log10(真值)，逐点一致
def test_db_equals_20_log10_magnitude_everywhere():
    r = analyze_with(STARTUP_ZPK)
    for m, db in zip(r.bode["magnitude"], r.bode["magnitude_db"]):
        assert db == pytest.approx(20.0 * math.log10(m), rel=1e-12, abs=1e-12)
    assert mag_to_db(1.0) == 0.0
    assert mag_to_db(11.0) == pytest.approx(20.0 * math.log10(11.0), abs=1e-12)


# 8) 穿越在网格点之间时必须加密，不能拿邻近点冒充
def test_crossover_refined_between_grid_points():
    coarse = loggrid(0.01, 100.0, n=31)  # 故意很稀
    r = analyze(TransferFunction(validate_plant(STARTUP_ZPK)), validate_grid(coarse))
    assert r.omega_c is not None and r.omega_c not in coarse
    assert abs(r.omega_c - 0.7844079) < 1e-5
    # 加密点上的幅值应当就是 1
    m = TransferFunction(validate_plant(STARTUP_ZPK)).magnitude(r.omega_c)
    assert m == pytest.approx(1.0, abs=1e-7)
    assert r.gain_margin == pytest.approx(11.0, rel=1e-6)
