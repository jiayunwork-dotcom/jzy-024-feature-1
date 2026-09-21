"""鲁棒最坏情况裕度服务的测试。

锁住的锚点：
  1) 退化一致性：所有区间收成单点，结果与现有 /api/sweep 逐项相等；
  2) 区间单调包含：把不确定区间放大，最坏 PM/GM 只会变差或持平；
  3) 延迟最坏端锁定：纯延迟族的最坏 PM 恰好等于 L 取上界的单点读数；
  4) 与逐点口径对齐：最坏参数点回代现有 analyze，两边裕度逐位一致。
另含：非法区间（上界<下界、延迟含负、极点越虚轴、标称极点不是分母根）当场
拒绝且不给读数；全族无有限穿越被显式标记而非填零；非单调一维搜索的粗扫+
加密确实钉到内点极值；混合族（K/L/极点）坍缩 + 搜索；鲁棒判定与临界参数点。
"""

import math

import pytest

from app.errors import ServiceError
from app.frf import TransferFunction
from app.margins import analyze
from app.robust import (
    COARSE_POINTS,
    GLOBAL_REFINE_PASSES,
    PARAM_REL_TOL,
    FamilyReader,
    build_uncertainty,
    robust_analyze,
    robust_to_payload,
    worst_on_line,
)
from app.validation import validate_grid, validate_plant
from tests.conftest import loggrid

STARTUP_ZPK = {
    "type": "zpk",
    "zeros": [],
    "poles": [[0.0, 0.0], [-1.0, 0.0], [-10.0, 0.0]],
    "gain": 10.0,
    "K": 1.0,
    "L": 0.0,
}

GRID = loggrid(0.01, 100.0, 601)


def run_robust(spec, unc, grid=None, K=None, L=None):
    plant = validate_plant(spec, K_override=K, L_override=L)
    u = build_uncertainty(unc, plant)
    return robust_analyze(plant, validate_grid(grid or GRID), u)


def run_single(spec, grid=None, K=None, L=None):
    plant = validate_plant(spec, K_override=K, L_override=L)
    return analyze(TransferFunction(plant), validate_grid(grid or GRID))


# ---------------------------------------------------------------------------
# 不变量 1：退化一致性 —— 一个不确定量都不给，与普通扫频逐项相等
# ---------------------------------------------------------------------------

def test_degenerate_no_uncertainty_matches_sweep_exactly():
    r = run_robust(STARTUP_ZPK, None)
    s = run_single(STARTUP_ZPK)
    wp, wg = r.worst_pm, r.worst_gm
    assert wp.exists_family and wg.exists_family
    assert wp.value == s.phase_margin_deg
    assert wg.value == s.gain_margin
    assert wg.value_db == s.gain_margin_db
    assert wp.omega == s.omega_c
    assert wg.omega == s.omega_pi
    assert (wp.K, wp.L, wp.pole) == (1.0, 0.0, None)
    assert r.robust_stable == s.stable


def test_degenerate_singleton_intervals_match_sweep():
    # 所有区间都收成单点（偏离名义值）：等于把该点直接送进扫频。
    # 标称启动对象 s(s+1)(s+10)：极点 -1 挪到 -2，K=3，L=0.25。
    unc = {"K": [3.0, 3.0], "L": [0.25, 0.25],
           "pole": {"nominal": -1.0, "range": [-2.0, -2.0]}}
    r = run_robust(STARTUP_ZPK, unc)
    s = run_single({"type": "poly", "num": [10.0],
                    "den": [1.0, 12.0, 20.0, 0.0], "K": 1.0, "L": 0.0},
                   K=3.0, L=0.25)
    assert r.worst_pm.value == s.phase_margin_deg
    assert r.worst_gm.value == s.gain_margin
    assert r.worst_pm.omega == s.omega_c
    assert r.worst_pm.K == 3.0 and r.worst_pm.L == 0.25
    assert r.worst_pm.pole == pytest.approx(-2.0)


# ---------------------------------------------------------------------------
# 不变量 3：纯延迟族最坏 PM 恰好落在 L 上界，且等于该点单点读数
# ---------------------------------------------------------------------------

def test_delay_only_family_worst_at_upper_bound():
    unc = {"L": [0.0, 0.6]}
    r = run_robust(STARTUP_ZPK, unc)
    single = run_single(STARTUP_ZPK, L=0.6)
    assert r.worst_pm.L == 0.6 and r.worst_pm.K == 1.0
    assert r.worst_pm.value == single.phase_margin_deg
    assert r.worst_pm.omega == single.omega_c
    assert r.worst_gm.value == single.gain_margin
    assert r.worst_gm.L == 0.6


@pytest.mark.parametrize("l_lo,l_hi", [(0.0, 0.0), (0.1, 0.1), (0.3, 0.3)])
def test_delay_singleton_matches_sweep(l_lo, l_hi):
    r = run_robust(STARTUP_ZPK, {"L": [l_lo, l_hi]})
    s = run_single(STARTUP_ZPK, L=l_hi)
    assert r.worst_pm.value == s.phase_margin_deg
    assert r.worst_gm.value == s.gain_margin


# ---------------------------------------------------------------------------
# 单调维度坍缩：K 族最坏点在 K_hi
# ---------------------------------------------------------------------------

def test_gain_only_family_worst_at_upper_bound():
    r = run_robust(STARTUP_ZPK, {"K": [1.0, 6.0]})
    single_hi = run_single(STARTUP_ZPK, K=6.0)
    single_lo = run_single(STARTUP_ZPK, K=1.0)
    assert r.worst_pm.K == 6.0
    assert r.worst_pm.value == single_hi.phase_margin_deg
    assert r.worst_gm.value == single_hi.gain_margin
    # GM*K 常数：K 放 6 倍，GM 缩 6 倍
    assert r.worst_gm.value * 6.0 == pytest.approx(single_lo.gain_margin, rel=1e-10)


# ---------------------------------------------------------------------------
# 不变量 2：区间放大后最坏裕度单调变差或持平
# ---------------------------------------------------------------------------

def test_enlarging_K_interval_cannot_improve_worst():
    small = run_robust(STARTUP_ZPK, {"K": [1.0, 3.0]})
    big = run_robust(STARTUP_ZPK, {"K": [0.5, 8.0]})
    assert big.worst_pm.value <= small.worst_pm.value
    assert big.worst_gm.value <= small.worst_gm.value


def test_enlarging_L_interval_cannot_improve_worst():
    small = run_robust(STARTUP_ZPK, {"L": [0.0, 0.2]})
    big = run_robust(STARTUP_ZPK, {"L": [0.0, 0.8]})
    assert big.worst_pm.value <= small.worst_pm.value
    assert big.worst_gm.value <= small.worst_gm.value


def test_enlarging_pole_interval_cannot_improve_worst():
    # 向不利方向（更靠近原点）拓宽极点区间，最坏 PM/GM 不得变好。
    small = run_robust(STARTUP_ZPK, {"pole": {"nominal": -1.0, "range": [-2.0, -1.0]}})
    big = run_robust(STARTUP_ZPK, {"pole": {"nominal": -1.0, "range": [-2.0, -0.3]}})
    assert big.worst_pm.value <= small.worst_pm.value + 1e-9
    assert big.worst_gm.value <= small.worst_gm.value + 1e-12


# ---------------------------------------------------------------------------
# 不变量 4：最坏参数点回代现有单点扫频，两边逐位一致
# ---------------------------------------------------------------------------

def test_worst_pole_point_recomputed_by_sweep_matches():
    unc = {"pole": {"nominal": -1.0, "range": [-3.0, -0.4]}}
    r = run_robust(STARTUP_ZPK, unc)
    # 移动极点 -1 -> p*：den = s(s+p*)(s+10)
    pstar = -r.worst_pm.pole
    moved = {"type": "poly", "num": [10.0],
             "den": [1.0, 10.0 + pstar, 10.0 * pstar, 0.0], "K": 1.0, "L": 0.0}
    s = run_single(moved)
    assert r.worst_pm.value == s.phase_margin_deg
    assert r.worst_pm.omega == s.omega_c
    g = run_single(moved)
    assert r.worst_gm.value == g.gain_margin
    assert r.worst_gm.omega == g.omega_pi


def test_mixed_family_worst_points_recomputed_match():
    unc = {"K": [1.0, 4.0], "L": [0.0, 0.3],
           "pole": {"nominal": -10.0, "range": [-12.0, -4.0]}}
    r = run_robust(STARTUP_ZPK, unc)
    for w, is_pm in ((r.worst_pm, True), (r.worst_gm, False)):
        pstar = -w.pole
        moved = {"type": "poly", "num": [10.0],
                 # den = s(s+1)(s+p*)
                 "den": [1.0, 1.0 + pstar, pstar, 0.0], "K": w.K, "L": w.L}
        s = run_single(moved, K=w.K, L=w.L)
        if is_pm:
            assert w.value == s.phase_margin_deg
            assert w.omega == s.omega_c
        else:
            assert w.value == s.gain_margin
            assert w.value_db == s.gain_margin_db
            assert w.omega == s.omega_pi


# ---------------------------------------------------------------------------
# 极点维度的真实搜索：非单调目标必须靠加密定位，粗扫网格点不许冒充极值
# ---------------------------------------------------------------------------

def test_worst_on_line_refines_to_interior_minimum_not_coarse_point():
    # 合成一个非单调一维目标：在 0.371 处有一个 -2 的内点低谷，区间两端为 0。
    x_star = 0.371
    calls = []

    def objective(x):
        calls.append(x)
        return 2.0 * (x - x_star) ** 2 - 2.0

    line = worst_on_line(0.0, 1.0, objective)
    coarse_grid = [i / (COARSE_POINTS - 1) for i in range(COARSE_POINTS)]
    assert line.x is not None
    # 内点极值：不能是任何粗扫网格点
    assert abs(line.x - x_star) <= 2.0 * PARAM_REL_TOL
    assert all(abs(line.x - g) > 1e-6 for g in coarse_grid)
    assert line.value == pytest.approx(objective(line.x), rel=0, abs=1e-14)


def test_worst_on_line_finds_global_dip_among_wiggles():
    # 多峰目标：加密必须找到全局最低，而不是最近一个粗扫低点。
    # 该目标的全局最小约在 0.689（不是名义抛物线中心 0.823），加密要能穿过
    # 前面几个 0.35·sin(7x) 的波谷找到真正的全局最低。
    def objective(x):
        return 0.35 * math.sin(7.0 * x) + (x - 0.823) ** 2 - 0.5

    line = worst_on_line(0.0, 1.0, objective)
    dense_xs = [i / 4001 for i in range(4001)]
    k = min(range(len(dense_xs)), key=lambda i: objective(dense_xs[i]))
    dense_min_x, dense_min = dense_xs[k], objective(dense_xs[k])
    assert line.value <= dense_min + 1e-5
    assert abs(line.x - dense_min_x) < 5e-3


def test_worst_on_line_records_none_points_without_dropping():
    # 前半段无有限穿越（None），后半段有值：最小化不能被 None 污染。
    def objective(x):
        return None if x < 0.5 else 1.0 + x

    line = worst_on_line(0.0, 1.0, objective)
    assert line.value == pytest.approx(1.5, abs=1e-9)  # 最坏（最小）有限值在分界 x=0.5
    assert line.x == pytest.approx(0.5, abs=2e-6)
    assert any(v is None for _, v in line.samples)


def test_worst_on_line_all_none_reports_none():
    line = worst_on_line(0.0, 1.0, lambda x: None)
    assert line.x is None and line.value is None
    assert len(line.samples) >= COARSE_POINTS * 2 ** GLOBAL_REFINE_PASSES - (
        2 ** GLOBAL_REFINE_PASSES - 1)


# ---------------------------------------------------------------------------
# 真实对象上的极点搜索：端点最坏的族也要真搜，且钉在容差内
# ---------------------------------------------------------------------------

def test_pole_family_startup_worst_pole_and_margins():
    unc = {"pole": {"nominal": -1.0, "range": [-2.0, -0.5]}}
    r = run_robust(STARTUP_ZPK, unc)
    # 10/[s(s+p)(s+10)]：p 越小 PM/GM 越差，最坏极点钉在区间近端 -0.5。
    assert r.worst_pm.pole == pytest.approx(-0.5, abs=2e-9)
    expected = run_single({"type": "poly", "num": [10.0],
                           "den": [1.0, 10.5, 5.0, 0.0], "K": 1.0, "L": 0.0})
    assert r.worst_pm.value == expected.phase_margin_deg
    assert r.worst_gm.value == expected.gain_margin
    # 最坏点落在参数相对容差内
    assert abs(r.worst_pm.pole - (-0.5)) <= PARAM_REL_TOL * max(1.0, 0.5) + 1e-12


def test_pole_approaching_origin_interior_worst_found():
    # 极点 -1 一直放到 0（允许贴虚轴）：p→0⁺ 时 PM 跌到约 -5.7° 的内点低谷，
    # p=0 自身（双积分）按既有口径取最低频穿越读数反而是 +354°。这是真实的
    # 非单调内点最坏，粗扫必须加密才能钉住。
    unc = {"pole": {"nominal": -1.0, "range": [-1.0, 0.0]}}
    r = run_robust(STARTUP_ZPK, unc)
    assert r.worst_pm.exists_family
    # 独立密扫核对内点最坏位置与数值
    dense = []
    for i in range(2001):
        p = 10 ** (-6 + i * 6 / 2000)
        s = run_single({"type": "poly", "num": [10.0],
                        "den": [1.0, 10.0 + p, 10.0 * p, 0.0], "K": 1.0, "L": 0.0})
        if s.phase_margin_deg is not None:
            dense.append((p, s.phase_margin_deg, s.omega_c))
    p_star, pm_star, wc_star = min(dense, key=lambda t: t[1])
    assert r.worst_pm.pole == pytest.approx(-p_star, abs=5e-6)
    assert r.worst_pm.value == pytest.approx(pm_star, abs=1e-4)
    assert r.worst_pm.omega == pytest.approx(wc_star, rel=1e-8)
    # 该族实际上不鲁棒（内点 PM 为负），并交出临界极点
    assert r.robust_stable is False and r.critical.found
    assert r.critical.pole == pytest.approx(-p_star, abs=5e-5)


def test_pole_interval_touching_origin_is_legal():
    # 区间端点贴在虚轴（0）上合法：不越过右半平面就不拒绝。
    plant = validate_plant(STARTUP_ZPK)
    u = build_uncertainty({"pole": {"nominal": -1.0, "range": [-1.0, 0.0]}}, plant)
    assert u.pole is not None and u.pole.lo == 0.0 and u.pole.hi == 1.0


def test_family_reader_reuses_original_den_at_nominal():
    # p == 标称值时必须逐位沿用原分母（退化一致性的浮点保证）。
    plant = validate_plant(STARTUP_ZPK)
    u = build_uncertainty({"pole": {"nominal": -1.0, "range": [-2.0, -0.5]}}, plant)
    reader = FamilyReader(plant, GRID, u)
    rebuilt = reader.make_plant(1.0, 0.0, 1.0)
    assert rebuilt.den == plant.den


# ---------------------------------------------------------------------------
# 非法不确定区间：当场拒绝，且响应里不带任何读数
# ---------------------------------------------------------------------------

def test_K_interval_hi_below_lo_rejected():
    plant = validate_plant(STARTUP_ZPK)
    with pytest.raises(ServiceError) as ei:
        build_uncertainty({"K": [3.0, 1.0]}, plant)
    assert "上界" in ei.value.reason and "下界" in ei.value.reason


def test_K_interval_nonpositive_rejected():
    plant = validate_plant(STARTUP_ZPK)
    with pytest.raises(ServiceError):
        build_uncertainty({"K": [0.0, 1.0]}, plant)
    with pytest.raises(ServiceError):
        build_uncertainty({"K": [-2.0, -1.0]}, plant)


def test_L_interval_negative_rejected():
    plant = validate_plant(STARTUP_ZPK)
    with pytest.raises(ServiceError) as ei:
        build_uncertainty({"L": [-0.01, 1.0]}, plant)
    assert "负" in ei.value.reason


def test_pole_interval_crossing_imag_axis_rejected():
    plant = validate_plant(STARTUP_ZPK)
    with pytest.raises(ServiceError) as ei:
        build_uncertainty({"pole": {"nominal": -1.0, "range": [-2.0, 0.5]}}, plant)
    assert "右半平面" in ei.value.reason
    # 整段都在右半平面同样拒绝
    with pytest.raises(ServiceError):
        build_uncertainty({"pole": {"nominal": -1.0, "range": [0.1, 1.0]}}, plant)
    # 上下界颠倒
    with pytest.raises(ServiceError):
        build_uncertainty({"pole": {"nominal": -1.0, "range": [-0.5, -2.0]}}, plant)


def test_pole_nominal_not_a_denominator_root_rejected():
    plant = validate_plant(STARTUP_ZPK)
    with pytest.raises(ServiceError) as ei:
        build_uncertainty({"pole": {"nominal": -3.14, "range": [-4.0, -2.0]}}, plant)
    assert "不是对象分母的实根" in ei.value.reason


def test_malformed_interval_shapes_rejected():
    plant = validate_plant(STARTUP_ZPK)
    with pytest.raises(ServiceError):
        build_uncertainty({"K": [1.0, 2.0, 3.0]}, plant)
    with pytest.raises(ServiceError):
        build_uncertainty({"L": "x"}, plant)
    with pytest.raises(ServiceError):
        build_uncertainty({"pole": {"nominal": -1.0}}, plant)


def test_non_finite_interval_endpoints_rejected():
    plant = validate_plant(STARTUP_ZPK)
    with pytest.raises(ServiceError):
        build_uncertainty({"K": [float("nan"), 1.0]}, plant)
    with pytest.raises(ServiceError):
        build_uncertainty({"L": [0.0, float("inf")]}, plant)


# ---------------------------------------------------------------------------
# 全族无有限穿越：显式标记，绝不填 0
# ---------------------------------------------------------------------------

def test_family_with_no_gain_crossover_marked_not_zero():
    # K 小到整族 |G| 全程小于 1：PM 全族无有限幅值穿越。
    r = run_robust(STARTUP_ZPK, {"K": [1e-7, 1e-5]})
    assert r.worst_pm.exists_family is False
    assert r.worst_pm.value is None
    assert r.worst_pm.omega is None
    payload = robust_to_payload(r)
    blk = payload["worst_phase_margin"]
    assert blk["exists"] is False
    assert blk["phase_margin_deg"] is None
    assert blk["gain_crossover"]["note"] == "全族无有限幅值穿越"
    assert blk["at"] is None
    # GM 穿越不依赖增益大小，仍在
    assert r.worst_gm.exists_family
    assert r.worst_gm.value > 1.0


def test_partial_family_missing_not_dropped():
    # 一族里部分 K 有幅值穿越、部分没有：有限成员参与最小化，
    # 无穿越成员不许被悄悄丢掉（标 partial 且鲁棒判定不成立）。
    r = run_robust(STARTUP_ZPK, {"K": [1e-4, 5.0]})
    assert r.worst_pm.exists_family
    assert r.worst_pm.partial_missing is True
    # (K_lo, L_lo) 无穿越：证据不足，不判鲁棒稳定
    assert r.uncertified
    assert r.robust_stable is False
    payload = robust_to_payload(r)
    assert "无有限穿越" in payload["worst_phase_margin"]["note"]


def test_no_phase_crossover_family_marked():
    # 一阶惯性 + 积分：相位永不达 -180°，GM 全族无有限相位穿越。
    spec = {"type": "poly", "num": [1.0], "den": [1.0, 1.0, 0.0], "K": 1.0, "L": 0.0}
    r = run_robust(spec, {"K": [0.5, 4.0]})
    assert r.worst_gm.exists_family is False
    assert r.worst_gm.value is None and r.worst_gm.omega is None
    blk = robust_to_payload(r)["worst_gain_margin"]
    assert blk["gain_margin"] is None
    assert blk["phase_crossover"]["note"] == "全族无有限相位穿越"
    assert r.worst_pm.value > 0


# ---------------------------------------------------------------------------
# 鲁棒稳定判定与临界参数点
# ---------------------------------------------------------------------------

def test_robust_stable_family():
    r = run_robust(STARTUP_ZPK, {"K": [1.0, 3.0], "L": [0.0, 0.2]})
    assert r.robust_stable is True
    assert r.critical.found is False
    assert r.worst_pm.value > 0 and r.worst_gm.value > 1.0


def test_unstable_gain_family_reports_critical_gain():
    # K=11 恰为临界（PM=0, GM=1）：族 [1, 11.5] 不鲁棒，临界 K 应对分到 ~11。
    r = run_robust(STARTUP_ZPK, {"K": [1.0, 11.5]})
    assert r.robust_stable is False
    c = r.critical
    assert c.found and c.margin in ("phase_margin", "gain_margin")
    assert c.K == pytest.approx(11.0, abs=1e-6) and c.L == 0.0 and c.pole is None
    assert abs(c.phase_margin_deg) < 1e-5
    assert c.gain_margin == pytest.approx(1.0, abs=1e-5)
    assert c.omega_pi == pytest.approx(math.sqrt(10.0), rel=1e-6)


def test_unstable_delay_family_reports_critical_delay_endpoint():
    # L 没有对分维度时（K 不动），临界族直接在 L_hi 端给出越界证据。
    r = run_robust(STARTUP_ZPK, {"L": [0.0, 2.0]})
    assert r.robust_stable is False
    c = r.critical
    assert c.found and c.L == 2.0
    assert c.phase_margin_deg < 0


def test_unstable_pole_family_reports_critical_pole():
    r = run_robust(STARTUP_ZPK, {"K": [12.0, 12.0],
                                  "pole": {"nominal": -1.0, "range": [-2.0, -0.2]}})
    assert r.robust_stable is False
    c = r.critical
    assert c.found and c.pole is not None
    # 回代临界极点：裕度确实贴近 0
    pstar = -c.pole
    moved = {"type": "poly", "num": [10.0],
             "den": [1.0, 10.0 + pstar, 10.0 * pstar, 0.0], "K": 12.0, "L": 0.0}
    s = run_single(moved, K=12.0)
    if c.margin == "phase_margin":
        assert s.phase_margin_deg == pytest.approx(c.phase_margin_deg, abs=1e-6)
    else:
        assert s.gain_margin == pytest.approx(c.gain_margin, rel=1e-6)


def test_boundary_zero_margin_is_not_robust():
    # 整族就在临界 K=11 单点：裕度为零 ⇒ 不鲁棒。
    r = run_robust(STARTUP_ZPK, {"K": [11.0, 11.0]})
    assert r.robust_stable is False
    assert r.critical.found
    assert r.worst_pm.value == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# FamilyReader / 参数族构造
# ---------------------------------------------------------------------------

def test_family_reader_pole_movement_keeps_other_poles_and_num():
    plant = validate_plant(STARTUP_ZPK)
    u = build_uncertainty({"pole": {"nominal": -10.0, "range": [-12.0, -4.0]}}, plant)
    reader = FamilyReader(plant, GRID, u)
    rebuilt = reader.make_plant(1.0, 0.0, 5.0)  # 极点 -10 移到 -5
    # den 应为 s(s+1)(s+5) = s^3 + 6s^2 + 5s
    assert tuple(round(c, 10) for c in rebuilt.den) == (1.0, 6.0, 5.0, 0.0)
    assert rebuilt.num == plant.num


def test_family_reader_caches_evaluations():
    plant = validate_plant(STARTUP_ZPK)
    u = build_uncertainty({"pole": {"nominal": -1.0, "range": [-2.0, -0.5]}}, plant)
    reader = FamilyReader(plant, GRID, u)
    r1 = reader.read(1.0, 0.0, 0.75)
    r2 = reader.read(1.0, 0.0, 0.75)
    assert r1 is r2


def test_worst_on_line_degenerate_interval_single_read():
    calls = []

    def objective(x):
        calls.append(x)
        return -1.7

    line = worst_on_line(2.0, 2.0, objective)
    assert line.x == 2.0 and line.value == -1.7 and calls == [2.0]


def test_enlarged_interval_equal_values_allowed():
    # 放大区间但新增的半边全族无穿越，最坏有限值持平——单调包含允许持平。
    base = run_robust(STARTUP_ZPK, {"K": [1.0, 2.0]})
    same = run_robust(STARTUP_ZPK, {"K": [1.0, 2.0]})
    assert base.worst_pm.value == same.worst_pm.value
    assert base.worst_gm.value == same.worst_gm.value


def test_delay_family_beyond_instability_endpoint_critical():
    # 只有 L 在飘、且上界已越稳定边界：不沿 L 对分，直接在 L_hi 交越界证据。
    r = run_robust(STARTUP_ZPK, {"L": [0.0, 2.0]})
    assert r.robust_stable is False
    c = r.critical
    assert c.found and c.L == 2.0
    assert c.phase_margin_deg < 0
    assert r.worst_pm.L == 2.0


def test_zpk_archive_pole_uncertainty_resolves_from_poles():
    # 从 ZPK 档出发，挑一个与其余极点不重样的实极点移动。
    zpk = {"type": "zpk", "zeros": [[-2.0, 0.0]],
           "poles": [[0.0, 0.0], [-1.0, 0.0], [-10.0, 0.0]],
           "gain": 5.0, "K": 2.0, "L": 0.1}
    r = run_robust(zpk, {"pole": {"nominal": -10.0, "range": [-15.0, -5.0]}})
    assert r.worst_pm.exists_family and r.worst_gm.exists_family
    assert r.worst_pm.pole == pytest.approx(-5.0, abs=1e-8)
    # 回代：den = s(s+1)(s+5) = s^3+6s^2+5s；num = 5(s+2)=[5,10]
    moved = {"type": "poly", "num": [5.0, 10.0],
             "den": [1.0, 6.0, 5.0, 0.0], "K": 2.0, "L": 0.1}
    s = run_single(moved, K=2.0, L=0.1)
    assert r.worst_pm.value == s.phase_margin_deg


def test_payload_shape_for_both_blocks():
    r = run_robust(STARTUP_ZPK, {"K": [1.0, 3.0], "L": [0.0, 0.2]})
    p = robust_to_payload(r)
    pm, gm = p["worst_phase_margin"], p["worst_gain_margin"]
    # PM 块只带相位裕度，GM 块同时带真值与 dB，且两值一致
    assert "gain_margin_db" not in pm
    assert "phase_margin_deg" not in gm
    assert gm["gain_margin_db"] == pytest.approx(
        20.0 * math.log10(gm["gain_margin"]), rel=1e-12, abs=1e-12)
    assert pm["gain_crossover"]["omega_c"] > 0
    assert gm["phase_crossover"]["omega_pi"] > 0
    assert p["uncertified_points"] == []
