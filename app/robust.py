"""参数族上的最坏情况裕度搜索与鲁棒稳定判定。

单调维度的坍缩（本服务约定并利用的剪枝方向，不做全区间傻扫）：
  * 纯延迟只压相位：PM 随 L 单调下降；相位穿越随 L 提前、GM 随 L 下降
    → 两个裕度的最坏点都在 L 区间上界；
  * 增益放大把幅值穿越推向高频、压低 PM；GM = 1/(K·|G₀(jω_π)|) 随 K 严格
    下降 → 两个裕度的最坏点都在 K 区间上界；
  * 极点位置非单调：在区间上先粗扫定位最坏子区间，再黄金分割加密，
    直到参数相对容差 PARAM_REL_TOL，绝不拿粗扫网格点冒充极值点。

穿越覆盖标记（crossover_coverage）在极点粗网格 × 单调维度两端上评估：
幅值穿越存在性随 K 单调（K 越大越可能穿越，与 L 无关）、相位穿越存在性随
L 单调（L 越大越可能穿 −180°，与 K 无关）。因此「部分覆盖」只可能出现在
良性方向（增益偏小端无幅值穿越、延迟偏小端无相位穿越），最坏值落在有穿越
的子族上，无穿越成员不会被悄悄丢掉，而是显式体现在覆盖标记里。

鲁棒稳定判定：最坏 PM 与最坏 GM(dB) 都存在且严格为正才判鲁棒稳定；
任一最坏裕度 ≤ 0 时交出该最坏参数点，作为设计需要收紧的第一处越界证据。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

from .frf import TransferFunction
from .margins import MarginResult, analyze
from .uncertainty import PlantFamily, Uncertainty
from .validation import Plant

# 最坏化搜索把参数钉到的相对容差（服务约定）。
PARAM_REL_TOL = 1e-6
# 极点区间粗扫点数；黄金分割加密的轮数上限（0.618^60 ≈ 3e-13，足够收敛）。
COARSE_SAMPLES = 25
MAX_REFINE_ROUNDS = 60


@dataclass
class WorstMargin:
    """一个裕度在整个参数族上的最坏情况读数。"""

    kind: str                                # "phase" | "gain"
    exists: bool                             # 最坏值是否存在（全族无穿越时为 False）
    value: float | None                      # PM（度）或 GM（真值）
    value_db: float | None                   # GM 分贝（PM 恒为 None）
    params: dict[str, float | None] | None   # 取到最坏值的参数点 {"K","L","pole"}
    omega: float | None                      # 该点处的 ω_c 或 ω_π
    coverage: str                            # "all" | "partial" | "none"


@dataclass
class RobustResult:
    worst_phase_margin: WorstMargin
    worst_gain_margin: WorstMargin
    robustly_stable: bool
    first_violation: dict[str, Any] | None


def _minimize_1d(func: Callable[[float], float | None], lo: float, hi: float,
                 rel_tol: float = PARAM_REL_TOL) -> tuple[float | None, float | None]:
    """在 [lo, hi] 上求 func 的最小值点。

    func 返回 None 表示该点无有限穿越（按 +∞ 处理，绝不会被选为最坏点）。
    先 COARSE_SAMPLES 点粗扫定位最坏子区间，再在子区间内黄金分割加密，
    直到子区间相对宽度 <= rel_tol。返回 (x_best, f_best)；
    整段都取不到有限值时返回 (None, None)。
    """

    def value(x: float) -> float:
        v = func(x)
        return math.inf if v is None else v

    if lo == hi:
        v = value(lo)
        return (lo, v) if math.isfinite(v) else (None, None)

    n = COARSE_SAMPLES
    xs = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    fs = [value(x) for x in xs]
    i_best = min(range(n), key=lambda i: fs[i])
    if not math.isfinite(fs[i_best]):
        return None, None
    best_x, best_f = xs[i_best], fs[i_best]
    # 最坏子区间：粗扫最坏点的左右邻点之间（端点处取一侧）。
    a = xs[i_best - 1] if i_best > 0 else xs[0]
    b = xs[i_best + 1] if i_best < n - 1 else xs[-1]

    invphi = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc, fd = value(c), value(d)
    for _ in range(MAX_REFINE_ROUNDS):
        mid = 0.5 * (a + b)
        if abs(b - a) <= rel_tol * max(1.0, abs(mid)):
            break
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = value(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = value(d)
        if fc < best_f:
            best_x, best_f = c, fc
        if fd < best_f:
            best_x, best_f = d, fd
    return best_x, best_f


def _analyze_at(family: PlantFamily, freqs: list[float],
                K: float, L: float, pole: float | None) -> MarginResult:
    """族内一点的单点读数：与 /api/sweep 共用同一个 analyze 内核。"""

    return analyze(TransferFunction(family.plant_at(K, L, pole)), freqs)


def _margin_value(r: MarginResult, kind: str) -> float | None:
    return r.phase_margin_deg if kind == "phase" else r.gain_margin


def _pole_sample_points(family: PlantFamily) -> list[float | None]:
    interval = family.pole_interval
    if interval is None:
        return [None]
    lo, hi = interval
    if lo == hi:
        return [lo]
    n = COARSE_SAMPLES
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def _coverage(family: PlantFamily, freqs: list[float], kind: str) -> str:
    """全族穿越覆盖：'all' 全族有穿越 / 'none' 全族无 / 'partial' 部分有。

    利用单调性只查两端：幅值穿越存在性随 K 单调（与 L 无关），相位穿越
    存在性随 L 单调（与 K 无关）；极点维度在粗网格上采样评估。
    """

    pts = _pole_sample_points(family)
    if kind == "phase":
        attr, mild, worst = "has_finite_gain_crossover", family.K_mild, family.K_worst
        fixed = family.L_worst
    else:
        attr, mild, worst = "has_finite_phase_crossover", family.L_mild, family.L_worst
        fixed = family.K_worst

    def exists_at(mono: float, pole: float | None) -> bool:
        K, L = (mono, fixed) if kind == "phase" else (fixed, mono)
        return getattr(_analyze_at(family, freqs, K, L, pole), attr)

    if all(exists_at(mild, p) for p in pts):
        return "all"
    return "partial" if any(exists_at(worst, p) for p in pts) else "none"


def _reading_to_worst(kind: str, r: MarginResult,
                      params: dict[str, float | None], coverage: str) -> WorstMargin:
    if kind == "phase":
        return WorstMargin("phase", r.phase_margin_deg is not None,
                           r.phase_margin_deg, None, params, r.omega_c, coverage)
    return WorstMargin("gain", r.gain_margin is not None,
                       r.gain_margin, r.gain_margin_db, params, r.omega_pi, coverage)


def _worst_margin(family: PlantFamily, freqs: list[float],
                  kind: str, coverage: str) -> WorstMargin:
    """一个裕度的最坏化：单调维度坍缩到上界，极点维度做一维最坏化搜索。"""

    K, L = family.K_worst, family.L_worst
    interval = family.pole_interval
    if interval is None:
        r = _analyze_at(family, freqs, K, L, None)
        return _reading_to_worst(kind, r, {"K": K, "L": L, "pole": None}, coverage)
    lo, hi = interval
    x, _ = _minimize_1d(
        lambda p: _margin_value(_analyze_at(family, freqs, K, L, p), kind), lo, hi
    )
    if x is None:
        # 全族（该裕度）无有限穿越：照现有口径标记，绝不填 0 或大数。
        return WorstMargin(kind, False, None, None, None, None, coverage)
    # 在最坏参数点上用同一个内核重读一次，保证与单点扫频口径一致。
    r = _analyze_at(family, freqs, K, L, x)
    return _reading_to_worst(kind, r, {"K": K, "L": L, "pole": x}, coverage)


def robust_analyze(family: PlantFamily, freqs: list[float]) -> RobustResult:
    """在整个参数族上求最坏 PM / 最坏 GM，并据此判定鲁棒稳定性。"""

    worst_pm = _worst_margin(family, freqs, "phase", _coverage(family, freqs, "phase"))
    worst_gm = _worst_margin(family, freqs, "gain", _coverage(family, freqs, "gain"))

    # 只有两个最坏裕度都存在且严格为正，才判这一族鲁棒稳定。
    pm_ok = worst_pm.value is not None and worst_pm.value > 0.0
    gm_ok = worst_gm.value_db is not None and worst_gm.value_db > 0.0
    stable = pm_ok and gm_ok

    # 第一处越界：任一最坏裕度落到零或负，交出该最坏参数点作为证据。
    violation = None
    if worst_pm.value is not None and worst_pm.value <= 0.0:
        violation = {
            "margin": "phase_margin",
            "params": worst_pm.params,
            "value_deg": worst_pm.value,
            "omega_c": worst_pm.omega,
        }
    elif worst_gm.value_db is not None and worst_gm.value_db <= 0.0:
        violation = {
            "margin": "gain_margin",
            "params": worst_gm.params,
            "value": worst_gm.value,
            "value_db": worst_gm.value_db,
            "omega_pi": worst_gm.omega,
        }
    return RobustResult(worst_pm, worst_gm, stable, violation)


def _uncertainty_payload(unc: Uncertainty) -> dict[str, Any]:
    return {
        "K": None if unc.K is None else {"min": unc.K.lo, "max": unc.K.hi},
        "L": None if unc.L is None else {"min": unc.L.lo, "max": unc.L.hi},
        "pole": None if unc.pole is None else {
            "nominal": unc.pole.nominal, "min": unc.pole.lo, "max": unc.pole.hi,
        },
    }


def _margin_payload(m: WorstMargin) -> dict[str, Any]:
    if m.kind == "phase":
        return {
            "exists": m.exists,
            "value_deg": m.value,
            "params": m.params,
            "omega_c": m.omega,
            "crossover_coverage": m.coverage,
            "note": None if m.exists else "全族无有限幅值穿越",
        }
    return {
        "exists": m.exists,
        "value": m.value,
        "value_db": m.value_db,
        "params": m.params,
        "omega_pi": m.omega,
        "crossover_coverage": m.coverage,
        "note": None if m.exists else "全族无有限相位穿越",
    }


def to_robust_payload(result: RobustResult, base: Plant, unc: Uncertainty) -> dict[str, Any]:
    """转成对外 JSON：无有限穿越时显式标记 + null，绝不填 0 或大数。"""

    return {
        "plant": {"num": list(base.num), "den": list(base.den), "K": base.K, "L": base.L},
        "uncertainty": _uncertainty_payload(unc),
        "worst_phase_margin": _margin_payload(result.worst_phase_margin),
        "worst_gain_margin": _margin_payload(result.worst_gain_margin),
        "robustly_stable": result.robustly_stable,
        "first_violation": result.first_violation,
    }
