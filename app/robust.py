"""参数族上的最坏情况裕度与鲁棒稳定判定。

三类允许飘动的量（都只在实轴/标量上变化）：
  * 开环增益 K ∈ [K_lo, K_hi]，区间为正；
  * 纯延迟 L ∈ [L_lo, L_hi]，区间非负（下界允许为 0）；
  * 对象的某一个实极点 -p（p ≥ 0）在区间 [p_lo, p_hi] 内移动，
    其余零极点保持标称不动；区间不允许把极点扫进虚轴右半平面。

单调性剪枝（复用既有频响/穿越内核即可验证的关系）：
  * 延迟只加相位滞后：L↑ 使相位裕度单调下降；对最小相位对象，加长延迟只会让
    −180° 穿越向低频推、|G(jω_π)| 单调变大，故幅值裕度对 L 也单调下降。
    ⇒ 两个裕度的 L 维度都坍缩到 L_hi。
  * 增益等比放大幅频：GM·K 在固定极点/延迟下为常数；增益穿越随 K↑ 推向高频，
    相位裕度单调下降。⇒ K 维度对两个裕度都坍缩到 K_hi。
  * 极点位置不保证单调（例如带超前零点时 PM(p) 可出现内点极大），必须在它的
    区间上真做一维最坏化搜索：先粗扫定位最坏子区间，再在子区间内对分加密到
    PARAM_REL_TOL 钉死最坏点，绝不拿粗扫网格点冒充极值。

分层：参数族的构造与搜索只调用 validation.Plant、frf.TransferFunction 和
margins.analyze —— 不复制任何扫频/穿越逻辑，最坏点回代与逐点口径天然一致。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

from .errors import ServiceError
from .frf import TransferFunction
from .margins import analyze
from .validation import Plant, _as_finite_number

# 极点维度的参数相对容差：|br-bl|/max(1,|p|) 小到该值才停止加密。
PARAM_REL_TOL = 1e-8
# 初始粗扫点数（含区间两端，端点必须在网格里——单调维度之外，角落不能漏）。
COARSE_POINTS = 25
# 粗扫后全区间加密的轮数（每轮在每个相邻点间插入一个中点）。
GLOBAL_REFINE_PASSES = 2
# 最坏子区间局部加密的最大轮数（1e-8 相对容差约 17 轮，给到 80 足够保险）。
MAX_LOCAL_ZOOM = 80
# 标称极点必须是分母的根：残差 |den(p_nom)| 相对系数尺度的容差。
_ROOT_RESIDUAL_TOL = 1e-8
# 裕度的数值零界：绝对量小于该值视为「贴到零」，不按严格为正放行。
ZERO_MARGIN_TOL = 1e-7


# ---------------------------------------------------------------------------
# 不确定性的结构
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float

    @property
    def degenerate(self) -> bool:
        return self.lo == self.hi


@dataclass(frozen=True)
class PoleUncertainty:
    """移动的实极点 -nominal（标称值必须是分母的根），-p 在 -hi..-lo 间移动。"""

    nominal: float
    interval: Interval  # 界的是 p（极点 = -p，p >= 0）

    @property
    def lo(self) -> float:
        return self.interval.lo

    @property
    def hi(self) -> float:
        return self.interval.hi


@dataclass(frozen=True)
class Uncertainty:
    K: Interval | None
    L: Interval | None
    pole: PoleUncertainty | None

    def any(self) -> bool:
        return self.K is not None or self.L is not None or self.pole is not None


def _interval(value: Any, what: str, *, positive: bool, nonnegative: bool) -> Interval:
    if not isinstance(value, list) or len(value) != 2:
        raise ServiceError(f"{what}的不确定区间必须是 [下界, 上界] 二元数组")
    lo = _as_finite_number(value[0], f"{what}区间下界")
    hi = _as_finite_number(value[1], f"{what}区间上界")
    if hi < lo:
        raise ServiceError(f"{what}区间非法：上界 {hi} 小于下界 {lo}")
    if positive and lo <= 0.0:
        raise ServiceError(f"{what}区间必须为正，收到下界 {lo}")
    if nonnegative and lo < 0.0:
        raise ServiceError(f"{what}区间不得含负值，收到下界 {lo}")
    return Interval(lo, hi)


def _poly_value(coeffs: tuple[float, ...], x: float) -> float:
    """降幂多项式在实轴点 x 上的霍纳求值。"""

    acc = 0.0
    for c in coeffs:
        acc = acc * x + c
    return acc


def _synthetic_divide_linear(den: tuple[float, ...], r: float) -> tuple[tuple[float, ...], float]:
    """把降幂多项式 den(x) 除以一次因式 (x - r)，返回 (商式系数, 余数)。"""

    n = len(den)
    q = [0.0] * (n - 1)
    q[0] = den[0]
    for i in range(1, n - 1):
        q[i] = den[i] + r * q[i - 1]
    remainder = den[-1] + r * q[-1]
    return tuple(q), remainder


def build_uncertainty(spec: Any, plant: Plant) -> Uncertainty:
    """把 HTTP 层收下来的 uncertainty 说明检查并规范化。

    极点合法性（不能扫到虚轴右半平面外去改变最小相位属性）在这里当场拒绝：
    界的是极点到原点的距离 p（极点 = -p），p_hi <= 0 必须满足，即整段都不许
    越过虚轴。
    """

    if spec is None:
        return Uncertainty(K=None, L=None, pole=None)
    if not isinstance(spec, dict):
        raise ServiceError("不确定性说明 uncertainty 必须是对象")

    k_int = _interval(spec["K"], "开环增益 K", positive=True, nonnegative=False) \
        if "K" in spec else None
    l_int = _interval(spec["L"], "纯延迟 L", positive=False, nonnegative=True) \
        if "L" in spec else None

    pole_unc: PoleUncertainty | None = None
    if "pole" in spec:
        pspec = spec["pole"]
        if not isinstance(pspec, dict) or "nominal" not in pspec or "range" not in pspec:
            raise ServiceError(
                "极点不确定性必须写成 {\"nominal\": 标称极点, \"range\": [下界, 上界]}"
            )
        nominal_pole = _as_finite_number(pspec["nominal"], "标称极点 nominal")
        rng = _interval(pspec["range"], "极点位置", positive=False, nonnegative=False)
        # 区间给的是极点本身（实轴坐标，越左越负）。合法域是 (-∞, 0]。
        if rng.hi > 0.0:
            raise ServiceError(
                f"极点区间 [{rng.lo}, {rng.hi}] 会把极点扫进虚轴右半平面，"
                "改变对象的最小相位属性，当场拒绝"
            )
        # 标称极点必须确实是分母的一个实根。
        scale = max((abs(c) for c in plant.den), default=1.0)
        residual = abs(_poly_value(plant.den, nominal_pole))
        if residual > _ROOT_RESIDUAL_TOL * max(1.0, scale):
            raise ServiceError(
                f"标称极点 {nominal_pole} 不是对象分母的实根（残差 {residual:g}），"
                "无法在保持其余零极点不动的前提下移动它"
            )
        # 内部以距离 p（极点 = -p，p >= 0）为搜索变量。
        pole_unc = PoleUncertainty(nominal=nominal_pole,
                                   interval=Interval(lo=-rng.hi, hi=-rng.lo))

    return Uncertainty(K=k_int, L=l_int, pole=pole_unc)


# ---------------------------------------------------------------------------
# 参数族构造：给定 (K, L, p) 复用现有内核读一次完整裕度
# ---------------------------------------------------------------------------

class FamilyReader:
    """以标称对象为底，按 (K, L, p) 现造 Plant 并走 margins.analyze 读数。

    极点移动通过一次合成除法完成：den(s) = (s + p_nom)·q(s)，移动后
    den_p(s) = (s + p)·q(s)，q(s)（其余零极点）原样不动。p 正好等于标称值时
    直接沿用原分母，保证退化情形与单次扫频逐位相等。
    """

    def __init__(self, plant: Plant, grid: list[float], unc: Uncertainty):
        self.plant = plant
        self.grid = grid
        self.unc = unc
        if unc.pole is not None:
            r = unc.pole.nominal  # 极点坐标（<=0）
            self._quotient, remainder = _synthetic_divide_linear(plant.den, r)
            # build_uncertainty 已用根残差校验过；这里只是浮点自洽保护。
            assert abs(remainder) <= 1e-6 * max(1.0, max(abs(c) for c in plant.den))
            self._nominal_p = -r
        self._cache: dict[tuple[float, float, float], Any] = {}

    def make_plant(self, K: float, L: float, p: float) -> Plant:
        if self.unc.pole is None or p == self._nominal_p:
            den = self.plant.den
        else:
            # 乘回 (s + p)：降幂系数 [1, p] 与 q 卷积
            q = self._quotient
            d = [0.0] * (len(q) + 1)
            for i, c in enumerate(q):
                d[i] += c
                d[i + 1] += c * p
            den = tuple(d)
        return Plant(num=self.plant.num, den=den, K=K, L=L, form=self.plant.form)

    def read(self, K: float, L: float, p: float) -> Any:
        key = (K, L, p)
        if key not in self._cache:
            plant = self.make_plant(K, L, p)
            self._cache[key] = analyze(TransferFunction(plant), self.grid)
        return self._cache[key]


# ---------------------------------------------------------------------------
# 一维最坏化搜索（先粗扫定位最坏子区间，再局部对分加密）
# ---------------------------------------------------------------------------

@dataclass
class LineMinimum:
    x: float | None                  # 最坏点；整条线都给不出有限值时为 None
    value: float | None
    samples: list[tuple[float, float | None]] = field(default_factory=list)


def _linspace(lo: float, hi: float, n: int) -> list[float]:
    if n == 1:
        return [lo]
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def worst_on_line(lo: float, hi: float,
                  objective: Callable[[float], float | None],
                  extra: tuple[float, ...] = ()) -> LineMinimum:
    """在 [lo, hi] 上求 objective 的最小值（None 视为 +∞，但记录不漏点）。

    步骤：等距粗扫（含端点）→ 全区间插中点加密 GLOBAL_REFINE_PASSES 轮定位
    最坏子区间 → 在最坏点邻域按三等分点对分收缩，直到区间相对宽度小于
    PARAM_REL_TOL。退化区间（lo == hi）只在该单点读一次。
    """

    def near(a: float, b: float) -> bool:
        return abs(a - b) <= 1e-14 * max(1.0, abs(a), abs(b))

    if near(lo, hi):
        v = objective(lo)
        return LineMinimum(lo if v is not None else None, v, [(lo, v)])

    xs = _linspace(lo, hi, COARSE_POINTS)
    for x in extra:
        if not any(near(x, y) for y in xs):
            xs.append(x)
    xs.sort()
    vals = [objective(x) for x in xs]

    # 全区间加密：每轮在每对相邻点之间插一个中点，把真正的最坏子区间显出来。
    for _ in range(GLOBAL_REFINE_PASSES):
        nxs: list[float] = []
        nvals: list[float | None] = []
        for i, x in enumerate(xs):
            nxs.append(x)
            nvals.append(vals[i])
            if i + 1 < len(xs):
                m = 0.5 * (x + xs[i + 1])
                nxs.append(m)
                nvals.append(objective(m))
        xs, vals = nxs, nvals

    finite = [i for i, v in enumerate(vals) if v is not None]
    if not finite:
        return LineMinimum(None, None, list(zip(xs, vals)))

    bi = min(finite, key=lambda i: vals[i])
    bl = xs[bi - 1] if bi > 0 else lo
    br = xs[bi + 1] if bi + 1 < len(xs) else hi

    def record(x: float, v: float | None) -> None:
        xs.append(x)
        vals.append(v)

    def fval(x: float) -> float:
        v = objective(x)
        record(x, v)
        return v if v is not None else math.inf

    # 局部对分加密：每轮取括号的两个三等分点，连同括号端点共 4 个候选值，
    # 最坏点与其相邻两点构成新括号 —— 最坏点贴在区间端点时也保证每轮至少
    # 收缩 1/3，绝不会卡死。None（无有限穿越）按 +∞ 处理但照样记录不漏点。
    for _ in range(MAX_LOCAL_ZOOM):
        if (br - bl) <= PARAM_REL_TOL * max(1.0, abs(0.5 * (bl + br))):
            break
        m1 = bl + (br - bl) / 3.0
        m2 = br - (br - bl) / 3.0
        pts = [(bl, fval(bl)), (m1, fval(m1)), (m2, fval(m2)), (br, fval(br))]
        j = min(range(4), key=lambda i: pts[i][1])
        if j == 0:
            bl2, br2 = bl, m1
        elif j == 1:
            bl2, br2 = bl, m2
        elif j == 2:
            bl2, br2 = m1, br
        else:
            bl2, br2 = m2, br
        bl, br = bl2, br2
    else:  # pragma: no cover - 保险：真跑满轮数也应已远优于容差
        raise ServiceError("极点维度最坏化加密未能在约定轮数内收敛")

    root = 0.5 * (bl + br)
    root_v = objective(root)
    record(root, root_v)
    # 最坏点恰在区间端点（如极点左移对常见对象最坏）时，中点会带一个容差量级
    # 的偏；与全程记录到的最优有限点比较后取更优，端点情形即返回精确端点。
    best_i = min((i for i, v in enumerate(vals) if v is not None),
                 key=lambda i: vals[i])
    if root_v is not None and root_v <= vals[best_i]:
        return LineMinimum(root, root_v, list(zip(xs, vals)))
    return LineMinimum(xs[best_i], vals[best_i], list(zip(xs, vals)))


# ---------------------------------------------------------------------------
# 鲁棒分析结果与对外口径
# ---------------------------------------------------------------------------

@dataclass
class WorstMargin:
    exists_family: bool               # 整个族里是否存在任何有限穿越
    value: float | None               # 最坏裕度值（PM 用度，GM 用真值）
    value_db: float | None
    K: float | None
    L: float | None
    pole: float | None                # 极点坐标（负值或 0）
    omega: float | None               # PM→ω_c，GM→ω_π
    partial_missing: bool = False     # 族内有参数点无有限穿越（未计入最小化）


@dataclass
class CriticalPoint:
    found: bool
    margin: str | None                # "phase_margin" / "gain_margin"
    K: float | None
    L: float | None
    pole: float | None
    phase_margin_deg: float | None
    gain_margin: float | None
    gain_margin_db: float | None
    omega_c: float | None
    omega_pi: float | None


@dataclass
class RobustResult:
    worst_pm: WorstMargin
    worst_gm: WorstMargin
    robust_stable: bool
    critical: CriticalPoint
    uncertified: list[dict[str, Any]]
    family: dict[str, Any]
    nominal: dict[str, Any]


def _worst_from_line(line: LineMinimum, *, kind: str, reader: FamilyReader,
                     K: float, L: float) -> WorstMargin:
    if line.value is None:
        return WorstMargin(False, None, None, None, None, None, None, partial_missing=False)
    p = line.x
    r = reader.read(K, L, p)
    partial = any(v is None for _, v in line.samples)
    if kind == "pm":
        return WorstMargin(True, r.phase_margin_deg, None, K, L,
                           -p if reader.unc.pole is not None else None,
                           r.omega_c, partial)
    return WorstMargin(True, r.gain_margin, r.gain_margin_db, K, L,
                       -p if reader.unc.pole is not None else None,
                       r.omega_pi, partial)


def _stability_scalars(res: Any) -> tuple[float | None, float | None]:
    """返回 (PM 度, GM dB)；对应穿越不存在则该分量为 None。"""

    pm = res.phase_margin_deg
    gm = res.gain_margin_db
    return pm, gm


def _is_positive_margin(v: float | None) -> bool:
    """严格为正才算守得住：贴零（绝对值 < ZERO_MARGIN_TOL）按越界处理。"""

    return v is not None and v > ZERO_MARGIN_TOL


def _is_nonpositive_margin(v: float | None) -> bool:
    return v is not None and v <= ZERO_MARGIN_TOL


def _bisect_zero(x_safe: float, x_bad: float, read_result: Callable[[float], Any],
                 ) -> tuple[float, Any]:
    """在 [x_safe, x_bad] 上对分 s(x)=min(PM,GM_dB) 的首个零点。

    s(x_safe)>0、s(x_bad)<=0，且沿 x_safe→x_bad 方向单调变坏（K↑/L↑/p↓ 都
    满足）。对分至参数相对容差 PARAM_REL_TOL 后，用最终括号两端的标量做一次
    线性插值钉住零点，再在该参数点上回代读数（与逐点扫频同一内核）。
    """

    lo, hi = x_safe, x_bad

    def scalar_of(res: Any) -> float:
        pm, gm = _stability_scalars(res)
        vals = [v for v in (pm, gm) if v is not None]
        return min(vals) if vals else math.inf

    res_lo, res_hi = read_result(lo), read_result(hi)
    for _ in range(MAX_LOCAL_ZOOM):
        if abs(hi - lo) <= PARAM_REL_TOL * max(1.0, abs(0.5 * (lo + hi))):
            break
        mid = 0.5 * (lo + hi)
        res_mid = read_result(mid)
        if scalar_of(res_mid) <= ZERO_MARGIN_TOL:
            hi, res_hi = mid, res_mid
        else:
            lo, res_lo = mid, res_mid

    s_lo, s_hi = scalar_of(res_lo), scalar_of(res_hi)
    if s_lo == s_hi:
        x0 = hi
    else:
        frac = s_lo / (s_lo - s_hi)  # 线性插零点；s_lo>0, s_hi<=0
        x0 = lo + frac * (hi - lo)
    return x0, read_result(x0)


def _make_critical(res: Any, *, K: float, L: float, pole_value: float | None) -> CriticalPoint:
    pm, gm = _stability_scalars(res)
    # 临界对分后零点上的两个裕度都在零附近；标出更贴近（或已越过）零界的那个。
    candidates: list[tuple[float, str]] = []
    if pm is not None:
        candidates.append((pm, "phase_margin"))
    if gm is not None:
        candidates.append((gm, "gain_margin"))
    which = min(candidates, key=lambda t: t[0])[1] if candidates else None
    return CriticalPoint(
        True, which, K, L, pole_value,
        res.phase_margin_deg, res.gain_margin, res.gain_margin_db,
        res.omega_c, res.omega_pi,
    )


def _find_critical(reader: FamilyReader, unc: Uncertainty,
                   pm_line: LineMinimum, gm_line: LineMinimum,
                   K_lo: float, K_hi: float, L_hi: float,
                   p_lo: float, p_hi: float) -> CriticalPoint:
    """找第一处任一裕度落到 0 或为负的临界参数点。

    最坏情形必然在坍缩面 (K_hi, L_hi) 上；若极点在飘，先沿 p 轴（p↓ 变坏）
    在搜索已评点里定位首个越界段并对分。极点退化为单点时，沿 K 轴（K↑ 变坏）
    对分（L 已取 L_hi）。
    """

    none_pt = CriticalPoint(False, None, None, None, None, None, None, None, None, None)

    if unc.pole is not None:
        evaluated: dict[float, Any] = {}

        def result_at_p(p: float) -> Any:
            if p not in evaluated:
                evaluated[p] = reader.read(K_hi, L_hi, p)
            return evaluated[p]

        for p, _ in pm_line.samples:
            result_at_p(p)
        for p, _ in gm_line.samples:
            result_at_p(p)

        def scalar_p(p: float) -> float:
            pm, gm = _stability_scalars(result_at_p(p))
            vals = [v for v in (pm, gm) if v is not None]
            return min(vals) if vals else math.inf

        ordered = sorted(evaluated)
        for i, p_bad in enumerate(ordered):
            if scalar_p(p_bad) <= ZERO_MARGIN_TOL:
                if i == 0:
                    p_crit, res = p_bad, result_at_p(p_bad)
                else:
                    p_crit, res = _bisect_zero(ordered[i - 1], p_bad, result_at_p)
                return _make_critical(res, K=K_hi, L=L_hi, pole_value=-p_crit)
        return none_pt

    # 极点不动：坍缩面退化为沿 K 的一维（L=L_hi）。
    if unc.K is None:
        # K、L 都没在飘：唯一参数点若越界即临界。
        res = reader.read(K_hi, L_hi, 0.0)
        pm, gm = _stability_scalars(res)
        if _is_nonpositive_margin(pm) or _is_nonpositive_margin(gm):
            return _make_critical(res, K=K_hi, L=L_hi, pole_value=None)
        return none_pt

    res_lo = reader.read(K_lo, L_hi, 0.0)
    res_hi = reader.read(K_hi, L_hi, 0.0)

    def pm_gm(r: Any) -> list[float]:
        pm, gm = _stability_scalars(r)
        return [v for v in (pm, gm) if v is not None]

    vals_lo, vals_hi = pm_gm(res_lo), pm_gm(res_hi)
    if not any(_is_nonpositive_margin(v) for v in vals_hi):
        return none_pt
    if not vals_lo or min(vals_lo) <= ZERO_MARGIN_TOL:
        res = res_lo if (not vals_lo or min(vals_lo) <= ZERO_MARGIN_TOL) else res_hi
        return _make_critical(res, K=K_lo, L=L_hi, pole_value=None)
    K_crit, res = _bisect_zero(K_lo, K_hi, lambda K: reader.read(K, L_hi, 0.0))
    return _make_critical(res, K=K_crit, L=L_hi, pole_value=None)


def robust_analyze(plant: Plant, grid: list[float], unc: Uncertainty) -> RobustResult:
    """参数族最坏情况读数 + 鲁棒稳定判定。"""

    reader = FamilyReader(plant, grid, unc)

    K_hi = unc.K.hi if unc.K is not None else plant.K
    K_lo = unc.K.lo if unc.K is not None else plant.K
    L_hi = unc.L.hi if unc.L is not None else plant.L
    L_lo = unc.L.lo if unc.L is not None else plant.L
    if unc.pole is not None:
        p_lo, p_hi = unc.pole.lo, unc.pole.hi
        p_nom = -unc.pole.nominal
    else:
        p_lo = p_hi = p_nom = 0.0

    def pm_at(p: float) -> float | None:
        return reader.read(K_hi, L_hi, p).phase_margin_deg

    def gm_at(p: float) -> float | None:
        return reader.read(K_hi, L_hi, p).gain_margin

    pm_line = worst_on_line(p_lo, p_hi, pm_at, extra=(p_nom,))
    gm_line = worst_on_line(p_lo, p_hi, gm_at, extra=(p_nom,))

    worst_pm = _worst_from_line(pm_line, kind="pm", reader=reader, K=K_hi, L=L_hi)
    worst_gm = _worst_from_line(gm_line, kind="gm", reader=reader, K=K_hi, L=L_hi)

    # 鲁棒判定与临界参数点：先看坍缩线（K_hi,L_hi 已含最坏裕度）。
    critical = _find_critical(reader, unc, pm_line, gm_line,
                              K_lo, K_hi, L_hi, p_lo, p_hi)

    # 穿越“存在性”随 K、L 单调：无穿越只可能出现在 (K_lo, L_lo) 这个对角，
    # 且随 p（极点到原点距离）越小相位越滞后，故只查极点区间两端即可覆盖
    # 全族。坍缩线本身已覆盖 K_hi/L_hi 端。这些点不参与最小化，只作证据。
    uncertified: list[dict[str, Any]] = []
    missing_pm_anywhere = False
    missing_gm_anywhere = False
    probe_ps = sorted({p_lo, p_hi}) if unc.pole is not None else [p_lo]
    for p in probe_ps:
        r = reader.read(K_lo, L_lo, p)
        pm, gm = _stability_scalars(r)
        finite = [v for v in (pm, gm) if v is not None]
        if pm is None:
            missing_pm_anywhere = True
        if gm is None:
            missing_gm_anywhere = True
        if pm is None or gm is None:
            # 任一有限穿越缺失：按现有单点口径「证据不足」，绝不当稳定证据。
            ok = False
        else:
            ok = _is_positive_margin(pm) and _is_positive_margin(gm)
        if not ok:
            uncertified.append({
                "K": K_lo,
                "L": L_lo,
                "pole": -p if unc.pole is not None else None,
                "reason": "该参数点无有限穿越，无法取得为正的裕度证据"
                          if not finite else "该参数点存在非正（或贴零）裕度",
                "phase_margin_deg": r.phase_margin_deg,
                "gain_margin_db": r.gain_margin_db,
            })

    # 搜索线（K_hi/L_hi 端）与证据探针（K_lo/L_lo 端）上任何一处无穿越，
    # 都按「族内部分点无有限穿越」标出，绝不悄悄丢掉不算。
    worst_pm.partial_missing = worst_pm.partial_missing or missing_pm_anywhere
    worst_gm.partial_missing = worst_gm.partial_missing or missing_gm_anywhere
    # 存在性是族级的：搜索线上全无、但对侧探针有有限穿越时也算「族内存在」。
    if not worst_pm.exists_family and not missing_pm_anywhere:
        # 探针点有 PM：用探针读数补一个最坏点（K_lo/L_lo 端）。
        for p in probe_ps:
            r = reader.read(K_lo, L_lo, p)
            if r.phase_margin_deg is not None:
                worst_pm.exists_family = True
                worst_pm.value = r.phase_margin_deg
                worst_pm.K, worst_pm.L = K_lo, L_lo
                worst_pm.pole = -p if unc.pole is not None else None
                worst_pm.omega = r.omega_c
                break
    if not worst_gm.exists_family and not missing_gm_anywhere:
        for p in probe_ps:
            r = reader.read(K_lo, L_lo, p)
            if r.gain_margin is not None:
                worst_gm.exists_family = True
                worst_gm.value = r.gain_margin
                worst_gm.value_db = r.gain_margin_db
                worst_gm.K, worst_gm.L = K_lo, L_lo
                worst_gm.pole = -p if unc.pole is not None else None
                worst_gm.omega = r.omega_pi
                break

    robust_stable = (
        worst_pm.exists_family and worst_gm.exists_family
        and _is_positive_margin(worst_pm.value)
        and worst_gm.value is not None
        and worst_gm.value_db is not None
        and _is_positive_margin(worst_gm.value_db)
        and not uncertified
    )

    family: dict[str, Any] = {
        "K": [unc.K.lo, unc.K.hi] if unc.K is not None else [plant.K, plant.K],
        "L": [unc.L.lo, unc.L.hi] if unc.L is not None else [plant.L, plant.L],
        "pole": [-unc.pole.hi, -unc.pole.lo] if unc.pole is not None else None,
    }
    nominal = {"num": list(plant.num), "den": list(plant.den),
               "K": plant.K, "L": plant.L}

    return RobustResult(worst_pm, worst_gm, robust_stable, critical,
                        uncertified, family, nominal)


def _worst_block(w: WorstMargin, *, kind: str, crossover_key: str, omega_key: str,
                 margin_key: str, missing_note: str) -> dict[str, Any]:
    crossover_note = (
        "全族无有限幅值穿越" if kind == "pm" else "全族无有限相位穿越"
    )
    if not w.exists_family:
        return {
            "exists": False,
            "note": missing_note,
            margin_key: None,
            **({"gain_margin_db": None} if kind == "gm" else {}),
            "at": None,
            crossover_key: {"exists": False, omega_key: None,
                            "note": crossover_note},
        }
    block: dict[str, Any] = {
        "exists": True,
        "note": "部分参数点无有限穿越，未计入最小化；已纳入鲁棒稳定判定"
        if w.partial_missing else None,
        margin_key: w.value,
        "at": {"K": w.K, "L": w.L, "pole": w.pole},
        crossover_key: {"exists": True, omega_key: w.omega, "note": None},
    }
    if kind == "gm":
        block["gain_margin_db"] = w.value_db
    return block


def robust_to_payload(r: RobustResult) -> dict[str, Any]:
    pm_block = _worst_block(
        r.worst_pm, kind="pm",
        crossover_key="gain_crossover", omega_key="omega_c",
        margin_key="phase_margin_deg",
        missing_note="全族无有限幅值穿越",
    )
    gm_block = _worst_block(
        r.worst_gm, kind="gm",
        crossover_key="phase_crossover", omega_key="omega_pi",
        margin_key="gain_margin",
        missing_note="全族无有限相位穿越",
    )
    c = r.critical
    critical_block = {
        "found": c.found,
        "margin": c.margin,
        "at": {"K": c.K, "L": c.L, "pole": c.pole} if c.found else None,
        "phase_margin_deg": c.phase_margin_deg,
        "gain_margin": c.gain_margin,
        "gain_margin_db": c.gain_margin_db,
        "omega_c": c.omega_c,
        "omega_pi": c.omega_pi,
    }
    return {
        "family": r.family,
        "nominal_plant": r.nominal,
        "worst_phase_margin": pm_block,
        "worst_gain_margin": gm_block,
        "robust_stable": r.robust_stable,
        "critical_point": critical_block,
        "uncertified_points": r.uncertified,
    }
