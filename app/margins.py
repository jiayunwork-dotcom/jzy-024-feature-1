"""穿越搜索、裕度读数与稳定性判定。

定义（与题目一致）：
  * 幅值穿越 ω_c：|G(jω)| 穿过 1 的频率；相位裕度 PM = ∠G(jω_c) + 180°。
  * 相位穿越 ω_π：∠G(jω) 穿过 −180° 的频率；幅值裕度 GM = 1 / |G(jω_π)|。

实现要点：
  * 先在调用方给出的网格上找异号区间，绝不用邻近网格点冒充交界；
  * 找到区间后在两点之间用对分加密，频率区间相对宽度小到服务钉死的容差
    FREQ_REL_TOL 才报；
  * 找不到有限穿越时一律标 None（「无有限穿越」），绝不填 0 或一个大数；
  * 对最小相位对象：两个已存在的裕度同号且都为正才判稳定。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .frf import TransferFunction, mag_to_db

# 服务钉死的频率相对容差：加密到 |hi-lo|/|mid| < 该值才读数。
FREQ_REL_TOL = 1e-10
# 对分的最大轮数（1e-10 相对容差约需 35 轮，给到 100 足够保险）。
MAX_BISECT = 100

_PHASE_CROSS_RAD = -math.pi
_PHASE_CROSS_DEG = -180.0


@dataclass
class MarginResult:
    omega_c: float | None            # 幅值穿越频率（无则 None）
    omega_pi: float | None           # 相位穿越频率（无则 None）
    phase_margin_deg: float | None   # 相位裕度（度）
    gain_margin: float | None        # 幅值裕度真值 1/|G(jω_π)|
    gain_margin_db: float | None     # 幅值裕度分贝 = 20*log10(真值)
    stable: bool
    has_finite_gain_crossover: bool
    has_finite_phase_crossover: bool
    bode: dict[str, list[float]] = field(default_factory=dict)


def _bisect_magnitude(tf: TransferFunction, lo: float, hi: float,
                      f_lo: float, f_hi: float,
                      anchor_lo: float, anchor_hi: float) -> tuple[float, float]:
    """在 [lo, hi] 内对分 |G|-1=0，返回 (频率, 连续相位弧度)。

    f_lo/f_hi 是两端点的 |G|-1；anchor_lo/anchor_hi 是两端点的连续相位，
    用来给对分中点挑正确的 2π 分支。
    """

    for _ in range(MAX_BISECT):
        mid = 0.5 * (lo + hi)
        if abs(hi - lo) <= FREQ_REL_TOL * abs(mid):
            break
        fm = tf.magnitude(mid) - 1.0
        if fm == 0.0:
            lo = hi = mid
            break
        if (f_lo < 0) == (fm < 0):
            lo, f_lo, anchor_lo = mid, fm, tf.continuous_phase_rad(mid, anchor_lo)
        else:
            hi, f_hi, anchor_hi = mid, fm, tf.continuous_phase_rad(mid, anchor_hi)
    root = 0.5 * (lo + hi)
    phase = tf.continuous_phase_rad(root, 0.5 * (anchor_lo + anchor_hi))
    return root, phase


def _bisect_phase(tf: TransferFunction, lo: float, hi: float,
                  p_lo: float, p_hi: float) -> tuple[float, float]:
    """在 [lo, hi] 内对分 ∠G+π=0，返回 (频率, 连续相位弧度)。"""

    anchor_lo, anchor_hi = p_lo, p_hi
    for _ in range(MAX_BISECT):
        mid = 0.5 * (lo + hi)
        if abs(hi - lo) <= FREQ_REL_TOL * abs(mid):
            break
        p_mid = tf.continuous_phase_rad(mid, 0.5 * (anchor_lo + anchor_hi))
        fm = p_mid - _PHASE_CROSS_RAD
        if fm == 0.0:
            return mid, p_mid
        if (p_lo - _PHASE_CROSS_RAD < 0) == (fm < 0):
            lo, p_lo, anchor_lo = mid, fm + _PHASE_CROSS_RAD, p_mid
        else:
            hi, p_hi, anchor_hi = mid, fm + _PHASE_CROSS_RAD, p_mid
    root = 0.5 * (lo + hi)
    phase = tf.continuous_phase_rad(root, 0.5 * (anchor_lo + anchor_hi))
    return root, phase


def _find_gain_crossover(tf: TransferFunction, freqs: list[float],
                         mags: list[float], phase_grid: list[float]
                         ) -> tuple[float | None, float | None]:
    """返回 (ω_c, ω_c 处连续相位弧度)。取最低频的一个幅值穿越。"""

    n = len(freqs)
    for i in range(n - 1):
        f_lo, f_hi = mags[i] - 1.0, mags[i + 1] - 1.0
        if f_lo == 0.0:
            w = freqs[i]
            return w, phase_grid[i]
        if f_lo * f_hi < 0.0:
            return _bisect_magnitude(tf, freqs[i], freqs[i + 1], f_lo, f_hi,
                                     phase_grid[i], phase_grid[i + 1])
    if mags[-1] - 1.0 == 0.0:
        return freqs[-1], phase_grid[-1]
    return None, None


def _find_phase_crossover(tf: TransferFunction, freqs: list[float],
                          phase_grid: list[float]
                          ) -> tuple[float | None, float | None]:
    """返回 (ω_π, ω_π 处连续相位弧度)。取最低频的一个 −180° 穿越。"""

    n = len(freqs)
    for i in range(n - 1):
        d_lo = phase_grid[i] - _PHASE_CROSS_RAD
        d_hi = phase_grid[i + 1] - _PHASE_CROSS_RAD
        if d_lo == 0.0:
            return freqs[i], phase_grid[i]
        if d_lo * d_hi < 0.0:
            return _bisect_phase(tf, freqs[i], freqs[i + 1],
                                 phase_grid[i], phase_grid[i + 1])
    if phase_grid[-1] == _PHASE_CROSS_RAD:
        return freqs[-1], phase_grid[-1]
    return None, None


def analyze(tf: TransferFunction, freqs: list[float]) -> MarginResult:
    """在给定网格上求整条 Bode 曲线、两个穿越、两个裕度与稳定性。"""

    bode = tf.bode_grid(freqs)
    phase_rad = [p * math.pi / 180.0 for p in bode.phase_deg]

    omega_c, phase_at_c = _find_gain_crossover(tf, freqs, bode.mag, phase_rad)
    omega_pi, _ = _find_phase_crossover(tf, freqs, phase_rad)

    pm = None
    if omega_c is not None and phase_at_c is not None:
        pm = phase_at_c * 180.0 / math.pi + 180.0

    gm = gm_db = None
    if omega_pi is not None:
        gm = 1.0 / tf.magnitude(omega_pi)
        gm_db = mag_to_db(gm)

    # 最小相位判据：已有的裕度必须全部为正才算稳定；
    # 两个穿越都不在（有限）网格范围内时，证据不足，不宣称稳定。
    margins = [m for m in (pm, gm_db) if m is not None]
    stable = bool(margins) and all(m > 0.0 for m in margins)

    return MarginResult(
        omega_c=omega_c,
        omega_pi=omega_pi,
        phase_margin_deg=pm,
        gain_margin=gm,
        gain_margin_db=gm_db,
        stable=stable,
        has_finite_gain_crossover=omega_c is not None,
        has_finite_phase_crossover=omega_pi is not None,
        bode={
            "omega": list(bode.freqs),
            "magnitude": list(bode.mag),
            "magnitude_db": list(bode.mag_db),
            "phase_deg": list(bode.phase_deg),
        },
    )


def to_payload(r: MarginResult) -> dict[str, Any]:
    """转成对外 JSON：无有限穿越时显式给出标记与 null，绝不填 0。"""

    return {
        "bode": r.bode,
        "gain_crossover": {
            "exists": r.has_finite_gain_crossover,
            "omega_c": r.omega_c,
            "note": None if r.has_finite_gain_crossover else "无有限幅值穿越",
        },
        "phase_crossover": {
            "exists": r.has_finite_phase_crossover,
            "omega_pi": r.omega_pi,
            "note": None if r.has_finite_phase_crossover else "无有限相位穿越",
        },
        "phase_margin_deg": r.phase_margin_deg,
        "gain_margin": r.gain_margin,
        "gain_margin_db": r.gain_margin_db,
        "stable": r.stable,
    }
