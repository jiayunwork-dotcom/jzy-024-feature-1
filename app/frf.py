"""频响求值：把 s 换成 jω。

对 G(s) = K * num(s)/den(s) * e^{-Ls}，在给定 ω 处计算 G(jω)：
分子分母分别在虚轴上各得到一个复数，相除即开环频率响应；
纯延迟 e^{-jωL} 只动相位（相位额外减去 ωL 弧度），幅值曲线保持原样。

对外相位一律用度，相位做解卷绕（unwrap），因此返回的是连续相位而不是 ±180°
之间的主值——延迟和高阶对象带来的相位滞后可以一直往下走。
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

from .validation import Plant

TWO_PI = 2.0 * math.pi
_DEG_PER_RAD = 180.0 / math.pi


def _poly_eval(coeffs: tuple[float, ...], z: complex) -> complex:
    """降幂多项式在复数点求值（霍纳法）。"""

    acc = 0.0j
    for c in coeffs:
        acc = acc * z + c
    return acc


def mag_to_db(mag: float) -> float:
    """分贝必须等于 20 * log10(真值)。"""

    return 20.0 * math.log10(mag)


def unwrap_radians(phases: list[float]) -> list[float]:
    """沿频率序列解卷绕：把相邻跳变都折回 (-π, π] 后累加。"""

    if not phases:
        return []
    out = [phases[0]]
    for prev_raw, cur in zip(phases, phases[1:]):
        delta = (cur - prev_raw + math.pi) % TWO_PI - math.pi
        out.append(out[-1] + delta)
    return out


@dataclass(frozen=True)
class FrequencyResponse:
    freqs: list[float]
    mag: list[float]          # 幅值真值
    mag_db: list[float]       # 20*log10(幅值)
    phase_deg: list[float]    # 解卷绕后的连续相位（度）


class TransferFunction:
    """一个已规范化的真有理对象，可在任意正频率点上求频响。"""

    def __init__(self, plant: Plant):
        self.plant = plant
        self._den_degree = len(plant.den) - 1

    def evaluate_raw(self, w: float) -> tuple[float, float]:
        """返回 (幅值, 主值相位弧度，含延迟)。"""

        s = complex(0.0, w)
        num = _poly_eval(self.plant.num, s)
        den = _poly_eval(self.plant.den, s)
        if den == 0.0j:
            # 真有理实系数系统的极点在正虚轴上：G(jω) 发散，无法作为有限穿越。
            raise ZeroDivisionError(f"频率 {w} 落在虚轴极点上")
        g = self.plant.K * num / den
        phase = cmath.phase(g) - w * self.plant.L  # 纯延迟只减相位
        return abs(g), phase

    def magnitude(self, w: float) -> float:
        num = _poly_eval(self.plant.num, complex(0.0, w))
        den = _poly_eval(self.plant.den, complex(0.0, w))
        return self.plant.K * abs(num) / abs(den)

    def bode_grid(self, freqs: list[float]) -> FrequencyResponse:
        mags: list[float] = []
        raw_phases: list[float] = []
        for w in freqs:
            m, p = self.evaluate_raw(w)
            mags.append(m)
            raw_phases.append(p)
        phases = unwrap_radians(raw_phases)
        return FrequencyResponse(
            freqs=list(freqs),
            mag=mags,
            mag_db=[mag_to_db(m) for m in mags],
            phase_deg=[p * _DEG_PER_RAD for p in phases],
        )

    def continuous_phase_rad(self, w: float, anchor: float) -> float:
        """求 ω 处的连续相位（弧度）。

        evaluate_raw 只给主值；这里把主值加上 2π 的整数倍，使其落在锚点 anchor
        （相邻网格点上的连续相位）最近的那一支上。
        """

        _, principal = self.evaluate_raw(w)
        k = round((anchor - principal) / TWO_PI)
        return principal + k * TWO_PI
