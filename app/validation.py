"""入参语义检查与规范化。

这一层不碰 HTTP、不碰频响计算，只负责：
  * 系数 / 零极点 / K / L 必须是有限数（bool 不算数）；
  * 传递函数必须是真有理（分子次数不高于分母次数）；
  * 频率网格必须非空、严格递增、且全为正；
  * ZPK（零点、极点、增益）形式就地展开成多项式系数。

系数一律按降幂排列：[a_n, a_{n-1}, ..., a_0]。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .errors import ServiceError

# ZPK 展开后，共轭极点/零点理论上应给出实系数；虚部残差超过该值视为参数不自洽。
_IMAG_TOL = 1e-9


@dataclass(frozen=True)
class Plant:
    """规范化后的被控对象：真有理传递函数 K * num(s)/den(s) * e^{-Ls}。"""

    num: tuple[float, ...]
    den: tuple[float, ...]
    K: float
    L: float
    form: str  # 来源形式："poly" 或 "zpk"


def _as_finite_number(value: Any, what: str) -> float:
    # bool 是 int 的子类，明确挡掉。
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ServiceError(f"{what}必须是有限数，收到的是：{value!r}")
    f = float(value)
    if not math.isfinite(f):
        raise ServiceError(f"{what}必须是有限数，收到的是：{value!r}")
    return f


def _coeffs(value: Any, what: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ServiceError(f"{what}必须是非空系数数组（按降幂排列）")
    return [_as_finite_number(c, f"{what}的系数") for c in value]


def _roots(value: Any, what: str) -> list[complex]:
    """零点/极点允许写成标量（实数）或 [实部, 虚部] 二元数组。"""

    if not isinstance(value, list):
        raise ServiceError(f"{what}必须是数组")
    roots: list[complex] = []
    for i, item in enumerate(value):
        if isinstance(item, list) and len(item) == 2:
            re = _as_finite_number(item[0], f"{what}[{i}] 的实部")
            im = _as_finite_number(item[1], f"{what}[{i}] 的虚部")
            roots.append(complex(re, im))
        else:
            roots.append(complex(_as_finite_number(item, f"{what}[{i}]"), 0.0))
    return roots


def _multiply_linear_factors(roots: list[complex]) -> list[complex]:
    """由根列表构造多项式 ∏(s - r)，返回复系数（降幂）。"""

    poly: list[complex] = [1.0 + 0j]
    for r in roots:
        # 当前多项式乘上 (s - r)
        nxt = [0.0j] * (len(poly) + 1)
        for i, c in enumerate(poly):
            nxt[i] += c           # 乘 s
            nxt[i + 1] -= c * r   # 乘 -r
        poly = nxt
    return poly


def _to_real(poly: list[complex], what: str) -> list[float]:
    out: list[float] = []
    scale = max((abs(c) for c in poly), default=0.0)
    tol = _IMAG_TOL * max(1.0, scale)
    for c in poly:
        if abs(c.imag) > tol:
            raise ServiceError(f"{what}展开后存在不可忽略的虚部（{c.imag:.3g}），系数不自洽")
        out.append(c.real)
    return out


def _strip_leading_zeros(coeffs: list[float]) -> list[float]:
    for i, c in enumerate(coeffs):
        if c != 0.0:
            return coeffs[i:]
    return [0.0]


def _degree(coeffs: list[float]) -> int:
    return len(_strip_leading_zeros(coeffs)) - 1


def validate_plant(spec: Any, *, K_override: float | None = None,
                   L_override: float | None = None) -> Plant:
    """把一份对象描述（poly 或 zpk）检查并规范化为 Plant。

    当次扫频可通过 K_override / L_override 覆盖档内的 K、L（仅覆盖，不改档）。
    """

    if not isinstance(spec, dict):
        raise ServiceError("对象描述必须是对象：{type: 'poly', ...} 或 {type: 'zpk', ...}")
    form = spec.get("type")
    if form not in ("poly", "zpk"):
        raise ServiceError("对象描述必须显式给出 type='poly'（分子/分母）或 type='zpk'（零点/极点/增益）")

    K = _as_finite_number(K_override if K_override is not None else spec.get("K"), "开环增益 K")
    if K <= 0:
        raise ServiceError(f"开环增益 K 必须为正数，收到：{K}")
    L = _as_finite_number(L_override if L_override is not None else spec.get("L", 0.0), "纯延迟 L")
    if L < 0:
        raise ServiceError(f"纯延迟 L 必须非负，收到：{L}")

    if form == "poly":
        num = _strip_leading_zeros(_coeffs(spec.get("num"), "分子 num"))
        den = _strip_leading_zeros(_coeffs(spec.get("den"), "分母 den"))
        if num == [0.0]:
            raise ServiceError("分子 num 全为零，传递函数恒为零，无法定义裕度")
        if den == [0.0]:
            raise ServiceError("分母 den 全为零")
    else:
        if "gain" not in spec:
            raise ServiceError("zpk 形式必须给出增益 gain")
        zpk_gain = _as_finite_number(spec["gain"], "zpk 增益 gain")
        if zpk_gain == 0:
            raise ServiceError("zpk 增益 gain 不能为 0")
        zeros = _roots(spec.get("zeros", []), "零点 zeros")
        poles = _roots(spec.get("poles", []), "极点 poles")
        num = _strip_leading_zeros(_to_real(_multiply_linear_factors(zeros), "零点"))
        den = _strip_leading_zeros(_to_real(_multiply_linear_factors(poles), "极点"))
        num = [c * zpk_gain for c in num]

    # 真有理性：分子次数不得高于分母次数。
    if _degree(num) > _degree(den):
        raise ServiceError(
            f"非真有理传递函数：分子次数 {_degree(num)} 高于分母次数 {_degree(den)}，拒绝扫频"
        )

    return Plant(tuple(num), tuple(den), K, L, form)


def validate_grid(freqs: Any) -> list[float]:
    if not isinstance(freqs, list) or not freqs:
        raise ServiceError("频率网格必须是非空数组")
    grid = [_as_finite_number(w, f"频率点 {i}") for i, w in enumerate(freqs)]
    for i, w in enumerate(grid):
        if w <= 0:
            raise ServiceError(f"频率网格必须全为正，第 {i} 个点为 {w}")
        if i > 0 and w <= grid[i - 1]:
            raise ServiceError(
                f"频率网格必须严格递增，第 {i - 1} 个点 {grid[i - 1]} 不小于第 {i} 个点 {w}"
            )
    return grid


def validate_name(name: Any) -> str:
    if not isinstance(name, str) or not name:
        raise ServiceError("档名必须是非空字符串")
    if not all(ch.isalnum() or ch in "._-" for ch in name):
        raise ServiceError("档名只能包含字母、数字、点、下划线与连字符")
    if name in (".", ".."):
        raise ServiceError("非法档名")
    return name
