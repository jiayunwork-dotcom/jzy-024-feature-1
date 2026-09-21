"""不确定性说明的语义检查与参数族构造。

允许飘动的量有三类（都可缺省，缺省即取名义值、不参与最坏化）：
  * 开环增益 K：区间 [min, max]，必须整体为正；
  * 纯延迟 L：区间 [min, max]，不得含负值（下界可以为 0）；
  * 某一个实极点：在实轴区间 [min, max] 上移动，其余零极点不动；
    区间不得越过虚轴进入右半平面（max <= 0）；nominal 用来标识
    移动的是哪一个极点，必须对应对象的一个真实极点。

三类都不给（或干脆不传 uncertainty）是退化情形：参数族只剩名义点，
最坏化读数必须与普通扫频逐项相等。

本层只负责「参数族长什么样、族内一点怎么映射成规范化 Plant」，
不做任何裕度计算；最坏化搜索与鲁棒判定在 robust.py，
频响与穿越读数复用 frf.py / margins.py。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .errors import ServiceError
from .validation import Plant, validate_plant

# 极点定位容差：|候选极点 - nominal| 的相对偏差不超过该值即认为找到对应极点。
_POLE_MATCH_TOL = 1e-9


@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float


@dataclass(frozen=True)
class PoleInterval:
    nominal: float
    lo: float
    hi: float


@dataclass(frozen=True)
class Uncertainty:
    """规范化后的不确定性说明；None 表示该维不飘（取名义值）。"""

    K: Interval | None
    L: Interval | None
    pole: PoleInterval | None


def _finite(value: Any, what: str) -> float:
    # bool 是 int 的子类，明确挡掉（与 validation 层同口径）。
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ServiceError(f"{what} 必须是有限数，收到的是：{value!r}")
    f = float(value)
    if not math.isfinite(f):
        raise ServiceError(f"{what} 必须是有限数，收到的是：{value!r}")
    return f


def _interval(raw: Any, key: str) -> Interval | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ServiceError(f"不确定性 {key} 必须是区间对象：{{\"min\": ..., \"max\": ...}}")
    lo = _finite(raw.get("min"), f"不确定性 {key}.min")
    hi = _finite(raw.get("max"), f"不确定性 {key}.max")
    if hi < lo:
        raise ServiceError(f"不确定性 {key} 区间非法：上界 {hi} 小于下界 {lo}")
    return Interval(lo, hi)


def _pole_interval(raw: Any) -> PoleInterval | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ServiceError("不确定性 pole 必须是对象：{\"nominal\": ..., \"min\": ..., \"max\": ...}")
    nominal = _finite(raw.get("nominal"), "不确定性 pole.nominal")
    lo = _finite(raw.get("min"), "不确定性 pole.min")
    hi = _finite(raw.get("max"), "不确定性 pole.max")
    if hi < lo:
        raise ServiceError(f"极点区间非法：上界 {hi} 小于下界 {lo}")
    if hi > 0.0:
        raise ServiceError(
            f"极点区间 [{lo}, {hi}] 越过虚轴进入右半平面，"
            "会改变对象的最小相位属性，拒绝分析"
        )
    return PoleInterval(nominal, lo, hi)


def validate_uncertainty(raw: Any) -> Uncertainty:
    """把一份不确定性说明检查并规范化；raw 为 None 或空对象时全部维度不飘。"""

    if raw is None:
        return Uncertainty(None, None, None)
    if not isinstance(raw, dict):
        raise ServiceError(
            "不确定性说明必须是对象："
            "{\"K\": {\"min\", \"max\"}, \"L\": {\"min\", \"max\"}, "
            "\"pole\": {\"nominal\", \"min\", \"max\"}}，每个键都可缺省"
        )
    k = _interval(raw.get("K"), "K")
    if k is not None and k.lo <= 0.0:
        raise ServiceError(f"增益 K 的不确定区间必须为正，收到下界 {k.lo}")
    delay = _interval(raw.get("L"), "L")
    if delay is not None and delay.lo < 0.0:
        raise ServiceError(f"纯延迟 L 的不确定区间不得含负值，收到下界 {delay.lo}")
    return Uncertainty(k, delay, _pole_interval(raw.get("pole")))


def _zpk_view(spec: dict[str, Any]) -> dict[str, Any] | None:
    """取出对象的 zpk 视角：内联 zpk 直接用；存档记录带 source_zpk 时还原。"""

    if not isinstance(spec, dict):
        return None
    if spec.get("type") == "zpk":
        return {
            "type": "zpk",
            "zeros": spec.get("zeros", []),
            "poles": spec.get("poles", []),
            "gain": spec.get("gain"),
            "K": spec.get("K"),
            "L": spec.get("L", 0.0),
        }
    sz = spec.get("source_zpk")
    if isinstance(sz, dict) and "poles" in sz:
        return {
            "type": "zpk",
            "zeros": sz.get("zeros", []),
            "poles": sz.get("poles", []),
            "gain": sz.get("gain"),
            "K": spec.get("K"),
            "L": spec.get("L", 0.0),
        }
    return None


def _find_real_pole(poles: list[Any], nominal: float) -> int:
    """在 zpk 极点表里找与 nominal 匹配的实极点，返回下标；找不到当场拒绝。"""

    for i, item in enumerate(poles):
        if isinstance(item, list) and len(item) == 2:
            re, im = item[0], item[1]
        else:
            re, im = item, 0.0
        if isinstance(re, bool) or isinstance(im, bool):
            continue
        if not isinstance(re, (int, float)) or not isinstance(im, (int, float)):
            continue
        if abs(float(im)) <= 1e-12 and abs(float(re) - nominal) <= _POLE_MATCH_TOL * max(1.0, abs(nominal)):
            return i
    raise ServiceError(f"对象中找不到与 nominal={nominal} 对应的实极点")


def _synthetic_quotient(den: list[float], p: float) -> list[float]:
    """分母对 (s - p) 做综合除法，返回商（降幂）；余数不可忽略说明 p 不是根。"""

    q = [den[0]]
    for c in den[1:]:
        q.append(c + q[-1] * p)
    remainder = q.pop()
    scale = max(1.0, max(abs(c) for c in den) * max(1.0, abs(p)) ** (len(den) - 1))
    if abs(remainder) > _POLE_MATCH_TOL * scale:
        raise ServiceError(f"nominal={p} 不是分母的根：找不到与该值对应的实极点")
    return q


class PoleMover:
    """把名义极点搬到区间内的任意位置，其余零极点保持标称值不动。

    zpk：在极点表里替换匹配到的那个实极点再展开——与直接提交改动后的
    zpk 描述走同一条代码路径，系数逐位一致；
    poly：分母对 (s - nominal) 综合除法取商，再乘回 (s - p)。
    p == nominal 时原样返回原描述，保证退化情形与名义读数逐项相等。
    """

    def __init__(self, spec: dict[str, Any], pole: PoleInterval):
        self._nominal = pole.nominal
        self._original_spec = spec
        zpk = _zpk_view(spec)
        if zpk is not None:
            self._kind = "zpk"
            self._zpk = zpk
            self._index = _find_real_pole(zpk["poles"], pole.nominal)
        else:
            self._kind = "poly"
            plant = validate_plant(spec)
            self._num = list(plant.num)
            self._den = list(plant.den)
            self._K, self._L = plant.K, plant.L
            self._quotient = _synthetic_quotient(self._den, pole.nominal)

    def spec_at(self, p: float) -> dict[str, Any]:
        if p == self._nominal:
            # 退化捷径：不重建，逐项等于名义对象。
            return self._original_spec
        if self._kind == "zpk":
            poles = [list(item) if isinstance(item, list) else item for item in self._zpk["poles"]]
            poles[self._index] = [p, 0.0]
            return {**self._zpk, "poles": poles}
        # poly：den_new(s) = quotient(s) * (s - p)
        den_new = [0.0] * (len(self._quotient) + 1)
        for i, c in enumerate(self._quotient):
            den_new[i] += c
            den_new[i + 1] -= c * p
        return {"type": "poly", "num": list(self._num), "den": den_new,
                "K": self._K, "L": self._L}


class PlantFamily:
    """一个参数族：名义对象 + 不确定性说明。

    负责把族内一点 (K, L, pole) 映射成规范化 Plant，并给出单调维度
    （K、L）各自的最坏端/温和端；最坏化搜索本身在 robust.py。
    """

    def __init__(self, spec: dict[str, Any], unc: Uncertainty):
        self.base = validate_plant(spec)  # 名义对象先过一遍语义检查
        self.unc = unc
        self._spec = spec
        self._mover = PoleMover(spec, unc.pole) if unc.pole is not None else None

    @property
    def K_worst(self) -> float:
        """增益单调维度的最坏端：区间上界（无不确定则取名义值）。"""

        return self.unc.K.hi if self.unc.K is not None else self.base.K

    @property
    def K_mild(self) -> float:
        return self.unc.K.lo if self.unc.K is not None else self.base.K

    @property
    def L_worst(self) -> float:
        """延迟单调维度的最坏端：区间上界（无不确定则取名义值）。"""

        return self.unc.L.hi if self.unc.L is not None else self.base.L

    @property
    def L_mild(self) -> float:
        return self.unc.L.lo if self.unc.L is not None else self.base.L

    @property
    def pole_interval(self) -> tuple[float, float] | None:
        if self.unc.pole is None:
            return None
        return (self.unc.pole.lo, self.unc.pole.hi)

    def plant_at(self, K: float, L: float, pole: float | None = None) -> Plant:
        spec = self._spec
        if self._mover is not None and pole is not None:
            spec = self._mover.spec_at(pole)
        return validate_plant(spec, K_override=K, L_override=L)
