"""HTTP 请求体模型（结构/类型层）。

语义规则（有限数、真有理性、网格合法性等）仍在 validation.py 里手工检查，
这里只负责把 JSON 收下来，并拦住 NaN/Infinity 这类 JSON 里不合法的数。
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, Field, field_validator

JsonValue = float | int | str | bool | list["JsonValue"] | dict[str, "JsonValue"] | None


class SweepRequest(BaseModel):
    # plant: 字符串（点具名档）或对象描述（poly / zpk），当次有效。
    plant: str | dict[str, Any]
    freqs: list[float]
    # 仅当次覆盖档内 K、L；不传就用档自身的值。
    K: float | None = None
    L: float | None = None

    @field_validator("freqs")
    @classmethod
    def _freqs_finite(cls, v: list[float]) -> list[float]:
        for w in v:
            if not isinstance(w, (int, float)) or isinstance(w, bool) or not math.isfinite(w):
                raise ValueError("频率点必须是有限数")
        return v

    @field_validator("K", "L")
    @classmethod
    def _override_finite(cls, v: float | None) -> float | None:
        if v is not None and not math.isfinite(v):
            raise ValueError("K / L 必须是有限数")
        return v


class PlantUpsertRequest(BaseModel):
    spec: dict[str, Any] = Field(..., description="poly 或 zpk 对象描述")
