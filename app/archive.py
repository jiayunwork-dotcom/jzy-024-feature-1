"""具名对象档：本地 JSON 文件存储，不另起数据库进程。

持久化的是对象档本身（分子分母或零极点、K、L）；扫频结果不建流水。
写盘采用临时文件 + os.replace 原子替换；进程重启后档仍在。

落盘统一存规范化后的实系数多项式；若对象最初按 ZPK 提交，同时保留原始
零点/极点/增益，列档时一并还回。
"""

from __future__ import annotations

import json
import os
from typing import Any

from .errors import ServiceError
from .validation import validate_name, validate_plant


class PlantArchive:
    def __init__(self, directory: str):
        self.directory = directory
        os.makedirs(self.directory, exist_ok=True)

    def _path(self, name: str) -> str:
        name = validate_name(name)
        return os.path.join(self.directory, f"{name}.json")

    def exists(self, name: str) -> bool:
        return os.path.isfile(self._path(name))

    def save(self, name: str, spec: dict[str, Any]) -> dict[str, Any]:
        """新增或整档替换。写入前先跑语义检查，非法档不落盘。"""

        path = self._path(name)
        plant = validate_plant(spec)
        record: dict[str, Any] = {
            "name": name,
            "type": "poly",
            "num": list(plant.num),
            "den": list(plant.den),
            "K": plant.K,
            "L": plant.L,
        }
        if plant.form == "zpk":
            record["source_zpk"] = {
                "zeros": spec.get("zeros", []),
                "poles": spec.get("poles", []),
                "gain": spec.get("gain"),
            }
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
        return record

    def get_spec(self, name: str) -> dict[str, Any]:
        path = self._path(name)
        if not os.path.isfile(path):
            raise ServiceError(f"对象档 '{name}' 不存在", status_code=404)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                spec = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            raise ServiceError(f"对象档 '{name}' 已损坏：{exc}")
        # 读盘时再校验一次：损坏/旧档当场说明，绝不给假读数。
        validate_plant(spec)
        return spec

    def load(self, name: str):
        return validate_plant(self.get_spec(name))

    def delete(self, name: str) -> None:
        path = self._path(name)
        if not os.path.isfile(path):
            raise ServiceError(f"对象档 '{name}' 不存在", status_code=404)
        os.remove(path)

    def list(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for fname in sorted(os.listdir(self.directory)):
            if not fname.endswith(".json"):
                continue
            name = fname[:-5]
            try:
                spec = self.get_spec(name)
            except ServiceError:
                continue
            plant = validate_plant(spec)
            entry: dict[str, Any] = {
                "name": name,
                "type": "poly",
                "num": list(plant.num),
                "den": list(plant.den),
                "K": plant.K,
                "L": plant.L,
            }
            if "source_zpk" in spec:
                entry["zpk"] = spec["source_zpk"]
            out.append(entry)
        return out
