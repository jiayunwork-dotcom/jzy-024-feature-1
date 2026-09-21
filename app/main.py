"""FastAPI 入口：对象档管理 + 开环扫频裕度读数。

启动时载入一套「二阶环节串联一个积分器」的对象档：
    G0(s) = 10 / [ s (s+1) (s+10) ]
其幅值穿越约 0.78 rad/s、相位裕度约 47°（几十度量级、为正），
ω_π = √10 rad/s，GM = 11（约 20.8 dB），穿越附近的幅相可手算核对。
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .archive import PlantArchive
from .errors import ServiceError
from .frf import TransferFunction
from .margins import analyze, to_payload
from .models import PlantUpsertRequest, SweepRequest
from .validation import validate_grid, validate_plant

PLANT_DIR = os.environ.get("PLANT_DIR", os.path.join(os.getcwd(), "data", "plants"))

# 启动对象：10 / [s(s+1)(s+10)] —— 两个实极点串一个积分器。
STARTUP_NAME = "startup_plant"
STARTUP_SPEC: dict[str, Any] = {
    "type": "zpk",
    "zeros": [],
    "poles": [[0.0, 0.0], [-1.0, 0.0], [-10.0, 0.0]],
    "gain": 10.0,
    "K": 1.0,
    "L": 0.0,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.archive = PlantArchive(PLANT_DIR)
    # 启动档不存在才写入；已存在（持久卷重启）则保持用户现状。
    if not app.state.archive.exists(STARTUP_NAME):
        app.state.archive.save(STARTUP_NAME, STARTUP_SPEC)
    yield


app = FastAPI(
    title="开环频域稳定裕度服务",
    description="对象档管理 + 对数网格扫频，返回相位裕度/幅值裕度。范围限定开环频域裕度。",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(ServiceError)
async def service_error_handler(_: Request, exc: ServiceError) -> JSONResponse:
    # 当场说明原因就停，响应里不带任何扫频读数。
    return JSONResponse(status_code=exc.status_code, content={"error": exc.reason})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "open-loop-margin",
        "scope": "开环频域稳定裕度（相位裕度/幅值裕度）",
        "startup_plant": STARTUP_NAME,
        "endpoints": [
            "GET  /api/plants",
            "POST /api/plants/{name}",
            "PUT  /api/plants/{name}",
            "GET  /api/plants/{name}",
            "DELETE /api/plants/{name}",
            "POST /api/sweep",
        ],
    }


@app.get("/api/plants")
def list_plants() -> dict[str, list[dict[str, Any]]]:
    return {"plants": app.state.archive.list()}


@app.post("/api/plants/{name}", status_code=201)
def create_plant(name: str, body: PlantUpsertRequest) -> dict[str, Any]:
    if app.state.archive.exists(name):
        raise ServiceError(f"对象档 '{name}' 已存在，请用 PUT 整档替换", status_code=409)
    record = app.state.archive.save(name, body.spec)
    return {"saved": record}


@app.put("/api/plants/{name}")
def replace_plant(name: str, body: PlantUpsertRequest) -> dict[str, Any]:
    record = app.state.archive.save(name, body.spec)  # 校验通过后原子替换
    return {"saved": record}


@app.get("/api/plants/{name}")
def get_plant(name: str) -> dict[str, Any]:
    return app.state.archive.get_spec(name)


@app.delete("/api/plants/{name}")
def delete_plant(name: str) -> dict[str, str]:
    app.state.archive.delete(name)
    return {"deleted": name}


@app.post("/api/sweep")
def sweep(body: SweepRequest) -> dict[str, Any]:
    # 1) 对象：点名字调档，或当次内联。
    if isinstance(body.plant, str):
        plant = app.state.archive.load(body.plant)
        if body.K is not None or body.L is not None:
            # 当次覆盖 K/L：重新规范化（仅作用于本次扫频，档本身不动）。
            spec = app.state.archive.get_spec(body.plant)
            plant = validate_plant(spec, K_override=body.K, L_override=body.L)
    else:
        plant = validate_plant(body.plant, K_override=body.K, L_override=body.L)

    # 2) 网格：非空、严格递增、全为正。
    grid = validate_grid(body.freqs)

    # 3) 扫频 + 穿越搜索（落在网格点之间一律对分加密）。
    tf = TransferFunction(plant)
    try:
        result = analyze(tf, grid)
    except ZeroDivisionError as exc:
        raise ServiceError(f"扫频失败：{exc}")

    payload = to_payload(result)
    payload["plant"] = {
        "num": list(plant.num),
        "den": list(plant.den),
        "K": plant.K,
        "L": plant.L,
    }
    return payload
