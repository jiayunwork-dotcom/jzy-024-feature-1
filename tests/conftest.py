"""pytest 夹具：每个测试用独立临时档目录，互不污染。

PLANT_DIR 必须在导入 app.main 之前设置，所以本文件位于 conftest 顶部。
"""

import importlib
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("PLANT_DIR", str(tmp_path / "plants"))
    import app.main as main
    importlib.reload(main)  # 让 lifespan 重新按新 PLANT_DIR 建档
    from fastapi.testclient import TestClient

    with TestClient(main.app) as c:
        yield c


@pytest.fixture
def startup_spec():
    return {
        "type": "zpk",
        "zeros": [],
        "poles": [[0.0, 0.0], [-1.0, 0.0], [-10.0, 0.0]],
        "gain": 10.0,
        "K": 1.0,
        "L": 0.0,
    }


def loggrid(w_lo=0.01, w_hi=100.0, n=601):
    """对数间隔频率网格（严格递增、全为正）。"""

    import math

    return [10.0 ** (math.log10(w_lo) + i * (math.log10(w_hi) - math.log10(w_lo)) / (n - 1))
            for i in range(n)]
