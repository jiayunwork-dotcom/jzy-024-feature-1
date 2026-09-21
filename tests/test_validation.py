"""入参检查：非真有理、网格非法、系数非有限数等必须当场拒绝。"""

import pytest

from app.errors import ServiceError
from app.validation import validate_grid, validate_plant


def test_nonproper_transfer_function_rejected():
    # 分子 2 阶、分母 1 阶
    spec = {"type": "poly", "num": [1.0, 2.0, 3.0], "den": [1.0, 1.0], "K": 1.0, "L": 0.0}
    with pytest.raises(ServiceError) as ei:
        validate_plant(spec)
    assert "非真有理" in ei.value.reason


def test_proper_equal_degree_accepted():
    spec = {"type": "poly", "num": [1.0, 2.0], "den": [3.0, 4.0, 5.0], "K": 1.0, "L": 0.0}
    p = validate_plant(spec)
    assert p.num == (1.0, 2.0)


def test_empty_grid_rejected():
    with pytest.raises(ServiceError) as ei:
        validate_grid([])
    assert "非空" in ei.value.reason


def test_nonpositive_frequency_rejected():
    with pytest.raises(ServiceError):
        validate_grid([0.0, 1.0])
    with pytest.raises(ServiceError):
        validate_grid([1.0, -2.0])


def test_non_increasing_grid_rejected():
    with pytest.raises(ServiceError) as ei:
        validate_grid([1.0, 2.0, 2.0, 3.0])
    assert "严格递增" in ei.value.reason
    with pytest.raises(ServiceError):
        validate_grid([3.0, 2.0, 1.0])


def test_non_finite_coefficients_rejected():
    with pytest.raises(ServiceError):
        validate_plant({"type": "poly", "num": [1.0, float("nan")],
                        "den": [1.0, 2.0], "K": 1.0})
    with pytest.raises(ServiceError):
        validate_plant({"type": "poly", "num": [1.0], "den": [1.0, float("inf")],
                        "K": 1.0})
    with pytest.raises(ServiceError):
        validate_plant({"type": "poly", "num": [1.0], "den": [1.0, 2.0],
                        "K": float("nan")})
    with pytest.raises(ServiceError):
        validate_plant({"type": "poly", "num": [1.0], "den": [1.0, 2.0],
                        "K": -1.0})
    with pytest.raises(ServiceError):
        validate_plant({"type": "poly", "num": [1.0], "den": [1.0, 2.0],
                        "K": 1.0, "L": -0.1})


def test_bool_is_not_a_number():
    with pytest.raises(ServiceError):
        validate_plant({"type": "poly", "num": [True], "den": [1.0, 2.0], "K": 1.0})


def test_zero_numerator_rejected():
    with pytest.raises(ServiceError):
        validate_plant({"type": "poly", "num": [0.0, 0.0], "den": [1.0, 2.0], "K": 1.0})


def test_zpk_nonproper_rejected():
    with pytest.raises(ServiceError) as ei:
        validate_plant({"type": "zpk", "zeros": [-1.0, -2.0], "poles": [-3.0],
                        "gain": 1.0, "K": 1.0})
    assert "非真有理" in ei.value.reason
