"""Array validation and centered ridge prediction."""
from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike, NDArray
FloatArray = NDArray[np.float64]

def _responses(values: ArrayLike) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 4 or min(array.shape) < 2:
        raise ValueError(
            "responses must be participant-by-visit-by-node-by-feature with all axes at least two"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("responses must be finite")
    return array

def _indices(
    values: tuple[int, ...],
    *,
    limit: int,
    name: str,
) -> tuple[int, ...]:
    if (
        not values
        or tuple(sorted(set(values))) != values
        or any(not isinstance(value, (int, np.integer)) or not 0 <= value < limit for value in values)
    ):
        raise ValueError(f"{name} must be unique increasing valid indices")
    return tuple(int(value) for value in values)

def _ridge_predict(
    x: FloatArray,
    y: FloatArray,
    query: FloatArray,
    ridge: float,
) -> FloatArray:
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    centered_x = x - x_mean
    centered_y = y - y_mean
    centered_query = query - x_mean
    if centered_x.shape[1] <= centered_x.shape[0]:
        coefficients = np.linalg.solve(
            centered_x.T @ centered_x + ridge * np.eye(centered_x.shape[1]),
            centered_x.T @ centered_y,
        )
        return y_mean + centered_query @ coefficients
    dual = np.linalg.solve(
        centered_x @ centered_x.T + ridge * np.eye(centered_x.shape[0]),
        centered_y,
    )
    return y_mean + (centered_query @ centered_x.T) @ dual
