from __future__ import annotations

import gzip
import pickle
from pathlib import Path
from typing import Any

import pandas as pd


PACKAGE_VERSION = 1


def save_package(path: str | Path, kind: str, data: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "package_version": PACKAGE_VERSION,
        "kind": str(kind),
        "data": data,
    }
    with gzip.open(path, "wb", compresslevel=5) as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_package(path: str | Path, expected_kind: str) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing prepared input package: {path}")
    with gzip.open(path, "rb") as handle:
        payload = pickle.load(handle)
    if payload.get("package_version") != PACKAGE_VERSION:
        raise ValueError(f"Unsupported package version in {path}")
    if payload.get("kind") != expected_kind:
        raise ValueError(f"Expected a {expected_kind!r} package, found {payload.get('kind')!r}")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise ValueError(f"Package data must be a dictionary: {path}")
    return data


def align_matrix(
    frame: pd.DataFrame,
    zone_ids: pd.Index,
    *,
    numeric: bool,
    fill_value,
) -> pd.DataFrame:
    matrix = pd.DataFrame(frame).copy()
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    matrix = matrix.reindex(index=zone_ids, columns=zone_ids, fill_value=fill_value)
    if numeric:
        return matrix.astype(float)
    return matrix.fillna("").astype(str)
