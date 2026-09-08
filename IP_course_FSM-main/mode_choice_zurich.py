from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import MODE_CHOICE_FILE, MODE_ORDER, NO_PATH_TIME_MIN

UNAVAILABLE_UTILITY = -1e9


def load_mode_choice_parameters(path: str | Path = MODE_CHOICE_FILE) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload.get("parameters", payload))


def _matrix(value, index: pd.Index, columns: pd.Index, fill: float) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame(fill, index=index, columns=columns, dtype=float)
    frame = value.copy()
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)
    return frame.reindex(index=index, columns=columns, fill_value=fill).astype(float)


def _availability(*matrices: np.ndarray, no_path: float) -> np.ndarray:
    available = np.ones(matrices[0].shape, dtype=bool)
    for matrix in matrices:
        available &= np.isfinite(matrix) & (matrix < no_path)
    return available


def mode_split_aggregated(
    travel_times: dict[str, pd.DataFrame],
    lengths: dict[str, pd.DataFrame],
    total_od: pd.DataFrame,
    *,
    car_affinity: float = 1.0,
    walk_affinity: float = 1.0,
    bike_affinity: float = 1.0,
    pt_affinity: float = 1.0,
    walk_allowed_mask: pd.DataFrame | None = None,
    parameters: dict | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Allocate every OD cell across five modes with a multinomial logit model."""
    p = dict(parameters or load_mode_choice_parameters())
    no_path = float(p.get("NO_PATH_TIME_MIN", NO_PATH_TIME_MIN))
    max_utility_clip = float(p.get("MAX_UTILITY_CLIP", 700.0))
    index = total_od.index.astype(str)
    columns = total_od.columns.astype(str)
    demand = _matrix(total_od, index, columns, 0.0)

    demand_values = demand.to_numpy(dtype=float, copy=False)
    drive = _matrix(travel_times.get("drive"), index, columns, no_path).to_numpy(dtype=float, copy=False)
    walk = _matrix(travel_times.get("walk"), index, columns, no_path).to_numpy(dtype=float, copy=False)
    bike = _matrix(travel_times.get("bike"), index, columns, no_path).to_numpy(dtype=float, copy=False)
    pt_walk_ivt = _matrix(travel_times.get("ivt_pt_walk"), index, columns, no_path).to_numpy(dtype=float, copy=False)
    pt_walk_ovt = _matrix(travel_times.get("ovt_pt_walk"), index, columns, no_path).to_numpy(dtype=float, copy=False)
    pt_bike_ivt = _matrix(travel_times.get("ivt_pt_bike"), index, columns, no_path).to_numpy(dtype=float, copy=False)
    pt_bike_ovt = _matrix(travel_times.get("ovt_pt_bike"), index, columns, no_path).to_numpy(dtype=float, copy=False)
    drive_distance = _matrix(lengths.get("drive"), index, columns, 0.0).to_numpy(dtype=float, copy=False)
    pt_walk_distance = _matrix(lengths.get("pt_walk"), index, columns, 0.0).to_numpy(dtype=float, copy=False)
    pt_bike_distance = _matrix(lengths.get("pt_bike"), index, columns, 0.0).to_numpy(dtype=float, copy=False)
    transfer_walk = _matrix(travel_times.get("transfer_count_pt_walk"), index, columns, 0.0).to_numpy(dtype=float, copy=False)
    transfer_bike = _matrix(travel_times.get("transfer_count_pt_bike"), index, columns, 0.0).to_numpy(dtype=float, copy=False)

    car_cost = drive_distance / 1000.0 * float(p["CAR_COST_CHF_PER_KM"])
    pt_walk_fare = np.minimum(
        float(p["PT_MIN_FARE_CHF"])
        + pt_walk_distance / 1000.0 * float(p["PT_FARE_CHF_PER_KM"]),
        float(p["PT_MAX_FARE_CHF"]),
    )
    pt_bike_fare = np.minimum(
        float(p["PT_MIN_FARE_CHF"])
        + pt_bike_distance / 1000.0 * float(p["PT_FARE_CHF_PER_KM"]),
        float(p["PT_MAX_FARE_CHF"]),
    )

    # Affinities are positive multipliers on odds, implemented as ASC shifts.
    asc = {
        "drive": float(p["ASC_CAR"]) + np.log(max(float(car_affinity), 1e-6)),
        "walk": float(p["ASC_WALK"]) + np.log(max(float(walk_affinity), 1e-6)),
        "bike": float(p["ASC_BIKE"]) + np.log(max(float(bike_affinity), 1e-6)),
        "pt_walk": float(p["ASC_PT_WALK"]) + np.log(max(float(pt_affinity), 1e-6)),
        "pt_bike": float(p["ASC_PT_BIKE"]) + np.log(max(float(pt_affinity), 1e-6)),
    }
    utility_values = {
        "drive": asc["drive"] + float(p["B_CAR_TIME"]) * drive + float(p["B_COST"]) * car_cost,
        "walk": asc["walk"] + float(p["B_WALK_TIME"]) * walk,
        "bike": asc["bike"] + float(p["B_BIKE_TIME"]) * bike,
        "pt_walk": (
            asc["pt_walk"]
            + float(p["B_PT_IVT"]) * pt_walk_ivt
            + float(p["B_PT_OVT"]) * pt_walk_ovt
            + float(p["B_PT_FARE"]) * pt_walk_fare
            + float(p["B_TRANSFER"]) * transfer_walk
        ),
        "pt_bike": (
            asc["pt_bike"]
            + float(p["B_PT_IVT"]) * pt_bike_ivt
            + float(p["B_PT_OVT"]) * pt_bike_ovt
            + float(p["B_PT_FARE"]) * pt_bike_fare
            + float(p["B_TRANSFER"]) * transfer_bike
        ),
    }
    available = {
        "drive": _availability(drive, no_path=no_path),
        "walk": _availability(walk, no_path=no_path),
        "bike": _availability(bike, no_path=no_path),
        "pt_walk": _availability(pt_walk_ivt, pt_walk_ovt, no_path=no_path),
        "pt_bike": _availability(pt_bike_ivt, pt_bike_ovt, no_path=no_path),
    }
    if walk_allowed_mask is not None:
        allowed = walk_allowed_mask.reindex(index=index, columns=columns, fill_value=False).to_numpy(dtype=bool)
        available["walk"] &= allowed

    utility_stack = np.stack([utility_values[m] for m in MODE_ORDER])
    availability_stack = np.stack([available[m] for m in MODE_ORDER])
    utility_stack = np.where(availability_stack, utility_stack, UNAVAILABLE_UTILITY)
    row_max = np.max(utility_stack, axis=0)
    exp_utility = np.where(
        availability_stack,
        np.exp(
            np.clip(
                utility_stack - row_max,
                -max_utility_clip,
                max_utility_clip,
            )
        ),
        0.0,
    )
    denominator = exp_utility.sum(axis=0)
    probabilities = np.divide(
        exp_utility,
        denominator,
        out=np.zeros_like(exp_utility),
        where=denominator > 0,
    )
    fallback = denominator <= 0
    if np.any(fallback):
        probabilities[:, fallback] = 0.0
        drive_available = availability_stack[0] & fallback
        probabilities[0, drive_available] = 1.0
        remaining = fallback & ~drive_available
        if np.any(remaining):
            coordinates = np.argwhere(remaining)
            best_modes = np.argmax(utility_stack[:, remaining], axis=0)
            for best_mode, (row, column) in zip(best_modes, coordinates):
                probabilities[best_mode, row, column] = 1.0

    od_by_mode = {
        mode: pd.DataFrame(
            demand_values * probabilities[position],
            index=index,
            columns=columns,
        )
        for position, mode in enumerate(MODE_ORDER)
    }
    utilities = {
        mode: pd.DataFrame(
            np.where(available[mode], utility_values[mode], UNAVAILABLE_UTILITY),
            index=index,
            columns=columns,
        )
        for mode in MODE_ORDER
    }
    return od_by_mode, utilities
