from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    CAR_TERMINAL_CANTON_MIN,
    CAR_TERMINAL_CITY_MIN,
    DRIVE_CONNECTOR_CANTON_MIN,
    DRIVE_CONNECTOR_CITY_MIN,
    DRIVE_INTRAZONAL_MIN,
    EBIKE_SPEED_MULTIPLIER,
    MAX_WALK_DISTANCE_KM,
    NO_PATH_TIME_MIN,
    PT_ACCESS_FLOOR_MIN,
    PT_EGRESS_FLOOR_MIN,
    PT_INTRAZONAL_MIN,
    PT_TRANSFER_PHYSICAL_FLOOR_MIN,
    SKIM_PACKAGE_FILE,
)
from input_packages import align_matrix, load_package


NUMERIC_SKIMS = {
    "drive": "drive_time_min.parquet",
    "walk": "walk_time_min.parquet",
    "bike": "bike_time_min.parquet",
    "ivt_pt_walk": "pt_walk_ivt_min.parquet",
    "access_pt_walk": "pt_walk_access_min.parquet",
    "egress_pt_walk": "pt_walk_egress_min.parquet",
    "initial_wait_pt_walk": "pt_walk_initial_wait_min.parquet",
    "transfer_physical_pt_walk": "pt_walk_transfer_physical_min.parquet",
    "transfer_wait_pt_walk": "pt_walk_transfer_wait_min.parquet",
    "transfer_count_pt_walk": "pt_walk_transfer_count.parquet",
    "ivt_pt_bike": "pt_bike_ivt_min.parquet",
    "access_pt_bike": "pt_bike_access_min.parquet",
    "egress_pt_bike": "pt_bike_egress_min.parquet",
    "initial_wait_pt_bike": "pt_bike_initial_wait_min.parquet",
    "transfer_physical_pt_bike": "pt_bike_transfer_physical_min.parquet",
    "transfer_wait_pt_bike": "pt_bike_transfer_wait_min.parquet",
    "transfer_count_pt_bike": "pt_bike_transfer_count.parquet",
}

STRING_SKIMS = {
    "chosen_origin_stop_pt_walk": "pt_walk_chosen_origin_stop.parquet",
    "chosen_destination_stop_pt_walk": "pt_walk_chosen_destination_stop.parquet",
    "chosen_origin_stop_pt_bike": "pt_bike_chosen_origin_stop.parquet",
    "chosen_destination_stop_pt_bike": "pt_bike_chosen_destination_stop.parquet",
}

LENGTH_SKIMS = {
    "drive": "drive_distance_m.parquet",
    "walk": "walk_distance_m.parquet",
    "bike": "bike_distance_m.parquet",
    "pt_walk": "pt_walk_distance_m.parquet",
    "pt_bike": "pt_bike_distance_m.parquet",
}


def rebuild_pt_out_of_vehicle_times(
    travel_times: dict[str, pd.DataFrame],
    modes: tuple[str, ...] = ("walk", "bike"),
) -> None:
    for mode in modes:
        pieces = [
            travel_times[f"access_pt_{mode}"],
            travel_times[f"egress_pt_{mode}"],
            travel_times[f"initial_wait_pt_{mode}"],
            travel_times[f"transfer_physical_pt_{mode}"],
            travel_times[f"transfer_wait_pt_{mode}"],
        ]
        total = pieces[0].copy()
        for piece in pieces[1:]:
            total = total + piece
        ivt = travel_times[f"ivt_pt_{mode}"]
        no_path = ~np.isfinite(ivt.to_numpy(dtype=float)) | (ivt.to_numpy(dtype=float) >= NO_PATH_TIME_MIN)
        for piece in pieces:
            piece_values = piece.to_numpy(dtype=float)
            no_path |= ~np.isfinite(piece_values) | (piece_values >= NO_PATH_TIME_MIN)
        values = total.to_numpy(dtype=float)
        values[no_path] = NO_PATH_TIME_MIN
        ovt = pd.DataFrame(values, index=ivt.index, columns=ivt.columns)
        travel_times[f"ovt_pt_{mode}"] = ovt
        travel_times[f"pt_{mode}"] = ivt + ovt


def load_travel_times(
    zones: pd.DataFrame,
    package_path=SKIM_PACKAGE_FILE,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Load raw prepared skims. Perceived-time policies are applied separately."""
    package = load_package(package_path, "skims")
    packaged_times = package.get("travel_times", {})
    packaged_lengths = package.get("lengths", {})
    zone_ids = pd.Index(zones["grid_id"].astype(str), dtype="object")
    travel_times = {
        key: align_matrix(
            packaged_times[key],
            zone_ids,
            numeric=True,
            fill_value=NO_PATH_TIME_MIN,
        )
        for key in NUMERIC_SKIMS
    }
    travel_times.update(
        {
            key: align_matrix(
                packaged_times[key],
                zone_ids,
                numeric=False,
                fill_value="",
            )
            for key in STRING_SKIMS
        }
    )
    lengths = {
        key: align_matrix(
            packaged_lengths[key],
            zone_ids,
            numeric=True,
            fill_value=np.inf,
        )
        for key in LENGTH_SKIMS
    }
    rebuild_pt_out_of_vehicle_times(travel_times)
    return travel_times, lengths


def _valid(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values) & (values < NO_PATH_TIME_MIN)


def _apply_drive_perceived_times(
    drive_time: pd.DataFrame,
    zones: pd.DataFrame,
    drive_connector_city_min: float = DRIVE_CONNECTOR_CITY_MIN,
) -> pd.DataFrame:
    zone_ids = drive_time.index.astype(str)
    levels = (
        zones.set_index(zones["grid_id"].astype(str))
        .reindex(zone_ids)["Level"]
        .astype(str)
    )
    is_city = levels.eq("Quartier").to_numpy(dtype=bool)
    endpoint = np.where(is_city, float(drive_connector_city_min), DRIVE_CONNECTOR_CANTON_MIN)
    values = drive_time.to_numpy(dtype=float, copy=True)
    valid = _valid(values)
    values = np.where(valid, values + endpoint[:, None] + endpoint[None, :], values)

    diagonal = np.diag_indices_from(values)
    values[diagonal] = np.maximum(values[diagonal], DRIVE_INTRAZONAL_MIN)

    city_involved = np.logical_or.outer(is_city, is_city)
    terminal = np.where(city_involved, CAR_TERMINAL_CITY_MIN, CAR_TERMINAL_CANTON_MIN)
    valid = _valid(values)
    values = np.where(valid, values + terminal, values)
    return pd.DataFrame(values, index=zone_ids, columns=drive_time.columns.astype(str))


def _apply_pt_component_floors(travel_times: dict[str, pd.DataFrame]) -> None:
    for mode in ("walk", "bike"):
        ivt = travel_times[f"ivt_pt_{mode}"]
        ovt = travel_times[f"ovt_pt_{mode}"]
        valid_non_diagonal = _valid(ivt.to_numpy(dtype=float)) & _valid(ovt.to_numpy(dtype=float))
        valid_non_diagonal &= ivt.index.to_numpy(dtype=object)[:, None] != ivt.columns.to_numpy(dtype=object)[None, :]

        for component, floor in (
            ("access", PT_ACCESS_FLOOR_MIN),
            ("egress", PT_EGRESS_FLOOR_MIN),
        ):
            key = f"{component}_pt_{mode}"
            values = travel_times[key].to_numpy(dtype=float, copy=True)
            eligible = valid_non_diagonal & _valid(values) & (values < float(floor))
            values[eligible] = float(floor)
            travel_times[key] = pd.DataFrame(values, index=ivt.index, columns=ivt.columns)

        physical_key = f"transfer_physical_pt_{mode}"
        physical = travel_times[physical_key].to_numpy(dtype=float, copy=True)
        count = travel_times[f"transfer_count_pt_{mode}"].to_numpy(dtype=float)
        eligible = (
            valid_non_diagonal
            & (count > 0.0)
            & _valid(physical)
            & (physical < PT_TRANSFER_PHYSICAL_FLOOR_MIN)
        )
        physical[eligible] = PT_TRANSFER_PHYSICAL_FLOOR_MIN
        travel_times[physical_key] = pd.DataFrame(physical, index=ivt.index, columns=ivt.columns)

    rebuild_pt_out_of_vehicle_times(travel_times)


def apply_perceived_time_policies(
    raw_travel_times: dict[str, pd.DataFrame],
    zones: pd.DataFrame,
    drive_connector_city_min: float = DRIVE_CONNECTOR_CITY_MIN,
) -> dict[str, pd.DataFrame]:
    """Apply connector, terminal, intrazonal, and PT component assumptions."""
    adjusted = {
        key: value.copy() if isinstance(value, pd.DataFrame) else value
        for key, value in raw_travel_times.items()
    }
    adjusted["drive"] = _apply_drive_perceived_times(
        adjusted["drive"],
        zones,
        drive_connector_city_min=drive_connector_city_min,
    )
    _apply_pt_component_floors(adjusted)
    for mode in ("walk", "bike"):
        key = f"ovt_pt_{mode}"
        values = adjusted[key].to_numpy(dtype=float, copy=True)
        diagonal = np.diag_indices_from(values)
        values[diagonal] = np.maximum(values[diagonal], PT_INTRAZONAL_MIN)
        adjusted[key] = pd.DataFrame(values, index=adjusted[key].index, columns=adjusted[key].columns)
        adjusted[f"pt_{mode}"] = adjusted[f"ivt_pt_{mode}"] + adjusted[key]
    return adjusted


def apply_ebike_share(
    travel_times: dict[str, pd.DataFrame],
    ebike_share: float,
    speed_multiplier: float = EBIKE_SPEED_MULTIPLIER,
) -> tuple[dict[str, pd.DataFrame], dict]:
    share = float(np.clip(float(ebike_share or 0.0), 0.0, 1.0))
    ebike_time_factor = 1.0 / max(float(speed_multiplier), 1e-9)
    time_multiplier = (1.0 - share) + share * ebike_time_factor
    adjusted = {
        key: value.copy() if isinstance(value, pd.DataFrame) else value
        for key, value in travel_times.items()
    }
    scaled = []
    missing_components = []
    if share > 0.0:
        if isinstance(adjusted.get("bike"), pd.DataFrame):
            bike = adjusted["bike"].to_numpy(dtype=float, copy=True)
            bike_available = _valid(bike)
            bike[bike_available] *= time_multiplier
            adjusted["bike"] = pd.DataFrame(
                bike,
                index=adjusted["bike"].index,
                columns=adjusted["bike"].columns,
            )
            scaled.append("bike")

        component_keys = (
            "access_pt_bike",
            "egress_pt_bike",
            "initial_wait_pt_bike",
            "transfer_physical_pt_bike",
            "transfer_wait_pt_bike",
        )
        required_keys = ("ivt_pt_bike", *component_keys)
        missing_components = [
            key for key in required_keys if not isinstance(adjusted.get(key), pd.DataFrame)
        ]
        if not missing_components:
            ivt = adjusted["ivt_pt_bike"]
            available = _valid(ivt.to_numpy(dtype=float))
            for key in component_keys:
                available &= _valid(adjusted[key].to_numpy(dtype=float))
            non_diagonal = (
                ivt.index.to_numpy(dtype=object)[:, None]
                != ivt.columns.to_numpy(dtype=object)[None, :]
            )
            for key, floor_min in (
                ("access_pt_bike", PT_ACCESS_FLOOR_MIN),
                ("egress_pt_bike", PT_EGRESS_FLOOR_MIN),
            ):
                values = adjusted[key].to_numpy(dtype=float, copy=True)
                values[available] *= time_multiplier
                floor_mask = available & non_diagonal & (values < float(floor_min))
                values[floor_mask] = float(floor_min)
                adjusted[key] = pd.DataFrame(values, index=ivt.index, columns=ivt.columns)
                scaled.append(key)

            rebuild_pt_out_of_vehicle_times(adjusted, modes=("bike",))
            ovt = adjusted["ovt_pt_bike"].to_numpy(dtype=float, copy=True)
            diagonal = ~non_diagonal
            intrazonal_floor = available & diagonal & (ovt < float(PT_INTRAZONAL_MIN))
            ovt[intrazonal_floor] = float(PT_INTRAZONAL_MIN)
            adjusted["ovt_pt_bike"] = pd.DataFrame(ovt, index=ivt.index, columns=ivt.columns)
            adjusted["pt_bike"] = adjusted["ivt_pt_bike"] + adjusted["ovt_pt_bike"]
    return adjusted, {
        "applied": bool(scaled),
        "ebike_share": share,
        "speed_multiplier": float(speed_multiplier),
        "time_multiplier": time_multiplier,
        "multiplier_basis": "travel_time_average",
        "scaled_time_keys": scaled,
        "missing_pt_bike_components": missing_components,
    }


def standalone_walk_mask(lengths: dict[str, pd.DataFrame], zones: pd.DataFrame) -> pd.DataFrame:
    """Allow every reachable standalone walk trip up to 5 km."""
    distance = lengths["walk"]
    values = distance.to_numpy(dtype=float)
    allowed = np.isfinite(values) & (values <= MAX_WALK_DISTANCE_KM * 1000.0)
    np.fill_diagonal(allowed, True)
    return pd.DataFrame(allowed, index=distance.index, columns=distance.columns)
