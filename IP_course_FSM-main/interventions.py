from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    LOOKUP_PACKAGE_FILE,
    NO_PATH_TIME_MIN,
    PT_ACCESS_FLOOR_MIN,
    PT_EGRESS_FLOOR_MIN,
    PT_TRANSFER_PHYSICAL_FLOOR_MIN,
)
from input_packages import load_package
from travel_times import rebuild_pt_out_of_vehicle_times





def _as_list(value) -> list[dict]:
    if not value:
        return []
    return value if isinstance(value, list) else [value]


def _selector_zone_ids(selector: dict, zones: pd.DataFrame) -> set[str]:
    if not isinstance(selector, dict) or not selector:
        raise ValueError("Each area selector must contain at least one zone attribute.")
    selected = np.ones(len(zones), dtype=bool)
    for column, requested in selector.items():
        if column not in zones.columns:
            raise ValueError(f"Zone selector uses missing column: {column}")
        values = requested if isinstance(requested, (list, tuple, set)) else [requested]
        requested_values = {str(value) for value in values}
        selected &= zones[column].astype(str).isin(requested_values).to_numpy()
    zone_ids = set(zones.loc[selected, "grid_id"].astype(str))
    if not zone_ids:
        raise ValueError(f"Zone selector did not match any zones: {selector}")
    return zone_ids


def _od_mask(
    index: pd.Index,
    columns: pd.Index,
    intervention: dict,
    zones: pd.DataFrame | None = None,
) -> np.ndarray:
    mask = np.zeros((len(index), len(columns)), dtype=bool)
    row_position = {str(value): i for i, value in enumerate(index)}
    column_position = {str(value): i for i, value in enumerate(columns)}
    for pair in intervention.get("od_pairs", []):
        if len(pair) != 2:
            raise ValueError(f"OD pairs must contain one origin and destination: {pair}")
        origin, destination = str(pair[0]), str(pair[1])
        if origin in row_position and destination in column_position:
            mask[row_position[origin], column_position[destination]] = True
        if intervention.get("both_directions", True):
            if destination in row_position and origin in column_position:
                mask[row_position[destination], column_position[origin]] = True

    area_pairs = intervention.get("area_pairs", [])
    if area_pairs and zones is None:
        raise ValueError("Area-pair interventions require the model zone table.")
    for pair in area_pairs:
        if not isinstance(pair, dict) or "origin" not in pair or "destination" not in pair:
            raise ValueError(f"Area pairs need origin and destination selectors: {pair}")
        origins = _selector_zone_ids(pair["origin"], zones)
        destinations = _selector_zone_ids(pair["destination"], zones)
        origin_rows = np.array([str(value) in origins for value in index], dtype=bool)
        destination_columns = np.array(
            [str(value) in destinations for value in columns], dtype=bool
        )
        mask |= origin_rows[:, None] & destination_columns[None, :]
        if intervention.get("both_directions", True):
            destination_rows = np.array(
                [str(value) in destinations for value in index], dtype=bool
            )
            origin_columns = np.array(
                [str(value) in origins for value in columns], dtype=bool
            )
            mask |= destination_rows[:, None] & origin_columns[None, :]
    return mask


def _bike_highway_mask(travel_times, lengths, intervention, zones=None) -> np.ndarray:
    reference = travel_times["bike"]
    mask = _od_mask(reference.index, reference.columns, intervention, zones)
    distance_km = lengths["bike"].to_numpy(dtype=float) / 1000.0
    minimum = float(intervention.get("min_distance_km", 0.0))
    maximum = float(intervention.get("max_distance_km", np.inf))
    if minimum < 0.0 or maximum <= minimum:
        raise ValueError("Bike-highway distance limits must satisfy 0 <= minimum < maximum.")
    return (
        mask
        & np.isfinite(distance_km)
        & (distance_km >= minimum)
        & (distance_km <= maximum)
    )


def _railway_expansion_mask(travel_times, intervention, zones=None) -> np.ndarray:
    reference = travel_times["ivt_pt_walk"]
    return _od_mask(reference.index, reference.columns, intervention, zones)


def _percentage(effects: dict, key: str) -> float:
    value = float(effects.get(key, 0.0))
    if value < 0.0 or value >= 100.0:
        raise ValueError(f"{key} must be between 0 and 100")
    return value


def _reduction_factor(effects: dict, key: str) -> float:
    return 1.0 - _percentage(effects, key) / 100.0


def _speed_factor(effects: dict) -> float:
    increase = float(effects.get("speed_increase_pct", 0.0))
    if increase < 0.0:
        raise ValueError("speed_increase_pct cannot be negative")
    return 1.0 / (1.0 + increase / 100.0)


def _multiply_selected(
    frame: pd.DataFrame,
    mask: np.ndarray,
    factor: float,
    *,
    floor: float | None = None,
    time_matrix: bool = True,
) -> int:
    values = frame.to_numpy(copy=False)
    reachable = np.isfinite(values)
    if time_matrix:
        reachable &= values < NO_PATH_TIME_MIN
    selected = mask & reachable
    scaled = values[selected] * float(factor)
    if floor is not None:
        scaled = np.where(values[selected] > 0.0, np.maximum(scaled, float(floor)), scaled)
    values[selected] = scaled
    return int(selected.sum())


def _hub_stop_ids(
    lookups: dict[str, pd.DataFrame] | None,
    mode: str,
    zone_ids: set[str],
    stops_per_zone: int,
    configured_stop_ids=None,
) -> set[str]:
    if configured_stop_ids:
        values = (
            configured_stop_ids.get(mode, [])
            if isinstance(configured_stop_ids, dict)
            else configured_stop_ids
        )
        return {str(value) for value in values}
    if lookups is None:
        raise ValueError("A hub without explicit stop_ids requires the prepared stop lookups.")
    key = f"pt_zone_stop_access_{mode}"
    if key not in lookups:
        raise KeyError(f"The lookup package is missing {key}")
    candidates = lookups[key].copy()
    candidates["zone_id"] = candidates["zone_id"].astype(str)
    candidates["stop_id"] = candidates["stop_id"].astype(str)
    candidates["rank"] = pd.to_numeric(candidates["rank"], errors="coerce")
    selected = candidates[
        candidates["zone_id"].isin(zone_ids)
        & candidates["rank"].le(int(stops_per_zone))
    ]
    return set(selected["stop_id"])


def _hub_mode_masks(travel_times, intervention, lookups=None) -> dict[str, dict]:
    configured_stops = intervention.get("stop_ids")
    zones = intervention.get("zones", intervention.get("zone", []))
    zone_ids = {str(value) for value in (zones if isinstance(zones, list) else [zones])}
    zone_ids.discard("")
    stops_per_zone = int(intervention.get("stops_per_zone", 2))
    masks = {}
    for mode in ("walk", "bike"):
        stop_ids = _hub_stop_ids(
            lookups,
            mode,
            zone_ids,
            stops_per_zone,
            configured_stop_ids=configured_stops,
        )
        chosen_origin = travel_times[f"chosen_origin_stop_pt_{mode}"].to_numpy(dtype=str)
        chosen_destination = travel_times[f"chosen_destination_stop_pt_{mode}"].to_numpy(dtype=str)
        observed_stops = set(np.unique(chosen_origin)) | set(np.unique(chosen_destination))
        origin_mask = np.isin(chosen_origin, list(stop_ids))
        destination_mask = np.isin(chosen_destination, list(stop_ids))
        masks[mode] = {
            "origin": origin_mask,
            "destination": destination_mask,
            "path": origin_mask | destination_mask,
            "stop_ids": stop_ids,
            "matched_stop_ids": stop_ids.intersection(observed_stops),
        }
    return masks


def _apply_bike_highway(travel_times, lengths, intervention, zones=None) -> dict:
    effects = intervention.get("effects", {})
    mask = _bike_highway_mask(travel_times, lengths, intervention, zones)
    distance_factor = _reduction_factor(effects, "distance_reduction_pct")
    time_factor = (
        distance_factor
        * _reduction_factor(effects, "travel_time_reduction_pct")
        * _speed_factor(effects)
    )
    changed = _multiply_selected(travel_times["bike"], mask, time_factor)
    _multiply_selected(lengths["bike"], mask, distance_factor, time_matrix=False)
    return {
        "type": "bike_highway",
        "name": intervention.get("name", "bike highway"),
        "level": intervention.get("level"),
        "od_cells": changed,
        "time_factor": time_factor,
        "distance_factor": distance_factor,
    }


def _apply_railway_expansion(travel_times, lengths, intervention, zones=None) -> dict:
    effects = intervention.get("effects", {})
    mask = _railway_expansion_mask(travel_times, intervention, zones)
    distance_factor = _reduction_factor(effects, "distance_reduction_pct")
    ivt_factor = (
        distance_factor
        * _reduction_factor(effects, "travel_time_reduction_pct")
        * _speed_factor(effects)
    )
    component_factors = {
        "initial_wait": _reduction_factor(effects, "initial_wait_reduction_pct"),
        "transfer_wait": _reduction_factor(effects, "transfer_wait_reduction_pct"),
        "transfer_physical": _reduction_factor(effects, "transfer_time_reduction_pct"),
        "access": _reduction_factor(effects, "access_time_reduction_pct"),
        "egress": _reduction_factor(effects, "egress_time_reduction_pct"),
    }
    changed = 0
    for mode in ("walk", "bike"):
        changed += _multiply_selected(travel_times[f"ivt_pt_{mode}"], mask, ivt_factor)
        _multiply_selected(
            travel_times[f"initial_wait_pt_{mode}"],
            mask,
            component_factors["initial_wait"],
        )
        _multiply_selected(
            travel_times[f"transfer_wait_pt_{mode}"],
            mask,
            component_factors["transfer_wait"],
        )
        _multiply_selected(
            travel_times[f"transfer_physical_pt_{mode}"],
            mask,
            component_factors["transfer_physical"],
            floor=PT_TRANSFER_PHYSICAL_FLOOR_MIN,
        )
        _multiply_selected(
            travel_times[f"access_pt_{mode}"],
            mask,
            component_factors["access"],
            floor=PT_ACCESS_FLOOR_MIN,
        )
        _multiply_selected(
            travel_times[f"egress_pt_{mode}"],
            mask,
            component_factors["egress"],
            floor=PT_EGRESS_FLOOR_MIN,
        )
        _multiply_selected(lengths[f"pt_{mode}"], mask, distance_factor, time_matrix=False)
    return {
        "type": "railway_expansion",
        "name": intervention.get("name", "railway expansion"),
        "level": intervention.get("level"),
        "od_cells": changed // 2,
        "ivt_factor": ivt_factor,
        "distance_factor": distance_factor,
        **{f"{key}_factor": value for key, value in component_factors.items()},
    }


def _apply_mobility_hub(travel_times, intervention, lookups) -> dict:
    effects = intervention.get("effects", {})
    zones = intervention.get("zones", intervention.get("zone", []))
    zone_ids = {str(value) for value in (zones if isinstance(zones, list) else [zones])}
    zone_ids.discard("")
    feeder_speed_factor = _speed_factor(effects)
    access_factor = _reduction_factor(effects, "access_time_reduction_pct") * feeder_speed_factor
    egress_factor = _reduction_factor(effects, "egress_time_reduction_pct") * feeder_speed_factor
    initial_wait_factor = _reduction_factor(effects, "initial_wait_reduction_pct")
    transfer_wait_factor = _reduction_factor(effects, "transfer_wait_reduction_pct")
    transfer_factor = _reduction_factor(effects, "transfer_time_reduction_pct")

    mode_masks = _hub_mode_masks(travel_times, intervention, lookups)
    selected_stops: set[str] = set()
    affected_cells = 0
    for mode in ("walk", "bike"):
        selected_stops.update(mode_masks[mode]["matched_stop_ids"])
        origin_mask = mode_masks[mode]["origin"]
        destination_mask = mode_masks[mode]["destination"]
        path_mask = mode_masks[mode]["path"]
        _multiply_selected(
            travel_times[f"access_pt_{mode}"],
            origin_mask,
            access_factor,
            floor=PT_ACCESS_FLOOR_MIN,
        )
        _multiply_selected(
            travel_times[f"egress_pt_{mode}"],
            destination_mask,
            egress_factor,
            floor=PT_EGRESS_FLOOR_MIN,
        )
        _multiply_selected(
            travel_times[f"initial_wait_pt_{mode}"],
            origin_mask,
            initial_wait_factor,
        )
        _multiply_selected(
            travel_times[f"transfer_wait_pt_{mode}"],
            path_mask,
            transfer_wait_factor,
        )
        _multiply_selected(
            travel_times[f"transfer_physical_pt_{mode}"],
            path_mask,
            transfer_factor,
            floor=PT_TRANSFER_PHYSICAL_FLOOR_MIN,
        )
        affected_cells += int(path_mask.sum())
    return {
        "type": "mobility_hub",
        "name": intervention.get("name", "mobility hub"),
        "level": intervention.get("level"),
        "zones": sorted(zone_ids),
        "configured_stops": sorted(
            {
                str(stop_id)
                for mode in ("walk", "bike")
                for stop_id in mode_masks[mode]["stop_ids"]
            }
        ),
        "baseline_stops": sorted(selected_stops),
        "od_cells": affected_cells,
        "access_factor": access_factor,
        "egress_factor": egress_factor,
        "transfer_factor": transfer_factor,
    }


def intervention_masks(
    travel_times: dict[str, pd.DataFrame],
    lengths: dict[str, pd.DataFrame],
    scenario: dict,
    *,
    zones: pd.DataFrame | None = None,
    lookup_path: str | Path = LOOKUP_PACKAGE_FILE,
) -> list[dict]:
    """Return the OD cells represented by each selected intervention."""
    coverage = []
    for intervention in _as_list(scenario.get("bike_highways")):
        coverage.append(
            {
                "intervention_type": "bike_highway",
                "intervention_level": intervention.get("level"),
                "intervention_name": intervention.get("name", "bike highway"),
                "mask": _bike_highway_mask(travel_times, lengths, intervention, zones),
            }
        )
    for intervention in _as_list(scenario.get("railway_expansions")):
        coverage.append(
            {
                "intervention_type": "railway_expansion",
                "intervention_level": intervention.get("level"),
                "intervention_name": intervention.get("name", "railway expansion"),
                "mask": _railway_expansion_mask(travel_times, intervention, zones),
            }
        )

    hubs = _as_list(scenario.get("mobility_hubs"))
    lookup_required = any(not intervention.get("stop_ids") for intervention in hubs)
    lookups = load_package(lookup_path, "lookups") if lookup_required else None
    for intervention in hubs:
        mode_masks = _hub_mode_masks(travel_times, intervention, lookups)
        mask = mode_masks["walk"]["path"] | mode_masks["bike"]["path"]
        coverage.append(
            {
                "intervention_type": "mobility_hub",
                "intervention_level": intervention.get("level"),
                "intervention_name": intervention.get("name", "mobility hub"),
                "mask": mask,
            }
        )
    return coverage


def apply_interventions(
    baseline_travel_times: dict[str, pd.DataFrame],
    baseline_lengths: dict[str, pd.DataFrame],
    scenario: dict,
    lookup_path: str | Path = LOOKUP_PACKAGE_FILE,
    *,
    zones: pd.DataFrame | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], list[dict]]:
    """Apply selected interventions to copies of the prepared skims."""
    travel_times = {key: value.copy() for key, value in baseline_travel_times.items()}
    lengths = {key: value.copy() for key, value in baseline_lengths.items()}
    diagnostics: list[dict] = []

    for intervention in _as_list(scenario.get("bike_highways")):
        diagnostics.append(_apply_bike_highway(travel_times, lengths, intervention, zones))

    for intervention in _as_list(scenario.get("railway_expansions")):
        diagnostics.append(_apply_railway_expansion(travel_times, lengths, intervention, zones))

    hubs = _as_list(scenario.get("mobility_hubs"))
    if hubs:
        lookup_required = any(not intervention.get("stop_ids") for intervention in hubs)
        lookups = load_package(lookup_path, "lookups") if lookup_required else None
        for intervention in hubs:
            diagnostics.append(_apply_mobility_hub(travel_times, intervention, lookups))

    rebuild_pt_out_of_vehicle_times(travel_times)
    return travel_times, lengths, diagnostics
