"""Section coverage and optional existing passengers, shared by FSM and appraisal.

One physical section can affect PT, car, bicycle or walking journeys. An optional
existing-person cohort shares that section; it never enters OD generation, mode
choice or road loading. Compact coverage is prepared separately, without GTFS.
Run ``python code/additional/section_flows.py --help`` for the count/calibration workflow.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata
from types import MappingProxyType

import numpy as np
import pandas as pd

# Support the documented direct script command as well as package imports.
_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import parameters as p

_ROOT = _CODE_DIR.parent
_MODES = {"PT": ("pt_walk", "pt_bike"), "CAR": ("drive",), "BIKE": ("bike",), "WALK": ("walk",)}
_DEFAULT_COVERAGE = "data/processed/section_coverage.npz"


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, allow_nan=False).encode()).hexdigest()


def _mode(value):
    value = str(value).upper()
    value = {"DRIVE": "CAR", "BICYCLE": "BIKE", "CYCLING": "BIKE", "WALKING": "WALK"}.get(value, value)
    if value not in _MODES:
        raise ValueError("Section mode must be PT, CAR, BIKE or WALK.")
    return value


def section_config(params=None):
    """Resolve SECTION, or a direct section dictionary; generic default is off."""
    params = params or {}
    defaults = p.SECTION_DEFAULTS
    source = params.get("SECTION", params if "active" in params else p.SECTION)
    if not isinstance(source, dict):
        raise ValueError("SECTION must be a dictionary.")
    result = {**deepcopy(defaults), **deepcopy(source)}
    result["mode"] = _mode(result["mode"])
    for key in ("active", "both_directions", "crowding_enabled"):
        if not isinstance(result[key], (bool, np.bool_)):
            raise ValueError(f"SECTION {key} must be True or False.")
        result[key] = bool(result[key])
    if result["active"]:
        for key in ("origin", "destination"):
            result[key] = _selector(result[key], key)
        if result["crowding_enabled"] and result["mode"] != "PT":
            raise ValueError("This crowding formula is calibrated for PT only; other effects need explicit valuation.")
    return result


def external_flow_config(params=None):
    """Resolve one optional cohort; all counts are person-trips per weekday."""
    params = params or {}
    defaults = p.EXTERNAL_FLOW_DEFAULTS
    source = params.get("EXTERNAL_FLOW", params if "enabled" in params else p.EXTERNAL_FLOW)
    if not isinstance(source, dict):
        raise ValueError("EXTERNAL_FLOW must be a dictionary.")
    result = {**deepcopy(defaults), **deepcopy(source)}
    section = section_config(params)
    result["mode"] = _mode(result.get("mode") or section["mode"])
    if not isinstance(result["enabled"], (bool, np.bool_)):
        raise ValueError("EXTERNAL_FLOW enabled must be True or False.")
    result["enabled"] = bool(result["enabled"])
    count = float(result["additional_trips_daily"])
    if not np.isfinite(count) or count < 0:
        raise ValueError("additional_trips_daily must be finite and nonnegative.")
    result["additional_trips_daily"] = count
    if result["growth"] not in ("general", "fixed"):
        raise ValueError("External growth must be 'general' or 'fixed'.")
    if result["enabled"] and (not section["active"] or result["mode"] != section["mode"]):
        raise ValueError("An enabled external cohort needs an active SECTION of the same mode.")
    value = result.get("value_of_time_chf_per_hour")
    if value is None:
        key = "C_TT_" + result["mode"]
        value = params.get(key, p.NOMINAL_PARAMS.get(key))
    if value is not None:
        value = float(value)
        if not np.isfinite(value) or value < 0:
            raise ValueError("External value of time must be finite and nonnegative.")
    if result["enabled"] and value is None:
        raise ValueError("Configure the selected mode's value of time in CHF/person-hour.")
    result["value_of_time_chf_per_hour"] = value
    return result


def _selector(value, name):
    if not isinstance(value, dict) or len(value) != 1 or next(iter(value), None) not in ("municipality_name", "zone_ids"):
        raise ValueError(f"Section {name} must select municipality_name or zone_ids.")
    key, values = next(iter(value.items()))
    values = values if isinstance(values, (list, tuple, set)) else [values]
    if not values or any(item is None or not str(item).strip() for item in values):
        raise ValueError(f"Section {name} cannot be empty.")
    return {key: sorted({str(item) for item in values})}


def _selected_zones(context, labels, selector):
    key, allowed = next(iter(selector.items()))
    if key == "zone_ids":
        values = labels
    else:
        zones = context.zones.set_index(context.zones["grid_id"].astype(str))
        if zones.index.has_duplicates:
            raise ValueError("Section zone metadata has duplicate grid_id values.")
        values = pd.Index(zones.reindex(labels)[key].astype(str))
    missing = set(allowed) - set(values)
    if missing:
        raise ValueError(f"Section selector is absent from model: {sorted(missing)}.")
    return values.isin(allowed)


def _path(config):
    path = Path(config.get("coverage_file") or _DEFAULT_COVERAGE)
    return path if path.is_absolute() else _ROOT / path


def _name(value):
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(re.sub(r"[_\-\u2010-\u2015]", " ", value).split())


@lru_cache(maxsize=8)
def _read_coverage(path, stamp, metadata_stamp):
    with np.load(path, allow_pickle=False) as package:
        arrays = {key: package[key].copy() for key in package.files}
    catalog = json.loads(Path(path).with_suffix(".json").read_text(encoding="utf-8"))
    for array in arrays.values():
        array.flags.writeable = False
    digest = hashlib.sha256(Path(path).read_bytes() + json.dumps(catalog, sort_keys=True).encode()).hexdigest()
    return arrays, catalog, digest


def _coverage(config):
    path = _path(config)
    meta = path.with_suffix(".json")
    if not path.is_file() or not meta.is_file():
        raise FileNotFoundError(f"Prepare section coverage first: {path} and {meta}. Use code/additional/section_flows.py --prepare.")
    st, mt = path.stat(), meta.stat()
    arrays, catalog, digest = _read_coverage(str(path), (st.st_mtime_ns, st.st_size), (mt.st_mtime_ns, mt.st_size))
    override = config.get("section_override")
    candidates = [item for item in catalog["sections"] if _mode(item.get("mode", "PT")) == config["mode"]]
    if override is None:
        matches = [item for item in candidates if item["id"] == catalog["default_section"]]
    elif isinstance(override, str):
        matches = [item for item in candidates if _name(override) in {_name(alias) for alias in
            [item["id"], item.get("name", ""), *item.get("station_aliases", []),
             "-".join(item.get("stations", [])),
             *( ["-".join(reversed(item.get("stations", [])))] if item.get("both_directions", True) else [] ) ]}]
    elif isinstance(override, (list, tuple)) and len(override) == 2:
        pair = list(map(_name, override))
        matches = [item for item in candidates if
                   (sorted(map(_name, item.get("stations", []))) == sorted(pair)
                    if item.get("both_directions", True) else list(map(_name, item.get("stations", []))) == pair)]
    else:
        raise ValueError("section_override must be a registered name, station or station pair.")
    if len(matches) != 1:
        raise ValueError(f"Section {override!r} is not uniquely available for {config['mode']}. "
                         "Use --list-sections, or --prepare for a new section.")
    selected = matches[0]
    if bool(selected.get("both_directions", True)) != config["both_directions"]:
        raise ValueError("Coverage direction does not match SECTION; prepare the requested directions explicitly.")
    return arrays, selected, digest


def available_sections(coverage_file=None):
    """Show feasible registered names without running transport."""
    catalog = json.loads(_path({"coverage_file": coverage_file}).with_suffix(".json").read_text(encoding="utf-8"))
    return pd.DataFrame([{"section_override": item["id"], "mode": item.get("mode", "PT"),
        "station_pair": item.get("stations", []), "both_directions": item.get("both_directions", True),
        "description": item.get("description", ""),
        "historical_daily_check": item.get("nominal_daily_passengers_check")}
        for item in catalog["sections"]])


def physical_signature(config=None):
    """Hash physical section choices, excluding cohort, capacity and prices."""
    config = section_config(config)
    if not config["active"]:
        return _hash({"schema": "general-section-v1", "active": False})
    _, section, digest = _coverage(config)
    return _hash({"schema": "general-section-v1", **{key: config[key] for key in
        ("active", "mode", "origin", "destination", "both_directions")},
        "coverage": digest, "section": section["id"]})


def appraisal_signature(params=None):
    """Identify economic/section settings used by cached notebook appraisals."""
    from stages import get_stages, package_appraisal_settings
    resolved = {**p.NOMINAL_PARAMS, **(params or {})}
    return _hash({"physical": physical_signature(section_config(params)), "parameters": resolved,
        "section": section_config(params), "external": external_flow_config(params),
        "stages": get_stages(resolved), **package_appraisal_settings(), "horizon": p.N_YEARS,
        "uncertainties": p.STRUCTURAL_UNCERTAINTIES})


def coverage_masks(labels, config=None):
    config = section_config(config)
    if not config["active"]:
        return {}
    labels = pd.Index(map(str, labels))
    if labels.has_duplicates:
        raise ValueError("Section coverage requires unique OD zone labels.")
    arrays, section, _ = _coverage(config)
    stored = pd.Index(arrays["labels"].astype(str))
    indexer = stored.get_indexer(labels)
    if stored.has_duplicates or (indexer < 0).any():
        raise ValueError("Section coverage does not match model zone labels; prepare it for these inputs.")
    keys = section.get("masks", {"pt_walk": section.get("walk_key"), "pt_bike": section.get("bike_key")})
    result = {}
    for mode in _MODES[config["mode"]]:
        matrix = arrays[keys[mode]]
        if matrix.dtype != bool or matrix.shape != (len(stored), len(stored)):
            raise ValueError(f"Invalid boolean coverage matrix for {mode}.")
        result[mode] = matrix[np.ix_(indexer, indexer)]
    return result


def _values(frame, labels):
    frame = frame.rename(index=str, columns=str)
    if frame.index.has_duplicates or frame.columns.has_duplicates:
        raise ValueError("Section skims require unique OD zone labels.")
    return frame.reindex(index=labels, columns=labels).to_numpy(dtype=float)


def route_input_signature(context, mode):
    """Identify prepared base skims used by the fixed coverage, cached per context."""
    cache = getattr(context, "_section_input_signatures", {})
    if mode in cache:
        return cache[mode]
    keys = list(_MODES[mode])
    if mode == "PT":
        keys += [f"{prefix}_{native}" for native in _MODES[mode]
                 for prefix in ("ivt", "chosen_origin_stop", "chosen_destination_stop")]
    digest = hashlib.sha256()
    for key in sorted(keys):
        frame = context.travel_times.get(key)
        if frame is None and key == "drive":
            frame = context.travel_times.get("car")
        if frame is None:
            raise ValueError(f"Coverage provenance requires prepared skim {key!r}.")
        digest.update(key.encode())
        digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
        digest.update(pd.util.hash_pandas_object(frame.columns).to_numpy().tobytes())
    cache[mode] = digest.hexdigest()
    context._section_input_signatures = cache
    return cache[mode]


def _validate_route_inputs(context, config):
    """Reject fixed masks from another prepared skim set with identical labels."""
    _, section, _ = _coverage(config)
    expected = section.get("route_input_signature")
    if expected is not None and expected != route_input_signature(context, config["mode"]):
        raise ValueError("Section coverage predates the prepared skims; regenerate/revalidate section coverage before running the model.")


def _saving(stage):
    value = float(stage.get("section_time_saving_min", 0.0))
    if not np.isfinite(value) or value < 0:
        raise ValueError("section_time_saving_min must be finite and nonnegative.")
    return value


def _no_path_time(context):
    return float(context.modules["config"].NO_PATH_TIME_MIN)


def apply_section_time_saving(context, travel_times, stage_spec, config=None):
    """Subtract fixed minutes once, before mode choice; stage values are cumulative."""
    config = section_config(config if config is not None else stage_spec.get("section_config"))
    saving = _saving(stage_spec) if config["active"] else 0.0
    metadata = {"section_config": config, "section_time_saving_min": saving,
                "section_signature": physical_signature(config)}
    output = dict(travel_times)
    if not config["active"] or saving == 0:
        if config["active"]:
            _validate_route_inputs(context, config)
        return output, metadata
    _validate_route_inputs(context, config)
    labels = context.baseline_od.index.astype(str)
    no_path = _no_path_time(context)
    for mode, mask in coverage_masks(labels, config).items():
        key = f"ivt_{mode}" if mode.startswith("pt_") else mode
        values = _values(travel_times[key], labels).copy()
        valid = mask & np.isfinite(values) & (values >= 0) & (values < no_path)
        reduction = saving
        if mode == "drive":
            physical = context.travel_times.get("drive", context.travel_times.get("car"))
            available = _values(physical, labels)
            # Perceived connector/parking constants cannot become a physical
            # time saving when the requested reduction exceeds actual driving.
            reduction = np.minimum(saving, np.maximum(available[valid], 0.0))
        values[valid] = np.maximum(values[valid] - reduction, 0.0)
        output[key] = pd.DataFrame(values, index=labels, columns=labels)
        if mode.startswith("pt_"):
            output[mode] = output[key] + output[f"ovt_{mode}"]
    return output, metadata


def physical_car_times(context, mode_result):
    """Physical road minutes without perceived parking/connector penalties."""
    config = section_config(mode_result.scenario.get("section_config"))
    frame = context.travel_times.get("drive", context.travel_times.get("car"))
    if frame is None:
        raise ValueError("Section car appraisal requires a physical drive/car skim.")
    if not config["active"] or config["mode"] != "CAR" or _saving(mode_result.scenario) == 0:
        return frame
    # Use the same mask/minute operation on physical rather than perceived skim.
    times, _ = apply_section_time_saving(context, {"drive": frame}, mode_result.scenario, config)
    return times["drive"]


def _nominal(context):
    """One unassigned nominal baseline mode-choice call, reused for fixed weights."""
    from stages import get_stages
    baseline = get_stages()[0]
    nominal_key = _hash({key: value for key, value in baseline.items() if key not in ("section_config", "section_signature")})
    cached = getattr(context, "_section_nominal_cache", None)
    if cached is not None and cached.get("nominal_key", nominal_key) == nominal_key:
        return cached
    modules = context.modules
    times = modules["travel_times"].apply_perceived_time_policies(context.travel_times, context.zones)
    times, _ = modules["travel_times"].apply_ebike_share(times, float(baseline.get("ebike_share", 0)),
        speed_multiplier=float(baseline.get("EBIKE_SPEED_MULTIPLIER", 1.5)))
    native = modules["mode_choice_zurich"].load_mode_choice_parameters()
    native.update({key: value for key, value in baseline.items() if key in native})
    od, _ = modules["mode_choice_zurich"].mode_split_aggregated(times, context.lengths, context.baseline_od,
        walk_allowed_mask=modules["travel_times"].standalone_walk_mask(context.lengths, context.zones), parameters=native)
    cached = {"times": times, "quantities": od, "selections": {}, "nominal_key": nominal_key}
    context._section_nominal_cache = cached
    return cached


def _reference(context, mode_result, config):
    labels = context.baseline_od.index.astype(str)
    nominal = _nominal(context)
    signature = physical_signature(config)
    selection = nominal["selections"].get(signature)
    no_path = _no_path_time(context)
    if selection is None:
        origin = _selected_zones(context, labels, config["origin"])
        destination = _selected_zones(context, labels, config["destination"])
        pair = origin[:, None] & destination[None, :]
        if config["both_directions"]:
            pair |= destination[:, None] & origin[None, :]
        masks = coverage_masks(labels, config)
        selection, total = {}, 0.0
        for mode in _MODES[config["mode"]]:
            key = f"ivt_{mode}" if mode.startswith("pt_") else mode
            raw_car = context.travel_times.get("drive", context.travel_times.get("car"))
            if mode == "drive" and raw_car is None:
                raise ValueError("Section car appraisal requires a physical drive/car skim.")
            t = _values(raw_car if mode == "drive" else nominal["times"][key], labels)
            q = _values(nominal["quantities"][mode], labels)
            # The representative pair must traverse the same selected section.
            valid = pair & masks[mode] & np.isfinite(q) & (q > 0) & np.isfinite(t) & (t >= 0) & (t < no_path)
            row, col = np.nonzero(valid)
            selection[mode] = (row, col, q[row, col], t[row, col])
            total += float(q[row, col].sum())
        if total <= 0:
            raise ValueError("Selected reference OD pair has no valid nominal passengers crossing this section.")
        selection = {mode: (row, col, q / total, t) for mode, (row, col, q, t) in selection.items()}
        nominal["selections"][signature] = selection
    baseline, current, delay = 0.0, 0.0, 0.0
    uncongested = getattr(mode_result, "uncongested_travel_times", None) or mode_result.travel_times
    for mode, (row, col, weights, old) in selection.items():
        key = f"ivt_{mode}" if mode.startswith("pt_") else mode
        frame = physical_car_times(context, mode_result) if mode == "drive" else uncongested[key]
        values = _values(frame, labels)[row, col]
        if not (np.isfinite(values) & (values >= 0) & (values < no_path)).all():
            raise ValueError("Project skim invalid for fixed section reference cohort; do not redistribute its weights.")
        baseline += float(np.dot(weights, old))
        current += float(np.dot(weights, values))
        if mode == "drive":
            if getattr(mode_result, "uncongested_travel_times", None) is None:
                raise ValueError("External car time requires coupled assignment with uncongested skims.")
            actual = _values(mode_result.travel_times["drive"], labels)[row, col]
            free = _values(uncongested["drive"], labels)[row, col]
            if not np.isfinite(actual).all() or (actual >= no_path).any():
                raise ValueError("Assigned car skim invalid for the fixed section reference cohort.")
            delay += float(np.dot(weights, np.maximum(actual - free, 0)))
    return baseline, current, delay


def section_reference(context, mode_result, config=None):
    """Fixed nominal OD/submode weights and reference minutes, read-only arrays.

    Each mode maps to (origin indices, destination indices, weights, minutes).
    Weights sum to one over the selected modes and OD pairs.
    """
    config = section_config(config if config is not None else mode_result.scenario.get("section_config"))
    if not config["active"]:
        return MappingProxyType({})
    _reference(context, mode_result, config)
    selection = _nominal(context)["selections"][physical_signature(config)]
    for terms in selection.values():
        for values in terms:
            values.flags.writeable = False
    return MappingProxyType(selection)


def section_welfare_times(context, mode_result, labels):
    config = section_config(mode_result.scenario.get("section_config"))
    labels = pd.Index(map(str, labels))
    result = {f"{mode}_section": np.zeros((len(labels), len(labels))) for mode in _MODES["PT"]}
    if not config["active"] or config["mode"] != "PT":
        return result
    _, reference, _ = _reference(context, mode_result, config)
    for mode, mask in coverage_masks(labels, config).items():
        values = _values(mode_result.travel_times[f"ivt_{mode}"], labels)
        valid = mask & np.isfinite(values) & (values >= 0) & (values < _no_path_time(context))
        result[f"{mode}_section"][valid] = np.minimum(values[valid], reference) / 60
    return result


def section_metrics(context, mode_result):
    """Count covered person-trips over all retained ODs, independent of cordon."""
    config = section_config(mode_result.scenario.get("section_config"))
    result = {"section_active": config["active"], "section_mode": config["mode"],
        "section_signature": physical_signature(config), "section_modeled_trips_peak": 0.0,
        "section_modeled_person_hours_peak": 0.0, "section_reference_minutes0": 0.0,
        "section_reference_minutes": 0.0, "section_reference_delay_minutes": 0.0}
    if not config["active"]:
        return result
    _validate_route_inputs(context, config)
    labels = context.baseline_od.index.astype(str)
    old, current, delay = _reference(context, mode_result, config)
    result.update(section_reference_minutes0=old, section_reference_minutes=current, section_reference_delay_minutes=delay)
    pt_hours = section_welfare_times(context, mode_result, labels) if config["mode"] == "PT" else {}
    for mode, mask in coverage_masks(labels, config).items():
        q = _values(mode_result.od_by_mode[mode], labels)
        if not np.isfinite(q[mask]).all() or (q[mask] < 0).any():
            raise ValueError("Section quantities must be finite nonnegative person-trips.")
        if mode.startswith("pt_"):
            hours = pt_hours[f"{mode}_section"]
        else:
            times = physical_car_times(context, mode_result) if mode == "drive" else mode_result.travel_times[mode]
            values = _values(times, labels)
            if not np.isfinite(values[mask & (q > 0)]).all():
                raise ValueError("Invalid section travel times.")
            hours = np.minimum(values, current) / 60
        result["section_modeled_trips_peak"] += float(q[mask].sum())
        result["section_modeled_person_hours_peak"] += float(np.dot(q[mask], hours[mask]))
    return result


def calibrate_section(context, mode_result, observed_trips_daily=None, params=None):
    """A baseline diagnostic, never an automatic stage/year recalibration."""
    config = section_config(params if params is not None else mode_result.scenario.get("section_config"))
    if not config["active"]:
        raise ValueError("Activate a physical SECTION before counting it.")
    if "stage" not in mode_result.scenario or int(mode_result.scenario["stage"]) != 0:
        raise ValueError("Calibrate once against nominal Stage 0, not against a project stage.")
    assignment_metadata = getattr(mode_result, "assigned_metadata", None) or {}
    for key in ("road_freight_multiplier", "background_multiplier", "passenger_demand_multiplier"):
        if key in assignment_metadata and not np.isclose(float(assignment_metadata[key]), 1.0):
            raise ValueError("Calibration requires nominal passenger and road-background demand multipliers.")
    from stages import get_stages
    baseline = get_stages()[0]
    native = context.modules["mode_choice_zurich"].load_mode_choice_parameters()
    native.update({key: value for key, value in baseline.items() if key in native})
    required_preferences = {"ASC_PT_WALK", "ASC_PT_BIKE", "ASC_BIKE"}
    if not required_preferences.issubset(mode_result.scenario):
        raise ValueError("Calibration requires recorded nominal baseline mode preferences; rerun the baseline FSM.")
    # Native defaults need not all be duplicated in the scenario. Any explicit
    # override must match the known Stage-0 value, including time, cost, fare
    # and transfer coefficients as well as alternative-specific constants.
    for key, nominal in native.items():
        if key in mode_result.scenario and not np.isclose(
                float(mode_result.scenario[key]), float(nominal), rtol=1e-9, atol=1e-12):
            raise ValueError(f"Calibration requires nominal baseline mode-choice parameter {key}; rerun the baseline FSM.")
    for key in ("ebike_share", "EBIKE_SPEED_MULTIPLIER"):
        if not np.isclose(float(mode_result.scenario.get(key, baseline.get(key, 0))), float(baseline.get(key, 0))):
            raise ValueError("Calibration requires nominal baseline bicycle assumptions.")
    quantity = sum(_values(frame, context.baseline_od.index.astype(str)) for frame in mode_result.od_by_mode.values())
    if not np.allclose(quantity, context.baseline_od.to_numpy(float), rtol=1e-7, atol=1e-7):
        raise ValueError("Calibration requires nominal baseline OD demand, not a growth/dashboard demand scenario.")
    if _saving(mode_result.scenario) != 0:
        raise ValueError("Calibration requires zero baseline section saving.")
    original = mode_result.scenario
    mode_result.scenario = {**original, "section_config": config}
    try:
        metrics = section_metrics(context, mode_result)
    finally:
        mode_result.scenario = original
    values = {**p.NOMINAL_PARAMS, **(params or {})}
    share = float(values[config["mode"] + "_PEAK_SHARE"])
    if not 0 < share <= 1:
        raise ValueError("Peak share must lie in (0,1].")
    daily = metrics["section_modeled_trips_peak"] / share
    result = {**metrics, "modeled_trips_daily": daily, "peak_share": share,
        "units": "person-trips, total across the configured directions",
        "calibration_basis": "nominal baseline post-mode-choice; fixed prepared routes",
        "nominal_baseline_validation": "passed: stage, all explicit native mode-choice parameters, bicycle technology, every OD demand cell and supplied assignment demand multipliers",
        "observed_trips_daily": observed_trips_daily, "additional_trips_daily": None,
        "suggested_comfort_capacity_peak": None, "warning": None}
    metadata = assignment_metadata
    result["assignment_diagnostics"] = {key: value.item() if isinstance(value, np.generic) else value for key, value in metadata.items()
        if value is None or isinstance(value, (str, bool, int, float, np.integer, np.floating))}
    result["solver_basis"] = metadata.get("algorithm", "No assignment diagnostics supplied; unassigned/reference mode choice only")
    if observed_trips_daily is not None:
        observed = float(observed_trips_daily)
        if not np.isfinite(observed) or observed < 0:
            raise ValueError("Observed person-trips/day must be finite and nonnegative.")
        result.update(observed_trips_daily=observed, additional_trips_daily=max(observed - daily, 0),
            suggested_comfort_capacity_peak=observed * share)
        if observed < daily:
            result["warning"] = "Modeled count exceeds observation: no negative external cohort; investigate boundary/year/mode definitions."
    return result


def _route_inputs(context, mode, route_file=None):
    """Read existing routing inputs; never download/reprocess GTFS or OSM."""
    import pickle
    import networkx as nx
    from scipy.spatial import cKDTree
    labels = context.baseline_od.index.astype(str)
    if mode == "PT":
        path = Path(route_file or _ROOT / "data/processed/section_routes.pkl")
        if not path.is_file():
            raise FileNotFoundError("New PT sections require a prepared route-only graph (--route-file). "
                                    "Scalar PT skims cannot recover intermediate routes.")
        with path.open("rb") as handle:
            data = pickle.load(handle)
        graph, stops = data["graph"], data["stops"]
        nodes = list(graph)
        positions = {str(node): i for i, node in enumerate(nodes)}
        endpoints = {}
        for native in _MODES[mode]:
            endpoint = {}
            for side in ("origin", "destination"):
                frame = context.travel_times[f"chosen_{side}_stop_{native}"].rename(index=str, columns=str)
                strings = frame.reindex(index=labels, columns=labels).to_numpy(dtype=str)
                endpoint[side] = pd.Series(strings.ravel()).map(positions).fillna(-1).to_numpy(dtype=np.int32).reshape(strings.shape)
            endpoints[native] = endpoint
        indexed = stops.set_index(stops.stop_id.astype(str)).reindex(list(map(str, nodes)))
        from pyproj import Transformer
        transformer = Transformer.from_crs(4326, 2056, always_xy=True)
        x, y = transformer.transform(indexed.stop_lon.to_numpy(float), indexed.stop_lat.to_numpy(float))
        coordinates = np.column_stack((x, y))
        names = {str(node): str(name) for node, name in zip(nodes, indexed.stop_name)}
        # Service edges can skip intermediate stations, so geometric screenlines
        # inspect all transit edges, not only services stopping at the override.
        candidates = [(u, v) for u, v, attributes in graph.edges(data=True)
                      if attributes.get("edge_type") == "transit"]
        return graph, nodes, coordinates, endpoints, names, candidates, str(path)
    native = _MODES[mode][0]
    import importlib
    network = context.modules.get("network") or importlib.import_module("transport_core.network")
    config = context.modules["config"]
    path = Path(route_file or getattr(config, {"CAR": "ROAD_GRAPH_FILE", "BIKE": "BIKE_GRAPH_FILE", "WALK": "PT_ACCESS_WALK_GRAPH_FILE"}[mode]))
    if not path.is_file():
        raise FileNotFoundError(f"Prepared {mode} routing graph is missing: {path}")
    with path.open("rb") as handle:
        original = pickle.load(handle)
    originals = list(original)
    coordinates = np.array([[original.nodes[node]["x"], original.nodes[node]["y"]] for node in originals], dtype=float)
    if mode == "WALK":
        # The all-canton walk skim uses the complete walk graph, bounded to5km;
        # city-only pairs are separately overridden below by their city graph.
        edges = []
        for u, v, attributes in original.edges(data=True):
            length = float(attributes.get("length", np.nan))
            if np.isfinite(length) and length > 0:
                edges.append((u, v, length / 1000 / float(config.WALK_SPEED_KPH) * 60))
                edges.append((v, u, length / 1000 / float(config.WALK_SPEED_KPH) * 60))
        graph = nx.DiGraph()
        graph.add_nodes_from(originals)
        for u, v, weight in edges:
            if not graph.has_edge(u, v) or graph[u][v]["weight"] > weight:
                graph.add_edge(u, v, weight=weight)
    else:
        edges = network._direct_mode_edges(original, native, network._network_policy_city_polygon())
        graph = nx.DiGraph()
        graph.add_nodes_from(originals)
        for edge in edges.itertuples(index=False):
            graph.add_edge(originals[edge.source], originals[edge.target], weight=float(edge.time_min))
    zones = context.zones.set_index(context.zones.grid_id.astype(str)).reindex(labels)
    xy = np.column_stack((zones.centroid.x.to_numpy(float), zones.centroid.y.to_numpy(float)))
    _, snapped = cKDTree(coordinates).query(xy, k=1)
    origin = np.repeat(np.asarray(snapped)[:, None], len(labels), axis=1)
    destination = np.repeat(np.asarray(snapped)[None, :], len(labels), axis=0)
    endpoints = {native: {"origin": origin, "destination": destination}}
    return graph, originals, coordinates, endpoints, {}, list(graph.edges), str(path)


def available_stations(route_file=None):
    """Names accepted by new PT section preparation (registered aliases separate)."""
    import pickle
    path = Path(route_file or _ROOT / "data/processed/section_routes.pkl")
    with path.open("rb") as handle:
        stops = pickle.load(handle)["stops"]
    return pd.DataFrame({"station": sorted(stops.stop_name.astype(str).unique())})


def _trace_masks(graph, nodes, endpoints, selected_edges):
    """Boolean path incidence via predecessor trees, counting each OD once."""
    import networkx as nx
    from scipy.sparse.csgraph import dijkstra
    position = {node: i for i, node in enumerate(nodes)}
    size = len(nodes)
    adjacency = nx.to_scipy_sparse_array(graph, nodelist=nodes, weight="weight", format="csr", dtype=float)
    adjacency.indices = adjacency.indices.astype(np.int32)
    adjacency.indptr = adjacency.indptr.astype(np.int32)
    encoded_edges = np.array([position[u] * size + position[v] for u, v in selected_edges], dtype=np.int64)
    results = {mode: np.zeros(endpoint["origin"].shape, dtype=bool) for mode, endpoint in endpoints.items()}
    origins = sorted(set(int(value) for endpoint in endpoints.values() for value in np.unique(endpoint["origin"]) if value >= 0))
    grouped = {}
    for mode, endpoint in endpoints.items():
        flat = endpoint["origin"].ravel()
        order = np.argsort(flat, kind="stable")
        unique, counts = np.unique(flat, return_counts=True)
        grouped[mode] = dict(zip(unique, np.split(order, np.cumsum(counts)[:-1])))
    for offset in range(0, len(origins), 16):
        batch = origins[offset:offset + 16]
        _, predecessor = dijkstra(adjacency, directed=True, indices=batch, return_predecessors=True)
        node_ids = np.arange(size)[None, :]
        rows = np.arange(len(batch))[:, None]
        parent = np.where(predecessor >= 0, predecessor, node_ids)
        flags = np.isin(predecessor.astype(np.int64) * size + node_ids, encoded_edges)
        for _ in range(int(np.ceil(np.log2(max(size, 2)))) + 1):
            flags |= flags[rows, parent]
            advanced = parent[rows, parent]
            if np.array_equal(advanced, parent):
                break
            parent = advanced
        else:
            raise RuntimeError("Route predecessor propagation did not converge.")
        for index, origin in enumerate(batch):
            for mode, endpoint in endpoints.items():
                positions = grouped[mode].get(origin)
                if positions is None:
                    continue
                destinations = endpoint["destination"].ravel()[positions]
                valid = destinations >= 0
                results[mode].ravel()[positions[valid]] = flags[index, destinations[valid]]
    return results


def prepare_section_coverage(context, config, output, *, section_id="custom_section", route_file=None,
                             station_override=None, half_width_m=2000.0):
    """Prepare a visible midpoint screenline from existing route graphs.

    A station anchors the screenline to the representative route's nearest
    segment. A pair chooses a route between those stations. The perpendicular
    screenline is a local teaching approximation, shown in the exported JSON;
    inspect parallel/express edges and calibrate counts against that same line.
    Registered MehrSpur coverage should be selected directly, not regenerated.
    """
    import networkx as nx
    config = section_config(config)
    if not config["active"]:
        raise ValueError("Enable SECTION to prepare a counting section.")
    if not np.isfinite(half_width_m) or half_width_m <= 0:
        raise ValueError("Screenline half-width must be positive metres.")
    graph, nodes, xy, endpoints, names, candidates, source = _route_inputs(context, config["mode"], route_file)
    labels = context.baseline_od.index.astype(str)
    nominal = _nominal(context)
    origin = _selected_zones(context, labels, config["origin"])
    destination = _selected_zones(context, labels, config["destination"])
    pair = origin[:, None] & destination[None, :]
    best = None
    for mode in endpoints:
        q = _values(nominal["quantities"][mode], labels)
        key = f"ivt_{mode}" if mode.startswith("pt_") else mode
        base_times = _values(context.travel_times[key], labels)
        eligible = (pair & np.isfinite(q) & (q > 0) & np.isfinite(base_times)
                    & (base_times > 0) & (base_times < _no_path_time(context))
                    & (endpoints[mode]["origin"] >= 0) & (endpoints[mode]["destination"] >= 0))
        if eligible.any():
            index = np.unravel_index(np.argmax(np.where(eligible, q, -1)), q.shape)
            value = q[index]
            if best is None or value > best[0]:
                best = (value, nodes[endpoints[mode]["origin"][index]], nodes[endpoints[mode]["destination"][index]], mode, index)
    if best is None:
        raise ValueError("The reference OD selection has no positive nominal trips with route endpoints.")
    _, route_origin, route_destination, native, reference_od = best
    override = station_override or config.get("section_override")
    if isinstance(override, str):
        override = [override]
    if override and len(override) not in (1, 2):
        raise ValueError("Choose one station or a pair of stations.")
    positions = {node: i for i, node in enumerate(nodes)}
    station_nodes = []
    for name in override or []:
        matches = [node for node in nodes if _name(names.get(str(node), "")) == _name(name)]
        if not matches:
            raise ValueError(f"Station {name!r} is absent. Use --list-stations; for non-PT select municipalities/zones as endpoints.")
        station_nodes.append(matches)
    if len(station_nodes) == 2:
        paths = []
        for start in station_nodes[0]:
            lengths, paths_from = nx.single_source_dijkstra(graph, start, weight="weight")
            paths.extend((lengths[end], paths_from[end]) for end in station_nodes[1] if end in paths_from)
        if not paths:
            raise ValueError("No prepared route connects the selected station pair.")
        route = min(paths, key=lambda value: value[0])[1]
    else:
        route = nx.shortest_path(graph, route_origin, route_destination, weight="weight")
    if len(route) < 2:
        raise ValueError("Representative route has no link; choose distinct reference endpoints.")
    route_xy = xy[[positions[node] for node in route]]
    city_route_bundle = None
    if config["mode"] == "WALK":
        zones = context.zones.set_index(context.zones.grid_id.astype(str)).reindex(labels)
        city = zones.Level.astype(str).eq("Quartier").to_numpy() if "Level" in zones else np.zeros(len(labels), dtype=bool)
        if city[reference_od[0]] and city[reference_od[1]]:
            city_route_bundle = _route_inputs(context, "WALK", getattr(context.modules["config"], "WALK_GRAPH_FILE"))
            city_graph, city_nodes, city_xy, city_endpoints, _, _, _ = city_route_bundle
            city_origin = city_nodes[city_endpoints["walk"]["origin"][reference_od]]
            city_destination = city_nodes[city_endpoints["walk"]["destination"][reference_od]]
            city_route = nx.shortest_path(city_graph, city_origin, city_destination, weight="weight")
            if len(city_route) < 2:
                raise ValueError("Reference city walking route has no link; select different endpoints.")
            city_positions = {node: index for index, node in enumerate(city_nodes)}
            route_xy = city_xy[[city_positions[node] for node in city_route]]
    lengths = np.linalg.norm(np.diff(route_xy, axis=0), axis=1)
    if not np.isfinite(lengths).all() or lengths.sum() <= 0:
        raise ValueError("Prepared route coordinates cannot define a screenline.")
    distance = lengths.sum() / 2
    if len(station_nodes) == 1:
        station_xy = xy[[positions[node] for node in station_nodes[0]]].mean(axis=0)
        candidates_with_length = np.flatnonzero(lengths > 1e-9)
        starts = route_xy[:-1][candidates_with_length]
        vectors = route_xy[1:][candidates_with_length] - starts
        fraction = np.clip(np.sum((station_xy - starts) * vectors, axis=1)
                           / np.square(lengths[candidates_with_length]), 0, 1)
        projected = starts + fraction[:, None] * vectors
        selected_projection = int(np.argmin(np.linalg.norm(projected - station_xy, axis=1)))
        segment = int(candidates_with_length[selected_projection])
        center = projected[selected_projection]
    else:
        candidates_with_length = np.flatnonzero(lengths > 1e-9)
        segment = int(candidates_with_length[min(int(np.searchsorted(np.cumsum(lengths[candidates_with_length]), distance)), len(candidates_with_length) - 1)])
        previous = float(lengths[:segment].sum())
        center = route_xy[segment] + (distance - previous) / max(lengths[segment], 1e-9) * (route_xy[segment + 1] - route_xy[segment])
    tangent = route_xy[segment + 1] - route_xy[segment]
    tangent /= np.linalg.norm(tangent)
    perpendicular = np.array([-tangent[1], tangent[0]])
    selected = []
    for u, v in candidates:
        a, b = xy[positions[u]] - center, xy[positions[v]] - center
        start, end = np.dot(a, tangent), np.dot(b, tangent)
        if start * end > 0 or np.isclose(start, end):
            continue
        crossing = a + (-start / (end - start)) * (b - a)
        if abs(np.dot(crossing, perpendicular)) <= half_width_m and (config["both_directions"] or end > start):
            selected.append((u, v))
    if not selected:
        raise ValueError("No route edges cross the proposed screenline; inspect the route and width.")
    masks = _trace_masks(graph, nodes, endpoints, selected)
    # Bound routes by the actual mode skim availability. Walk's city-only skim
    # is produced on a different graph, so refuse to label it exact incidence.
    for mode, mask in masks.items():
        key = f"ivt_{mode}" if mode.startswith("pt_") else mode
        values = _values(context.travel_times[key], labels)
        mask &= np.isfinite(values) & (values >= 0) & (values < _no_path_time(context))
    if config["mode"] == "WALK":
        zones = context.zones.set_index(context.zones.grid_id.astype(str)).reindex(labels)
        city = zones.Level.astype(str).eq("Quartier").to_numpy() if "Level" in zones else np.zeros(len(labels), dtype=bool)
        if city.any():
            # Recompute city-only routing on the same graph used by the skim.
            original_graph = getattr(context.modules["config"], "WALK_GRAPH_FILE")
            city_graph, city_nodes, city_xy, city_endpoints, _, city_candidates, _ = (city_route_bundle or _route_inputs(context, "WALK", original_graph))
            city_positions = {node: i for i, node in enumerate(city_nodes)}
            city_edges = []
            for u, v in city_candidates:
                a, b = city_xy[city_positions[u]] - center, city_xy[city_positions[v]] - center
                start, end = np.dot(a, tangent), np.dot(b, tangent)
                if start * end > 0 or np.isclose(start, end):
                    continue
                crossing = a + (-start / (end - start)) * (b - a)
                if abs(np.dot(crossing, perpendicular)) <= half_width_m and (config["both_directions"] or end > start):
                    city_edges.append((u, v))
            city_masks = _trace_masks(city_graph, city_nodes, city_endpoints, city_edges)["walk"]
            city_pairs = city[:, None] & city[None, :]
            masks["walk"][city_pairs] = city_masks[city_pairs]
            walk_values = _values(context.travel_times["walk"], labels)
            masks["walk"] &= np.isfinite(walk_values) & (walk_values >= 0) & (walk_values < _no_path_time(context))
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    keys = {mode: f"{section_id}_{mode}" for mode in masks}
    np.savez_compressed(target, labels=np.asarray(labels, dtype=str), **{keys[mode]: mask for mode, mask in masks.items()})
    entry = {"id": section_id, "name": section_id, "mode": config["mode"], "masks": keys,
        "route_input_signature": route_input_signature(context, config["mode"]),
        "both_directions": config["both_directions"], "stations": list(override or []),
        "description": "Prepared midpoint screenline; inspect selected edges and calibrate the same observed boundary.",
        "screenline_center_lv95": center.tolist(), "screenline_half_width_m": half_width_m,
        "reference_od": [str(labels[reference_od[0]]), str(labels[reference_od[1]])],
        "selected_edges": [{"origin": str(u), "destination": str(v), "origin_name": names.get(str(u)),
                            "destination_name": names.get(str(v))} for u, v in selected]}
    metadata = {"schema_version": 2, "default_section": section_id, "sections": [entry],
        "provenance": {"route_file": source, "route_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                       "method": "fixed prepared shortest paths, each person counted once if any selected edge occurs"},
        "limitations": ["Fixed baseline routes; no route reassignment after an intervention.",
                        "Straight-line service edges approximate geometry; inspect skipped stops and parallel branches.",
                        "Observed count must match mode, directions, section, period and year."]}
    target.with_suffix(".json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {**config, "coverage_file": str(target.resolve()), "section_override": section_id}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list-sections", action="store_true", help="List registered station/section overrides.")
    parser.add_argument("--list-stations", action="store_true", help="List station names in the optional prepared PT route graph.")
    parser.add_argument("--prepare", action="store_true", help="Prepare a new section at a representative route midpoint.")
    parser.add_argument("--count", action="store_true", help="Run nominal native FSM/MSA and estimate a calibration residual.")
    parser.add_argument("--mode", choices=list(_MODES))
    parser.add_argument("--origin", help="Origin municipality name (or use --origin-zones).")
    parser.add_argument("--destination", help="Destination municipality name.")
    parser.add_argument("--origin-zones", nargs="+")
    parser.add_argument("--destination-zones", nargs="+")
    parser.add_argument("--one-direction", action="store_true")
    parser.add_argument("--section", help="Registered section name for counting; --list-sections shows choices.")
    parser.add_argument("--station", nargs="+", help="One station or two stations for NEW section preparation.")
    parser.add_argument("--coverage-file", help="Existing imported boolean OD coverage NPZ and accompanying JSON.")
    parser.add_argument("--route-file", help="Existing prepared mode route graph; never a GTFS feed.")
    parser.add_argument("--half-width-m", type=float, default=2000, help="Half width of new local screenline (metres).")
    parser.add_argument("--observed-daily", type=float, help="Observed existing person-trips/day across the same section/directions.")
    parser.add_argument("--iterations", type=int, help="MSA maximum iterations; defaults to ASSIGNMENT_SETTINGS.")
    parser.add_argument("--fixed-iterations", action="store_true", help="Use the same minimum and maximum MSA iterations for reproducible calibration.")
    parser.add_argument("--output", type=Path, default=Path("data/processed"), help="Directory for count report/new compact coverage.")
    args = parser.parse_args(argv)
    if args.list_sections:
        print(available_sections(args.coverage_file).to_string(index=False))
    if args.list_stations:
        print(available_stations(args.route_file).to_string(index=False))
    if not (args.prepare or args.count):
        if not (args.list_sections or args.list_stations):
            parser.print_help()
        return
    config = section_config()
    config["active"] = True
    if args.mode:
        config["mode"] = args.mode
        config["crowding_enabled"] = bool(args.mode == "PT" and config["crowding_enabled"])
    for key in ("origin", "destination"):
        zones, municipality = getattr(args, key + "_zones"), getattr(args, key)
        if zones and municipality:
            parser.error(f"Choose --{key} or --{key}-zones, not both.")
        if zones or municipality:
            config[key] = {"zone_ids": zones} if zones else {"municipality_name": municipality}
    if args.coverage_file:
        config["coverage_file"] = args.coverage_file
    if args.section:
        config["section_override"] = args.section
    if args.one_direction:
        config["both_directions"] = False
    from transport_model_interface import load_transport_context, run_simulation, get_zone_ids_for_municipalities, ASSIGNMENT_SETTINGS
    from stages import get_stages
    context = load_transport_context(_ROOT)
    if args.prepare:
        config = prepare_section_coverage(context, config, args.output / "section_coverage.npz", route_file=args.route_file,
            station_override=args.station, half_width_m=args.half_width_m)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "section_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Prepared coverage and configuration in {args.output}. Inspect its JSON edge list before using an observation.")
    if args.count:
        specs = get_stages({"SECTION": config, "EXTERNAL_FLOW": {"enabled": False}})
        specs[0]["section_config"] = config
        specs[0]["section_time_saving_min"] = 0
        settings = dict(ASSIGNMENT_SETTINGS)
        try:
            ASSIGNMENT_SETTINGS.update(method="MSA", modal_feedback=True)
            if args.iterations is not None:
                if args.iterations <= 0:
                    parser.error("--iterations must be a positive integer.")
                ASSIGNMENT_SETTINGS["max_iterations"] = args.iterations
            if args.fixed_iterations:
                ASSIGNMENT_SETTINGS["min_iterations"] = ASSIGNMENT_SETTINGS["max_iterations"]
            result, _ = run_simulation(context, stage=0, stage_specs=specs,
                corridor_zone_ids=get_zone_ids_for_municipalities(context.zones, p.CORRIDOR_MUNICIPALITIES),
                corridor_municipalities=list(p.CORRIDOR_MUNICIPALITIES))
        finally:
            ASSIGNMENT_SETTINGS.clear()
            ASSIGNMENT_SETTINGS.update(settings)
        report = calibrate_section(context, result, args.observed_daily, {"SECTION": config})
        report["solver_settings"] = {**settings, "method": "MSA", "modal_feedback": True,
            "max_iterations": args.iterations or settings.get("max_iterations", 8),
            "min_iterations": (args.iterations or settings.get("max_iterations", 8)) if args.fixed_iterations else settings.get("min_iterations", 2)}
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "calibration.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
