"""
transport_notebook.py
=====================
Notebook-facing bridge to the Canton Zürich transport model (FSM).

This module serves as a project-neutral, modular interface between Jupyter
notebooks and the underlying Canton Zürich spatial transport model
(IP_course_FSM-main with 1,223 zones, multimodal skims, and logit mode choice).

WHAT THIS MODULE PROVIDES
-------------------------
1. Preflight checks for prepared transport data inputs (load_transport_context).
2. Spatial lookup utilities by municipality / region (get_zone_ids_for_municipalities).
3. Stage-aware multimodal mode choice simulation (run_transport_mode_choice).
4. Automated extraction of corridor metrics for CBA evaluation (extract_corridor_metrics).
5. Interactive dashboard widgets for exploratory analysis (mode_choice_dashboard).
6. Optional WebGL / Lonboard network map visualization (full_network_explorer).

Students do NOT need to modify this file; all project-specific interventions
and economic assumptions are configured in stages.py and parameters.py.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields
import importlib
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Default Constants
# ---------------------------------------------------------------------------

DEFAULT_CORRIDOR_BUFFER_M = 500.0
DEFAULT_MAX_GATES = 100
DEFAULT_GATE_SEPARATION_M = 500.0
DEFAULT_PASSTHROUGH_FRACTION = 0.05

# ---------------------------------------------------------------------------
# Global Settings & Toggles
# ---------------------------------------------------------------------------

ASSIGNMENT_SETTINGS = {
    # MSA is the teaching-model default: link flows update BPR travel times
    # explicitly.  LUT remains available as an opt-in fast approximation.
    "method": "LUT",  # 'MSA' or 'LUT'
    "regenerate_lut": False,  # If True, re-runs multi-scale MSA assignment to regenerate & overwrite JSON files
    "force_regenerate_corridor_network": True,  # If True, forces redownload and rebuild of detailed OSM network
    "max_iterations": 8,
    "min_iterations": 2,
    "relative_gap_threshold": 0.10,
    "od_threshold": 1.0,
    "drive_occupancy": 1.14,
    "modal_feedback": True,
}

ROUTE_ASSIGNMENT_DEFAULTS = ASSIGNMENT_SETTINGS  # keep alias for backwards compatibility


# ---------------------------------------------------------------------------
# NumPy unpickling compatibility shim (handles NumPy 1.x vs 2.x pickles)
# ---------------------------------------------------------------------------

def _alias_numpy_core_for_pickle_compat() -> None:
    """Let NumPy 1.26 unpickle arrays saved under NumPy 2.x."""
    try:
        import numpy._core.numeric  # noqa: F401
        return
    except ImportError:
        pass

    try:
        import numpy.core.numeric
        sys.modules["numpy._core.numeric"] = numpy.core.numeric
    except Exception:
        pass


_alias_numpy_core_for_pickle_compat()


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TransportContext:
    """Prepared inputs and modules shared across notebook transport runs."""
    project_root: Path
    model_dir: Path
    zones: Any
    baseline_od: pd.DataFrame
    road_background_od: pd.DataFrame
    travel_times: dict[str, pd.DataFrame]
    lengths: dict[str, pd.DataFrame]
    assignment_network: dict[str, Any]
    modules: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"'{key}' not found in TransportContext. Available keys: {list(self.keys())}")

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def __iter__(self):
        return iter(self.keys())

    def keys(self) -> list[str]:
        return [f.name for f in fields(self)]

    def values(self) -> list[Any]:
        return [getattr(self, f.name) for f in fields(self)]

    def items(self) -> list[tuple[str, Any]]:
        return [(f.name, getattr(self, f.name)) for f in fields(self)]

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass
class ModeChoiceResult:
    """Memory-conscious output of a multimodal mode-choice run."""
    drive_od: pd.DataFrame
    od_by_mode: dict[str, pd.DataFrame]
    summary: pd.DataFrame
    scenario: dict[str, Any]
    travel_times: dict[str, pd.DataFrame]
    lengths: dict[str, pd.DataFrame]
    cache_key: tuple[Any, ...]
    assigned_edges: Any = None
    assigned_metadata: dict = None
    # Preserve the stage-adjusted, uncongested skim when a coupled assignment
    # returns an updated congested skim. This makes later-year reruns stable.
    uncongested_travel_times: dict[str, pd.DataFrame] | None = None

    @property
    def stage(self) -> int:
        if isinstance(self.scenario, dict):
            return int(self.scenario.get("stage", 0))
        return 0

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"'{key}' not found in ModeChoiceResult. Available keys: {list(self.keys())}")

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def __iter__(self):
        return iter(self.keys())

    def keys(self) -> list[str]:
        return [f.name for f in fields(self)] + ["stage"]

    def values(self) -> list[Any]:
        return [getattr(self, k) for k in self.keys()]

    def items(self) -> list[tuple[str, Any]]:
        return [(k, getattr(self, k)) for k in self.keys()]

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.keys()}


@dataclass
class CorridorContext:
    """Topology, cordon gates, and OD mappings for one model corridor.

    Keeping these related objects together prevents notebook cells from
    rebuilding slightly different corridor definitions for maps and audit
    tables.  The class is project-neutral: MehrSpur is selected through
    ``parameters.CORRIDOR_MUNICIPALITIES`` rather than hard-coded here.
    """

    polygon: Any
    polygon_gdf: Any
    zones: Any
    zone_ids: list[str]
    edges: Any
    nodes: Any
    gates: Any
    zone_node_map: dict[str, int]
    external_zone_to_entry_gate: dict[str, str]
    external_zone_to_exit_gate: dict[str, str]
    metadata: dict[str, Any]


@dataclass
class AssignmentResult:
    """One corridor assignment plus the data needed to audit it."""

    links: Any
    history: pd.DataFrame
    diagnostics: dict[str, Any]
    demand_matrix: pd.DataFrame
    demand_breakdown: pd.DataFrame
    gate_totals: pd.DataFrame
    # Populated only by the coupled congestion--mode-choice calculation.
    mode_result: ModeChoiceResult | None = None


# =============================================================================
# 1. PREFLIGHT AND CONTEXT LOADING
# =============================================================================

def transport_input_status(project_root: str | Path) -> pd.DataFrame:
    """Check availability of prepared FSM transport model input files."""
    root = Path(project_root).resolve()
    model_dir = root / "IP_course_FSM-main"
    prepared = model_dir / "input_data" / "prepared"
    
    rows = [
        ("Zones", prepared / "zones.parquet", True, "1,223 spatial zones & attributes"),
        ("Passenger & Background Demand", prepared / "demand.pkl.gz", True, "OD matrices"),
        ("Multimodal Skims", prepared / "skims.pkl.gz", True, "Travel times & distances"),
        ("Road Network", prepared / "assignment_network.pkl", True, "Road links and nodes"),
        ("Mode-choice Parameters", model_dir / "config" / "mode_choice.json", True, "Logit coefficients"),
    ]
    
    status = pd.DataFrame(rows, columns=["item", "path", "required", "purpose"])
    status["present"] = status["path"].map(lambda p: Path(p).exists())
    status["relative_path"] = status["path"].map(
        lambda p: str(Path(p).relative_to(root)) if Path(p).is_relative_to(root) else str(p)
    )
    return status[["item", "required", "present", "relative_path", "purpose"]]


def readiness_flags(status: pd.DataFrame) -> dict[str, bool]:
    """Summarise data readiness."""
    by_item = status.set_index("item")["present"].to_dict()
    ready = all(by_item.values())
    return {
        "network": bool(by_item.get("Zones", False) and by_item.get("Road Network", False)),
        "mode_choice": ready,
    }


def readiness_message(status: pd.DataFrame) -> str:
    """Return a styled HTML readiness callout for notebook display."""
    missing = status.loc[status["required"] & ~status["present"], "relative_path"].tolist()
    if not missing:
        return (
            "<div style='padding:10px;border-left:5px solid #2e7d32;background:#edf7ed'>"
            "<b>Transport model ready.</b> 1,223 Canton Zürich zones and skims loaded.</div>"
        )
    items = "".join(f"<li><code>{value}</code></li>" for value in missing)
    return (
        "<div style='padding:10px;border-left:5px solid #c47f00;background:#fff8e1'>"
        f"<b>Missing prepared inputs:</b><ul>{items}</ul></div>"
    )


def _import_fsm_modules(model_dir: Path) -> dict[str, Any]:
    """Dynamically import the FSM standalone packages."""
    model_dir = model_dir.resolve()
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))

    module_names = (
        "config",
        "input_packages",
        "zoning",
        "travel_times",
        "scenario_generation",
        "interventions",
        "mode_choice_zurich"
    )
    modules: dict[str, Any] = {}
    for name in module_names:
        modules[name] = importlib.import_module(name)
    return modules

def _load_assignment_network(prepared_dir: Path) -> dict[str, Any]:
    """Load prepared road network without importing pandana/network.py."""
    network_file = prepared_dir / "assignment_network.pkl"
    if not network_file.exists():
        return {}
    import pickle
    with open(network_file, "rb") as f:
        return pickle.load(f)


def load_transport_context(
    project_root: str | Path,
    *,
    require_mode_choice: bool = True,
) -> TransportContext:
    """Load the complete Canton Zürich transport context and prepared skims."""
    import pickle
    
    root = Path(project_root).resolve()
    status = transport_input_status(root)
    flags = readiness_flags(status)
    
    if require_mode_choice and not flags["mode_choice"]:
        missing = status.loc[status["required"] & ~status["present"], "relative_path"]
        raise FileNotFoundError("Missing prepared transport inputs: " + ", ".join(missing.astype(str)))

    model_dir = root / "IP_course_FSM-main"
    prepared_dir = model_dir / "input_data" / "prepared"
    modules = _import_fsm_modules(model_dir)
    
    cache_dir = root / "cache"
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / "tmi_context.pkl"
    
    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)
            return TransportContext(
                project_root=root,
                model_dir=model_dir,
                zones=cached_data["zones"],
                baseline_od=cached_data["baseline_od"],
                road_background_od=cached_data["road_background"],
                travel_times=cached_data["travel_times"],
                lengths=cached_data["lengths"],
                assignment_network=cached_data["assignment_network"],
                modules=modules,
            )
        except Exception as e:
            print(f"Warning: Failed to load cached context ({e}). Regenerating...")

    zones = modules["zoning"].load_zones()
    assignment_network = _load_assignment_network(prepared_dir)
    zones, baseline_od, road_background = modules["zoning"].load_model_inputs()
    travel_times, lengths = modules["travel_times"].load_travel_times(zones)
    
    cached_data = {
        "zones": zones,
        "baseline_od": baseline_od,
        "road_background": road_background,
        "travel_times": travel_times,
        "lengths": lengths,
        "assignment_network": assignment_network,
    }
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(cached_data, f)
    except Exception as e:
        print(f"Warning: Failed to write context cache ({e}).")

    return TransportContext(
        project_root=root,
        model_dir=model_dir,
        zones=zones,
        baseline_od=baseline_od,
        road_background_od=road_background,
        travel_times=travel_times,
        lengths=lengths,
        assignment_network=assignment_network,
        modules=modules,
    )


# =============================================================================
# 2. SPATIAL HELPERS & CORRIDOR SELECTION
# =============================================================================

def get_zone_ids_for_municipalities(
    context: TransportContext | Any,
    municipality_names: list[str] | set[str] | None = None,
) -> list[str]:
    """Return a list of grid_id strings matching the given municipality names or explicit corridor zones."""
    try:
        import parameters as p
        if getattr(p, "CORRIDOR_DEFINITION_MODE", "") == "zones":
            zone_ids = getattr(p, "CORRIDOR_ZONE_IDS", [])
            if zone_ids:
                return [str(z) for z in zone_ids]
    except Exception:
        pass
    zones = context.zones if hasattr(context, "zones") else context
    muni_col = zones.get("municipality_name", pd.Series("", index=zones.index)).astype(str)
    matched = zones.loc[muni_col.isin(municipality_names or []), "grid_id"].astype(str).tolist()
    return matched


def _get_core_zones(zones: pd.DataFrame, corridor_municipalities: list[str]) -> pd.DataFrame:
    """Get core network zones based on the student's chosen definition mode in parameters.py."""
    try:
        import parameters as p
        mode = getattr(p, "CORRIDOR_DEFINITION_MODE", "municipalities")
        zone_ids = getattr(p, "CORRIDOR_ZONE_IDS", [])
    except Exception:
        mode = "municipalities"
        zone_ids = []
        
    if mode == "zones":
        if not zone_ids:
            raise ValueError("CORRIDOR_DEFINITION_MODE is 'zones' but CORRIDOR_ZONE_IDS is empty.")
        return zones.loc[zones["grid_id"].astype(str).isin([str(x) for x in zone_ids])].copy()
    else:
        return zones.loc[zones["municipality_name"].isin(corridor_municipalities)].copy()


# =============================================================================
# 3. NATIVE FSM SCENARIO GENERATION & MODE CHOICE
# =============================================================================


def _corridor_mode_summary(
    od_by_mode: dict[str, pd.DataFrame],
    corridor_zone_ids: list[str] | set[str] | None = None,
) -> pd.DataFrame:
    """Summarise modal split across all OD pairs touching the corridor."""
    reference = next(iter(od_by_mode.values()))
    labels = reference.index.astype(str)
    
    if corridor_zone_ids is not None and len(corridor_zone_ids) > 0:
        internal = np.isin(labels, np.asarray(list(corridor_zone_ids), dtype=str))
        involved = internal[:, None] | internal[None, :]
    else:
        involved = np.ones((len(labels), len(labels)), dtype=bool)

    mode_matrices = {
        "Car (Driver)": od_by_mode["drive"],
        "Public Transport": od_by_mode["pt_walk"] + od_by_mode["pt_bike"],
        "Bicycle": od_by_mode["bike"],
        "Walking": od_by_mode["walk"],
    }
    
    rows = []
    for mode, frame in mode_matrices.items():
        val = frame.to_numpy(dtype=float)
        rows.append({
            "mode": mode,
            "trips": float(val[involved].sum()),
        })
        
    summary = pd.DataFrame(rows)
    total_trips = float(summary["trips"].sum())
    summary["share"] = summary["trips"] / max(total_trips, 1e-9)
    return summary


def run_transport_mode_choice(
    context: TransportContext,
    *,
    stage: int,
    stage_specs: Mapping[int, Mapping[str, Any]],
    corridor_zone_ids: list[str] | set[str] | None = None,
    corridor_municipalities: list[str] | None = None,
    trip_rate_multiplier: float = 1.0,
    demand_multiplier: float | None = None,
    car_affinity: float | None = None,
    bike_affinity: float | None = None,
    pt_affinity: float | None = None,
    ebike_share: float | None = None,
    run_assignment: bool = False,
    assignment_iterations: int = 8,
) -> ModeChoiceResult:
    """
    Run 5-alternative mode choice across Canton Zürich using fixed skims and stage specs.
    
    Parameters
    ----------
    context : TransportContext
        Loaded transport model context.
    stage : int
        Active stage ID (0, 1, 2, ...).
    stage_specs : Mapping
        Stage specifications dict from stages.py.
    corridor_zone_ids : list[str] | None
        Optional subset of zone IDs for localized corridor summary.
    """
    stage = int(stage)
    
    if not isinstance(stage_specs, dict):
        raise TypeError(
            f"Expected 'stage_specs' to be a dictionary of stage definitions, "
            f"but got {type(stage_specs).__name__}. "
            f"\nDid you pass the 'stages' module instead of calling stages.get_stages(params)? "
            f"\nExample fix: stage_specs = stages.get_stages(p.NOMINAL_PARAMS)"
        )
        
    if stage not in stage_specs:
        raise KeyError(f"Unknown stage: {stage}. Available stages: {list(stage_specs.keys())}")
        
    mult = demand_multiplier if demand_multiplier is not None else trip_rate_multiplier
    trip_multiplier = max(float(mult), 0.0)
    modules = context.modules
    
    # Pass down the native lists directly from stages.py
    stage_spec = dict(stage_specs[int(stage)])
    scenario = {
        "name": f"stage_{stage}",
        "population_multiplier": trip_multiplier,
        "jobs_multiplier": trip_multiplier,
        "trip_rate_multiplier": 1.0,
        "stage": stage,
    }
    
    # 1. Base values from stage spec
    for key, value in stage_spec.items():
        if key not in ("name",):
            scenario[key] = value

    # 2. Overrides from explicit function arguments (e.g. from the dashboard)
    if car_affinity is not None:
        scenario["car_affinity"] = float(car_affinity)
    if bike_affinity is not None:
        scenario["bike_affinity"] = float(bike_affinity)
    if pt_affinity is not None:
        scenario["pt_affinity"] = float(pt_affinity)
    if ebike_share is not None:
        scenario["ebike_share"] = float(ebike_share)

    if "scenario_generation" in modules:
        _, total_od, _ = modules["scenario_generation"].generate_scenario_demand(
            context.baseline_od,
            context.zones,
            scenario
        )
    else:
        total_od = context.baseline_od * trip_multiplier

    travel_times = modules["travel_times"].apply_perceived_time_policies(
        context.travel_times,
        context.zones,
    )
    lengths = {key: value.copy() for key, value in context.lengths.items()}
    
    # Apply interventions directly using the native FSM schema
    intervention_keys = ["bike_highways", "railway_expansions", "mobility_hubs", "road_capacity"]
    has_interventions = any(scenario.get(k) for k in intervention_keys)
    if has_interventions:
        travel_times, lengths, _ = modules["interventions"].apply_interventions(
            travel_times,
            lengths,
            scenario,
            zones=context.zones,
        )
        
    travel_times, _ = modules["travel_times"].apply_ebike_share(
        travel_times,
        float(scenario.get("ebike_share", 0.0)),
        speed_multiplier=float(stage_spec.get("EBIKE_SPEED_MULTIPLIER", 1.5)),
    )

    parameters = modules["mode_choice_zurich"].load_mode_choice_parameters()
    # Override native parameters if provided in stage_specs
    for k in parameters:
        if k in stage_spec:
            parameters[k] = stage_spec[k]
    od_by_mode, _ = modules["mode_choice_zurich"].mode_split_aggregated(
        travel_times,
        lengths,
        total_od,
        walk_allowed_mask=modules["travel_times"].standalone_walk_mask(
            lengths,
            context.zones,
        ),
        parameters=parameters,
        car_affinity=float(scenario.get("car_affinity", 1.0)),
        walk_affinity=float(scenario.get("walk_affinity", 1.0)),
        bike_affinity=float(scenario.get("bike_affinity", 1.0)),
        pt_affinity=float(scenario.get("pt_affinity", 1.0)),
    )

    summary = _corridor_mode_summary(od_by_mode, corridor_zone_ids)
    
    cache_key = (
        stage,
        round(trip_multiplier, 6),
        round(float(scenario.get("car_affinity", 1.0)), 6),
        round(float(scenario.get("bike_affinity", 1.0)), 6),
        round(float(scenario.get("pt_affinity", 1.0)), 6),
        round(float(scenario.get("ebike_share", 0.0)), 6),
    )
    
    assigned_edges = None
    assigned_metadata = None
    if run_assignment and corridor_municipalities:
        temp_res = ModeChoiceResult(
            drive_od=od_by_mode["drive"],
            od_by_mode=od_by_mode,
            summary=summary,
            scenario=scenario,
            travel_times=travel_times,
            lengths=lengths,
            cache_key=cache_key,
        )
        assigned_edges, congested_time, congested_distance, metadata = run_route_assignment(
            context,
            temp_res,
            corridor_municipalities=corridor_municipalities,
            iterations=assignment_iterations,
            return_skims=True,
            generate_lut=True,
        )
        assigned_metadata = metadata
        if congested_time is not None and not congested_time.empty:
            drive_tt = travel_times.get("drive", context.travel_times.get("drive", travel_times.get("car"))).copy()
            for r in congested_time.index:
                if r in drive_tt.index:
                    cols = [c for c in congested_time.columns if c in drive_tt.columns]
                    drive_tt.loc[r, cols] = congested_time.loc[r, cols]
            travel_times["drive"] = drive_tt

    return ModeChoiceResult(
        drive_od=od_by_mode["drive"],
        od_by_mode=od_by_mode,
        summary=summary,
        scenario=scenario,
        travel_times=travel_times,
        lengths=lengths,
        cache_key=cache_key,
        assigned_edges=assigned_edges,
        assigned_metadata=assigned_metadata,
    )


# =============================================================================
# 4. CORRIDOR METRICS EXTRACTION (FOR CBA CALCULATOR)
# =============================================================================

def extract_corridor_metrics(
    context: TransportContext,
    mode_result: ModeChoiceResult,
    corridor_zone_ids: list[str] | set[str] | None = None,
    *,
    p_co2_kg_per_km: float = 0.139,
    min_distance_km: float = 0.0,
) -> dict[str, float]:
    """
    Extract aggregate physical and demand indicators from a mode choice run.
    
    The resulting dictionary is a direct drop-in input for simulation_engine.py!
    
    Returns:
        total_trips     : Total daily passenger trips
        car_trips       : Daily car driver trips
        pt_trips        : Daily public transport trips
        bike_trips      : Daily cycling trips
        walk_trips      : Daily walking trips
        car_share_trips : Car modal split share (raw trips, unfiltered)
        pt_share_trips  : PT modal split share (raw trips, unfiltered)
        bike_share_trips: Bike modal split share (raw trips, unfiltered)
        walk_share_trips: Walk modal split share (raw trips, unfiltered)
        car_share       : Car modal split share (PKM-based, strategic trips >= 5km)
        pt_share        : PT modal split share (PKM-based, strategic trips >= 5km)
        bike_share      : Bike modal split share (PKM-based, strategic trips >= 5km)
        walk_share      : Walk modal split share (PKM-based, strategic trips >= 5km)
        car_dist_km     : Daily vehicle-km travelled by car (VKT)
        car_tt_hours    : Daily total car in-vehicle travel time (hours)
        pt_tt_hours     : Daily total PT in-vehicle travel time (hours)
        co2_tonnes      : Daily car CO2 emissions (tonnes)
        avg_tt_min      : Average travel time across all corridor trips (minutes)
    """
    od_by_mode = mode_result.od_by_mode
    reference = next(iter(od_by_mode.values()))
    labels = reference.index.astype(str)
    
    if corridor_zone_ids is not None and len(corridor_zone_ids) > 0:
        internal = np.isin(labels, np.asarray(list(corridor_zone_ids), dtype=str))
        involved = internal[:, None] | internal[None, :]
    else:
        involved = np.ones((len(labels), len(labels)), dtype=bool)

    # Trips
    car_trips = float(od_by_mode["drive"].to_numpy(dtype=float)[involved].sum())
    pt_trips = float((od_by_mode["pt_walk"] + od_by_mode["pt_bike"]).to_numpy(dtype=float)[involved].sum())
    bike_trips = float(od_by_mode["bike"].to_numpy(dtype=float)[involved].sum())
    walk_trips = float(od_by_mode["walk"].to_numpy(dtype=float)[involved].sum())
    total_trips = car_trips + pt_trips + bike_trips + walk_trips

    # Distances & Travel times (always use baseline free-flow skims so delay is not double-counted)
    car_lengths = context.lengths.get("drive", context.lengths.get("car", mode_result.lengths.get("drive")))
    car_tt = context.travel_times.get("drive", context.travel_times.get("car", mode_result.travel_times.get("drive")))
    pt_tt = mode_result.travel_times.get("ivt_pt_walk", mode_result.travel_times.get("pt_walk", mode_result.travel_times.get("pt")))
    bike_tt = mode_result.travel_times.get("bike")

    car_lengths_km = np.nan_to_num(car_lengths.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0) / 1000.0 if car_lengths is not None else np.zeros((len(labels), len(labels)))
    car_tt_h = np.nan_to_num(car_tt.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0) / 60.0 if car_tt is not None else np.zeros((len(labels), len(labels)))
    pt_tt_h = np.nan_to_num(pt_tt.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0) / 60.0 if pt_tt is not None else np.zeros((len(labels), len(labels)))
    bike_tt_h = np.nan_to_num(bike_tt.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0) / 60.0 if bike_tt is not None else np.zeros((len(labels), len(labels)))

    car_vkt = float((od_by_mode["drive"].to_numpy(dtype=float) * car_lengths_km)[involved].sum())
    
    # === IDEA 1 & 2: Distance-weighted PKM Modal Split with Spatial Filtering ===
    # Filter out local trips (< min_distance_km) to avoid dilution of the core PT/Car metrics.
    # We use car_lengths_km as a proxy for physical OD distance for all modes.
    strategic_mask = involved & (car_lengths_km >= min_distance_km)
    
    car_pkm = float((od_by_mode["drive"].to_numpy(dtype=float) * car_lengths_km)[strategic_mask].sum())
    pt_pkm = float(((od_by_mode["pt_walk"] + od_by_mode["pt_bike"]).to_numpy(dtype=float) * car_lengths_km)[strategic_mask].sum())
    bike_pkm = float((od_by_mode["bike"].to_numpy(dtype=float) * car_lengths_km)[strategic_mask].sum())
    walk_pkm = float((od_by_mode["walk"].to_numpy(dtype=float) * car_lengths_km)[strategic_mask].sum())
    
    total_pkm = car_pkm + pt_pkm + bike_pkm + walk_pkm

    car_tt_total_hours = float((od_by_mode["drive"].to_numpy(dtype=float) * car_tt_h)[involved].sum())
    pt_tt_total_hours = float(((od_by_mode["pt_walk"] + od_by_mode["pt_bike"]).to_numpy(dtype=float) * pt_tt_h)[involved].sum())
    bike_tt_total_hours = float((od_by_mode["bike"].to_numpy(dtype=float) * bike_tt_h)[involved].sum())

    co2_kg = car_vkt * float(p_co2_kg_per_km)
    
    total_tt_hours = car_tt_total_hours + pt_tt_total_hours
    motorized_trips = car_trips + pt_trips
    avg_tt_min = (total_tt_hours / max(motorized_trips, 1.0)) * 60.0

    return {
        "total_trips": total_trips,
        "car_trips": car_trips,
        "pt_trips": pt_trips,
        "bike_trips": bike_trips,
        "walk_trips": walk_trips,
        
        # Original trip-based modal split (kept for reference/legacy checks)
        "car_share_trips": car_trips / max(total_trips, 1e-9),
        "pt_share_trips": pt_trips / max(total_trips, 1e-9),
        "bike_share_trips": bike_trips / max(total_trips, 1e-9),
        "walk_share_trips": walk_trips / max(total_trips, 1e-9),
        
        # Primary modal split: PKM-based for strategic trips (>= 5km)
        "car_share": car_pkm / max(total_pkm, 1e-9),
        "pt_share": pt_pkm / max(total_pkm, 1e-9),
        "bike_share": bike_pkm / max(total_pkm, 1e-9),
        "walk_share": walk_pkm / max(total_pkm, 1e-9),

        "car_dist_km": car_vkt,
        "car_tt_hours": car_tt_total_hours,
        "pt_tt_hours": pt_tt_total_hours,
        "co2_tonnes": co2_kg / 1000.0,
        "avg_tt_min": avg_tt_min,
    }


def run_simulation(
    context: TransportContext,
    *,
    stage: int,
    stage_specs: Mapping[int, Mapping[str, Any]],
    corridor_zone_ids: list[str] | set[str] | None = None,
    corridor_municipalities: list[str] | None = None,
    trip_rate_multiplier: float = 1.0,
    demand_multiplier: float | None = None,
    car_affinity: float | None = None,
    bike_affinity: float | None = None,
    pt_affinity: float | None = None,
    ebike_share: float | None = None,
) -> tuple[ModeChoiceResult, dict[str, float]]:
    """
    Unified function that executes mode choice and route assignment based on ASSIGNMENT_SETTINGS.
    
    Returns:
        mode_result (ModeChoiceResult): The raw FSM result.
        metrics (dict): The aggregated physical/demand metrics, augmented with delay if MSA is used.
    """
    # 1. Run mode choice
    mode_result = run_transport_mode_choice(
        context,
        stage=stage,
        stage_specs=stage_specs,
        corridor_zone_ids=corridor_zone_ids,
        corridor_municipalities=corridor_municipalities,
        trip_rate_multiplier=trip_rate_multiplier,
        demand_multiplier=demand_multiplier,
        car_affinity=car_affinity,
        bike_affinity=bike_affinity,
        pt_affinity=pt_affinity,
        ebike_share=ebike_share,
        run_assignment=False, # We handle it dynamically below
    )
    
    # 2. Extract base metrics
    min_dist_km = stage_specs[stage].get("_min_distance_km", 0.0) if stage_specs and stage in stage_specs else 0.0
    print("  • [Simulation] Extracting corridor metrics...")
    metrics = extract_corridor_metrics(context, mode_result, corridor_zone_ids, min_distance_km=min_dist_km)
    
    # 3. Handle Assignment based on toggle
    mult = demand_multiplier if demand_multiplier is not None else trip_rate_multiplier
    method = ASSIGNMENT_SETTINGS.get("method", "MSA").upper()
    regen = bool(ASSIGNMENT_SETTINGS.get("regenerate_lut", False))

    if method == "MSA" and corridor_municipalities:
        corridor = build_corridor_context(
            context,
            corridor_municipalities=list(corridor_municipalities),
            name="configured project corridor",
        )
        if bool(ASSIGNMENT_SETTINGS.get("modal_feedback", True)):
            assignment = run_coupled_corridor_assignment(
                context,
                corridor,
                mode_result,
                # ``mode_result`` already contains the requested demand scale.
                demand_multiplier=1.0,
                drive_occupancy=ASSIGNMENT_SETTINGS.get("drive_occupancy", 1.14),
                max_iterations=ASSIGNMENT_SETTINGS.get("max_iterations", 8),
                min_iterations=ASSIGNMENT_SETTINGS.get("min_iterations", 2),
                relative_gap_threshold=ASSIGNMENT_SETTINGS.get("relative_gap_threshold", 0.10),
                od_threshold=ASSIGNMENT_SETTINGS.get("od_threshold", 1.0),
            )
            mode_result = assignment.mode_result or mode_result
            # Mode choice has changed, so all physical indicators must be
            # recomputed from the converged multimodal matrices.
            metrics = extract_corridor_metrics(
                context,
                mode_result,
                corridor_zone_ids or corridor.zone_ids,
                min_distance_km=min_dist_km,
            )
        else:
            assignment = run_corridor_assignment(
                context,
                corridor,
                mode_result,
                drive_occupancy=ASSIGNMENT_SETTINGS.get("drive_occupancy", 1.14),
                max_iterations=ASSIGNMENT_SETTINGS.get("max_iterations", 8),
                min_iterations=ASSIGNMENT_SETTINGS.get("min_iterations", 2),
                relative_gap_threshold=ASSIGNMENT_SETTINGS.get("relative_gap_threshold", 0.10),
                od_threshold=ASSIGNMENT_SETTINGS.get("od_threshold", 1.0),
            )
        mode_result.assigned_edges = assignment.links
        mode_result.assigned_metadata = assignment.diagnostics
        metrics["congestion_delay_hours"] = assignment.diagnostics.get(
            "total_delay_hours", 0.0
        )
    elif regen and corridor_municipalities:
        # LUT regeneration remains an explicit maintenance operation.
        run_route_assignment(
            context,
            mode_result,
            corridor_municipalities=corridor_municipalities,
            iterations=ASSIGNMENT_SETTINGS.get("max_iterations", 8),
            generate_lut=True,
        )
        import simulation_engine as se
        metrics["congestion_delay_hours"] = se.get_lut_delay(stage, mult)
    elif method == "LUT":
        import simulation_engine as se
        metrics["congestion_delay_hours"] = se.get_lut_delay(stage, mult)
    
    return mode_result, metrics


# =============================================================================
# 5. INTERACTIVE DASHBOARD WIDGETS
# =============================================================================

def mode_choice_dashboard(
    context: TransportContext,
    stage_specs: Mapping[int, Mapping[str, Any]],
    corridor_zone_ids: list[str] | set[str] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Create a fast, responsive interactive Jupyter widget for mode-choice exploration."""
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    from IPython.display import clear_output, display, HTML

    state: dict[str, Any] = {"result": None, "metrics": None, "cache_key": None}
    
    controls = {
        "stage": widgets.Dropdown(
            options=[(f"{spec.get('name', f'Stage {s}')}", s) for s, spec in sorted(stage_specs.items())],
            value=0,
            description="Stage:",
            layout=widgets.Layout(width="340px"),
            style={"description_width": "60px"},
        ),
        "trip_rate_multiplier": widgets.FloatSlider(
            value=1.0, min=0.75, max=1.50, step=0.05,
            description="Demand:", continuous_update=False, readout_format=".2f",
            layout=widgets.Layout(width="280px"),
            style={"description_width": "65px"},
        ),
        "car_affinity": widgets.FloatSlider(
            value=1.0, min=0.50, max=1.50, step=0.05,
            description="Car beta:", continuous_update=False, readout_format=".2f",
            layout=widgets.Layout(width="240px"),
            style={"description_width": "70px"},
        ),
        "pt_affinity": widgets.FloatSlider(
            value=1.0, min=0.50, max=1.50, step=0.05,
            description="PT beta:", continuous_update=False, readout_format=".2f",
            layout=widgets.Layout(width="240px"),
            style={"description_width": "60px"},
        ),
        "bike_affinity": widgets.FloatSlider(
            value=1.0, min=0.50, max=2.00, step=0.05,
            description="Bike beta:", continuous_update=False, readout_format=".2f",
            layout=widgets.Layout(width="240px"),
            style={"description_width": "70px"},
        ),
    }
    
    output = widgets.Output(layout=widgets.Layout(overflow="visible", width="100%"))

    def update(**values: Any) -> None:
        key = tuple(values[name] for name in controls)
        started = time.perf_counter()
        try:
            if state["cache_key"] != key:
                state["result"] = run_transport_mode_choice(
                    context,
                    corridor_zone_ids=corridor_zone_ids,
                    stage_specs=stage_specs,
                    **values,
                )
                stage_val = int(values.get("stage", 0))
                min_dist_km = stage_specs[stage_val].get("_min_distance_km", 0.0) if stage_specs and stage_val in stage_specs else 0.0
                state["metrics"] = extract_corridor_metrics(
                    context,
                    state["result"],
                    corridor_zone_ids=corridor_zone_ids,
                    min_distance_km=min_dist_km,
                )
                state["cache_key"] = key
                
            result = state["result"]
            metrics = state["metrics"]
            runtime = time.perf_counter() - started
            
            with output:
                clear_output(wait=True)
                
                # 2. Side-by-Side: Table and Chart
                fig, (ax_table, ax_bar) = plt.subplots(
                    1, 2, figsize=(11.5, 2.7),
                    gridspec_kw={"width_ratios": [1.15, 1.25]}
                )
                
                # Build unified comparison data
                mode_rows = [
                    ("Car (Driver)", metrics["car_trips"], metrics["car_share_trips"], metrics["car_share"]),
                    ("Public Transport", metrics["pt_trips"], metrics["pt_share_trips"], metrics["pt_share"]),
                    ("Bicycle", metrics["bike_trips"], metrics["bike_share_trips"], metrics["bike_share"]),
                    ("Walking", metrics["walk_trips"], metrics["walk_share_trips"], metrics["walk_share"]),
                ]
                
                # Render clean table in left axes
                ax_table.axis("off")
                table_data = [["Mode", "Trips/day", "Trip Share", "PKM (>5km)"]]
                for m_name, m_trips, m_t_share, m_p_share in mode_rows:
                    table_data.append([m_name, f"{m_trips:,.0f}", f"{m_t_share:.1%}", f"{m_p_share:.1%}"])
                
                table = ax_table.table(
                    cellText=table_data,
                    loc="center",
                    cellLoc="center",
                )
                table.auto_set_font_size(False)
                table.set_fontsize(9.0)
                table.scale(1.0, 1.40)
                for (row_idx, col_idx), cell in table.get_celld().items():
                    if row_idx == 0:
                        cell.set_facecolor("#edf2f7")
                        cell.set_text_props(weight="bold", color="#2d3748")
                    cell.set_edgecolor("#cbd5e0")
                
                # Render Side-by-Side Bar Chart in right axes
                modes = [r[0] for r in mode_rows]
                trip_shares = [r[2] * 100 for r in mode_rows]
                pkm_shares = [r[3] * 100 for r in mode_rows]
                colors = ["#4C78A8", "#F28E2B", "#54A24B", "#76B7B2"]
                
                x = np.arange(len(modes))
                w = 0.35
                
                bars1 = ax_bar.bar(x - w/2, trip_shares, w, label="Trip Share", color=colors, alpha=0.45, edgecolor="black", linewidth=0.8)
                bars2 = ax_bar.bar(x + w/2, pkm_shares, w, label="PKM (>5km)", color=colors, hatch="//", edgecolor="black", linewidth=0.8)
                
                max_s = max(max(trip_shares), max(pkm_shares), 50.0)
                ax_bar.set_ylim(0, max_s * 1.25)
                ax_bar.set_xticks(x)
                ax_bar.set_xticklabels(["Car", "PT", "Bike", "Walk"], fontsize=9)
                ax_bar.set_ylabel("Modal share (%)", fontsize=9.0)
                ax_bar.set_title(f"Corridor Modal Split (Stage {int(values['stage'])})", fontsize=10.0, fontweight="bold", pad=8)
                ax_bar.grid(axis="y", linestyle="--", alpha=0.3)
                ax_bar.legend(loc="upper right", fontsize=8, frameon=True)
                ax_bar.set_axisbelow(True)
                
                for b in bars1:
                    h = b.get_height()
                    ax_bar.text(b.get_x() + b.get_width()/2.0, h + 0.8, f"{h:.0f}%", ha="center", va="bottom", fontsize=7.5)
                for b in bars2:
                    h = b.get_height()
                    ax_bar.text(b.get_x() + b.get_width()/2.0, h + 0.8, f"{h:.0f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
                
                plt.tight_layout()
                plt.show()
                
        except Exception as error:
            state["result"] = None
            state["cache_key"] = None
            with output:
                clear_output(wait=True)
                print(f"Error: {type(error).__name__}: {error}")

    interactive = widgets.interactive_output(update, controls)
    interactive.layout.display = "none"
    
    # Clean top control bar layout
    row1 = widgets.HBox([controls["stage"], controls["trip_rate_multiplier"]], layout=widgets.Layout(margin="0 0 6px 0"))
    row2 = widgets.HBox([controls["car_affinity"], controls["pt_affinity"], controls["bike_affinity"]], layout=widgets.Layout(margin="0 0 8px 0"))
    control_panel = widgets.VBox([row1, row2], layout=widgets.Layout(
        background_color="#ffffff",
        padding="10px 14px",
        border="1px solid #e2e8f0",
        border_radius="8px",
        margin="0 0 10px 0"
    ))
    
    ui = widgets.VBox([control_panel, output, interactive], layout=widgets.Layout(overflow="visible", width="100%"))
    return ui, state


def _get_or_generate_corridor(
    context: TransportContext,
    corridor_municipalities: list[str] | None = None,
    buffer_m: float = DEFAULT_CORRIDOR_BUFFER_M,
    max_gates: int = DEFAULT_MAX_GATES,
) -> tuple[Any, Any]:
    """Helper to extract the clipped corridor network and cordon gates."""
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    
    zones = context.zones.copy()
    if not corridor_municipalities:
        import parameters as p
        corridor_municipalities = getattr(p, "CORRIDOR_MUNICIPALITIES", None)
        if not corridor_municipalities and context.assignment_network and "metadata" in context.assignment_network:
            corridor_municipalities = context.assignment_network["metadata"].get("corridor_municipalities", [])
        if not corridor_municipalities:
            return context.assignment_network["edges"].copy(), None
            
    try:
        import parameters as p
        max_gates = getattr(p, "MAX_GATES", max_gates)
    except Exception:
        pass
            
    core_zones = _get_core_zones(zones, corridor_municipalities)
    if core_zones.empty:
        return context.assignment_network["edges"].copy(), None
        
    polygon = core_zones.geometry.union_all().buffer(float(buffer_m))
    network_edges = context.assignment_network["edges"]
    network_nodes = context.assignment_network["nodes"]
    
    node_inside = network_nodes.geometry.intersects(polygon)
    inside_by_node = dict(zip(network_nodes["node_id"].astype(int), node_inside.astype(bool)))
    source_inside = network_edges["source"].map(inside_by_node).fillna(False).astype(bool)
    target_inside = network_edges["target"].map(inside_by_node).fillna(False).astype(bool)
    
    local_edges = network_edges.loc[source_inside & target_inside].copy()
    used_nodes = set(local_edges["source"].astype(int)) | set(local_edges["target"].astype(int))
    local_nodes = network_nodes.loc[network_nodes["node_id"].astype(int).isin(used_nodes)]
    
    crossing = network_edges.loc[source_inside ^ target_inside].copy()
    crossing["inside_node"] = np.where(source_inside.loc[crossing.index], crossing["source"], crossing["target"]).astype(int)
    crossing["can_exit"] = source_inside.loc[crossing.index].to_numpy(dtype=bool)
    crossing["can_enter"] = target_inside.loc[crossing.index].to_numpy(dtype=bool)
    crossing = crossing.loc[crossing["inside_node"].isin(used_nodes)]
    
    node_geometry = local_nodes.set_index("node_id").geometry
    candidate_rows = []
    for node_id, group in crossing.groupby("inside_node", sort=False):
        point = node_geometry.get(node_id)
        if point is None:
            continue
        candidate_rows.append({
            "node_id": int(node_id),
            "geometry": point,
            "crossing_count": int(len(group)),
            "can_enter": bool(group["can_enter"].any()),
            "can_exit": bool(group["can_exit"].any()),
        })

    # Robust fallback: If road network was pre-clipped and has no crossing edges,
    # detect entry/exit gates from perimeter boundary nodes of the clipped network
    if not candidate_rows and not local_edges.empty:
        deg = local_edges["source"].value_counts().add(local_edges["target"].value_counts(), fill_value=0)
        perimeter_ids = deg[deg <= 2].index.astype(int)
        perim_nodes = local_nodes.loc[local_nodes["node_id"].astype(int).isin(perimeter_ids)]
        for _, r in perim_nodes.iterrows():
            candidate_rows.append({
                "node_id": int(r["node_id"]),
                "geometry": r["geometry"],
                "crossing_count": 1,
                "can_enter": True,
                "can_exit": True,
            })

    if not candidate_rows:
        return local_edges, gpd.GeoDataFrame(columns=["gate_id", "node_id", "geometry", "can_enter", "can_exit"], crs=local_edges.crs)

    candidate_gates = gpd.GeoDataFrame(candidate_rows, crs=network_nodes.crs)
    candidate_gates = candidate_gates.sort_values(
        by=["crossing_count", "can_enter", "can_exit"],
        ascending=[False, False, False],
    )

    # Spatial deduplication: distribute gates around perimeter
    selected_gates = []
    min_dist = float(DEFAULT_GATE_SEPARATION_M)
    for _, row in candidate_gates.iterrows():
        if len(selected_gates) >= max_gates:
            break
        point = row["geometry"]
        if not selected_gates or all(point.distance(s["geometry"]) >= min_dist for s in selected_gates):
            selected_gates.append(row)

    if not selected_gates and not candidate_gates.empty:
        selected_gates = [row for _, row in candidate_gates.head(max_gates).iterrows()]

    gates = gpd.GeoDataFrame(selected_gates, crs=network_nodes.crs).reset_index(drop=True)
    gates["gate_id"] = [f"G_{i+1:02d}" for i in range(len(gates))]
    return local_edges, gates


# =============================================================================
# 6. CORRIDOR CONTEXT AND CLICKABLE MAP EXPLORERS
# =============================================================================

# Shared palettes keep all network views visually consistent.  Continuous
# metrics use a blue-to-red scale; road classes use stable categorical colours.
_CONTINUOUS_PALETTE = np.asarray(
    [
        [44, 123, 182, 230],
        [102, 194, 165, 230],
        [255, 255, 191, 230],
        [253, 174, 97, 230],
        [215, 25, 28, 235],
    ],
    dtype=np.uint8,
)

_CATEGORY_PALETTE = np.asarray(
    [
        [76, 120, 168, 230],
        [245, 133, 24, 230],
        [84, 162, 75, 230],
        [228, 87, 86, 230],
        [178, 121, 162, 230],
        [114, 183, 178, 230],
        [255, 157, 167, 230],
        [156, 117, 95, 230],
        [186, 176, 172, 230],
        [237, 201, 72, 230],
    ],
    dtype=np.uint8,
)


def _numeric_style(
    values: pd.Series,
    *,
    breaks: list[float] | None = None,
    palette: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    """Return link colours and an HTML legend for one numeric metric."""
    full_palette = _CONTINUOUS_PALETTE if palette is None else np.asarray(palette, dtype=np.uint8)
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = numeric[np.isfinite(numeric)]
    if finite.size == 0:
        colors = np.repeat([[160, 160, 160, 210]], len(values), axis=0).astype(np.uint8)
        return colors, "<i>No finite values.</i>"

    if breaks is None:
        breaks_array = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, len(full_palette) + 1)))
        if len(breaks_array) < 3:
            minimum, maximum = float(finite.min()), float(finite.max())
            if np.isclose(minimum, maximum):
                maximum = minimum + 1.0
            breaks_array = np.linspace(minimum, maximum, len(full_palette) + 1)
    else:
        breaks_array = np.asarray(breaks, dtype=float)
        if len(breaks_array) < 2:
            raise ValueError("Numeric map breaks must contain at least two values.")

    n_colors = min(len(full_palette), max(len(breaks_array) - 1, 1))
    selected_palette = full_palette[:n_colors]
    indices = np.digitize(numeric, breaks_array[1:-1], right=False)
    indices = np.clip(indices, 0, n_colors - 1)
    colors = selected_palette[indices].copy()
    colors[~np.isfinite(numeric)] = [160, 160, 160, 180]

    swatches = []
    for index in range(n_colors):
        low, high = breaks_array[index], breaks_array[index + 1]
        if breaks is not None and index == 0:
            interval = f"&lt; {high:,.2f}"
        elif breaks is not None and index == n_colors - 1:
            interval = f"≥ {low:,.2f}"
        else:
            interval = f"{low:,.2f}–{high:,.2f}"
        rgba = selected_palette[index]
        colour = f"rgb({rgba[0]},{rgba[1]},{rgba[2]})"
        swatches.append(
            "<span style='display:inline-block;margin-right:14px'>"
            f"<span style='display:inline-block;width:14px;height:10px;background:{colour};"
            f"margin-right:4px'></span>{interval}</span>"
        )
    return colors.astype(np.uint8), "".join(swatches)


def _categorical_style(values: pd.Series) -> tuple[np.ndarray, str]:
    """Return stable colours and an HTML legend for categorical values."""
    labels = values.fillna("missing").astype(str)
    order = labels.value_counts().index.tolist()
    if len(order) > len(_CATEGORY_PALETTE):
        retained = order[: len(_CATEGORY_PALETTE) - 1]
        labels = labels.where(labels.isin(retained), "other")
        order = retained + ["other"]
    colour_lookup = {
        label: _CATEGORY_PALETTE[position % len(_CATEGORY_PALETTE)]
        for position, label in enumerate(order)
    }
    colors = np.vstack([colour_lookup[label] for label in labels]).astype(np.uint8)
    swatches = []
    for label in order:
        rgba = colour_lookup[label]
        colour = f"rgb({rgba[0]},{rgba[1]},{rgba[2]})"
        swatches.append(
            "<span style='display:inline-block;margin-right:14px'>"
            f"<span style='display:inline-block;width:14px;height:10px;background:{colour};"
            f"margin-right:4px'></span>{label}</span>"
        )
    return colors, "".join(swatches)


def _metric_colours(frame: pd.DataFrame, spec: Mapping[str, Any]) -> tuple[np.ndarray, str]:
    """Dispatch map styling for a metric specification."""
    column = str(spec["column"])
    if column not in frame:
        raise KeyError(f"Map metric column not found: {column}")
    if spec.get("kind", "numeric") == "categorical":
        return _categorical_style(frame[column])
    return _numeric_style(frame[column], breaks=spec.get("breaks"), palette=spec.get("palette"))


def _slim_geodataframe(frame: Any, columns: list[str], *, simplify_m: float = 0.0) -> Any:
    """Keep browser-facing fields only, simplify a copy, and reproject to WGS84."""
    keep = [column for column in columns if column in frame.columns]
    if "geometry" not in keep:
        keep.append("geometry")
    result = frame[keep].copy()
    if result.crs is None:
        raise ValueError("Map data need a declared CRS.")
    if simplify_m > 0 and not result.crs.is_geographic:
        result.geometry = result.geometry.simplify(float(simplify_m), preserve_topology=True)
    return result.to_crs(4326)


def _lonboard_map(layers: list[Any], *, height: int = 620) -> Any:
    """Construct a clickable Lonboard map across supported API versions."""
    from lonboard import Map

    kwargs = {"show_tooltip": True, "show_side_panel": True, "picking_radius": 5}
    try:
        return Map(layers, height=height, **kwargs)
    except TypeError:
        result = Map(layers, **kwargs)
        if hasattr(result, "_height"):
            result._height = int(height)
        try:
            result.layout.height = f"{int(height)}px"
        except Exception:
            pass
        return result


def _select_spaced_gates(
    candidates: Any,
    *,
    max_gates: int,
    minimum_separation_m: float,
) -> Any:
    """Prefer high-capacity bidirectional gates while keeping them spatially distinct."""
    if candidates.empty or int(max_gates) < 1:
        return candidates.iloc[0:0].copy()
    ordered = candidates.sort_values(
        ["capacity_vph", "both_directions"], ascending=[False, False]
    )
    selected = []
    both = ordered.loc[ordered["both_directions"].astype(bool)]
    if len(both):
        selected.append(next(both.itertuples(index=False)))
    else:
        entries = ordered.loc[ordered["can_enter"].astype(bool)]
        exits = ordered.loc[ordered["can_exit"].astype(bool)]
        if entries.empty or exits.empty or int(max_gates) < 2:
            raise ValueError("The cordon needs at least one valid entry gate and one exit gate.")
        selected.extend([next(entries.itertuples(index=False)), next(exits.itertuples(index=False))])

    for row in ordered.itertuples(index=False):
        if len(selected) >= int(max_gates):
            break
        if any(row.node_id == existing.node_id for existing in selected):
            continue
        if selected and min(row.geometry.distance(existing.geometry) for existing in selected) < float(minimum_separation_m):
            continue
        selected.append(row)
    selected_ids = [row.node_id for row in selected]
    return candidates.loc[candidates["node_id"].isin(selected_ids)].copy()


def build_corridor_context(
    context: TransportContext,
    corridor_municipalities: list[str] | None = None,
    *,
    buffer_m: float = DEFAULT_CORRIDOR_BUFFER_M,
    max_gates: int = DEFAULT_MAX_GATES,
    gate_separation_m: float = DEFAULT_GATE_SEPARATION_M,
    name: str = "MehrSpur",
) -> CorridorContext:
    """Build one auditable corridor topology shared by maps, OD tables, and assignment.

    Complete inside-to-inside links are retained; links are never geometrically
    cut.  Each gate is the inside endpoint of a directed boundary-crossing link.
    External zones are mapped separately to eligible entry and exit gates.
    """
    import geopandas as gpd

    if not context.assignment_network or "edges" not in context.assignment_network:
        raise ValueError("The transport context does not contain an assignment network.")
    if not corridor_municipalities:
        import parameters as p
        corridor_municipalities = list(getattr(p, "CORRIDOR_MUNICIPALITIES", []))
    if not corridor_municipalities:
        raise ValueError("No corridor municipalities were supplied or configured.")

    zones = context.zones.copy()
    core_zones = _get_core_zones(zones, list(corridor_municipalities))
    if core_zones.empty:
        raise ValueError(f"No zones matched the configured {name} corridor.")
    polygon = core_zones.geometry.union_all().buffer(float(buffer_m))
    polygon_gdf = gpd.GeoDataFrame(
        {"name": [f"{name} modelling cordon"]}, geometry=[polygon], crs=zones.crs
    )

    network_edges = context.assignment_network["edges"].copy()
    network_nodes = context.assignment_network["nodes"].copy()
    node_inside = network_nodes.geometry.intersects(polygon)
    inside_by_node = dict(zip(network_nodes["node_id"].astype(int), node_inside.astype(bool)))
    source_inside = network_edges["source"].map(inside_by_node).fillna(False).astype(bool)
    target_inside = network_edges["target"].map(inside_by_node).fillna(False).astype(bool)

    local_edges = network_edges.loc[source_inside & target_inside].copy()
    if local_edges.empty:
        raise ValueError(f"The {name} cordon contains no complete road links.")
    used_nodes = set(local_edges["source"].astype(int)) | set(local_edges["target"].astype(int))
    local_nodes = network_nodes.loc[network_nodes["node_id"].astype(int).isin(used_nodes)].copy()

    crossing = network_edges.loc[source_inside ^ target_inside].copy()
    crossing["inside_node"] = np.where(
        source_inside.loc[crossing.index], crossing["source"], crossing["target"]
    ).astype(int)
    crossing["can_exit"] = source_inside.loc[crossing.index].to_numpy(dtype=bool)
    crossing["can_enter"] = target_inside.loc[crossing.index].to_numpy(dtype=bool)
    crossing = crossing.loc[crossing["inside_node"].isin(used_nodes)].copy()

    node_geometry = local_nodes.set_index("node_id").geometry
    candidate_rows = []
    for node_id, group in crossing.groupby("inside_node", sort=False):
        capacity_values = group.get(
            "capacity_vph", pd.Series(0.0, index=group.index)
        )
        capacity = pd.to_numeric(capacity_values, errors="coerce").max()
        highway = group.get("highway", pd.Series("unknown", index=group.index)).astype(str).mode()
        can_enter, can_exit = bool(group["can_enter"].any()), bool(group["can_exit"].any())
        candidate_rows.append({
            "node_id": int(node_id),
            "capacity_vph": float(capacity) if np.isfinite(capacity) else 0.0,
            "highway": highway.iloc[0] if len(highway) else "unknown",
            "can_enter": can_enter,
            "can_exit": can_exit,
            "both_directions": can_enter and can_exit,
            "direction": "entry + exit" if can_enter and can_exit else "entry" if can_enter else "exit",
            "geometry": node_geometry.loc[int(node_id)],
        })
    candidates = gpd.GeoDataFrame(
        candidate_rows,
        columns=["node_id", "capacity_vph", "highway", "can_enter", "can_exit", "both_directions", "direction", "geometry"],
        geometry="geometry",
        crs=zones.crs,
    )
    gates = _select_spaced_gates(
        candidates,
        max_gates=int(max_gates),
        minimum_separation_m=float(gate_separation_m),
    ).reset_index(drop=True)
    if gates.empty:
        raise ValueError(f"The {name} cordon produced no usable boundary gates.")
    gates["gate_id"] = [f"G{position:02d}" for position in range(1, len(gates) + 1)]

    original_zone_node_map = {
        str(zone): int(node) for zone, node in context.assignment_network["zone_node_map"].items()
    }
    centroids = gpd.GeoSeries(zones["centroid"], crs=zones.crs)
    internal_ids = [
        str(zone_id)
        for zone_id in zones.loc[centroids.intersects(polygon), "grid_id"].astype(str)
        if str(zone_id) in original_zone_node_map and original_zone_node_map[str(zone_id)] in used_nodes
    ]
    corridor_zones = zones.loc[zones["grid_id"].astype(str).isin(internal_ids)].copy()
    zone_node_map = {zone_id: original_zone_node_map[zone_id] for zone_id in internal_ids}
    zone_node_map.update(dict(zip(gates["gate_id"].astype(str), gates["node_id"].astype(int))))

    external = zones.loc[~zones["grid_id"].astype(str).isin(internal_ids)].copy()
    external_centroids = gpd.GeoSeries(external["centroid"], crs=zones.crs)
    external_xy = np.column_stack((external_centroids.x, external_centroids.y))

    def nearest_gate_mapping(eligible_gates: Any) -> dict[str, str]:
        if eligible_gates.empty:
            raise ValueError("Direction-aware external demand needs an eligible gate.")
        if external.empty:
            return {}
        gate_xy = np.column_stack((eligible_gates.geometry.x, eligible_gates.geometry.y))
        squared_distance = (
            (external_xy[:, None, 0] - gate_xy[None, :, 0]) ** 2
            + (external_xy[:, None, 1] - gate_xy[None, :, 1]) ** 2
        )
        nearest = np.argmin(squared_distance, axis=1)
        return dict(zip(external["grid_id"].astype(str), eligible_gates.iloc[nearest]["gate_id"].astype(str)))

    entry_map = nearest_gate_mapping(gates.loc[gates["can_enter"].astype(bool)])
    exit_map = nearest_gate_mapping(gates.loc[gates["can_exit"].astype(bool)])
    metadata = {
        "name": name,
        "buffer_m": float(buffer_m),
        "max_gates": int(max_gates),
        "gate_separation_m": float(gate_separation_m),
        "internal_zones": len(internal_ids),
        "local_links": len(local_edges),
        "local_nodes": len(local_nodes),
        "candidate_boundary_nodes": len(candidates),
        "selected_gates": len(gates),
        "entry_gates": int(gates["can_enter"].sum()),
        "exit_gates": int(gates["can_exit"].sum()),
        "gate_definition": "inside endpoint of a boundary-crossing directed link",
    }
    return CorridorContext(
        polygon=polygon,
        polygon_gdf=polygon_gdf,
        zones=corridor_zones,
        zone_ids=internal_ids,
        edges=local_edges,
        nodes=local_nodes,
        gates=gates,
        zone_node_map=zone_node_map,
        external_zone_to_entry_gate=entry_map,
        external_zone_to_exit_gate=exit_map,
        metadata=metadata,
    )


def build_mehrspur_corridor(context: TransportContext, **kwargs: Any) -> CorridorContext:
    """Convenience wrapper using the corridor configured in ``parameters.py``."""
    return build_corridor_context(context, name="MehrSpur", **kwargs)


def network_explorer(
    edges: Any,
    metrics: Mapping[str, Mapping[str, Any]],
    *,
    nodes: Any | None = None,
    gates: Any | None = None,
    polygon: Any | None = None,
    width_column: str | None = None,
    title: str = "Network explorer",
    simplify_m: float = 3.0,
    height: int = 620,
) -> Any:
    """Create one clickable, metric-selectable network widget for notebooks.

    Only lightweight WGS84 copies are sent to the browser. Model data remain
    in their projected CRS and are never modified. Link attributes appear on
    hover or, after a click, in Lonboard's persistent side panel.
    """
    import ipywidgets as widgets
    from lonboard import PathLayer, ScatterplotLayer, SolidPolygonLayer

    if not metrics:
        raise ValueError("At least one map metric must be supplied.")
    if edges is None or len(edges) == 0:
        raise ValueError("The network explorer received no road links.")

    metric_columns = [str(spec["column"]) for spec in metrics.values()]
    popup_columns = [
        "edge_id", "source", "target", "source_id", "target_id", "highway",
        "drive_area", "length_m", "speed_kph", "lanes", "capacity_vph",
        "free_flow_time_min", "flow_vehicles", "volume_capacity_ratio",
        "time_min", "delay_min", "assigned_speed_kph", "stage2_lane_converted",
        *metric_columns,
    ]
    links_map = _slim_geodataframe(
        edges, list(dict.fromkeys(popup_columns)), simplify_m=simplify_m
    ).dropna(subset=["geometry"])
    links_map = links_map.loc[~links_map.geometry.is_empty].reset_index(drop=True)
    if links_map.empty:
        raise ValueError("The network explorer received no valid link geometry.")

    first_label = next(iter(metrics))
    link_colours, legend_html = _metric_colours(links_map, metrics[first_label])
    if width_column and width_column in links_map:
        width_values = pd.to_numeric(links_map[width_column], errors="coerce").fillna(0.0)
        upper = max(float(width_values.quantile(0.98)), 1e-9)
        link_width = 1.0 + 5.0 * np.sqrt(np.clip(width_values / upper, 0.0, 1.0))
    else:
        link_width = np.full(len(links_map), 1.6)

    link_layer = PathLayer.from_geopandas(
        links_map,
        get_color=link_colours,
        get_width=np.asarray(link_width, dtype=np.float32),
        width_units="pixels",
        width_min_pixels=1,
        width_max_pixels=7,
        auto_highlight=True,
        # pyrefly: ignore [unexpected-keyword]
        highlight_color=[20, 20, 20, 180],
        pickable=True,
    )

    layers: list[Any] = []
    if polygon is not None and len(polygon):
        polygon_map = _slim_geodataframe(polygon, ["name"], simplify_m=5.0)
        layers.append(
            SolidPolygonLayer.from_geopandas(
                polygon_map,
                get_fill_color=[35, 120, 180, 25],
                get_line_color=[35, 120, 180, 180],
                filled=True,
                pickable=True,
            )
        )
    layers.append(link_layer)

    node_layer = None
    if nodes is not None and len(nodes):
        node_columns = ["node_id", "nodeID", "zone_id", "is_zone", "x", "y", "geometry"]
        nodes_map = _slim_geodataframe(nodes, node_columns).reset_index(drop=True)
        node_layer = ScatterplotLayer.from_geopandas(
            nodes_map,
            get_fill_color=[48, 92, 160, 180],
            get_line_color=[255, 255, 255, 230],
            get_radius=3.0,
            radius_units="pixels",
            radius_min_pixels=2,
            radius_max_pixels=6,
            stroked=True,
            auto_highlight=True,
            pickable=True,
        )
        layers.append(node_layer)

    gate_layer = None
    if gates is not None and len(gates):
        gate_columns = [
            "gate_id", "node_id", "direction", "can_enter", "can_exit",
            "capacity_vph", "highway", "geometry",
        ]
        gates_map = _slim_geodataframe(gates, gate_columns).reset_index(drop=True)
        gate_layer = ScatterplotLayer.from_geopandas(
            gates_map,
            get_fill_color=[128, 0, 128, 235],
            get_line_color=[255, 255, 255, 255],
            get_radius=7.0,
            radius_units="pixels",
            radius_min_pixels=5,
            radius_max_pixels=11,
            line_width_min_pixels=2,
            stroked=True,
            auto_highlight=True,
            pickable=True,
        )
        layers.append(gate_layer)

    map_widget = _lonboard_map(layers, height=height)
    selector = widgets.Dropdown(
        options=list(metrics), value=first_label, description="Colour:",
        layout=widgets.Layout(width="420px"),
    )
    legend = widgets.HTML(
        value=f"<b>{first_label}</b><br>{legend_html}",
        layout=widgets.Layout(width="100%"),
    )
    toggles = []
    if node_layer is not None:
        node_toggle = widgets.Checkbox(value=True, description="show nodes/zones")
        node_toggle.observe(
            lambda change: setattr(node_layer, "visible", bool(change["new"])),
            names="value",
        )
        toggles.append(node_toggle)
    if gate_layer is not None:
        gate_toggle = widgets.Checkbox(value=True, description="show gates")
        gate_toggle.observe(
            lambda change: setattr(gate_layer, "visible", bool(change["new"])),
            names="value",
        )
        toggles.append(gate_toggle)

    def recolour(change: dict[str, Any]) -> None:
        label = change["new"]
        colours, html = _metric_colours(links_map, metrics[label])
        link_layer.get_color = colours
        legend.value = f"<b>{label}</b><br>{html}"

    selector.observe(recolour, names="value")
    controls = widgets.HBox([selector, *toggles])
    help_text = widgets.HTML(
        "<span style='color:#555'>Hover for a tooltip; click a segment or "
        "point to keep its attributes in the map side panel.</span>"
    )
    heading = widgets.HTML(f"<h4 style='margin:4px 0'>{title}</h4>")
    return widgets.VBox([heading, controls, legend, help_text, map_widget])


def _network_characteristic_metrics() -> dict[str, dict[str, Any]]:
    """Metric catalogue shared by full-network and corridor maps."""
    return {
        "Road class": {"column": "highway", "kind": "categorical"},
        "Free-flow speed (km/h)": {"column": "speed_kph"},
        "Directional capacity (veh/h)": {"column": "capacity_vph"},
        "Lanes": {"column": "lanes"},
        "Link length (m)": {"column": "length_m"},
        "Free-flow time (min)": {"column": "free_flow_time_min"},
    }


def corridor_explorer(corridor: CorridorContext) -> Any:
    """Show retained links, internal zone nodes, cordon gates, and attributes."""
    zone_nodes = corridor.nodes
    if "is_zone" in zone_nodes:
        zone_nodes = zone_nodes.loc[zone_nodes["is_zone"].astype(bool)].copy()
    return network_explorer(
        corridor.edges,
        _network_characteristic_metrics(),
        nodes=zone_nodes,
        gates=corridor.gates,
        polygon=corridor.polygon_gdf,
        title="MehrSpur topology and cordon gates (no assignment)",
        simplify_m=2.0,
        height=650,
    )

def _legacy_corridor_network_explorer(
    context: TransportContext,
    corridor_municipalities: list[str] | None = None,
) -> Any:
    """Clickable map of the clipped corridor road network and cordon gates."""
    try:
        from lonboard import Map, PathLayer, ScatterplotLayer
    except ImportError:
        print("Lonboard is not installed; interactive WebGL map skipped.")
        return None

    if context.assignment_network is None or "edges" not in context.assignment_network:
        print("No assignment network loaded in context.")
        return None

    edges, gates = _get_or_generate_corridor(context, corridor_municipalities)
    if edges is None or len(edges) == 0:
        print("Corridor assignment network edges are empty.")
        return None

    if edges.crs is None:
        edges = edges.set_crs(2056, allow_override=True)
    if not edges.crs.is_geographic:
        edges = edges.to_crs(4326)
        
    edges = edges.dropna(subset=["geometry"])
    edges = edges.loc[~edges.geometry.is_empty]
        
    cols = [c for c in ["highway", "speed_kph", "lanes"] if c in edges.columns]
    layer = PathLayer.from_geopandas(
        edges[cols + ["geometry"]],
        width_min_pixels=1,
        width_max_pixels=5,
    )
    layers = [layer]
    
    if gates is not None and not gates.empty:
        gates = gates.copy()
        if gates.crs is None:
            gates = gates.set_crs(2056, allow_override=True)
        if not gates.crs.is_geographic:
            gates = gates.to_crs(4326)
        
        import numpy as np
        colors = np.zeros((len(gates), 4), dtype=np.uint8)
        colors[:] = [122, 1, 119, 255]
        
        gate_layer = ScatterplotLayer.from_geopandas(
            gates[["gate_id", "geometry"]],
            get_fill_color=colors,
            get_radius=80,
            radius_min_pixels=6,
            radius_max_pixels=12,
        )
        layers.append(gate_layer)
        
    return Map(layers)


def _legacy_full_network_explorer(
    context: TransportContext,
    corridor_municipalities: list[str] | None = None,
) -> Any:
    """Clickable map of road network characteristics (full canton if corridor_municipalities is None)."""
    if corridor_municipalities is not None:
        return _legacy_corridor_network_explorer(context, corridor_municipalities)

    try:
        from lonboard import Map, PathLayer
    except ImportError:
        print("Lonboard is not installed; interactive WebGL map skipped.")
        return None

    if context.assignment_network is None or "edges" not in context.assignment_network:
        print("No assignment network loaded in context.")
        return None

    # Load full Canton Zürich road network
    edges = context.assignment_network["edges"].copy()
    if edges is None or len(edges) == 0:
        print("Assignment network edges are empty.")
        return None

    if edges.crs is None:
        edges = edges.set_crs(2056, allow_override=True)
    if not edges.crs.is_geographic:
        edges = edges.to_crs(4326)
        
    edges = edges.dropna(subset=["geometry"])
    edges = edges.loc[~edges.geometry.is_empty]
        
    cols = [c for c in ["highway", "speed_kph", "lanes"] if c in edges.columns]
    layer = PathLayer.from_geopandas(
        edges[cols + ["geometry"]],
        width_min_pixels=1,
        width_max_pixels=5,
    )
    return Map([layer])


def corridor_network_explorer(
    context: TransportContext,
    corridor_municipalities: list[str] | None = None,
) -> Any:
    """Backward-compatible shortcut for the richer corridor explorer."""
    try:
        corridor = build_corridor_context(
            context,
            corridor_municipalities=corridor_municipalities,
            name="MehrSpur",
        )
        return corridor_explorer(corridor)
    except ImportError:
        print("Lonboard or ipywidgets is not installed; interactive map skipped.")
        return None


def full_network_explorer(
    context: TransportContext,
    corridor_municipalities: list[str] | None = None,
) -> Any:
    """Clickable metric explorer for the full network or a selected corridor."""
    if corridor_municipalities is not None:
        return corridor_network_explorer(context, corridor_municipalities)
    if context.assignment_network is None or "edges" not in context.assignment_network:
        print("No assignment network loaded in context.")
        return None
    try:
        nodes = context.assignment_network["nodes"]
        # Drawing every intersection marker on a large graph is costly and
        # obscures the roads, so the canton-wide view keeps zone nodes only.
        if "is_zone" in nodes and len(nodes) > 20_000:
            nodes = nodes.loc[nodes["is_zone"].astype(bool)].copy()
        return network_explorer(
            context.assignment_network["edges"],
            _network_characteristic_metrics(),
            nodes=nodes,
            title="Canton Zurich road network - characteristics only (no assignment)",
            simplify_m=4.0,
            height=650,
        )
    except ImportError:
        print("Lonboard or ipywidgets is not installed; interactive map skipped.")
        return None


def collapse_od_to_gates(
    matrix: pd.DataFrame,
    context: TransportContext,
    corridor: CorridorContext,
    *,
    passthrough_fraction: float = DEFAULT_PASSTHROUGH_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse canton-wide demand into internal zones and cordon gates.

    The reduced matrix preserves internal-to-internal, inbound, and outbound
    demand exactly. External-to-external demand is grouped by its nearest
    direction-compatible gates and multiplied by ``passthrough_fraction``:
    proximity alone cannot establish which of those trips crosses the cordon.
    """
    fraction = float(np.clip(passthrough_fraction, 0.0, 1.0))
    zone_ids = context.zones["grid_id"].astype(str).tolist()
    values = (
        matrix.copy()
        .set_axis(matrix.index.astype(str), axis=0)
        .set_axis(matrix.columns.astype(str), axis=1)
        .reindex(index=zone_ids, columns=zone_ids, fill_value=0.0)
    )

    internal = [zone for zone in corridor.zone_ids if zone in values.index]
    internal_set = set(internal)
    external = [zone for zone in zone_ids if zone not in internal_set]
    gates = corridor.gates["gate_id"].astype(str).tolist()
    labels = internal + gates
    reduced = pd.DataFrame(0.0, index=labels, columns=labels)
    if internal:
        reduced.loc[internal, internal] = values.loc[internal, internal].to_numpy(dtype=float)

    entry_groups: dict[str, list[str]] = {gate: [] for gate in gates}
    exit_groups: dict[str, list[str]] = {gate: [] for gate in gates}
    for zone in external:
        entry_gate = corridor.external_zone_to_entry_gate.get(zone)
        exit_gate = corridor.external_zone_to_exit_gate.get(zone)
        if entry_gate in entry_groups:
            entry_groups[entry_gate].append(zone)
        if exit_gate in exit_groups:
            exit_groups[exit_gate].append(zone)

    # Inbound trips enter at a gate; outbound trips leave at a gate.
    for gate, external_zones in entry_groups.items():
        if external_zones and internal:
            reduced.loc[gate, internal] += values.loc[external_zones, internal].sum(axis=0)
    for gate, external_zones in exit_groups.items():
        if external_zones and internal:
            reduced.loc[internal, gate] += values.loc[internal, external_zones].sum(axis=1)

    # Retain only the calibrated share of external trips likely to cross the
    # corridor, excluding same-gate movements that do not traverse it.
    for origin_gate, origin_zones in entry_groups.items():
        if not origin_zones:
            continue
        for destination_gate, destination_zones in exit_groups.items():
            if origin_gate == destination_gate or not destination_zones:
                continue
            reduced.loc[origin_gate, destination_gate] += (
                values.loc[origin_zones, destination_zones].to_numpy(dtype=float).sum()
                * fraction
            )

    category_values = [
        float(reduced.loc[internal, internal].to_numpy(dtype=float).sum()) if internal else 0.0,
        float(reduced.loc[gates, internal].to_numpy(dtype=float).sum()) if internal and gates else 0.0,
        float(reduced.loc[internal, gates].to_numpy(dtype=float).sum()) if internal and gates else 0.0,
        float(reduced.loc[gates, gates].to_numpy(dtype=float).sum()) if gates else 0.0,
    ]
    breakdown = pd.DataFrame(
        {
            "category": [
                "internal -> internal", "gate -> internal",
                "internal -> gate", "gate -> gate (pass-through)",
            ],
            "trips": category_values,
        }
    )
    breakdown["share_of_retained"] = breakdown["trips"] / max(
        float(breakdown["trips"].sum()), 1e-9
    )

    if not np.isclose(
        float(breakdown["trips"].sum()),
        float(reduced.to_numpy(dtype=float).sum()),
        rtol=1e-9,
        atol=1e-6,
    ):
        raise AssertionError("Cordon OD categories do not reconcile with the reduced matrix.")
    missing = sorted(set(labels).difference(corridor.zone_node_map))
    if missing:
        raise AssertionError(f"Reduced OD labels have no routing node: {missing}")
    return reduced, breakdown


def gate_totals(demand: pd.DataFrame, corridor: CorridorContext) -> pd.DataFrame:
    """Summarise inbound, outbound, and pass-through demand at every gate."""
    gates = corridor.gates["gate_id"].astype(str).tolist()
    internal = corridor.zone_ids
    rows = []
    for gate in gates:
        inbound = float(demand.loc[gate, internal].sum()) if internal else 0.0
        outbound = float(demand.loc[internal, gate].sum()) if internal else 0.0
        pass_in = float(demand.loc[gates, gate].sum() - demand.loc[gate, gate])
        pass_out = float(demand.loc[gate, gates].sum() - demand.loc[gate, gate])
        rows.append(
            {
                "gate_id": gate,
                "to_internal": inbound,
                "from_internal": outbound,
                "passthrough_in": pass_in,
                "passthrough_out": pass_out,
                "cordon_total": inbound + outbound + 0.5 * (pass_in + pass_out),
            }
        )
    return pd.DataFrame(rows)


def corridor_detail_table(edges: pd.DataFrame) -> pd.DataFrame:
    """Summarise retained link count and length by road class."""
    if edges is None or len(edges) == 0:
        return pd.DataFrame(
            columns=["highway", "edges", "total_length_km", "mean_length_m", "share_%"]
        )
    frame = edges.copy()
    highway_values = frame.get("highway", pd.Series("unknown", index=frame.index))
    length_values = frame.get("length_m", pd.Series(0.0, index=frame.index))
    frame["highway"] = highway_values.fillna("unknown").astype(str)
    frame["length_m"] = pd.to_numeric(length_values, errors="coerce").fillna(0.0)
    table = (
        frame.groupby("highway", dropna=False)
        .agg(
            edges=("length_m", "size"),
            total_length_km=("length_m", "sum"),
            mean_length_m=("length_m", "mean"),
        )
        .sort_values("edges", ascending=False)
    )
    table["total_length_km"] = (table["total_length_km"] / 1000.0).round(1)
    table["mean_length_m"] = table["mean_length_m"].round(0)
    table["share_%"] = (100.0 * table["edges"] / table["edges"].sum()).round(1)
    return table.reset_index()


def corridor_dashboard(
    context: TransportContext,
    corridor: CorridorContext,
    *,
    demand_matrix: pd.DataFrame | None = None,
    passthrough_fraction: float = DEFAULT_PASSTHROUGH_FRACTION,
) -> Any:
    """Package the corridor map and audit tables into one tabbed widget.

    This keeps notebook cells short while leaving every generated table
    available through the standalone functions above for later calculations.
    """
    import ipywidgets as widgets
    from IPython.display import display

    def table_output(value: Any) -> Any:
        output = widgets.Output()
        with output:
            display(value)
        return output

    gate_columns = [
        column for column in
        ["gate_id", "node_id", "direction", "highway", "capacity_vph"]
        if column in corridor.gates
    ]
    overview = widgets.VBox(
        [
            table_output(pd.Series(corridor.metadata, name="value").to_frame()),
            table_output(corridor.gates[gate_columns]),
        ]
    )
    children = [corridor_explorer(corridor), overview, table_output(corridor_detail_table(corridor.edges))]
    titles = ["Interactive map", "Overview and gates", "Road detail"]

    selected_demand = context.baseline_od if demand_matrix is None else demand_matrix
    if isinstance(selected_demand, pd.DataFrame) and not selected_demand.empty:
        reduced, breakdown = collapse_od_to_gates(
            selected_demand,
            context,
            corridor,
            passthrough_fraction=passthrough_fraction,
        )
        children.append(widgets.VBox([table_output(breakdown), table_output(gate_totals(reduced, corridor))]))
        titles.append("Cordon demand")

    tabs = widgets.Tab(children=children)
    for index, title in enumerate(titles):
        tabs.set_title(index, title)
    return tabs


def _legacy_assignment_explorer(assigned_edges: Any) -> Any:
    """Clickable result map of assigned corridor links showing flow and V/C ratios with traffic colors."""
    try:
        from lonboard import Map, PathLayer
    except ImportError:
        print("Lonboard is not installed; interactive WebGL map skipped.")
        return None

    if assigned_edges is None or len(assigned_edges) == 0:
        print("No assigned edges to display.")
        return None

    edges = assigned_edges.copy()
    if edges.crs is None:
        edges = edges.set_crs(2056, allow_override=True)
    if not edges.crs.is_geographic:
        edges = edges.to_crs(4326)

    edges = edges.dropna(subset=["geometry"])
    edges = edges.loc[~edges.geometry.is_empty]

    # Calculate dynamic congestion colors based on Volume/Capacity (V/C) ratio
    import numpy as np
    import pandas as pd
    
    vc = pd.to_numeric(edges.get("volume_capacity_ratio", 0.0), errors="coerce").fillna(0.0).to_numpy()
    flows = pd.to_numeric(edges.get("flow_vehicles", 0.0), errors="coerce").fillna(0.0).to_numpy()

    colors = np.zeros((len(edges), 4), dtype=np.uint8)
    for i, val in enumerate(vc):
        if val < 0.60:
            colors[i] = [46, 204, 113, 220]   # Emerald Green (Free Flow)
        elif val < 0.85:
            colors[i] = [241, 196, 15, 230]  # Amber Yellow (Moderate Traffic)
        elif val < 1.00:
            colors[i] = [230, 126, 34, 245]  # Orange (Heavy Traffic)
        else:
            colors[i] = [231, 76, 60, 255]   # Crimson Red (Congested / Bottleneck)

    # Scale width dynamically with volume of vehicles
    p95 = float(np.percentile(flows, 95)) if len(flows) > 0 else 1.0
    max_f = max(p95, 1.0)
    widths = np.clip(1.5 + 6.5 * (flows / max_f), 1.5, 8.0).astype(np.float32)

    columns = [c for c in ["highway", "flow_vehicles", "volume_capacity_ratio", "time_min", "assigned_speed_kph", "delay_min"] if c in edges.columns]
    layer = PathLayer.from_geopandas(
        edges[columns + ["geometry"]],
        get_color=colors,
        get_width=widths,
        width_min_pixels=1.5,
        width_max_pixels=8.0,
    )
    
    from IPython.display import HTML, display
    legend_html = """
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; font-size: 13px; color: #000000; margin-bottom: 8px; padding: 10px 14px; background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 6px; display: inline-block;">
        <strong style="margin-right: 14px; color: #000000;">Traffic Congestion (V/C Ratio):</strong>
        <span style="margin-right: 14px; color: #000000;"><span style="color: #2ecc71; font-size: 16px; vertical-align: -2px;">■</span> Free Flow (&lt;0.60)</span>
        <span style="margin-right: 14px; color: #000000;"><span style="color: #f1c40f; font-size: 16px; vertical-align: -2px;">■</span> Moderate (0.60 - 0.85)</span>
        <span style="margin-right: 14px; color: #000000;"><span style="color: #e67e22; font-size: 16px; vertical-align: -2px;">■</span> Heavy (0.85 - 1.00)</span>
        <span style="color: #000000;"><span style="color: #e74c3c; font-size: 16px; vertical-align: -2px;">■</span> Congested (&ge;1.00)</span>
    </div>
    """
    display(HTML(legend_html))
    
    return Map([layer])


def assignment_explorer(
    assignment: AssignmentResult | Any,
    corridor: CorridorContext | None = None,
) -> Any:
    """Interactive assignment map with selectable link-performance metrics.

    Passing a bare link GeoDataFrame is retained for older notebooks.  A full
    ``AssignmentResult`` together with its corridor additionally displays the
    modelling cordon and direction-aware gates.
    """

    links = assignment.links if isinstance(assignment, AssignmentResult) else assignment
    if links is None or len(links) == 0:
        raise ValueError("No assigned road links are available to display.")

    metrics = {
        "Assigned load (veh/h)": {"column": "flow_vehicles"},
        "Volume / capacity": {
            "column": "volume_capacity_ratio",
            "breaks": [0.0, 0.50, 0.80, 1.00, 1.20, 2.50],
        },
        "Congested time (min)": {"column": "time_min"},
        "Delay (min)": {"column": "delay_min"},
        "Assigned speed (km/h)": {"column": "assigned_speed_kph"},
        "Directional capacity (veh/h)": {"column": "capacity_vph"},
    }
    return network_explorer(
        links,
        metrics,
        gates=corridor.gates if corridor is not None else None,
        polygon=corridor.polygon_gdf if corridor is not None else None,
        width_column="flow_vehicles",
        title="Corridor road assignment - link width represents assigned flow",
        simplify_m=1.0,
        height=650,
    )


def static_network_plot(
    context: TransportContext,
    mode_result: Any = None,
    corridor_municipalities: list[str] | None = None,
    project_name: str = "Corridor",
    assigned_edges: Any = None,
) -> Any:
    """Render a publication-ready static Matplotlib map of the assigned corridor network."""
    import matplotlib.pyplot as plt
    import geopandas as gpd
    
    # Extract edges to plot flexibly
    edges_plot = None
    gates_plot = None
    
    if assigned_edges is not None and isinstance(assigned_edges, gpd.GeoDataFrame):
        edges_plot = assigned_edges.copy()
        gates_plot = assigned_edges.attrs.get("metadata", {}).get("gates", None)
    elif mode_result is not None and isinstance(mode_result, gpd.GeoDataFrame):
        edges_plot = mode_result.copy()
        gates_plot = mode_result.attrs.get("metadata", {}).get("gates", None)
    elif mode_result is not None and hasattr(mode_result, "assigned_edges") and mode_result.assigned_edges is not None:
        edges_plot = mode_result.assigned_edges.copy()
        if hasattr(mode_result, "assigned_metadata") and mode_result.assigned_metadata is not None:
            gates_plot = mode_result.assigned_metadata.get("gates", None)
    else:
        edges_plot, gates_plot = _get_or_generate_corridor(context, corridor_municipalities)

    # Fallback to ensure gates are ALWAYS available for plotting
    if gates_plot is None or (isinstance(gates_plot, gpd.GeoDataFrame) and gates_plot.empty):
        _, gates_plot = _get_or_generate_corridor(context, corridor_municipalities)

    if edges_plot is None or len(edges_plot) == 0:
        print("No assigned edges available to plot.")
        return None

    munis = corridor_municipalities or []
    zones_plot = _get_core_zones(context.zones, munis) if munis else gpd.GeoDataFrame()

    road_colours = {
        "motorway": "#d73027", "motorway_link": "#fc8d59",
        "trunk": "#f46d43", "trunk_link": "#fdae61",
        "primary": "#fee08b", "primary_link": "#fff2a8",
        "secondary": "#91bfdb", "secondary_link": "#abd9e9",
        "tertiary": "#74add1", "tertiary_link": "#a6cee3",
        "residential": "#969696", "unclassified": "#bdbdbd",
        "living_street": "#d9d9d9",
    }

    fig, ax = plt.subplots(figsize=(14, 8))

    if not zones_plot.empty:
        zones_plot.plot(
            ax=ax, facecolor="#3182bd", edgecolor="#08519c",
            linewidth=1.8, alpha=0.12, zorder=1,
        )

    if "highway" in edges_plot.columns:
        hw_col = edges_plot["highway"].fillna("unclassified").astype(str)
    else:
        hw_col = pd.Series("unclassified", index=edges_plot.index)
    plot_df = edges_plot.assign(_hw_group=hw_col)

    for highway, group in plot_df.groupby("_hw_group"):
        highway = str(highway).lower()
        colour = road_colours.get(highway, "#bdbdbd")
        linewidth = 1.8 if highway in {"motorway", "trunk", "primary"} else 0.7
        group.plot(ax=ax, color=colour, linewidth=linewidth, alpha=0.85, label=highway, zorder=3)

    if gates_plot is not None and not gates_plot.empty:
        gates_plot.plot(
            ax=ax, color="#7a0177", edgecolor="white", linewidth=1.2,
            markersize=80, zorder=5, label="Cordon gates",
        )
        for gate in gates_plot.itertuples(index=False):
            ax.annotate(
                gate.gate_id, xy=(gate.geometry.x, gate.geometry.y),
                xytext=(4, 4), textcoords="offset points",
                fontsize=8.5, fontweight="bold", color="#4a004a", zorder=6,
            )

    ax.set_title(f"Detailed {project_name} Road Network & Cordon Gates", fontsize=14, fontweight="bold")
    ax.set_xlabel("Swiss projected coordinate — Easting (m)")
    ax.set_ylabel("Swiss projected coordinate — Northing (m)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)

    handles, labels = ax.get_legend_handles_labels()
    if labels:
        unique_legend = dict(zip(labels, handles))
        ax.legend(
            unique_legend.values(), unique_legend.keys(),
            title="Road class", loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9,
        )
    plt.tight_layout()
    plt.show()
    return None


# =============================================================================
# 7. ROUTE ASSIGNMENT
# =============================================================================

def run_route_assignment(
    context: TransportContext,
    mode_result: ModeChoiceResult,
    corridor_municipalities: list[str],
    *,
    demand_multiplier: float = 1.0,
    buffer_m: float = DEFAULT_CORRIDOR_BUFFER_M,
    max_gates: int = DEFAULT_MAX_GATES,
    iterations: int = ROUTE_ASSIGNMENT_DEFAULTS["max_iterations"],
    return_skims: bool = False,
    generate_lut: bool = False,
) -> Any:
    """Run a fast sub-network Frank-Wolfe road assignment on the corridor."""
    if not generate_lut and bool(ASSIGNMENT_SETTINGS.get("regenerate_lut", False)):
        generate_lut = True
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    
    zones = context.zones.copy()
    core_zones = _get_core_zones(zones, corridor_municipalities)
    if core_zones.empty:
        raise ValueError("No matching corridor municipalities found.")
        
    polygon = core_zones.geometry.union_all().buffer(float(buffer_m))
    network_edges = context.assignment_network["edges"].copy()
    network_nodes = context.assignment_network["nodes"].copy()
    
    # 1. Clip Network
    node_inside = network_nodes.geometry.intersects(polygon)
    inside_by_node = dict(zip(network_nodes["node_id"].astype(int), node_inside.astype(bool)))
    source_inside = network_edges["source"].map(inside_by_node).fillna(False).astype(bool)
    target_inside = network_edges["target"].map(inside_by_node).fillna(False).astype(bool)

    local_edges = network_edges.loc[source_inside & target_inside].copy()
    used_nodes = set(local_edges["source"].astype(int)) | set(local_edges["target"].astype(int))
    local_nodes = network_nodes.loc[network_nodes["node_id"].astype(int).isin(used_nodes)].copy()
    
    crossing = network_edges.loc[source_inside ^ target_inside].copy()
    crossing["inside_node"] = np.where(source_inside.loc[crossing.index], crossing["source"], crossing["target"]).astype(int)
    crossing["can_exit"] = source_inside.loc[crossing.index].to_numpy(dtype=bool)
    crossing["can_enter"] = target_inside.loc[crossing.index].to_numpy(dtype=bool)
    crossing = crossing.loc[crossing["inside_node"].isin(used_nodes)].copy()

    # 2. Build Gates
    node_geometry = local_nodes.set_index("node_id").geometry
    candidate_rows = []
    for node_id, group in crossing.groupby("inside_node", sort=False):
        capacity = pd.to_numeric(group["capacity_vph"], errors="coerce").max()
        candidate_rows.append({
            "node_id": int(node_id),
            "capacity_vph": float(capacity) if np.isfinite(capacity) else 0.0,
            "can_enter": bool(group["can_enter"].any()),
            "can_exit": bool(group["can_exit"].any()),
            "geometry": node_geometry.loc[int(node_id)],
        })
        
    candidates = gpd.GeoDataFrame(candidate_rows, geometry="geometry", crs=zones.crs)
    if candidates.empty:
        raise ValueError("No boundary crossing roads found.")
        
    candidates = candidates.sort_values("capacity_vph", ascending=False)
    
    try:
        import parameters as p
        max_gates = getattr(p, "MAX_GATES", max_gates)
    except Exception:
        pass

    selected_gates = []
    for _, row in candidates.iterrows():
        if len(selected_gates) >= max_gates:
            break
        point = row["geometry"]
        if not selected_gates:
            selected_gates.append(row)
            continue
            
        distances = [point.distance(s["geometry"]) for s in selected_gates]
        if min(distances) >= float(DEFAULT_GATE_SEPARATION_M):
            selected_gates.append(row)
            
    ordered = gpd.GeoDataFrame(selected_gates, crs=zones.crs).reset_index(drop=True)
    ordered["gate_id"] = [f"G{position:02d}" for position in range(1, len(ordered) + 1)]
    
    # 3. Setup Zone Mapping
    original_zone_node_map = {str(z): int(n) for z, n in context.assignment_network["zone_node_map"].items()}
    zone_centroids = gpd.GeoSeries(zones["centroid"], crs=zones.crs)
    internal_zone_ids = [
        str(z) for z in zones.loc[zone_centroids.intersects(polygon), "grid_id"]
        if str(z) in original_zone_node_map and original_zone_node_map[str(z)] in used_nodes
    ]
    
    zone_node_map = {z: original_zone_node_map[z] for z in internal_zone_ids}
    zone_node_map.update(dict(zip(ordered["gate_id"].astype(str), ordered["node_id"].astype(int))))

    # 4. Map External Demand to Gates
    external = zones.loc[~zones["grid_id"].astype(str).isin(internal_zone_ids)].copy()
    external_centroids = gpd.GeoSeries(external["centroid"], crs=zones.crs)
    ext_xy = np.column_stack((external_centroids.x, external_centroids.y))

    def map_gates(eligible_gates):
        if eligible_gates.empty: return {}
        gate_xy = np.column_stack((eligible_gates.geometry.x, eligible_gates.geometry.y))
        nearest = np.argmin(((ext_xy[:, None, 0] - gate_xy[None, :, 0])**2 + (ext_xy[:, None, 1] - gate_xy[None, :, 1])**2), axis=1)
        return dict(zip(external["grid_id"].astype(str), eligible_gates.iloc[nearest]["gate_id"].astype(str)))

    origin_map = {z: z for z in internal_zone_ids}
    origin_map.update(map_gates(ordered.loc[ordered["can_enter"].astype(bool)]))
    
    dest_map = {z: z for z in internal_zone_ids}
    dest_map.update(map_gates(ordered.loc[ordered["can_exit"].astype(bool)]))
    
    drive_od = mode_result.drive_od.copy()
    drive_od.index = drive_od.index.astype(str).map(lambda x: origin_map.get(x, None))
    drive_od.columns = drive_od.columns.astype(str).map(lambda x: dest_map.get(x, None))
    drive_od = drive_od.loc[drive_od.index.notna(), drive_od.columns.notna()]
    compressed_od = drive_od.groupby(level=0).sum().T.groupby(level=0).sum().T
    
    gate_zones = gpd.GeoDataFrame({"grid_id": ordered["gate_id"]}, geometry=ordered["geometry"], crs=zones.crs)
    sub_zones = pd.concat([zones.loc[zones["grid_id"].astype(str).isin(internal_zone_ids)], gate_zones], ignore_index=True)
    
    # 5. Transparent LUT Generation (Runs quietly in the background)
    if generate_lut:
        import json
        from pathlib import Path
   
        import concurrent.futures

        lut_scales = [0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
    
        delay_lut = {}
        try:
            import parameters as p
            p2a = getattr(p, "PEAK_TO_ANNUAL", 1200.0)
        except ImportError:
            p2a = 1200.0
        peak_factor = 365.0 / p2a
        
        def run_msa_for_scale(scale):
            scaled_od = compressed_od * scale * peak_factor
            _, _, meta = coarse_msa_assignment(
                edges=local_edges,
                nodes=local_nodes,
                zone_node_map=zone_node_map,
                demand=scaled_od,
                max_iterations=iterations,
                min_iterations=2,
            )
            return str(scale), {
                "delay_hours": meta.get("total_delay_hours", 0.0) / peak_factor,
                "vkt": meta.get("total_vkt", 0.0) / peak_factor,
                "vht": meta.get("total_vht", 0.0) / peak_factor,
                "avg_speed_kph": meta.get("avg_speed_kph", 0.0)
            }
            
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(run_msa_for_scale, lut_scales)
            for scale_str, res in results:
                delay_lut[scale_str] = res
            
        project_root = Path(__file__).resolve().parent.parent
        out_dir = project_root / "data" / "processed"
        out_dir.mkdir(parents=True, exist_ok=True)
        stage_idx = getattr(mode_result, "stage", 0)
        if isinstance(stage_idx, property) or not isinstance(stage_idx, (int, str)):
            stage_idx = mode_result.scenario.get("stage", 0) if hasattr(mode_result, "scenario") else 0
        lut_path = out_dir / f"delay_lut_stage_{stage_idx}.json"
        with open(lut_path, "w") as f:
            json.dump(delay_lut, f)
    
    # 6. Run Primary Assignment via Local Cordon MSA / BPR
    try:
        import parameters as p
        p2a = getattr(p, "PEAK_TO_ANNUAL", 1200.0)
    except ImportError:
        p2a = 1200.0
    peak_factor = 365.0 / p2a
    primary_od = compressed_od * demand_multiplier * peak_factor
    edges_gdf, history_df, metadata = coarse_msa_assignment(
        edges=local_edges,
        nodes=local_nodes,
        zone_node_map=zone_node_map,
        demand=primary_od,
        max_iterations=iterations,
        min_iterations=2,
    )
    metadata["total_delay_hours"] = metadata.get("total_delay_hours", 0.0) / peak_factor
    metadata["gates"] = ordered
    edges_gdf.attrs["metadata"] = metadata
    
    if return_skims:
        return edges_gdf, None, None, metadata
    return edges_gdf


# =============================================================================
# 8. LOCAL COARSE MSA / BPR ROAD ASSIGNMENT
# =============================================================================

def coarse_msa_assignment(
    edges: Any,
    zone_node_map: Mapping[str, int],
    demand: pd.DataFrame,
    *,
    nodes: Any | None = None,
    max_iterations: int = 8,
    min_iterations: int = 2,
    relative_gap_threshold: float = 0.10,
    od_threshold: float = 1.0,
) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    """Run a transparent local MSA/BPR assignment for notebook use.

    ``relative_gap_threshold`` is a relative L1 link-flow change, not a formal
    Wardrop equilibrium gap. ``nodes`` remains in the signature for backwards
    compatibility; endpoints are derived from the directed link table.
    """

    started = time.perf_counter()
    local_edges = edges.copy().reset_index(drop=True)
    required = {
        "source", "target", "free_flow_time_min", "capacity_vph", "length_m", "geometry"
    }
    missing = required.difference(local_edges.columns)
    if missing:
        raise ValueError(f"Local assignment links are missing: {sorted(missing)}")
    if int(max_iterations) < int(min_iterations):
        raise ValueError("max_iterations must be at least min_iterations")

    local_demand = demand.copy()
    local_demand.index = local_demand.index.astype(str)
    local_demand.columns = local_demand.columns.astype(str)
    if local_demand.shape[0] != local_demand.shape[1] or list(local_demand.index) != list(local_demand.columns):
        raise ValueError("Assignment demand must be a square, equally labelled OD matrix.")
    demand_values = local_demand.to_numpy(dtype=float)
    if not np.isfinite(demand_values).all() or (demand_values < 0).any():
        raise ValueError("Assignment demand must be finite and non-negative.")

    intrazonal_demand = float(np.trace(demand_values))
    interzonal_demand = max(float(demand_values.sum()) - intrazonal_demand, 0.0)
    retained_mask = demand_values > float(od_threshold)
    np.fill_diagonal(retained_mask, False)
    retained_demand = float(demand_values[retained_mask].sum())
    retained_pairs = int(retained_mask.sum())
    omitted_below_cutoff = max(interzonal_demand - retained_demand, 0.0)

    flows = np.zeros(len(local_edges), dtype=float)
    history: list[dict[str, float]] = []
    served = unserved = 0.0
    unserved_pairs = 0
    for iteration in range(1, int(max_iterations) + 1):
        current_travel_time = _bpr_link_times(local_edges, flows)
        aon_flows, served, unserved, unserved_pairs = _all_or_nothing_sparse(
            local_edges,
            zone_node_map,
            local_demand,
            current_travel_time,
            od_threshold=float(od_threshold),
        )
        step = 1.0 / float(iteration)
        updated = flows + step * (aon_flows - flows)
        rel_gap = float(np.abs(updated - flows).sum()) / max(
            float(np.abs(flows).sum()), 1e-9
        )
        flows = updated
        history.append(
            {
                "iteration": iteration,
                "msa_step": step,
                "relative_l1_flow_change": rel_gap,
                "served_vehicles": float(served),
            }
        )
        if iteration >= int(min_iterations) and rel_gap <= float(relative_gap_threshold):
            break

    capacity = pd.to_numeric(local_edges["capacity_vph"], errors="coerce").fillna(1.0).clip(lower=1.0).to_numpy()
    free_flow_time = pd.to_numeric(local_edges["free_flow_time_min"], errors="coerce").fillna(1.0).to_numpy()
    length_m = pd.to_numeric(local_edges["length_m"], errors="coerce").fillna(0.0).to_numpy()
    final_vc = np.maximum(flows / capacity, 0.0)
    final_travel_time = _bpr_link_times(local_edges, flows)
    final_speed = np.divide(
        length_m / 1000.0,
        final_travel_time / 60.0,
        out=np.zeros(len(local_edges), dtype=float),
        where=final_travel_time > 0,
    )
    assigned_edges = local_edges.copy()
    assigned_edges["flow_vehicles"] = flows
    assigned_edges["volume_capacity_ratio"] = final_vc
    assigned_edges["time_min"] = final_travel_time
    assigned_edges["assigned_speed_kph"] = np.clip(final_speed, 0.0, 130.0)
    assigned_edges["delay_min"] = np.maximum(final_travel_time - free_flow_time, 0.0)
    
    total_delay_hours = float((assigned_edges["flow_vehicles"] * (assigned_edges["delay_min"] / 60.0)).sum())
    total_vkt = float((assigned_edges["flow_vehicles"] * (length_m / 1000.0)).sum())
    total_vht = float((assigned_edges["flow_vehicles"] * (final_travel_time / 60.0)).sum())
    avg_speed_kph = total_vkt / total_vht if total_vht > 0 else 0.0
    
    history_df = pd.DataFrame(history)
    metadata = {
        "algorithm": "method of successive averages (MSA) with BPR link times",
        "runtime_s": time.perf_counter() - started,
        "iterations": len(history_df),
        "max_iterations": int(max_iterations),
        "min_iterations": int(min_iterations),
        "total_delay_hours": total_delay_hours,
        "total_vkt": total_vkt,
        "total_vht": total_vht,
        "avg_speed_kph": avg_speed_kph,
        "relative_l1_threshold": float(relative_gap_threshold),
        "final_relative_l1_change": float(history_df.iloc[-1]["relative_l1_flow_change"]) if len(history_df) else 0.0,
        "stopping_rule_met": bool(
            len(history_df) >= int(min_iterations)
            and len(history_df)
            and history_df.iloc[-1]["relative_l1_flow_change"] <= float(relative_gap_threshold)
        ),
        "od_threshold_veh_h": float(od_threshold),
        "requested_total_vehicles": float(demand_values.sum()),
        "intrazonal_not_assigned_vehicles": intrazonal_demand,
        "requested_interzonal_vehicles": interzonal_demand,
        "retained_above_cutoff_vehicles": retained_demand,
        "retained_od_pairs": retained_pairs,
        "omitted_below_cutoff_vehicles": omitted_below_cutoff,
        "served_vehicles": float(served),
        "unserved_vehicles": float(unserved),
        "unserved_od_pairs": int(unserved_pairs),
        "served_share_of_retained": float(served) / max(retained_demand, 1e-9),
        "retained_trips": retained_demand,
        "max_vc": float(final_vc.max()) if len(final_vc) > 0 else 0.0,
        "mean_vc": float(final_vc.mean()) if len(final_vc) > 0 else 0.0,
    }
    assigned_edges.attrs["metadata"] = metadata
    return assigned_edges, history_df, metadata


def run_corridor_assignment(
    context: TransportContext,
    corridor: CorridorContext,
    mode_result: ModeChoiceResult,
    *,
    passthrough_fraction: float = DEFAULT_PASSTHROUGH_FRACTION,
    drive_occupancy: float = ASSIGNMENT_SETTINGS["drive_occupancy"],
    max_iterations: int = ASSIGNMENT_SETTINGS["max_iterations"],
    min_iterations: int = ASSIGNMENT_SETTINGS["min_iterations"],
    relative_gap_threshold: float = ASSIGNMENT_SETTINGS["relative_gap_threshold"],
    od_threshold: float = ASSIGNMENT_SETTINGS["od_threshold"],
) -> AssignmentResult:
    """Collapse full-canton car demand and assign it to one local corridor.

    Passenger car trips are converted to vehicles using ``drive_occupancy``;
    prepared background road demand is already expressed as vehicles.  Both
    matrices use exactly the same internal-zone/gate representation.
    """

    passenger, passenger_breakdown = collapse_od_to_gates(
        mode_result.drive_od,
        context,
        corridor,
        passthrough_fraction=passthrough_fraction,
    )
    background, background_breakdown = collapse_od_to_gates(
        context.road_background_od,
        context,
        corridor,
        passthrough_fraction=passthrough_fraction,
    )
    occupancy = max(float(drive_occupancy), 1e-9)
    vehicle_demand = passenger / occupancy + background

    breakdown = passenger_breakdown.rename(
        columns={"trips": "passenger_car_trips"}
    ).copy()
    breakdown["passenger_vehicles"] = breakdown["passenger_car_trips"] / occupancy
    breakdown["background_vehicles"] = background_breakdown["trips"].to_numpy()
    breakdown["assignment_vehicles"] = (
        breakdown["passenger_vehicles"] + breakdown["background_vehicles"]
    )

    links, history, diagnostics = coarse_msa_assignment(
        edges=corridor.edges,
        nodes=corridor.nodes,
        zone_node_map=corridor.zone_node_map,
        demand=vehicle_demand,
        max_iterations=int(max_iterations),
        min_iterations=int(min_iterations),
        relative_gap_threshold=float(relative_gap_threshold),
        od_threshold=float(od_threshold),
    )
    diagnostics.update(
        {
            "stage": int(mode_result.stage),
            "modal_feedback": False,
            "drive_occupancy": float(drive_occupancy),
            "passthrough_fraction": float(passthrough_fraction),
            "od_threshold_veh_h": float(od_threshold),
            "requested_total_vehicles": float(vehicle_demand.to_numpy().sum()),
            "mode_choice_cache_key": list(mode_result.cache_key),
        }
    )
    links.attrs["metadata"] = diagnostics
    return AssignmentResult(
        links=links,
        history=history,
        diagnostics=diagnostics,
        demand_matrix=vehicle_demand,
        demand_breakdown=breakdown,
        gate_totals=gate_totals(vehicle_demand, corridor),
    )


def _bpr_link_times(edges: pd.DataFrame, flow: np.ndarray) -> np.ndarray:
    """Return BPR travel time for every directed link at ``flow``."""

    free_flow = pd.to_numeric(
        edges["free_flow_time_min"], errors="coerce"
    ).fillna(1.0).to_numpy(dtype=float)
    capacity = pd.to_numeric(
        edges["capacity_vph"], errors="coerce"
    ).fillna(1.0).clip(lower=1.0).to_numpy(dtype=float)
    alpha = pd.to_numeric(
        edges.get("alpha", pd.Series(0.15, index=edges.index)), errors="coerce"
    ).fillna(0.15).to_numpy(dtype=float)
    beta = pd.to_numeric(
        edges.get("beta", pd.Series(4.0, index=edges.index)), errors="coerce"
    ).fillna(4.0).to_numpy(dtype=float)
    ratio = np.maximum(np.asarray(flow, dtype=float) / capacity, 0.0)
    return np.clip(free_flow * (1.0 + alpha * ratio ** beta), 1e-4, 500.0)


def _sparse_shortest_path_inputs(
    edges: pd.DataFrame,
    edge_times: np.ndarray,
) -> tuple[Any, np.ndarray, dict[int, int], dict[tuple[int, int], int]]:
    """Build a sparse graph while retaining the fastest parallel link."""

    from scipy.sparse import csr_matrix

    sources = pd.to_numeric(edges["source"], errors="raise").to_numpy(dtype=int)
    targets = pd.to_numeric(edges["target"], errors="raise").to_numpy(dtype=int)
    unique_nodes = np.unique(np.concatenate([sources, targets]))
    node_to_position = {int(node): i for i, node in enumerate(unique_nodes)}
    source_positions = np.asarray([node_to_position[int(node)] for node in sources])
    target_positions = np.asarray([node_to_position[int(node)] for node in targets])

    # scipy sums duplicate sparse entries.  Road networks can contain parallel
    # links, so select the currently fastest one before creating the matrix.
    edge_lookup: dict[tuple[int, int], int] = {}
    for edge_position, (source, target) in enumerate(
        zip(source_positions, target_positions)
    ):
        pair = (int(source), int(target))
        previous = edge_lookup.get(pair)
        if previous is None or edge_times[edge_position] < edge_times[previous]:
            edge_lookup[pair] = edge_position
    selected_positions = np.fromiter(edge_lookup.values(), dtype=int)
    graph = csr_matrix(
        (
            np.asarray(edge_times, dtype=float)[selected_positions],
            (
                source_positions[selected_positions],
                target_positions[selected_positions],
            ),
        ),
        shape=(len(unique_nodes), len(unique_nodes)),
    )
    return graph, unique_nodes, node_to_position, edge_lookup


def _all_or_nothing_sparse(
    edges: pd.DataFrame,
    zone_node_map: Mapping[str, int],
    demand: pd.DataFrame,
    edge_times: np.ndarray,
    *,
    od_threshold: float,
) -> tuple[np.ndarray, float, float, int]:
    """Load retained OD demand on current shortest paths."""

    from scipy.sparse.csgraph import shortest_path

    graph, _, node_to_position, edge_lookup = _sparse_shortest_path_inputs(
        edges, edge_times
    )
    labels = demand.index.astype(str).to_numpy()
    values = demand.to_numpy(dtype=float)
    retained = np.isfinite(values) & (values > float(od_threshold))
    np.fill_diagonal(retained, False)
    origins, destinations = np.nonzero(retained)

    origin_positions = np.asarray(
        [node_to_position.get(int(zone_node_map.get(labels[i], -1)), -1) for i in origins],
        dtype=int,
    )
    destination_positions = np.asarray(
        [node_to_position.get(int(zone_node_map.get(labels[j], -1)), -1) for j in destinations],
        dtype=int,
    )
    trips = values[origins, destinations]
    valid = (origin_positions >= 0) & (destination_positions >= 0)
    unserved = float(trips[~valid].sum())
    unserved_pairs = int((~valid).sum())
    same_node = valid & (origin_positions == destination_positions)
    # Different zone labels can share one routing connector.  Their demand is
    # served locally without loading any physical road link.
    served = float(trips[same_node].sum())
    routed = valid & ~same_node
    origin_positions = origin_positions[routed]
    destination_positions = destination_positions[routed]
    trips = trips[routed]

    flow = np.zeros(len(edges), dtype=float)
    unique_origins = np.unique(origin_positions)
    if len(unique_origins):
        _, predecessors = shortest_path(
            graph,
            directed=True,
            indices=unique_origins,
            return_predecessors=True,
        )
        origin_to_row = {int(origin): row for row, origin in enumerate(unique_origins)}
        for origin, destination, volume in zip(
            origin_positions, destination_positions, trips
        ):
            current = int(destination)
            path_edges: list[int] = []
            while current != int(origin):
                predecessor = int(predecessors[origin_to_row[int(origin)], current])
                if predecessor < 0:
                    path_edges = []
                    break
                edge_position = edge_lookup.get((predecessor, current))
                if edge_position is None:
                    path_edges = []
                    break
                path_edges.append(edge_position)
                current = predecessor
            if not path_edges:
                unserved += float(volume)
                unserved_pairs += 1
                continue
            flow[np.asarray(path_edges, dtype=int)] += float(volume)
            served += float(volume)
    return flow, served, unserved, unserved_pairs


def _corridor_shortest_time_table(
    edges: pd.DataFrame,
    zone_node_map: Mapping[str, int],
    edge_times: np.ndarray,
) -> pd.DataFrame:
    """Shortest-path minutes between every internal-zone/gate routing node."""

    from scipy.sparse.csgraph import shortest_path

    graph, _, node_to_position, _ = _sparse_shortest_path_inputs(edges, edge_times)
    labels = [str(label) for label in zone_node_map]
    valid_labels = [
        label for label in labels if int(zone_node_map[label]) in node_to_position
    ]
    node_positions = np.asarray(
        [node_to_position[int(zone_node_map[label])] for label in valid_labels],
        dtype=int,
    )
    distances = shortest_path(
        graph, directed=True, indices=node_positions, return_predecessors=False
    )
    distances = distances[:, node_positions]
    return pd.DataFrame(distances, index=valid_labels, columns=valid_labels)


def _project_corridor_delay_to_full_od(
    context: TransportContext,
    corridor: CorridorContext,
    delay_table: pd.DataFrame,
    labels: pd.Index,
) -> pd.DataFrame:
    """Map node-specific corridor delay back to each full-canton OD pair."""

    internal = set(corridor.zone_ids)
    origin_labels = [
        label if label in internal else corridor.external_zone_to_entry_gate.get(label)
        for label in labels.astype(str)
    ]
    destination_labels = [
        label if label in internal else corridor.external_zone_to_exit_gate.get(label)
        for label in labels.astype(str)
    ]
    lookup = {label: i for i, label in enumerate(delay_table.index.astype(str))}
    origin_positions = np.asarray([lookup.get(label, -1) for label in origin_labels])
    destination_positions = np.asarray([lookup.get(label, -1) for label in destination_labels])
    valid = (origin_positions[:, None] >= 0) & (destination_positions[None, :] >= 0)
    projected = np.zeros((len(labels), len(labels)), dtype=float)
    safe_origins = np.maximum(origin_positions, 0)
    safe_destinations = np.maximum(destination_positions, 0)
    candidates = delay_table.to_numpy(dtype=float)[
        safe_origins[:, None], safe_destinations[None, :]
    ]
    projected[valid] = candidates[valid]
    projected[~np.isfinite(projected)] = 0.0
    projected = np.maximum(projected, 0.0)
    return pd.DataFrame(projected, index=labels, columns=labels)


def run_coupled_corridor_assignment(
    context: TransportContext,
    corridor: CorridorContext,
    mode_result: ModeChoiceResult,
    *,
    demand_multiplier: float = 1.0,
    passthrough_fraction: float = DEFAULT_PASSTHROUGH_FRACTION,
    drive_occupancy: float = ASSIGNMENT_SETTINGS["drive_occupancy"],
    max_iterations: int = ASSIGNMENT_SETTINGS["max_iterations"],
    min_iterations: int = ASSIGNMENT_SETTINGS["min_iterations"],
    relative_gap_threshold: float = ASSIGNMENT_SETTINGS["relative_gap_threshold"],
    od_threshold: float = ASSIGNMENT_SETTINGS["od_threshold"],
) -> AssignmentResult:
    """Solve link flows and five-mode demand together in one MSA loop.

    Each iteration obtains OD-specific corridor delays from the same road graph
    used by assignment.  Those delays update the car skim, the multinomial
    logit model updates car demand, and both flows and demand are averaged with
    the classical ``1 / iteration`` MSA step.  Only delay inside the local
    cordon is fed back; the remainder of every whole-trip skim stays fixed.
    """

    started = time.perf_counter()
    edges = corridor.edges.copy().reset_index(drop=True)
    flow = np.zeros(len(edges), dtype=float)
    occupancy = max(float(drive_occupancy), 1e-9)
    scale = max(float(demand_multiplier), 0.0)

    # Summing the five fixed-skims outputs exactly recovers the scenario demand
    # that produced ``mode_result`` without repeating trip generation/Furness.
    total_od = sum(
        (frame.copy() for frame in mode_result.od_by_mode.values()),
        start=pd.DataFrame(
            0.0,
            index=mode_result.drive_od.index,
            columns=mode_result.drive_od.columns,
        ),
    ) * scale
    labels = total_od.index.astype(str)
    total_od.index = labels
    total_od.columns = labels

    source_times = mode_result.uncongested_travel_times or mode_result.travel_times
    base_times = {name: frame.copy() for name, frame in source_times.items()}
    lengths = {name: frame.copy() for name, frame in mode_result.lengths.items()}
    drive_key = "drive" if "drive" in base_times else "car"
    if drive_key not in base_times:
        raise KeyError("The mode-choice result has no drive/car travel-time skim.")
    base_drive_time = base_times[drive_key].reindex(index=labels, columns=labels)

    mode_module = context.modules["mode_choice_zurich"]
    travel_time_module = context.modules["travel_times"]
    parameters = mode_module.load_mode_choice_parameters()
    for key in list(parameters):
        if key in mode_result.scenario:
            parameters[key] = mode_result.scenario[key]
    walk_mask = travel_time_module.standalone_walk_mask(lengths, context.zones)

    # Prepared background demand is fixed with respect to mode choice, but it
    # uses the same gate mapping and optional scenario demand multiplier.
    background, background_breakdown = collapse_od_to_gates(
        context.road_background_od * scale,
        context,
        corridor,
        passthrough_fraction=passthrough_fraction,
    )
    initial_passenger, _ = collapse_od_to_gates(
        mode_result.drive_od * scale,
        context,
        corridor,
        passthrough_fraction=passthrough_fraction,
    )
    vehicle_demand = initial_passenger / occupancy + background

    free_flow_times = _bpr_link_times(edges, np.zeros(len(edges), dtype=float))
    free_flow_table = _corridor_shortest_time_table(
        edges, corridor.zone_node_map, free_flow_times
    )
    history_rows: list[dict[str, float]] = []
    latest_od_by_mode = {
        name: frame * scale for name, frame in mode_result.od_by_mode.items()
    }
    latest_drive_time = base_drive_time.copy()
    served = unserved = 0.0
    unserved_pairs = 0

    for iteration in range(1, int(max_iterations) + 1):
        edge_times = _bpr_link_times(edges, flow)
        auxiliary, served, unserved, unserved_pairs = _all_or_nothing_sparse(
            edges,
            corridor.zone_node_map,
            vehicle_demand,
            edge_times,
            od_threshold=float(od_threshold),
        )
        step = 1.0 / float(iteration)
        updated_flow = flow + step * (auxiliary - flow)
        flow_change = float(np.abs(updated_flow - flow).sum()) / max(
            float(np.abs(flow).sum()), 1e-9
        )

        current_table = _corridor_shortest_time_table(
            edges, corridor.zone_node_map, edge_times
        )
        node_delay = (current_table - free_flow_table).clip(lower=0.0)
        full_delay = _project_corridor_delay_to_full_od(
            context, corridor, node_delay, labels
        )
        latest_drive_time = base_drive_time + full_delay
        updated_times = {name: frame.copy() for name, frame in base_times.items()}
        updated_times[drive_key] = latest_drive_time

        latest_od_by_mode, _ = mode_module.mode_split_aggregated(
            updated_times,
            lengths,
            total_od,
            walk_allowed_mask=walk_mask,
            parameters=parameters,
            car_affinity=float(mode_result.scenario.get("car_affinity", 1.0)),
            walk_affinity=float(mode_result.scenario.get("walk_affinity", 1.0)),
            bike_affinity=float(mode_result.scenario.get("bike_affinity", 1.0)),
            pt_affinity=float(mode_result.scenario.get("pt_affinity", 1.0)),
        )
        passenger_target, _ = collapse_od_to_gates(
            latest_od_by_mode["drive"],
            context,
            corridor,
            passthrough_fraction=passthrough_fraction,
        )
        target_vehicle_demand = passenger_target / occupancy + background
        updated_demand = vehicle_demand + step * (target_vehicle_demand - vehicle_demand)
        demand_change = float(
            np.abs(updated_demand.to_numpy() - vehicle_demand.to_numpy()).sum()
        ) / max(float(np.abs(vehicle_demand.to_numpy()).sum()), 1e-9)

        mode_summary = _corridor_mode_summary(latest_od_by_mode, corridor.zone_ids)
        shares = mode_summary.set_index("mode")["share"].to_dict()
        trip_weights = total_od.to_numpy(dtype=float)
        mean_delay = float(
            (full_delay.to_numpy(dtype=float) * trip_weights).sum()
            / max(trip_weights.sum(), 1e-9)
        )
        history_rows.append(
            {
                "iteration": iteration,
                "msa_step": step,
                "relative_l1_flow_change": flow_change,
                "relative_l1_demand_change": demand_change,
                "car_share": shares.get("Car (Driver)", np.nan),
                "pt_share": shares.get("Public Transport", np.nan),
                "bike_share": shares.get("Bicycle", np.nan),
                "walk_share": shares.get("Walking", np.nan),
                "demand_weighted_corridor_delay_min": mean_delay,
                "served_vehicles": served,
            }
        )
        flow, vehicle_demand = updated_flow, updated_demand
        if (
            iteration >= int(min_iterations)
            and max(flow_change, demand_change) <= float(relative_gap_threshold)
        ):
            break

    final_times = _bpr_link_times(edges, flow)
    # Re-evaluate the skim and modal split once at the final averaged flow so
    # the returned behavioural result corresponds to the displayed link times.
    final_table = _corridor_shortest_time_table(
        edges, corridor.zone_node_map, final_times
    )
    final_delay = _project_corridor_delay_to_full_od(
        context,
        corridor,
        (final_table - free_flow_table).clip(lower=0.0),
        labels,
    )
    latest_drive_time = base_drive_time + final_delay
    final_mode_times = {name: frame.copy() for name, frame in base_times.items()}
    final_mode_times[drive_key] = latest_drive_time
    latest_od_by_mode, _ = mode_module.mode_split_aggregated(
        final_mode_times,
        lengths,
        total_od,
        walk_allowed_mask=walk_mask,
        parameters=parameters,
        car_affinity=float(mode_result.scenario.get("car_affinity", 1.0)),
        walk_affinity=float(mode_result.scenario.get("walk_affinity", 1.0)),
        bike_affinity=float(mode_result.scenario.get("bike_affinity", 1.0)),
        pt_affinity=float(mode_result.scenario.get("pt_affinity", 1.0)),
    )
    capacity = pd.to_numeric(edges["capacity_vph"], errors="coerce").clip(lower=1.0)
    free_flow = pd.to_numeric(edges["free_flow_time_min"], errors="coerce")
    length_km = pd.to_numeric(edges["length_m"], errors="coerce").fillna(0.0) / 1000.0
    links = edges.copy()
    links["flow_vehicles"] = flow
    links["time_min"] = final_times
    links["volume_capacity_ratio"] = flow / capacity.to_numpy(dtype=float)
    links["delay_min"] = np.maximum(final_times - free_flow.to_numpy(dtype=float), 0.0)
    links["assigned_speed_kph"] = np.divide(
        length_km.to_numpy(dtype=float),
        final_times / 60.0,
        out=np.zeros(len(links), dtype=float),
        where=final_times > 0,
    )
    history = pd.DataFrame(history_rows)
    final_flow_change = float(history.iloc[-1]["relative_l1_flow_change"])
    final_demand_change = float(history.iloc[-1]["relative_l1_demand_change"])
    total_delay_hours = float((flow * links["delay_min"].to_numpy() / 60.0).sum())
    total_vkt = float((flow * length_km.to_numpy()).sum())
    total_vht = float((flow * final_times / 60.0).sum())

    # Report the actual MSA-averaged demand used by the final flow solution,
    # rather than only the last unaveraged modal-choice target.
    passenger_vehicle_matrix = (vehicle_demand - background).clip(lower=0.0)
    internal = corridor.zone_ids
    gates = corridor.gates["gate_id"].astype(str).tolist()
    passenger_categories = [
        float(passenger_vehicle_matrix.loc[internal, internal].to_numpy().sum()),
        float(passenger_vehicle_matrix.loc[gates, internal].to_numpy().sum()),
        float(passenger_vehicle_matrix.loc[internal, gates].to_numpy().sum()),
        float(passenger_vehicle_matrix.loc[gates, gates].to_numpy().sum()),
    ]
    breakdown = pd.DataFrame(
        {
            "category": background_breakdown["category"],
            "passenger_car_trips": np.asarray(passenger_categories) * occupancy,
            "passenger_vehicles": passenger_categories,
            "background_vehicles": background_breakdown["trips"].to_numpy(),
        }
    )
    breakdown["assignment_vehicles"] = (
        breakdown["passenger_vehicles"] + breakdown["background_vehicles"]
    )
    updated_times = {name: frame.copy() for name, frame in base_times.items()}
    updated_times[drive_key] = latest_drive_time
    coupled_mode_result = ModeChoiceResult(
        drive_od=latest_od_by_mode["drive"],
        od_by_mode=latest_od_by_mode,
        summary=_corridor_mode_summary(latest_od_by_mode, corridor.zone_ids),
        scenario=dict(mode_result.scenario),
        travel_times=updated_times,
        lengths=lengths,
        cache_key=tuple(mode_result.cache_key) + ("coupled_msa",),
        assigned_edges=links,
        uncongested_travel_times=base_times,
    )
    diagnostics = {
        "algorithm": "coupled MSA/BPR assignment and five-mode logit",
        "modal_feedback": True,
        "runtime_s": time.perf_counter() - started,
        "iterations": len(history),
        "max_iterations": int(max_iterations),
        "min_iterations": int(min_iterations),
        "relative_l1_threshold": float(relative_gap_threshold),
        "final_relative_l1_flow_change": final_flow_change,
        "final_relative_l1_demand_change": final_demand_change,
        "stopping_rule_met": bool(
            len(history) >= int(min_iterations)
            and max(final_flow_change, final_demand_change)
            <= float(relative_gap_threshold)
        ),
        "od_threshold_veh_h": float(od_threshold),
        "passthrough_fraction": float(passthrough_fraction),
        "drive_occupancy": float(drive_occupancy),
        "served_vehicles": float(served),
        "unserved_vehicles": float(unserved),
        "unserved_od_pairs": int(unserved_pairs),
        "total_delay_hours": total_delay_hours,
        "total_vkt": total_vkt,
        "total_vht": total_vht,
        "avg_speed_kph": total_vkt / max(total_vht, 1e-9),
        "corridor_delay_feedback": (
            "OD-specific local-network delay added to the fixed whole-trip drive skim"
        ),
    }
    links.attrs["metadata"] = diagnostics
    return AssignmentResult(
        links=links,
        history=history,
        diagnostics=diagnostics,
        demand_matrix=vehicle_demand,
        demand_breakdown=breakdown,
        gate_totals=gate_totals(vehicle_demand, corridor),
        mode_result=coupled_mode_result,
    )


def assignment_dashboard(
    context: TransportContext,
    corridor: CorridorContext,
    mode_state: dict[str, Any],
    *,
    modal_feedback: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Explicit-button MSA dashboard with cached settings combinations.

    ``modal_feedback=False`` holds the car OD matrix fixed, which isolates the
    assignment algorithm.  ``modal_feedback=True`` recomputes modal split from
    the OD-specific congested drive skim within the same MSA iteration.
    """

    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    from IPython.display import clear_output, display

    state: dict[str, Any] = {"result": None, "cache": {}, "max_cached_runs": 3}
    run_button = widgets.Button(
        description=(
            "Run coupled MSA + mode choice"
            if modal_feedback
            else "Run corridor MSA assignment"
        ),
        button_style="primary",
        icon="play",
    )
    flow_change = widgets.Dropdown(
        options=[
            ("0.05 (stricter)", 0.05),
            ("0.10 (default)", 0.10),
            ("0.20 (coarser)", 0.20),
        ],
        value=float(ASSIGNMENT_SETTINGS["relative_gap_threshold"]),
        description="Flow change:",
        style={"description_width": "95px"},
    )
    max_iterations = widgets.IntSlider(
        value=int(ASSIGNMENT_SETTINGS["max_iterations"]),
        min=2,
        max=20,
        step=1,
        description="Max iter.:",
        continuous_update=False,
    )
    od_threshold = widgets.Dropdown(
        options=[("0.1 veh/h", 0.1), ("1.0 veh/h (default)", 1.0), ("5.0 veh/h", 5.0)],
        value=float(ASSIGNMENT_SETTINGS["od_threshold"]),
        description="OD cutoff:",
    )
    passthrough = widgets.FloatSlider(
        value=DEFAULT_PASSTHROUGH_FRACTION,
        min=0.0,
        max=0.25,
        step=0.01,
        description="Pass-through:",
        continuous_update=False,
        readout_format=".0%",
    )
    output = widgets.Output()

    def run(_: Any) -> None:
        selected_mode_result = mode_state.get("result")
        with output:
            clear_output(wait=True)
            if selected_mode_result is None:
                print("Run the mode-choice dashboard once before road assignment.")
                return
            key = (
                selected_mode_result.cache_key,
                bool(modal_feedback),
                float(flow_change.value),
                int(max_iterations.value),
                float(od_threshold.value),
                float(passthrough.value),
            )
            print("Running MSA on the reduced corridor network...")
        run_button.disabled = True
        try:
            if key not in state["cache"]:
                runner = (
                    run_coupled_corridor_assignment
                    if modal_feedback
                    else run_corridor_assignment
                )
                state["cache"][key] = runner(
                    context,
                    corridor,
                    selected_mode_result,
                    passthrough_fraction=float(passthrough.value),
                    max_iterations=int(max_iterations.value),
                    min_iterations=int(ASSIGNMENT_SETTINGS["min_iterations"]),
                    relative_gap_threshold=float(flow_change.value),
                    od_threshold=float(od_threshold.value),
                )
                while len(state["cache"]) > int(state["max_cached_runs"]):
                    state["cache"].pop(next(iter(state["cache"])))
            result = state["cache"][key]
            state["result"] = result
            with output:
                clear_output(wait=True)
                diagnostic_table = pd.DataFrame(
                    {
                        "diagnostic": list(result.diagnostics),
                        "value": [
                            value if not isinstance(value, (dict, list)) else str(value)
                            for value in result.diagnostics.values()
                        ],
                    }
                )
                display(diagnostic_table)
                display(result.demand_breakdown)
                display(result.gate_totals)
                if result.mode_result is not None:
                    display(result.mode_result.summary.style.format({"trips": "{:,.1f}", "share": "{:.1%}"}))
                if len(result.history):
                    fig, ax = plt.subplots(figsize=(7.2, 3.2))
                    ax.plot(
                        result.history["iteration"],
                        result.history["relative_l1_flow_change"],
                        marker="o",
                        label="link-flow change",
                    )
                    if "relative_l1_demand_change" in result.history:
                        ax.plot(
                            result.history["iteration"],
                            result.history["relative_l1_demand_change"],
                            marker="s",
                            label="car-demand change",
                        )
                    ax.axhline(float(flow_change.value), color="crimson", linestyle="--", label="threshold")
                    ax.set_yscale("log")
                    ax.set_xlabel("MSA iteration")
                    ax.set_ylabel("Relative L1 change")
                    ax.set_title("Practical convergence diagnostic (not a formal Wardrop gap)")
                    ax.legend()
                    plt.show()
                    if result.mode_result is not None:
                        fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.2))
                        for column, label in (
                            ("car_share", "Car"),
                            ("pt_share", "Public transport"),
                            ("bike_share", "Bicycle"),
                            ("walk_share", "Walking"),
                        ):
                            axes[0].plot(
                                result.history["iteration"],
                                100.0 * result.history[column],
                                marker="o",
                                label=label,
                            )
                        axes[0].set_xlabel("MSA iteration")
                        axes[0].set_ylabel("Corridor trip share (%)")
                        axes[0].set_title("Modal response to congestion")
                        axes[0].legend(fontsize=8)
                        axes[1].plot(
                            result.history["iteration"],
                            result.history["demand_weighted_corridor_delay_min"],
                            color="crimson",
                            marker="o",
                        )
                        axes[1].set_xlabel("MSA iteration")
                        axes[1].set_ylabel("Minutes per trip")
                        axes[1].set_title("Demand-weighted local road delay")
                        fig.tight_layout()
                        plt.show()
                display(assignment_explorer(result, corridor))
        except Exception as error:
            with output:
                clear_output(wait=True)
                print(f"Assignment failed: {type(error).__name__}: {error}")
        finally:
            run_button.disabled = False

    run_button.on_click(run)
    controls = widgets.VBox(
        [flow_change, max_iterations, od_threshold, passthrough, run_button]
    )
    note = widgets.HTML(
        (
            "<b>Coupled calculation:</b> congestion updates OD-specific car "
            "travel times and modal split during MSA."
            if modal_feedback
            else "<b>Fixed-demand diagnostic:</b> MSA updates routes and link "
            "times while the selected car OD matrix remains fixed."
        )
    )
    return widgets.VBox([note, controls, output]), state


def assignment_convergence_test(
    context: TransportContext,
    corridor: CorridorContext,
    mode_result: ModeChoiceResult,
    *,
    n_runs: int = 30,
    max_iterations: int = 8,
    thresholds: tuple[float, ...] = (0.10, 0.05, 0.02),
    od_threshold: float = ASSIGNMENT_SETTINGS["od_threshold"],
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Monte Carlo diagnostic for the practical MSA stopping rule.

    The test perturbs total car demand, individual retained OD cells, and the
    uncertain pass-through share.  It always runs the requested number of
    iterations; thresholds are evaluated afterwards so every run is directly
    comparable.  This is a numerical stability diagnostic, not uncertainty
    analysis for the infrastructure project.
    """

    if int(n_runs) < 1 or int(max_iterations) < 2:
        raise ValueError("n_runs must be positive and max_iterations at least 2.")
    rng = np.random.default_rng(int(seed))
    rows: list[pd.DataFrame] = []
    occupancy = max(float(ASSIGNMENT_SETTINGS["drive_occupancy"]), 1e-9)

    for run in range(int(n_runs)):
        demand_multiplier = float(rng.uniform(0.80, 1.25))
        passthrough = float(rng.uniform(0.02, 0.12))
        passenger, _ = collapse_od_to_gates(
            mode_result.drive_od,
            context,
            corridor,
            passthrough_fraction=passthrough,
        )
        noise = rng.lognormal(mean=0.0, sigma=0.12, size=passenger.shape)
        passenger = passenger * demand_multiplier * noise
        background, _ = collapse_od_to_gates(
            context.road_background_od * demand_multiplier,
            context,
            corridor,
            passthrough_fraction=passthrough,
        )
        demand = passenger / occupancy + background
        _, history, _ = coarse_msa_assignment(
            edges=corridor.edges,
            nodes=corridor.nodes,
            zone_node_map=corridor.zone_node_map,
            demand=demand,
            max_iterations=int(max_iterations),
            min_iterations=int(max_iterations),
            relative_gap_threshold=0.0,
            od_threshold=float(od_threshold),
        )
        history = history.copy()
        history["run"] = run + 1
        history["demand_multiplier"] = demand_multiplier
        history["passthrough_fraction"] = passthrough
        rows.append(history)

    runs = pd.concat(rows, ignore_index=True)
    summary = (
        runs.groupby("iteration")["relative_l1_flow_change"]
        .agg(
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
        )
        .reset_index()
    )
    reached_rows = []
    for threshold in thresholds:
        for run, group in runs.groupby("run"):
            reached = group.loc[
                group["relative_l1_flow_change"] <= float(threshold), "iteration"
            ]
            reached_rows.append(
                {
                    "run": int(run),
                    "threshold": float(threshold),
                    "first_iteration": int(reached.iloc[0]) if len(reached) else np.nan,
                    "reached": bool(len(reached)),
                }
            )
    threshold_reach = pd.DataFrame(reached_rows)
    return {"runs": runs, "summary": summary, "threshold_reach": threshold_reach}


def plot_assignment_convergence_test(
    test_result: Mapping[str, pd.DataFrame],
    *,
    thresholds: tuple[float, ...] = (0.10, 0.05, 0.02),
) -> Any:
    """Plot median/IQR MSA convergence and threshold-reach frequencies."""

    import matplotlib.pyplot as plt

    summary = test_result["summary"]
    reached = test_result["threshold_reach"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    axes[0].plot(summary["iteration"], summary["median"], marker="o", label="median")
    axes[0].fill_between(
        summary["iteration"].to_numpy(dtype=float),
        summary["q25"].to_numpy(dtype=float),
        summary["q75"].to_numpy(dtype=float),
        alpha=0.25,
        label="interquartile range",
    )
    for threshold in thresholds:
        axes[0].axhline(float(threshold), linestyle="--", linewidth=1, label=f"{threshold:.0%}")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("MSA iteration")
    axes[0].set_ylabel("Relative L1 flow change")
    axes[0].set_title("Convergence across perturbed demand runs")
    axes[0].legend(ncol=2, fontsize=8)

    rates = reached.groupby("threshold")["reached"].mean().reindex(thresholds)
    axes[1].bar([f"{value:.0%}" for value in rates.index], rates.values * 100.0)
    axes[1].set_ylim(0, 100)
    axes[1].set_xlabel("Flow-change threshold")
    axes[1].set_ylabel("Runs reaching threshold (%)")
    axes[1].set_title("Threshold reached within iteration limit")
    fig.tight_layout()
    return fig


# =============================================================================
# 8. DETAILED OPENSTREETMAP NETWORK ENHANCER
# =============================================================================

def load_detailed_corridor_network(
    context: TransportContext,
    corridor_municipalities: list[str],
    *,
    cache_path: str | Path | None = None,
    buffer_m: float = 800.0,
    allow_download: bool = True,
    force_regenerate: bool | None = None,
) -> TransportContext:
    """
    Enhance the macroscopic road assignment network with high-detail OpenStreetMap roads
    for any specified project corridor (e.g. MehrSpur, Limmattal, or Forch).
    
    If cache_path exists and force_regenerate is False, it loads instantly from disk.
    Otherwise, if allow_download is True, it queries OSM via osmnx, converts to traffic
    assignment schema, stitches zone centroids, and caches the result.
    """
    import pickle
    import geopandas as gpd
    import numpy as np
    from scipy.spatial import cKDTree
    
    if force_regenerate is None:
        force_regenerate = ASSIGNMENT_SETTINGS.get(
            "force_regenerate_corridor_network"
        )
    
    if not force_regenerate and cache_path is not None and Path(cache_path).exists():
        with open(cache_path, "rb") as f:
            detailed_network = pickle.load(f)
        print(f"✓ Loaded cached detailed corridor network from {Path(cache_path).name}")
        return TransportContext(
            project_root=context.project_root,
            model_dir=context.model_dir,
            zones=context.zones,
            baseline_od=context.baseline_od,
            road_background_od=context.road_background_od,
            travel_times=context.travel_times,
            lengths=context.lengths,
            assignment_network=detailed_network,
            modules=context.modules,
        )
        
    if not allow_download:
        print("Detailed network cache not found and allow_download=False. Falling back to coarse macroscopic network.")
        return context
        
    try:
        import osmnx as ox
        import networkx as nx
        # Increase timeout and memory limits for large queries
        try:
            ox.settings.timeout = 600
            ox.settings.memory = 1073741824
        except Exception:
            pass
    except ImportError:
        print("The 'osmnx' library is not installed. Falling back to coarse macroscopic network.")
        return context

    zones = context.zones.copy()
    core_zones = _get_core_zones(zones, corridor_municipalities)
    if core_zones.empty:
        raise ValueError(f"None of the specified corridor municipalities {corridor_municipalities} were found in zoning data.")
        
    cordon_polygon = core_zones.geometry.union_all().buffer(float(buffer_m))
    cordon_polygon_wgs84 = gpd.GeoSeries([cordon_polygon], crs=zones.crs).to_crs(4326).iloc[0]
    
    # 1. Fetch OSM network (with per-municipality chunked fallback if full polygon fails)
    try:
        print(f"Downloading detailed OSM road network for {len(corridor_municipalities)} municipalities (this may take 1–3 minutes)...")
        G = ox.graph_from_polygon(cordon_polygon_wgs84, network_type="drive", simplify=True)
    except Exception as e_full:
        print(f"Single polygon query had an issue ({e_full}). Trying chunked per-municipality download...")
        graphs = []
        for muni in corridor_municipalities:
            m_zones = zones.loc[zones["municipality_name"] == muni]
            if m_zones.empty:
                continue
            m_poly = m_zones.geometry.union_all().buffer(float(buffer_m))
            m_poly_w84 = gpd.GeoSeries([m_poly], crs=zones.crs).to_crs(4326).iloc[0]
            try:
                print(f"  • Downloading OSM roads for {muni}...")
                g_muni = ox.graph_from_polygon(m_poly_w84, network_type="drive", simplify=True)
                if len(g_muni) > 0:
                    graphs.append(g_muni)
            except Exception as m_err:
                print(f"    Warning: Could not fetch OSM for {muni}: {m_err}")
                
        if not graphs:
            print(f"Failed to download OSM network: {e_full}. Falling back to coarse macroscopic network.")
            return context
            
        G = nx.compose_all(graphs)

    # Keep only the largest strongly connected component to remove isolated islands
    try:
        largest_scc = max(nx.strongly_connected_components(G), key=len)
        G = G.subgraph(largest_scc).copy()
    except Exception as e:
        print(f"Warning: Could not extract largest connected component: {e}")

    osm_nodes_gdf, osm_edges_gdf = ox.graph_to_gdfs(G)
    
    osm_nodes_gdf = osm_nodes_gdf.to_crs(zones.crs)
    osm_edges_gdf = osm_edges_gdf.to_crs(zones.crs)
    
    base_network = context.assignment_network
    base_nodes = base_network["nodes"].copy()
    base_edges = base_network["edges"].copy()
    
    # 2. Drop coarse edges inside the cordon
    node_inside = base_nodes.geometry.intersects(cordon_polygon)
    inside_node_ids = set(base_nodes.loc[node_inside, "node_id"].astype(int))
    kept_edges = base_edges.loc[
        ~base_edges["source"].isin(inside_node_ids) & 
        ~base_edges["target"].isin(inside_node_ids)
    ].copy()
    
    # 3. Format OSM nodes with unique node IDs
    start_id = int(base_nodes["node_id"].max()) + 1
    osm_osmid_to_id = {osmid: start_id + idx for idx, osmid in enumerate(osm_nodes_gdf.index)}
    
    new_nodes_records = []
    for osmid, row in osm_nodes_gdf.iterrows():
        new_nodes_records.append({
            "node_id": osm_osmid_to_id[osmid],
            "x": float(row.geometry.x),
            "y": float(row.geometry.y),
            "geometry": row.geometry,
            "is_zone": False,
            "zone_id": "",
        })
    new_nodes_gdf = gpd.GeoDataFrame(new_nodes_records, geometry="geometry", crs=zones.crs)
    
    # 4. Format OSM edges with speed & capacity (self-contained to avoid pandana dependency)
    capacity_lookup = {
        "motorway": {1: 1700.0, 2: 4000.0, 3: 5800.0, 4: 7850.0},
        "trunk": {1: 1700.0, 2: 4000.0, 3: 5800.0},
        "motorway_link": {1: 1000.0, 2: 1700.0, 3: 2800.0},
        "trunk_link": {1: 1000.0, 2: 1700.0, 3: 2800.0},
        "primary": {1: 1200.0, 2: 2100.0, 3: 3200.0},
        "primary_link": {1: 1200.0, 2: 2100.0, 3: 3200.0},
        "secondary": {1: 1000.0, 2: 1600.0, 3: 2600.0},
        "secondary_link": {1: 1000.0, 2: 1600.0, 3: 2600.0},
        "tertiary": {1: 900.0, 2: 1500.0, 3: 2400.0},
        "tertiary_link": {1: 900.0, 2: 1500.0, 3: 2400.0},
        "residential": {1: 700.0, 2: 1100.0},
        "unclassified": {1: 700.0, 2: 1100.0},
        "living_street": {1: 400.0},
    }
    speed_lookup = {
        "motorway": 100.0, "trunk": 90.0, "motorway_link": 80.0, "trunk_link": 80.0,
        "primary": 60.0, "primary_link": 60.0, "secondary": 55.0, "secondary_link": 55.0,
        "tertiary": 45.0, "tertiary_link": 45.0, "residential": 35.0, "unclassified": 45.0,
        "living_street": 20.0,
    }

    new_edges_records = []
    for idx, row in osm_edges_gdf.reset_index().iterrows():
        u = row["u"]
        v = row["v"]
        key = row.get("key", 0)
        highway_val = row.get("highway", "unclassified")
        if isinstance(highway_val, (list, tuple, set)):
            highway_val = list(highway_val)[0] if highway_val else "unclassified"
        highway = str(highway_val or "unclassified").strip().lower()
        
        lanes_raw = row.get("lanes", 1.0)
        lanes = pd.to_numeric(pd.Series([lanes_raw]), errors="coerce").fillna(1.0).iloc[0]
        lanes = max(float(lanes), 1.0)
        
        length_m = float(row.geometry.length)
        if length_m <= 0:
            continue
            
        speed_raw = row.get("maxspeed", np.nan)
        speed_kph = pd.to_numeric(pd.Series([speed_raw]), errors="coerce").fillna(np.nan).iloc[0]
        if not np.isfinite(speed_kph) or speed_kph <= 0:
            speed_kph = float(speed_lookup.get(highway, 45.0))
            
        free_flow_time_min = length_m / 1000.0 / speed_kph * 60.0
        
        # Calculate capacity
        cap_table = capacity_lookup.get(highway, {1: 700.0, 2: 1100.0})
        lane_keys = sorted(cap_table.keys())
        cap_vals = [cap_table[k] for k in lane_keys]
        capacity_vph = float(np.interp(lanes, lane_keys, cap_vals))
        
        src_id = osm_osmid_to_id.get(u)
        tgt_id = osm_osmid_to_id.get(v)
        if src_id is None or tgt_id is None:
            continue
            
        new_edges_records.append({
            "source": int(src_id),
            "target": int(tgt_id),
            "source_id": str(u),
            "target_id": str(v),
            "length_m": length_m,
            "free_flow_time_min": free_flow_time_min,
            "capacity_vph": capacity_vph,
            "speed_kph": speed_kph,
            "lanes": float(lanes),
            "highway": highway,
            "alpha": 0.15,
            "beta": 4.0,
            "edge_id": f"OSM_{u}_{v}_{key}",
            "geometry": row.geometry,
        })
        
    new_edges_gdf = gpd.GeoDataFrame(new_edges_records, geometry="geometry", crs=zones.crs)
    
    # 5. Reconnect internal zone centroids
    zone_node_map = dict(base_network["zone_node_map"])
    osm_node_xy = new_nodes_gdf[["x", "y"]].to_numpy(dtype=float)
    tree = cKDTree(osm_node_xy)
    
    internal_zones = _get_core_zones(zones, corridor_municipalities)
    int_centroids_xy = np.column_stack((internal_zones.geometry.centroid.x, internal_zones.geometry.centroid.y))
    _, nearest_osm_indices = tree.query(int_centroids_xy, k=1)
    
    for zone_id, osm_idx in zip(internal_zones["grid_id"].astype(str), nearest_osm_indices):
        zone_node_map[zone_id] = int(new_nodes_gdf.iloc[int(osm_idx)]["node_id"])
        
    # 6. Assemble merged network
    merged_edges = gpd.GeoDataFrame(
        pd.concat([kept_edges, new_edges_gdf], ignore_index=True),
        geometry="geometry",
        crs=zones.crs,
    )
    merged_nodes = gpd.GeoDataFrame(
        pd.concat([base_nodes, new_nodes_gdf], ignore_index=True),
        geometry="geometry",
        crs=zones.crs,
    )
    
    detailed_network = {
        "edges": merged_edges,
        "nodes": merged_nodes,
        "zone_node_map": zone_node_map,
        "metadata": {
            "corridor_municipalities": corridor_municipalities,
            "added_osm_nodes": len(new_nodes_gdf),
            "added_osm_edges": len(new_edges_gdf),
        }
    }
    
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(detailed_network, f)
            
    return TransportContext(
        project_root=context.project_root,
        model_dir=context.model_dir,
        zones=context.zones,
        baseline_od=context.baseline_od,
        road_background_od=context.road_background_od,
        travel_times=context.travel_times,
        lengths=context.lengths,
        assignment_network=detailed_network,
        modules=context.modules,
    )


# =============================================================================
# 9. CORRIDOR FLOW MAP & DESIRE-LINE EXPLORER
# =============================================================================

SPECTRUM_STOPS = [
    (0.0, (0x21, 0x66, 0xac)),   # blue  (lowest bin)
    (0.5, (0x1a, 0x98, 0x50)),   # green (middle bin)
    (1.0, (0xd7, 0x30, 0x27)),   # red   (highest bin)
]

def _spectrum_color(t: float) -> str:
    t = min(max(t, 0.0), 1.0)
    for (t0, c0), (t1, c1) in zip(SPECTRUM_STOPS, SPECTRUM_STOPS[1:]):
        if t0 <= t <= t1:
            f = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            r, g, b = (round(c0[i] + f * (c1[i] - c0[i])) for i in range(3))
            return f"rgb({r},{g},{b})"
    r, g, b = SPECTRUM_STOPS[-1][1]
    return f"rgb({r},{g},{b})"


def _bearing_deg(lon0: float, lat0: float, lon1: float, lat1: float) -> float:
    """Initial compass bearing (degrees clockwise from north), for marker.angle."""
    import math
    phi0, phi1 = math.radians(lat0), math.radians(lat1)
    dlon = math.radians(lon1 - lon0)
    x = math.sin(dlon) * math.cos(phi1)
    y = math.cos(phi0) * math.sin(phi1) - math.sin(phi0) * math.cos(phi1) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


SEQUENTIAL_BLUE = [
    [0.00, "#cde2fb"], [0.15, "#9ec5f4"], [0.30, "#6da7ec"],
    [0.45, "#3987e5"], [0.60, "#256abf"], [0.75, "#184f95"],
    [1.00, "#0d366b"],
]


def _ordered_cordon_matrix(
    matrix: pd.DataFrame,
    corridor: CorridorContext,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Validate and order a reduced OD matrix as internal zones then gates."""
    if not isinstance(matrix, pd.DataFrame) or matrix.empty:
        raise ValueError("The cordon OD matrix must be a non-empty DataFrame.")
    normalised = matrix.copy()
    normalised.index = normalised.index.astype(str)
    normalised.columns = normalised.columns.astype(str)
    if normalised.index.has_duplicates or normalised.columns.has_duplicates:
        raise ValueError("The cordon OD matrix must have unique origin and destination labels.")

    internal_ids = [str(zone) for zone in corridor.zone_ids if str(zone) in normalised.index]
    gate_ids = [
        str(gate) for gate in corridor.gates["gate_id"].astype(str)
        if str(gate) in normalised.index
    ]
    ordered = internal_ids + gate_ids
    missing_columns = [label for label in ordered if label not in normalised.columns]
    if not ordered:
        raise ValueError("The matrix contains no labels belonging to this corridor.")
    if missing_columns:
        raise ValueError(f"Cordon destinations missing from the matrix: {missing_columns}")

    result = normalised.loc[ordered, ordered].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if (result.to_numpy(dtype=float) < 0).any():
        raise ValueError("OD demand cannot contain negative trip values.")
    return result, internal_ids, gate_ids


def _ignore_private_plotly_relayout_properties(figure: Any) -> None:
    """Ignore frontend-only map state unsupported by Plotly's Python schema.

    Some Jupyter frontends send ``map._derived`` (or the legacy
    ``mapbox._derived``) whenever a FigureWidget map is panned or resized.
    These are private browser-calculated values, but certain Plotly version
    combinations try to validate them as public layout properties and raise a
    ``ValueError``. This instance-local handler drops only those private keys;
    ordinary zoom, centre, selection, and click events continue to work.
    """
    trait_name = "_js2py_relayout"
    notifiers = figure._trait_notifiers.get(trait_name, {}).get("change", [])
    default_handlers = [
        handler for handler in list(notifiers)
        if getattr(handler, "name", "") == "_handler_js2py_relayout"
    ]
    if not default_handlers:
        return
    for handler in default_handlers:
        figure.unobserve(handler, names=trait_name)

    def filtered_relayout(change: dict[str, Any]) -> None:
        message = change["new"]
        if not message:
            return
        relayout_data = {
            key: value
            for key, value in dict(message.get("relayout_data", {})).items()
            if key != "lastInputTime"
            and not key.endswith("._derived")
            and "._derived." not in key
        }
        if relayout_data:
            figure.plotly_relayout(
                relayout_data=relayout_data,
                source_view_id=message.get("source_view_id"),
            )
        figure.set_trait(trait_name, None)

    figure.observe(filtered_relayout, names=trait_name)


def od_matrix_explorer(
    matrix: pd.DataFrame,
    corridor: CorridorContext,
    *,
    height: int = 650,
) -> Any:
    """Interactive heatmap of a reduced corridor OD matrix.

    Quadrant controls distinguish internal and gate movements. Hover always
    reports raw trips, even when logarithmic colour scaling is enabled, and a
    clicked cell remains visible in the detail panel.
    """
    import ipywidgets as widgets
    import plotly.graph_objects as go

    mat, internal_ids, gate_ids = _ordered_cordon_matrix(matrix, corridor)
    ordered = internal_ids + gate_ids
    values = mat.to_numpy(dtype=float)
    is_internal = np.asarray([label in set(internal_ids) for label in ordered])
    row_internal = is_internal[:, None]
    column_internal = is_internal[None, :]
    quadrants = {
        "internal -> internal": row_internal & column_internal,
        "gate -> internal": ~row_internal & column_internal,
        "internal -> gate": row_internal & ~column_internal,
        "gate -> gate (pass-through)": ~row_internal & ~column_internal,
    }
    active = {label: True for label in quadrants}
    state = {"log": False}

    def visible_values() -> np.ndarray:
        mask = np.zeros_like(values, dtype=bool)
        for label, enabled in active.items():
            if enabled:
                mask |= quadrants[label]
        visible = np.where(mask, values, np.nan)
        return np.log1p(visible) if state["log"] else visible

    heatmap = go.Heatmap(
        z=visible_values(),
        x=ordered,
        y=ordered,
        customdata=values,
        colorscale=SEQUENTIAL_BLUE,
        colorbar={"title": "Trips"},
        hovertemplate=(
            "Origin: %{y}<br>Destination: %{x}<br>"
            "Trips: %{customdata:,.1f}<extra></extra>"
        ),
        zmin=0.0,
    )
    figure = go.FigureWidget(data=[heatmap])
    figure.update_layout(
        title="Cordon OD matrix - internal zones and gates",
        xaxis={"title": "Destination", "showticklabels": False, "showgrid": False},
        yaxis={
            "title": "Origin", "showticklabels": False,
            "showgrid": False, "autorange": "reversed",
        },
        height=int(height),
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    if internal_ids and gate_ids:
        boundary = len(internal_ids) - 0.5
        figure.add_shape(
            type="line", x0=boundary, x1=boundary, y0=-0.5, y1=len(ordered) - 0.5,
            line={"color": "#52514e", "width": 1, "dash": "dot"},
        )
        figure.add_shape(
            type="line", y0=boundary, y1=boundary, x0=-0.5, x1=len(ordered) - 0.5,
            line={"color": "#52514e", "width": 1, "dash": "dot"},
        )

    scale_label = widgets.HTML()

    def redraw() -> None:
        visible = visible_values()
        finite = visible[np.isfinite(visible)]
        with figure.batch_update():
            figure.data[0].z = visible
            figure.data[0].zmax = float(finite.max()) if finite.size else 1.0
        scale_label.value = (
            "<i>Colours use log(1 + trips); hover values remain unscaled.</i>"
            if state["log"] else "<i>Colours show trips on a linear scale.</i>"
        )

    checkboxes = {
        label: widgets.Checkbox(value=True, description=label, indent=False)
        for label in quadrants
    }
    for label, checkbox in checkboxes.items():
        def on_quadrant_toggle(change: dict[str, Any], label: str = label) -> None:
            active[label] = bool(change["new"])
            redraw()
        checkbox.observe(on_quadrant_toggle, names="value")

    log_toggle = widgets.Checkbox(value=False, description="Log colour scale", indent=False)

    def on_log_toggle(change: dict[str, Any]) -> None:
        state["log"] = bool(change["new"])
        redraw()

    log_toggle.observe(on_log_toggle, names="value")
    detail = widgets.HTML(value="<i>Click a cell to pin its value here.</i>")

    def on_click(trace: Any, points: Any, selector: Any) -> None:
        if not points.xs:
            return
        destination, origin = str(points.xs[0]), str(points.ys[0])
        detail.value = (
            f"<b>{origin} -> {destination}</b><br>"
            f"Trips: {float(mat.loc[origin, destination]):,.1f}"
        )

    figure.data[0].on_click(on_click)
    redraw()
    controls = widgets.VBox(
        [widgets.HTML("<b>Show:</b>"), *checkboxes.values(),
         widgets.HTML("<br><b>Scale:</b>"), log_toggle, scale_label]
    )
    side = widgets.VBox(
        [controls, detail], layout=widgets.Layout(width="280px", padding="8px")
    )
    return widgets.HBox([figure, side])


def corridor_flow_map_explorer(
    matrix: pd.DataFrame,
    corridor: CorridorContext,
    *,
    n_bins: int = 5,
    max_flows: int = 250,
    min_trips: float = 0.0,
    height: int = 680,
) -> Any:
    """Interactive, project-neutral desire-line map for a reduced OD matrix.

    The matrix is assumed to have already been reduced by
    :func:`collapse_od_to_gates`; pass-through demand is therefore not scaled
    again. Quantile bins and their legend are recomputed after each category
    filter, while ``max_flows`` limits browser rendering rather than demand.
    """
    import geopandas as gpd
    import ipywidgets as widgets
    import plotly.graph_objects as go

    mat, internal_ids, gate_ids = _ordered_cordon_matrix(matrix, corridor)
    internal_set = set(internal_ids)
    zone_rows = corridor.zones.loc[
        corridor.zones["grid_id"].astype(str).isin(internal_ids)
    ].copy()
    zone_centroids = gpd.GeoSeries(zone_rows["centroid"], crs=zone_rows.crs).to_crs(4326)
    zone_xy = {
        str(row.grid_id): (point.x, point.y)
        for row, point in zip(zone_rows.itertuples(index=False), zone_centroids)
    }
    gates_wgs84 = corridor.gates.set_geometry("geometry").to_crs(4326)
    gate_xy = {
        str(gate): (point.x, point.y)
        for gate, point in zip(corridor.gates["gate_id"].astype(str), gates_wgs84.geometry)
    }
    coordinates = {**zone_xy, **gate_xy}
    polygon_wgs84 = gpd.GeoSeries(
        [corridor.polygon], crs=corridor.zones.crs
    ).to_crs(4326).iloc[0]
    polygons = (
        list(polygon_wgs84.geoms)
        if hasattr(polygon_wgs84, "geoms") else [polygon_wgs84]
    )

    records = []
    for origin, row in mat.iterrows():
        if origin not in coordinates:
            continue
        for destination, value in row.items():
            value = float(value)
            if destination == origin or destination not in coordinates or value <= float(min_trips):
                continue
            origin_internal = origin in internal_set
            destination_internal = destination in internal_set
            if origin_internal and destination_internal:
                category = "internal -> internal"
            elif not origin_internal and destination_internal:
                category = "gate -> internal"
            elif origin_internal and not destination_internal:
                category = "internal -> gate"
            else:
                category = "gate -> gate (pass-through)"
            records.append((origin, destination, value, category))

    flows = pd.DataFrame(records, columns=["origin", "destination", "value", "category"])
    active_category = {label: True for label in flows["category"].unique()} if len(flows) else {}
    active_bin = {index: True for index in range(max(int(n_bins), 1))}
    all_lons = [coordinate[0] for coordinate in coordinates.values()]
    all_lats = [coordinate[1] for coordinate in coordinates.values()]
    center = {"lon": float(np.mean(all_lons)), "lat": float(np.mean(all_lats))}

    figure = go.FigureWidget()
    _ignore_private_plotly_relayout_properties(figure)
    figure.update_layout(
        title="Cordon OD flows - desire lines",
        map={"style": "open-street-map", "center": center, "zoom": 11.0},
        height=int(height),
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
        showlegend=False,
    )
    detail = widgets.HTML(value="<i>Click a flow line to pin its value here.</i>")
    flow_count_label = widgets.HTML()
    bin_controls_box = widgets.VBox()

    def click_handler(origin: str, destination: str, value: float) -> Any:
        def handler(trace: Any, points: Any, selector: Any) -> None:
            if points.point_inds:
                detail.value = f"<b>{origin} -> {destination}</b><br>Trips: {value:,.1f}"
        return handler

    def redraw() -> None:
        category_mask = (
            flows["category"].map(active_category).fillna(False)
            if len(flows) else pd.Series(dtype=bool)
        )
        binned = flows.loc[category_mask].copy() if len(flows) else flows.copy()
        bins_info = []
        if len(binned):
            codes, _ = pd.qcut(
                binned["value"], q=max(int(n_bins), 1), labels=False,
                retbins=True, duplicates="drop",
            )
            binned["bin"] = codes.astype(int)
            actual_bins = int(binned["bin"].max()) + 1
            for bin_index in range(actual_bins):
                bin_values = binned.loc[binned["bin"] == bin_index, "value"]
                position = bin_index / (actual_bins - 1) if actual_bins > 1 else 0.5
                bins_info.append({
                    "index": bin_index,
                    "count": int(len(bin_values)),
                    "minimum": float(bin_values.min()),
                    "maximum": float(bin_values.max()),
                    "color": _spectrum_color(position),
                })
        else:
            actual_bins = 0

        bin_rows = []
        for info in bins_info:
            bin_index = info["index"]
            swatch = widgets.HTML(
                value=(
                    f'<div style="width:14px;height:14px;background:{info["color"]};'
                    'border-radius:3px;margin-top:3px;"></div>'
                )
            )
            checkbox = widgets.Checkbox(
                value=active_bin.get(bin_index, True),
                description=(
                    f'{info["minimum"]:,.1f}-{info["maximum"]:,.1f} trips '
                    f'(n={info["count"]})'
                ),
                indent=False,
                layout=widgets.Layout(width="240px"),
            )
            def on_bin_toggle(change: dict[str, Any], bin_index: int = bin_index) -> None:
                active_bin[bin_index] = bool(change["new"])
                redraw()
            checkbox.observe(on_bin_toggle, names="value")
            bin_rows.append(widgets.HBox([swatch, checkbox]))
        bin_controls_box.children = bin_rows

        subset = (
            binned.loc[binned["bin"].map(lambda value: active_bin.get(int(value), True))]
            if actual_bins else binned
        )
        selected_count = len(subset)
        subset = subset.sort_values("value", ascending=False).head(int(max_flows))
        base_traces = []
        for part in polygons:
            longitudes, latitudes = part.exterior.xy
            base_traces.append(go.Scattermap(
                lon=list(longitudes), lat=list(latitudes), fill="toself",
                fillcolor="rgba(35,120,180,0.12)",
                line={"color": "rgba(35,120,180,0.8)", "width": 2},
                hoverinfo="skip", showlegend=False,
            ))
        base_traces.append(go.Scattermap(
            lon=[coordinates[zone][0] for zone in internal_ids if zone in coordinates],
            lat=[coordinates[zone][1] for zone in internal_ids if zone in coordinates],
            mode="markers", marker={"size": 5, "color": "#898781"},
            hoverinfo="skip", showlegend=False,
        ))
        visible_gates = [gate for gate in gate_ids if gate in coordinates]
        base_traces.append(go.Scattermap(
            lon=[coordinates[gate][0] for gate in visible_gates],
            lat=[coordinates[gate][1] for gate in visible_gates],
            mode="markers", marker={"size": 11, "color": "#7a0177"},
            text=visible_gates, hovertemplate="Gate %{text}<extra></extra>", showlegend=False,
        ))

        width_upper = max(float(subset["value"].quantile(0.98)), 1e-9) if len(subset) else 1.0
        colours = {info["index"]: info["color"] for info in bins_info}
        flow_traces, trace_metadata = [], []
        arrow_lon, arrow_lat, arrow_angle, arrow_colour, arrow_size = [], [], [], [], []
        for flow in subset.itertuples(index=False):
            lon0, lat0 = coordinates[flow.origin]
            lon1, lat1 = coordinates[flow.destination]
            colour = colours.get(int(flow.bin), "rgb(128,128,128)")
            width = 1.0 + 4.0 * np.sqrt(min(float(flow.value) / width_upper, 1.0))
            flow_traces.append(go.Scattermap(
                lon=[lon0, lon1], lat=[lat0, lat1], mode="lines",
                line={"color": colour, "width": width},
                text=[
                    f"{flow.origin} -> {flow.destination}<br>{flow.category}<br>"
                    f"Trips: {float(flow.value):,.1f}"
                ] * 2,
                hovertemplate="%{text}<extra></extra>", showlegend=False,
            ))
            trace_metadata.append((flow.origin, flow.destination, float(flow.value)))
            arrow_lon.append(lon0 + 0.85 * (lon1 - lon0))
            arrow_lat.append(lat0 + 0.85 * (lat1 - lat0))
            arrow_angle.append(_bearing_deg(lon0, lat0, lon1, lat1))
            arrow_colour.append(colour)
            arrow_size.append(8 + 6 * min(float(flow.value) / width_upper, 1.0))
        arrow_trace = go.Scattermap(
            lon=arrow_lon, lat=arrow_lat, mode="markers",
            marker={
                "symbol": "triangle", "size": arrow_size,
                "angle": arrow_angle, "color": arrow_colour,
            },
            hoverinfo="skip", showlegend=False,
        )

        with figure.batch_update():
            figure.data = []
            figure.add_traces(base_traces)
            figure.add_traces(flow_traces)
            flow_start = len(base_traces)
            figure.add_trace(arrow_trace)
        for trace, metadata in zip(
            figure.data[flow_start:flow_start + len(flow_traces)], trace_metadata
        ):
            trace.on_click(click_handler(*metadata))
        flow_count_label.value = (
            f"Showing {len(subset)} of {selected_count} selected flows "
            f"(capped at {int(max_flows)})."
        )

    category_checkboxes = {}
    for label in sorted(active_category):
        checkbox = widgets.Checkbox(value=True, description=label, indent=False)
        def on_category_toggle(change: dict[str, Any], label: str = label) -> None:
            active_category[label] = bool(change["new"])
            redraw()
        checkbox.observe(on_category_toggle, names="value")
        category_checkboxes[label] = checkbox
    redraw()
    controls = widgets.VBox([
        widgets.HTML("<b>Show category:</b>"), *category_checkboxes.values(),
        widgets.HTML("<br><b>Show range (equal-count bins):</b>"),
        bin_controls_box, widgets.HTML("<br>"), flow_count_label,
    ])
    side = widgets.VBox(
        [controls, detail], layout=widgets.Layout(width="320px", padding="8px")
    )
    return widgets.HBox([figure, side])


def corridor_od_explorer(
    matrix: pd.DataFrame,
    corridor: CorridorContext,
    **flow_map_kwargs: Any,
) -> Any:
    """Combine the OD heatmap and desire-line map in one compact tab widget."""
    import ipywidgets as widgets

    tabs = widgets.Tab(children=[
        od_matrix_explorer(matrix, corridor),
        corridor_flow_map_explorer(matrix, corridor, **flow_map_kwargs),
    ])
    tabs.set_title(0, "OD matrix")
    tabs.set_title(1, "Desire lines")
    return tabs


def aggregate_od_by_municipality(
    context: TransportContext,
    municipalities: list[str] | tuple[str, ...] | set[str],
    *,
    matrix: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate an FSM zone-level OD matrix into directed municipality flows.

    The function is project-agnostic: the caller supplies the municipalities
    and may replace the cantonal baseline with any square FSM OD matrix.
    Returns both the municipality matrix and a long directional-link table.
    """
    requested = list(dict.fromkeys(map(str, municipalities)))
    if not requested:
        raise ValueError("At least one municipality must be supplied.")

    source = context.baseline_od if matrix is None else matrix
    if not isinstance(source, pd.DataFrame) or source.empty:
        raise ValueError("The source OD matrix must be a non-empty DataFrame.")
    source = source.copy()
    source.index = source.index.astype(str)
    source.columns = source.columns.astype(str)
    if source.index.has_duplicates or source.columns.has_duplicates:
        raise ValueError("The source OD matrix must have unique zone labels.")

    required_columns = {"grid_id", "municipality_name"}
    missing_columns = required_columns.difference(context.zones.columns)
    if missing_columns:
        raise ValueError(f"Zone data are missing columns: {sorted(missing_columns)}")
    zones = context.zones[["grid_id", "municipality_name"]].copy()
    zones["grid_id"] = zones["grid_id"].astype(str)
    zones["municipality_name"] = zones["municipality_name"].astype(str)
    selected_zones = zones.loc[zones["municipality_name"].isin(requested)].copy()
    zone_ids = [
        zone_id for zone_id in selected_zones["grid_id"]
        if zone_id in source.index and zone_id in source.columns
    ]
    if not zone_ids:
        raise ValueError("No FSM zones matched the selected municipalities.")

    zone_to_municipality = (
        selected_zones.drop_duplicates("grid_id")
        .set_index("grid_id")
        .loc[zone_ids, "municipality_name"]
    )
    selected_od = source.loc[zone_ids, zone_ids].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if (selected_od.to_numpy(dtype=float) < 0).any():
        raise ValueError("OD demand cannot contain negative trip values.")

    # Aggregate origins, transpose, then aggregate destinations. This avoids
    # the deprecated ``groupby(axis=1)`` API and works across supported pandas versions.
    municipality_od = selected_od.groupby(zone_to_municipality, sort=False).sum()
    municipality_od = municipality_od.T.groupby(zone_to_municipality, sort=False).sum().T
    municipality_order = [name for name in requested if name in municipality_od.index]
    municipality_od = municipality_od.reindex(
        index=municipality_order,
        columns=municipality_order,
        fill_value=0.0,
    )

    directed_links = municipality_od.rename_axis("origin").reset_index().melt(
        id_vars="origin",
        var_name="destination",
        value_name="peak-hour passenger trips",
    )
    directed_links = directed_links.loc[
        directed_links["origin"] != directed_links["destination"]
    ].copy()
    directed_links["OD relation"] = (
        directed_links["origin"] + " -> " + directed_links["destination"]
    )
    directed_links = directed_links.sort_values(
        "peak-hour passenger trips", ascending=False
    ).reset_index(drop=True)
    return municipality_od, directed_links


def municipality_od_explorer(
    context: TransportContext,
    municipalities: list[str] | tuple[str, ...] | set[str],
    *,
    matrix: pd.DataFrame | None = None,
    height: int = 700,
) -> Any:
    """Return a static Plotly heatmap of directed municipality-level demand.

    A regular ``Figure`` is intentional: it keeps hover and zoom in the browser
    without FigureWidget relayout callbacks, avoiding frontend/backend Plotly
    version conflicts involving private ``map._derived`` state.
    """
    import plotly.graph_objects as go

    municipality_od, _ = aggregate_od_by_municipality(
        context,
        municipalities,
        matrix=matrix,
    )
    values = municipality_od.to_numpy(dtype=float)
    figure = go.Figure(go.Heatmap(
        z=values,
        x=municipality_od.columns,
        y=municipality_od.index,
        colorscale="Blues",
        colorbar={"title": "Passenger trips"},
        text=np.round(values, 1),
        texttemplate="%{text:,.1f}",
        hovertemplate=(
            "Origin: %{y}<br>Destination: %{x}<br>"
            "Peak-hour passenger trips: %{z:,.1f}<extra></extra>"
        ),
    ))
    figure.update_layout(
        title="Cantonal-reference OD demand by municipality",
        xaxis_title="Destination municipality",
        yaxis_title="Origin municipality",
        yaxis_autorange="reversed",
        height=int(height),
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    return figure


def municipality_flow_map_explorer(
    context: TransportContext,
    municipalities: list[str] | tuple[str, ...] | set[str],
    *,
    matrix: pd.DataFrame | None = None,
    n_bins: int = 5,
    trip_bin_edges: list[float] | None = None,
    min_trips: float = 0.0,
    max_flows: int = 150,
    height: int = 700,
) -> Any:
    """Map directional municipality OD desire lines from any FSM matrix.

    Municipal centroids anchor straight desire lines; they are not assigned
    routes. A regular Plotly ``Figure`` keeps the map independent of Jupyter
    widget relayout callbacks. Fixed trip intervals may be supplied through
    ``trip_bin_edges``; otherwise equal-count quantile bins are used.
    """
    import geopandas as gpd
    import plotly.graph_objects as go

    _, links = aggregate_od_by_municipality(
        context, municipalities, matrix=matrix
    )
    links = links.loc[
        links["peak-hour passenger trips"] > float(min_trips)
    ].copy()
    if links.empty:
        raise ValueError("No municipality OD flows exceed min_trips.")

    requested = list(dict.fromkeys(map(str, municipalities)))
    zones = context.zones.loc[
        context.zones["municipality_name"].astype(str).isin(requested),
        ["municipality_name", "geometry"],
    ].copy()
    if zones.crs is None:
        raise ValueError("Zone geometries need a declared CRS.")
    zones["municipality_name"] = zones["municipality_name"].astype(str)
    municipality_shapes = zones.dissolve(by="municipality_name")
    centres_projected = municipality_shapes.geometry.centroid
    centres_wgs84 = gpd.GeoSeries(
        centres_projected, index=municipality_shapes.index, crs=zones.crs
    ).to_crs(4326)
    coordinates = {
        str(name): (float(point.x), float(point.y))
        for name, point in centres_wgs84.items()
    }
    links = links.loc[
        links["origin"].isin(coordinates)
        & links["destination"].isin(coordinates)
    ].copy()
    # A valid attribute name keeps ``itertuples`` access independent of how
    # pandas sanitises the human-readable column containing spaces and hyphens.
    links["value"] = links["peak-hour passenger trips"].astype(float)

    values = links["value"]
    if trip_bin_edges is not None:
        edges = np.asarray(trip_bin_edges, dtype=float)
        if len(edges) < 2 or not np.all(np.diff(edges) > 0):
            raise ValueError(
                "trip_bin_edges must contain at least two strictly increasing values."
            )
        links["bin"] = pd.cut(
            values, bins=edges, labels=False, include_lowest=True, right=False
        )
        links = links.dropna(subset=["bin"]).copy()
        links["bin"] = links["bin"].astype(int)
        bin_labels = {
            index: (
                f"{edges[index]:,.0f}+ trips"
                if np.isinf(edges[index + 1])
                else f"{edges[index]:,.0f}-{edges[index + 1]:,.0f} trips"
            )
            for index in sorted(links["bin"].unique())
        }
    else:
        requested_bins = max(1, min(int(n_bins), len(links)))
        links["bin"] = pd.qcut(
            values, q=requested_bins, labels=False, duplicates="drop"
        ).astype(int)
        bin_labels = {}
        for index in sorted(links["bin"].unique()):
            bin_values = links.loc[
                links["bin"] == index, "peak-hour passenger trips"
            ]
            bin_labels[index] = (
                f"{float(bin_values.min()):,.0f}-{float(bin_values.max()):,.0f} trips"
            )

    links = links.sort_values(
        "peak-hour passenger trips", ascending=False
    ).head(int(max_flows))
    populated_bins = sorted(links["bin"].unique())
    colours = {
        bin_index: _spectrum_color(
            position / (len(populated_bins) - 1) if len(populated_bins) > 1 else 0.5
        )
        for position, bin_index in enumerate(populated_bins)
    }
    width_upper = max(
        float(links["peak-hour passenger trips"].quantile(0.98)), 1e-9
    )

    figure = go.Figure()
    arrow_lon, arrow_lat, arrow_angle, arrow_colour, arrow_size = [], [], [], [], []
    for flow in links.itertuples(index=False):
        lon0, lat0 = coordinates[flow.origin]
        lon1, lat1 = coordinates[flow.destination]
        value = float(flow.value)
        bin_index = int(flow.bin)
        colour = colours[bin_index]
        width = 1.0 + 5.0 * np.sqrt(min(value / width_upper, 1.0))
        figure.add_trace(go.Scattermap(
            lon=[lon0, lon1], lat=[lat0, lat1], mode="lines",
            line={"color": colour, "width": width},
            text=[
                f"{flow.origin} -> {flow.destination}<br>"
                f"Peak-hour passenger trips: {value:,.1f}"
            ] * 2,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        ))
        arrow_lon.append(lon0 + 0.85 * (lon1 - lon0))
        arrow_lat.append(lat0 + 0.85 * (lat1 - lat0))
        arrow_angle.append(_bearing_deg(lon0, lat0, lon1, lat1))
        arrow_colour.append(colour)
        arrow_size.append(8 + 6 * min(value / width_upper, 1.0))

    figure.add_trace(go.Scattermap(
        lon=arrow_lon, lat=arrow_lat, mode="markers",
        marker={
            "symbol": "triangle", "size": arrow_size,
            "angle": arrow_angle, "color": arrow_colour,
        },
        hoverinfo="skip", showlegend=False,
    ))
    names = [name for name in requested if name in coordinates]
    figure.add_trace(go.Scattermap(
        lon=[coordinates[name][0] for name in names],
        lat=[coordinates[name][1] for name in names],
        mode="markers+text",
        marker={"size": 10, "color": "#5b2c83"},
        text=names,
        textposition="top center",
        hovertemplate="%{text}<extra></extra>",
        name="Municipalities",
    ))
    for bin_index in populated_bins:
        figure.add_trace(go.Scattermap(
            lon=[None], lat=[None], mode="markers",
            marker={"size": 10, "color": colours[bin_index]},
            name=bin_labels[bin_index], hoverinfo="skip",
        ))
    figure.update_layout(
        title=(
            "Cantonal-reference municipality OD desire lines "
            f"(showing {len(links)} flows)"
        ),
        map={
            "style": "open-street-map",
            "center": {
                "lon": float(np.mean([coordinates[name][0] for name in names])),
                "lat": float(np.mean([coordinates[name][1] for name in names])),
            },
            "zoom": 9.3,
        },
        height=int(height),
        margin={"l": 0, "r": 0, "t": 55, "b": 0},
        legend={"title": "Flow range"},
    )
    return figure


def flow_map_explorer(
    context: TransportContext,
    mode_result: ModeChoiceResult | None = None,
    corridor_municipalities: list[str] | None = None,
    *,
    matrix: pd.DataFrame | None = None,
    buffer_m: float = DEFAULT_CORRIDOR_BUFFER_M,
    n_bins: int = 5,
    max_flows: int = 250,
    pass_through_factor: float = 0.05,
    min_trips: float = 0.0,
) -> Any:
    """
    Desire-line map on OpenStreetMap basemap: OD flows as arrowed lines, split into
    n_bins equal-count quantile bins with interactive category and range filtering.
    """
    import math
    import geopandas as gpd
    import ipywidgets as widgets
    import plotly.graph_objects as go
    
    zones = context.zones.copy()
    if corridor_municipalities is None:
        corridor_municipalities = list(zones["municipality_name"].unique())
        
    core_zones = _get_core_zones(zones, corridor_municipalities)
    if core_zones.empty:
        raise ValueError("No matching corridor municipalities found.")
        
    polygon = core_zones.geometry.union_all().buffer(float(buffer_m))
    
    # Check for gates in mode_result or generate them
    gates_gdf = None
    if mode_result is not None and mode_result.assigned_metadata is not None and "gates" in mode_result.assigned_metadata:
        gates_gdf = mode_result.assigned_metadata["gates"].copy()
    else:
        # Generate gates using helper
        _, gates_gdf = _get_or_generate_corridor(context, corridor_municipalities, buffer_m=buffer_m, max_gates=DEFAULT_MAX_GATES)

    # Zone centroids and gate coordinates in WGS84
    zone_centroids = gpd.GeoSeries(zones["centroid"], crs=zones.crs)
    internal_mask = zone_centroids.intersects(polygon)
    internal_zones = zones.loc[internal_mask].copy()
    internal_ids = set(internal_zones["grid_id"].astype(str))
    
    int_centroids_wgs84 = gpd.GeoSeries(internal_zones["centroid"], crs=zones.crs).to_crs(4326)
    zone_xy = {
        str(row.grid_id): (pt.x, pt.y)
        for row, pt in zip(internal_zones.itertuples(index=False), int_centroids_wgs84)
    }
    
    gates_wgs84 = gates_gdf.set_geometry("geometry").to_crs(4326)
    gate_xy = {
        str(gid): (pt.x, pt.y)
        for gid, pt in zip(gates_gdf["gate_id"].astype(str), gates_wgs84.geometry)
    }
    coords = {**zone_xy, **gate_xy}
    
    # Polygon in WGS84
    polygon_wgs84 = gpd.GeoSeries([polygon], crs=zones.crs).to_crs(4326).iloc[0]
    polygons = list(polygon_wgs84.geoms) if hasattr(polygon_wgs84, "geoms") else [polygon_wgs84]
    
    # Build or use matrix
    if matrix is None:
        if mode_result is None:
            raise ValueError("Either matrix or mode_result must be provided.")
        # Map external demand to nearest gate
        external = zones.loc[~zones["grid_id"].astype(str).isin(internal_ids)].copy()
        external_centroids = gpd.GeoSeries(external["centroid"], crs=zones.crs)
        ext_xy = np.column_stack((external_centroids.x, external_centroids.y))
        
        def map_gates(eligible_gates):
            if eligible_gates.empty: return {}
            gate_xy_loc = np.column_stack((eligible_gates.geometry.x, eligible_gates.geometry.y))
            nearest = np.argmin(((ext_xy[:, None, 0] - gate_xy_loc[None, :, 0])**2 + (ext_xy[:, None, 1] - gate_xy_loc[None, :, 1])**2), axis=1)
            return dict(zip(external["grid_id"].astype(str), eligible_gates.iloc[nearest]["gate_id"].astype(str)))

        origin_map = {z: z for z in internal_ids}
        origin_map.update(map_gates(gates_gdf.loc[gates_gdf["can_enter"].astype(bool)]))
        
        dest_map = {z: z for z in internal_ids}
        dest_map.update(map_gates(gates_gdf.loc[gates_gdf["can_exit"].astype(bool)]))
        
        total_od = sum(mode_result.od_by_mode.values()).copy()
        total_od.index = total_od.index.astype(str).map(lambda x: origin_map.get(x, None))
        total_od.columns = total_od.columns.astype(str).map(lambda x: dest_map.get(x, None))
        total_od = total_od.loc[total_od.index.notna(), total_od.columns.notna()]
        matrix = total_od.groupby(level=0).sum().groupby(level=0, axis=1).sum()
        
    gate_ids = set(gates_gdf["gate_id"].astype(str)) if gates_gdf is not None else set()
    
    records = []
    for origin in matrix.index:
        orig_str = str(origin)
        if orig_str not in coords:
            continue
        for destination in matrix.columns:
            dest_str = str(destination)
            if dest_str == orig_str or dest_str not in coords:
                continue
            value = float(matrix.loc[origin, destination])
            if value <= min_trips:
                continue
            origin_internal = orig_str in internal_ids
            destination_internal = dest_str in internal_ids
            if origin_internal and destination_internal:
                category = "internal → internal"
            elif not origin_internal and destination_internal:
                category = "gate → internal"
            elif origin_internal and not destination_internal:
                category = "internal → gate"
            else:
                category = "gate → gate (pass-through)"
                value = value * float(pass_through_factor)
                if value <= min_trips:
                    continue
            records.append((orig_str, dest_str, value, category))

    flows = pd.DataFrame(records, columns=["origin", "destination", "value", "category"])
    active_category = {key: True for key in flows["category"].unique()} if len(flows) else {}
    active_bin = {i: True for i in range(n_bins)}

    all_lons = [c[0] for c in coords.values()]
    all_lats = [c[1] for c in coords.values()]
    center = dict(lon=float(np.mean(all_lons)), lat=float(np.mean(all_lats)))

    fig = go.FigureWidget()
    _ignore_private_plotly_relayout_properties(fig)
    fig.update_layout(
        title="Corridor OD Flows — Desire Lines",
        map=dict(style="open-street-map", center=center, zoom=11.0),
        height=680,
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=False,
    )

    detail = widgets.HTML(value="<i>Click a flow line to pin its value here.</i>")
    flow_count_label = widgets.HTML()
    bin_controls_box = widgets.VBox()

    def on_click(origin, destination, value):
        def handler(trace, points, sel_state):
            if points.point_inds:
                detail.value = f"<b>{origin} → {destination}</b><br>Trips: {value:,.1f}"
        return handler

    def redraw():
        category_mask = flows["category"].map(active_category).fillna(False) if len(flows) else pd.Series(dtype=bool)
        by_category = flows.loc[category_mask] if len(flows) else flows

        bins_info = []
        binned = by_category.copy()
        if len(by_category):
            codes, edges = pd.qcut(
                by_category["value"], q=n_bins, labels=False, retbins=True, duplicates="drop"
            )
            binned["bin"] = codes.values
            actual_bins = int(binned["bin"].max()) + 1
            for b in range(actual_bins):
                bin_values = binned.loc[binned["bin"] == b, "value"]
                t = b / (actual_bins - 1) if actual_bins > 1 else 0.5
                bins_info.append({
                    "index": b,
                    "count": int(len(bin_values)),
                    "vmin": float(bin_values.min()),
                    "vmax": float(bin_values.max()),
                    "color": _spectrum_color(t),
                })
        else:
            actual_bins = 0

        bin_rows = []
        for info in bins_info:
            b = info["index"]
            swatch = widgets.HTML(
                value=f'<div style="width:14px;height:14px;background:{info["color"]};'
                      f'border-radius:3px;margin-top:3px;"></div>'
            )
            cb = widgets.Checkbox(
                value=active_bin.get(b, True),
                description=f'{info["vmin"]:,.1f}–{info["vmax"]:,.1f} trips (n={info["count"]})',
                indent=False,
                layout=widgets.Layout(width="240px"),
            )
            def on_bin_toggle(change, b=b):
                active_bin[b] = change["new"]
                redraw()
            cb.observe(on_bin_toggle, names="value")
            bin_rows.append(widgets.HBox([swatch, cb]))
        bin_controls_box.children = bin_rows

        subset = (
            binned.loc[binned["bin"].map(lambda b: active_bin.get(b, True))]
            if actual_bins else binned.assign(bin=[])
        )
        subset = subset.sort_values("value", ascending=False).head(max_flows)

        base_traces = []
        for part in polygons:
            lons, lats = part.exterior.xy
            base_traces.append(go.Scattermap(
                lon=list(lons), lat=list(lats), fill="toself",
                fillcolor="rgba(35,120,180,0.12)",
                line=dict(color="rgba(35,120,180,0.8)", width=2),
                hoverinfo="skip", showlegend=False,
            ))
        base_traces.append(go.Scattermap(
            lon=[coords[z][0] for z in internal_ids if z in coords], lat=[coords[z][1] for z in internal_ids if z in coords],
            mode="markers", marker=dict(size=5, color="#898781"),
            hoverinfo="skip", showlegend=False,
        ))
        base_traces.append(go.Scattermap(
            lon=[coords[g][0] for g in gate_ids if g in coords], lat=[coords[g][1] for g in gate_ids if g in coords],
            mode="markers", marker=dict(size=11, color="#7a0177"),
            text=list(gate_ids), hovertemplate="Gate %{text}<extra></extra>", showlegend=False,
        ))

        width_upper = max(float(subset["value"].quantile(0.98)), 1e-9) if len(subset) else 1.0
        colors_by_bin = {info["index"]: info["color"] for info in bins_info}

        new_traces, meta = [], []
        arrow_lon, arrow_lat, arrow_angle, arrow_color, arrow_size = [], [], [], [], []
        for row in subset.itertuples(index=False):
            lon0, lat0 = coords[row.origin]
            lon1, lat1 = coords[row.destination]
            colour = colors_by_bin.get(int(row.bin), "rgb(128,128,128)")
            width = 1.0 + 4.0 * math.sqrt(min(row.value / width_upper, 1.0))
            new_traces.append(go.Scattermap(
                lon=[lon0, lon1], lat=[lat0, lat1], mode="lines",
                line=dict(color=colour, width=width),
                text=[f"{row.origin} → {row.destination}<br>{row.category}<br>Trips: {row.value:,.1f}"] * 2,
                hovertemplate="%{text}<extra></extra>", showlegend=False,
            ))
            meta.append((row.origin, row.destination, row.value))

            tlon, tlat = lon0 + 0.85 * (lon1 - lon0), lat0 + 0.85 * (lat1 - lat0)
            bearing = _bearing_deg(lon0, lat0, lon1, lat1)
            arrow_lon.append(tlon); arrow_lat.append(tlat)
            arrow_angle.append(bearing); arrow_color.append(colour)
            arrow_size.append(8 + 6 * min(row.value / width_upper, 1.0))

        arrow_trace = go.Scattermap(
            lon=arrow_lon, lat=arrow_lat, mode="markers",
            marker=dict(symbol="triangle", size=arrow_size, angle=arrow_angle, color=arrow_color),
            hoverinfo="skip", showlegend=False,
        )

        with fig.batch_update():
            fig.data = []
            fig.add_traces(base_traces)
            fig.add_traces(new_traces)
            line_trace_start = len(base_traces)
            fig.add_traces([arrow_trace])

        for trace, (origin, destination, value) in zip(
            fig.data[line_trace_start:line_trace_start + len(new_traces)], meta
        ):
            trace.on_click(on_click(origin, destination, value))

        flow_count_label.value = (
            f"Showing {len(subset)} of {int(binned['bin'].map(lambda b: active_bin.get(b, True)).sum()) if actual_bins else 0} "
            f"selected-bin flows (capped at {max_flows})."
        )

    category_checkboxes = {}
    for key in sorted(active_category):
        cb = widgets.Checkbox(value=True, description=key, indent=False)
        def on_category_toggle(change, key=key):
            active_category[key] = change["new"]
            redraw()
        cb.observe(on_category_toggle, names="value")
        category_checkboxes[key] = cb

    redraw()

    controls = widgets.VBox([
        widgets.HTML("<b>Show category:</b>"), *category_checkboxes.values(),
        widgets.HTML("<br><b>Show range (equal-count bins):</b>"), bin_controls_box,
        widgets.HTML("<br>"), flow_count_label,
    ])
    side = widgets.VBox([controls, detail], layout=widgets.Layout(width="320px", padding="8px"))
    return widgets.HBox([fig, side])


# =============================================================================
# 10. MODAL SPLIT & OD RELATION SUMMARIES
# =============================================================================

def corridor_modal_split(
    context: TransportContext,
    mode_result: ModeChoiceResult,
    corridor_municipalities: list[str] | None = None,
    zone_ids: list[str] | set[str] | None = None,
    *,
    detailed_pt: bool = False,
) -> pd.DataFrame:
    """
    Compute internal-to-internal (Binnenverkehr) modal split within the specified corridor.
    
    Parameters:
        context: Loaded transport context.
        mode_result: Output from run_transport_mode_choice.
        corridor_municipalities: Municipalities to include in the internal corridor.
        detailed_pt: If False (default), aggregates into 4 modes (Car, PT, Bike, Walk).
                     If True, breaks PT into PT (Walk access) and PT (Bike access).
    """
    if zone_ids is None and corridor_municipalities is not None:
        zone_ids = get_zone_ids_for_municipalities(context, corridor_municipalities)
    elif zone_ids is None:
        zone_ids = list(context.zones["grid_id"].astype(str))
        
    od_by_mode = mode_result.od_by_mode
    reference = next(iter(od_by_mode.values()))
    labels = reference.index.astype(str)
    internal = np.isin(labels, list(map(str, zone_ids)))
    internal_to_internal = internal[:, None] & internal[None, :]

    car_total = float(od_by_mode["drive"].to_numpy(dtype=float)[internal_to_internal].sum()) if "drive" in od_by_mode else 0.0
    pt_walk_total = float(od_by_mode["pt_walk"].to_numpy(dtype=float)[internal_to_internal].sum()) if "pt_walk" in od_by_mode else 0.0
    pt_bike_total = float(od_by_mode["pt_bike"].to_numpy(dtype=float)[internal_to_internal].sum()) if "pt_bike" in od_by_mode else 0.0
    bike_total = float(od_by_mode["bike"].to_numpy(dtype=float)[internal_to_internal].sum()) if "bike" in od_by_mode else 0.0
    walk_total = float(od_by_mode["walk"].to_numpy(dtype=float)[internal_to_internal].sum()) if "walk" in od_by_mode else 0.0
    
    if detailed_pt:
        rows = [
            {"mode": "Car (Driver)", "trips": car_total},
            {"mode": "PT (Walk access)", "trips": pt_walk_total},
            {"mode": "PT (Bike access)", "trips": pt_bike_total},
            {"mode": "Bicycle", "trips": bike_total},
            {"mode": "Walking", "trips": walk_total},
        ]
    else:
        rows = [
            {"mode": "Car (Driver)", "trips": car_total},
            {"mode": "Public Transport", "trips": pt_walk_total + pt_bike_total},
            {"mode": "Bicycle", "trips": bike_total},
            {"mode": "Walking", "trips": walk_total},
        ]

    df = pd.DataFrame(rows)
    total_trips = float(df["trips"].sum())
    df["share"] = df["trips"] / max(total_trips, 1e-9)
    return df


def corridor_od_summary(
    context: TransportContext,
    mode_result: ModeChoiceResult,
    pairs: list[tuple[str, str]],
    macro_regions: dict[str, list[str]] | None = None,
    *,
    detailed_pt: bool = False,
) -> pd.DataFrame:
    """
    Summarize total passenger trips and modal split shares for specific macro-region or municipality pairs.
    
    Parameters:
        detailed_pt: If False (default), outputs 4-mode shares (Car, PT, Bike, Walk).
                     If True, splits PT into pt_walk and pt_bike.
    """
    zones = context.zones
    # If macro_regions is provided and not empty, map by region; otherwise map directly by municipality name
    if macro_regions:
        first_list = next(iter(macro_regions.values()), [])
        if first_list and any(str(x).isdigit() and len(str(x)) >= 6 for x in first_list):
            # Dict maps region name -> list of zone IDs
            zone_to_region = {str(z): r for r, z_list in macro_regions.items() for z in z_list}
        else:
            # Dict maps region name -> list of municipality names
            muni_to_region = {m: r for r, munis in macro_regions.items() for m in munis}
            zone_to_region = (
                zones.set_index("grid_id")["municipality_name"]
                .astype(str)
                .map(muni_to_region)
                .fillna("Rest of Canton")
                .to_dict()
            )
    else:
        # Direct municipality-level OD mapping
        zone_to_region = zones.set_index("grid_id")["municipality_name"].astype(str).to_dict()

    od_by_mode = mode_result.od_by_mode
    reference = next(iter(od_by_mode.values()))
    labels = reference.index.astype(str)
    
    origin_regions = np.array([zone_to_region.get(str(z), "Other") for z in labels])
    dest_regions = np.array([zone_to_region.get(str(z), "Other") for z in labels])
    
    car_mat = od_by_mode["drive"].to_numpy(dtype=float) if "drive" in od_by_mode else np.zeros((len(labels), len(labels)))
    pt_walk_mat = od_by_mode["pt_walk"].to_numpy(dtype=float) if "pt_walk" in od_by_mode else np.zeros((len(labels), len(labels)))
    pt_bike_mat = od_by_mode["pt_bike"].to_numpy(dtype=float) if "pt_bike" in od_by_mode else np.zeros((len(labels), len(labels)))
    bike_mat = od_by_mode["bike"].to_numpy(dtype=float) if "bike" in od_by_mode else np.zeros((len(labels), len(labels)))
    walk_mat = od_by_mode["walk"].to_numpy(dtype=float) if "walk" in od_by_mode else np.zeros((len(labels), len(labels)))
    
    rows = []
    for orig, dest in pairs:
        mask = (origin_regions[:, None] == orig) & (dest_regions[None, :] == dest)
        c_trips = float(car_mat[mask].sum())
        pw_trips = float(pt_walk_mat[mask].sum())
        pb_trips = float(pt_bike_mat[mask].sum())
        b_trips = float(bike_mat[mask].sum())
        w_trips = float(walk_mat[mask].sum())
        tot = c_trips + pw_trips + pb_trips + b_trips + w_trips
        
        if detailed_pt:
            rows.append({
                "OD relation": f"{orig} → {dest}",
                "total_trips": tot,
                "car_trips": c_trips,
                "pt_walk_trips": pw_trips,
                "pt_bike_trips": pb_trips,
                "bike_trips": b_trips,
                "walk_trips": w_trips,
                "car_share": c_trips / max(tot, 1e-9),
                "pt_walk_share": pw_trips / max(tot, 1e-9),
                "pt_bike_share": pb_trips / max(tot, 1e-9),
                "bike_share": b_trips / max(tot, 1e-9),
                "walk_share": w_trips / max(tot, 1e-9),
            })
        else:
            p_trips = pw_trips + pb_trips
            rows.append({
                "OD relation": f"{orig} → {dest}",
                "total_trips": tot,
                "car_trips": c_trips,
                "pt_trips": p_trips,
                "bike_trips": b_trips,
                "walk_trips": w_trips,
                "car_share": c_trips / max(tot, 1e-9),
                "pt_share": p_trips / max(tot, 1e-9),
                "bike_share": b_trips / max(tot, 1e-9),
                "walk_share": w_trips / max(tot, 1e-9),
            })
        
    return pd.DataFrame(rows)


# =============================================================================
# 11. MULTI-STAGE HORIZON PERFORMANCE COMPARISON HELPER
# =============================================================================

def compare_stages_summary(
    context: TransportContext,
    stage_specs: Mapping[int, Mapping[str, Any]] | Any,
    corridor_zone_ids: list[str] | set[str] | None = None,
    corridor_municipalities: list[str] | None = None,
    *,
    stage_results: dict[int, Any] | None = None,
    g_d_base: float | None = None,
    n_years: int | None = None,
    stage_names: dict[int, str] | None = None,
    display_tables: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compare physical indicators across infrastructure stages for Year 1 and Year 40.
    
    Generates and optionally displays the 4 standard mode-specific performance tables:
      1. 🚗 Car (Driver) Performance
      2. 🚆 Public Transport Performance
      3. 🚲 Active Mobility (Bike & Walk) Performance
      4. 🌐 Overall Multimodal System Summary
    """
    import simulation_engine as m
    import parameters as p
    from IPython.display import display

    # 1. Resolve stage specifications
    if hasattr(stage_specs, "get_stages"):
        specs = stage_specs.get_stages(vars(p))
    elif isinstance(stage_specs, dict):
        specs = stage_specs
    else:
        specs = {0: {"name": "Stage 0"}, 1: {"name": "Stage 1"}, 2: {"name": "Stage 2"}}

    stage_keys = sorted(specs.keys())

    # 2. Planning horizon growth
    ny = n_years if n_years is not None else getattr(p, "N_YEARS", 40)
    if g_d_base is not None:
        g_cum_40 = (1.0 + float(g_d_base)) ** (ny - 1) - 1.0
    elif hasattr(p, "DEMAND_GROWTH_Y40"):
        g_cum_40 = float(p.DEMAND_GROWTH_Y40)
    else:
        g_cum_40 = (1.0 + getattr(p, "G_D_BASE", 0.006)) ** (ny - 1) - 1.0
    scale_y1 = 1.0
    scale_y40 = 1.0 + g_cum_40

    if stage_names is None:
        stage_names = {}

    car_data = []
    pt_data = []
    active_data = []
    system_data = []

    c_munis = corridor_municipalities if corridor_municipalities is not None else getattr(p, "CORRIDOR_MUNICIPALITIES", None)

    for s in stage_keys:
        st_name = stage_names.get(s, specs[s].get("name", f"Stage {s}"))
        
        # Retrieve stage metrics
        min_dist_km = specs[s].get("_min_distance_km", 0.0) if s in specs else 0.0
        
        if stage_results is not None and s in stage_results and stage_results[s] is not None:
            res_s = stage_results[s]
            metrics = extract_corridor_metrics(context, res_s, corridor_zone_ids=corridor_zone_ids, min_distance_km=min_dist_km)
        else:
            res_s = run_transport_mode_choice(
                context,
                stage=s,
                stage_specs=specs,
                corridor_zone_ids=corridor_zone_ids,
                corridor_municipalities=c_munis,
            )
            metrics = extract_corridor_metrics(context, res_s, corridor_zone_ids=corridor_zone_ids, min_distance_km=min_dist_km)

        # Physical simulations via simulation_engine (with full MSA support)
        y1 = m.simulate_year(metrics, stage=s, year_idx=0, g_cum=0.0,
                             context=context, mode_result=res_s, corridor_municipalities=c_munis)
        y40 = m.simulate_year(metrics, stage=s, year_idx=ny - 1, g_cum=g_cum_40,
                              context=context, mode_result=res_s, corridor_municipalities=c_munis)

        # Peak hour trips
        car_y1 = metrics.get("car_trips", 0.0)
        pt_y1 = metrics.get("pt_trips", 0.0)
        bike_y1 = metrics.get("bike_trips", 0.0)
        walk_y1 = metrics.get("walk_trips", 0.0)
        total_y1 = metrics.get("total_trips", 0.0)

        car_y40 = car_y1 * scale_y40
        pt_y40 = pt_y1 * scale_y40
        bike_y40 = bike_y1 * scale_y40
        walk_y40 = walk_y1 * scale_y40
        total_y40 = total_y1 * scale_y40

        # Delay & CO2 (peak hour)
        p2a = getattr(p, "PEAK_TO_ANNUAL", 1200.0)
        delay_y1_peak = y1.get("congestion_delay_hours", 0.0) / p2a
        delay_y40_peak = y40.get("congestion_delay_hours", 0.0) / p2a
        co2_y1_peak = y1.get("co2_tonnes", 0.0) / p2a
        co2_y40_peak = y40.get("co2_tonnes", 0.0) / p2a

        # Mode Travel Times
        car_tt_y1 = ((metrics.get("car_tt_hours", 0.0) * scale_y1 + delay_y1_peak) / max(car_y1, 1e-6)) * 60.0
        car_tt_y40 = ((metrics.get("car_tt_hours", 0.0) * scale_y40 + delay_y40_peak) / max(car_y40, 1e-6)) * 60.0
        pt_tt_y1 = (metrics.get("pt_tt_hours", 0.0) / max(pt_y1, 1e-6)) * 60.0
        pt_tt_y40 = pt_tt_y1

        # --- Mode 1: Car ---
        car_data.append({
            "Stage": st_name,
            "Year 1 Trips": car_y1,
            "Year 40 Trips": car_y40,
            "Trip Share": car_y1 / max(total_y1, 1e-9),
            "PKM Share (>5km)": metrics.get("car_share", car_y1 / max(total_y1, 1e-9)),
            "Year 1 TT": car_tt_y1,
            "Year 40 TT": car_tt_y40,
            "Year 1 Delay": delay_y1_peak,
            "Year 40 Delay": delay_y40_peak,
            "Year 1 CO2": co2_y1_peak,
            "Year 40 CO2": co2_y40_peak,
        })

        # --- Mode 2: Public Transport ---
        pt_data.append({
            "Stage": st_name,
            "Year 1 Trips": pt_y1,
            "Year 40 Trips": pt_y40,
            "Trip Share": pt_y1 / max(total_y1, 1e-9),
            "PKM Share (>5km)": metrics.get("pt_share", pt_y1 / max(total_y1, 1e-9)),
            "Year 1 TT": pt_tt_y1,
            "Year 40 TT": pt_tt_y40,
        })

        # --- Mode 3: Active Mobility (Bike & Walk) ---
        active_data.append({
            "Stage": st_name,
            "Year 1 Bike Trips": bike_y1,
            "Year 40 Bike Trips": bike_y40,
            "Bike Trip Share": bike_y1 / max(total_y1, 1e-9),
            "Bike PKM Share": metrics.get("bike_share", bike_y1 / max(total_y1, 1e-9)),
            "Year 1 Walk Trips": walk_y1,
            "Year 40 Walk Trips": walk_y40,
            "Walk Trip Share": walk_y1 / max(total_y1, 1e-9),
            "Walk PKM Share": metrics.get("walk_share", walk_y1 / max(total_y1, 1e-9)),
        })

        # --- Mode 4: Overall System ---
        system_data.append({
            "Stage": st_name,
            "Year 1 Total Demand": total_y1,
            "Year 40 Total Demand": total_y40,
            "Year 1 System Avg TT": y1.get("avg_tt_min", 0.0),
            "Year 40 System Avg TT": y40.get("avg_tt_min", 0.0),
            "Year 1 Peak Delay": delay_y1_peak,
            "Year 40 Peak Delay": delay_y40_peak,
            "Year 1 Peak CO2": co2_y1_peak,
            "Year 40 Peak CO2": co2_y40_peak,
        })

    # Convert to DataFrames
    df_car = pd.DataFrame(car_data).set_index("Stage")
    df_pt = pd.DataFrame(pt_data).set_index("Stage")
    df_active = pd.DataFrame(active_data).set_index("Stage")
    df_system = pd.DataFrame(system_data).set_index("Stage")

    if display_tables:
        print("🚗 1. CAR (DRIVER) PERFORMANCE (Single Evening Peak Hour)")
        display(df_car.style.format({
            "Year 1 Trips": "{:,.0f} trips",
            "Year 40 Trips": "{:,.0f} trips",
            "Trip Share": "{:.1%}",
            "PKM Share (>5km)": "{:.1%}",
            "Year 1 TT": "{:.1f} min",
            "Year 40 TT": "{:.1f} min",
            "Year 1 Delay": "{:,.0f} h",
            "Year 40 Delay": "{:,.0f} h",
            "Year 1 CO2": "{:,.1f} t",
            "Year 40 CO2": "{:,.1f} t",
        }))

        print("\n🚆 2. PUBLIC TRANSPORT PERFORMANCE (Single Evening Peak Hour)")
        display(df_pt.style.format({
            "Year 1 Trips": "{:,.0f} trips",
            "Year 40 Trips": "{:,.0f} trips",
            "Trip Share": "{:.1%}",
            "PKM Share (>5km)": "{:.1%}",
            "Year 1 TT": "{:.1f} min",
            "Year 40 TT": "{:.1f} min",
        }))

        print("\n🚲 3. ACTIVE MOBILITY (BIKE & WALK) (Single Evening Peak Hour)")
        display(df_active.style.format({
            "Year 1 Bike Trips": "{:,.0f} trips",
            "Year 40 Bike Trips": "{:,.0f} trips",
            "Bike Trip Share": "{:.1%}",
            "Bike PKM Share": "{:.1%}",
            "Year 1 Walk Trips": "{:,.0f} trips",
            "Year 40 Walk Trips": "{:,.0f} trips",
            "Walk Trip Share": "{:.1%}",
            "Walk PKM Share": "{:.1%}",
        }))

        print("\n🌐 4. OVERALL MULTIMODAL SYSTEM SUMMARY (Single Evening Peak Hour)")
        display(df_system.style.format({
            "Year 1 Total Demand": "{:,.0f} trips",
            "Year 40 Total Demand": "{:,.0f} trips",
            "Year 1 System Avg TT": "{:.1f} min",
            "Year 40 System Avg TT": "{:.1f} min",
            "Year 1 Peak Delay": "{:,.0f} h",
            "Year 40 Peak Delay": "{:,.0f} h",
            "Year 1 Peak CO2": "{:,.1f} t",
            "Year 40 Peak CO2": "{:,.1f} t",
        }))

    return df_car, df_pt, df_active, df_system
