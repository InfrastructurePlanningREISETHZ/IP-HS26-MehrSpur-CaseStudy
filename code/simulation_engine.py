"""
cba.py
=============
MehrSpur – core economic evaluation engine.

WHAT THIS FILE DOES
-------------------
  1. Uncertainty paths       demand_growth_path
  2. Nuisance parameters     sample_perturbed_params
  3. Annual cost calculator  annual_mode_costs
  4. Single-year simulator   simulate_year
  5. NPC calculator          npc_by_component

HOW IT FITS IN THE PROJECT
--------------------------
    parameters.py      ──►┐
    transport_model_interface ──►├──  cba_engine.py  ──►  notebooks (EMA Workbench)
    pathways.py        ──►┘
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

from parameters import (
    N_YEARS, DISCOUNT_RATE,
    FIXED_PARAMS, PERTURBABLE_PARAMS, PERTURBABLE_SD_FRACTION,
    STRUCTURAL_UNCERTAINTIES,
)


def _z_score(u: float) -> float:
    """Uniform[0,1] -> standard normal draw."""
    u_safe = min(max(u, 1e-6), 1 - 1e-6)
    return norm.ppf(u_safe)


NOMINAL_PARAMS = {**FIXED_PARAMS, **PERTURBABLE_PARAMS}

# =============================================================================
# 1. CORE ECONOMIC & PHYSICAL FORMULAS (STUDENT-EDITABLE)
# =============================================================================
# ⚠️ STUDENTS: This is the engine room of the Cost-Benefit Analysis (CBA).
# If you want to change how CO2 costs are calculated, add a new mode (e.g. E-Bikes),
# or change the crowding penalty function, modify these two functions!

def annual_mode_costs(base_metrics: dict, year_idx: int, g_cum: float, stage: int, params: dict | None = None, extra_delay: float = 0.0) -> dict:
    """
    Return annual societal costs (CHF) for one year, broken out by mode and NIBA categories.

    base_metrics : Aggregated transport metrics from the FSM for the active stage.
                   Expected keys: car_tt_hours, car_dist_km, pt_tt_hours
    year_idx     : 0-based year index (year_idx=0 → Year 1 of the horizon)
    g_cum        : Cumulative demand growth factor for this year.
    stage        : The active infrastructure stage.
    params       : Nuisance parameters.
    extra_delay  : True congestion delay loaded from the LUT for this demand level.
    """
    if params is None:
        params = NOMINAL_PARAMS

    scale = 1.0 + g_cum
    c_co2_t = params["C_CO2"] + params.get("C_CO2_GROWTH", 0.0) * (year_idx + 1)
    
    # PEAK_TO_ANNUAL is now loaded dynamically from parameters.py
    p2a = params.get("PEAK_TO_ANNUAL", 1200.0)

    # ── Car: travel time + fuel + CO2 + Noise + Air + Accidents ──────────
    tt_car = (base_metrics.get("car_tt_hours", 0) * scale + extra_delay) * p2a
    d_car = base_metrics.get("car_dist_km", 0) * scale * p2a
    
    car_time_cost = tt_car * params["C_TT_CAR"]
    car_fuel_cost = params.get("F_FUEL", 0.04) * params.get("C_FUEL", 1.25) * d_car
    car_co2_cost = d_car * params.get("P_CO2", 0.139) * c_co2_t
    car_noise_cost = d_car * params.get("C_NOISE_CAR", 0.015)
    car_air_cost = d_car * params.get("C_AIR_CAR", 0.018)
    car_acc_cost = d_car * params.get("C_ACCIDENT_CAR", 0.084)
    
    car_cost = car_time_cost + car_fuel_cost + car_co2_cost + car_noise_cost + car_air_cost + car_acc_cost

    # ── PT: in-vehicle travel time + wait time + crowding penalty ────────
    tt_pt = base_metrics.get("pt_tt_hours", 0) * scale * p2a
    pt_trips = base_metrics.get("pt_trips", 0) * scale * p2a
    pt_trips_peak = base_metrics.get("pt_trips", 0) * scale # Peak hour trips
    
    # Get stage-specific headway and capacity dynamically (supports N stages!)
    headway = params.get(f"HEADWAY_STAGE{stage}", params.get("HEADWAY_STAGE0", 20.0))
    capacity = params.get(f"CAPACITY_STAGE{stage}", params.get("CAPACITY_STAGE0", 75000))
        
    wait_time_hours = min(headway / 2.0, 5.0) / 60.0
    total_pt_time = tt_pt + (pt_trips * wait_time_hours)
    
    # Crowding Penalty (NIBA overload proxy): quadratic penalty if capacity exceeded
    crowding_multiplier = max(1.0, (pt_trips_peak / max(capacity, 1.0))**2)
    
    pt_time_cost = total_pt_time * params["C_TT_PT"] * crowding_multiplier
    pt_cost = pt_time_cost

    return {
        "car": car_cost, 
        "pt": pt_cost,
        "car_time_cost": car_time_cost,
        "car_fuel_cost": car_fuel_cost,
        "car_co2_cost": car_co2_cost,
        "car_noise_cost": car_noise_cost,
        "car_air_cost": car_air_cost,
        "car_acc_cost": car_acc_cost,
        "pt_time_cost": pt_time_cost
    }


def annual_physical_indicators(base_metrics: dict, g_cum: float, params: dict | None = None, extra_delay: float = 0.0) -> dict:
    """Physical (non-monetised) societal indicators for one year."""
    if params is None:
        params = NOMINAL_PARAMS

    scale = 1.0 + g_cum
    p2a = params.get("PEAK_TO_ANNUAL", 1200.0)
    
    d_car = base_metrics.get("car_dist_km", 0) * scale * p2a
    co2_kg = d_car * params["P_CO2"]

    tt_car = (base_metrics.get("car_tt_hours", 0) * scale + extra_delay) * p2a
    tt_pt = base_metrics.get("pt_tt_hours", 0) * scale * p2a
    
    # Motorized trips only (exclude walking/biking so travel times reflect vehicular travel)
    car_trips = base_metrics.get("car_trips", 0) * scale * p2a
    pt_trips = base_metrics.get("pt_trips", 0) * scale * p2a
    motorized_trips = car_trips + pt_trips
    if motorized_trips <= 0:
        motorized_trips = (base_metrics.get("total_trips", 1) * scale * p2a) * (base_metrics.get("car_share", 0.0) + base_metrics.get("pt_share", 0.0))

    avg_tt_min = ((tt_car + tt_pt) / max(motorized_trips, 1.0)) * 60.0

    return {
        "co2_tonnes": co2_kg / 1000,
        "avg_tt_min": avg_tt_min,
        "congestion_delay_hours": extra_delay * p2a,
    }

# =============================================================================
# 2. INTERNAL ENGINE LOGIC (Uncertainty Trajectories & Runners)
# =============================================================================

def generate_trajectory(u: float, spec: dict) -> np.ndarray:
    """
    Project-agnostic 40-year trajectory generator from a uniform draw u in [0, 1].
    Linearly ramps the mean trend from nominal_start to nominal_end,
    and widens the standard deviation cone from sigma_start to sigma_end.
    """
    t = np.arange(1, N_YEARS + 1)
    mu = np.linspace(spec.get("nominal_start", 0.0), spec.get("nominal_end", 0.0), N_YEARS)
    sigma = np.linspace(spec.get("sigma_start", 0.005), spec.get("sigma_end", 0.20), N_YEARS)
    z = _z_score(u)
    traj = mu + z * sigma
    floor = spec.get("floor", None)
    if floor is not None:
        traj = np.maximum(traj, float(floor))
    return traj


def get_trajectory(name: str, u: float = 0.5) -> np.ndarray:
    """Retrieve 40-year trajectory for any structural uncertainty defined in parameters.py."""
    if name not in STRUCTURAL_UNCERTAINTIES:
        raise KeyError(f"Unknown structural uncertainty: {name}. Registered: {list(STRUCTURAL_UNCERTAINTIES.keys())}")
    return generate_trajectory(u, STRUCTURAL_UNCERTAINTIES[name])


def demand_growth_path(u: float) -> np.ndarray:
    """Modular helper for demand growth trajectory."""
    return get_trajectory("u_demand", u)


def pt_affinity_path(u: float) -> np.ndarray:
    """Modular helper for PT preference trajectory."""
    return get_trajectory("u_beta_pt", u)


def get_ema_uncertainties(include_nuisance: bool = True) -> list:
    """
    Build the list of EMA Workbench RealParameter objects dynamically
    from the registries defined in parameters.py.
    
    ⚠️ STUDENTS: You do NOT need to modify this function! 
    It automatically reads from `PERTURBABLE_PARAMS` in parameters.py.
    If you want to add a new uncertainty (e.g. e-bike battery cost), 
    just add it to the dictionary in parameters.py and it will appear here.
    """
    from ema_workbench import RealParameter
    uncertainties = [RealParameter(k, 0.0, 1.0) for k in STRUCTURAL_UNCERTAINTIES.keys()]
    if include_nuisance:
        for name in PERTURBABLE_PARAMS.keys():
            uncertainties.append(RealParameter(f"u_{name}", 0.0, 1.0))
    return uncertainties


def sample_perturbed_params(u_kwargs: dict | None = None) -> dict:
    """
    Transform per-parameter Uniform[0,1] draws into Normal(nominal, 10%)
    realizations for every PERTURBABLE parameter.
    """
    u_kwargs = u_kwargs or {}
    out = dict(FIXED_PARAMS)
    for name, nominal in PERTURBABLE_PARAMS.items():
        u = u_kwargs.get(f"u_{name}", 0.5)
        z = _z_score(u)
        out[name] = nominal + z * abs(nominal) * PERTURBABLE_SD_FRACTION
    return out


def get_lut_delay(stage: int, scale: float) -> float:
    """Auto-loads the Transparent LUT for the active stage and interpolates the true congestion delay."""
    project_root = Path(__file__).resolve().parent.parent
    lut_path = project_root / "data" / "processed" / f"delay_lut_stage_{stage}.json"
    if not lut_path.exists():
        # Fallback to local cwd relative path if run in a different working directory
        lut_path = Path(f"data/processed/delay_lut_stage_{stage}.json")
        if not lut_path.exists():
            return 0.0  # Linear baseline fallback if Notebook 2 hasn't run yet
    try:
        with open(lut_path, "r") as f:
            lut = json.load(f)
        scales = sorted([float(k) for k in lut.keys()])
        
        delays = []
        for k in scales:
            val = lut[str(k)]
            if isinstance(val, dict):
                delays.append(float(val.get("delay_hours", 0.0)))
            else:
                delays.append(float(val))
                
        return float(np.interp(scale, scales, delays))
    except Exception:
        return 0.0


def simulate_year(
    base_metrics: dict, 
    stage: int, 
    year_idx: int, 
    g_cum: float, 
    params: dict | None = None,
    context=None,
    mode_result=None,
    corridor_municipalities=None
) -> dict:
    """Simulate one year and return a results dict."""
    if params is None:
        params = NOMINAL_PARAMS
        
    scale = 1.0 + g_cum
    p2a = params.get("PEAK_TO_ANNUAL", 1200.0)
    total_demand = base_metrics.get("total_trips", 0) * scale * p2a
    
    try:
        from transport_model_interface import (
            ASSIGNMENT_SETTINGS,
            build_corridor_context,
            extract_corridor_metrics,
            run_coupled_corridor_assignment,
            run_route_assignment,
        )
        use_msa = (ASSIGNMENT_SETTINGS.get("method") == "MSA")
    except ImportError:
        use_msa = False

    if use_msa:
        if not (context and mode_result and corridor_municipalities):
            raise ValueError(
                "MSA is enabled in ASSIGNMENT_SETTINGS, but `context`, `mode_result`, "
                "or `corridor_municipalities` were not passed to simulate_year(). "
                "To run the pathways engine with MSA, you must pass these objects."
            )
        if bool(ASSIGNMENT_SETTINGS.get("modal_feedback", True)):
            corridor = build_corridor_context(
                context,
                corridor_municipalities=list(corridor_municipalities),
                name="configured project corridor",
            )
            assignment = run_coupled_corridor_assignment(
                context,
                corridor,
                mode_result,
                demand_multiplier=scale,
                drive_occupancy=ASSIGNMENT_SETTINGS.get("drive_occupancy", 1.14),
                max_iterations=ASSIGNMENT_SETTINGS.get("max_iterations", 8),
                min_iterations=ASSIGNMENT_SETTINGS.get("min_iterations", 2),
                relative_gap_threshold=ASSIGNMENT_SETTINGS.get(
                    "relative_gap_threshold", 0.10
                ),
                od_threshold=ASSIGNMENT_SETTINGS.get("od_threshold", 1.0),
            )
            # The coupled result already contains this year's demand scale and
            # modal response. Downstream cost formulas must therefore receive
            # zero additional growth, otherwise demand would be scaled twice.
            base_metrics = extract_corridor_metrics(
                context,
                assignment.mode_result,
                corridor.zone_ids,
            )
            g_cum_for_metrics = 0.0
            total_demand = base_metrics.get("total_trips", 0.0) * p2a
            extra_delay = assignment.diagnostics.get("total_delay_hours", 0.0)
        else:
            # Compatibility path for experiments that intentionally hold modal
            # split fixed while scaling road demand.
            _, _, _, meta = run_route_assignment(
                context,
                mode_result,
                corridor_municipalities=corridor_municipalities,
                demand_multiplier=scale,
                iterations=ASSIGNMENT_SETTINGS.get("max_iterations", 8),
                return_skims=True,
                generate_lut=False,
            )
            extra_delay = meta.get("total_delay_hours", 0.0)
            g_cum_for_metrics = g_cum
    else:
        extra_delay = get_lut_delay(stage, scale)
        if "congestion_delay_hours" in base_metrics and extra_delay == 0.0:
            extra_delay = base_metrics["congestion_delay_hours"]
        g_cum_for_metrics = g_cum

    costs = annual_mode_costs(
        base_metrics, year_idx, g_cum_for_metrics, stage, params, extra_delay
    )
    phys = annual_physical_indicators(
        base_metrics, g_cum_for_metrics, params, extra_delay
    )

    return {
        "year_idx":               year_idx,
        "stage":                  stage,
        "total_demand":           total_demand,
        "car_cost":               costs["car"],
        "car_time_cost":          costs["car_time_cost"],
        "car_fuel_cost":          costs["car_fuel_cost"],
        "car_co2_cost":           costs["car_co2_cost"],
        "car_noise_cost":         costs["car_noise_cost"],
        "car_air_cost":           costs["car_air_cost"],
        "car_acc_cost":           costs["car_acc_cost"],
        "pt_cost":                costs["pt"],
        "pt_time_cost":           costs["pt_time_cost"],
        "co2_tonnes":             phys["co2_tonnes"],
        "avg_tt_min":             phys["avg_tt_min"],
        "congestion_delay_hours": extra_delay * p2a,
    }


def npc_by_component(results: pd.DataFrame, baseline_results: pd.DataFrame = None, discount_rate: float = DISCOUNT_RATE) -> dict:
    """Discounted Net Present Cost, broken out by component (millions CHF) matching NIBA."""
    df = results.copy()

    if baseline_results is not None:
        base = baseline_results.copy()
        
        # Rule of a Half: Benefit = 0.5 * (Trips_base + Trips_proj) * (Cost_per_trip_base - Cost_per_trip_proj)
        # Avoid division by zero by replacing 0 with 1
        car_trips_base = base["annual_car_trips"].replace(0, 1)
        car_trips_proj = df["annual_car_trips"].replace(0, 1)
        pt_trips_base = base["annual_pt_trips"].replace(0, 1)
        pt_trips_proj = df["annual_pt_trips"].replace(0, 1)
        
        car_cpt_base = base["car_time_cost"] / car_trips_base
        car_cpt_proj = df["car_time_cost"] / car_trips_proj
        pt_cpt_base = base["pt_time_cost"] / pt_trips_base
        pt_cpt_proj = df["pt_time_cost"] / pt_trips_proj
        
        car_time_benefit = 0.5 * (base["annual_car_trips"] + df["annual_car_trips"]) * (car_cpt_base - car_cpt_proj)
        pt_time_benefit = 0.5 * (base["annual_pt_trips"] + df["annual_pt_trips"]) * (pt_cpt_base - pt_cpt_proj)
        
        # A positive benefit implies a negative cost in the NPC framework
        car_time_diff = -car_time_benefit
        pt_time_diff = -pt_time_benefit
        
        df["car_cost"] = df["car_cost"] - df["car_time_cost"] + car_time_diff
        df["pt_cost"] = df["pt_cost"] - df["pt_time_cost"] + pt_time_diff
        if "total_cost" in df.columns:
            df["total_cost"] = df["total_cost"] - df["car_time_cost"] - df["pt_time_cost"] + car_time_diff + pt_time_diff
            
        df["car_time_cost"] = car_time_diff
        df["pt_time_cost"] = pt_time_diff

    df["d"] = 1 / (1 + discount_rate) ** (df["year_idx"] + 1)
    s = 1e6   # convert CHF → MCHF
    
    # Optional fields might not be present if results are from an older run
    def get_sum(col: str) -> float:
        if col in df.columns:
            return (df[col] * df["d"]).sum() / s
        return 0.0

    return {
        "car":   get_sum("car_cost"),
        "car_time": get_sum("car_time_cost"),
        "car_fuel": get_sum("car_fuel_cost"),
        "car_co2":  get_sum("car_co2_cost"),     # NIBA 6.1
        "car_noise": get_sum("car_noise_cost"),  # NIBA 2.1
        "car_air":  get_sum("car_air_cost"),     # NIBA 1.1
        "car_acc":  get_sum("car_acc_cost"),     # NIBA 20.1
        "pt":    get_sum("pt_cost"),
        "pt_time": get_sum("pt_time_cost"),      # NIBA 11.1
        "inv":   get_sum("inv_cost"),            # NIBA 10.6
        "op":    get_sum("op_cost"),             # NIBA 10.5
        "total": get_sum("total_cost"),
    }
