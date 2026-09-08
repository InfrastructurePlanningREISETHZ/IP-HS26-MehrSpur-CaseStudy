"""
pathways.py
===========
Planning policy (strategy) definitions and the adaptive-pathway simulation engine.

This file contains BOTH the user-friendly definitions of the 9 pathways (at the top)
and the more complex simulation engine logic that runs them (at the bottom).
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from parameters import (
    N_YEARS,
    STRUCTURAL_UNCERTAINTIES,
    PERTURBABLE_PARAMS,
    FIXED_PARAMS,
    NOMINAL_PARAMS,
    MAX_AVG_TT,
    PT_SHARE_TARGET,
)
import stages
import simulation_engine as m


# =============================================================================
# 1. ADAPTIVE TRIGGERS (STUDENT-EDITABLE)
# =============================================================================
# Triggers are OPERATIONAL ACTION THRESHOLDS that initiate building a stage:
#   - Signpost: the real-time operational metric being monitored
#   - Threshold: the value that indicates the current system needs an upgrade
#   - Persistence: consecutive years the threshold must be exceeded before deciding
#   - Lead time: construction/implementation duration from decision to opening

# VALID SIGNPOSTS (Supported variables for adaptive triggers):
#   "pt_trips"               : Peak-Hour PT Demand (trips)
#   "annual_pt_trips"        : Annual PT Demand (trips)
#   "pt_share"               : Public Transport Mode Share (e.g. 0.25)
#   "avg_tt_min"             : Average Corridor Travel Time (min)
#   "congestion_delay_hours" : Total Congestion Delay (hours)

DEFAULT_TRIGGERS = {
    # Trigger 1: Transit Demand Volume -> Activates Stage 1 (Local Stations & Access Package)
    "trigger1": {
        "name":            "Trigger 1 — Public Transport Demand Surge (Station Package)",
        "signpost":        "pt_trips",
        "signpost_label":  "Peak-Hour PT Demand (trips)",
        "threshold":       78_000,  # 78,000 peak trips (baseline starts at ~71,200 trips; triggers ~Year 12-14)
        "persistence":     2,       # 2 consecutive years above threshold
        "lead_time":       2,       # 2 years planning & station civil works
        "stage_activated": 1,
        "description":     "If peak-hour PT demand exceeds 78,000 trips for 2 years, deploy Stage 1 Station Package & Mobility Hubs."
    },
    # Trigger 2: Highway Congestion / Delay -> Activates Stage 2 (Brüttenertunnel)
    "trigger2": {
        "name":            "Trigger 2 — Network Travel Time & Bottleneck Delay (Core Tunnel)",
        "signpost":        "avg_tt_min",
        "signpost_label":  "Average Corridor Travel Time (min)",
        "threshold":       17.5,    # 17.5 mins (highway bottleneck delay reaching critical levels)
        "persistence":     2,       # 2 consecutive years above threshold
        "lead_time":       5,       # 5 years tunneling, fleet expansion & timetable integration
        "stage_activated": 2,
        "description":     "If average travel time exceeds 17.5 min for 2 years, construct Stage 2 Core Tunnel & 15-min rhythm."
    }
}


def get_triggers(params: dict | None = None) -> dict:
    """Return adaptive triggers (with optional parameter overrides)."""
    trig = deepcopy(DEFAULT_TRIGGERS)
    if params is not None:
        for t_key in ["trigger1", "trigger2"]:
            prefix = t_key.upper() + "_"
            if f"{prefix}THRESHOLD" in params:
                trig[t_key]["threshold"] = params[f"{prefix}THRESHOLD"]
            if f"{prefix}SIGNPOST" in params:
                trig[t_key]["signpost"] = params[f"{prefix}SIGNPOST"]
            if f"{prefix}PERSISTENCE" in params:
                trig[t_key]["persistence"] = params[f"{prefix}PERSISTENCE"]
            if f"{prefix}LEAD_TIME" in params:
                trig[t_key]["lead_time"] = params[f"{prefix}LEAD_TIME"]
    return trig


# Module-level trigger shortcuts
TRIGGER_1 = DEFAULT_TRIGGERS["trigger1"]
TRIGGER_2 = DEFAULT_TRIGGERS["trigger2"]


# =============================================================================
# 2. POLICY PATHWAY DEFINITIONS
# =============================================================================

def get_pathways(params: dict | None = None, triggers: dict | None = None) -> dict:
    """Return pathway definitions enriched with policy-compatible costs."""
    if params is None:
        params = NOMINAL_PARAMS
    if triggers is None:
        triggers = get_triggers(params)
        
    c_inv1 = params.get("C_INV_STAGE1", params.get("C_INV1", 925_000_000))
    c_op1  = params.get("C_OP_STAGE1", params.get("C_OP1", 15_000_000))
    c_inv2 = params.get("C_INV_STAGE2", params.get("C_INV2", 2_302_600_000))
    c_op2  = params.get("C_OP_STAGE2", params.get("C_OP2", 35_000_000))
    c_flex = params.get("C_FLEX", 20_000_000)

    trigger1 = triggers.get("trigger1", DEFAULT_TRIGGERS["trigger1"])
    trigger2 = triggers.get("trigger2", DEFAULT_TRIGGERS["trigger2"])

    PATHWAYS = {
        "baseline": {
            "name": "Baseline — Stage 0 throughout",
            "description": "No infrastructure investment for the full 40-year horizon.",
            "initial_stage": 0, "to1": None, "to2": None,
        },
        "static1": {
            "name": "Static 1 — Stage 1 from Year 1",
            "description": "Stage 1 built immediately and held fixed for the full horizon; never expands to Stage 2.",
            "initial_stage": 1, "to1": None, "to2": None,
        },
        "static2": {
            "name": "Static 2 — Stage 2 from Year 1",
            "description": "Full build (Stage 1+2) immediately, maximum capacity upfront.",
            "initial_stage": 2, "to1": None, "to2": None,
        },
        "staged1": {
            "name": "Staged 1 — Stage 1 in Year 1, Stage 2 in Year 10",
            "description": "Committed two-phase plan; Stage 2 arrives early regardless of demand.",
            "initial_stage": 1, "to1": None, "to2": {"type": "fixed", "year": 10},
        },
        "staged2": {
            "name": "Staged 2 — Stage 1 in Year 1, Stage 2 in Year 30",
            "description": "Committed two-phase plan; Stage 2 arrives late regardless of demand.",
            "initial_stage": 1, "to1": None, "to2": {"type": "fixed", "year": 30},
        },
        "staged3": {
            "name": "Staged 3 — Stage 1 in Year 10, Stage 2 in Year 30",
            "description": "Committed two-phase plan; both phases delayed from Year 1.",
            "initial_stage": 0, "to1": {"type": "fixed", "year": 10}, "to2": {"type": "fixed", "year": 30},
        },
        "flexible1": {
            "name": "Flexible 1 — Stage 1 via Trigger 1",
            "description": "Stage 1 only built once congestion pressure warrants it.",
            "initial_stage": 0, "to1": {"type": "trigger", **trigger1}, "to2": None,
        },
        "flexible2": {
            "name": "Flexible 2 — Stage 1 in Year 1, Stage 2 via Trigger 2",
            "description": "Stage 1 committed upfront; Stage 2 only if network travel time exceeds trigger threshold.",
            "initial_stage": 1, "to1": None, "to2": {"type": "trigger", **trigger2},
        },
        "flexible3": {
            "name": "Flexible 3 — Stage 1 via Trigger 1, Stage 2 via Trigger 2",
            "description": "Fully adaptive: both stages only built once their own trigger fires.",
            "initial_stage": 0, "to1": {"type": "trigger", **trigger1}, "to2": {"type": "trigger", **trigger2},
        },
    }

    pathways = deepcopy(PATHWAYS)
    for key, spec in pathways.items():
        initial_stage = spec["initial_stage"]
        option_premium_stage1 = (
            c_flex
            if spec["to1"] is not None and spec["to1"]["type"] == "trigger"
            else 0.0
        )
        option_premium_stage2 = (
            c_flex
            if spec["to2"] is not None and spec["to2"]["type"] == "trigger"
            else 0.0
        )

        spec.update({
            "stage2_adaptive": (
                spec["to2"] is not None and spec["to2"]["type"] == "trigger"
            ),
            "stage2_year": (
                spec["to2"]["year"]
                if spec["to2"] is not None and spec["to2"]["type"] == "fixed"
                else None
            ),
            "option_premium_stage1": option_premium_stage1,
            "option_premium_stage2": option_premium_stage2,
            "upfront_fee": option_premium_stage1 + option_premium_stage2,
            "inv_stage1": (
                c_inv1 if initial_stage >= 1 or spec["to1"] is not None else 0.0
            ),
            "op_stage1": (
                c_op1 if initial_stage >= 1 or spec["to1"] is not None else 0.0
            ),
            "inv_stage2": (
                c_inv2 if initial_stage >= 2 or spec["to2"] is not None else 0.0
            ),
            "op_stage2": (
                c_op2 if initial_stage >= 2 or spec["to2"] is not None else 0.0
            ),
        })

    return pathways


# Module-level defaults for convenience across notebooks
PATHWAYS = get_pathways(NOMINAL_PARAMS)
PATHWAY_NAMES = list(PATHWAYS.keys())


# =============================================================================
# 2. SHAPED UNCERTAINTY TRAJECTORIES
# =============================================================================

SHAPES = ("linear", "early", "late", "logistic", "random_walk", "almost_flat")

_OU_REVERSION = 0.35        # fraction of the gap to the trend pulled back each year
_OU_STEP_FRAC = 0.55        # per-year noise, as a fraction of the trend's local sigma
_LOGISTIC_STEEPNESS = 10.0  # higher = sharper, more S-shaped transition
_ALMOST_FLAT_FRACTION = 0.20  # 'almost_flat' reaches only this fraction of the normal change


def _shape_fraction(t_frac: np.ndarray, shape: str) -> np.ndarray:
    if shape == "linear":
        return t_frac
    if shape == "early":
        return t_frac ** 0.4
    if shape == "late":
        return t_frac ** 2.5
    if shape == "logistic":
        raw = 1.0 / (1.0 + np.exp(-_LOGISTIC_STEEPNESS * (t_frac - 0.5)))
        raw0 = 1.0 / (1.0 + np.exp(_LOGISTIC_STEEPNESS * 0.5))
        raw1 = 1.0 / (1.0 + np.exp(-_LOGISTIC_STEEPNESS * 0.5))
        return (raw - raw0) / (raw1 - raw0)          # normalised so f(0)=0, f(1)=1 exactly
    if shape == "almost_flat":
        return _ALMOST_FLAT_FRACTION * t_frac
    raise ValueError(f"{shape!r} is not an interpolated shape")


def _random_walk_trajectory(u: float, mu1: float, mu40: float, sigma1: float, sigma40: float) -> np.ndarray:
    seed = int(u * (2**31 - 1))
    rng = np.random.default_rng(seed)
    t_frac = np.linspace(0.0, 1.0, N_YEARS)
    mu_trend = mu1 + t_frac * (mu40 - mu1)
    sigma_trend = sigma1 + t_frac * (sigma40 - sigma1)
    x = np.empty(N_YEARS)
    x[0] = mu_trend[0] + rng.normal(0.0, sigma_trend[0])
    for t in range(1, N_YEARS):
        step = rng.normal(0.0, _OU_STEP_FRAC * sigma_trend[t])
        x[t] = x[t - 1] + _OU_REVERSION * (mu_trend[t] - x[t - 1]) + step
    return x


def pt_affinity_trajectory(u: float, shape: str = "linear") -> np.ndarray:
    """PT-preference trajectory following `shape`."""
    spec = STRUCTURAL_UNCERTAINTIES["u_beta_pt"]
    mu1, mu40 = spec["nominal_start"], spec["nominal_end"]
    sig1, sig40 = spec["sigma_start"], spec["sigma_end"]
    if shape == "random_walk":
        return _random_walk_trajectory(u, mu1, mu40, sig1, sig40)
    t_frac = np.linspace(0.0, 1.0, N_YEARS)
    f = _shape_fraction(t_frac, shape)
    mu = mu1 + f * (mu40 - mu1)
    sigma = sig1 + f * (sig40 - sig1)
    z = m._z_score(u)
    traj = mu + z * sigma
    return np.maximum(traj, float(spec["floor"]))


def demand_growth_trajectory(u: float, shape: str = "linear") -> np.ndarray:
    """Cumulative demand-growth trajectory following `shape`."""
    spec = STRUCTURAL_UNCERTAINTIES["u_demand"]
    mu1, mu40 = spec["nominal_start"], spec["nominal_end"]
    sig1, sig40 = spec["sigma_start"], spec["sigma_end"]
    if shape == "random_walk":
        return _random_walk_trajectory(u, mu1, mu40, sig1, sig40)
    t_frac = np.linspace(0.0, 1.0, N_YEARS)
    f = _shape_fraction(t_frac, shape)
    mu = mu1 + f * (mu40 - mu1)
    sigma = sig1 + f * (sig40 - sig1)
    z = m._z_score(u)
    traj = mu + z * sigma
    return np.maximum(traj, float(spec["floor"]))


# =============================================================================
# 3. THE PATHWAY RUNNER
# =============================================================================

def run_pathway(pathway_def: str | dict, stage_metrics: dict, u_demand: float,
                 demand_shape: str = "linear", params: dict | None = None,
                 u_beta_pt: float | None = None, beta_shape: str = "linear",
                 context=None, mode_results_by_stage: dict | None = None, corridor_municipalities: list | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Simulate one pathway for one scenario over the full 40-year horizon.
    Requires pre-calculated `stage_metrics` from the FSM for the active scenario.
    """
    g_traj = demand_growth_trajectory(u_demand, demand_shape)
    pt_traj = pt_affinity_trajectory(u_beta_pt, beta_shape) if u_beta_pt is not None else None
    return run_pathway_from_trajectories(pathway_def, stage_metrics, g_traj, params, pt_traj=pt_traj,
                                         context=context, mode_results_by_stage=mode_results_by_stage, corridor_municipalities=corridor_municipalities)


def run_pathway_from_trajectories(pathway_def: str | dict, stage_metrics: dict, g_traj: np.ndarray,
                                   params: dict | None = None,
                                   pt_traj: np.ndarray | None = None,
                                   context=None, mode_results_by_stage: dict | None = None, corridor_municipalities: list | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Same as run_pathway(), but takes ready-made 40-year `g_traj` and optional `pt_traj`.
    """
    if params is None:
        params = m.NOMINAL_PARAMS
        
    if isinstance(pathway_def, str):
        pathway_def = get_pathways(params)[pathway_def]
    
    to1, to2 = pathway_def["to1"], pathway_def["to2"]

    stage = pathway_def["initial_stage"]
    decision1_year = activation1_year = None
    decision2_year = activation2_year = None
    persist1 = persist2 = 0
    prev_tt = None
    prev_pt_share = None
    prev_pt_trips = None
    prev_delay = None

    records = []

    for t in range(N_YEARS):
        year = t + 1

        # ---- fixed-year transitions (deterministic, no lag needed) --------
        if to1 and to1["type"] == "fixed" and stage < 1 and year == to1["year"]:
            stage = 1
            activation1_year = year
        if to2 and to2["type"] == "fixed" and stage < 2 and year == to2["year"]:
            stage = 2
            activation2_year = year

        # ---- triggered transitions (based on the PREVIOUS year's signpost) --
        if to1 and to1["type"] == "trigger" and stage < 1 and activation1_year is None:
            sp1 = to1.get("signpost", "pt_trips")
            if sp1 == "pt_trips":
                val1 = prev_pt_trips
            elif sp1 == "annual_pt_trips":
                val1 = prev_annual_pt_trips
            elif sp1 == "pt_share":
                val1 = prev_pt_share
            elif sp1 == "congestion_delay_hours":
                val1 = prev_delay
            else:
                val1 = prev_tt  # default to avg_tt_min
                
            if val1 is not None and val1 > to1["threshold"]:
                persist1 += 1
            else:
                persist1 = 0
            if persist1 >= to1["persistence"] and decision1_year is None:
                decision1_year = year
                activation1_year = year + to1["lead_time"]
        if to1 and activation1_year is not None and stage < 1 and year >= activation1_year:
            stage = 1

        if to2 and to2["type"] == "trigger" and stage >= 1 and stage < 2 and activation2_year is None:
            sp2 = to2.get("signpost", "avg_tt_min")
            if sp2 == "pt_trips":
                val2 = prev_pt_trips
            elif sp2 == "annual_pt_trips":
                val2 = prev_annual_pt_trips
            elif sp2 == "pt_share":
                val2 = prev_pt_share
            elif sp2 == "congestion_delay_hours":
                val2 = prev_delay
            else:
                val2 = prev_tt  # default to avg_tt_min
                
            if val2 is not None and val2 > to2["threshold"]:
                persist2 += 1
            else:
                persist2 = 0
            if persist2 >= to2["persistence"] and decision2_year is None:
                decision2_year = year
                activation2_year = year + to2["lead_time"]
        if to2 and activation2_year is not None and stage < 2 and year >= activation2_year:
            stage = 2

        # ---- simulate this year at the resulting stage ---------------------
        base_m = stage_metrics.get(stage, {})
        mode_result = mode_results_by_stage.get(stage) if mode_results_by_stage else None
        row = m.simulate_year(base_m, stage, t, g_traj[t], params,
                              context=context, mode_result=mode_result, corridor_municipalities=corridor_municipalities)

        total_daily_trips = base_m.get("total_trips", 0) * (1 + g_traj[t])
        base_pt = base_m.get("pt_share", 0.0)
        base_car = base_m.get("car_share", 0.0)
        if pt_traj is not None:
            pt_affinity = pt_traj[t]
            pt_share = base_pt * pt_affinity
            delta_pt = pt_share - base_pt
            car_share = max(base_car - delta_pt, 0.0)
            pt_mult = pt_affinity
        else:
            pt_share = base_pt
            car_share = base_car
            pt_mult = 1.0

        # Match Year-1 accounting conventions
        inv_this_year = pathway_def.get("upfront_fee", 0.0) if year == 1 else 0.0
        if year == 1:
            if pathway_def["initial_stage"] >= 1:
                inv_this_year += pathway_def.get("inv_stage1", 0.0)
            if pathway_def["initial_stage"] >= 2:
                inv_this_year += pathway_def.get("inv_stage2", 0.0)
        if activation1_year == year:
            inv_this_year += pathway_def.get("inv_stage1", 0.0)
        if activation2_year == year:
            inv_this_year += pathway_def.get("inv_stage2", 0.0)
        
        op_this_year = {
            0: 0.0, 
            1: pathway_def.get("op_stage1", 0.0), 
            2: pathway_def.get("op_stage1", 0.0) + pathway_def.get("op_stage2", 0.0)
        }[stage]

        pt_peak_trips = base_m.get("pt_trips", 0.0) * (1 + g_traj[t]) * pt_mult
        row.update({
            "year": year,
            "stage": stage,
            "car_share": car_share,
            "pt_share": pt_share,
            "bike_share": base_m.get("bike_share", 0.0),
            "walk_share": base_m.get("walk_share", 0.0),
            "pt_trips": pt_peak_trips,
            "annual_pt_trips": pt_peak_trips * params.get("PEAK_TO_ANNUAL", 1200.0),
            "car_trips": base_m.get("car_trips", 0.0) * (1 + g_traj[t]) * (car_share / base_car if base_car > 0 else 1.0),
            "annual_car_trips": base_m.get("car_trips", 0.0) * (1 + g_traj[t]) * (car_share / base_car if base_car > 0 else 1.0) * params.get("PEAK_TO_ANNUAL", 1200.0),
            "avg_tt_min": row.get("avg_tt_min", 0.0),
            "total_travel_time_hours": row.get("avg_tt_min", 0.0) * total_daily_trips / 60,
            "inv_cost": inv_this_year,
            "op_cost": op_this_year,
            "total_cost": (
                row.get("car_cost", row.get("car", 0.0))
                + row.get("pt_cost", row.get("pt", 0.0))
                + inv_this_year
                + op_this_year
            ),
        })
        records.append(row)

        prev_tt = row["avg_tt_min"]
        prev_pt_share = pt_share
        prev_pt_trips = pt_peak_trips
        prev_annual_pt_trips = row["annual_pt_trips"]
        prev_delay = row.get("congestion_delay_hours", 0.0)

    df = pd.DataFrame(records)
    meta = {
        "decision1_year": decision1_year, "activation1_year": activation1_year,
        "decision2_year": decision2_year, "activation2_year": activation2_year,
    }
    return df, meta


# =============================================================================
# 4. PERFORMANCE REQUIREMENT, TIPPING POINTS, OPPORTUNITY POINTS
# =============================================================================

def is_acceptable_year(avg_tt_min: float, pt_share: float, max_avg_tt: float | None = None, pt_share_target: float | None = None) -> bool:
    if max_avg_tt is None:
        max_avg_tt = MAX_AVG_TT
    if pt_share_target is None:
        pt_share_target = PT_SHARE_TARGET
    return (avg_tt_min <= max_avg_tt) and (pt_share >= pt_share_target)


def _stage_forever_trajectory(stage: int, stage_metrics: dict, g_traj: np.ndarray,
                               params: dict | None = None,
                               context=None, mode_results_by_stage: dict | None = None, corridor_municipalities: list | None = None) -> pd.DataFrame:
    """Simulate one infrastructure stage held fixed for the whole horizon."""
    if params is None:
        params = m.NOMINAL_PARAMS
    rows = []
    base_m = stage_metrics.get(stage, {})
    mode_result = mode_results_by_stage.get(stage) if mode_results_by_stage else None
    for t in range(N_YEARS):
        row = m.simulate_year(base_m, stage, t, g_traj[t], params,
                              context=context, mode_result=mode_result, corridor_municipalities=corridor_municipalities)
        row["year"] = t + 1
        row["pt_share"] = base_m.get("pt_share", 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def find_tipping_point(stage: int, stage_metrics: dict, g_traj: np.ndarray,
                        params: dict | None = None,
                        context=None, mode_results_by_stage: dict | None = None, corridor_municipalities: list | None = None) -> int | None:
    """Adaptation tipping point for one stage."""
    if params is None:
        params = m.NOMINAL_PARAMS
    max_tt = params.get("MAX_AVG_TT", MAX_AVG_TT)
    pt_target = params.get("PT_SHARE_TARGET", PT_SHARE_TARGET)
    if max_tt is None:
        max_tt = MAX_AVG_TT
    if pt_target is None:
        pt_target = PT_SHARE_TARGET
    
    df = _stage_forever_trajectory(stage, stage_metrics, g_traj, params,
                                   context=context, mode_results_by_stage=mode_results_by_stage, corridor_municipalities=corridor_municipalities)
    failing = df[~df.apply(lambda r: is_acceptable_year(r.get("avg_tt_min", 0), r.get("pt_share", 0), max_tt, pt_target), axis=1)]
    return int(failing.iloc[0]["year"]) if not failing.empty else None


def find_opportunity_point(stage_from: int, stage_to: int, stage_metrics: dict, g_traj: np.ndarray,
                            params: dict | None = None, benefit_margin: float = 2.0,
                            context=None, mode_results_by_stage: dict | None = None, corridor_municipalities: list | None = None) -> int | None:
    """Opportunity point for upgrading stage_from -> stage_to.
    Checks if avg_tt_min drops by at least benefit_margin (e.g., 2 mins)."""
    df_from = _stage_forever_trajectory(stage_from, stage_metrics, g_traj, params,
                                        context=context, mode_results_by_stage=mode_results_by_stage, corridor_municipalities=corridor_municipalities)
    df_to = _stage_forever_trajectory(stage_to, stage_metrics, g_traj, params,
                                      context=context, mode_results_by_stage=mode_results_by_stage, corridor_municipalities=corridor_municipalities)
    relief = df_from["avg_tt_min"].values - df_to["avg_tt_min"].values
    crossing = np.where(relief >= benefit_margin)[0]
    return int(crossing[0] + 1) if len(crossing) else None
