"""
simulation_engine.py
=============
MehrSpur – core economic evaluation engine.

WHAT THIS FILE DOES
-------------------
  1. Receive a native or surrogate transport state.
  2. Convert peak quantities into annual physical quantities.
  3. Value time, emissions, externalities and infrastructure costs.
  4. Match baseline/project OD welfare and discount the annual components.

HOW IT FITS IN THE PROJECT
--------------------------
    parameters.py      ──►┐
    transport_model_interface ──►├──  simulation_engine.py  ──►  notebooks (EMA Workbench)
    adaptive_planning.py        ──►┘

STUDENT SURROGATE RULE
----------------------
This module is the correct place for uncertainties that revalue an already
computed transport state (for example unit costs, emission factors, and
the number of peak hours). Such variables stay outside the GP. If a proposed change can
alter demand, mode choice, routes, link flows, travel time, or congestion, it
must instead be evaluated inside the MSA and represented by a surrogate input.
See surrogate_model/README.md before adding a new uncertainty.
"""

from __future__ import annotations

import zlib
import numpy as np
import pandas as pd

from parameters import (
    N_YEARS, DISCOUNT_RATE,
    NOMINAL_PARAMS,
)
from additional import uncertainty as uc
from stages import stage_capacity, stage_headway, stage_components

APPRAISAL_WELFARE_VERSION = "matched_zone_od_sections_external_component_vot_v4"
APPRAISAL_RESIDUAL_VERSION = "straight_line_terminal_v1"
APPRAISAL_CONSTRUCTION_VERSION = "construction_spread_time_zero_v1"


# Disjoint annual cost fields, their NPC labels and parent totals. All values
# are CHF/year; upfront premiums are paid at time zero when discounting.
COST_COMPONENTS = {
    name: {"field": field, "parent": parent, "unit": "CHF/year"}
    for name, field, parent in (
        ("car_time", "car_time_cost", "car_cost"),
        ("car_co2", "car_co2_cost", "car_cost"),
        ("car_noise", "car_noise_cost", "car_cost"),
        ("car_air", "car_air_cost", "car_cost"),
        ("car_acc", "car_acc_cost", "car_cost"),
        ("pt_time", "pt_time_cost", "pt_cost"),
        ("pt_crowding", "pt_crowding_cost", "pt_cost"),
        ("pt_waiting", "pt_waiting_cost", "pt_cost"),
        ("pt_access", "pt_access_cost", "pt_cost"),
        ("pt_egress", "pt_egress_cost", "pt_cost"),
        ("pt_transfer_walk", "pt_transfer_walk_cost", "pt_cost"),
        ("pt_co2", "pt_co2_cost", "pt_cost"),
        ("pt_noise", "pt_noise_cost", "pt_cost"),
        ("pt_air", "pt_air_cost", "pt_cost"),
        ("pt_acc", "pt_acc_cost", "pt_cost"),
        ("pt_socioeconomic", "pt_socioeconomic_cost", "pt_cost"),
        ("bike_time", "bike_time_cost", "bike_cost"),
        ("bike_health", "bike_health_cost", "bike_cost"),
        ("walk_time", "walk_time_cost", "walk_cost"),
        ("walk_health", "walk_health_cost", "walk_cost"),
        ("external_time", "external_time_cost", "external_cost"),
        ("external_delay", "external_delay_cost", "external_cost"),
        ("external_crowding", "external_crowding_cost", "external_cost"),
        ("construction_co2", "construction_co2_cost", None),
        ("investment", "inv_cost", None),
        ("upfront", "upfront_cost", None),
        ("operation", "op_cost", None),
    )
}
TIME_COMPONENTS = {
    field: {"metric": metric, "mode": mode, "vot": vot, "welfare_vot": welfare_vot}
    for field, metric, mode, vot, welfare_vot in (
        ("car_time_cost", "car_tt_hours", "car", "C_TT_CAR", "welfare_vot_car"),
        ("pt_time_cost", "pt_tt_hours", "pt", "C_TT_PT", "welfare_vot_pt"),
        ("pt_waiting_cost", "pt_wait_hours", "pt", "C_TT_PT_WAITING", "welfare_vot_pt_wait"),
        ("pt_access_cost", "pt_access_hours", "pt", "C_TT_PT_ACCESS", "welfare_vot_pt_access"),
        ("pt_egress_cost", "pt_egress_hours", "pt", "C_TT_PT_ACCESS", "welfare_vot_pt_access"),
        ("pt_transfer_walk_cost", "pt_transfer_walk_hours", "pt", "C_TT_PT_TRANSFER", "welfare_vot_pt_transfer"),
        ("bike_time_cost", "bike_tt_hours", "bike", "C_TT_BIKE", "welfare_vot_bike"),
        ("walk_time_cost", "walk_tt_hours", "walk", "C_TT_WALK", "welfare_vot_walk"),
    )
}


def resolved_parameters(params: dict | None = None) -> dict:
    """Apply overrides to the single project configuration."""
    return {**NOMINAL_PARAMS, **(params or {})}


def annualizers(params: dict | None = None) -> dict[str, float]:
    """All-day person-trip factors, peak-only delays, and independent train supply."""
    params = resolved_parameters(params)
    days = float(params["EQUIVALENT_DAYS_PER_YEAR"])
    peak_hours = float(params["NUMBER_OF_PEAK_HOURS"])
    if not np.isfinite(days) or days <= 0:
        raise ValueError("Equivalent days per year must be finite and positive.")
    if not np.isfinite(peak_hours) or not peak_hours.is_integer() or not 1 <= peak_hours <= 24:
        raise ValueError("NUMBER_OF_PEAK_HOURS must be an integer between 1 and 24.")
    factors = {}
    for mode in ("car", "pt", "bike", "walk"):
        key = mode.upper() + "_PEAK_SHARE"
        share = float(params[key])
        if not np.isfinite(share) or not 0 < share <= 1:
            raise ValueError(f"{key} must be in (0, 1].")
        factors[mode] = days / share
    factors.update(congestion=days * peak_hours,
                   crowding=days * peak_hours,
                   train=float(params["PEAK_TO_ANNUAL_TRAIN"]))
    if any(
        not np.isfinite(value) or value < 0 for value in factors.values()
    ):
        raise ValueError("Peak-to-year conversion factors must be finite and nonnegative.")
    return factors


_annualizers = annualizers  # Existing notebook extensions can retain their import.


def _annual_journey_hours(metrics, scale, factors, extra_delay, *, appraisal=False, headway=None):
    """Annual person-hours; appraisal can select the matched corridor OD scope."""
    hours = {}
    for field, spec in TIME_COMPONENTS.items():
        key = spec["metric"]
        value = metrics.get(key, 0.0)
        if appraisal:
            value = metrics.get("appraisal_" + key, value)
            if key == "pt_wait_hours" and key not in metrics:
                value = metrics.get("pt_trips", 0.0) * headway / 240.0
        hours[field] = float(value) * scale * factors[spec["mode"]]
    delay = extra_delay
    if appraisal:
        delay = metrics.get("appraisal_car_delay_person_hours", extra_delay / scale if scale else 0.0) * scale
    hours["car_time_cost"] += delay * factors["congestion"]
    return hours


def _section_and_external_costs(base_metrics: dict, scale: float, growth: float,
                                stage: int, params: dict, factors: dict | None = None) -> dict:
    """Price the same existing external cohort in every alternative after mode choice.

    The shared transport interface supplies physical section exposure. Only PT
    receives comfort crowding; CAR delay is valued at peak. No added person is
    assigned to roads, generates a train, or changes the modeled mode shares.
    """
    from additional.section_flows import section_config, external_flow_config, physical_signature

    section = section_config(params)
    external = external_flow_config(params)
    active = bool(section["active"])
    enabled = bool(external["enabled"])
    mode = str(section["mode"]).upper()
    signature = physical_signature(section)  # Disabled sections do not load coverage files.
    factors = annualizers(params) if factors is None else factors
    result = {
        "section_active": active, "section_mode": mode, "section_signature": signature,
        "section_modeled_peak": 0.0, "section_modeled_daily": 0.0,
        "section_external_peak": 0.0, "section_external_daily": 0.0,
        "section_combined_peak": 0.0, "section_combined_daily": 0.0,
        "section_capacity_peak": 0.0, "section_load_ratio": 0.0,
        "section_crowding_multiplier": 1.0,
        "section_reference_minutes0": 0.0, "section_reference_minutes": 0.0,
        "section_reference_delay_minutes": 0.0,
        "external_flow_enabled": enabled, "external_flow_mode": mode,
        "external_flow_daily": 0.0, "external_flow_peak": 0.0,
        "external_flow_signature": "none",
        "external_time_cost": 0.0, "external_delay_cost": 0.0,
        "external_crowding_cost": 0.0, "external_cost": 0.0,
        "pt_crowding_cost": 0.0,
    }
    if not active:
        if enabled:
            raise ValueError("Enable SECTION before adding an external flow to it.")
        return result

    required = ("section_modeled_trips_peak", "section_modeled_person_hours_peak",
                "section_reference_minutes0", "section_reference_minutes",
                "section_reference_delay_minutes")
    if base_metrics.get("section_signature") != signature or any(key not in base_metrics for key in required):
        raise ValueError("Transport section metrics are missing or stale. Rebuild the transport "
                         "state with the selected section before appraisal.")
    values = {key: float(base_metrics[key]) for key in required}
    if any(not np.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError("Section quantities and times must be finite and nonnegative.")
    share = float(params[mode + "_PEAK_SHARE"])
    if not 0 < share <= 1:
        raise ValueError(f"{mode}_PEAK_SHARE must be in (0, 1].")
    external_growth = growth if external["growth"] == "general" else 1.0
    daily = float(external["additional_trips_daily"]) * external_growth if enabled else 0.0
    if not np.isfinite(daily) or daily < 0:
        raise ValueError("External daily flow must be finite and nonnegative.")
    modeled_peak = values["section_modeled_trips_peak"] * scale
    external_peak = daily * share
    combined = modeled_peak + external_peak
    capacity, ratio, crowd = 0.0, 0.0, 1.0
    if mode == "PT" and section.get("crowding_enabled", False):
        capacity = stage_capacity(stage, params)
        slope = float(params["CROWDING_SLOPE"])
        cap = float(params["CROWDING_MAX"])
        if not np.isfinite(capacity) or capacity <= 0 or not np.isfinite(slope) or slope < 0 or not np.isfinite(cap) or cap < 1:
            raise ValueError("Crowding requires positive capacity, nonnegative slope and a multiplier cap >= 1.")
        ratio = combined / capacity
        crowd = min(cap, 1.0 + slope * max(ratio - 1.0, 0.0))
    modeled_crowding = (values["section_modeled_person_hours_peak"] * scale
                        * factors["crowding"] * params["C_TT_PT"] * (crowd - 1.0))
    external_vot = external.get("value_of_time_chf_per_hour")
    external_vot = float(params["C_TT_" + mode] if external_vot is None else external_vot)
    if not np.isfinite(external_vot) or external_vot < 0:
        raise ValueError("External-flow value of time must be finite and nonnegative.")
    days = float(params["EQUIVALENT_DAYS_PER_YEAR"])
    if not np.isfinite(days) or days < 0:
        raise ValueError("Equivalent appraisal days must be finite and nonnegative.")
    external_time = daily * days * values["section_reference_minutes"] / 60.0 * external_vot
    external_delay = (external_peak * factors["congestion"] * values["section_reference_delay_minutes"]
                      / 60.0 * external_vot if mode == "CAR" else 0.0)
    external_crowd = (external_peak * factors["crowding"] * values["section_reference_minutes"]
                      / 60.0 * external_vot * (crowd - 1.0))
    result.update(section_signature=signature,
                  section_modeled_peak=modeled_peak, section_modeled_daily=modeled_peak / share,
                  section_external_peak=external_peak, section_external_daily=daily,
                  section_combined_peak=combined, section_combined_daily=combined / share,
                  section_capacity_peak=capacity, section_load_ratio=ratio,
                  section_crowding_multiplier=crowd,
                  external_flow_mode=mode, external_flow_daily=daily, external_flow_peak=external_peak,
                  external_flow_signature=signature if enabled else "none",
                  external_time_cost=external_time, external_delay_cost=external_delay,
                  external_crowding_cost=external_crowd,
                  external_cost=external_time + external_delay + external_crowd,
                  pt_crowding_cost=modeled_crowding)
    result.update({key: values[key] for key in required if key.startswith("section_reference_")})
    return result


def _value_of_time(params: dict, key: str) -> float:
    value = params.get(key)
    if value is None or not np.isfinite(value) or value < 0:
        raise ValueError(f"Set {key} in parameters.py to the course value in CHF/person-hour before appraisal.")
    return float(value)


def carbon_value_chf_per_tonne(year_idx: int, params: dict | None = None) -> float:
    """Carbon value in constant 2019 CHF, compounded from its reference year."""
    params = resolved_parameters(params)
    calendar_year = params["APPRAISAL_START_YEAR"] + year_idx
    return float(params["CO2_VALUE_CHF_PER_TONNE"] * (1.0 + params["CO2_VALUE_ANNUAL_GROWTH"])
                 ** (calendar_year - params["CO2_REFERENCE_YEAR"]))


def _annual_train_km(stage: int, params: dict, factors: dict) -> float:
    """Train supply follows the timetable, independently of passenger demand."""
    frequency = stage_headway(0, params) / max(stage_headway(stage, params), 1.0)
    return float(params["PT_TRAIN_KM_PEAK_STAGE0"] * factors["train"] * frequency)


def _annual_carbon_tonnes(car_vehicle_km: float, train_km: float, params: dict) -> dict:
    """Convert road vehicle-km and rail gross-tonne-km to physical tonnes CO2."""
    car = car_vehicle_km * params["CAR_CO2_KG_PER_VEHICLE_KM"] / 1_000.0
    rail_kwh = train_km * params["TRAIN_GROSS_TONNES"] * params["RAIL_ENERGY_WH_PER_GROSS_TONNE_KM"] / 1_000.0
    rail = rail_kwh * params["RAIL_ELECTRICITY_CO2_G_PER_KWH"] / 1_000_000.0
    return {"car_co2_tonnes": float(car), "pt_co2_tonnes": float(rail), "co2_tonnes": float(car + rail)}

# =============================================================================
# 1. CORE ECONOMIC & PHYSICAL FORMULAS (STUDENT-EDITABLE)
# =============================================================================
# ⚠️ STUDENTS: This is the engine room of the Cost-Benefit Analysis (CBA).
# Values are configured in parameters.py. The helpers above define annual
# conversions, physical emissions and section crowding; annual_costs prices them.

def annual_costs(
    base_metrics: dict,
    year_idx: int,
    g_cum: float,
    stage: int,
    params: dict | None = None,
    extra_delay: float = 0.0,
    inv_cost: float | None = None,
    op_cost: float | None = None,
    flex_cost: float = 0.0,
    demand_scaled: bool = False,
) -> dict:
    """
    Return all annual societal and infrastructure costs (CHF) for one year.
    Broken out by road, public transport, active modes, and infrastructure categories.

    base_metrics      : Aggregated transport metrics from the FSM for the active stage.
    year_idx          : 0-based year index (year_idx=0 → Year 1 of the horizon).
    g_cum             : Cumulative demand growth factor for this year.
    stage             : Transport state: 0 baseline, 1 stations, 2 tunnel, 3 both.
    params            : Parameter dictionary (defaults to NOMINAL_PARAMS).
    extra_delay       : Car passenger-hours of congestion, excluding background vehicles.
    inv_cost          : Capital investment cost (CHF) incurred in this year (optional).
    op_cost           : Annual operations & maintenance cost (CHF) for this year (optional;
                        if None, automatically computed from the active stage).
    flex_cost         : Time-zero option premium (CHF), recorded only in Year 1.
    """
    params = resolved_parameters(params)
    scale = 1.0 if demand_scaled else (1.0 + g_cum)
    c_co2_t = carbon_value_chf_per_tonne(year_idx, params)

    # NPVM mode-specific peak shares give ordinary all-day quantities. Road
    # congestion and section crowding accrue only in the configured peak hours.
    factors = annualizers(params)
    section_costs = _section_and_external_costs(base_metrics, scale, 1.0 + g_cum, stage, params, factors)
    hours = _annual_journey_hours(base_metrics, scale, factors, extra_delay,
                                  appraisal=True, headway=stage_headway(stage, params))
    time_costs = {field: hours[field] * _value_of_time(params, spec["vot"])
                  for field, spec in TIME_COMPONENTS.items()}
    d_car = base_metrics.get("car_vehicle_km", base_metrics.get("car_dist_km", 0)) * scale * factors["car"]
    zugkm_pt = _annual_train_km(stage, params, factors)
    emissions = _annual_carbon_tonnes(d_car, zugkm_pt, params)

    # Optional external benefits are negative societal costs. Value person-km
    # for standalone active trips and person-trips for PT, not boardings.
    benefit_rates = {key: float(params.get(key, 0.0)) for key in (
        "BENEFIT_HEALTH_BIKE_PER_KM", "BENEFIT_HEALTH_WALK_PER_KM",
        "BENEFIT_SOCIOECONOMIC_PT_PER_TRIP",
    )}
    if any(not np.isfinite(value) or value < 0 for value in benefit_rates.values()):
        raise ValueError("Optional benefit rates must be finite and nonnegative.")
    bike_health_cost = (-base_metrics.get("bike_dist_km", 0.0) * scale * factors["bike"]
                        * benefit_rates["BENEFIT_HEALTH_BIKE_PER_KM"])
    walk_health_cost = (-base_metrics.get("walk_dist_km", 0.0) * scale * factors["walk"]
                        * benefit_rates["BENEFIT_HEALTH_WALK_PER_KM"])
    pt_person_trips = base_metrics.get("appraisal_pt_trips", base_metrics.get("pt_trips", 0.0)) * scale * factors["pt"]
    if section_costs["external_flow_mode"] == "PT":
        pt_person_trips += section_costs["external_flow_daily"] * params["EQUIVALENT_DAYS_PER_YEAR"]
    pt_socioeconomic_cost = -pt_person_trips * benefit_rates["BENEFIT_SOCIOECONOMIC_PT_PER_TRIP"]

    car_time_cost = time_costs["car_time_cost"]
    car_co2_cost = emissions["car_co2_tonnes"] * c_co2_t
    car_noise_cost = d_car * params["C_NOISE_CAR"]
    local_air_multiplier = params["LOCAL_AIR_COST_MULTIPLIER"]
    accident_multiplier = params["ACCIDENT_COST_MULTIPLIER"]
    car_air_cost = d_car * params["C_AIR_CAR"] * local_air_multiplier
    car_acc_cost = d_car * params["C_ACCIDENT_CAR"] * accident_multiplier

    car_cost = car_time_cost + car_co2_cost + car_noise_cost + car_air_cost + car_acc_cost

    # ── 2. PT: In-Vehicle Travel Time + Wait Time + Crowding Penalty ──
    pt_time_cost = time_costs["pt_time_cost"]
    pt_crowding_cost = section_costs["pt_crowding_cost"]
    pt_waiting_cost = time_costs["pt_waiting_cost"]
    pt_access_cost = time_costs["pt_access_cost"]
    pt_egress_cost = time_costs["pt_egress_cost"]
    pt_transfer_walk_cost = time_costs["pt_transfer_walk_cost"]

    # ── 3. PT: Externalities (Air, Noise, CO2, Accidents) ────────────
    # Rail air/noise rates and energy consumption use gross-tonne-km:
    # annual train-km × gross tonnes per train.
    gross_tonnes_per_train = params["TRAIN_GROSS_TONNES"]
    btkm_pt = zugkm_pt * gross_tonnes_per_train

    pt_air_cost = btkm_pt * params["C_AIR_PT"] * local_air_multiplier
    pt_noise_cost = btkm_pt * params["C_NOISE_PT"]
    pt_co2_cost = emissions["pt_co2_tonnes"] * c_co2_t
    pt_acc_cost = zugkm_pt * params["C_ACCIDENT_PT"] * accident_multiplier


    pt_cost = (
        pt_time_cost
        + pt_crowding_cost
        + pt_waiting_cost
        + pt_access_cost
        + pt_egress_cost
        + pt_transfer_walk_cost
        + pt_co2_cost
        + pt_noise_cost
        + pt_air_cost
        + pt_acc_cost
        + pt_socioeconomic_cost
    )

    # ── 4. Active Mobility (Bikes / Walk) ────────────
    # Standalone active trips are separate mode-choice alternatives from PT
    # access/egress, so their own journey times are counted exactly once.
    bike_time_cost = time_costs["bike_time_cost"]
    walk_time_cost = time_costs["walk_time_cost"]
    bike_cost = bike_time_cost + bike_health_cost
    walk_cost = walk_time_cost + walk_health_cost

    # ── 5. Infrastructure: OPEX, CAPEX, Flexibility ─────────────────
    cost_mult = params["OPEX_MULTIPLIER"]

    if op_cost is None:
        c_inv1 = params["C_INV_STAGE1"]
        c_inv2 = params["C_INV_STAGE2"]
        opex_rate = params["OPEX_RATE"]
        c_op1 = c_inv1 * opex_rate
        c_op2 = c_inv2 * opex_rate
        stations_active, tunnel_active = stage_components(stage)
        op_cost = (c_op1 * stations_active + c_op2 * tunnel_active) * cost_mult

    if inv_cost is None:
        inv_cost = 0.0
    if flex_cost is None:
        flex_cost = 0.0
    if not np.isfinite(flex_cost) or flex_cost < 0 or (flex_cost and year_idx != 0):
        raise ValueError("The time-zero option premium must be nonnegative and recorded only in Year 1.")

    total_cost = car_cost + pt_cost + bike_cost + walk_cost + section_costs["external_cost"] + inv_cost + op_cost + flex_cost

    return {
        # High-level totals
        "car": car_cost,
        "pt": pt_cost,
        "bike": bike_cost,
        "walk": walk_cost,
        "inv": inv_cost + flex_cost,
        "upfront": flex_cost,
        "op": op_cost,
        "total": total_cost,

        # Explicit cost fields (matching trajectory DataFrame columns)
        "car_cost": car_cost,
        "pt_cost": pt_cost,
        "bike_cost": bike_cost,
        "walk_cost": walk_cost,
        "bike_time_cost": bike_time_cost,
        "walk_time_cost": walk_time_cost,
        "bike_health_cost": bike_health_cost,
        "walk_health_cost": walk_health_cost,
        "inv_cost": inv_cost,
        "upfront_cost": flex_cost,
        "op_cost": op_cost,
        "total_cost": total_cost,

        # Road / car breakdown
        "car_time_cost": car_time_cost,
        "car_co2_cost": car_co2_cost,
        "car_noise_cost": car_noise_cost,
        "car_air_cost": car_air_cost,
        "car_acc_cost": car_acc_cost,

        # Rail / PT breakdown
        "pt_time_cost": pt_time_cost,
        "pt_crowding_cost": pt_crowding_cost,
        "pt_waiting_cost": pt_waiting_cost,
        "pt_access_cost": pt_access_cost,
        "pt_egress_cost": pt_egress_cost,
        "pt_transfer_walk_cost": pt_transfer_walk_cost,
        "pt_co2_cost": pt_co2_cost,
        "pt_noise_cost": pt_noise_cost,
        "pt_air_cost": pt_air_cost,
        "pt_acc_cost": pt_acc_cost,
        "pt_socioeconomic_cost": pt_socioeconomic_cost,
        **section_costs,
    }


def annual_physical_indicators(base_metrics: dict, g_cum: float, params: dict | None = None, extra_delay: float = 0.0, demand_scaled: bool = False, stage: int = 0) -> dict:
    """Annual physical indicators; stage determines the supplied rail timetable."""
    params = resolved_parameters(params)

    scale = 1.0 if demand_scaled else (1.0 + g_cum)
    factors = annualizers(params)
    p2a_car, p2a_congestion = factors["car"], factors["congestion"]

    d_car = base_metrics.get("car_vehicle_km", base_metrics.get("car_dist_km", 0)) * scale * p2a_car
    emissions = _annual_carbon_tonnes(d_car, _annual_train_km(stage, params, factors), params)

    # The signpost uses peak car/PT in-vehicle minutes plus road delay.
    tt_car_peak = base_metrics.get("car_tt_hours", 0) * scale + extra_delay
    tt_pt_peak = base_metrics.get("pt_tt_hours", 0) * scale
    car_trips_peak = base_metrics.get("car_trips", 0) * scale
    pt_trips_peak = base_metrics.get("pt_trips", 0) * scale
    motorized_trips_peak = car_trips_peak + pt_trips_peak
    if motorized_trips_peak <= 0:
        motorized_trips_peak = (base_metrics.get("total_trips", 1) * scale) * (
            base_metrics.get("car_share", 0.0) + base_metrics.get("pt_share", 0.0)
        )

    avg_tt_min = ((tt_car_peak + tt_pt_peak) / max(motorized_trips_peak, 1.0)) * 60.0

    # Annual modeled corridor journey-hours, including PT waiting and walking.
    # Ordinary time uses each mode's daily conversion; congestion is peak-only.
    total_travel_time_hours = sum(_annual_journey_hours(base_metrics, scale, factors, extra_delay).values())

    return {
        **emissions,
        "avg_tt_min": avg_tt_min,
        "congestion_delay_hours": extra_delay * p2a_congestion,
        "total_travel_time_hours": total_travel_time_hours,
    }

# =============================================================================
# 2. SHARED TRANSPORT-STATE APPRAISAL
# =============================================================================

def add_construction_emissions(row: dict, annual_tonnes: float, upfront_tonnes: float = 0.0,
                               *, params: dict | None = None) -> dict:
    """Return a row with construction CO2e priced at its time of emission.

    Pre-horizon emissions are booked at time zero, the start of Year 1, using
    that year's carbon value (2026 for MehrSpur). Their separate subtotal is
    excluded from year-end discounting. Total construction fields include it.
    This replaces any existing construction values, permitting safe row reuse.
    """
    tonnes = np.asarray([annual_tonnes, upfront_tonnes], dtype=float)
    year_idx = int(row["year_idx"])
    if not np.isfinite(tonnes).all() or (tonnes < 0).any():
        raise ValueError("Construction emissions must be finite and nonnegative.")
    if upfront_tonnes and year_idx != 0:
        raise ValueError("Time-zero construction emissions belong in the Year-1 row.")
    result = dict(row)
    upfront_cost = float(upfront_tonnes) * carbon_value_chf_per_tonne(0, params)
    total_cost = float(annual_tonnes) * carbon_value_chf_per_tonne(year_idx, params) + upfront_cost
    total_tonnes = float(tonnes.sum())
    cost_change = total_cost - float(result.get("construction_co2_cost", 0.0))
    result["co2_tonnes"] += total_tonnes - float(result.get("construction_co2_tonnes", 0.0))
    result["total_cost"] += cost_change
    if "total" in result:
        result["total"] += cost_change
    result.update(construction_co2_tonnes=total_tonnes, construction_co2_cost=total_cost,
                  upfront_construction_co2_tonnes=float(upfront_tonnes),
                  upfront_construction_co2_cost=upfront_cost,
                  appraisal_construction_version=APPRAISAL_CONSTRUCTION_VERSION)
    return result


def simulate_year(
    base_metrics: dict,
    stage: int,
    year_idx: int,
    g_cum: float,
    params: dict | None = None,
    context=None,
    mode_result=None,
    corridor_municipalities=None,
    corridor_context=None,
    return_details: bool = False,
    assignment_settings: dict | None = None,
    assignment_result=None,
    inv_cost: float | None = None,
    op_cost: float | None = None,
    flex_cost: float = 0.0,
    include_welfare: bool = False,
    transport_state: dict | None = None,
) -> dict:
    """Simulate one year and return a results dict.

    ``corridor_context`` lets interactive callers reuse an already constructed
    local network. ``return_details`` adds lightweight mode and assignment
    summaries without changing the default result consumed by the plan runner.
    ``assignment_settings`` preserves notebook-side MSA settings when a
    year is evaluated in a separate worker process. ``assignment_result`` can
    reuse an identical coupled solve prepared earlier in the notebook.
    ``include_welfare`` retains matched OD samples or compact paired moments
    for appraisal; both use the same annual monetary coefficients.
    ``transport_state`` is the shared boundary between the transport solver and
    appraisal. It may come from the exact coupled MSA or its validated fast
    surrogate, but always enters the same annualisation and cost functions.
    """
    params = resolved_parameters(params)
    scale = 1.0 + g_cum
    factors = annualizers(params)
    p2a_congestion = factors["congestion"]

    # Notebook callers can request native transport here; the interface owns
    # mode choice and assignment. Every backend then enters the same appraisal.
    if transport_state is None:
        if context is None or mode_result is None:
            raise ValueError("Supply a transport_state or the native context and mode result.")
        from transport_model_interface import acquire_native_transport_state
        transport_state = acquire_native_transport_state(
            context, mode_result, stage=stage,
            corridor_municipalities=corridor_municipalities,
            corridor_context=corridor_context, passenger_demand_multiplier=scale,
            pt_asc_shift=params["PT_ASC_SHIFT"], bike_asc_shift=params["BIKE_ASC_SHIFT"],
            ebike_share=params["EBIKE_SHARE"], road_freight_multiplier=max(0.0, 1.0 + params["ROAD_FREIGHT_GROWTH"]),
            assignment_settings=assignment_settings, assignment_result=assignment_result,
            include_welfare=include_welfare,
        )
    if int(transport_state.get("stage", stage)) != int(stage):
        raise ValueError("Transport-state stage does not match simulate_year(stage).")
    if not isinstance(transport_state.get("metrics"), dict):
        raise ValueError("transport_state must contain a 'metrics' dictionary.")
    base_metrics = dict(transport_state["metrics"])
    demand_scaled = True
    extra_delay = float(base_metrics.get("congestion_delay_hours", 0.0))
    calculation_method = str(transport_state.get("source", "transport_state"))
    welfare_od, welfare_summary = None, None
    if include_welfare:
        welfare_od = transport_state.get("welfare_od")
        if not isinstance(welfare_od, dict):
            welfare_od = None
            welfare_summary = transport_state.get("welfare_summary")
            if not isinstance(welfare_summary, dict):
                raise ValueError("Request include_welfare when creating the transport state for matched appraisal.")
            if welfare_summary.get("stage") != int(stage):
                raise ValueError("Matched welfare summary stage differs from the transport state.")
            if _welfare_summary_inputs(welfare_summary.get("input_values")) != _welfare_summary_inputs(
                transport_state.get("input_values")
            ):
                raise ValueError("Matched welfare summary and transport state use different physical inputs.")

    costs = annual_costs(
        base_metrics,
        year_idx,
        g_cum,
        stage,
        params=params,
        extra_delay=extra_delay,
        inv_cost=inv_cost,
        op_cost=op_cost,
        flex_cost=flex_cost,
        demand_scaled=demand_scaled,
    )
    phys = annual_physical_indicators(
        base_metrics,
        g_cum,
        params=params,
        extra_delay=extra_delay,
        demand_scaled=demand_scaled,
        stage=stage,
    )

    scale_for_metrics = 1.0 if demand_scaled else (1.0 + g_cum)
    total_demand = scale_for_metrics * sum(
        float(base_metrics.get(mode + "_trips", 0.0)) * factors[mode]
        for mode in ("car", "pt", "bike", "walk")
    )

    result = {
        "year_idx":               year_idx,
        "calendar_year":          params["APPRAISAL_START_YEAR"] + year_idx,
        "carbon_value_chf_per_tonne": carbon_value_chf_per_tonne(year_idx, params),
        "stage":                  stage,
        "total_demand":           total_demand,
        **{key: value for key, value in costs.items() if key.endswith("_cost")},
        "co2_tonnes":             phys["co2_tonnes"],
        "car_co2_tonnes":         phys["car_co2_tonnes"],
        "pt_co2_tonnes":          phys["pt_co2_tonnes"],
        # Construction is added by the plan's build schedule, not an operating snapshot.
        "construction_co2_tonnes": 0.0,
        "upfront_construction_co2_tonnes": 0.0,
        "construction_co2_cost": 0.0,
        "upfront_construction_co2_cost": 0.0,
        "avg_tt_min":             phys["avg_tt_min"],
        "total_travel_time_hours": phys["total_travel_time_hours"],
        "congestion_delay_hours": extra_delay * p2a_congestion,
        "car_share":              base_metrics.get("car_share", base_metrics.get("car_share_trips", 0.0)),
        "pt_share":               base_metrics.get("pt_share", base_metrics.get("pt_share_trips", 0.0)),
        "bike_share":             base_metrics.get("bike_share", base_metrics.get("bike_share_trips", 0.0)),
        "walk_share":             base_metrics.get("walk_share", base_metrics.get("walk_share_trips", 0.0)),
        "car_share_trips":        base_metrics.get("car_share_trips", 0.0),
        "pt_share_trips":         base_metrics.get("pt_share_trips", 0.0),
        "bike_share_trips":       base_metrics.get("bike_share_trips", 0.0),
        "walk_share_trips":       base_metrics.get("walk_share_trips", 0.0),
        "car_trips":              base_metrics.get("car_trips", 0.0) * scale_for_metrics,
        "pt_trips":               base_metrics.get("pt_trips", 0.0) * scale_for_metrics,
        "bike_trips":             base_metrics.get("bike_trips", 0.0) * scale_for_metrics,
        "walk_trips":             base_metrics.get("walk_trips", 0.0) * scale_for_metrics,
        "car_tt_hours":           base_metrics.get("car_tt_hours", 0.0) * scale_for_metrics,
        "pt_tt_hours":            base_metrics.get("pt_tt_hours", 0.0) * scale_for_metrics,
        "pt_wait_hours":          base_metrics.get("pt_wait_hours", 0.0) * scale_for_metrics,
        "calculation_method":     calculation_method,
        # Retain each side's annual cost coefficients for matched-OD appraisal.
        "welfare_demand_scale":   scale,
        "welfare_p2a_congestion": p2a_congestion,
        "welfare_p2a_crowding":   factors["crowding"],
        **{"welfare_p2a_" + mode: factors[mode] for mode in ("car", "pt", "bike", "walk")},
        **{spec["welfare_vot"]: _value_of_time(params, spec["vot"]) for spec in TIME_COMPONENTS.values()},
        "welfare_crowding":       costs["section_crowding_multiplier"],
    }
    result.update({key: value for key, value in costs.items()
                   if key.startswith(("section_", "external_"))})
    if welfare_od is not None:
        result["_welfare_od"] = welfare_od
    if welfare_summary is not None:
        result["_welfare_summary"] = welfare_summary
    if return_details:
        # Numeric-only additions keep parallel dashboard results inexpensive to
        # serialise while exposing the modal response calculated inside MSA.
        result["mode_metrics"] = dict(base_metrics)
        if "assignment_diagnostics" in transport_state:
            result["assignment_diagnostics"] = dict(transport_state["assignment_diagnostics"])
    return result


WELFARE_COMPONENTS = {
    "car_freeflow": ("car", "car_time_cost"),
    "car_delay": ("car", "car_time_cost"),
    "pt_walk_ivt": ("pt_walk", "pt_time_cost"),
    "pt_walk_section": ("pt_walk", "pt_crowding_cost"),
    "pt_walk_wait": ("pt_walk", "pt_waiting_cost"),
    "pt_walk_access": ("pt_walk", "pt_access_cost"),
    "pt_walk_egress": ("pt_walk", "pt_egress_cost"),
    "pt_walk_transfer_walk": ("pt_walk", "pt_transfer_walk_cost"),
    "pt_bike_ivt": ("pt_bike", "pt_time_cost"),
    "pt_bike_section": ("pt_bike", "pt_crowding_cost"),
    "pt_bike_wait": ("pt_bike", "pt_waiting_cost"),
    "pt_bike_access": ("pt_bike", "pt_access_cost"),
    "pt_bike_egress": ("pt_bike", "pt_egress_cost"),
    "pt_bike_transfer_walk": ("pt_bike", "pt_transfer_walk_cost"),
    "bike_time": ("bike", "bike_time_cost"),
    "walk_time": ("walk", "walk_time_cost"),
}
_WELFARE_COMPONENTS = WELFARE_COMPONENTS  # Compatibility for saved response tooling.


def _welfare_coefficient(row: pd.Series, component: str) -> float:
    """Per-unit CHF/year value to multiply a component's OD time-moment by:
    the matching person-hour value (mode and PT journey component) times the
    appropriate annualizer (congestion for car delay, linear otherwise).
    Excludes crowding, which _welfare_crowding_coefficient adds separately."""
    if component.endswith("_section"):
        return _welfare_crowding_coefficient(row)
    spec = TIME_COMPONENTS[WELFARE_COMPONENTS[component][1]]
    annualizer = row["welfare_p2a_congestion"] if component == "car_delay" else row["welfare_p2a_" + spec["mode"]]
    value = row[spec["welfare_vot"]]
    return float(annualizer * value)


def _welfare_crowding_coefficient(row: pd.Series) -> float:
    """Incremental IVT coefficient attributable only to crowding."""
    return float(
        row["welfare_p2a_crowding"]
        * row["welfare_vot_pt"]
        * (row["welfare_crowding"] - 1.0)
    )


def _welfare_summary_inputs(values: dict | None) -> dict[str, float]:
    """Canonical physical inputs identify the baseline paired with a summary."""
    if not isinstance(values, dict) or set(values) != set(uc.TRANSPORT_SURROGATE_INPUTS):
        raise ValueError("Matched welfare summary lacks the declared physical transport inputs.")
    try:
        inputs = {name: float(value) for name, value in values.items()}
    except (TypeError, ValueError) as error:
        raise ValueError("Matched welfare summary has invalid physical inputs.") from error
    if not all(np.isfinite(value) for value in inputs.values()):
        raise ValueError("Matched welfare summary has non-finite physical inputs.")
    return inputs


def _matched_time_benefits(project: pd.Series, baseline: pd.Series) -> dict:
    """Annual matched OD/submode benefits, positive for a cost reduction."""
    required = ("welfare_demand_scale",
                "welfare_p2a_congestion", "welfare_vot_car", "welfare_vot_pt",
                "welfare_vot_pt_wait", "welfare_vot_pt_access", "welfare_vot_pt_transfer", "welfare_vot_bike",
                "welfare_vot_walk", "welfare_crowding", "welfare_p2a_car", "welfare_p2a_pt",
                "welfare_p2a_bike", "welfare_p2a_walk", "welfare_p2a_crowding")
    if any(key not in row or not np.isfinite(row[key]) for row in (baseline, project) for key in required):
        raise ValueError("Matched welfare metadata is missing; rerun annual trajectories before appraisal.")
    if int(baseline["stage"]) != 0:
        raise ValueError("Matched welfare requires the same year's stage-0 counterfactual.")
    for year_field in ("year_idx", "year"):
        if year_field in baseline and year_field in project and baseline[year_field] != project[year_field]:
            raise ValueError("Matched welfare requires baseline and project from the same year.")
    if not np.isclose(baseline["welfare_demand_scale"], project["welfare_demand_scale"]):
        raise ValueError("Matched welfare requires baseline and project under the same demand future.")
    if baseline.get("section_signature", "none") != project.get("section_signature", "none"):
        raise ValueError("Baseline and project must use the same counting section and reference OD.")
    enabled = [bool(row.get("external_flow_enabled", False)) for row in (baseline, project)]
    if any(enabled):
        if not all(enabled) or any(
            baseline.get(key) != project.get(key)
            for key in ("external_flow_mode", "external_flow_signature")
        ):
            raise ValueError("Baseline and project must contain the same existing external cohort.")
        quantities = [float(row.get("external_flow_daily", np.nan)) for row in (baseline, project)]
        if not all(np.isfinite(value) and value >= 0 for value in quantities) or not np.isclose(*quantities):
            raise ValueError("External passengers must be identical in the same baseline/project year; do not recalibrate each stage.")
    methods = {str(row.get("calculation_method", "")) for row in (baseline, project)}
    benefit = dict.fromkeys(
        (*dict.fromkeys(output for _, output in WELFARE_COMPONENTS.values()), "pt_crowding_cost"), 0.0
    )
    # Exact FSM/GP states retain OD arrays. The optional response table retains
    # paired OD moments instead; annual prices are applied only after matching.
    samples = [row.get("_welfare_od") for row in (baseline, project)]
    if all(isinstance(sample, dict) for sample in samples):
        if any(not set(WELFARE_COMPONENTS).issubset(sample.get("times", {})) for sample in samples):
            raise ValueError("Welfare state lacks complete zone-OD time components; regenerate the transport trajectories.")
        if samples[0].get("od_keys") != samples[1].get("od_keys") or not samples[0].get("od_keys"):
            raise ValueError("Baseline and project transport states use different OD layouts.")
        def array(sample, section, name):
            value = sample[section][name]
            if sample.get("encoding") == "zlib-float64-v1":
                result = np.frombuffer(zlib.decompress(value), dtype="<f8")
                if result.size != sample["od_count"]:
                    raise ValueError("Invalid compressed matched OD state.")
                return result
            return np.asarray(value, dtype=float)

        # Decode only one mode/component at a time. Lossless packing retains
        # every zone OD while keeping annual trajectories reasonably compact.
        current_mode = None
        for component, (mode, output) in WELFARE_COMPONENTS.items():
            if mode != current_mode:
                q0, qs = (array(sample, "quantities", mode) for sample in samples)
                current_mode = mode
            t0, ts = (array(sample, "times", component) for sample in samples)
            arrays = (q0, qs, t0, ts)
            if any(a.ndim != 1 or a.shape != q0.shape or not np.all(np.isfinite(a)) for a in arrays):
                raise ValueError(f"Invalid matched MSA arrays for {component}.")
            if np.any(q0 < 0) or np.any(qs < 0):
                raise ValueError("Matched MSA quantities must be nonnegative.")
            benefit[output] += float(.5 * np.dot(q0 + qs,
                t0 * _welfare_coefficient(baseline, component)
                - ts * _welfare_coefficient(project, component)))
        return benefit
    summaries = [row.get("_welfare_summary") for row in (baseline, project)]
    if any(isinstance(summary, dict) for summary in summaries):
        if not all(isinstance(summary, dict) for summary in summaries) or methods != {"surrogate_msa"}:
            raise ValueError("Both trajectories must contain compatible matched welfare summaries.")
        expected = set(WELFARE_COMPONENTS)
        moments = []
        identities = []
        for row, summary in zip((baseline, project), summaries):
            if summary.get("schema") != "matched_moments_v1":
                raise ValueError("Unsupported matched welfare summary schema.")
            if summary.get("stage") != int(row["stage"]):
                raise ValueError("Matched welfare summary stage differs from its annual row.")
            if summary.get("section_signature") != row.get("section_signature", "none"):
                raise ValueError("Matched welfare summary uses a different counting section.")
            table_id = summary.get("table_id")
            if not isinstance(table_id, str) or not table_id:
                raise ValueError("Matched welfare summary has no response-table identity.")
            identities.append((table_id, _welfare_summary_inputs(summary.get("input_values"))))
            names = summary.get("components", ())
            if not isinstance(names, (list, tuple)) or len(names) != len(expected) or set(names) != expected:
                raise ValueError("Matched welfare summary lacks complete OD/submode components.")
            paired = {}
            for field in ("delta_hours", "project_hours"):
                values = summary.get(field)
                try:
                    if isinstance(values, dict):
                        if set(values) != expected:
                            raise ValueError("incomplete component moments")
                        vector = np.asarray([values[name] for name in names], dtype=float)
                    else:
                        vector = np.asarray(values, dtype=float)
                    if vector.shape != (len(names),) or not np.isfinite(vector).all():
                        raise ValueError("invalid component moments")
                    if field == "project_hours" and np.any(vector < 0):
                        raise ValueError("negative quantity-weighted time")
                except (TypeError, ValueError) as error:
                    raise ValueError(f"Invalid matched welfare summary {field}.") from error
                paired[field] = dict(zip(names, vector))
            if int(row["stage"]) == 0 and any(value != 0 for value in paired["delta_hours"].values()):
                raise ValueError("A baseline welfare summary must have zero time differences.")
            moments.append(paired)
        if identities[0] != identities[1]:
            raise ValueError("Matched welfare summaries use different tables or physical inputs.")
        # delta=.5*sum((q0+qs)*(t0-ts)); project=.5*sum((q0+qs)*ts).
        # Keeping both moments also handles different crowding coefficients.
        for component, (_, output) in WELFARE_COMPONENTS.items():
            c0 = _welfare_coefficient(baseline, component)
            cs = _welfare_coefficient(project, component)
            benefit[output] += float(c0 * moments[1]["delta_hours"][component]
                                    + (c0 - cs) * moments[1]["project_hours"][component])
        return benefit
    raise ValueError(
        "Matched appraisal requires OD welfare state or paired moments. "
        "Rebuild both trajectories with include_welfare=True."
    )


def matched_appraisal_trajectory(results: pd.DataFrame,
                                 baseline_results: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return annual appraisal rows after the matched Rule-of-Half adjustment.

    Keeping this transformation separate from discounting makes it possible to
    reuse already evaluated annual operating transport states when testing
    alternative adaptive pathways. The trigger state machine determines which
    state operates, when CAPEX is paid, and the construction-emissions schedule;
    those plan-dependent costs must be recalculated for the selected pathway.
    """
    df = results.copy()

    if "appraisal_welfare_version" in df:
        if not df["appraisal_welfare_version"].eq(APPRAISAL_WELFARE_VERSION).all():
            raise ValueError("Stale or mixed appraisal rows; regenerate the trajectories.")
        if baseline_results is not None:
            raise ValueError("These rows already contain matched welfare; discount them without applying a baseline again.")
        return df

    if baseline_results is not None:
        if df["year_idx"].duplicated().any() or baseline_results["year_idx"].duplicated().any():
            raise ValueError("Matched welfare requires one row per year in each trajectory.")
        if set(df["year_idx"]) != set(baseline_results["year_idx"]):
            raise ValueError("Baseline and project welfare trajectories must contain the same years.")
        base = baseline_results.set_index("year_idx").loc[df["year_idx"]].reset_index()
        base.index = df.index
        benefits = pd.DataFrame(
            [_matched_time_benefits(project, reference)
             for (_, project), (_, reference) in zip(df.iterrows(), base.iterrows())],
            index=df.index,
        )
        # External travelers are an existing identical cohort on both sides.
        # Their separate absolute costs already yield the full cost saving;
        # only the retained model quantities enter the Rule-of-Half adjustment.
        # Anchor time costs to baseline absolute costs. Therefore baseline NPC
        # keeps its meaning and baseline NPC minus project NPC is net benefit.
        matched_fields = {field for _, field in WELFARE_COMPONENTS.values()}
        for spec in COST_COMPONENTS.values():
            field, total_field = spec["field"], spec["parent"]
            if field not in matched_fields:
                continue
            adjusted = base[field] - benefits[field]
            change = adjusted - df[field]
            df[field] = adjusted
            df[total_field] += change
            if "total_cost" in df.columns:
                df["total_cost"] += change

    if baseline_results is not None or ("stage" in df and df["stage"].eq(0).all()):
        df["appraisal_welfare_version"] = APPRAISAL_WELFARE_VERSION

    return df


def npc_by_component(results: pd.DataFrame, baseline_results: pd.DataFrame = None, discount_rate: float | None = None) -> dict:
    """Discount annual flows at year end; option premiums are paid at time zero.

    ``inv`` includes the upfront premium; ``upfront`` is its diagnostic subtotal.
    ``residual`` is a separate negative cost: the remaining asset value credited
    at the end of the horizon. The trajectory's total_cost already includes it.
    Scalar trajectory exports retain ``upfront_cost`` for the same CSV replay.
    Construction CO2e is discounted when emitted; its recorded time-zero
    subtotal is included without discounting in ``construction_co2``.
    """
    df = matched_appraisal_trajectory(results, baseline_results)

    rate = (discount_rate if discount_rate is not None else
            df["discount_rate"].iloc[0] if "discount_rate" in df else DISCOUNT_RATE)
    if (not np.isfinite(rate) or rate <= -1 or
            (discount_rate is None and "discount_rate" in df
             and not np.allclose(df["discount_rate"], rate))):
        raise ValueError("Appraisal requires one finite discount rate per future, greater than -1.")
    df["d"] = 1 / (1 + rate) ** (df["year_idx"] + 1)
    s = 1e6   # convert CHF → MCHF

    # Optional fields might not be present if results are from an older run
    def get_sum(col: str) -> float:
        if col in df.columns:
            return (df[col] * df["d"]).sum() / s
        return 0.0

    upfront = df.get("upfront_cost", pd.Series(0.0, index=df.index))
    if (not np.isfinite(upfront).all() or (upfront < 0).any()
            or ((upfront != 0) & (df["year_idx"] != 0)).any()
            or ((upfront != 0) & (df["year_idx"] == 0)).sum() > 1):
        raise ValueError("Record the nonnegative time-zero premium once, in the Year-1 upfront_cost field.")
    upfront_npc = upfront.sum() / s

    construction_upfront = df.get("upfront_construction_co2_cost", pd.Series(0.0, index=df.index))
    construction = df.get("construction_co2_cost", pd.Series(0.0, index=df.index))
    if (not np.isfinite(construction).all() or not np.isfinite(construction_upfront).all()
            or (construction < 0).any() or (construction_upfront < 0).any()
            or (construction_upfront > construction + 1e-8).any()
            or ((construction_upfront != 0) & (df["year_idx"] != 0)).any()
            or (construction_upfront != 0).sum() > 1):
        raise ValueError("Time-zero construction cost must be a nonnegative subtotal recorded once in Year 1.")
    construction_upfront_npc = construction_upfront.sum() / s
    construction_discount_correction = construction_upfront_npc - get_sum("upfront_construction_co2_cost")

    residual = df.get("residual_value", pd.Series(0.0, index=df.index))
    if (not np.isfinite(residual).all() or (residual < 0).any()
            or ((residual != 0) & (df["year_idx"] != N_YEARS - 1)).any()
            or (residual != 0).sum() > 1):
        raise ValueError("Record the nonnegative residual asset value once, at the end of the appraisal horizon.")

    return {
        **{name: get_sum(spec["field"]) for name, spec in COST_COMPONENTS.items() if spec["parent"] is not None},
        **{mode: get_sum(mode + "_cost") for mode in ("car", "pt", "bike", "walk", "external")},
        "inv":   get_sum("inv_cost") + upfront_npc,  # NIBA 10.6, including options
        "upfront": upfront_npc,                 # Subtotal of inv; do not add twice
        "residual": -get_sum("residual_value"), # Terminal asset value; negative NPC, positive NPV
        "op":    get_sum("op_cost"),             # NIBA 10.5
        "construction_co2": get_sum("construction_co2_cost") + construction_discount_correction,
        "total": get_sum("total_cost") - get_sum("upfront_cost") + upfront_npc + construction_discount_correction,
    }
