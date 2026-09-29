"""Fast, exact trigger-threshold optimization from evaluated annual stage states.

This auxiliary analysis module is intentionally separate from the transport
surrogate package: it consumes completed appraisal trajectories but does not
build or alter the surrogate model.

The transport and appraisal result in a given future/year depends on the active
infrastructure state, not on the threshold that selected that state. Four
constant-state trajectories form the annual lookup cube. Grid search replays
the independent package decisions and selects the corresponding annual row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import adaptive_planning as ap
import parameters as p
import simulation_engine as m
import stages
from additional import uncertainty as uc


def _future_paths(future: pd.Series) -> dict:
    draws = {name: float(future[name]) for name in p.STRUCTURAL_UNCERTAINTIES
             if name in future.index and pd.notna(future[name])}
    shapes = {name: future[name + "__shape"] for name in p.STRUCTURAL_UNCERTAINTIES
              if name + "__shape" in future.index and pd.notna(future[name + "__shape"])}
    return uc.build_paths(draws, shapes=shapes)


def build_stage_lookup(traj_df: pd.DataFrame, futures: pd.DataFrame,
                       discount_rate: float | None = None, *, plan_name="flexible") -> dict:
    """Build scenario x year x stage arrays from existing appraisal results.

    Static-plan CAPEX, construction emissions and residual value are removed
    from the annual state cost. Construction is rebooked over each decision's
    lead time; CAPEX at activation; residual value at the resulting asset age.
    OPEX remains in the annual
    stage cost. Notebook 05 exports costs already adjusted by the matched
    Rule of Half. Their version marker prevents reapplying that adjustment
    after the underlying OD arrays have been discarded. Each future retains its
    recorded discount rate unless an explicit scalar override is supplied.
    """
    if futures["scenario"].duplicated().any():
        raise ValueError("Each scenario must have exactly one future definition.")
    futures = futures.sort_values("scenario")
    scenario_ids = futures["scenario"].astype(int).to_numpy()
    if len(set(scenario_ids)) != len(scenario_ids) or not np.equal(futures["scenario"], scenario_ids).all():
        raise ValueError("Scenario IDs must be unique integers.")
    if not len(scenario_ids):
        raise ValueError("Threshold search needs at least one future.")
    for key in ("scenario", "stage"):
        values = traj_df[key].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"Trajectory {key} values must be finite integers.")
    n_scenarios = len(scenario_ids)
    n_years = p.N_YEARS

    plans = ap.get_plans(p.NOMINAL_PARAMS)
    selected_plan = plans[plan_name]
    rules = [selected_plan[key] for key in ("to1", "to2")]
    if any(rule is not None and rule["type"] != "trigger" for rule in rules):
        raise ValueError("Threshold search requires trigger-based or absent transitions.")
    if not any(rules):
        raise ValueError("Threshold search needs at least one adaptive transition.")
    annual_cost = np.empty((n_scenarios, n_years, max(stages.STATE_IDS) + 1), dtype=float)
    triggers = dict(zip(("trigger1", "trigger2"), rules))
    signposts = [rule["signpost"] if rule is not None else None for rule in rules]
    trigger_values = [np.empty_like(annual_cost) for _ in signposts]
    capex_multiplier = np.empty((n_scenarios, n_years), dtype=float)
    investment1 = np.empty_like(capex_multiplier)
    investment2 = np.empty_like(capex_multiplier)
    option_premium = np.empty_like(capex_multiplier)
    discount = np.empty_like(capex_multiplier)
    carbon_value = np.empty_like(capex_multiplier)
    baseline_npc = np.empty(n_scenarios, dtype=float)

    # Registered constant plans and supplied constant-state references identify
    # their physical state through the annual stage column, not the plan name.
    indexed = {}
    for (name, scenario), group in traj_df.groupby(["plan", "scenario"], sort=False):
        if name in plans and (plans[name]["to1"] is not None or plans[name]["to2"] is not None):
            continue
        if group["stage"].nunique() != 1:
            continue
        stage = int(group["stage"].iloc[0])
        if stage not in stages.STATE_IDS:
            raise ValueError(f"Unknown reference transport state {stage}.")
        ordered = group.sort_values("year_idx").reset_index(drop=True)
        if not np.array_equal(ordered["year_idx"].to_numpy(), np.arange(n_years)):
            raise ValueError(f"Reference '{name}', scenario {scenario} needs each year_idx 0..{n_years - 1} exactly once.")
        key = (stage, int(scenario))
        if key in indexed:
            raise ValueError(f"Multiple constant references for state {stage}, scenario {scenario}.")
        indexed[key] = ordered

    for i, scenario in enumerate(scenario_ids):
        if (0, scenario) not in indexed:
            raise ValueError(f"Missing baseline reference for scenario {scenario}.")
        base = indexed[(0, scenario)]
        rate = (discount_rate if discount_rate is not None else
                float(base["discount_rate"].iloc[0]) if "discount_rate" in base else p.DISCOUNT_RATE)
        if (not np.isfinite(rate) or rate <= -1 or
                (discount_rate is None and "discount_rate" in base
                 and not np.allclose(base["discount_rate"], rate))):
            raise ValueError("Each future requires one finite discount rate greater than -1.")
        discount[i] = 1.0 / (1.0 + rate) ** np.arange(1, n_years + 1)
        baseline_npc[i] = m.npc_by_component(base, discount_rate=rate)["total"]
        if "carbon_value_chf_per_tonne" not in base:
            raise ValueError("Missing annual carbon values; rerun Notebook 05 Section 2.")
        carbon_value[i] = base["carbon_value_chf_per_tonne"].to_numpy(float)
        if not np.isfinite(carbon_value[i]).all() or (carbon_value[i] < 0).any():
            raise ValueError("Annual carbon values must be finite and nonnegative.")

        for stage in stages.STATE_IDS:
            if (stage, scenario) not in indexed:
                raise ValueError(
                    f"Missing annual reference for state {stage}, scenario {scenario}; "
                    "rerun Notebook 05 Section 2 to include all four transport states."
                )
            raw = indexed[(stage, scenario)]
            if "appraisal_welfare_version" in raw:
                if not raw["appraisal_welfare_version"].eq(m.APPRAISAL_WELFARE_VERSION).all():
                    raise ValueError("Stale appraisal trajectories; rerun Notebook 05 Section 2.")
                adjusted = raw
            elif stage == 0:
                adjusted = raw
            elif "_welfare_od" in raw or "_welfare_summary" in raw:
                adjusted = m.matched_appraisal_trajectory(raw, baseline_results=base)
            else:
                raise ValueError(
                    "Trajectory costs have no matched-welfare marker or OD state. "
                    "Rerun Notebook 05 Section 2 before optimizing thresholds."
                )
            # Keep operating costs only. Rebook investment at actual opening,
            # and terminal residual value at the resulting asset age.
            annual_cost[i, :, stage] = (
                adjusted["total_cost"].to_numpy(float)
                - adjusted["inv_cost"].to_numpy(float)
                - adjusted.get("upfront_cost", pd.Series(0.0, index=adjusted.index)).to_numpy(float)
                - adjusted.get("construction_co2_cost", pd.Series(0.0, index=adjusted.index)).to_numpy(float)
                + adjusted.get("residual_value", pd.Series(0.0, index=adjusted.index)).to_numpy(float)
            )
            for signpost, values in zip(signposts, trigger_values):
                if signpost is None:
                    values[i, :, stage] = 0.0
                    continue
                if signpost not in raw:
                    raise ValueError(
                        f"Missing trigger signpost '{signpost}'; rerun Notebook 05 Section 2."
                    )
                values[i, :, stage] = raw[signpost].to_numpy(float)
                if signpost == "co2_tonnes":
                    values[i, :, stage] -= raw.get("construction_co2_tonnes", pd.Series(0.0, index=raw.index)).to_numpy(float)
                elif signpost in {"construction_co2_tonnes", "construction_co2_cost",
                                  "upfront_construction_co2_tonnes", "upfront_construction_co2_cost"}:
                    values[i, :, stage] = 0.0

        future = futures.loc[futures["scenario"].astype(int) == scenario].iloc[0]
        paths = _future_paths(future)
        capex_multiplier[i] = uc.target_path(paths, "CAPEX_MULTIPLIER")
        investment1[i] = uc.target_path(paths, "C_INV_STAGE1") * capex_multiplier[i]
        investment2[i] = uc.target_path(paths, "C_INV_STAGE2") * capex_multiplier[i]
        fee_scale = uc.target_path(paths, "C_FLEX") / p.C_FLEX if p.C_FLEX else np.ones(n_years)
        option_premium[i] = selected_plan["upfront_fee"] * fee_scale * capex_multiplier[i]

    return {
        "scenario_ids": scenario_ids,
        "annual_cost": annual_cost,
        "trigger1_values": trigger_values[0],
        "trigger2_values": trigger_values[1],
        "triggers": triggers,
        "capex_multiplier": capex_multiplier,
        "investment1": investment1,
        "investment2": investment2,
        "option_premium": option_premium,
        "baseline_npc": baseline_npc,
        "discount": discount,
        "carbon_value": carbon_value,
        "construction_lead_times": {package: ap.construction_lead_time(selected_plan, package)
                                    for package in (1, 2)},
        "initial_stage": selected_plan["initial_stage"],
    }


def simulate_threshold_grid(lookup: dict, stage1_thresholds, stage2_thresholds,
                            return_scenario_values: bool = False):
    """Evaluate every Cartesian-product threshold pair in one vectorized pass."""
    s1_values = np.asarray(stage1_thresholds, dtype=float)
    s2_values = np.asarray(stage2_thresholds, dtype=float)
    if any(values.ndim != 1 or not len(values) or not np.isfinite(values).all()
           for values in (s1_values, s2_values)):
        raise ValueError("Threshold arrays must be nonempty, finite and one-dimensional.")
    th1, th2 = np.meshgrid(s1_values, s2_values, indexing="ij")
    th1, th2 = th1.ravel(), th2.ravel()

    n_grid = len(th1)
    trigger1, trigger2 = lookup["triggers"]["trigger1"], lookup["triggers"]["trigger2"]
    n_scenarios, n_years, _ = lookup["annual_cost"].shape
    scenario_index = np.arange(n_scenarios)[None, :]

    stage = np.zeros((n_grid, n_scenarios), dtype=np.int8)
    initially_station, initially_tunnel = stages.stage_components(lookup["initial_stage"])
    stations_active = np.full_like(stage, initially_station, dtype=bool)
    tunnel_active = np.full_like(stage, initially_tunnel, dtype=bool)
    persistence1 = np.zeros_like(stage, dtype=np.int16)
    persistence2 = np.zeros_like(stage, dtype=np.int16)
    activation1 = np.zeros_like(stage, dtype=np.int16)
    activation2 = np.zeros_like(stage, dtype=np.int16)
    prev1 = np.full((n_grid, n_scenarios), np.nan)
    prev2 = np.full((n_grid, n_scenarios), np.nan)
    capital1 = np.zeros((n_grid, n_scenarios), dtype=float)
    capital2 = np.zeros_like(capital1)
    construction_npc = np.zeros_like(capital1)
    construction_tonnes = np.zeros_like(capital1)

    # Flexibility is purchased at time zero in every future, whether exercised or
    # not. This mirrors run_plan_from_trajectories exactly.
    npc_chf = np.broadcast_to(
        lookup["option_premium"][:, 0][None, :],
        (n_grid, n_scenarios),
    ).copy()
    for active, opening, capital, investment in (
            (stations_active, activation1, capital1, lookup["investment1"]),
            (tunnel_active, activation2, capital2, lookup["investment2"])):
        opening[active] = 1
        capital[:] = np.where(active, investment[:, 0][None, :], 0.0)
        npc_chf += capital * lookup["discount"][:, 0][None, :]
    state_map = np.asarray([[stages.state_for_components(station, tunnel)
                             for tunnel in (False, True)] for station in (False, True)])

    for t in range(n_years):
        year = t + 1

        activate1 = np.zeros_like(stations_active)
        if trigger1 is not None:
            stations_active, activation1, persistence1, _, activate1 = ap.advance_trigger(
                year, stations_active, activation1, persistence1, prev1, trigger1, threshold=th1[:, None])
        capital1[activate1] = np.broadcast_to(lookup["investment1"][:, t], capital1.shape)[activate1]
        npc_chf[activate1] += (
            np.broadcast_to(lookup["investment1"][:, t] * lookup["discount"][:, t],
                            (n_grid, n_scenarios))[activate1]
        )
        activate2 = np.zeros_like(tunnel_active)
        if trigger2 is not None:
            tunnel_active, activation2, persistence2, _, activate2 = ap.advance_trigger(
                year, tunnel_active, activation2, persistence2, prev2, trigger2, threshold=th2[:, None])
        capital2[activate2] = np.broadcast_to(lookup["investment2"][:, t], capital2.shape)[activate2]
        npc_chf[activate2] += (
            np.broadcast_to(lookup["investment2"][:, t] * lookup["discount"][:, t],
                            (n_grid, n_scenarios))[activate2]
        )
        stage = state_map[stations_active.astype(int), tunnel_active.astype(int)]

        annual = lookup["annual_cost"][scenario_index, t, stage]
        npc_chf += annual * lookup["discount"][:, t][None, :]
        emitted = np.zeros_like(capital1)
        upfront = np.zeros_like(capital1)
        for package, opening in ((1, activation1), (2, activation2)):
            during, before = ap.construction_emissions_in_year(
                package, year, opening, lookup["construction_lead_times"][package])
            emitted += during
            upfront += before
        construction_cost = emitted * lookup["carbon_value"][:, t][None, :]
        upfront_cost = upfront * lookup["carbon_value"][:, 0][None, :]
        construction_present_value = construction_cost * lookup["discount"][:, t][None, :] + upfront_cost
        npc_chf += construction_present_value
        construction_npc += construction_present_value
        construction_tonnes += emitted + upfront
        # Construction belongs to the realized pathway, not its operating state.
        feedback = {"co2_tonnes": emitted + upfront,
                    "construction_co2_tonnes": emitted + upfront,
                    "construction_co2_cost": construction_cost + upfront_cost,
                    "upfront_construction_co2_tonnes": upfront,
                    "upfront_construction_co2_cost": upfront_cost}
        prev1 = lookup["trigger1_values"][scenario_index, t, stage] + feedback.get(
            trigger1["signpost"] if trigger1 else None, 0.0)
        prev2 = lookup["trigger2_values"][scenario_index, t, stage] + feedback.get(
            trigger2["signpost"] if trigger2 else None, 0.0)

    # A unit-cost factor reuses the shared straight-line formula. Ages take at
    # most n_years values, keeping large threshold grids inexpensive.
    residual = np.zeros_like(npc_chf)
    for package, capital, opening, active in (
            (1, capital1, activation1, stations_active),
            (2, capital2, activation2, tunnel_active)):
        remaining = np.asarray([stages.asset_residual_value(package, 1.0, age)
                                for age in range(n_years + 1)])
        age = np.clip(n_years - opening + 1, 0, n_years)
        residual += np.where(active, capital * remaining[age], 0.0)
    npc_chf -= residual * lookup["discount"][:, -1][None, :]

    npc = npc_chf / 1e6
    npv = lookup["baseline_npc"][None, :] - npc
    trigger1 = trigger1 or {"signpost": "none", "signpost_label": "No station trigger"}
    trigger2 = trigger2 or {"signpost": "none", "signpost_label": "No tunnel trigger"}
    table = pd.DataFrame({
        "stage1_threshold": th1,
        "stage2_threshold": th2,
        "stage1_signpost": trigger1["signpost"],
        "stage2_signpost": trigger2["signpost"],
        "stage1_signpost_label": trigger1["signpost_label"],
        "stage2_signpost_label": trigger2["signpost_label"],
        "mean_NPC_MCHF": npc.mean(axis=1),
        "mean_NPV_MCHF": npv.mean(axis=1),
        "median_NPV_MCHF": np.median(npv, axis=1),
        "probability_positive_NPV": (npv > 0).mean(axis=1),
    })
    if not return_scenario_values:
        return table
    return table, {
        "npc_MCHF": npc,
        "npv_MCHF": npv,
        "activation1_year": activation1,
        "activation2_year": activation2,
        "residual_value_CHF": residual,
        "construction_co2_NPC_CHF": construction_npc,
        "construction_co2_tonnes": construction_tonnes,
    }
