from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from config import (
    DEFAULT_EBIKE_SHARE,
    DRIVE_CONNECTOR_CITY_MIN,
    MODE_CHOICE_FILE,
    OUTPUT_DIR,
)
from interventions import (
    apply_interventions,
    intervention_masks,
    selected_interventions,
)
from mode_choice_zurich import load_mode_choice_parameters, mode_split_aggregated
from model_outputs import (
    intervention_summary,
    logsum_accessibility,
    mode_summary,
    save_outputs,
)
from scenario_generation import generate_scenario_demand
from travel_times import (
    apply_ebike_share,
    apply_perceived_time_policies,
    load_travel_times,
    standalone_walk_mask,
)
from zoning import load_model_inputs


# =====================================================================
# RUN SETTINGS: edit these values before running Main.py
# =====================================================================

RUN_NAME = "baseline"
RUN_ASSIGNMENT = True

APPLY_SCENARIO = False
SCENARIO_SETTINGS = {
    "population_multiplier": 1.0,
    "jobs_multiplier": 1.0,
    "trip_rate_multiplier": 1.0,
    "city_population_growth_share": 0.5,
    "city_jobs_growth_share": 0.5,
    "car_affinity": 1.0,
    "walk_affinity": 1.0,
    "bike_affinity": 1.0,
    "pt_affinity": 1.0,
    "ebike_share": 0.25,
}

# Select 0 for no intervention, or option 1 or 2 from interventions.py.
INTERVENTION_SELECTION = {
    "bike_highway": 0,
    "railway_expansion": 0,
    "mobility_hub": 0,
}


def build_scenario() -> dict:
    scenario = {
        "population_multiplier": 1.0,
        "jobs_multiplier": 1.0,
        "trip_rate_multiplier": 1.0,
        "city_population_growth_share": 0.5,
        "city_jobs_growth_share": 0.5,
        "car_affinity": 1.0,
        "walk_affinity": 1.0,
        "bike_affinity": 1.0,
        "pt_affinity": 1.0,
        "ebike_share": DEFAULT_EBIKE_SHARE,
    }
    if APPLY_SCENARIO:
        scenario.update(SCENARIO_SETTINGS)
    scenario.update(selected_interventions(INTERVENTION_SELECTION))
    scenario["intervention_selection"] = dict(INTERVENTION_SELECTION)

    label = [RUN_NAME]
    if APPLY_SCENARIO:
        label.append("scenario")
    abbreviations = {
        "bike_highway": "bike",
        "railway_expansion": "rail",
        "mobility_hub": "hub",
    }
    for key, abbreviation in abbreviations.items():
        option = int(INTERVENTION_SELECTION.get(key, 0))
        if option:
            label.append(f"{abbreviation}{option}")
    scenario["name"] = "_".join(label)
    return scenario


def run_model(
    scenario: dict,
    *,
    run_assignment: bool = True,
    output_dir: str | Path | None = None,
) -> dict:
    zones, baseline_od, road_background = load_model_inputs()
    future_zones, total_od, demand_diagnostics = generate_scenario_demand(
        baseline_od,
        zones,
        scenario,
    )
    raw_travel_times, raw_lengths = load_travel_times(zones)

    def scenario_inputs(base_times, base_lengths, active_scenario):
        perceived = apply_perceived_time_policies(
            base_times,
            zones,
            drive_connector_city_min=float(
                active_scenario.get("drive_connector_city_min", DRIVE_CONNECTOR_CITY_MIN)
            ),
        )
        intervened_times, intervened_lengths, intervention_diagnostics = apply_interventions(
            perceived,
            base_lengths,
            active_scenario,
            zones=zones,
        )
        adjusted_times, ebike_diagnostics = apply_ebike_share(
            intervened_times,
            float(active_scenario.get("ebike_share", DEFAULT_EBIKE_SHARE)),
        )
        return (
            adjusted_times,
            intervened_lengths,
            intervention_diagnostics,
            ebike_diagnostics,
        )

    travel_times, lengths, intervention_diagnostics, ebike_diagnostics = scenario_inputs(
        raw_travel_times,
        raw_lengths,
        scenario,
    )
    parameters = load_mode_choice_parameters(scenario.get("mode_choice_config", MODE_CHOICE_FILE))
    affinities = {
        "car_affinity": float(scenario.get("car_affinity", 1.0)),
        "walk_affinity": float(scenario.get("walk_affinity", 1.0)),
        "bike_affinity": float(scenario.get("bike_affinity", 1.0)),
        "pt_affinity": float(scenario.get("pt_affinity", 1.0)),
    }

    preliminary_od, preliminary_utilities = mode_split_aggregated(
        travel_times,
        lengths,
        total_od,
        walk_allowed_mask=standalone_walk_mask(lengths, zones),
        parameters=parameters,
        **affinities,
    )

    link_flows = None
    assignment_diagnostics = {"enabled": False}
    final_raw_times = raw_travel_times
    final_raw_lengths = raw_lengths
    if run_assignment:
        from network import load_assignment_network
        from trip_assignment_zurich import trip_assignment_baseline

        assignment_network = load_assignment_network()
        (
            link_flows,
            congested_drive_time,
            congested_drive_distance,
            assignment_diagnostics,
        ) = trip_assignment_baseline(
            assignment_network,
            zones,
            preliminary_od["drive"],
            road_background,
        )
        assignment_diagnostics["enabled"] = True

        assigned_raw_times = {key: value.copy() for key, value in raw_travel_times.items()}
        assigned_raw_times["drive"] = congested_drive_time.combine_first(raw_travel_times["drive"])
        assigned_raw_lengths = {key: value.copy() for key, value in raw_lengths.items()}
        assigned_raw_lengths["drive"] = congested_drive_distance.combine_first(raw_lengths["drive"])
        final_raw_times = assigned_raw_times
        final_raw_lengths = assigned_raw_lengths
        travel_times, lengths, intervention_diagnostics, ebike_diagnostics = scenario_inputs(
            final_raw_times,
            final_raw_lengths,
            scenario,
        )
        final_od, final_utilities = mode_split_aggregated(
            travel_times,
            lengths,
            total_od,
            walk_allowed_mask=standalone_walk_mask(lengths, zones),
            parameters=parameters,
            **affinities,
        )
    else:
        final_od, final_utilities = preliminary_od, preliminary_utilities

    coverage = intervention_masks(
        travel_times,
        lengths,
        scenario,
        zones=zones,
    )
    if coverage:
        counterfactual_scenario = deepcopy(scenario)
        for key in ("bike_highways", "railway_expansions", "mobility_hubs"):
            counterfactual_scenario[key] = []
        counterfactual_times, counterfactual_lengths, _, _ = scenario_inputs(
            final_raw_times,
            final_raw_lengths,
            counterfactual_scenario,
        )
        counterfactual_od, _ = mode_split_aggregated(
            counterfactual_times,
            counterfactual_lengths,
            total_od,
            walk_allowed_mask=standalone_walk_mask(counterfactual_lengths, zones),
            parameters=parameters,
            **affinities,
        )
    else:
        counterfactual_times = travel_times
        counterfactual_lengths = lengths
        counterfactual_od = final_od
    intervention_report = intervention_summary(
        final_od,
        travel_times,
        lengths,
        counterfactual_od,
        counterfactual_times,
        counterfactual_lengths,
        coverage,
    )

    summary = mode_summary(final_od, lengths)
    accessibility = logsum_accessibility(final_utilities, future_zones)
    diagnostics = {
        "demand": demand_diagnostics,
        "interventions": intervention_diagnostics,
        "ebike": ebike_diagnostics,
        "assignment": assignment_diagnostics,
    }
    run_metadata = {
        "scenario": scenario,
        "assignment_enabled": bool(run_assignment),
        "mode_choice_config": str(MODE_CHOICE_FILE.relative_to(Path(__file__).resolve().parent)),
    }

    if output_dir is None:
        scenario_name = str(scenario.get("name", "scenario")).replace(" ", "_")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_dir = OUTPUT_DIR / f"{scenario_name}_{timestamp}"
    save_outputs(
        output_dir=output_dir,
        run_metadata=run_metadata,
        summary=summary,
        accessibility=accessibility,
        od_by_mode=final_od,
        link_flows=link_flows,
        diagnostics=diagnostics,
        zones=zones,
        lengths=lengths,
        intervention_report=intervention_report,
    )
    return {
        "summary": summary,
        "accessibility": accessibility,
        "od_by_mode": final_od,
        "link_flows": link_flows,
        "travel_times": travel_times,
        "lengths": lengths,
        "diagnostics": diagnostics,
        "intervention_summary": intervention_report,
        "output_dir": Path(output_dir),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Canton Zurich course model.")
    parser.add_argument(
        "--no-assignment",
        action="store_true",
        help="Use prepared free-flow road times.",
    )
    parser.add_argument("--output-dir", default="", help="Optional exact output directory.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = run_model(
        build_scenario(),
        run_assignment=RUN_ASSIGNMENT and not args.no_assignment,
        output_dir=args.output_dir or None,
    )
    print(result["summary"].to_string(index=False))
    print(f"\nOutputs: {result['output_dir']}")
