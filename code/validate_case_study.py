"""Fast, read-only preflight for a student-adapted case study.

Run this after editing parameters.py, stages.py, or adaptive_planning.py and
before starting a costly surrogate rebuild. It performs no MSA runs and writes
no files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import adaptive_planning as ap
import parameters as p
import stages
import transport_model_interface as tmi
from additional import section_flows as sf
from additional import uncertainty as uc
from surrogate_model.transport_surrogate import artifact_status


ROOT = Path(__file__).resolve().parent.parent


def validate_case_study(project_root: str | Path = ROOT) -> dict:
    """Validate the centralized student configuration and report rebuild state."""
    root = Path(project_root).resolve()
    uc.validate_params()
    ap.validate_case_study_configuration()

    stage_specs = stages.get_stages(p.NOMINAL_PARAMS)
    stage_ids = list(map(int, stage_specs))
    for package in (1, 2):
        stages.asset_residual_value(package, 1.0, 0.0)
        stages.construction_emissions_tonnes(package)
    zones_file = tmi.configured_input_paths(root)["zones"]
    zones = pd.read_parquet(zones_file, columns=["grid_id", "municipality_name"])
    selected_zones = tmi.resolve_corridor_zone_ids(zones)

    section = sf.section_config()
    external = sf.external_flow_config()
    if external["enabled"] and not section["active"]:
        raise ValueError("External flows require an active, prepared SECTION.")
    # Also validates available section names and coverage alignment. Inactive
    # generic projects take the fast path and need no section files.
    sf.coverage_masks(zones["grid_id"], section)
    artifacts = artifact_status(root, stage_specs=stage_specs)

    return {
        "corridor_zones": len(selected_zones),
        "municipalities": int(zones.loc[zones["grid_id"].astype(str).isin(selected_zones), "municipality_name"].nunique()),
        "stage_ids": stage_ids,
        "state_headways_min": {stage: stages.stage_headway(stage) for stage in stage_ids},
        "state_section_savings_min": {stage: stage_specs[stage]["section_time_saving_min"] for stage in stage_ids},
        "construction_emissions_tonnes": {name: stages.construction_emissions_tonnes(name)
                                          for name in ("stations", "tunnel")},
        "plans": list(ap.PLANS),
        "transport_gp_inputs": list(uc.TRANSPORT_SURROGATE_INPUTS),
        "scenario_uncertainties": uc.select_uncertainties(use="scenarios"),
        "sensitivity_uncertainties": uc.select_uncertainties(use="sensitivity"),
        "surrogate_status": artifacts["gp_status"],
        "response_table_status": artifacts["response_status"],
        "diagnostics_status": artifacts["diagnostics_status"],
        "artifacts_ready": artifacts["ready"],
        "artifact_notes": artifacts["reasons"],
        "section_active": section["active"],
        "section_mode": section["mode"],
        "external_flow_enabled": external["enabled"],
        "external_additional_trips_daily": external["additional_trips_daily"] if external["enabled"] else 0.0,
    }


if __name__ == "__main__":
    report = validate_case_study()
    print("Case-study configuration: PASS")
    print(json.dumps(report, indent=2))

