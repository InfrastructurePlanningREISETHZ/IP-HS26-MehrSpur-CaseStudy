"""Check native matched annual benefits against independent OD-GP holdouts.

Both exact heldout OD delays and GP predictions pass through the same native
mode choice, annualisation and matched OD/submode rule of half. No new MSA or
training occurs. These are paired annual snapshots, not full-future NPV tests.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
import sys

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
import parameters as p
import simulation_engine as m
from surrogate_model.transport_surrogate import load_transport_surrogate

APPRAISAL_OUTPUTS = ("total_cost", "co2_tonnes", "avg_tt_min")
# Disjoint costs: aggregate car/pt/external subtotals would double count.
COMPONENT_FIELDS = {name: spec["field"] for name, spec in m.COST_COMPONENTS.items()}


def _annual_snapshot(state, stage, demand_multiplier):
    return m.simulate_year(state["metrics"], stage, p.N_YEARS - 1, demand_multiplier - 1,
                           params=p.NOMINAL_PARAMS, inv_cost=0.0, flex_cost=0.0,
                           transport_state=state, include_welfare=True)


def _benefits(rows):
    base = pd.DataFrame([rows[0]])
    benefits = {}
    for stage, row in rows.items():
        if stage == 0:
            continue
        matched = m.matched_appraisal_trajectory(pd.DataFrame([row]), base).iloc[0]
        values = {name: float(rows[0][field] - matched[field]) for name, field in COMPONENT_FIELDS.items()}
        total = float(rows[0]["total_cost"] - matched["total_cost"])
        if not np.isclose(sum(values.values()), total, rtol=1e-10, atol=1e-4):
            raise ValueError("Matched benefit components do not reconcile with annual total.")
        benefits[stage] = {**values, "total": total}
    return benefits


def _statistics(actual, mean):
    actual, mean = np.asarray(actual), np.asarray(mean)
    error = mean - actual
    absolute = np.abs(error)
    scale = max(float(np.ptp(actual)), abs(float(actual.mean()))*.05, 1e-9)
    return {"n": len(actual), "mean_reference": float(actual.mean()),
            "mean_absolute_error": float(absolute.mean()), "rmse": float(np.sqrt(np.mean(error**2))),
            "nrmse": float(np.sqrt(np.mean(error**2))/scale),
            "median_absolute_relative_error": float(np.median(absolute/np.maximum(np.abs(actual),1e-9))),
            "max_absolute_relative_error": float(np.max(absolute/np.maximum(np.abs(actual),1e-9)))}


def _summarize(paired, absolute):
    total = paired.loc[paired.component == "total"]
    actual, mean = total.actual.to_numpy(), total.predicted.to_numpy()
    error = np.abs(mean-actual)
    checks = {
        "total_benefit_mae": {"value": float(error.mean()), "limit": max(.05*float(np.abs(actual).mean()),1e6)},
        "total_benefit_p95_error": {"value": float(np.quantile(error,.95)),
                                    "limit": max(.10*float(np.quantile(np.abs(actual),.95)),5e6)},
        "material_total_sign_errors": {"value": int(np.sum((np.abs(actual)>5e6)&(np.sign(actual)!=np.sign(mean)))), "limit": 0},
    }
    components = {}
    for name, group in paired.groupby("component"):
        stats = _statistics(group.actual, group.predicted)
        magnitude = float(np.abs(group.actual).mean())
        applicable = name != "total" and magnitude >= 5e6
        limit = max(.10*magnitude,1e6)
        stats.update(material=applicable, mean_absolute_reference=float(magnitude),
                     error_limit_chf_per_year=limit,
                     passed=bool(stats["mean_absolute_error"] <= limit) if applicable else None)
        if magnitude < 5e6:
            stats["median_absolute_relative_error"] = None
            stats["max_absolute_relative_error"] = None
        components[name] = stats
    checks["material_component_failures"] = {
        "value": sum(item["material"] and not item["passed"] for item in components.values()), "limit": 0}
    shares = absolute.loc[absolute.metric.str.endswith("_share")]
    if not shares.empty:
        difference = (shares.predicted-shares.actual).abs()*100
        checks["maximum_mode_share_error_pp"] = {"value": float(difference.max()), "limit": 1.0}
        checks["median_mode_share_error_pp"] = {"value": float(difference.median()), "limit": .25}
    section = absolute.loc[absolute.metric == "section_modeled_peak"]
    if not section.empty:
        difference = (section.predicted-section.actual).abs()/np.maximum(np.abs(section.actual),1.0)
        checks["maximum_section_load_relative_error"] = {"value": float(difference.max()), "limit": .03}
        checks["median_section_load_relative_error"] = {"value": float(difference.median()), "limit": .01}
    for check in checks.values():
        check["passed"] = bool(check["value"] <= check["limit"])
    metrics = {str(stage): {name: _statistics(group.actual,group.predicted)
                           for name,group in subset.groupby("metric")}
               for stage,subset in absolute.groupby("stage")}
    return {"passed": all(check["passed"] for check in checks.values()), "checks": checks,
            "paired_components": components, "metrics": metrics}


def validate_emulator(emulator, *, design=None, write_outputs=False, project_root=ROOT):
    """Validate a loaded (or builder-created pending) emulator without refitting."""
    directory = Path(emulator.artifact_dir)
    for name, key in (("od_validation.npz", "validation_data_sha256"),
                      (emulator.manifest.get("artifact_file", "od_emulator.joblib"), "artifact_sha256")):
        if key in emulator.manifest:
            digest = hashlib.sha256()
            with (directory/name).open("rb") as stream:
                for block in iter(lambda: stream.read(1024*1024), b""):
                    digest.update(block)
            if digest.hexdigest() != emulator.manifest[key]:
                raise ValueError("Fitted model/validation payload changed after the manifest was recorded.")
    if design is None:
        design = pd.read_parquet(directory / "msa_design_and_outputs.parquet")
    holdout = design.loc[design.role.isin(["test","development","validation"])].copy()
    if holdout.empty or holdout.id.duplicated().any():
        raise ValueError("Require uniquely identified independent heldout cases.")
    names = list(emulator.input_names)
    with np.load(directory / "od_validation.npz", allow_pickle=False) as stored:
        ids, labels = list(map(str,stored["ids"])), list(map(str,stored["labels"]))
        actual, mean = stored["actual"].copy(), stored["mean"].copy()
        stages, inputs = stored["stages"].copy(), stored["inputs"].copy()
    if set(ids) != set(holdout.id.astype(str)) or len(ids) != len(set(ids)):
        raise ValueError("Validation matrices and design have different case IDs.")
    holdout = holdout.set_index("id").loc[ids].reset_index()
    expected_labels = list(map(str,emulator.manifest.get("od_labels", labels)))
    if labels != expected_labels or len(set(labels)) != len(labels):
        raise ValueError("Validation OD labels differ from the trained output order.")
    shape = (len(holdout),len(labels),len(labels))
    if (actual.shape != shape or mean.shape != shape or not np.isfinite(actual).all()
            or not np.isfinite(mean).all() or (actual < 0).any() or (mean < 0).any()
            or not np.array_equal(stages,holdout.stage.to_numpy())
            or not np.array_equal(inputs,holdout[names].to_numpy())):
        raise ValueError("Validation matrices or physical input alignment are invalid.")
    # A complete identical-input stage set is required. Grouping by inputs also
    # supports imported experiment development rows with a different pair label.
    holdout["_position"] = np.arange(len(holdout))
    stages_expected = set(map(int,emulator.manifest["stages"]))
    if 0 not in stages_expected or len(stages_expected)<2:
        raise ValueError("Paired appraisal requires baseline stage 0 and at least one project stage.")
    absolute, paired = [], []
    groups = holdout.groupby(names, sort=False, dropna=False) if names else [((), holdout)]
    for pair_number, (_, group) in enumerate(groups):
        if set(group.stage.astype(int)) != stages_expected or len(group) != len(stages_expected):
            raise ValueError("Every heldout future must contain exactly one row for every stage.")
        side_rows = []
        for matrices in (actual,mean):
            rows = {}
            for _, record in group.sort_values("stage").iterrows():
                stage, position = int(record.stage),int(record["_position"])
                values = {name: float(record[name]) for name in names}
                state = emulator.transport_state_from_od_delay(
                    stage=stage, delay_table=pd.DataFrame(matrices[position],index=labels,columns=labels),
                    include_welfare=True, **values)
                from additional.uncertainty import complete_transport_inputs
                rows[stage] = _annual_snapshot(state,stage,1.0 + complete_transport_inputs(values)["PASSENGER_DEMAND_GROWTH"])
            benefits = _benefits(rows)
            scalar = {stage:{key:value for key,value in row.items() if key!="_welfare_od"} for stage,row in rows.items()}
            side_rows.append((scalar,benefits))
            del rows, state
        (truth,truth_benefits),(prediction,predicted_benefits) = side_rows
        for stage in sorted(stages_expected):
            fields = [*APPRAISAL_OUTPUTS, *[mode+"_share" for mode in ("car","pt","bike","walk")]]
            if "section_modeled_peak" in truth[stage]:
                fields.append("section_modeled_peak")
            for field in fields:
                observed,predicted = float(truth[stage][field]),float(prediction[stage][field])
                if not np.isfinite([observed,predicted]).all():
                    raise ValueError("Nonfinite native annual validation output.")
                absolute.append({"pair":pair_number,"stage":stage,"metric":field,
                                 "actual":observed,"predicted":predicted})
            if stage != 0:
                for component,value in truth_benefits[stage].items():
                    paired.append({"pair":pair_number,"stage":stage,"component":component,
                                   "actual":value,"predicted":predicted_benefits[stage][component]})
    absolute,paired = pd.DataFrame(absolute),pd.DataFrame(paired)
    report = _summarize(paired,absolute)
    report.update(appraisal_welfare_version=m.APPRAISAL_WELFARE_VERSION,
                  reference="Independent heldout MSA OD delays; identical native postprocessing; no new MSA.",
                  scope="Paired annual baseline/project matched benefits and physical indicators, not full-future NPV.",
                  n_holdouts_per_stage=int(holdout.groupby("stage").size().min()),
                  n_independent_physical_futures=int(len(holdout)/len(stages_expected)),
                  surrogate_fingerprint=emulator.manifest["model_fingerprint"],
                  component_rule="Material mean benefit >= CHF5m/year: MAE <= max(10%, CHF1m/year).",
                  current_parameters=p.NOMINAL_PARAMS)
    if write_outputs:
        root = Path(project_root)
        (root/"results").mkdir(exist_ok=True)
        (root/"results/05_gp_appraisal_validation.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
        absolute.to_csv(root/"results/05_gp_appraisal_holdout_comparison.csv",index=False)
        paired.to_csv(root/"results/05_gp_appraisal_paired_benefits.csv",index=False)
    return report


def validate(project_root=ROOT):
    return validate_emulator(load_transport_surrogate(Path(project_root), use_response_table=False),write_outputs=True,
                             project_root=project_root)


if __name__ == "__main__":
    result = validate()
    print(json.dumps(result,indent=2))
    if not result["passed"]:
        warnings.warn("GP appraisal validation targets were not met; results remain available for analysis.",
                      UserWarning)
