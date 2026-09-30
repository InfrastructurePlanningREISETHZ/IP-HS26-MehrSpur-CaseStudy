"""Saved physical transport metrics and matched OD welfare for planning queries."""
from __future__ import annotations
from additional import uncertainty as uc

from collections import OrderedDict
from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path
import time

import numpy as np

SCHEMA = "transport_response_v1"
FILENAME = "response_table.npz"


def _fingerprint(manifest):
    import parameters as p
    import simulation_engine as m
    from additional import section_flows as sf
    from .transport_surrogate import stable_fingerprint, equation_identity, model_identity

    section = sf.section_config()
    return stable_fingerprint({
        "schema": SCHEMA, "gp_sha256": manifest["artifact_sha256"],
        "physical_model": model_identity(manifest), "section": sf.physical_signature(section),
        "equations": equation_identity({
            "code/surrogate_model/response_table.py": ("_ratios", "_moments", "response_grid", "ResponseTable"),
            "code/surrogate_model/prepared_transport.py": ("PreparedTransport",),
            "code/surrogate_model/od_gp.py": ("SharedKernelGP", "ODDelayGP"),
            "code/transport_model_interface.py": ("_technology_base_components",),
            "code/additional/section_flows.py": ("section_config", "coverage_masks", "physical_car_times",
                                                   "section_metrics", "section_reference"),
        }),
        "components": m.WELFARE_COMPONENTS,
        "minimum_strategic_distance": p.MIN_STRATEGIC_PKM_DISTANCE,
        "nominal_inputs": {name: uc.TRANSPORT_SURROGATE_INPUTS[name]["nominal"]
                           for name in manifest["input_names"]},
    })


def response_grid(input_names, bounds):
    """Three evenly spaced values per input, plus its exact nominal value."""
    return tuple(np.unique(np.r_[np.linspace(lo, hi, 3), uc.TRANSPORT_SURROGATE_INPUTS[name]["nominal"]])
                 for name, (lo, hi) in zip(input_names, bounds))


def response_table_design(stages=None):
    """Show preparation size without loading a model or running transport."""
    if stages is None:
        import stages as stage_definitions
        stages = stage_definitions.STATE_IDS
    names = list(uc.TRANSPORT_SURROGATE_INPUTS)
    bounds = [(uc.TRANSPORT_SURROGATE_INPUTS[name]["minimum"], uc.TRANSPORT_SURROGATE_INPUTS[name]["maximum"])
              for name in names]
    knots = response_grid(names, bounds)
    count = int(np.prod([len(axis) for axis in knots]))
    return {"input_knots": dict(zip(names, [len(axis) for axis in knots])),
            "conditions": count, "stage_evaluations": count * len(stages)}


def _ratios(metrics):
    # Peak car diagnostic retained in saved tables; annual carbon is priced by
    # simulation_engine using its current car and rail emissions assumptions.
    metrics["car_co2_tonnes_peak"] = metrics["co2_tonnes"]
    modes = ("car", "pt", "bike", "walk")
    total = sum(metrics[f"{mode}_trips"] for mode in modes)
    pkm = sum(metrics[f"strategic_{mode}_person_km"] for mode in modes)
    metrics["total_trips"] = total
    for mode in modes:
        metrics[f"{mode}_share_trips"] = metrics[f"{mode}_trips"] / max(total, 1e-9)
        metrics[f"{mode}_share"] = metrics[f"strategic_{mode}_person_km"] / max(pkm, 1e-9)
    metrics["avg_tt_min"] = 60 * (metrics["car_tt_hours"] + metrics["pt_tt_hours"]) / max(
        metrics["car_trips"] + metrics["pt_trips"], 1.)
    return metrics


@dataclass
class ResponseTable:
    knots: tuple
    values: np.ndarray
    metadata: dict

    def __post_init__(self):
        self.knots = tuple(np.asarray(axis, dtype=float) for axis in self.knots)
        self.values = np.asarray(self.values, dtype=float)
        self.stages = tuple(map(int, self.metadata["stages"]))
        self.metric_names = tuple(self.metadata["metric_names"])
        self.components = tuple(self.metadata["components"])
        width = len(self.stages) * (len(self.metric_names) + 2 * len(self.components))
        expected = tuple(len(axis) for axis in self.knots) + (width,)
        if (self.values.shape != expected or not np.isfinite(self.values).all()
                or len(self.knots) != len(self.metadata["input_names"])
                or any(axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all()
                       or np.any(np.diff(axis) <= 0) for axis in self.knots)):
            raise ValueError("Invalid transport response table dimensions or values.")
        self._corners = tuple(product((0, 1), repeat=len(self.knots)))
        self._cache = OrderedDict()

    def _interpolate(self, point):
        lower, fractions = [], []
        for axis, value in zip(self.knots, point):
            if not np.isfinite(value) or value < axis[0] - 1e-12 or value > axis[-1] + 1e-12:
                raise ValueError("Transport inputs lie outside the prepared response table.")
            value = min(max(value, axis[0]), axis[-1])
            upper = min(max(int(np.searchsorted(axis, value)), 1), len(axis) - 1)
            lo = upper - 1
            lower.append(lo)
            fractions.append((value - axis[lo]) / (axis[upper] - axis[lo]))
        result = np.zeros(self.values.shape[-1], dtype=float)
        for corner in self._corners:
            weight = float(np.prod([fraction if side else 1 - fraction
                                    for side, fraction in zip(corner, fractions)]))
            if weight:
                result += weight * self.values[tuple(lo + side for lo, side in zip(lower, corner))]
        return result

    def state(self, stage, inputs, *, include_welfare=False):
        stage = int(stage)
        if stage not in self.stages:
            raise ValueError(f"Stage {stage} is absent from the response table.")
        point = tuple(float(inputs[name]) for name in self.metadata["input_names"])
        if point not in self._cache:
            self._cache[point] = self._interpolate(point)
            if len(self._cache) > 512:
                self._cache.popitem(last=False)
        values = self._cache[point]
        self._cache.move_to_end(point)
        n_metrics, n_components = len(self.metric_names), len(self.components)
        offset = self.stages.index(stage) * (n_metrics + 2 * n_components)
        metrics = dict(self.metadata["constant_metrics"][str(stage)])
        metrics.update(zip(self.metric_names, map(float, values[offset:offset + n_metrics])))
        state = {"schema_version": 1, "source": "surrogate_msa", "stage": stage,
                 "metrics": _ratios(metrics), "input_values": dict(zip(self.metadata["input_names"], point)),
                 "surrogate_representation": SCHEMA}
        if include_welfare:
            start = offset + n_metrics
            state["welfare_summary"] = {
                "schema": "matched_moments_v1", "table_id": self.metadata["table_id"],
                "stage": stage, "input_values": dict(state["input_values"]),
                "section_signature": metrics.get("section_signature", "none"),
                "components": list(self.components),
                "delta_hours": dict(zip(self.components, map(float, values[start:start + n_components]))),
                "project_hours": dict(zip(self.components, map(float, values[start + n_components:start + 2 * n_components]))),
            }
        return state

    def save(self, path):
        path = Path(path)
        temporary = path.with_suffix(".tmp.npz")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, values=self.values,
                metadata=np.asarray(json.dumps(self.metadata, sort_keys=True, allow_nan=False)),
                **{f"knots_{i}": axis for i, axis in enumerate(self.knots)})
        temporary.replace(path)


def load_response_table(directory, manifest):
    from .transport_surrogate import SurrogateError, _file_digest
    path = Path(directory) / FILENAME
    record = manifest.get("response_table", {})
    if not path.is_file() or not record:
        raise SurrogateError("Prepare the transport response table in Notebook 03, Section 3.1 before running later analyses.")
    if record.get("fingerprint") != _fingerprint(manifest):
        raise SurrogateError("The transport response table needs updating. Run its preparation cell in Notebook 03, Section 3.1.")
    stat = path.stat()
    if _file_digest(str(path), stat.st_size, stat.st_mtime_ns) != record.get("sha256"):
        raise SurrogateError("The response table and manifest differ. Prepare the table in Notebook 03, Section 3.1.")
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"]))
        # A migrated manifest keeps the ids stored in the unchanged table payload.
        table_ids = {record["fingerprint"], *record.get("fingerprint_history", [])}
        if (meta.get("schema") != SCHEMA or meta.get("table_id") not in table_ids
                or meta.get("input_names") != manifest["input_names"]
                or meta.get("stages") != manifest["stages"]):
            raise SurrogateError("The response table belongs to a different transport model.")
        return ResponseTable(tuple(data[f"knots_{i}"].copy() for i in range(len(meta["input_names"]))),
                             data["values"].copy(), meta)


def _moments(baseline, project, components):
    if not baseline.get("od_keys") or baseline["od_keys"] != project.get("od_keys"):
        raise ValueError("Response preparation requires a common OD layout across stages.")
    weights = {mode: .5 * (baseline["quantities"][mode] + project["quantities"][mode])
               for mode in dict.fromkeys(mode for mode, _ in components.values())}
    delta, current = [], []
    for name, (mode, _) in components.items():
        t0, ts = baseline["times"][name], project["times"][name]
        delta.append(float(np.einsum("i,i->", weights[mode], t0 - ts)))
        current.append(float(np.einsum("i,i->", weights[mode], ts)))
    return delta, current


def prepare_response_table(project_root=None, *, force=False, progress=True, artifact_dir=None):
    """Prepare all stages on the configured input grid, including nominal conditions."""
    import parameters as p
    import simulation_engine as m
    import transport_model_interface as tmi
    from threadpoolctl import threadpool_limits
    from .prepared_transport import PreparedTransport
    from .transport_surrogate import load_transport_surrogate, SurrogateError, _file_digest

    emulator = load_transport_surrogate(project_root, artifact_dir=artifact_dir, use_response_table=False)
    directory = emulator.artifact_dir
    if not force:
        try:
            return load_response_table(directory, emulator.manifest)
        except SurrogateError:
            pass
    started = time.perf_counter()
    identity = _fingerprint(emulator.manifest)
    names = emulator.input_names
    bundle = emulator._get_bundle()
    bounds = np.asarray(bundle["input_bounds"], dtype=float).reshape(-1, 2)
    knots = response_grid(names, bounds)
    if any(axis[0] != bounds[i, 0] or axis[-1] != bounds[i, 1] for i, axis in enumerate(knots)):
        raise ValueError("Nominal transport inputs must lie within the fitted domain.")
    stage_ids = tuple(sorted(map(int, emulator.manifest["stages"])))
    if not stage_ids or stage_ids[0] != 0:
        raise ValueError("Response preparation requires baseline stage 0.")
    components = dict(m.WELFARE_COMPONENTS)
    backend = emulator._get_backend()
    prepared = {}
    shape = tuple(len(axis) for axis in knots)
    n_nodes = int(np.prod(shape))
    if progress:
        print(f"Preparing transport response table: {n_nodes} conditions × {len(stage_ids)} stages.", flush=True)
    with threadpool_limits(limits=1):
        for stage in stage_ids:
            corridor = backend["corridors"][stage]
            mode = tmi.run_transport_mode_choice(backend["context"], stage=stage,
                stage_specs=backend["stage_specs"], corridor_zone_ids=corridor.zone_ids,
                demand_multiplier=1., pt_asc_shift=0., bike_asc_shift=0.)
            prepared[stage] = PreparedTransport(backend["context"], corridor, mode,
                                              backend["stage_specs"][stage], bundle["od_labels"])
            del mode
        n_labels = len(bundle["od_labels"])
        reachable = np.asarray(bundle.get("reachable", np.ones((n_labels, n_labels), dtype=bool)), dtype=bool).reshape(n_labels, n_labels)
        metric_names, constants, values = None, {}, None
        for completed, index in enumerate(np.ndindex(shape), 1):
            point = np.array([axis[i] for axis, i in zip(knots, index)])
            inputs = dict(zip(names, map(float, point)))
            unit = (point - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
            delays = bundle["model"].predict(np.repeat(unit[None, :], len(stage_ids), axis=0),
                        np.asarray(stage_ids), clip_nonnegative=True).reshape(len(stage_ids), n_labels, n_labels)
            delays[:, ~reachable] = 0.
            row, baseline = [], None
            for stage, delay in zip(stage_ids, delays):
                np.fill_diagonal(delay, 0.)
                state = prepared[stage].evaluate(delay, inputs, return_welfare=True)
                metrics = state["metrics"]
                numeric = [key for key, value in metrics.items()
                           if isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_))]
                static = {key: value for key, value in metrics.items() if key not in numeric}
                if metric_names is None:
                    metric_names = numeric
                    width = len(stage_ids) * (len(metric_names) + 2 * len(components))
                    values = np.empty(shape + (width,), dtype=float)
                if numeric != metric_names:
                    raise ValueError("Stages must expose the same transport metric schema.")
                if completed == 1:
                    constants[str(stage)] = static
                elif constants[str(stage)] != static:
                    raise ValueError("Section identities changed during response preparation.")
                if stage == 0:
                    baseline = state["welfare_od"]
                delta, current = _moments(baseline, state["welfare_od"], components)
                row.extend([float(metrics[key]) for key in metric_names] + delta + current)
                del state
            if not np.isfinite(row).all():
                raise ValueError(f"Non-finite transport response at {inputs}.")
            values[index] = row
            del baseline
            if progress and (completed % 16 == 0 or completed == n_nodes):
                print(f"Response table: {completed}/{n_nodes} conditions ({time.perf_counter() - started:.0f}s).", flush=True)
    if identity != _fingerprint(emulator.manifest):
        raise RuntimeError("Physical response inputs changed during preparation; prepare the table again.")
    meta = {"schema": SCHEMA, "table_id": identity, "input_names": names,
            "stages": list(stage_ids), "metric_names": metric_names,
            "constant_metrics": constants, "components": list(components),
            "n_conditions": n_nodes, "preparation_seconds": time.perf_counter() - started,
            "nominal_inputs": {name: float(uc.TRANSPORT_SURROGATE_INPUTS[name]["nominal"]) for name in names},
            "interpolation": "multilinear physical metrics and matched OD/submode moments"}
    table = ResponseTable(knots, values, meta)
    path = directory / FILENAME
    table.save(path)
    stat = path.stat()
    manifest = dict(emulator.manifest)
    manifest["response_table"] = {"file": FILENAME, "schema": SCHEMA, "fingerprint": identity,
        "sha256": _file_digest(str(path), stat.st_size, stat.st_mtime_ns),
        "n_conditions": n_nodes, "preparation_seconds": meta["preparation_seconds"]}
    temporary = directory / "response_manifest.tmp.json"
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(directory / "manifest.json")
    return table
