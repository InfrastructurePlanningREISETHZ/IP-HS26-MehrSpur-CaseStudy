"""Prepared NumPy mode choice and accounting for repeated OD-delay queries.

Preparation aligns the fixed stage skims and masks once. Each query evaluates
all five native alternatives for every positive-demand OD affecting the cordon
or configured counting section. Zero-demand and irrelevant ODs contribute
nothing to these outputs and are omitted; no positive flows are cut or sampled.
The utility, availability, annual-appraisal anchors and section definitions are
those used by transport_model_interface. Matched welfare arrays remain exact
zone/submode quantities and hours, without repeated compression.
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pandas as pd


def _dot(a, b):
    return float(np.einsum("i,i->", a, b, optimize=False))


class PreparedTransport:
    """Prepared stage arrays with demand, preference, e-bike and delay inputs."""

    def __init__(self, context, corridor, base_mode, stage_spec, delay_labels):
        from additional import section_flows as sf
        import parameters as p
        from transport_model_interface import ASSIGNMENT_SETTINGS, _technology_base_components

        started = time.perf_counter()
        self.stage = int(base_mode.scenario.get("stage", 0))
        self.labels = pd.Index(sorted(map(str, base_mode.drive_od.index)))
        self.delay_labels = pd.Index(map(str, delay_labels))
        if self.labels.has_duplicates or self.delay_labels.has_duplicates:
            raise ValueError("Unique labels required.")
        self.n_labels = len(self.labels)
        self.n_delay = len(self.delay_labels)
        native = context.modules["mode_choice_zurich"]
        self.modes = tuple(native.MODE_ORDER)
        self.position = {m: i for i, m in enumerate(self.modes)}
        pars = native.load_mode_choice_parameters()
        pars.update({k: base_mode.scenario[k] for k in pars if k in base_mode.scenario})
        self.parameters = pars
        self.no_path = float(pars.get("NO_PATH_TIME_MIN", native.NO_PATH_TIME_MIN))
        self.utility_clip = float(pars.get("MAX_UTILITY_CLIP", 700.0))

        def full(frame, missing=0.0):
            if frame is None:
                return np.full((self.n_labels, self.n_labels), missing)
            frame = frame.rename(index=str, columns=str)
            return frame.reindex(index=self.labels, columns=self.labels,
                                 fill_value=missing).to_numpy(dtype=float)

        total = np.zeros((self.n_labels, self.n_labels))
        for frame in base_mode.od_by_mode.values():
            total += full(frame)
        if not np.isfinite(total).all() or (total < 0).any():
            raise ValueError("Finite nonnegative baseline person demand required.")
        internal = self.labels.isin(set(map(str, corridor.zone_ids)))
        involved = internal[:, None] | internal[None, :]
        config = sf.section_config(base_mode.scenario.get("section_config"))
        section_masks = sf.coverage_masks(self.labels, config)
        relevant = involved.copy()
        for mask in section_masks.values():
            relevant |= mask
        support = relevant & (total > 0.0)
        self.flat_indices = np.flatnonzero(support)
        self.row, self.col = np.unravel_index(self.flat_indices, total.shape)
        self.baseline_total_od = total.ravel()[self.flat_indices].copy()
        self.involved = involved.ravel()[self.flat_indices]
        self.section_masks = {m: a.ravel()[self.flat_indices] for m, a in section_masks.items()}
        self.appraisal_masks = {
            m: self.involved | self.section_masks.get(m, False) for m in self.modes
        }
        digest = hashlib.sha256(json.dumps(list(self.labels)).encode("utf-8"))
        digest.update(self.flat_indices.astype("<i8").tobytes())
        self.od_keys = digest.hexdigest()

        def vector(frame, missing=0.0, sanitize=False):
            values = full(frame, missing).ravel()[self.flat_indices].copy()
            if sanitize:
                np.nan_to_num(values, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            return values

        uncongested = base_mode.uncongested_travel_times or base_mode.travel_times
        lengths = base_mode.lengths
        get_time = lambda key, missing=0.0: vector(uncongested.get(key), missing)
        drive = get_time("drive", self.no_path)
        walk = get_time("walk", self.no_path)
        bike = get_time("bike", self.no_path)
        pw_i, pw_o = get_time("ivt_pt_walk", self.no_path), get_time("ovt_pt_walk", self.no_path)
        pb_i, pb_o = get_time("ivt_pt_bike", self.no_path), get_time("ovt_pt_bike", self.no_path)
        technology_base = _technology_base_components(context, base_mode)
        self.raw_bike = vector(technology_base.get("bike"), self.no_path)
        self.raw_pt_bike = {part: vector(technology_base.get(f"{part}_pt_bike"), self.no_path)
                            for part in ("access", "egress")}
        self.fixed_pt_bike = {part: get_time(f"{part}_pt_bike", self.no_path)
                              for part in ("ivt", "initial_wait", "transfer_physical", "transfer_wait")}
        self.ebike_speed_multiplier = float(stage_spec.get("EBIKE_SPEED_MULTIPLIER", 1.5))
        if not np.isfinite(self.ebike_speed_multiplier) or self.ebike_speed_multiplier <= 0:
            raise ValueError("E-bike speed multiplier must be finite and positive.")
        native_times = context.modules["travel_times"]
        self.ebike_time_factor = native_times.ebike_time_factor
        self.pt_access_floor = float(getattr(native_times, "PT_ACCESS_FLOOR_MIN", 2.0))
        self.pt_egress_floor = float(getattr(native_times, "PT_EGRESS_FLOOR_MIN", 2.0))
        self.pt_intrazonal_floor = float(getattr(native_times, "PT_INTRAZONAL_MIN", 5.0))
        self.non_diagonal = self.row != self.col
        self.section_config = config
        self.section_saving = float(stage_spec.get("section_time_saving_min", 0.0)) if config["active"] else 0.0
        drive_dist = vector(lengths.get("drive"))
        pw_dist, pb_dist = vector(lengths.get("pt_walk")), vector(lengths.get("pt_bike"))
        fare = lambda d: np.minimum(float(pars["PT_MIN_FARE_CHF"]) + d / 1000.0 * float(pars["PT_FARE_CHF_PER_KM"]), float(pars["PT_MAX_FARE_CHF"]))
        utilities = {
            "drive": float(pars["ASC_CAR"]) + float(pars["B_CAR_TIME"]) * drive + float(pars["B_COST"]) * (drive_dist / 1000.0 * float(pars["CAR_COST_CHF_PER_KM"])),
            "walk": float(pars["ASC_WALK"]) + float(pars["B_WALK_TIME"]) * walk,
            "bike": float(pars["ASC_BIKE"]) + float(pars["B_BIKE_TIME"]) * bike,
            "pt_walk": float(pars["ASC_PT_WALK"]) + float(pars["B_PT_IVT"]) * pw_i + float(pars["B_PT_OVT"]) * pw_o + float(pars["B_PT_FARE"]) * fare(pw_dist) + float(pars["B_TRANSFER"]) * get_time("transfer_count_pt_walk"),
            "pt_bike": float(pars["ASC_PT_BIKE"]) + float(pars["B_PT_IVT"]) * pb_i + float(pars["B_PT_OVT"]) * pb_o + float(pars["B_PT_FARE"]) * fare(pb_dist) + float(pars["B_TRANSFER"]) * get_time("transfer_count_pt_bike"),
        }
        available = lambda *arrays: np.logical_and.reduce([np.isfinite(a) & (a < self.no_path) for a in arrays])
        allowed = context.modules["travel_times"].standalone_walk_mask(lengths, context.zones)
        availability = {
            "drive": available(drive), "walk": available(walk) & vector(allowed).astype(bool),
            "bike": available(bike), "pt_walk": available(pw_i, pw_o),
            "pt_bike": available(pb_i, pb_o),
        }
        self.utilities = np.stack([utilities[m] for m in self.modes])
        self.pt_bike_utility_without_ovt = (float(pars["ASC_PT_BIKE"]) + float(pars["B_PT_IVT"]) * pb_i
            + float(pars["B_PT_FARE"]) * fare(pb_dist) + float(pars["B_TRANSFER"]) * get_time("transfer_count_pt_bike"))
        self.availability = np.stack([availability[m] for m in self.modes])
        self.drive_skim = drive
        self.drive_km = vector(context.lengths.get("drive", context.lengths.get("car", lengths.get("drive"))), sanitize=True) / 1000.0
        self.bike_km = vector(lengths.get("bike"), sanitize=True) / 1000.0
        self.walk_km = vector(lengths.get("walk"), sanitize=True) / 1000.0
        min_dist = stage_spec.get("_min_distance_km")
        if min_dist is None:
            min_dist = getattr(p, "MIN_STRATEGIC_PKM_DISTANCE", 5.0)
        self.strategic = self.involved & (self.drive_km >= float(min_dist))
        self.occupancy = float((base_mode.assigned_metadata or {}).get("drive_occupancy", ASSIGNMENT_SETTINGS.get("drive_occupancy", 1.14)))
        if self.occupancy <= 0 or not np.isfinite(self.occupancy):
            raise ValueError("Positive finite occupancy required.")
        self.times = {"car_freeflow": vector(sf.physical_car_times(context, base_mode), sanitize=True) / 60.0,
                      "bike_time": vector(uncongested.get("bike"), sanitize=True) / 60.0,
                      "walk_time": vector(uncongested.get("walk"), sanitize=True) / 60.0}
        for mode in ("pt_walk", "pt_bike"):
            for component, suffixes in {
                "ivt": ("ivt",), "wait": ("initial_wait", "transfer_wait"),
                "access": ("access",), "egress": ("egress",),
                "transfer_walk": ("transfer_physical",),
            }.items():
                values = np.zeros(len(self.flat_indices))
                for suffix in suffixes:
                    frame = uncongested.get(f"{suffix}_{mode}")
                    if suffix == "ivt" and frame is None:
                        frame = uncongested.get(mode)
                    values += vector(frame, sanitize=True) / 60.0
                self.times[f"{mode}_{component}"] = values

        self._gate_index = {str(label): i for i, label in enumerate(self.delay_labels)}
        self._internal = set(map(str, corridor.zone_ids))
        self._entry = corridor.external_zone_to_entry_gate
        self._exit = corridor.external_zone_to_exit_gate
        self.delay_positions, self.delay_valid = self._delay_indices(self.labels[self.row], self.labels[self.col])

        # One native preparation preserves the selected section/reference rules;
        # only its quantities are replaced in subsequent vector evaluations.
        self.section_template = sf.section_metrics(context, base_mode)
        self.section_template["section_modeled_trips_peak"] = 0.0
        self.section_template["section_modeled_person_hours_peak"] = 0.0
        self.section_hours = {}
        reference = float(self.section_template["section_reference_minutes"])
        for mode, mask in self.section_masks.items():
            if mode.startswith("pt_"):
                raw = vector(uncongested.get(f"ivt_{mode}"))
                valid = mask & np.isfinite(raw) & (raw >= 0) & (raw < 999)
                hours = np.zeros(len(raw))
                hours[valid] = np.minimum(raw[valid], reference) / 60.0
            else:
                values = self.times["car_freeflow" if mode == "drive" else f"{mode}_time"] * 60
                hours = np.minimum(values, reference) / 60.0
            self.section_hours[mode] = hours
        zeros = np.zeros(len(self.flat_indices))
        for mode in ("pt_walk", "pt_bike"):
            self.times[f"{mode}_section"] = self.section_hours.get(mode, zeros) if config["mode"] == "PT" else zeros
        self.reference_delay_terms = []
        self.reference_bike_terms = []
        self.section_no_path = float(getattr(context.modules.get("config"), "NO_PATH_TIME_MIN", 999.0))
        if config["active"] and config["mode"] == "CAR":
            selected = sf.section_reference(context, base_mode, config)
            ref_labels = context.baseline_od.index.astype(str)
            for mode, (row, col, weights, old) in selected.items():
                if mode == "drive":
                    positions, valid = self._delay_indices(ref_labels[row], ref_labels[col])
                    free = uncongested["drive"].rename(index=str, columns=str).reindex(
                        index=ref_labels, columns=ref_labels).to_numpy(dtype=float)[row, col]
                    self.reference_delay_terms.append((positions, valid, weights, free))
        if config["active"] and config["mode"] == "BIKE":
            selected = sf.section_reference(context, base_mode, config)
            ref_labels = context.baseline_od.index.astype(str)
            for mode, (row, col, weights, old) in selected.items():
                if mode == "bike":
                    raw = technology_base["bike"].rename(index=str, columns=str).reindex(
                        index=ref_labels, columns=ref_labels).to_numpy(dtype=float)[row, col]
                    self.reference_bike_terms.append((raw, weights))
        # Reuse immutable fixed components across annual welfare states.
        for values in self.times.values():
            values.flags.writeable = False
        self.prepare_seconds = time.perf_counter() - started

    def _delay_indices(self, origins, destinations):
        origin = np.fromiter((self._gate_index.get(str(z) if str(z) in self._internal else self._entry.get(str(z)), -1) for z in origins), dtype=np.int64)
        destination = np.fromiter((self._gate_index.get(str(z) if str(z) in self._internal else self._exit.get(str(z)), -1) for z in destinations), dtype=np.int64)
        valid = (origin >= 0) & (destination >= 0)
        return np.maximum(origin, 0) * self.n_delay + np.maximum(destination, 0), valid

    def _technology_times(self, share):
        """Match native e-bike scaling, PT component floors and section savings."""
        share = float(share)
        if not np.isfinite(share) or not 0.0 <= share <= 1.0:
            raise ValueError("E-bike share must be finite and between zero and one.")
        factor = self.ebike_time_factor(share, self.ebike_speed_multiplier)
        valid = lambda values: np.isfinite(values) & (values < self.no_path)
        bike = self.raw_bike.copy()
        bike[valid(bike)] *= factor
        if self.section_config["active"] and self.section_config["mode"] == "BIKE":
            covered = self.section_masks.get("bike", False) & valid(bike) & (bike >= 0)
            bike[covered] = np.maximum(bike[covered] - self.section_saving, 0.0)
        components = {**self.fixed_pt_bike, **{key: value.copy() for key, value in self.raw_pt_bike.items()}}
        available = np.logical_and.reduce([valid(value) for value in components.values()])
        if share > 0.0:
            for part, floor in (("access", self.pt_access_floor), ("egress", self.pt_egress_floor)):
                values = components[part]
                values[available] *= factor
                floors = available & self.non_diagonal & (values < floor)
                values[floors] = floor
        ovt = sum(value for part, value in components.items() if part != "ivt")
        ovt[~available] = self.no_path
        intrazonal = ~self.non_diagonal & available
        ovt[intrazonal] = np.maximum(ovt[intrazonal], self.pt_intrazonal_floor)
        return bike, components["access"], components["egress"], ovt, factor

    def evaluate(self, delay_array, input_values, *, return_welfare=True):
        if isinstance(delay_array, pd.DataFrame):
            if not delay_array.index.astype(str).equals(self.delay_labels) or not delay_array.columns.astype(str).equals(self.delay_labels):
                raise ValueError("Delay labels must match the prepared order.")
            delay_array = delay_array.to_numpy(dtype=float)
        table = np.asarray(delay_array, dtype=float)
        if table.shape != (self.n_delay, self.n_delay) or not np.isfinite(table).all():
            raise ValueError("Finite delay matrix with prepared dimensions required.")
        table = np.maximum(table, 0.0).copy()
        np.fill_diagonal(table, 0.0)
        delay = table.ravel()[self.delay_positions]
        delay[~self.delay_valid] = 0.0
        from additional.uncertainty import complete_transport_inputs
        input_values = complete_transport_inputs(input_values)
        demand = self.baseline_total_od
        multiplier = 1.0 + float(input_values["PASSENGER_DEMAND_GROWTH"])
        if not np.isfinite(multiplier) or multiplier < 0:
            raise ValueError("Finite nonnegative demand multiplier required.")
        utility = self.utilities.copy()
        bike, access, egress, pt_bike_ovt, bike_factor = self._technology_times(input_values["EBIKE_SHARE"])
        times = {**self.times, "bike_time": np.nan_to_num(bike, nan=0.0, posinf=0.0, neginf=0.0) / 60.0,
                 "pt_bike_access": np.nan_to_num(access, nan=0.0, posinf=0.0, neginf=0.0) / 60.0,
                 "pt_bike_egress": np.nan_to_num(egress, nan=0.0, posinf=0.0, neginf=0.0) / 60.0}
        utility[self.position["bike"]] = float(self.parameters["ASC_BIKE"]) + float(self.parameters["B_BIKE_TIME"]) * bike
        utility[self.position["pt_bike"]] = self.pt_bike_utility_without_ovt + float(self.parameters["B_PT_OVT"]) * pt_bike_ovt
        drive_i = self.position["drive"]
        utility[drive_i] += float(self.parameters["B_CAR_TIME"]) * delay
        utility[self.position["bike"]] += float(input_values["BIKE_ASC_SHIFT"])
        for mode in ("pt_walk", "pt_bike"):
            utility[self.position[mode]] += float(input_values["PT_ASC_SHIFT"])
        available = self.availability.copy()
        available[self.position["bike"]] = np.isfinite(bike) & (bike < self.no_path)
        available[self.position["pt_bike"]] = (np.isfinite(self.fixed_pt_bike["ivt"])
            & (self.fixed_pt_bike["ivt"] < self.no_path) & np.isfinite(pt_bike_ovt) & (pt_bike_ovt < self.no_path))
        available[drive_i] = np.isfinite(self.drive_skim + delay) & (self.drive_skim + delay < self.no_path)
        utility = np.where(available, utility, -1e9)
        exponents = np.where(available, np.exp(np.clip(utility - utility.max(axis=0), -self.utility_clip, self.utility_clip)), 0.0)
        denominator = exponents.sum(axis=0)
        probabilities = np.divide(exponents, denominator, out=np.zeros_like(exponents), where=denominator > 0)
        fallback = denominator <= 0
        if fallback.any():
            # Native fallback chooses first available drive or the maximum
            # utility; with all unavailable, native MODE_ORDER's first mode wins.
            positions = np.flatnonzero(fallback)
            best = np.argmax(utility[:, fallback], axis=0)
            probabilities[:, fallback] = 0.0
            probabilities[best, positions] = 1.0
        quantities = probabilities * (demand * multiplier)
        q = {m: quantities[i] for i, m in enumerate(self.modes)}
        delay_hours = delay / 60.0
        hours = lambda mode, values, appraisal=False: _dot(q[mode], values * (self.appraisal_masks[mode] if appraisal else self.involved))
        trips = {m: _dot(q[m], self.involved) for m in self.modes}
        pt_trips = trips["pt_walk"] + trips["pt_bike"]
        total = sum(trips.values())
        pkm = {m: _dot(q[m], self.drive_km * self.strategic) for m in self.modes}
        total_pkm = sum(pkm.values())
        car_person_km = _dot(q["drive"], self.drive_km * self.involved)
        car_vkt = car_person_km / self.occupancy
        car_hours = hours("drive", times["car_freeflow"])
        pt_hours = {component: sum(hours(mode, times[f"{mode}_{component}"]) for mode in ("pt_walk", "pt_bike")) for component in ("ivt", "wait", "access", "egress", "transfer_walk")}
        metrics = {
            "total_trips": total, "car_trips": trips["drive"], "pt_trips": pt_trips,
            "bike_trips": trips["bike"], "walk_trips": trips["walk"],
            "car_share_trips": trips["drive"] / max(total, 1e-9), "pt_share_trips": pt_trips / max(total, 1e-9),
            "bike_share_trips": trips["bike"] / max(total, 1e-9), "walk_share_trips": trips["walk"] / max(total, 1e-9),
            "car_share": pkm["drive"] / max(total_pkm, 1e-9), "pt_share": (pkm["pt_walk"] + pkm["pt_bike"]) / max(total_pkm, 1e-9),
            "bike_share": pkm["bike"] / max(total_pkm, 1e-9), "walk_share": pkm["walk"] / max(total_pkm, 1e-9),
            "strategic_car_person_km": pkm["drive"],
            "strategic_pt_person_km": pkm["pt_walk"] + pkm["pt_bike"],
            "strategic_bike_person_km": pkm["bike"], "strategic_walk_person_km": pkm["walk"],
            "car_dist_km": car_vkt, "car_vehicle_km": car_vkt, "car_person_km": car_person_km,
            "car_vehicle_trips": trips["drive"] / self.occupancy, "drive_occupancy": self.occupancy,
            "pt_dist_km": pkm["pt_walk"] + pkm["pt_bike"], "car_tt_hours": car_hours,
            "car_delay_person_hours": hours("drive", delay_hours), "congestion_delay_hours": hours("drive", delay_hours),
            "pt_tt_hours": pt_hours["ivt"], "pt_wait_hours": pt_hours["wait"],
            "pt_access_hours": pt_hours["access"], "pt_egress_hours": pt_hours["egress"],
            "pt_transfer_walk_hours": pt_hours["transfer_walk"],
            "bike_tt_hours": hours("bike", times["bike_time"]), "walk_tt_hours": hours("walk", times["walk_time"]),
            "bike_dist_km": _dot(q["bike"], self.bike_km * self.involved), "walk_dist_km": _dot(q["walk"], self.walk_km * self.involved),
            "co2_tonnes": car_vkt * 0.139 / 1000.0,
            "car_co2_tonnes_peak": car_vkt * 0.139 / 1000.0,
            "avg_tt_min": (car_hours + pt_hours["ivt"]) / max(trips["drive"] + pt_trips, 1.0) * 60.0,
        }
        for key, modes in {"car": ("drive",), "pt": ("pt_walk", "pt_bike"), "bike": ("bike",), "walk": ("walk",)}.items():
            metrics[f"appraisal_{key}_trips"] = sum(_dot(q[m], self.appraisal_masks[m]) for m in modes)
        metrics["appraisal_car_tt_hours"] = hours("drive", times["car_freeflow"], True)
        metrics["appraisal_car_delay_person_hours"] = hours("drive", delay_hours, True)
        for mode in ("bike", "walk"):
            metrics[f"appraisal_{mode}_tt_hours"] = hours(mode, times[f"{mode}_time"], True)
        for component, key in {"ivt": "tt", "wait": "wait", "access": "access", "egress": "egress", "transfer_walk": "transfer_walk"}.items():
            metrics[f"appraisal_pt_{key}_hours"] = sum(hours(mode, times[f"{mode}_{component}"], True) for mode in ("pt_walk", "pt_bike"))
        section = dict(self.section_template)
        section_hours = self.section_hours
        if self.reference_bike_terms:
            reference = sum(_dot(np.maximum(raw * bike_factor - self.section_saving, 0.0), weights)
                            for raw, weights in self.reference_bike_terms)
            section["section_reference_minutes"] = reference
            section_hours = {**section_hours, "bike": np.minimum(times["bike_time"], reference / 60.0)}
        for mode, mask in self.section_masks.items():
            section["section_modeled_trips_peak"] += _dot(q[mode], mask)
            section["section_modeled_person_hours_peak"] += _dot(q[mode], section_hours[mode] * mask)
        if self.reference_delay_terms:
            section["section_reference_delay_minutes"] = 0.0
            for positions, valid, weights, free in self.reference_delay_terms:
                reference_delay = table.ravel()[positions] * valid
                actual = free + reference_delay
                if not np.isfinite(actual).all() or (actual >= self.section_no_path).any():
                    raise ValueError("Assigned car skim invalid for the fixed section reference cohort.")
                section["section_reference_delay_minutes"] += _dot(reference_delay, weights)
        metrics.update(section)
        state = {"schema_version": 1, "source": "surrogate_msa", "stage": self.stage,
                 "representation": "prepared native OD mode choice",
                 "metrics": metrics, "input_values": dict(input_values)}
        if return_welfare:
            keys = {"drive": "car", "pt_walk": "pt_walk", "pt_bike": "pt_bike", "bike": "bike", "walk": "walk"}
            state["welfare_od"] = {"od_keys": self.od_keys,
                "quantities": {keys[m]: q[m] * self.appraisal_masks[m] for m in self.modes},
                "times": {**times, "car_delay": delay_hours}}
        return state
