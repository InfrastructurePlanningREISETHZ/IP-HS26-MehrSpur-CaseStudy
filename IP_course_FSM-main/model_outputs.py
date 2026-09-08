from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from config import MODE_ORDER, NO_PATH_TIME_MIN

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPORTING_MODE_ORDER = ("drive", "pt", "bike", "walk")
MODE_LABELS = {"drive": "Car", "pt": "Public transport", "bike": "Bicycle", "walk": "Walk"}
MODE_COLORS = {"drive": "#4D4D4D", "pt": "#2F6B9A", "bike": "#3A8D5D", "walk": "#D69A2D"}
AREA_NAMES = {
    "all": "All modelled origins",
    "city": "City of Zurich (quartier origins)",
    "rest": "Rest of Canton Zurich (gemeinde origins)",
}
DISTANCE_BINS = (
    ("intrazonal / very short trips", 0.0, 0.0),
    ("0-2 km", 0.0, 2.0),
    ("2-5 km", 2.0, 5.0),
    ("5-10 km", 5.0, 10.0),
    ("10-25 km", 10.0, 25.0),
    ("25+ km", 25.0, np.inf),
)
PT_COMPONENTS = {
    "pt_ivt": "ivt_pt_{access}",
    "pt_access": "access_pt_{access}",
    "pt_egress": "egress_pt_{access}",
    "pt_initial_wait": "initial_wait_pt_{access}",
    "pt_transfer_wait": "transfer_wait_pt_{access}",
    "pt_transfer_physical": "transfer_physical_pt_{access}",
}
INTERVENTION_SUMMARY_COLUMNS = [
    "intervention_type",
    "intervention_level",
    "intervention_name",
    "comparison_basis",
    "affected_od_cells",
    "affected_total_trips",
    "mode",
    "counterfactual_trips",
    "intervention_trips",
    "trip_change",
    "counterfactual_share_pct",
    "intervention_share_pct",
    "share_change_pp",
    "counterfactual_mean_time_min",
    "intervention_mean_time_min",
    "mean_time_change_min",
    "counterfactual_mean_distance_km",
    "intervention_mean_distance_km",
    "mean_distance_change_km",
]
for _component in PT_COMPONENTS:
    INTERVENTION_SUMMARY_COLUMNS.extend(
        [
            f"counterfactual_{_component}_min",
            f"intervention_{_component}_min",
            f"{_component}_change_min",
        ]
    )


def _mode_matrices(
    od_by_mode: dict[str, pd.DataFrame],
    *,
    detailed: bool,
) -> dict[str, pd.DataFrame]:
    if detailed:
        return {mode: od_by_mode[mode] for mode in MODE_ORDER}
    return {
        "drive": od_by_mode["drive"],
        "pt": od_by_mode["pt_walk"] + od_by_mode["pt_bike"],
        "bike": od_by_mode["bike"],
        "walk": od_by_mode["walk"],
    }


def _zone_reporting_data(
    zones: pd.DataFrame,
    zone_ids: pd.Index,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    indexed = zones.copy()
    indexed.index = indexed["grid_id"].astype(str)
    indexed = indexed.reindex(zone_ids)
    if indexed["geometry"].isna().any():
        raise ValueError("Every model zone needs geometry for distance-bin reporting.")

    is_city = indexed["Level"].eq("Quartier").to_numpy(dtype=bool)
    area_masks = {
        "all": np.ones(len(indexed), dtype=bool),
        "city": is_city,
        "rest": ~is_city,
    }
    centroids = indexed["geometry"].map(lambda geometry: geometry.centroid)
    x = centroids.map(lambda point: point.x).to_numpy(dtype=float)
    y = centroids.map(lambda point: point.y).to_numpy(dtype=float)
    centroid_distance_km = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :]) / 1000.0
    return area_masks, centroid_distance_km


def mode_share_by_distance_bin(
    od_by_mode: dict[str, pd.DataFrame],
    zones: pd.DataFrame,
    *,
    detailed: bool = False,
) -> pd.DataFrame:
    """Summarise trips by centroid-distance bin and origin area."""
    matrices = _mode_matrices(od_by_mode, detailed=detailed)
    mode_order = MODE_ORDER if detailed else REPORTING_MODE_ORDER
    zone_ids = od_by_mode["drive"].index.astype(str)
    arrays = {
        mode: matrix.reindex(index=zone_ids, columns=zone_ids).to_numpy(dtype=float)
        for mode, matrix in matrices.items()
    }
    area_masks, centroid_distance_km = _zone_reporting_data(zones, zone_ids)
    diagonal = np.eye(len(zone_ids), dtype=bool)
    rows = []

    for area_key, origin_mask in area_masks.items():
        area_pair_mask = origin_mask[:, None]
        area_mode_totals = {
            mode: float(np.where(area_pair_mask, values, 0.0).sum())
            for mode, values in arrays.items()
        }
        area_total = sum(area_mode_totals.values())

        for label, lower, upper in DISTANCE_BINS:
            if label == "intrazonal / very short trips":
                bin_mask = diagonal
            else:
                bin_mask = (~diagonal) & (centroid_distance_km >= lower)
                if np.isfinite(upper):
                    bin_mask &= centroid_distance_km < upper
            mask = bin_mask & area_pair_mask
            bin_mode_totals = {
                mode: float(values[mask].sum())
                for mode, values in arrays.items()
            }
            bin_total = sum(bin_mode_totals.values())

            for mode in mode_order:
                trips = bin_mode_totals[mode]
                centroid_pkm = float((arrays[mode][mask] * centroid_distance_km[mask]).sum())
                rows.append(
                    {
                        "area": AREA_NAMES[area_key],
                        "area_basis": "origin zone",
                        "distance_bin": label,
                        "distance_bin_min_km": lower,
                        "distance_bin_max_km": upper if np.isfinite(upper) else np.nan,
                        "distance_bin_basis": "zone_centroid_distance_km",
                        "mode": mode,
                        "trips": trips,
                        "trip_modal_split_pct": 100.0 * trips / bin_total if bin_total else np.nan,
                        "share_of_mode_trips_pct": (
                            100.0 * trips / area_mode_totals[mode]
                            if area_mode_totals[mode]
                            else np.nan
                        ),
                        "bin_share_of_area_trips_pct": (
                            100.0 * bin_total / area_total if area_total else np.nan
                        ),
                        "centroid_passenger_km": centroid_pkm,
                        "mean_centroid_distance_km": centroid_pkm / trips if trips else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def mode_summary_by_area(
    od_by_mode: dict[str, pd.DataFrame],
    lengths: dict[str, pd.DataFrame],
    zones: pd.DataFrame,
    *,
    detailed: bool = False,
) -> pd.DataFrame:
    matrices = _mode_matrices(od_by_mode, detailed=detailed)
    mode_order = MODE_ORDER if detailed else REPORTING_MODE_ORDER
    zone_ids = od_by_mode["drive"].index.astype(str)
    area_masks, _ = _zone_reporting_data(zones, zone_ids)

    mode_pkm_arrays = {}
    for mode in MODE_ORDER:
        demand = od_by_mode[mode].reindex(index=zone_ids, columns=zone_ids).to_numpy(dtype=float)
        distance = lengths[mode].reindex(index=zone_ids, columns=zone_ids).to_numpy(dtype=float)
        mode_pkm_arrays[mode] = demand * np.where(np.isfinite(distance), distance, 0.0) / 1000.0
    if detailed:
        pkm_arrays = mode_pkm_arrays
    else:
        pkm_arrays = {
            "drive": mode_pkm_arrays["drive"],
            "pt": mode_pkm_arrays["pt_walk"] + mode_pkm_arrays["pt_bike"],
            "bike": mode_pkm_arrays["bike"],
            "walk": mode_pkm_arrays["walk"],
        }

    rows = []
    for area_key, origin_mask in area_masks.items():
        trips = {
            mode: float(matrix.reindex(index=zone_ids, columns=zone_ids).to_numpy(dtype=float)[origin_mask, :].sum())
            for mode, matrix in matrices.items()
        }
        passenger_km = {
            mode: float(values[origin_mask, :].sum())
            for mode, values in pkm_arrays.items()
        }
        total_trips = sum(trips.values())
        total_pkm = sum(passenger_km.values())
        for mode in mode_order:
            rows.append(
                {
                    "area": AREA_NAMES[area_key],
                    "area_basis": "origin zone",
                    "mode": mode,
                    "trips": trips[mode],
                    "trip_share_pct": 100.0 * trips[mode] / total_trips if total_trips else np.nan,
                    "passenger_km": passenger_km[mode],
                    "passenger_km_share_pct": (
                        100.0 * passenger_km[mode] / total_pkm if total_pkm else np.nan
                    ),
                    "mean_km_per_trip": (
                        passenger_km[mode] / trips[mode] if trips[mode] else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def mode_summary(
    od_by_mode: dict[str, pd.DataFrame],
    lengths: dict[str, pd.DataFrame],
    *,
    detailed: bool = False,
) -> pd.DataFrame:
    matrices = _mode_matrices(od_by_mode, detailed=detailed)
    mode_order = MODE_ORDER if detailed else REPORTING_MODE_ORDER
    trips = {
        mode: float(matrix.to_numpy(dtype=float).sum())
        for mode, matrix in matrices.items()
    }
    detailed_pkm = {}
    for mode in MODE_ORDER:
        demand = od_by_mode[mode].to_numpy(dtype=float)
        distance = lengths[mode].to_numpy(dtype=float)
        safe_distance = np.where(np.isfinite(distance), distance, 0.0)
        detailed_pkm[mode] = float((demand * safe_distance / 1000.0).sum())
    if detailed:
        passenger_km = detailed_pkm
    else:
        passenger_km = {
            "drive": detailed_pkm["drive"],
            "pt": detailed_pkm["pt_walk"] + detailed_pkm["pt_bike"],
            "bike": detailed_pkm["bike"],
            "walk": detailed_pkm["walk"],
        }

    total_trips = sum(trips.values())
    total_pkm = sum(passenger_km.values())
    rows = []
    for mode in mode_order:
        rows.append(
            {
                "mode": mode,
                "trips": trips[mode],
                "trip_share_pct": 100.0 * trips[mode] / total_trips if total_trips else np.nan,
                "passenger_km": passenger_km[mode],
                "passenger_km_share_pct": (
                    100.0 * passenger_km[mode] / total_pkm if total_pkm else np.nan
                ),
                "mean_km_per_trip": (
                    passenger_km[mode] / trips[mode] if trips[mode] else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _reporting_alternatives(mode: str) -> tuple[str, ...]:
    return ("pt_walk", "pt_bike") if mode == "pt" else (mode,)


def _masked_trips(
    od_by_mode: dict[str, pd.DataFrame],
    mode: str,
    mask: np.ndarray,
) -> float:
    return float(
        sum(
            od_by_mode[alternative].to_numpy(dtype=float)[mask].sum()
            for alternative in _reporting_alternatives(mode)
        )
    )


def _weighted_mean(
    od_by_mode: dict[str, pd.DataFrame],
    matrices: dict[str, pd.DataFrame],
    alternatives: tuple[str, ...],
    matrix_keys: tuple[str, ...],
    mask: np.ndarray,
    *,
    time_matrix: bool,
    scale: float = 1.0,
) -> float:
    weighted_sum = 0.0
    weight_sum = 0.0
    for alternative, key in zip(alternatives, matrix_keys):
        weights = od_by_mode[alternative].to_numpy(dtype=float)
        values = matrices[key].to_numpy(dtype=float) * scale
        valid = mask & np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
        if time_matrix:
            valid &= values < NO_PATH_TIME_MIN
        weighted_sum += float((weights[valid] * values[valid]).sum())
        weight_sum += float(weights[valid].sum())
    return weighted_sum / weight_sum if weight_sum else np.nan


def _mode_mean(
    od_by_mode: dict[str, pd.DataFrame],
    matrices: dict[str, pd.DataFrame],
    mode: str,
    mask: np.ndarray,
    *,
    time_matrix: bool,
) -> float:
    alternatives = _reporting_alternatives(mode)
    return _weighted_mean(
        od_by_mode,
        matrices,
        alternatives,
        alternatives,
        mask,
        time_matrix=time_matrix,
        scale=1.0 if time_matrix else 1.0 / 1000.0,
    )


def _pt_component_mean(
    od_by_mode: dict[str, pd.DataFrame],
    travel_times: dict[str, pd.DataFrame],
    component_template: str,
    mask: np.ndarray,
) -> float:
    alternatives = ("pt_walk", "pt_bike")
    keys = tuple(component_template.format(access=access) for access in ("walk", "bike"))
    return _weighted_mean(
        od_by_mode,
        travel_times,
        alternatives,
        keys,
        mask,
        time_matrix=True,
    )


def intervention_summary(
    intervention_od: dict[str, pd.DataFrame],
    intervention_times: dict[str, pd.DataFrame],
    intervention_lengths: dict[str, pd.DataFrame],
    counterfactual_od: dict[str, pd.DataFrame],
    counterfactual_times: dict[str, pd.DataFrame],
    counterfactual_lengths: dict[str, pd.DataFrame],
    coverage: list[dict],
) -> pd.DataFrame:
    """Summarise selected interventions against a matched no-intervention case."""
    if not coverage:
        return pd.DataFrame(columns=INTERVENTION_SUMMARY_COLUMNS)

    entries = list(coverage)
    if len(entries) > 1:
        union = np.logical_or.reduce([entry["mask"] for entry in entries])
        entries.append(
            {
                "intervention_type": "combined",
                "intervention_level": np.nan,
                "intervention_name": "All selected interventions",
                "mask": union,
            }
        )

    rows = []
    comparison = (
        "same scenario demand and final road skim; all selected interventions removed; "
        "mean times and distances use counterfactual mode-trip weights"
    )
    for entry in entries:
        mask = np.asarray(entry["mask"], dtype=bool)
        intervention_trips = {
            mode: _masked_trips(intervention_od, mode, mask)
            for mode in REPORTING_MODE_ORDER
        }
        counterfactual_trips = {
            mode: _masked_trips(counterfactual_od, mode, mask)
            for mode in REPORTING_MODE_ORDER
        }
        intervention_total = sum(intervention_trips.values())
        counterfactual_total = sum(counterfactual_trips.values())

        for mode in REPORTING_MODE_ORDER:
            current_share = (
                100.0 * intervention_trips[mode] / intervention_total
                if intervention_total
                else np.nan
            )
            counterfactual_share = (
                100.0 * counterfactual_trips[mode] / counterfactual_total
                if counterfactual_total
                else np.nan
            )
            current_time = _mode_mean(
                counterfactual_od,
                intervention_times,
                mode,
                mask,
                time_matrix=True,
            )
            counterfactual_time = _mode_mean(
                counterfactual_od,
                counterfactual_times,
                mode,
                mask,
                time_matrix=True,
            )
            current_distance = _mode_mean(
                counterfactual_od,
                intervention_lengths,
                mode,
                mask,
                time_matrix=False,
            )
            counterfactual_distance = _mode_mean(
                counterfactual_od,
                counterfactual_lengths,
                mode,
                mask,
                time_matrix=False,
            )
            row = {
                "intervention_type": entry["intervention_type"],
                "intervention_level": entry.get("intervention_level"),
                "intervention_name": entry["intervention_name"],
                "comparison_basis": comparison,
                "affected_od_cells": int(mask.sum()),
                "affected_total_trips": intervention_total,
                "mode": mode,
                "counterfactual_trips": counterfactual_trips[mode],
                "intervention_trips": intervention_trips[mode],
                "trip_change": intervention_trips[mode] - counterfactual_trips[mode],
                "counterfactual_share_pct": counterfactual_share,
                "intervention_share_pct": current_share,
                "share_change_pp": current_share - counterfactual_share,
                "counterfactual_mean_time_min": counterfactual_time,
                "intervention_mean_time_min": current_time,
                "mean_time_change_min": current_time - counterfactual_time,
                "counterfactual_mean_distance_km": counterfactual_distance,
                "intervention_mean_distance_km": current_distance,
                "mean_distance_change_km": current_distance - counterfactual_distance,
            }
            for component, template in PT_COMPONENTS.items():
                if mode == "pt":
                    before = _pt_component_mean(
                        counterfactual_od,
                        counterfactual_times,
                        template,
                        mask,
                    )
                    after = _pt_component_mean(
                        counterfactual_od,
                        intervention_times,
                        template,
                        mask,
                    )
                else:
                    before = after = np.nan
                row[f"counterfactual_{component}_min"] = before
                row[f"intervention_{component}_min"] = after
                row[f"{component}_change_min"] = after - before
            rows.append(row)
    return pd.DataFrame(rows, columns=INTERVENTION_SUMMARY_COLUMNS)


def save_modal_share_figure(summary: pd.DataFrame, path: str | Path) -> Path:
    values = summary.set_index("mode").reindex(REPORTING_MODE_ORDER)
    labels = [MODE_LABELS[mode] for mode in REPORTING_MODE_ORDER]
    colors = [MODE_COLORS[mode] for mode in REPORTING_MODE_ORDER]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    for axis, column, title in (
        (axes[0], "trip_share_pct", "Mode share by trips"),
        (axes[1], "passenger_km_share_pct", "Mode share by distance"),
    ):
        shares = values[column].fillna(0.0).to_numpy(dtype=float)
        axis.pie(
            shares,
            labels=labels,
            colors=colors,
            autopct=lambda value: f"{value:.1f}%" if value >= 1.0 else "",
            startangle=90,
            counterclock=False,
            wedgeprops={"edgecolor": "white", "linewidth": 1.0},
        )
        axis.set_title(title)
    figure.tight_layout()
    path = Path(path)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def logsum_accessibility(
    utilities: dict[str, pd.DataFrame],
    zones: pd.DataFrame,
) -> pd.DataFrame:
    zone_ids = utilities["drive"].index.astype(str)
    opportunity_column = "future_jobs" if "future_jobs" in zones.columns else "baseline_jobs"
    opportunities = (
        zones.set_index(zones["grid_id"].astype(str))
        .reindex(zone_ids)[opportunity_column]
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    stack = np.stack([utilities[mode].to_numpy(dtype=float) for mode in MODE_ORDER])
    maximum = np.max(stack, axis=0)
    maximum[~np.isfinite(maximum)] = 0.0
    mode_logsum = maximum + np.log(
        np.maximum(np.exp(np.clip(stack - maximum, -700.0, 700.0)).sum(axis=0), 1e-300)
    )
    weighted = np.exp(np.clip(mode_logsum, -700.0, 700.0)) * opportunities[None, :]
    accessibility = np.log1p(weighted.sum(axis=1))
    return pd.DataFrame({"grid_id": zone_ids, "logsum_accessibility": accessibility})


def save_outputs(
    output_dir: str | Path,
    run_metadata: dict,
    summary: pd.DataFrame,
    accessibility: pd.DataFrame,
    od_by_mode: dict[str, pd.DataFrame],
    link_flows=None,
    diagnostics: dict | None = None,
    zones: pd.DataFrame | None = None,
    lengths: dict[str, pd.DataFrame] | None = None,
    intervention_report: pd.DataFrame | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "mode_summary.csv", index=False)
    mode_summary(od_by_mode, lengths, detailed=True).to_csv(
        output_dir / "mode_summary_detailed.csv",
        index=False,
    )
    save_modal_share_figure(summary, output_dir / "modal_share_pies.png")
    accessibility.to_parquet(output_dir / "zone_accessibility.parquet", index=False)
    for mode, matrix in od_by_mode.items():
        matrix.to_parquet(output_dir / f"od_{mode}.parquet")
    if zones is not None:
        mode_share_by_distance_bin(od_by_mode, zones).to_csv(
            output_dir / "mode_share_by_distance_bin.csv", index=False
        )
        mode_share_by_distance_bin(od_by_mode, zones, detailed=True).to_csv(
            output_dir / "mode_share_by_distance_bin_detailed.csv", index=False
        )
    if zones is not None and lengths is not None:
        mode_summary_by_area(od_by_mode, lengths, zones).to_csv(
            output_dir / "mode_summary_by_area.csv", index=False
        )
        mode_summary_by_area(od_by_mode, lengths, zones, detailed=True).to_csv(
            output_dir / "mode_summary_by_area_detailed.csv", index=False
        )
    if link_flows is not None:
        link_flows.to_parquet(output_dir / "road_link_flows.parquet", index=False)
    if intervention_report is None:
        intervention_report = pd.DataFrame(columns=INTERVENTION_SUMMARY_COLUMNS)
    intervention_report.to_csv(output_dir / "intervention_summary.csv", index=False)
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2),
        encoding="utf-8",
    )
    (output_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics or {}, indent=2, default=str),
        encoding="utf-8",
    )
    return output_dir
