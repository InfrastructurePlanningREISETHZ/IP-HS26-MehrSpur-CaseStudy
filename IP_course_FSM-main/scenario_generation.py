from __future__ import annotations

import numpy as np
import pandas as pd


def _growth_allocation(
    baseline: pd.Series,
    multiplier: float,
    is_city: pd.Series,
    city_growth_share: float,
) -> pd.Series:
    baseline = pd.to_numeric(baseline, errors="coerce").fillna(0.0).clip(lower=0.0)
    multiplier = max(float(multiplier), 0.0)
    total_change = float(baseline.sum()) * (multiplier - 1.0)
    if abs(total_change) < 1e-12:
        return baseline.copy()

    city_growth_share = float(np.clip(city_growth_share, 0.0, 1.0))
    result = baseline.copy()
    for mask, share in ((is_city, city_growth_share), (~is_city, 1.0 - city_growth_share)):
        group = baseline.loc[mask]
        if group.empty:
            continue
        weights = group / group.sum() if group.sum() > 0 else pd.Series(1.0 / len(group), index=group.index)
        result.loc[mask] += total_change * share * weights
    return result.clip(lower=0.0)


def furness_balance(
    seed: pd.DataFrame,
    productions: pd.Series,
    attractions: pd.Series,
    max_iterations: int = 300,
    tolerance: float = 1e-7,
) -> pd.DataFrame:
    """Balance an OD seed to row and column targets using iterative scaling."""
    matrix = seed.to_numpy(dtype=float, copy=True)
    matrix[~np.isfinite(matrix) | (matrix < 0.0)] = 0.0
    row_target = productions.reindex(seed.index).fillna(0.0).to_numpy(dtype=float)
    col_target = attractions.reindex(seed.columns).fillna(0.0).to_numpy(dtype=float)

    positive_rows = (row_target > 0) & (matrix.sum(axis=1) <= 0)
    positive_cols = (col_target > 0) & (matrix.sum(axis=0) <= 0)
    if positive_rows.any() or positive_cols.any():
        support = np.outer(row_target > 0, col_target > 0)
        matrix[support & (matrix <= 0)] = 1e-12

    for _ in range(int(max_iterations)):
        row_sum = matrix.sum(axis=1)
        row_factor = np.divide(row_target, row_sum, out=np.ones_like(row_target), where=row_sum > 0)
        matrix *= row_factor[:, None]

        col_sum = matrix.sum(axis=0)
        col_factor = np.divide(col_target, col_sum, out=np.ones_like(col_target), where=col_sum > 0)
        matrix *= col_factor[None, :]

        row_error = np.max(np.abs(matrix.sum(axis=1) - row_target) / np.maximum(row_target, 1.0))
        col_error = np.max(np.abs(matrix.sum(axis=0) - col_target) / np.maximum(col_target, 1.0))
        if max(row_error, col_error) <= tolerance:
            break

    return pd.DataFrame(matrix, index=seed.index, columns=seed.columns)


def generate_scenario_demand(
    baseline_od: pd.DataFrame,
    zones: pd.DataFrame,
    scenario: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Create future population, jobs and OD demand from a small set of controls."""
    zone_table = zones.set_index(zones["grid_id"].astype(str), drop=False).reindex(baseline_od.index).copy()
    is_city = zone_table["Level"].astype(str).eq("Quartier")

    population_multiplier = float(scenario.get("population_multiplier", 1.0))
    jobs_multiplier = float(scenario.get("jobs_multiplier", 1.0))
    trip_rate_multiplier = float(scenario.get("trip_rate_multiplier", 1.0))
    city_population_share = float(scenario.get("city_population_growth_share", 0.5))
    city_jobs_share = float(scenario.get("city_jobs_growth_share", 0.5))

    base_population = pd.to_numeric(zone_table["baseline_population"], errors="coerce").fillna(0.0)
    base_jobs = pd.to_numeric(zone_table["baseline_jobs"], errors="coerce").fillna(0.0)
    future_population = _growth_allocation(
        base_population,
        population_multiplier,
        is_city,
        city_population_share,
    )
    future_jobs = _growth_allocation(base_jobs, jobs_multiplier, is_city, city_jobs_share)

    population_factor = np.divide(
        future_population,
        base_population,
        out=np.ones(len(zone_table), dtype=float),
        where=base_population.to_numpy(dtype=float) > 0,
    )
    jobs_factor = np.divide(
        future_jobs,
        base_jobs,
        out=np.ones(len(zone_table), dtype=float),
        where=base_jobs.to_numpy(dtype=float) > 0,
    )

    productions = baseline_od.sum(axis=1) * population_factor * max(trip_rate_multiplier, 0.0)
    attractions = baseline_od.sum(axis=0) * jobs_factor
    if productions.sum() > 0 and attractions.sum() > 0:
        attractions *= productions.sum() / attractions.sum()

    future_od = furness_balance(baseline_od, productions, attractions)
    zone_table["future_population"] = future_population
    zone_table["future_jobs"] = future_jobs
    metadata = {
        "population_multiplier": population_multiplier,
        "jobs_multiplier": jobs_multiplier,
        "trip_rate_multiplier": trip_rate_multiplier,
        "future_population": float(future_population.sum()),
        "future_jobs": float(future_jobs.sum()),
        "future_trips": float(future_od.to_numpy(dtype=float).sum()),
    }
    return zone_table.reset_index(drop=True), future_od, metadata
