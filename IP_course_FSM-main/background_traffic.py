from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import (
    EXTERNAL_BACKGROUND_GATEWAYS_FILE,
)


def _zone_ids(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def prepare_road_background(
    zones: pd.DataFrame,
    source: pd.DataFrame | str | Path,
    gateway_path: str | Path = EXTERNAL_BACKGROUND_GATEWAYS_FILE,
    output_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Aggregate external NPVM road traffic onto internal gateway zones."""
    model_ids = set(zones["grid_id"].astype(str))
    gateways = pd.read_csv(gateway_path, dtype=str)
    required = {"external_zone_id", "gateway_zone_id"}
    if not required.issubset(gateways.columns):
        raise ValueError(f"Gateway table must contain {sorted(required)}.")

    gateways["external_zone_id"] = _zone_ids(gateways["external_zone_id"])
    gateways["gateway_zone_id"] = _zone_ids(gateways["gateway_zone_id"])
    if gateways["external_zone_id"].duplicated().any():
        raise ValueError("Each external zone may appear only once in the gateway table.")
    unknown_gateways = set(gateways["gateway_zone_id"]) - model_ids
    if unknown_gateways:
        raise ValueError(f"Gateway zones are absent from the model: {sorted(unknown_gateways)}")

    external_map = dict(zip(gateways["external_zone_id"], gateways["gateway_zone_id"]))
    endpoint_map = {zone_id: zone_id for zone_id in model_ids}
    endpoint_map.update(external_map)
    relevant_ids = set(endpoint_map)

    if not isinstance(source, pd.DataFrame):
        source = pd.read_parquet(source, columns=["Origin", "Destination", "Trips"])
    else:
        source = source[["Origin", "Destination", "Trips"]].copy()
    source["Origin"] = _zone_ids(source["Origin"])
    source["Destination"] = _zone_ids(source["Destination"])
    source["Trips"] = pd.to_numeric(source["Trips"], errors="coerce").fillna(0.0)
    source = source.loc[
        source["Origin"].isin(relevant_ids)
        & source["Destination"].isin(relevant_ids)
    ].copy()

    source_total = float(source["Trips"].sum())
    source["origin"] = source["Origin"].map(endpoint_map)
    source["destination"] = source["Destination"].map(endpoint_map)
    prepared = (
        source.groupby(["origin", "destination"], as_index=False, sort=True)["Trips"]
        .sum()
        .rename(columns={"Trips": "trips"})
    )
    prepared = prepared.loc[prepared["trips"].ne(0.0)].reset_index(drop=True)
    prepared_total = float(prepared["trips"].sum())
    if abs(source_total - prepared_total) > 1e-6:
        raise RuntimeError("Road-background trips changed during gateway aggregation.")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        prepared.to_parquet(output_path, index=False)
    diagnostics = {
        "model_zone_count": len(model_ids),
        "external_zone_count": len(external_map),
        "gateway_zone_count": len(set(external_map.values())),
        "road_background_trips": prepared_total,
        "road_background_od_pairs": len(prepared),
    }
    return prepared, diagnostics
