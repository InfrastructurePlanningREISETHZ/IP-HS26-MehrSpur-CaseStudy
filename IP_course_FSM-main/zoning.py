from __future__ import annotations

import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from config import (
    DEMAND_PACKAGE_FILE,
    MUNICIPAL_BOUNDARIES_ARCHIVE,
    NETWORK_POLICY_CITY_FILE,
    NPVM_ZONES_ARCHIVE,
    QUARTIER_BOUNDARIES_ARCHIVE,
    STATENT_ARCHIVE,
    STATPOP_ARCHIVE,
    ZONES_FILE,
)
from input_packages import load_package


REQUIRED_ZONE_COLUMNS = {
    "grid_id",
    "IntIDs",
    "Level",
    "is_city",
    "municipality_name",
    "city_quartier",
    "geometry",
    "centroid",
    "baseline_population",
    "baseline_jobs",
}


def load_zones(path: str | Path = ZONES_FILE) -> gpd.GeoDataFrame:
    zones = gpd.read_parquet(path).copy()
    missing = REQUIRED_ZONE_COLUMNS.difference(zones.columns)
    if missing:
        raise ValueError(f"Zone file is missing columns: {sorted(missing)}")

    zones["grid_id"] = zones["grid_id"].astype(str)
    zones["IntIDs"] = pd.to_numeric(zones["IntIDs"], errors="raise").astype(int)
    zones["is_city"] = zones["Level"].astype(str).eq("Quartier")
    if zones["grid_id"].duplicated().any():
        raise ValueError("Zone IDs must be unique.")
    return zones.sort_values("IntIDs").reset_index(drop=True)


def load_model_inputs(
    zones_path: str | Path = ZONES_FILE,
    demand_path: str | Path = DEMAND_PACKAGE_FILE,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame]:
    zones = load_zones(zones_path)
    zone_ids = zones["grid_id"].astype(str)
    package = load_package(demand_path, "demand")
    demand = load_od_table(package["passenger_od"], zone_ids)
    background = load_od_table(package["road_background"], zone_ids)
    return zones, demand, background


def load_od_table(
    table: pd.DataFrame,
    zone_ids: pd.Index | list[str] | None = None,
) -> pd.DataFrame:
    required = {"origin", "destination", "trips"}
    if not required.issubset(table.columns):
        raise ValueError(f"OD table must contain {sorted(required)}.")

    values = table.copy()
    values["origin"] = values["origin"].astype(str)
    values["destination"] = values["destination"].astype(str)
    values["trips"] = pd.to_numeric(values["trips"], errors="coerce").fillna(0.0)
    matrix = values.pivot_table(
        index="origin",
        columns="destination",
        values="trips",
        aggfunc="sum",
        fill_value=0.0,
    )
    if zone_ids is not None:
        labels = pd.Index(zone_ids, dtype="object").astype(str)
        matrix = matrix.reindex(index=labels, columns=labels, fill_value=0.0)
    matrix.index.name = "origin"
    matrix.columns.name = "destination"
    return matrix.astype(float)


def city_zone_ids(zones: gpd.GeoDataFrame) -> set[str]:
    return set(zones.loc[zones["Level"].eq("Quartier"), "grid_id"].astype(str))


def _zip_uri(archive: Path, member: str) -> str:
    return f"zip://{archive.resolve().as_posix()}!{member}"


def _read_boundaries() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    member = "Gemeindegrenzen_-OGD/Gemeindegrenzen_-OGD.gpkg"
    source = _zip_uri(Path(MUNICIPAL_BOUNDARIES_ARCHIVE), member)
    cantons = gpd.read_file(source, layer="UP_KANTON_F")
    municipalities = gpd.read_file(source, layer="UP_GEMEINDEN_F")
    canton = cantons.loc[cantons["ABKUERZUNG"].astype(str).eq("ZH")].copy()
    city = municipalities.loc[pd.to_numeric(municipalities["BFS"]).eq(261)].copy()
    if canton.empty or city.empty:
        raise ValueError("The raw boundary archive does not contain Canton or City of Zurich.")
    return canton.to_crs(2056), city.to_crs(2056)


def _read_city_quartiers() -> gpd.GeoDataFrame:
    archive = Path(QUARTIER_BOUNDARIES_ARCHIVE)
    with zipfile.ZipFile(archive) as source:
        members = [
            member
            for member in source.namelist()
            if member.endswith("UP_STADTQUARTIERE_F.shp")
        ]
    if len(members) != 1:
        raise ValueError("The quartier archive must contain one UP_STADTQUARTIERE_F shapefile.")
    quartiers = gpd.read_file(_zip_uri(archive, members[0])).to_crs(2056)
    required = {"BFS", "GEMEINDENA", "QUARTIERNA"}
    if not required.issubset(quartiers.columns):
        raise ValueError(f"The quartier archive is missing columns: {sorted(required)}")
    quartiers = quartiers.loc[pd.to_numeric(quartiers["BFS"], errors="coerce").eq(261)].copy()
    if quartiers.empty:
        raise ValueError("The quartier archive does not contain City of Zurich quartiers.")
    return quartiers


def _assign_city_quartiers(zones: gpd.GeoDataFrame) -> pd.Series:
    quartiers = _read_city_quartiers()
    centroids = gpd.GeoDataFrame(
        zones[["grid_id"]].copy(),
        geometry=zones["centroid"],
        crs=zones.crs,
    )
    joined = gpd.sjoin(
        centroids,
        quartiers[["QUARTIERNA", "geometry"]],
        how="left",
        predicate="covered_by",
    )
    names = joined.groupby("grid_id")["QUARTIERNA"].first()
    result = zones["grid_id"].map(names).fillna("").astype(str)
    return result.where(zones["is_city"], "")


def _allocate_grid_values(
    archive: Path,
    member: str,
    value_column: str,
    zones: gpd.GeoDataFrame,
) -> pd.Series:
    with zipfile.ZipFile(archive) as source:
        with source.open(member) as table:
            values = pd.read_csv(
                table,
                sep=";",
                usecols=["E_KOORD", "N_KOORD", value_column],
                low_memory=False,
            )
    for column in ("E_KOORD", "N_KOORD", value_column):
        values[column] = pd.to_numeric(values[column], errors="coerce")
    values = values.dropna(subset=["E_KOORD", "N_KOORD", value_column])
    minimum_x, minimum_y, maximum_x, maximum_y = zones.total_bounds
    values["E_KOORD"] += 50.0
    values["N_KOORD"] += 50.0
    values = values.loc[
        values["E_KOORD"].between(minimum_x, maximum_x)
        & values["N_KOORD"].between(minimum_y, maximum_y)
        & values[value_column].ne(0.0)
    ].copy()
    points = gpd.GeoDataFrame(
        values[[value_column]],
        geometry=gpd.points_from_xy(values["E_KOORD"], values["N_KOORD"]),
        crs=2056,
    )
    joined = gpd.sjoin(
        points,
        zones[["grid_id", "geometry"]],
        how="inner",
        predicate="within",
    )
    totals = joined.groupby("grid_id")[value_column].sum()
    return zones["grid_id"].map(totals).fillna(0.0).astype(float)


def prepare_zones(output_path: str | Path = ZONES_FILE) -> gpd.GeoDataFrame:
    """Build the direct NPVM zoning and attach baseline population and jobs."""
    npvm_member = "1_Verkehrszonen_Schweiz_NPVM_2023.gpkg"
    raw_zones = gpd.read_file(_zip_uri(Path(NPVM_ZONES_ARCHIVE), npvm_member)).to_crs(2056)
    canton, city = _read_boundaries()
    canton_polygon = canton.geometry.union_all()
    city_polygon = city.geometry.union_all()

    overlap_share = raw_zones.geometry.intersection(canton_polygon).area / raw_zones.geometry.area
    zones = raw_zones.loc[overlap_share.gt(0.40)].copy()
    zones["grid_id"] = pd.to_numeric(zones["No"], errors="raise").astype(int).astype(str)
    zones = zones.sort_values("grid_id").reset_index(drop=True)
    zones["IntIDs"] = np.arange(1, len(zones) + 1, dtype=int)
    zones["centroid"] = zones.geometry.centroid
    zones["is_city"] = zones["centroid"].covered_by(city_polygon)
    zones["Level"] = np.where(zones["is_city"], "Quartier", "Canton")
    zones["municipality_id"] = pd.to_numeric(
        zones["ID_Gem"], errors="coerce"
    ).astype("Int64")
    zones["municipality_name"] = zones["N_Gem"].astype(str)
    zones["city_quartier"] = _assign_city_quartiers(zones)
    zones["Name"] = zones["municipality_name"] + " / NPVM " + zones["grid_id"]
    zones["baseline_population"] = _allocate_grid_values(
        Path(STATPOP_ARCHIVE),
        "STATPOP2024.csv",
        "BBTOT",
        zones,
    )
    zones["baseline_jobs"] = _allocate_grid_values(
        Path(STATENT_ARCHIVE),
        "STATENT_2023.csv",
        "B08EMPT",
        zones,
    )

    columns = [
        "grid_id",
        "IntIDs",
        "Level",
        "Name",
        "is_city",
        "municipality_id",
        "municipality_name",
        "city_quartier",
        "geometry",
        "centroid",
        "baseline_population",
        "baseline_jobs",
    ]
    zones = gpd.GeoDataFrame(zones[columns], geometry="geometry", crs=2056)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    zones.to_parquet(output_path, index=False)

    Path(NETWORK_POLICY_CITY_FILE).parent.mkdir(parents=True, exist_ok=True)
    city[["geometry"]].dissolve().to_parquet(NETWORK_POLICY_CITY_FILE, index=False)
    return zones
