"""Build ASP public-transport skims from GTFS and full-canton access graphs."""

from __future__ import annotations

import json
import pickle
import time
import zipfile
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import matplotlib

# Render plots to files reliably on fresh or headless environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm

from config import (
    ASP_END,
    ASP_START,
    GTFS_FILE,
    GTFS_SERVICE_DATE,
    PT_ACCESS_BIKE_GRAPH_FILE,
    PT_ACCESS_WALK_GRAPH_FILE,
    SKIM_DIR,
    WORK_GTFS_DIR,
    ZONES_FILE,
)


# =====================================================================
# CONFIGURATION
# =====================================================================

GTFS_ZIP = GTFS_FILE
ACCESS_WALK_GRAPH_FILE = PT_ACCESS_WALK_GRAPH_FILE
ACCESS_BIKE_GRAPH_FILE = PT_ACCESS_BIKE_GRAPH_FILE

OUTPUT_DIR = WORK_GTFS_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TIME_PERIODS = {
    "ASP": (ASP_START, ASP_END),
}

CRS_LV95 = "EPSG:2056"

# Use a typical weekday inside the 2026 GTFS validity window.
SERVICE_DATE = GTFS_SERVICE_DATE
STUDY_AREA_BUFFER_M = 10_000.0
TRIM_STOPS_TO_STUDY_AREA = True
SPATIAL_PERIOD_FILTER = True

ACCESS_SPEED_KPH = {"walk": 4.71, "bike": 13.0}
ACCESS_MAX_MIN = {"walk": 25.0, "bike": 20.0}
ACCESS_TOP_K = {"walk": 8, "bike": 8}
EXTRA_RAIL_ACCESS_TOP_K = {"walk": 4, "bike": 4}
WAIT_DEFAULT_MIN = 10.0
WAITING_TIME_FACTOR = 0.5
PT_IVT_MAX_MIN = 240.0
PT_NO_PATH_MIN = 999.0
ASSUME_TWOWAY_ACCESS = True
STOP_TO_STOP_DISTANCE_FACTOR = 1.25

USE_GTFS_TRANSFERS = True
USE_PARENT_STATION_TRANSFERS = True
USE_NEARBY_STOP_TRANSFERS = True
TRANSFER_DEFAULT_MIN = 2.0
PARENT_STATION_TRANSFER_MIN = 1.0
NEARBY_TRANSFER_MAX_M = 300.0
NEARBY_TRANSFER_TOP_K = 12
TRANSFER_WALK_SPEED_KPH = 4.71

RAIL_ROUTE_TYPES = {
    1, 2, 100, 101, 102, 103, 104, 105, 106, 108, 109, 110, 111, 113, 114,
    115, 116, 117, 400, 401, 402, 403, 404, 405,
}
TRAM_ROUTE_TYPES = {0, 900, 901, 902, 903, 904, 905, 906}
BUS_ROUTE_TYPES = {3, 11, 700, 701, 702, 704, 705, 706, 707, 708, 711, 712, 713, 714, 715, 716, 800}
FERRY_ROUTE_TYPES = {4, 1000, 1200}
LIFT_ROUTE_TYPES = {5, 6, 7, 1300, 1400}

# GTFS processed outputs
OUTPUT_STOPS = OUTPUT_DIR / "gtfs_stops_processed.parquet"
OUTPUT_ROUTES = OUTPUT_DIR / "gtfs_routes_processed.parquet"
OUTPUT_WAIT = OUTPUT_DIR / "pt_wait_times.json"
OUTPUT_DIAG = OUTPUT_DIR / "pt_skim_diagnostics.json"
OUTPUT_VIZ = OUTPUT_DIR / "pt_network_visualization.png"

# Access/egress candidate tables
OUTPUT_ZONE_STOP_ACCESS_WALK = OUTPUT_DIR / "pt_zone_stop_access_walk.parquet"
OUTPUT_ZONE_STOP_ACCESS_BIKE = OUTPUT_DIR / "pt_zone_stop_access_bike.parquet"
OUTPUT_STOP_ZONE_EGRESS_WALK = OUTPUT_DIR / "pt_stop_zone_egress_walk.parquet"
OUTPUT_STOP_ZONE_EGRESS_BIKE = OUTPUT_DIR / "pt_stop_zone_egress_bike.parquet"

# Stop-level IVT edges
OUTPUT_IVT_ASP = OUTPUT_DIR / "pt_ivt_matrix_asp.parquet"

# Zone-level PT components
OUTPUT_IVT_ZONE_ASP = OUTPUT_DIR / "pt_ivt_zone_matrix_asp.parquet"
OUTPUT_IVT_WALK_ASP = OUTPUT_DIR / "pt_ivt_walk_matrix_asp.parquet"
OUTPUT_IVT_BIKE_ASP = OUTPUT_DIR / "pt_ivt_bike_matrix_asp.parquet"
OUTPUT_OVT_WALK_ASP = OUTPUT_DIR / "pt_ovt_walk_matrix_asp.parquet"
OUTPUT_OVT_BIKE_ASP = OUTPUT_DIR / "pt_ovt_bike_matrix_asp.parquet"
OUTPUT_DIST_WALK_ASP = OUTPUT_DIR / "pt_distance_walk_matrix_asp.parquet"
OUTPUT_DIST_BIKE_ASP = OUTPUT_DIR / "pt_distance_bike_matrix_asp.parquet"
OUTPUT_TRANSFER_WALK_ASP = OUTPUT_DIR / "pt_transfer_time_walk_matrix_asp.parquet"
OUTPUT_TRANSFER_BIKE_ASP = OUTPUT_DIR / "pt_transfer_time_bike_matrix_asp.parquet"
OUTPUT_TRANSFER_COUNT_WALK_ASP = OUTPUT_DIR / "pt_transfer_count_walk_matrix_asp.parquet"
OUTPUT_TRANSFER_COUNT_BIKE_ASP = OUTPUT_DIR / "pt_transfer_count_bike_matrix_asp.parquet"

PT_OVT_COMPONENTS = (
    "access",
    "egress",
    "initial_wait",
    "transfer_physical",
    "transfer_wait",
    "transfer_total",
    "transfer_count",
)


def component_matrix_path(component: str, mode: str, period_name: str) -> Path:
    period = str(period_name).lower()
    return OUTPUT_DIR / f"pt_{component}_component_{mode}_matrix_{period}.parquet"


def chosen_stop_matrix_path(kind: str, mode: str, period_name: str) -> Path:
    period = str(period_name).lower()
    return OUTPUT_DIR / f"pt_chosen_{kind}_stop_{mode}_matrix_{period}.parquet"


def stop_usage_path(mode: str, period_name: str) -> Path:
    period = str(period_name).lower()
    return OUTPUT_DIR / f"pt_stop_usage_{mode}_{period}.csv"


def component_consistency_path(period_name: str) -> Path:
    period = str(period_name).lower()
    return OUTPUT_DIR / f"pt_component_consistency_{period}.csv"


def access_component_floors_path(period_name: str) -> Path:
    period = str(period_name).lower()
    return OUTPUT_DIR / f"pt_access_component_floors_{period}.json"


def parse_time(time_str: str) -> float:
    try:
        h, m, s = map(int, str(time_str).split(":"))
        return h * 60 + m + s / 60.0
    except Exception:
        return np.nan


def period_bounds(period_name: str) -> tuple[float, float]:
    start, end = TIME_PERIODS[period_name]
    return parse_time(start + ":00"), parse_time(end + ":00")


def load_gtfs_tables(gtfs_zip_path: Path) -> dict[str, pd.DataFrame]:
    required = ["stops", "routes", "trips", "stop_times"]
    tables: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(gtfs_zip_path, "r") as zf:
        names = set(zf.namelist())
        for table in required:
            fn = f"{table}.txt"
            if fn not in names:
                raise FileNotFoundError(f"GTFS missing required table: {fn}")
            tables[table] = pd.read_csv(zf.open(fn), low_memory=False)
        for optional in ("calendar", "calendar_dates", "transfers"):
            fn = f"{optional}.txt"
            if fn in names:
                tables[optional] = pd.read_csv(zf.open(fn), low_memory=False)
        if "shapes.txt" in names:
            tables["shapes"] = pd.read_csv(zf.open("shapes.txt"), low_memory=False)
    return tables


def route_family(route_type: object) -> str:
    try:
        rt = int(route_type)
    except Exception:
        return "unknown"
    if rt in RAIL_ROUTE_TYPES:
        return "rail"
    if rt in TRAM_ROUTE_TYPES:
        return "tram"
    if rt in BUS_ROUTE_TYPES:
        return "bus"
    if rt in FERRY_ROUTE_TYPES:
        return "ferry"
    if rt in LIFT_ROUTE_TYPES:
        return "lift/funicular"
    return "other"


def validate_gtfs(gtfs_tables: dict[str, pd.DataFrame]) -> None:
    required_columns = {
        "stops": ["stop_id", "stop_name", "stop_lat", "stop_lon"],
        "routes": ["route_id", "route_type"],
        "trips": ["trip_id", "route_id"],
        "stop_times": ["trip_id", "stop_id", "stop_sequence", "arrival_time", "departure_time"],
    }
    for table_name, cols in required_columns.items():
        if table_name not in gtfs_tables:
            raise ValueError(f"Missing GTFS table: {table_name}")
        missing = [c for c in cols if c not in gtfs_tables[table_name].columns]
        if missing:
            raise ValueError(f"{table_name} missing columns: {missing}")


def process_stops(gtfs_tables: dict[str, pd.DataFrame]) -> gpd.GeoDataFrame:
    stops = gtfs_tables["stops"].copy()
    stops["stop_id"] = stops["stop_id"].astype(str)
    if "parent_station" in stops.columns:
        stops["parent_station"] = stops["parent_station"].fillna("").astype(str)
    stops_gdf = gpd.GeoDataFrame(
        stops,
        geometry=gpd.points_from_xy(stops["stop_lon"], stops["stop_lat"]),
        crs="EPSG:4326",
    ).to_crs(CRS_LV95)
    stops_gdf["x"] = stops_gdf.geometry.x.astype(float)
    stops_gdf["y"] = stops_gdf.geometry.y.astype(float)
    return stops_gdf


def process_routes(gtfs_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    routes = gtfs_tables["routes"].copy()
    routes["route_id"] = routes["route_id"].astype(str)
    routes["route_type_int"] = pd.to_numeric(routes["route_type"], errors="coerce").astype("Int64")
    routes["route_family"] = routes["route_type_int"].map(route_family)
    trip_counts = gtfs_tables["trips"].copy()
    trip_counts["route_id"] = trip_counts["route_id"].astype(str)
    counts = trip_counts.groupby("route_id").size().rename("trip_count")
    routes = routes.merge(counts, left_on="route_id", right_index=True, how="left")
    routes["trip_count"] = routes["trip_count"].fillna(0).astype(int)
    return routes


def load_zones(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Zones file not found: {path}")
    zones = gpd.read_parquet(path).copy()
    if zones.crs is None:
        raise ValueError(f"{path} has no CRS.")
    zones = zones.to_crs(CRS_LV95)
    if "grid_id" not in zones.columns:
        raise ValueError(f"{path} missing required column: grid_id")
    if "centroid" not in zones.columns:
        zones["centroid"] = zones.geometry.centroid
    zones["grid_id"] = zones["grid_id"].astype(str)
    return zones.sort_values("grid_id").reset_index(drop=True)


def study_area_stop_ids(
    stops_gdf: gpd.GeoDataFrame,
    zones_gdf: gpd.GeoDataFrame,
    buffer_m: float,
) -> set[str]:
    union = zones_gdf.geometry.union_all() if hasattr(zones_gdf.geometry, "union_all") else zones_gdf.geometry.unary_union
    area = union.buffer(float(buffer_m)).buffer(0)
    mask = stops_gdf.geometry.within(area)
    return set(stops_gdf.loc[mask, "stop_id"].astype(str))


def normalize_service_date(service_date: str | None) -> str | None:
    if service_date is None:
        return None
    value = str(service_date).strip()
    if not value:
        return None
    return value.replace("-", "")


def active_service_ids(gtfs_tables: dict[str, pd.DataFrame], service_date: str | None) -> set[str] | None:
    date_value = normalize_service_date(service_date)
    if date_value is None:
        return None
    date_int = int(date_value)
    date_ts = pd.to_datetime(date_value, format="%Y%m%d")
    weekday_col = date_ts.day_name().lower()

    active: set[str] = set()
    calendar = gtfs_tables.get("calendar")
    if calendar is not None and not calendar.empty:
        cal = calendar.copy()
        cal["service_id"] = cal["service_id"].astype(str)
        if {"start_date", "end_date", weekday_col}.issubset(cal.columns):
            cal["start_date"] = pd.to_numeric(cal["start_date"], errors="coerce")
            cal["end_date"] = pd.to_numeric(cal["end_date"], errors="coerce")
            weekday_active = pd.to_numeric(cal[weekday_col], errors="coerce").fillna(0).astype(int).eq(1)
            in_range = cal["start_date"].le(date_int) & cal["end_date"].ge(date_int)
            active.update(cal.loc[weekday_active & in_range, "service_id"].astype(str))

    calendar_dates = gtfs_tables.get("calendar_dates")
    if calendar_dates is not None and not calendar_dates.empty and {"service_id", "date", "exception_type"}.issubset(calendar_dates.columns):
        cd = calendar_dates.copy()
        cd["service_id"] = cd["service_id"].astype(str)
        cd["date"] = pd.to_numeric(cd["date"], errors="coerce")
        cd["exception_type"] = pd.to_numeric(cd["exception_type"], errors="coerce")
        cd = cd[cd["date"].eq(date_int)]
        active.update(cd.loc[cd["exception_type"].eq(1), "service_id"].astype(str))
        active.difference_update(set(cd.loc[cd["exception_type"].eq(2), "service_id"].astype(str)))

    if not active:
        print(f"WARNING: No active GTFS service_ids found for service_date={service_date}; using all services.")
        return None
    return active


def filter_trips_by_service_date(
    gtfs_tables: dict[str, pd.DataFrame],
    service_date: str | None,
) -> pd.DataFrame:
    trips = gtfs_tables["trips"].copy()
    trips["trip_id"] = trips["trip_id"].astype(str)
    trips["route_id"] = trips["route_id"].astype(str)
    if "service_id" not in trips.columns:
        return trips
    trips["service_id"] = trips["service_id"].astype(str)
    active_ids = active_service_ids(gtfs_tables, service_date)
    if active_ids is None:
        return trips
    return trips[trips["service_id"].isin(active_ids)].copy()


def annotate_stops_with_route_families(
    stops_gdf: gpd.GeoDataFrame,
    routes_df: pd.DataFrame,
    trips_df: pd.DataFrame,
    stop_times: pd.DataFrame,
) -> gpd.GeoDataFrame:
    routes = routes_df[["route_id", "route_family"]].copy()
    routes["route_id"] = routes["route_id"].astype(str)
    trips = trips_df[["trip_id", "route_id"]].copy()
    trips["trip_id"] = trips["trip_id"].astype(str)
    trips["route_id"] = trips["route_id"].astype(str)
    trip_family = trips.merge(routes, on="route_id", how="left").set_index("trip_id")["route_family"]

    tmp = stop_times[["stop_id", "trip_id"]].copy()
    tmp["stop_id"] = tmp["stop_id"].astype(str)
    tmp["trip_id"] = tmp["trip_id"].astype(str)
    tmp["route_family"] = tmp["trip_id"].map(trip_family)
    tmp = tmp.dropna(subset=["route_family"])
    families = (
        tmp.groupby("stop_id")["route_family"]
        .agg(lambda s: "|".join(sorted(set(str(v) for v in s if pd.notna(v)))))
        .rename("route_families")
    )
    out = stops_gdf.copy()
    out["route_families"] = out["stop_id"].astype(str).map(families).fillna("")
    out["serves_rail"] = out["route_families"].str.contains("rail", regex=False)
    return out


def load_access_graph(path: Path) -> nx.MultiDiGraph:
    if not path.exists():
        raise FileNotFoundError(f"Missing PT access graph: {path}")
    with open(path, "rb") as f:
        G = pickle.load(f)
    if not isinstance(G, nx.MultiDiGraph):
        raise ValueError(f"Unexpected graph type in {path}: {type(G).__name__}")
    if len(G.nodes) == 0:
        raise ValueError(f"Access graph is empty: {path}")
    return G


def add_weighted_edge_min(G: nx.DiGraph, u, v, w: float) -> None:
    if G.has_edge(u, v):
        if w < float(G[u][v]["weight"]):
            G[u][v]["weight"] = float(w)
    else:
        G.add_edge(u, v, weight=float(w))


def add_transit_edge_min_time(G: nx.DiGraph, u, v, ivt_min: float, distance_m: float) -> None:
    if not np.isfinite(distance_m) or float(distance_m) < 0:
        distance_m = 0.0
    if G.has_edge(u, v):
        current = float(G[u][v]["weight"])
        if float(ivt_min) < current:
            G[u][v]["weight"] = float(ivt_min)
            G[u][v]["ivt_min"] = float(ivt_min)
            G[u][v]["transfer_min"] = 0.0
            G[u][v]["transfer_count"] = 0.0
            G[u][v]["distance_m"] = float(distance_m)
            G[u][v]["edge_type"] = "transit"
        elif float(ivt_min) == current:
            G[u][v]["distance_m"] = min(float(G[u][v].get("distance_m", distance_m)), float(distance_m))
    else:
        G.add_edge(
            u,
            v,
            weight=float(ivt_min),
            ivt_min=float(ivt_min),
            transfer_min=0.0,
            transfer_count=0.0,
            distance_m=float(distance_m),
            edge_type="transit",
        )


def build_access_time_graph(
    G_raw: nx.MultiDiGraph,
    speed_kph: float,
    force_twoway: bool = True,
) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(G_raw.nodes())
    for u, v, _k, data in G_raw.edges(keys=True, data=True):
        length_m = data.get("length", np.nan)
        if length_m is None or not np.isfinite(length_m):
            continue
        length_m = float(length_m)
        if length_m <= 0:
            continue
        time_min = (length_m / 1000.0 / speed_kph) * 60.0
        add_weighted_edge_min(G, u, v, time_min)
        if force_twoway:
            add_weighted_edge_min(G, v, u, time_min)
    if len(G.edges) == 0:
        raise ValueError("Access time graph has zero edges.")
    return G


def nearest_graph_node_ids(points, G) -> np.ndarray:
    node_ids = list(G.nodes())
    if len(node_ids) == 0:
        raise ValueError("Graph has zero nodes for nearest-node lookup.")

    node_xy = np.column_stack(
        (
            np.array([float(G.nodes[n].get("x")) for n in node_ids], dtype=float),
            np.array([float(G.nodes[n].get("y")) for n in node_ids], dtype=float),
        )
    )
    point_xy = np.column_stack(
        (
            np.array([float(pt.x) for pt in points], dtype=float),
            np.array([float(pt.y) for pt in points], dtype=float),
        )
    )
    tree = cKDTree(node_xy)
    _, idx = tree.query(point_xy, k=1)
    idx = np.asarray(idx, dtype=int)
    return np.array([node_ids[i] for i in idx], dtype=object)


def nearest_point_indices(query_xy: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
    if target_xy.size == 0:
        raise ValueError("target_xy is empty for nearest-point lookup.")
    tree = cKDTree(target_xy)
    _, idx = tree.query(query_xy, k=1)
    return np.asarray(idx, dtype=int)


def compute_zone_stop_candidates(
    zones_gdf: gpd.GeoDataFrame,
    stops_gdf: gpd.GeoDataFrame,
    G_time: nx.DiGraph,
    G_raw: nx.MultiDiGraph,
    mode: str,
    top_k: int,
    max_access_min: float,
    extra_stop_mask: np.ndarray | None = None,
    extra_top_k: int = 0,
    extra_label: str = "extra",
) -> pd.DataFrame:
    zones = zones_gdf.copy().sort_values("grid_id").reset_index(drop=True)
    if "centroid" not in zones.columns:
        zones["centroid"] = zones.geometry.centroid

    zone_ids = zones["grid_id"].astype(str).to_numpy()
    stop_ids = stops_gdf["stop_id"].astype(str).to_numpy()

    zone_nodes = nearest_graph_node_ids(zones["centroid"].tolist(), G_raw)
    stop_nodes = nearest_graph_node_ids(stops_gdf.geometry.tolist(), G_raw)

    stop_node_to_idx: dict[object, list[int]] = {}
    for idx, node in enumerate(stop_nodes):
        stop_node_to_idx.setdefault(node, []).append(idx)

    stop_xy = np.column_stack((stops_gdf["x"].to_numpy(dtype=float), stops_gdf["y"].to_numpy(dtype=float)))
    zone_xy = np.column_stack((zones["centroid"].x.to_numpy(dtype=float), zones["centroid"].y.to_numpy(dtype=float)))
    nearest_stop_idx = nearest_point_indices(zone_xy, stop_xy)

    unique_zone_nodes = pd.unique(zone_nodes)
    zone_node_to_times: dict[object, np.ndarray] = {}

    for zn in tqdm(unique_zone_nodes, desc=f"Computing {mode} access dijkstra"):
        lengths = nx.single_source_dijkstra_path_length(
            G_time,
            source=zn,
            cutoff=float(max_access_min),
            weight="weight",
        )
        arr = np.full(len(stop_ids), np.inf, dtype=float)
        for node, t in lengths.items():
            idxs = stop_node_to_idx.get(node)
            if idxs is None:
                continue
            for si in idxs:
                if float(t) < arr[si]:
                    arr[si] = float(t)
        zone_node_to_times[zn] = arr

    records = []
    for i, zone_id in enumerate(zone_ids):
        zn = zone_nodes[i]
        row = zone_node_to_times.get(zn)
        if row is None:
            row = np.full(len(stop_ids), np.inf, dtype=float)
        valid = np.where(np.isfinite(row) & (row <= float(max_access_min)))[0]

        if len(valid) == 0:
            fallback_idx = int(nearest_stop_idx[i])
            records.append(
                {
                    "zone_id": str(zone_id),
                    "stop_id": str(stop_ids[fallback_idx]),
                    "access_min": float(max_access_min),
                    "rank": 1,
                    "mode": mode,
                    "candidate_type": "nearest_fallback",
                    "fallback": True,
                }
            )
            continue

        selected: list[tuple[int, str]] = []
        ranked = valid[np.argsort(row[valid])][: int(top_k)]
        selected.extend((int(stop_idx), "nearest") for stop_idx in ranked)

        if extra_stop_mask is not None and int(extra_top_k) > 0:
            mask = np.asarray(extra_stop_mask, dtype=bool)
            if len(mask) == len(stop_ids):
                extra_valid = valid[mask[valid]]
                extra_ranked = extra_valid[np.argsort(row[extra_valid])][: int(extra_top_k)]
                seen = {idx for idx, _ in selected}
                selected.extend((int(stop_idx), extra_label) for stop_idx in extra_ranked if int(stop_idx) not in seen)

        for rank, (stop_idx, candidate_type) in enumerate(selected, start=1):
            records.append(
                {
                    "zone_id": str(zone_id),
                    "stop_id": str(stop_ids[stop_idx]),
                    "access_min": float(row[stop_idx]),
                    "rank": int(rank),
                    "mode": mode,
                    "candidate_type": candidate_type,
                    "fallback": False,
                }
            )

    out = pd.DataFrame(records)
    out = out.sort_values(["zone_id", "rank", "stop_id"]).reset_index(drop=True)
    return out


def build_stop_zone_egress_from_access(access_df: pd.DataFrame, mode: str) -> pd.DataFrame:
    out = access_df[["zone_id", "stop_id", "access_min", "rank"]].copy()
    out = out.rename(columns={"zone_id": "zone_id", "access_min": "egress_min"})
    out["mode"] = mode
    return out


def prepare_stop_times(gtfs_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    st = gtfs_tables["stop_times"].copy()
    st["trip_id"] = st["trip_id"].astype(str)
    st["stop_id"] = st["stop_id"].astype(str)
    st["arrival_min"] = st["arrival_time"].apply(parse_time)
    st["departure_min"] = st["departure_time"].apply(parse_time)
    st["stop_sequence"] = pd.to_numeric(st["stop_sequence"], errors="coerce")
    st = st.dropna(subset=["stop_sequence"]).copy()
    return st


def stop_event_overlaps_period(stop_times: pd.DataFrame, period_name: str) -> pd.Series:
    start_min, end_min = period_bounds(period_name)
    arr = pd.to_numeric(stop_times["arrival_min"], errors="coerce")
    dep = pd.to_numeric(stop_times["departure_min"], errors="coerce")
    event_in_period = arr.between(start_min, end_min, inclusive="both") | dep.between(start_min, end_min, inclusive="both")
    dwell_overlaps = arr.le(end_min) & dep.ge(start_min)
    return event_in_period | dwell_overlaps


def add_transfer_edge_min_time(
    G: nx.DiGraph,
    u: str,
    v: str,
    transfer_min: float,
    transfer_wait_min: float = 0.0,
) -> bool:
    if not np.isfinite(transfer_min) or float(transfer_min) < 0:
        return False
    u = str(u)
    v = str(v)
    if not u or not v or u == "nan" or v == "nan" or u == v:
        return False
    physical_min = max(float(transfer_min), float(TRANSFER_DEFAULT_MIN))
    if not np.isfinite(transfer_wait_min) or float(transfer_wait_min) < 0:
        transfer_wait_min = float(WAIT_DEFAULT_MIN)
    total_transfer_min = physical_min + float(transfer_wait_min)
    if G.has_edge(u, v):
        if float(total_transfer_min) < float(G[u][v]["weight"]):
            G[u][v]["weight"] = float(total_transfer_min)
            G[u][v]["ivt_min"] = 0.0
            G[u][v]["transfer_min"] = float(total_transfer_min)
            G[u][v]["transfer_physical_min"] = float(physical_min)
            G[u][v]["transfer_wait_min"] = float(transfer_wait_min)
            G[u][v]["transfer_count"] = 1.0
            G[u][v]["distance_m"] = 0.0
            G[u][v]["edge_type"] = "transfer"
            return True
        return False
    G.add_edge(
        u,
        v,
        weight=float(total_transfer_min),
        ivt_min=0.0,
        transfer_min=float(total_transfer_min),
        transfer_physical_min=float(physical_min),
        transfer_wait_min=float(transfer_wait_min),
        transfer_count=1.0,
        distance_m=0.0,
        edge_type="transfer",
    )
    return True


def add_gtfs_transfer_edges(
    G_pt: nx.DiGraph,
    gtfs_tables: dict[str, pd.DataFrame],
    wait_times: dict[str, float],
) -> dict[str, int]:
    diag = {
        "gtfs_transfer_rows": 0,
        "gtfs_transfer_edges_added_or_improved": 0,
        "parent_station_edges_added_or_improved": 0,
        "nearby_transfer_edges_added_or_improved": 0,
    }

    node_set = set(str(n) for n in G_pt.nodes)
    stops = gtfs_tables.get("_processed_stops_gdf")

    if stops is not None and not stops.empty and USE_PARENT_STATION_TRANSFERS and "parent_station" in stops.columns:
        parent_links = stops[["stop_id", "parent_station"]].copy()
        parent_links["stop_id"] = parent_links["stop_id"].astype(str)
        parent_links["parent_station"] = parent_links["parent_station"].fillna("").astype(str)
        existing_child_parent = parent_links[
            parent_links["stop_id"].isin(node_set)
            & parent_links["parent_station"].ne("")
            & parent_links["parent_station"].isin(set(stops["stop_id"].astype(str)))
        ]
        for parent in existing_child_parent["parent_station"].unique():
            G_pt.add_node(str(parent))
        node_set = set(str(n) for n in G_pt.nodes)

    transfers = gtfs_tables.get("transfers")
    if USE_GTFS_TRANSFERS and transfers is not None and not transfers.empty and {"from_stop_id", "to_stop_id"}.issubset(transfers.columns):
        tr = transfers.copy()
        tr["from_stop_id"] = tr["from_stop_id"].astype(str)
        tr["to_stop_id"] = tr["to_stop_id"].astype(str)
        tr = tr[tr["from_stop_id"].isin(node_set) & tr["to_stop_id"].isin(node_set)]
        if "transfer_type" in tr.columns:
            transfer_type = pd.to_numeric(tr["transfer_type"], errors="coerce")
            tr = tr[~transfer_type.eq(3)]
        if "min_transfer_time" in tr.columns:
            transfer_min = pd.to_numeric(tr["min_transfer_time"], errors="coerce") / 60.0
        else:
            transfer_min = pd.Series(np.nan, index=tr.index)
        transfer_min = transfer_min.fillna(float(TRANSFER_DEFAULT_MIN)).clip(lower=0.0)
        diag["gtfs_transfer_rows"] = int(len(tr))
        added = 0
        for from_stop, to_stop, minutes in zip(tr["from_stop_id"], tr["to_stop_id"], transfer_min):
            wait_min = float(wait_times.get(str(to_stop), WAIT_DEFAULT_MIN))
            added += int(add_transfer_edge_min_time(G_pt, from_stop, to_stop, float(minutes), wait_min))
        diag["gtfs_transfer_edges_added_or_improved"] = int(added)

    if stops is None or stops.empty:
        return diag

    if USE_PARENT_STATION_TRANSFERS and "parent_station" in stops.columns:
        added = 0
        parent_links = stops[["stop_id", "parent_station"]].copy()
        parent_links["stop_id"] = parent_links["stop_id"].astype(str)
        parent_links["parent_station"] = parent_links["parent_station"].fillna("").astype(str)
        parent_links = parent_links[
            parent_links["stop_id"].isin(node_set)
            & parent_links["parent_station"].isin(node_set)
            & parent_links["parent_station"].ne("")
        ]
        for child, parent in zip(parent_links["stop_id"], parent_links["parent_station"]):
            parent_wait = float(wait_times.get(str(parent), WAIT_DEFAULT_MIN))
            child_wait = float(wait_times.get(str(child), WAIT_DEFAULT_MIN))
            added += int(add_transfer_edge_min_time(G_pt, child, parent, float(PARENT_STATION_TRANSFER_MIN), parent_wait))
            added += int(add_transfer_edge_min_time(G_pt, parent, child, float(PARENT_STATION_TRANSFER_MIN), child_wait))
        diag["parent_station_edges_added_or_improved"] = int(added)

    if USE_NEARBY_STOP_TRANSFERS and len(node_set) > 1:
        transfer_stops = stops[stops["stop_id"].astype(str).isin(node_set)].copy()
        transfer_stops = transfer_stops[np.isfinite(transfer_stops["x"]) & np.isfinite(transfer_stops["y"])]
        if len(transfer_stops) > 1:
            stop_ids = transfer_stops["stop_id"].astype(str).to_numpy()
            xy = np.column_stack((transfer_stops["x"].to_numpy(dtype=float), transfer_stops["y"].to_numpy(dtype=float)))
            tree = cKDTree(xy)
            nearby = tree.query_ball_point(xy, r=float(NEARBY_TRANSFER_MAX_M))
            added = 0
            for i, js in enumerate(nearby):
                js = [j for j in js if j != i]
                if not js:
                    continue
                js = sorted(js, key=lambda j: float(np.hypot(*(xy[j] - xy[i]))))[: int(NEARBY_TRANSFER_TOP_K)]
                for j in js:
                    dist_m = float(np.hypot(*(xy[j] - xy[i])))
                    minutes = max(float(TRANSFER_DEFAULT_MIN), (dist_m / 1000.0 / float(TRANSFER_WALK_SPEED_KPH)) * 60.0)
                    wait_min = float(wait_times.get(str(stop_ids[j]), WAIT_DEFAULT_MIN))
                    added += int(add_transfer_edge_min_time(G_pt, stop_ids[i], stop_ids[j], minutes, wait_min))
            diag["nearby_transfer_edges_added_or_improved"] = int(added)

    return diag


def build_period_transit_graph(
    stop_times: pd.DataFrame,
    stops_gdf: gpd.GeoDataFrame,
    period_name: str,
    gtfs_tables: dict[str, pd.DataFrame],
    wait_times: dict[str, float],
    selection_stop_ids: set[str] | None = None,
    graph_stop_ids: set[str] | None = None,
) -> tuple[nx.DiGraph, pd.DataFrame, int, dict[str, int]]:
    if SPATIAL_PERIOD_FILTER and selection_stop_ids is not None:
        selection_mask = stop_times["stop_id"].astype(str).isin(selection_stop_ids)
    else:
        selection_mask = pd.Series(True, index=stop_times.index)
    in_period = stop_times[selection_mask & stop_event_overlaps_period(stop_times, period_name)]
    period_trips = in_period["trip_id"].unique()
    st_period = stop_times[stop_times["trip_id"].isin(period_trips)].copy()
    if graph_stop_ids is not None:
        st_period = st_period[st_period["stop_id"].astype(str).isin(graph_stop_ids)].copy()

    G_pt = nx.DiGraph()
    stop_xy = {
        str(row.stop_id): (float(row.x), float(row.y))
        for row in stops_gdf[["stop_id", "x", "y"]].itertuples(index=False)
    }
    records = []
    for trip_id, grp in tqdm(st_period.groupby("trip_id"), desc=f"Building PT graph {period_name}"):
        grp = grp.sort_values("stop_sequence")
        if len(grp) < 2:
            continue
        stops = grp["stop_id"].to_numpy(dtype=str)
        dep = grp["departure_min"].to_numpy(dtype=float)
        arr = grp["arrival_min"].to_numpy(dtype=float)
        shape_dist = (
            pd.to_numeric(grp["shape_dist_traveled"], errors="coerce").to_numpy(dtype=float)
            if "shape_dist_traveled" in grp.columns
            else np.full(len(grp), np.nan, dtype=float)
        )

        for i in range(len(stops) - 1):
            s_i = str(stops[i])
            s_j = str(stops[i + 1])
            dep_i = dep[i]
            arr_j = arr[i + 1]
            if not np.isfinite(dep_i) or not np.isfinite(arr_j):
                continue
            ivt = float(arr_j - dep_i)
            if ivt <= 0 or ivt > PT_IVT_MAX_MIN:
                continue
            distance_m = float(shape_dist[i + 1] - shape_dist[i])
            if not np.isfinite(distance_m) or distance_m < 0:
                xy_i = stop_xy.get(s_i)
                xy_j = stop_xy.get(s_j)
                if xy_i is not None and xy_j is not None:
                    distance_m = float(np.hypot(xy_j[0] - xy_i[0], xy_j[1] - xy_i[1]))
                    distance_m *= float(STOP_TO_STOP_DISTANCE_FACTOR)
                else:
                    distance_m = 0.0
            add_transit_edge_min_time(G_pt, s_i, s_j, ivt, distance_m)
            records.append({"stop_i": s_i, "stop_j": s_j, "ivt_minutes": ivt, "distance_m": distance_m})

    transfer_diag = add_gtfs_transfer_edges(G_pt, gtfs_tables, wait_times=wait_times)

    if len(records) == 0:
        return (
            G_pt,
            pd.DataFrame(columns=["stop_i", "stop_j", "ivt_minutes", "distance_m"]),
            int(len(period_trips)),
            transfer_diag,
        )

    direct_df = pd.DataFrame(records)
    direct_df = (
        direct_df.sort_values(["stop_i", "stop_j", "ivt_minutes", "distance_m"])
        .groupby(["stop_i", "stop_j"], as_index=False)
        .first()
        .sort_values(["stop_i", "stop_j"])
        .reset_index(drop=True)
    )
    return G_pt, direct_df, int(len(period_trips)), transfer_diag


def compute_wait_times(stop_times: pd.DataFrame, period_name: str) -> dict[str, float]:
    start_min, end_min = period_bounds(period_name)
    arr = stop_times[
        (stop_times["arrival_min"] >= start_min)
        & (stop_times["arrival_min"] <= end_min)
    ][["stop_id", "arrival_min"]].copy()

    waits: dict[str, float] = {}
    for stop_id, grp in arr.groupby("stop_id"):
        values = np.sort(grp["arrival_min"].to_numpy(dtype=float))
        values = values[np.isfinite(values)]
        if len(values) < 2:
            waits[str(stop_id)] = float(WAIT_DEFAULT_MIN)
            continue
        headways = np.diff(values)
        headways = headways[np.isfinite(headways) & (headways > 0)]
        if len(headways) == 0:
            waits[str(stop_id)] = float(WAIT_DEFAULT_MIN)
            continue
        waits[str(stop_id)] = float(np.mean(headways) * WAITING_TIME_FACTOR)
    return waits


def build_candidate_lookup(df: pd.DataFrame, time_col: str = "access_min") -> dict[str, list[tuple[str, float]]]:
    lookup: dict[str, list[tuple[str, float]]] = {}
    if df.empty:
        return lookup
    for zone_id, grp in df.sort_values(["zone_id", "rank"]).groupby("zone_id"):
        lookup[str(zone_id)] = [(str(r["stop_id"]), float(r[time_col])) for _, r in grp.iterrows()]
    return lookup


def _summarize_path_edges(G_pt: nx.DiGraph, path: list[str]) -> dict[str, object]:
    ivt_min = 0.0
    transfer_min = 0.0
    transfer_physical_min = 0.0
    transfer_wait_min = 0.0
    transfer_count = 0.0
    distance_m = 0.0
    transfer_stops: set[str] = set()

    for u, v in zip(path[:-1], path[1:]):
        edge = G_pt[u][v]
        edge_type = str(edge.get("edge_type", "transit"))
        ivt_min += float(edge.get("ivt_min", edge.get("weight", 0.0)))
        transfer_min += float(edge.get("transfer_min", 0.0))
        transfer_physical_min += float(edge.get("transfer_physical_min", 0.0))
        transfer_wait_min += float(edge.get("transfer_wait_min", 0.0))
        transfer_count += float(edge.get("transfer_count", 0.0))
        distance_m += float(edge.get("distance_m", 0.0))
        if edge_type == "transfer":
            transfer_stops.add(str(u))
            transfer_stops.add(str(v))

    return {
        "ivt_min": float(ivt_min),
        "transfer_min": float(transfer_min),
        "transfer_physical_min": float(transfer_physical_min),
        "transfer_wait_min": float(transfer_wait_min),
        "transfer_count": float(transfer_count),
        "distance_m": float(distance_m),
        "transfer_stops": tuple(sorted(transfer_stops)),
    }


STOP_PATH_METRICS = (
    "ivt_min",
    "transfer_min",
    "transfer_physical_min",
    "transfer_wait_min",
    "transfer_count",
    "distance_m",
)


@dataclass
class StopShortestMatrices:
    """Disk-backed stop-to-stop path metrics used by both PT access modes."""

    values: np.memmap
    origin_index: dict[str, int]
    destination_index: dict[str, int]
    path: Path

    def get(self, origin: str, destination: str) -> np.ndarray | None:
        origin_idx = self.origin_index.get(str(origin))
        destination_idx = self.destination_index.get(str(destination))
        if origin_idx is None or destination_idx is None:
            return None
        metrics = self.values[origin_idx, destination_idx]
        return metrics if np.isfinite(metrics[0]) else None

    def close(self) -> None:
        self.values.flush()
        mmap = getattr(self.values, "_mmap", None)
        if mmap is not None:
            mmap.close()
        self.path.unlink(missing_ok=True)


def compute_stop_shortest_matrices(
    G_pt: nx.DiGraph,
    origin_stops: set[str],
    destination_stops: set[str],
    temp_path: Path,
) -> StopShortestMatrices:
    """Compute exact path metrics once without retaining Python path dictionaries."""
    origins = sorted(str(stop_id) for stop_id in origin_stops)
    destinations = sorted(str(stop_id) for stop_id in destination_stops)
    origin_index = {stop_id: i for i, stop_id in enumerate(origins)}
    destination_index = {stop_id: i for i, stop_id in enumerate(destinations)}

    temp_path.unlink(missing_ok=True)
    values = np.lib.format.open_memmap(
        temp_path,
        mode="w+",
        dtype=np.float64,
        shape=(len(origins), len(destinations), len(STOP_PATH_METRICS)),
    )

    for origin_idx, origin in enumerate(
        tqdm(origins, desc="Transit shortest paths from candidate origin stops")
    ):
        row = values[origin_idx]
        row.fill(np.nan)
        if origin not in G_pt:
            continue
        lengths, paths = nx.single_source_dijkstra(
            G_pt,
            source=origin,
            cutoff=float(PT_IVT_MAX_MIN),
            weight="weight",
        )
        for destination, path_min in lengths.items():
            destination = str(destination)
            destination_idx = destination_index.get(destination)
            if destination_idx is None or not np.isfinite(path_min):
                continue
            path = paths.get(destination, [])
            summary = _summarize_path_edges(G_pt, [str(node) for node in path])
            row[destination_idx] = (
                float(summary["ivt_min"]),
                float(summary["transfer_min"]),
                float(summary["transfer_physical_min"]),
                float(summary["transfer_wait_min"]),
                float(summary["transfer_count"]),
                float(summary["distance_m"]),
            )
        if origin_idx % 100 == 0:
            values.flush()

    values.flush()
    return StopShortestMatrices(values, origin_index, destination_index, temp_path)


def compose_zone_components_for_mode(
    zone_ids: list[str],
    access_lookup: dict[str, list[tuple[str, float]]],
    egress_lookup: dict[str, list[tuple[str, float]]],
    wait_times: dict[str, float],
    stop_shortest_matrices: StopShortestMatrices,
    *,
    mode: str,
    stop_names: dict[str, str] | None = None,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict]:
    n_zones = len(zone_ids)
    no_path = float(PT_NO_PATH_MIN)
    matrices = {
        "ivt": np.full((n_zones, n_zones), no_path, dtype=float),
        "ovt": np.full((n_zones, n_zones), no_path, dtype=float),
        "total": np.full((n_zones, n_zones), no_path, dtype=float),
        "distance": np.zeros((n_zones, n_zones), dtype=float),
        "transfer_output": np.zeros((n_zones, n_zones), dtype=float),
        "transfer_count": np.zeros((n_zones, n_zones), dtype=float),
        "access": np.full((n_zones, n_zones), no_path, dtype=float),
        "egress": np.full((n_zones, n_zones), no_path, dtype=float),
        "initial_wait": np.full((n_zones, n_zones), no_path, dtype=float),
        "transfer_physical": np.full((n_zones, n_zones), no_path, dtype=float),
        "transfer_wait": np.full((n_zones, n_zones), no_path, dtype=float),
        "transfer_total": np.full((n_zones, n_zones), no_path, dtype=float),
        "chosen_origin_stop": np.full((n_zones, n_zones), "", dtype=object),
        "chosen_destination_stop": np.full((n_zones, n_zones), "", dtype=object),
    }
    for values in matrices.values():
        np.fill_diagonal(values, 0.0 if values.dtype != object else "")

    stop_usage: dict[str, dict[str, object]] = {}
    total_pairs = len(zone_ids) * len(zone_ids)
    no_path_pairs = 0
    stop_names = stop_names or {}

    def usage_entry(stop_id: str) -> dict[str, object]:
        sid = str(stop_id)
        entry = stop_usage.get(sid)
        if entry is None:
            entry = {
                "stop_id": sid,
                "stop_name": stop_names.get(sid, ""),
                "origin_access_count": 0,
                "destination_egress_count": 0,
                "transfer_count_on_paths": 0,
                "access_min_sum": 0.0,
                "egress_min_sum": 0.0,
            }
            stop_usage[sid] = entry
        return entry

    def add_stop_usage(
        origin_stop: str,
        destination_stop: str,
        access_min: float,
        egress_min: float,
    ) -> None:
        if origin_stop:
            entry = usage_entry(origin_stop)
            entry["origin_access_count"] = int(entry["origin_access_count"]) + 1
            entry["access_min_sum"] = float(entry["access_min_sum"]) + float(access_min)
        if destination_stop:
            entry = usage_entry(destination_stop)
            entry["destination_egress_count"] = int(entry["destination_egress_count"]) + 1
            entry["egress_min_sum"] = float(entry["egress_min_sum"]) + float(egress_min)

    for origin_idx, origin_zone in enumerate(
        tqdm(zone_ids, desc=f"Composing zone PT matrices ({mode})")
    ):
        o_cands = access_lookup.get(origin_zone, [])
        for destination_idx, destination_zone in enumerate(zone_ids):
            if origin_idx == destination_idx:
                continue

            d_cands = egress_lookup.get(destination_zone, [])
            best_total = np.inf
            best_ivt = np.inf
            best_ovt = np.inf
            best_distance_m = np.inf
            best_access_min = np.inf
            best_egress_min = np.inf
            best_initial_wait_min = np.inf
            best_transfer_min = np.inf
            best_transfer_physical_min = np.inf
            best_transfer_wait_min = np.inf
            best_transfer_count = np.inf
            best_origin_stop = ""
            best_destination_stop = ""

            for o_stop, access_min in o_cands:
                wait_min = float(wait_times.get(str(o_stop), WAIT_DEFAULT_MIN))
                for d_stop, egress_min in d_cands:
                    path_metrics = stop_shortest_matrices.get(o_stop, d_stop)
                    if path_metrics is None:
                        continue
                    ivt_min = float(path_metrics[0])
                    transfer_min = float(path_metrics[1])
                    transfer_physical_min = float(path_metrics[2])
                    transfer_wait_min = float(path_metrics[3])
                    transfer_count = float(path_metrics[4])
                    distance_m = float(path_metrics[5])
                    ovt_min = float(access_min) + float(wait_min) + float(egress_min) + float(transfer_min)
                    total_min = float(ivt_min) + float(ovt_min)
                    if total_min < best_total:
                        best_total = total_min
                        best_ivt = float(ivt_min)
                        best_ovt = float(ovt_min)
                        best_distance_m = distance_m
                        best_access_min = float(access_min)
                        best_egress_min = float(egress_min)
                        best_initial_wait_min = float(wait_min)
                        best_transfer_min = transfer_min
                        best_transfer_physical_min = transfer_physical_min
                        best_transfer_wait_min = transfer_wait_min
                        best_transfer_count = transfer_count
                        best_origin_stop = str(o_stop)
                        best_destination_stop = str(d_stop)

            if not np.isfinite(best_total):
                no_path_pairs += 1
                continue

            add_stop_usage(
                best_origin_stop,
                best_destination_stop,
                best_access_min,
                best_egress_min,
            )
            index = (origin_idx, destination_idx)
            matrices["ivt"][index] = best_ivt
            matrices["ovt"][index] = best_ovt
            matrices["total"][index] = best_total
            matrices["distance"][index] = best_distance_m
            matrices["transfer_output"][index] = best_transfer_min
            matrices["transfer_count"][index] = best_transfer_count
            matrices["access"][index] = best_access_min
            matrices["egress"][index] = best_egress_min
            matrices["initial_wait"][index] = best_initial_wait_min
            matrices["transfer_physical"][index] = best_transfer_physical_min
            matrices["transfer_wait"][index] = best_transfer_wait_min
            matrices["transfer_total"][index] = best_transfer_min
            matrices["chosen_origin_stop"][index] = best_origin_stop
            matrices["chosen_destination_stop"][index] = best_destination_stop

    usage_rows = []
    for entry in stop_usage.values():
        origin_count = int(entry["origin_access_count"])
        dest_count = int(entry["destination_egress_count"])
        transfer_path_count = int(entry["transfer_count_on_paths"])
        usage_rows.append(
            {
                "stop_id": entry["stop_id"],
                "stop_name": entry["stop_name"],
                "origin_access_count": origin_count,
                "destination_egress_count": dest_count,
                "transfer_count_on_paths": transfer_path_count,
                "total_chosen_path_count": origin_count + dest_count + transfer_path_count,
                "mean_access_min": float(entry["access_min_sum"]) / origin_count if origin_count else np.nan,
                "mean_egress_min": float(entry["egress_min_sum"]) / dest_count if dest_count else np.nan,
            }
        )
    stop_usage_df = pd.DataFrame(usage_rows)
    if not stop_usage_df.empty:
        stop_usage_df = stop_usage_df.sort_values(
            ["total_chosen_path_count", "origin_access_count", "destination_egress_count"],
            ascending=False,
        ).reset_index(drop=True)

    diag = {
        "total_pairs": int(total_pairs),
        "no_path_pairs": int(no_path_pairs),
        "coverage_pct": float(100.0 * (total_pairs - no_path_pairs) / max(1, total_pairs)),
        "ivt_p95": float(np.quantile(matrices["ivt"], 0.95)),
        "ovt_p95": float(np.quantile(matrices["ovt"], 0.95)),
        "transfer_time_p95": float(np.quantile(matrices["transfer_output"], 0.95)),
        "transfer_count_p95": float(np.quantile(matrices["transfer_count"], 0.95)),
        "total_p95": float(np.quantile(matrices["total"], 0.95)),
        "distance_p95_km": float(np.quantile(matrices["distance"], 0.95) / 1000.0),
    }
    return matrices, stop_usage_df, diag


def save_zone_matrix(
    zone_ids: list[str],
    values: np.ndarray,
    path: Path,
    value_col: str = "pt_time",
) -> None:
    ids = np.asarray(zone_ids, dtype=object)
    n_zones = len(ids)
    table = pd.DataFrame(
        {
            "origin": np.repeat(ids, n_zones),
            "destination": np.tile(ids, n_zones),
            value_col: values.reshape(-1),
        }
    )
    table.to_parquet(path, index=False)


def save_pt_component_outputs(
    matrices: dict[str, np.ndarray],
    zone_ids: list[str],
    mode: str,
    period_name: str,
) -> list[str]:
    saved: list[str] = []
    for component in PT_OVT_COMPONENTS:
        path = component_matrix_path(component, mode, period_name)
        save_zone_matrix(zone_ids, matrices[component], path)
        saved.append(str(path))
    for kind, component in (
        ("origin", "chosen_origin_stop"),
        ("destination", "chosen_destination_stop"),
    ):
        path = chosen_stop_matrix_path(kind, mode, period_name)
        save_zone_matrix(zone_ids, matrices[component], path, value_col="stop_id")
        saved.append(str(path))
    return saved


def save_mode_zone_outputs(
    matrices: dict[str, np.ndarray],
    zone_ids: list[str],
    mode: str,
    period_name: str,
) -> list[str]:
    saved = save_pt_component_outputs(matrices, zone_ids, mode, period_name)
    if mode == "walk":
        outputs = (
            ("ivt", OUTPUT_IVT_WALK_ASP, "pt_time"),
            ("ovt", OUTPUT_OVT_WALK_ASP, "pt_time"),
            ("distance", OUTPUT_DIST_WALK_ASP, "distance_m"),
            ("transfer_output", OUTPUT_TRANSFER_WALK_ASP, "pt_time"),
            ("transfer_count", OUTPUT_TRANSFER_COUNT_WALK_ASP, "transfer_count"),
        )
    elif mode == "bike":
        outputs = (
            ("ivt", OUTPUT_IVT_BIKE_ASP, "pt_time"),
            ("ovt", OUTPUT_OVT_BIKE_ASP, "pt_time"),
            ("distance", OUTPUT_DIST_BIKE_ASP, "distance_m"),
            ("transfer_output", OUTPUT_TRANSFER_BIKE_ASP, "pt_time"),
            ("transfer_count", OUTPUT_TRANSFER_COUNT_BIKE_ASP, "transfer_count"),
        )
    else:
        raise ValueError(f"Unsupported PT access mode: {mode}")

    for matrix_name, path, value_col in outputs:
        save_zone_matrix(zone_ids, matrices[matrix_name], path, value_col=value_col)
    return saved


def component_consistency_row(
    mode: str,
    matrices: dict[str, np.ndarray],
) -> dict[str, float | int | str]:
    ovt = matrices["ovt"]
    component_sum = np.zeros_like(ovt, dtype=float)
    for component in ("access", "initial_wait", "egress", "transfer_physical", "transfer_wait"):
        component_sum += matrices[component]
    valid = np.isfinite(ovt) & (ovt < float(PT_NO_PATH_MIN)) & np.isfinite(component_sum)
    errors = np.abs(component_sum[valid] - ovt[valid])
    return {
        "mode": mode,
        "n_od": int(ovt.size),
        "n_valid": int(valid.sum()),
        "mean_abs_ovt_component_error": float(errors.mean()) if errors.size else 0.0,
        "max_abs_ovt_component_error": float(errors.max()) if errors.size else 0.0,
        "p95_abs_ovt_component_error": float(np.quantile(errors, 0.95)) if errors.size else 0.0,
    }


def access_floor_from_candidates(access_df: pd.DataFrame, mode: str) -> dict[str, object]:
    floor_min = 1.0 if mode == "walk" else 0.5
    if access_df.empty or "access_min" not in access_df.columns:
        return {
            "access_floor_min": float(floor_min),
            "sample_size": 0,
            "source": "fallback_no_access_candidates",
        }
    values = pd.to_numeric(access_df["access_min"], errors="coerce")
    if "fallback" in access_df.columns:
        values = values.loc[~access_df["fallback"].astype(bool)]
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    values = values[values >= 0.0]
    if values.empty:
        return {
            "access_floor_min": float(floor_min),
            "sample_size": 0,
            "source": "fallback_no_nonfallback_access_candidates",
        }
    return {
        "access_floor_min": float(max(floor_min, values.quantile(0.10))),
        "sample_size": int(len(values)),
        "source": "p10_nonfallback_access_min",
    }


def plot_pt_network(stops_gdf: gpd.GeoDataFrame, zones_gdf: gpd.GeoDataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 12))
    zones_gdf.plot(ax=ax, alpha=0.08, edgecolor="lightgray", linewidth=0.5)
    stops_gdf.plot(ax=ax, color="red", markersize=2, alpha=0.65)
    bounds = zones_gdf.total_bounds
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_title("PT Stops and NPVM Zones", fontsize=14, fontweight="bold")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main(
    gtfs_zip: Path = GTFS_ZIP,
    service_date: str | None = SERVICE_DATE,
    extract_only: bool = False,
) -> None:
    t0 = time.time()
    print("\n" + "=" * 80)
    print("GTFS ANALYSIS")
    print("=" * 80)
    print(f"gtfs_zip={gtfs_zip}")
    print(f"zones_file={ZONES_FILE}")
    print(f"access_walk_graph={ACCESS_WALK_GRAPH_FILE}")
    print(f"access_bike_graph={ACCESS_BIKE_GRAPH_FILE}")
    print(f"service_date={service_date or 'all services'}")
    print("time_period=ASP")
    print(f"spatial_period_filter={SPATIAL_PERIOD_FILTER}")
    print(f"study_area_buffer_m={STUDY_AREA_BUFFER_M}")
    print(f"extract_only={extract_only}")

    # Load core inputs.
    if not gtfs_zip.exists():
        raise FileNotFoundError(f"GTFS zip not found: {gtfs_zip}")
    if not ZONES_FILE.exists():
        raise FileNotFoundError(f"Zones file not found: {ZONES_FILE}")

    gtfs_tables = load_gtfs_tables(gtfs_zip)
    validate_gtfs(gtfs_tables)
    stops_all_gdf = process_stops(gtfs_tables)
    zones = load_zones(ZONES_FILE)

    service_stop_ids = study_area_stop_ids(
        stops_all_gdf,
        zones,
        buffer_m=float(STUDY_AREA_BUFFER_M),
    )
    if TRIM_STOPS_TO_STUDY_AREA:
        stops_gdf = stops_all_gdf[stops_all_gdf["stop_id"].astype(str).isin(service_stop_ids)].copy()
    else:
        stops_gdf = stops_all_gdf.copy()
    graph_stop_ids = set(stops_gdf["stop_id"].astype(str))

    active_trips = filter_trips_by_service_date(gtfs_tables, service_date)
    gtfs_tables["trips"] = active_trips
    routes_df = process_routes(gtfs_tables)
    stop_times = prepare_stop_times(gtfs_tables)
    active_trip_ids = set(active_trips["trip_id"].astype(str))
    stop_times = stop_times[stop_times["trip_id"].astype(str).isin(active_trip_ids)].copy()
    stop_times_area = stop_times[stop_times["stop_id"].astype(str).isin(graph_stop_ids)].copy()

    stops_gdf = annotate_stops_with_route_families(
        stops_gdf=stops_gdf,
        routes_df=routes_df,
        trips_df=active_trips,
        stop_times=stop_times_area,
    )
    candidate_stops_gdf = stops_gdf[stops_gdf["route_families"].astype(str).ne("")].copy()
    if candidate_stops_gdf.empty:
        print("WARNING: No active-service stops found inside the study area; using all study-area stops as candidates.")
        candidate_stops_gdf = stops_gdf.copy()
    gtfs_tables["_processed_stops_gdf"] = stops_gdf

    # Save processed GTFS tables.
    stops_gdf.to_parquet(OUTPUT_STOPS, index=False)
    routes_df.to_parquet(OUTPUT_ROUTES, index=False)
    print(f"Saved: {OUTPUT_STOPS}")
    print(f"Saved: {OUTPUT_ROUTES}")
    print(
        "GTFS scope: "
        f"all_stops={len(stops_all_gdf):,}, study_stops={len(stops_gdf):,}, "
        f"active_candidate_stops={len(candidate_stops_gdf):,}, "
        f"active_trips={len(active_trips):,}, study_stop_times={len(stop_times_area):,}"
    )

    if extract_only:
        dt = time.time() - t0
        print("\n" + "=" * 80)
        print("GTFS INPUT EXTRACTION COMPLETE")
        print("=" * 80)
        print("Run this script again without --extract-only to build the PT skims.")
        print(f"elapsed_min={dt/60.0:.2f}")
        return

    # Load full-canton PT access graphs and build time graphs.
    G_walk_raw = load_access_graph(ACCESS_WALK_GRAPH_FILE)
    G_bike_raw = load_access_graph(ACCESS_BIKE_GRAPH_FILE)
    G_walk_time = build_access_time_graph(
        G_walk_raw,
        speed_kph=float(ACCESS_SPEED_KPH["walk"]),
        force_twoway=ASSUME_TWOWAY_ACCESS,
    )
    G_bike_time = build_access_time_graph(
        G_bike_raw,
        speed_kph=float(ACCESS_SPEED_KPH["bike"]),
        force_twoway=ASSUME_TWOWAY_ACCESS,
    )

    # Zone-stop candidate sets.
    access_walk = compute_zone_stop_candidates(
        zones_gdf=zones,
        stops_gdf=candidate_stops_gdf,
        G_time=G_walk_time,
        G_raw=G_walk_raw,
        mode="walk",
        top_k=int(ACCESS_TOP_K["walk"]),
        max_access_min=float(ACCESS_MAX_MIN["walk"]),
        extra_stop_mask=candidate_stops_gdf["serves_rail"].to_numpy(dtype=bool),
        extra_top_k=int(EXTRA_RAIL_ACCESS_TOP_K["walk"]),
        extra_label="nearest_rail",
    )
    access_bike = compute_zone_stop_candidates(
        zones_gdf=zones,
        stops_gdf=candidate_stops_gdf,
        G_time=G_bike_time,
        G_raw=G_bike_raw,
        mode="bike",
        top_k=int(ACCESS_TOP_K["bike"]),
        max_access_min=float(ACCESS_MAX_MIN["bike"]),
        extra_stop_mask=candidate_stops_gdf["serves_rail"].to_numpy(dtype=bool),
        extra_top_k=int(EXTRA_RAIL_ACCESS_TOP_K["bike"]),
        extra_label="nearest_rail",
    )
    egress_walk = build_stop_zone_egress_from_access(access_walk, mode="walk")
    egress_bike = build_stop_zone_egress_from_access(access_bike, mode="bike")

    access_walk.to_parquet(OUTPUT_ZONE_STOP_ACCESS_WALK, index=False)
    access_bike.to_parquet(OUTPUT_ZONE_STOP_ACCESS_BIKE, index=False)
    egress_walk.to_parquet(OUTPUT_STOP_ZONE_EGRESS_WALK, index=False)
    egress_bike.to_parquet(OUTPUT_STOP_ZONE_EGRESS_BIKE, index=False)
    print(f"Saved: {OUTPUT_ZONE_STOP_ACCESS_WALK}")
    print(f"Saved: {OUTPUT_ZONE_STOP_ACCESS_BIKE}")
    print(f"Saved: {OUTPUT_STOP_ZONE_EGRESS_WALK}")
    print(f"Saved: {OUTPUT_STOP_ZONE_EGRESS_BIKE}")

    # Build candidate lookups once.
    access_lookup_walk = build_candidate_lookup(access_walk, time_col="access_min")
    access_lookup_bike = build_candidate_lookup(access_bike, time_col="access_min")
    egress_lookup_walk = build_candidate_lookup(
        egress_walk.rename(columns={"egress_min": "access_min"}),
        time_col="access_min",
    )
    egress_lookup_bike = build_candidate_lookup(
        egress_bike.rename(columns={"egress_min": "access_min"}),
        time_col="access_min",
    )

    zone_ids = zones["grid_id"].astype(str).tolist()
    diagnostics: dict[str, object] = {
        "gtfs_zip": str(gtfs_zip),
        "service_date": service_date,
        "spatial_period_filter": bool(SPATIAL_PERIOD_FILTER),
        "study_area_buffer_m": float(STUDY_AREA_BUFFER_M),
        "stop_to_stop_distance_factor": float(STOP_TO_STOP_DISTANCE_FACTOR),
        "transfer_time_rule": "transfer walking/minimum time + half-headway wait at transfer boarding stop",
        "zones": int(len(zone_ids)),
        "stops": int(len(stops_gdf)),
        "all_feed_stops": int(len(stops_all_gdf)),
        "active_candidate_stops": int(len(candidate_stops_gdf)),
        "active_trips_after_service_date": int(len(active_trips)),
        "study_stop_times_after_service_date": int(len(stop_times_area)),
        "study_rail_stops": int(stops_gdf["serves_rail"].sum()) if "serves_rail" in stops_gdf.columns else None,
        "access": {
            "walk_candidates": int(len(access_walk)),
            "bike_candidates": int(len(access_bike)),
            "walk_avg_candidates_per_zone": float(len(access_walk) / max(1, len(zone_ids))),
            "bike_avg_candidates_per_zone": float(len(access_bike) / max(1, len(zone_ids))),
            "walk_fallback_rows": int(access_walk["fallback"].sum()) if "fallback" in access_walk.columns else None,
            "bike_fallback_rows": int(access_bike["fallback"].sum()) if "fallback" in access_bike.columns else None,
            "walk_rail_candidate_rows": int(access_walk["candidate_type"].eq("nearest_rail").sum())
            if "candidate_type" in access_walk.columns
            else None,
            "bike_rail_candidate_rows": int(access_bike["candidate_type"].eq("nearest_rail").sum())
            if "candidate_type" in access_bike.columns
            else None,
        },
        "periods": {},
    }

    wait_times_by_period: dict[str, dict[str, float]] = {}
    stop_names = dict(
        zip(stops_gdf["stop_id"].astype(str), stops_gdf.get("stop_name", pd.Series("", index=stops_gdf.index)).astype(str))
    )

    for period_name in TIME_PERIODS:
        print(f"\n--- Period: {period_name} ---")
        period_filter_stop_times = stop_times_area if SPATIAL_PERIOD_FILTER else stop_times
        wait_times = compute_wait_times(stop_times_area, period_name)
        wait_times_by_period[period_name] = wait_times
        G_pt, direct_ivt_df, period_trip_count, transfer_diag = build_period_transit_graph(
            period_filter_stop_times,
            stops_gdf,
            period_name,
            gtfs_tables=gtfs_tables,
            wait_times=wait_times,
            selection_stop_ids=graph_stop_ids,
            graph_stop_ids=graph_stop_ids,
        )

        direct_ivt_df.to_parquet(OUTPUT_IVT_ASP, index=False)
        print(f"Saved: {OUTPUT_IVT_ASP}")

        # Transit shortest paths among union of candidate stops.
        origin_union = set(access_walk["stop_id"].astype(str)).union(set(access_bike["stop_id"].astype(str)))
        dest_union = set(egress_walk["stop_id"].astype(str)).union(set(egress_bike["stop_id"].astype(str)))
        stop_shortest_matrices = compute_stop_shortest_matrices(
            G_pt,
            origin_union,
            dest_union,
            OUTPUT_DIR / f"_pt_stop_shortest_{period_name.lower()}.npy",
        )
        component_output_files: list[str] = []
        consistency_rows: list[dict[str, float | int | str]] = []
        try:
            matrices_walk, stop_usage_walk, diag_walk = compose_zone_components_for_mode(
                zone_ids=zone_ids,
                access_lookup=access_lookup_walk,
                egress_lookup=egress_lookup_walk,
                wait_times=wait_times,
                stop_shortest_matrices=stop_shortest_matrices,
                mode="walk",
                stop_names=stop_names,
            )
            consistency_rows.append(component_consistency_row("walk", matrices_walk))
            component_output_files.extend(
                save_mode_zone_outputs(matrices_walk, zone_ids, "walk", period_name)
            )
            stop_usage_walk.to_csv(stop_usage_path("walk", period_name), index=False)
            ivt_walk = matrices_walk["ivt"].copy()
            del matrices_walk, stop_usage_walk

            matrices_bike, stop_usage_bike, diag_bike = compose_zone_components_for_mode(
                zone_ids=zone_ids,
                access_lookup=access_lookup_bike,
                egress_lookup=egress_lookup_bike,
                wait_times=wait_times,
                stop_shortest_matrices=stop_shortest_matrices,
                mode="bike",
                stop_names=stop_names,
            )
            consistency_rows.append(component_consistency_row("bike", matrices_bike))
            component_output_files.extend(
                save_mode_zone_outputs(matrices_bike, zone_ids, "bike", period_name)
            )
            stop_usage_bike.to_csv(stop_usage_path("bike", period_name), index=False)

            ivt_shared = np.minimum(ivt_walk, matrices_bike["ivt"])
            save_zone_matrix(zone_ids, ivt_shared, OUTPUT_IVT_ZONE_ASP)
            del matrices_bike, stop_usage_bike, ivt_walk, ivt_shared
        finally:
            stop_shortest_matrices.close()

        consistency_df = pd.DataFrame(consistency_rows)
        consistency_df.to_csv(component_consistency_path(period_name), index=False)
        access_floors = {
            "period": period_name,
            "walk": access_floor_from_candidates(access_walk, "walk"),
            "bike": access_floor_from_candidates(access_bike, "bike"),
            "hub_specific": {
                "available": False,
                "note": "Target-hub-specific floors are computed later by interventions when hub stop sets are known.",
            },
        }
        with open(access_component_floors_path(period_name), "w", encoding="utf-8") as f:
            json.dump(access_floors, f, indent=2)
        print(f"Saved PT component matrices for {period_name}: {len(component_output_files)} files")
        print(f"Saved: {stop_usage_path('walk', period_name)}")
        print(f"Saved: {stop_usage_path('bike', period_name)}")
        print(f"Saved: {component_consistency_path(period_name)}")
        print(f"Saved: {access_component_floors_path(period_name)}")

        print(f"Saved: {OUTPUT_IVT_ZONE_ASP}")
        print(f"Saved: {OUTPUT_IVT_WALK_ASP}")
        print(f"Saved: {OUTPUT_IVT_BIKE_ASP}")
        print(f"Saved: {OUTPUT_OVT_WALK_ASP}")
        print(f"Saved: {OUTPUT_OVT_BIKE_ASP}")
        print(f"Saved: {OUTPUT_DIST_WALK_ASP}")
        print(f"Saved: {OUTPUT_DIST_BIKE_ASP}")
        print(f"Saved: {OUTPUT_TRANSFER_WALK_ASP}")
        print(f"Saved: {OUTPUT_TRANSFER_BIKE_ASP}")
        print(f"Saved: {OUTPUT_TRANSFER_COUNT_WALK_ASP}")
        print(f"Saved: {OUTPUT_TRANSFER_COUNT_BIKE_ASP}")

        diagnostics["periods"][period_name] = {
            "period_trip_count": int(period_trip_count),
            "pt_graph_nodes": int(len(G_pt.nodes)),
            "pt_graph_edges": int(len(G_pt.edges)),
            "wait_stops": int(len(wait_times)),
            "transfers": transfer_diag,
            "walk": diag_walk,
            "bike": diag_bike,
            "component_files": component_output_files,
            "component_consistency": consistency_df.to_dict(orient="records"),
            "access_component_floors": access_floors,
        }

    with open(OUTPUT_WAIT, "w", encoding="utf-8") as f:
        json.dump(wait_times_by_period, f, indent=2)
    print(f"Saved: {OUTPUT_WAIT}")

    with open(OUTPUT_DIAG, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"Saved: {OUTPUT_DIAG}")

    plot_pt_network(stops_gdf, zones, OUTPUT_VIZ)
    print(f"Saved: {OUTPUT_VIZ}")

    dt = time.time() - t0
    print("\n" + "=" * 80)
    print("GTFS PROCESSING COMPLETE")
    print("=" * 80)
    print(f"zones={len(zone_ids):,}, stops={len(stops_gdf):,}")
    print(f"elapsed_min={dt/60.0:.2f}")


def _wide_numeric(path: Path, value_column: str, zone_ids: pd.Index, fill_value: float) -> pd.DataFrame:
    table = pd.read_parquet(path)
    table["origin"] = table["origin"].astype(str)
    table["destination"] = table["destination"].astype(str)
    table[value_column] = pd.to_numeric(table[value_column], errors="coerce").fillna(fill_value)
    matrix = table.pivot_table(
        index="origin",
        columns="destination",
        values=value_column,
        aggfunc="min",
        fill_value=fill_value,
    )
    return matrix.reindex(index=zone_ids, columns=zone_ids, fill_value=fill_value).astype(float)


def _wide_text(path: Path, zone_ids: pd.Index) -> pd.DataFrame:
    table = pd.read_parquet(path)
    table["origin"] = table["origin"].astype(str)
    table["destination"] = table["destination"].astype(str)
    table["stop_id"] = table["stop_id"].fillna("").astype(str)
    matrix = table.pivot_table(
        index="origin",
        columns="destination",
        values="stop_id",
        aggfunc="first",
        fill_value="",
    )
    return matrix.reindex(index=zone_ids, columns=zone_ids, fill_value="").astype(str)


def export_prepared_pt_skims() -> None:
    """Convert processor tables to the wide matrices used by the model run."""
    zone_ids = pd.Index(load_zones(ZONES_FILE)["grid_id"].astype(str), dtype="object")
    SKIM_DIR.mkdir(parents=True, exist_ok=True)
    numeric_outputs = {
        "pt_walk_ivt_min.parquet": (OUTPUT_IVT_WALK_ASP, "pt_time", PT_NO_PATH_MIN),
        "pt_walk_distance_m.parquet": (OUTPUT_DIST_WALK_ASP, "distance_m", 0.0),
        "pt_walk_access_min.parquet": (component_matrix_path("access", "walk", "ASP"), "pt_time", 0.0),
        "pt_walk_egress_min.parquet": (component_matrix_path("egress", "walk", "ASP"), "pt_time", 0.0),
        "pt_walk_initial_wait_min.parquet": (component_matrix_path("initial_wait", "walk", "ASP"), "pt_time", 0.0),
        "pt_walk_transfer_physical_min.parquet": (component_matrix_path("transfer_physical", "walk", "ASP"), "pt_time", 0.0),
        "pt_walk_transfer_wait_min.parquet": (component_matrix_path("transfer_wait", "walk", "ASP"), "pt_time", 0.0),
        "pt_walk_transfer_count.parquet": (component_matrix_path("transfer_count", "walk", "ASP"), "pt_time", 0.0),
        "pt_bike_ivt_min.parquet": (OUTPUT_IVT_BIKE_ASP, "pt_time", PT_NO_PATH_MIN),
        "pt_bike_distance_m.parquet": (OUTPUT_DIST_BIKE_ASP, "distance_m", 0.0),
        "pt_bike_access_min.parquet": (component_matrix_path("access", "bike", "ASP"), "pt_time", 0.0),
        "pt_bike_egress_min.parquet": (component_matrix_path("egress", "bike", "ASP"), "pt_time", 0.0),
        "pt_bike_initial_wait_min.parquet": (component_matrix_path("initial_wait", "bike", "ASP"), "pt_time", 0.0),
        "pt_bike_transfer_physical_min.parquet": (component_matrix_path("transfer_physical", "bike", "ASP"), "pt_time", 0.0),
        "pt_bike_transfer_wait_min.parquet": (component_matrix_path("transfer_wait", "bike", "ASP"), "pt_time", 0.0),
        "pt_bike_transfer_count.parquet": (component_matrix_path("transfer_count", "bike", "ASP"), "pt_time", 0.0),
    }
    for output_name, (source, value_column, fill_value) in numeric_outputs.items():
        _wide_numeric(source, value_column, zone_ids, fill_value).to_parquet(SKIM_DIR / output_name)

    text_outputs = {
        "pt_walk_chosen_origin_stop.parquet": chosen_stop_matrix_path("origin", "walk", "ASP"),
        "pt_walk_chosen_destination_stop.parquet": chosen_stop_matrix_path("destination", "walk", "ASP"),
        "pt_bike_chosen_origin_stop.parquet": chosen_stop_matrix_path("origin", "bike", "ASP"),
        "pt_bike_chosen_destination_stop.parquet": chosen_stop_matrix_path("destination", "bike", "ASP"),
    }
    for output_name, source in text_outputs.items():
        _wide_text(source, zone_ids).to_parquet(SKIM_DIR / output_name)


def process_gtfs(
    gtfs_zip: Path = GTFS_ZIP,
    service_date: str | None = SERVICE_DATE,
    extract_only: bool = False,
) -> None:
    main(
        gtfs_zip=gtfs_zip,
        service_date=service_date,
        extract_only=extract_only,
    )
    if not extract_only:
        export_prepared_pt_skims()


def _build_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Process GTFS into Zurich PT skim inputs.")
    parser.add_argument("--gtfs-zip", default=str(GTFS_ZIP), help="Path to GTFS zip.")
    parser.add_argument(
        "--service-date",
        default=SERVICE_DATE or "",
        help="Representative service date, YYYY-MM-DD or YYYYMMDD. Use empty string for all services.",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only export processed GTFS stops and routes.",
    )
    parser.add_argument(
        "--no-spatial-period-filter",
        action="store_true",
        help="Use trips with any event in the period, regardless of study area.",
    )
    parser.add_argument(
        "--no-transfers",
        action="store_true",
        help="Disable GTFS, parent-station, and nearby walking transfer edges.",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    if args.no_spatial_period_filter:
        SPATIAL_PERIOD_FILTER = False
    if args.no_transfers:
        USE_GTFS_TRANSFERS = False
        USE_PARENT_STATION_TRANSFERS = False
        USE_NEARBY_STOP_TRANSFERS = False
    process_gtfs(
        gtfs_zip=Path(args.gtfs_zip),
        service_date=args.service_date or None,
        extract_only=bool(args.extract_only),
    )
