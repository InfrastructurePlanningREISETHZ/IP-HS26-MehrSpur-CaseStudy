from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import LineString

from config import (
    ASSIGNMENT_NETWORK_FILE,
    BASE_NETWORK_FILE,
    BIKE_GRAPH_FILE,
    BIKE_SPEED_CANTON_KPH,
    BIKE_SPEED_CITY_KPH,
    MAX_WALK_DISTANCE_KM,
    NETWORK_POLICY_CITY_FILE,
    NO_PATH_TIME_MIN,
    PT_ACCESS_WALK_GRAPH_FILE,
    ROAD_GRAPH_FILE,
    SKIM_DIR,
    WALK_SPEED_KPH,
    WALK_GRAPH_FILE,
)
from routing import RoutingNetwork


ROAD_CAPACITY_BY_TYPE_AND_LANES = {
    "motorway": {1: 1700.0, 2: 4000.0, 3: 5800.0, 4: 7850.0},
    "trunk_limited_access": {1: 1700.0, 2: 4000.0, 3: 5800.0},
    "motorway_link": {1: 1000.0, 2: 1700.0, 3: 2800.0},
    "trunk_link": {1: 1000.0, 2: 1700.0, 3: 2800.0},
    "primary_outer": {1: 1200.0, 2: 2100.0, 3: 3200.0},
    "primary_inner": {1: 1100.0, 2: 1700.0, 3: 2800.0},
    "secondary": {1: 1000.0, 2: 1600.0, 3: 2600.0},
    "tertiary": {1: 900.0, 2: 1500.0, 3: 2400.0},
    "residential_unclassified": {1: 700.0, 2: 1100.0},
    "living_street": {1: 400.0},
}
MANUAL_CONNECTOR_CAPACITY_VPH = 99999.0
DRIVE_SPEED_KPH_BY_HIGHWAY = {
    "motorway": {"city": 80.0, "rest": 100.0},
    "trunk": {"city": 70.0, "rest": 90.0},
    "motorway_link": {"city": 60.0, "rest": 80.0},
    "trunk_link": {"city": 60.0, "rest": 80.0},
    "primary": {"city": 40.0, "rest": 60.0},
    "primary_link": {"city": 40.0, "rest": 60.0},
    "secondary": {"city": 35.0, "rest": 55.0},
    "secondary_link": {"city": 35.0, "rest": 55.0},
    "tertiary": {"city": 30.0, "rest": 45.0},
    "tertiary_link": {"city": 30.0, "rest": 45.0},
    "residential": {"city": 25.0, "rest": 35.0},
    "unclassified": {"city": 30.0, "rest": 45.0},
    "road": {"city": 30.0, "rest": 45.0},
    "living_street": {"city": 15.0, "rest": 20.0},
}


def _load_pickle(path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing prepared network input: {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_base_network(path: str | Path = BASE_NETWORK_FILE) -> dict:
    payload = _load_pickle(path)
    edges = payload["edges"].copy()
    nodes = payload["nodes"].copy()

    # Keep the shared transport links separate from model-zone centroids.
    edges = edges.loc[~edges["highway"].astype(str).eq("manual_connector")].copy()
    nodes = nodes.loc[~nodes["nodeID"].astype(str).str.startswith("Z")].copy()
    return {"edges": edges, "nodes": nodes, "metadata": dict(payload.get("metadata", {}))}


def _mode_speed(mode: str, is_city: bool) -> float:
    if mode == "drive":
        return 20.0 if is_city else 55.0
    if mode == "walk":
        return float(WALK_SPEED_KPH)
    if mode == "bike":
        return float(BIKE_SPEED_CITY_KPH if is_city else BIKE_SPEED_CANTON_KPH)
    raise ValueError(f"Unsupported connector mode: {mode}")


def _mode_flags(mode: str) -> dict[str, int]:
    flags = {key: 0 for key in ("drive", "walk", "bike", "pt_walk", "pt_bike")}
    flags[mode] = 1
    if mode == "walk":
        flags["pt_walk"] = 1
    elif mode == "bike":
        flags["pt_bike"] = 1
    return flags


def _centroid_nodes(zones: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    nodes = gpd.GeoDataFrame(
        {
            "index": zones["grid_id"].astype(str),
            "x": zones["centroid"].x.astype(float),
            "y": zones["centroid"].y.astype(float),
            "nodeID": zones["grid_id"].astype(str),
            "geometry": zones["centroid"],
        },
        geometry="geometry",
        crs=zones.crs,
    )
    return nodes


def _connector_edges(
    zones: gpd.GeoDataFrame,
    base_nodes: gpd.GeoDataFrame,
    base_edges: gpd.GeoDataFrame,
    mode: str,
) -> gpd.GeoDataFrame:
    selected_zones = zones
    if mode == "walk":
        selected_zones = zones.loc[zones["Level"].astype(str).eq("Quartier")]

    mode_edges = base_edges.loc[pd.to_numeric(base_edges[mode], errors="coerce").fillna(0).gt(0)]
    mode_node_ids = set(mode_edges["source"].astype(str)).union(mode_edges["target"].astype(str))
    candidates = base_nodes.loc[base_nodes["nodeID"].astype(str).isin(mode_node_ids)].copy()
    candidates = candidates.drop_duplicates("nodeID").reset_index(drop=True)
    if candidates.empty:
        raise ValueError(f"The prepared base network has no nodes for mode {mode!r}.")

    candidate_xy = candidates[["x", "y"]].to_numpy(dtype=float)
    zone_xy = np.column_stack(
        (selected_zones["centroid"].x.to_numpy(dtype=float), selected_zones["centroid"].y.to_numpy(dtype=float))
    )
    _, nearest_positions = cKDTree(candidate_xy).query(zone_xy, k=1)
    nearest = candidates.iloc[np.asarray(nearest_positions, dtype=int)].reset_index(drop=True)

    records = []
    flags = _mode_flags(mode)
    for zone, road_node in zip(selected_zones.itertuples(index=False), nearest.itertuples(index=False)):
        zone_id = str(zone.grid_id)
        target_id = str(road_node.nodeID)
        x, y = float(zone.centroid.x), float(zone.centroid.y)
        nx_, ny_ = float(road_node.x), float(road_node.y)
        length_m = max(float(np.hypot(nx_ - x, ny_ - y)), 1.0)
        is_city = str(zone.Level) == "Quartier"
        speed_kph = _mode_speed(mode, is_city)
        time_min = length_m / 1000.0 / speed_kph * 60.0
        time = {mode: time_min}
        if mode == "walk":
            time["pt_walk"] = time_min
        elif mode == "bike":
            time["pt_bike"] = time_min

        common = {
            **flags,
            "length": length_m,
            "lanes": 1.0,
            "speed": speed_kph,
            "time": time,
            "highway": "manual_connector",
            "drive_area": "city" if is_city else "rest",
        }
        for source, target, coordinates, direction in (
            (zone_id, target_id, [(x, y), (nx_, ny_)], "out"),
            (target_id, zone_id, [(nx_, ny_), (x, y)], "in"),
        ):
            records.append(
                {
                    "source": source,
                    "target": target,
                    "edge": f"C_{mode}_{zone_id}_{direction}",
                    "geometry": LineString(coordinates),
                    **common,
                }
            )
    return gpd.GeoDataFrame(records, geometry="geometry", crs=zones.crs)


def build_model_network(zones: gpd.GeoDataFrame, base_path: str | Path = BASE_NETWORK_FILE) -> dict:
    """Attach direct NPVM centroids to the common prepared transport network."""
    base = load_base_network(base_path)
    edges = base["edges"].copy()
    nodes = base["nodes"].copy()
    connectors = [
        _connector_edges(zones, nodes, edges, mode)
        for mode in ("drive", "walk", "bike")
    ]
    edges = gpd.GeoDataFrame(
        pd.concat([edges, *connectors], ignore_index=True),
        geometry="geometry",
        crs=edges.crs,
    )
    nodes = gpd.GeoDataFrame(
        pd.concat([nodes, _centroid_nodes(zones)], ignore_index=True),
        geometry="geometry",
        crs=nodes.crs,
    ).drop_duplicates("nodeID")
    return {"edges": edges, "nodes": nodes, "metadata": base["metadata"]}


def _normalize_highway(value) -> str:
    if isinstance(value, (list, tuple, set)):
        value = list(value)[0] if value else "unclassified"
    return str(value or "unclassified").strip().lower()


def _capacity_category(highway, speed_kph: float, drive_area) -> str:
    highway = _normalize_highway(highway)
    if highway == "manual_connector":
        return "manual_connector"
    if highway == "motorway":
        return "motorway"
    if highway == "trunk":
        return "trunk_limited_access"
    if highway in {"motorway_link", "trunk_link"}:
        return highway
    if highway in {"primary", "primary_link"}:
        return "primary_inner" if str(drive_area).lower() == "city" or speed_kph <= 45.0 else "primary_outer"
    if highway in {"secondary", "secondary_link"}:
        return "secondary"
    if highway in {"tertiary", "tertiary_link"}:
        return "tertiary"
    if highway == "living_street":
        return "living_street"
    return "residential_unclassified"


def _capacity(highway, lanes, speed_kph: float, drive_area) -> float:
    category = _capacity_category(highway, speed_kph, drive_area)
    if category == "manual_connector":
        return MANUAL_CONNECTOR_CAPACITY_VPH
    table = ROAD_CAPACITY_BY_TYPE_AND_LANES[category]
    lane_value = pd.to_numeric(pd.Series([lanes]), errors="coerce").fillna(1.0).iloc[0]
    lane_value = max(float(lane_value), 1.0)
    lane_counts = np.asarray(sorted(table), dtype=float)
    capacities = np.asarray([table[int(value)] for value in lane_counts], dtype=float)
    return float(np.interp(lane_value, lane_counts, capacities))


def _time_value(value, mode: str) -> float:
    if isinstance(value, dict):
        return float(value.get(mode, np.inf))
    return float(value)


def build_assignment_network(
    zones: gpd.GeoDataFrame,
    model_network: dict,
    output_path: str | Path = ASSIGNMENT_NETWORK_FILE,
) -> dict:
    """Build the road-only assignment payload with NPVM capacity assumptions."""
    edges = model_network["edges"].copy()
    edges = edges.loc[pd.to_numeric(edges["drive"], errors="coerce").fillna(0).gt(0)].copy()
    edges["source_id"] = edges["source"].astype(str)
    edges["target_id"] = edges["target"].astype(str)
    edges["length_m"] = pd.to_numeric(edges["length"], errors="coerce")
    edges["free_flow_time_min"] = edges["time"].map(lambda value: _time_value(value, "drive"))
    edges["highway"] = edges["highway"].map(_normalize_highway)
    edges["lanes"] = pd.to_numeric(edges["lanes"], errors="coerce").fillna(1.0).clip(lower=1.0)
    edges["speed_kph"] = pd.to_numeric(edges["speed"], errors="coerce").fillna(0.0)
    edges["capacity_vph"] = [
        _capacity(highway, lanes, speed, area)
        for highway, lanes, speed, area in zip(edges["highway"], edges["lanes"], edges["speed_kph"], edges["drive_area"])
    ]
    edges = edges.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["source_id", "target_id", "length_m", "free_flow_time_min", "capacity_vph"]
    )
    edges = edges.loc[(edges["length_m"] > 0) & (edges["free_flow_time_min"] > 0)].copy()
    edges = edges.drop_duplicates(["source_id", "target_id"], keep="first")

    used_node_ids = set(edges["source_id"]).union(edges["target_id"])
    nodes = model_network["nodes"].copy()
    nodes["nodeID"] = nodes["nodeID"].astype(str)
    nodes = nodes.loc[nodes["nodeID"].isin(used_node_ids)].drop_duplicates("nodeID").copy()
    node_map = {node_id: position for position, node_id in enumerate(nodes["nodeID"])}
    nodes["node_id"] = nodes["nodeID"].map(node_map).astype(int)
    nodes["is_zone"] = nodes["nodeID"].isin(set(zones["grid_id"].astype(str)))
    nodes["zone_id"] = np.where(nodes["is_zone"], nodes["nodeID"], "")

    edges["source"] = edges["source_id"].map(node_map).astype(int)
    edges["target"] = edges["target_id"].map(node_map).astype(int)
    edges["alpha"] = 0.15
    edges["beta"] = 4.0
    edges["edge_id"] = edges["edge"].astype(str)
    zone_node_map = {
        zone_id: int(node_map[zone_id])
        for zone_id in zones["grid_id"].astype(str)
    }

    edge_columns = [
        "source",
        "target",
        "source_id",
        "target_id",
        "length_m",
        "free_flow_time_min",
        "highway",
        "drive_area",
        "speed_kph",
        "lanes",
        "capacity_vph",
        "alpha",
        "beta",
        "geometry",
        "edge_id",
    ]
    node_columns = ["node_id", "x", "y", "is_zone", "zone_id", "nodeID", "geometry"]
    payload = {
        "edges": gpd.GeoDataFrame(edges[edge_columns], geometry="geometry", crs=edges.crs),
        "nodes": gpd.GeoDataFrame(nodes[node_columns], geometry="geometry", crs=nodes.crs),
        "zone_node_map": zone_node_map,
        "metadata": {
            "capacity_policy": "npvm_directional_capacity_v1",
            "drive_speed_policy": "npvm_city_rest_road_type_v1",
            "connector_policy": "mode_speed_by_length",
        },
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return payload


def load_assignment_network(path: str | Path = ASSIGNMENT_NETWORK_FILE) -> dict:
    return _load_pickle(path)


def _routing_network(nodes: pd.DataFrame, edges: pd.DataFrame, weights: list[str], twoway: bool) -> RoutingNetwork:
    node_table = nodes.copy()
    node_table["node_id"] = node_table["node_id"].astype(np.int32)
    node_table = node_table.set_index("node_id")
    return RoutingNetwork(
        node_x=node_table["x"].astype(float),
        node_y=node_table["y"].astype(float),
        edge_from=edges["source"].astype(np.int32),
        edge_to=edges["target"].astype(np.int32),
        edge_weights=edges[weights].astype(float),
        twoway=twoway,
    )


def _all_pairs(
    network: RoutingNetwork,
    zone_nodes: np.ndarray,
    zone_ids: np.ndarray,
    impedance: str,
    no_path_value: float,
) -> pd.DataFrame:
    origins = np.repeat(zone_nodes, len(zone_nodes))
    destinations = np.tile(zone_nodes, len(zone_nodes))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unsigned integer: shortest path distance")
        values = np.asarray(
            network.shortest_path_lengths(origins, destinations, imp_name=impedance),
            dtype=float,
        ).reshape((len(zone_nodes), len(zone_nodes)))
    values[~np.isfinite(values) | (values >= 4_000_000.0)] = no_path_value
    np.fill_diagonal(values, 0.0)
    return pd.DataFrame(values, index=zone_ids, columns=zone_ids)


def _mode_graph_path(mode: str) -> Path:
    paths = {
        "drive": ROAD_GRAPH_FILE,
        "walk": WALK_GRAPH_FILE,
        "bike": BIKE_GRAPH_FILE,
    }
    try:
        return Path(paths[mode])
    except KeyError as error:
        raise ValueError(f"Unsupported direct mode: {mode}") from error


def _network_policy_city_polygon():
    area = gpd.read_parquet(NETWORK_POLICY_CITY_FILE)
    if area.empty:
        raise ValueError(f"Network policy area is empty: {NETWORK_POLICY_CITY_FILE}")
    return area.geometry.iloc[0]


def _edge_is_in_city(graph: nx.MultiDiGraph, source, target, data: dict, city_polygon) -> bool:
    geometry = data.get("geometry")
    if geometry is None:
        geometry = LineString(
            [
                (float(graph.nodes[source]["x"]), float(graph.nodes[source]["y"])),
                (float(graph.nodes[target]["x"]), float(graph.nodes[target]["y"])),
            ]
        )
    return bool(geometry.representative_point().within(city_polygon))


def _direct_mode_edges(graph: nx.MultiDiGraph, mode: str, city_polygon) -> pd.DataFrame:
    node_map = {node_id: position for position, node_id in enumerate(graph.nodes())}
    edge_weights: dict[tuple[int, int], list[float]] = {}

    def add_edge(source, target, time_min: float, length_m: float) -> None:
        key = (node_map[source], node_map[target])
        values = edge_weights.setdefault(key, [np.inf, np.inf])
        values[0] = min(values[0], time_min)
        values[1] = min(values[1], length_m)

    for source, target, _, data in graph.edges(keys=True, data=True):
        try:
            length_m = float(data.get("length"))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(length_m) or length_m <= 0.0:
            continue
        in_city = _edge_is_in_city(graph, source, target, data, city_polygon)
        if mode == "drive":
            speeds = DRIVE_SPEED_KPH_BY_HIGHWAY.get(
                _normalize_highway(data.get("highway")),
                DRIVE_SPEED_KPH_BY_HIGHWAY["unclassified"],
            )
            speed_kph = float(speeds["city" if in_city else "rest"])
        elif mode == "walk":
            speed_kph = float(WALK_SPEED_KPH)
        else:
            speed_kph = float(BIKE_SPEED_CITY_KPH if in_city else BIKE_SPEED_CANTON_KPH)
        time_min = length_m / 1000.0 / speed_kph * 60.0
        add_edge(source, target, time_min, length_m)
        if mode in {"walk", "bike"}:
            add_edge(target, source, time_min, length_m)

    return pd.DataFrame(
        [
            {
                "source": source,
                "target": target,
                "time_min": values[0],
                "length_m": values[1],
            }
            for (source, target), values in edge_weights.items()
        ]
    )


def mode_skims(zones: gpd.GeoDataFrame, mode: str, city_polygon) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build direct-mode skims by snapping each centroid to its mode graph."""
    graph = _load_pickle(_mode_graph_path(mode))
    node_ids = list(graph.nodes())
    node_xy = np.asarray(
        [[float(graph.nodes[node]["x"]), float(graph.nodes[node]["y"])] for node in node_ids],
        dtype=float,
    )
    zone_xy = np.column_stack(
        (zones["centroid"].x.to_numpy(dtype=float), zones["centroid"].y.to_numpy(dtype=float))
    )
    nearest_positions = np.asarray(
        [np.argmin(np.square(node_xy[:, 0] - x) + np.square(node_xy[:, 1] - y)) for x, y in zone_xy],
        dtype=np.int32,
    )

    nodes = pd.DataFrame(
        {
            "node_id": np.arange(len(node_ids), dtype=np.int32),
            "x": node_xy[:, 0],
            "y": node_xy[:, 1],
        }
    )
    edges = _direct_mode_edges(graph, mode, city_polygon)
    network = _routing_network(nodes, edges, ["time_min", "length_m"], twoway=False)
    zone_ids = zones["grid_id"].astype(str).to_numpy()
    zone_nodes = nearest_positions
    time = _all_pairs(network, zone_nodes, zone_ids, "time_min", NO_PATH_TIME_MIN)
    distance = _all_pairs(network, zone_nodes, zone_ids, "length_m", np.inf)
    return time, distance


def _full_walk_routing_network(graph: nx.MultiDiGraph) -> tuple[RoutingNetwork, np.ndarray]:
    node_ids = list(graph.nodes())
    node_map = {node_id: position for position, node_id in enumerate(node_ids)}
    node_xy = np.asarray(
        [[float(graph.nodes[node]["x"]), float(graph.nodes[node]["y"])] for node in node_ids],
        dtype=float,
    )

    sources = []
    targets = []
    lengths = []
    for source, target, data in graph.edges(data=True):
        try:
            length_m = float(data.get("length"))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(length_m) or length_m <= 0.0:
            continue
        sources.append(node_map[source])
        targets.append(node_map[target])
        lengths.append(length_m)

    nodes = pd.DataFrame(
        {
            "node_id": np.arange(len(node_ids), dtype=np.int32),
            "x": node_xy[:, 0],
            "y": node_xy[:, 1],
        }
    )
    edges = pd.DataFrame(
        {
            "source": np.asarray(sources, dtype=np.int32),
            "target": np.asarray(targets, dtype=np.int32),
            "length_m": np.asarray(lengths, dtype=float),
        }
    )
    return _routing_network(nodes, edges, ["length_m"], twoway=True), node_xy


def full_canton_walk_skims(
    zones: gpd.GeoDataFrame,
    city_polygon,
    graph_path: str | Path = PT_ACCESS_WALK_GRAPH_FILE,
    max_distance_km: float = MAX_WALK_DISTANCE_KM,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build bounded walk skims for all zones while retaining City walk paths."""
    graph = _load_pickle(graph_path)
    network, node_xy = _full_walk_routing_network(graph)
    zone_xy = np.column_stack(
        (zones["centroid"].x.to_numpy(dtype=float), zones["centroid"].y.to_numpy(dtype=float))
    )
    _, zone_nodes = cKDTree(node_xy).query(zone_xy, k=1)
    zone_nodes = np.asarray(zone_nodes, dtype=np.int32)
    snapped_xy = node_xy[zone_nodes]

    max_distance_m = float(max_distance_km) * 1000.0
    nearby = cKDTree(snapped_xy).query_ball_tree(cKDTree(snapped_xy), r=max_distance_m)
    origin_positions = np.repeat(
        np.arange(len(zones), dtype=np.int32),
        np.fromiter((len(items) for items in nearby), dtype=np.int32),
    )
    destination_positions = np.concatenate(nearby).astype(np.int32, copy=False)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unsigned integer: shortest path distance")
        route_distance = np.asarray(
            network.shortest_path_lengths(
                zone_nodes[origin_positions],
                zone_nodes[destination_positions],
                imp_name="length_m",
            ),
            dtype=float,
        )

    valid = (
        np.isfinite(route_distance)
        & (route_distance < 4_000_000.0)
        & (route_distance <= max_distance_m)
    )
    distance_values = np.full((len(zones), len(zones)), np.inf, dtype=float)
    time_values = np.full((len(zones), len(zones)), NO_PATH_TIME_MIN, dtype=float)
    rows = origin_positions[valid]
    columns = destination_positions[valid]
    distance_values[rows, columns] = route_distance[valid]
    time_values[rows, columns] = route_distance[valid] / 1000.0 / WALK_SPEED_KPH * 60.0
    np.fill_diagonal(distance_values, 0.0)
    np.fill_diagonal(time_values, 0.0)

    zone_ids = zones["grid_id"].astype(str).to_numpy()
    time = pd.DataFrame(time_values, index=zone_ids, columns=zone_ids)
    distance = pd.DataFrame(distance_values, index=zone_ids, columns=zone_ids)

    city_zones = zones.loc[zones["Level"].astype(str).eq("Quartier")].copy()
    if not city_zones.empty:
        city_time, city_distance = mode_skims(city_zones, "walk", city_polygon)
        city_ids = city_zones["grid_id"].astype(str)
        time.loc[city_ids, city_ids] = city_time
        distance.loc[city_ids, city_ids] = city_distance
    return time, distance


def process_networks(zones: gpd.GeoDataFrame) -> None:
    """Rebuild snapped direct-mode skims and road assignment inputs."""
    SKIM_DIR.mkdir(parents=True, exist_ok=True)
    model_network = build_model_network(zones)
    build_assignment_network(zones, model_network)
    city_polygon = _network_policy_city_polygon()
    for mode in ("drive", "walk", "bike"):
        if mode == "walk":
            time, distance = full_canton_walk_skims(zones, city_polygon)
        else:
            time, distance = mode_skims(zones, mode, city_polygon)
        time.to_parquet(SKIM_DIR / f"{mode}_time_min.parquet")
        distance.to_parquet(SKIM_DIR / f"{mode}_distance_m.parquet")
