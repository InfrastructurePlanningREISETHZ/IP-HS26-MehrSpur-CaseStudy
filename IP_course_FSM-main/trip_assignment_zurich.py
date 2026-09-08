from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd

from config import (
    ASSIGNMENT_MAX_ITERATIONS,
    ASSIGNMENT_MIN_ITERATIONS,
    ASSIGNMENT_OD_THRESHOLD,
    ASSIGNMENT_RELATIVE_GAP,
    ASSIGNMENT_STALL_ITERATIONS,
    DRIVE_OCCUPANCY,
    EXTERNAL_BACKGROUND_CANTON_FACTOR,
    EXTERNAL_BACKGROUND_CITY_FACTOR,
    NO_PATH_TIME_MIN,
)
from ta_lab.assignment.graph import Edge, Network
from ta_lab.assignment.line import cal_limit, cal_step
from routing import RoutingNetwork


def _pandana_network(nodes: pd.DataFrame, edges: pd.DataFrame) -> RoutingNetwork:
    node_table = nodes.copy()
    node_table["node_id"] = node_table["node_id"].astype(np.int32)
    node_table = node_table.set_index("node_id")
    return RoutingNetwork(
        node_x=node_table["x"].astype(float),
        node_y=node_table["y"].astype(float),
        edge_from=edges["source"].astype(np.int32),
        edge_to=edges["target"].astype(np.int32),
        edge_weights=edges[["time_min"]].astype(float),
        twoway=False,
    )


def _assignment_demand(
    drive_passenger_od: pd.DataFrame,
    road_background_od: pd.DataFrame,
    zones: pd.DataFrame,
    drive_occupancy: float,
) -> tuple[pd.DataFrame, dict]:
    index = drive_passenger_od.index.astype(str)
    columns = drive_passenger_od.columns.astype(str)
    drive_vehicles = (
        drive_passenger_od.reindex(index=index, columns=columns, fill_value=0.0)
        .to_numpy(dtype=float)
        / max(float(drive_occupancy), 1e-9)
    )
    background = road_background_od.reindex(
        index=index,
        columns=columns,
        fill_value=0.0,
    ).to_numpy(dtype=float)

    zone_table = zones.set_index(zones["grid_id"].astype(str)).reindex(index)
    endpoint_factor = np.where(
        zone_table["Level"].astype(str).eq("Quartier"),
        EXTERNAL_BACKGROUND_CITY_FACTOR,
        EXTERNAL_BACKGROUND_CANTON_FACTOR,
    )
    external_factor = endpoint_factor[:, None] + endpoint_factor[None, :]
    internal_road_demand = drive_vehicles + background
    external_background = internal_road_demand * external_factor
    total_values = internal_road_demand + external_background
    metadata = {
        "drive_vehicle_trips": float(drive_vehicles.sum()),
        "npvm_background_trips": float(background.sum()),
        "external_endpoint_background_trips": float(external_background.sum()),
        "assignment_vehicle_trips": float(total_values.sum()),
    }
    return pd.DataFrame(total_values, index=index, columns=columns), metadata


def _significant_od(
    matrix: pd.DataFrame,
    zone_node_map: dict[str, int],
    threshold: float,
) -> pd.DataFrame:
    values = matrix.to_numpy(dtype=float)
    selected = np.isfinite(values) & (values > float(threshold))
    np.fill_diagonal(selected, False)
    rows, columns = np.nonzero(selected)
    origins = matrix.index.astype(str).to_numpy()
    destinations = matrix.columns.astype(str).to_numpy()
    long = pd.DataFrame(
        {
            "origin": origins[rows],
            "destination": destinations[columns],
            "demand": values[rows, columns],
        }
    )
    long["origin_node"] = long["origin"].map(zone_node_map)
    long["destination_node"] = long["destination"].map(zone_node_map)
    return long.dropna(subset=["origin_node", "destination_node"])


def _all_or_nothing(
    network: RoutingNetwork,
    od: pd.DataFrame,
    edge_lookup: dict[tuple[int, int], str],
    edge_ids: list[str],
) -> tuple[dict[str, float], float]:
    edge_position = {edge_id: position for position, edge_id in enumerate(edge_ids)}
    node_base = max(max(pair) for pair in edge_lookup) + 1
    edge_codes = np.fromiter(
        (source * node_base + target for source, target in edge_lookup),
        dtype=np.int64,
    )
    positions = np.fromiter(
        (edge_position[edge_id] for edge_id in edge_lookup.values()),
        dtype=np.int64,
    )
    order = np.argsort(edge_codes)
    edge_codes = edge_codes[order]
    positions = positions[order]

    volumes = np.zeros(len(edge_ids), dtype=float)
    served = 0.0
    batch_size = 25_000
    for start in range(0, len(od), batch_size):
        batch = od.iloc[start : start + batch_size]
        paths = network.shortest_paths(
            batch["origin_node"].to_numpy(dtype=np.int32),
            batch["destination_node"].to_numpy(dtype=np.int32),
            imp_name="time_min",
        )
        demands = batch["demand"].to_numpy(dtype=float)
        valid_paths = []
        valid_demands = []
        for path, demand in zip(paths, demands):
            if path is None or len(path) < 2:
                continue
            valid_paths.append(np.asarray(path, dtype=np.int64))
            valid_demands.append(float(demand))
        if not valid_paths:
            continue

        lengths = np.fromiter((len(path) - 1 for path in valid_paths), dtype=np.int64)
        sources = np.concatenate([path[:-1] for path in valid_paths])
        targets = np.concatenate([path[1:] for path in valid_paths])
        route_codes = sources * node_base + targets
        matches = np.searchsorted(edge_codes, route_codes)
        in_network = matches < len(edge_codes)
        in_network[in_network] &= edge_codes[matches[in_network]] == route_codes[in_network]
        if not np.all(in_network):
            raise RuntimeError("A shortest path contains an edge outside the assignment network.")

        edge_demands = np.repeat(np.asarray(valid_demands), lengths)
        volumes += np.bincount(
            positions[matches],
            weights=edge_demands,
            minlength=len(edge_ids),
        )
        served += float(np.sum(valid_demands))

    return dict(zip(edge_ids, volumes.tolist())), served


def _edge_times(edges: pd.DataFrame, volumes: dict[str, float]) -> np.ndarray:
    flow = edges["edge_id"].map(volumes).fillna(0.0).to_numpy(dtype=float)
    free_flow = edges["free_flow_time_min"].to_numpy(dtype=float)
    capacity = edges["capacity_vph"].to_numpy(dtype=float)
    alpha = edges["alpha"].to_numpy(dtype=float)
    beta = edges["beta"].to_numpy(dtype=float)
    times = free_flow * (1.0 + alpha * np.power(flow / np.maximum(capacity, 1e-9), beta))
    return np.clip(times, 1e-4, 500.0)


def _selected_route_skims(
    network: RoutingNetwork,
    od: pd.DataFrame,
    edges: pd.DataFrame,
    zone_ids: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Return final time and distance only for OD pairs loaded in assignment."""
    position = {str(zone): i for i, zone in enumerate(zone_ids)}
    time_values = np.full((len(zone_ids), len(zone_ids)), np.nan, dtype=float)
    distance_values = np.full_like(time_values, np.nan)

    source = edges["source"].to_numpy(dtype=np.int64)
    target = edges["target"].to_numpy(dtype=np.int64)
    node_base = int(max(source.max(initial=0), target.max(initial=0)) + 1)
    edge_codes = source * node_base + target
    edge_lengths = edges["length_m"].to_numpy(dtype=float)
    edge_times = edges["time_min"].to_numpy(dtype=float)
    order = np.argsort(edge_codes)
    edge_codes = edge_codes[order]
    edge_lengths = edge_lengths[order]
    edge_times = edge_times[order]

    valid_route_count = 0
    batch_size = 25_000
    for start in range(0, len(od), batch_size):
        batch = od.iloc[start : start + batch_size]
        origin_nodes = batch["origin_node"].to_numpy(dtype=np.int32)
        destination_nodes = batch["destination_node"].to_numpy(dtype=np.int32)
        paths = network.shortest_paths(origin_nodes, destination_nodes, imp_name="time_min")

        valid_positions = []
        valid_paths = []
        for local_position, path in enumerate(paths):
            if path is None or len(path) < 2:
                continue
            valid_positions.append(local_position)
            valid_paths.append(np.asarray(path, dtype=np.int64))
        if not valid_paths:
            continue

        edge_counts = np.fromiter((len(path) - 1 for path in valid_paths), dtype=np.int64)
        route_sources = np.concatenate([path[:-1] for path in valid_paths])
        route_targets = np.concatenate([path[1:] for path in valid_paths])
        route_codes = route_sources * node_base + route_targets
        matches = np.searchsorted(edge_codes, route_codes)
        in_network = matches < len(edge_codes)
        in_network[in_network] &= edge_codes[matches[in_network]] == route_codes[in_network]
        if not np.all(in_network):
            raise RuntimeError("A final shortest path contains an edge outside the assignment network.")
        starts = np.concatenate(([0], np.cumsum(edge_counts)[:-1]))
        route_times = np.add.reduceat(edge_times[matches], starts)
        route_distances = np.add.reduceat(edge_lengths[matches], starts)

        selected = batch.iloc[valid_positions]
        rows = selected["origin"].astype(str).map(position).to_numpy(dtype=int)
        columns = selected["destination"].astype(str).map(position).to_numpy(dtype=int)
        usable = np.isfinite(route_times) & (route_times < NO_PATH_TIME_MIN)
        usable &= np.isfinite(route_distances)
        time_values[rows[usable], columns[usable]] = route_times[usable]
        distance_values[rows[usable], columns[usable]] = route_distances[usable]
        valid_route_count += int(usable.sum())

    return (
        pd.DataFrame(time_values, index=zone_ids, columns=zone_ids),
        pd.DataFrame(distance_values, index=zone_ids, columns=zone_ids),
        valid_route_count,
    )


def trip_assignment_baseline(
    assignment_network: dict,
    zones: pd.DataFrame,
    drive_passenger_od: pd.DataFrame,
    road_background_od: pd.DataFrame,
    *,
    drive_occupancy: float = DRIVE_OCCUPANCY,
    max_iterations: int = ASSIGNMENT_MAX_ITERATIONS,
    min_iterations: int = ASSIGNMENT_MIN_ITERATIONS,
    relative_gap_threshold: float = ASSIGNMENT_RELATIVE_GAP,
    od_threshold: float = ASSIGNMENT_OD_THRESHOLD,
    stall_iterations: int = ASSIGNMENT_STALL_ITERATIONS,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Assign road demand once, using Frank-Wolfe iterations within the assignment."""
    edges = assignment_network["edges"].copy().reset_index(drop=True)
    nodes = assignment_network["nodes"]
    bpr_columns = ["free_flow_time_min", "capacity_vph", "alpha", "beta"]
    if not np.isfinite(edges[bpr_columns].to_numpy(dtype=float)).all():
        raise ValueError("The assignment network contains non-finite BPR inputs. Rebuild it with prepare_inputs.py.")
    zone_node_map = {str(k): int(v) for k, v in assignment_network["zone_node_map"].items()}
    zone_ids = zones["grid_id"].astype(str).to_numpy()

    demand, metadata = _assignment_demand(
        drive_passenger_od,
        road_background_od,
        zones,
        drive_occupancy,
    )
    demand_values = demand.to_numpy(dtype=float)
    intrazonal = float(np.trace(demand_values))
    interzonal = float(demand_values.sum() - intrazonal)
    metadata["intrazonal_assignment_vehicle_trips"] = intrazonal
    metadata["interzonal_assignment_vehicle_trips"] = interzonal
    od = _significant_od(demand, zone_node_map, od_threshold)
    retained = float(od["demand"].sum())
    metadata["retained_assignment_trips"] = retained
    metadata["retained_interzonal_share"] = retained / max(interzonal, 1e-9)
    metadata["omitted_low_volume_interzonal_trips"] = max(interzonal - retained, 0.0)
    metadata["assigned_od_pairs"] = int(len(od))
    print(
        f"Assignment OD pairs: {len(od):,}; retained demand: {retained:,.1f} vehicles "
        f"at threshold {float(od_threshold):g}."
    )

    edge_ids = edges["edge_id"].astype(str).tolist()
    edge_lookup = {
        (int(source), int(target)): edge_id
        for source, target, edge_id in zip(edges["source"], edges["target"], edge_ids)
    }
    ta_network = Network("road")
    for row in edges[["edge_id", "source", "target", "free_flow_time_min", "capacity_vph", "alpha", "beta"]].itertuples(index=False):
        ta_network.add_edge(Edge([str(value) for value in row]))

    edges["time_min"] = edges["free_flow_time_min"]
    network = _pandana_network(nodes, edges)
    volumes, served = _all_or_nothing(network, od, edge_lookup, edge_ids)
    history = []
    best_relative_gap = np.inf
    stall_count = 0

    for iteration in range(1, int(max_iterations) + 1):
        old_volumes = volumes.copy()
        edges["time_min"] = _edge_times(edges, old_volumes)
        network = _pandana_network(nodes, edges)
        potential, served = _all_or_nothing(network, od, edge_lookup, edge_ids)
        step = float(cal_step(ta_network, old_volumes, potential))
        volumes = {
            edge_id: old_volumes[edge_id] + step * (potential[edge_id] - old_volumes[edge_id])
            for edge_id in edge_ids
        }
        absolute_change = float(cal_limit(volumes, old_volumes))
        relative_gap = absolute_change / max(sum(abs(value) for value in old_volumes.values()), 1e-9)
        if relative_gap < best_relative_gap:
            best_relative_gap = relative_gap
            stall_count = 0
        else:
            stall_count += 1
        history.append(
            {
                "iteration": iteration,
                "step": step,
                "relative_l1_gap": relative_gap,
                "served_trips": float(served),
                "stall_count": stall_count,
            }
        )
        print(
            f"Frank-Wolfe iteration {iteration}: step={step:.4f}, "
            f"relative flow change={relative_gap:.4f}."
        )
        if iteration >= int(min_iterations) and relative_gap <= float(relative_gap_threshold):
            break
        if stall_count >= max(int(stall_iterations), 1):
            print(f"Assignment stopped after {stall_count} non-improving iterations.")
            break

    edges["flow_vehicles"] = edges["edge_id"].map(volumes).fillna(0.0)
    edges["time_min"] = _edge_times(edges, volumes)
    edges["volume_capacity_ratio"] = edges["flow_vehicles"] / edges["capacity_vph"].clip(lower=1e-9)
    final_network = _pandana_network(nodes, edges)
    congested_time, congested_distance, skimmed_pairs = _selected_route_skims(
        final_network,
        od,
        edges,
        zone_ids,
    )
    retained_trips = float(metadata["retained_assignment_trips"])
    unserved_trips = max(retained_trips - float(served), 0.0)
    unskimmed_pairs = max(len(od) - int(skimmed_pairs), 0)
    if unserved_trips > 1e-6 or unskimmed_pairs:
        print(
            "Assignment connectivity: "
            f"{unskimmed_pairs:,} unreachable OD pairs and "
            f"{unserved_trips:.1f} unserved vehicle trips "
            f"({100.0 * unserved_trips / max(retained_trips, 1e-9):.3f}% of retained demand)."
        )

    metadata.update(
        {
            "served_assignment_trips": float(served),
            "unserved_assignment_trips": unserved_trips,
            "served_retained_share": float(served) / max(retained_trips, 1e-9),
            "iterations": len(history),
            "final_relative_l1_gap": history[-1]["relative_l1_gap"] if history else 0.0,
            "best_relative_l1_gap": float(best_relative_gap) if history else 0.0,
            "stall_iterations": int(stall_iterations),
            "skimmed_assignment_od_pairs": skimmed_pairs,
            "unskimmed_assignment_od_pairs": unskimmed_pairs,
            "history": history,
        }
    )
    return (
        gpd.GeoDataFrame(edges, geometry="geometry", crs=assignment_network["edges"].crs),
        congested_time,
        congested_distance,
        metadata,
    )
