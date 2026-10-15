"""Optional PT approach, cycling-route and station coverage for FSM and appraisal.

Named recipes generate reusable OD coverage from shared prepared routing inputs.
Supplementary passengers enter appraisal after mode choice. Station components
and cycling-route exposure identify the journeys benefiting from an intervention.
Run ``python code/additional/section_flows.py --help`` for the count/calibration workflow.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata
from zipfile import BadZipFile
from types import MappingProxyType

import numpy as np
import pandas as pd

# Support the documented direct script command as well as package imports.
_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import parameters as p

_ROOT = _CODE_DIR.parent
_MODES = {"PT": ("pt_walk", "pt_bike"), "CAR": ("drive",), "BIKE": ("bike",), "WALK": ("walk",)}
_DEFAULT_COVERAGE = "data/processed/section_coverage.npz"
_RECIPE_VERSION = 1
_INTERNAL_SECTION = {"active": False, "mode": "PT", "origin": {}, "destination": {},
    "both_directions": True, "crowding_enabled": False, "coverage_file": None,
    "section_override": None}
_INTERNAL_EXTERNAL = {"enabled": False, "mode": None, "additional_trips_daily": 0.0, "growth": "general"}


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, allow_nan=False).encode()).hexdigest()


def _mode(value):
    value = str(value).upper()
    value = {"DRIVE": "CAR", "BICYCLE": "BIKE", "CYCLING": "BIKE", "WALKING": "WALK"}.get(value, value)
    if value not in _MODES:
        raise ValueError("Section mode must be PT, CAR, BIKE or WALK.")
    return value


def section_config(params=None):
    """Resolve SECTION, or a direct section dictionary; generic default is off."""
    params = params or {}
    defaults = {**_INTERNAL_SECTION, **getattr(p, "SECTION_DEFAULTS", {})}
    source = params.get("SECTION", params if "active" in params or "kind" in params else getattr(p, "SECTION", {}))
    if not isinstance(source, dict):
        raise ValueError("SECTION must be a dictionary.")
    result = {**deepcopy(defaults), **deepcopy(source)}
    result["mode"] = _mode(result["mode"])
    for key in ("active", "both_directions", "crowding_enabled"):
        if not isinstance(result[key], (bool, np.bool_)):
            raise ValueError(f"SECTION {key} must be True or False.")
        result[key] = bool(result[key])
    if result["active"]:
        kind = result.get("kind", "screenline")
        if kind not in ("pt_approach", "pt_stop", "bike_route", "screenline"):
            raise ValueError("SECTION kind must be pt_approach, pt_stop, bike_route or screenline.")
        result["kind"] = kind
        expected = {"pt_approach": "PT", "pt_stop": "PT", "bike_route": "BIKE"}.get(kind)
        if expected and result["mode"] != expected:
            raise ValueError(f"SECTION {kind} requires mode={expected}.")
        for key in ("origin", "destination"):
            if result.get(key):
                result[key] = _selector(result[key], key)
        if bool(result.get("origin")) != bool(result.get("destination")):
            raise ValueError("Supply both reference origin and destination, or neither.")
        if kind in ("pt_approach", "pt_stop"):
            if not str(result.get("station", "")).strip():
                raise ValueError(f"SECTION {kind} needs a station name.")
            if kind == "pt_approach" and not str(result.get("towards", "")).strip():
                raise ValueError("SECTION pt_approach needs a towards station name.")
        if kind == "pt_stop" and result["crowding_enabled"]:
            raise ValueError("PT stop coverage measures access and transfers; disable corridor crowding for this kind.")
        if result["crowding_enabled"] and result["mode"] != "PT":
            raise ValueError("This crowding formula is calibrated for PT only; other effects need explicit valuation.")
    return result


def external_flow_config(params=None):
    """Resolve one optional cohort; all counts are person-trips per weekday."""
    params = params or {}
    defaults = {**_INTERNAL_EXTERNAL, **getattr(p, "EXTERNAL_FLOW_DEFAULTS", {})}
    source = params.get("EXTERNAL_FLOW", params if "enabled" in params else getattr(p, "EXTERNAL_FLOW", {}))
    if not isinstance(source, dict):
        raise ValueError("EXTERNAL_FLOW must be a dictionary.")
    result = {**deepcopy(defaults), **deepcopy(source)}
    section = section_config(params)
    result["mode"] = _mode(result.get("mode") or section["mode"])
    if not isinstance(result["enabled"], (bool, np.bool_)):
        raise ValueError("EXTERNAL_FLOW enabled must be True or False.")
    result["enabled"] = bool(result["enabled"])
    count = float(result["additional_trips_daily"])
    if not np.isfinite(count) or count < 0:
        raise ValueError("additional_trips_daily must be finite and nonnegative.")
    result["additional_trips_daily"] = count
    if result["growth"] not in ("general", "fixed"):
        raise ValueError("External growth must be 'general' or 'fixed'.")
    if result["enabled"] and (not section["active"] or result["mode"] != section["mode"]):
        raise ValueError("An enabled external cohort needs an active SECTION of the same mode.")
    value = result.get("value_of_time_chf_per_hour")
    if value is None:
        key = "C_TT_" + result["mode"]
        value = params.get(key, p.NOMINAL_PARAMS.get(key))
    if value is not None:
        value = float(value)
        if not np.isfinite(value) or value < 0:
            raise ValueError("External value of time must be finite and nonnegative.")
    if result["enabled"] and value is None:
        raise ValueError("Configure the selected mode's value of time in CHF/person-hour.")
    result["value_of_time_chf_per_hour"] = value
    return result


def _selector(value, name):
    if not isinstance(value, dict) or len(value) != 1 or next(iter(value), None) not in ("municipality_name", "zone_ids"):
        raise ValueError(f"Section {name} must select municipality_name or zone_ids.")
    key, values = next(iter(value.items()))
    values = values if isinstance(values, (list, tuple, set)) else [values]
    if not values or any(item is None or not str(item).strip() for item in values):
        raise ValueError(f"Section {name} cannot be empty.")
    return {key: sorted({str(item) for item in values})}


def _selected_zones(context, labels, selector):
    key, allowed = next(iter(selector.items()))
    if key == "zone_ids":
        values = labels
    else:
        zones = context.zones.set_index(context.zones["grid_id"].astype(str))
        if zones.index.has_duplicates:
            raise ValueError("Section zone metadata has duplicate grid_id values.")
        values = pd.Index(zones.reindex(labels)[key].astype(str))
    missing = set(allowed) - set(values)
    if missing:
        raise ValueError(f"Section selector is absent from model: {sorted(missing)}.")
    return values.isin(allowed)


def _path(config):
    path = Path(config.get("coverage_file") or _DEFAULT_COVERAGE)
    return path if path.is_absolute() else _ROOT / path


def _name(value):
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(re.sub(r"[_\-\u2010-\u2015]", " ", value).split())


@lru_cache(maxsize=32)
def _file_digest(path, stamp):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest_path(path):
    path = Path(path)
    stat = path.stat()
    return _file_digest(str(path.resolve()), (stat.st_size, stat.st_mtime_ns))


def _routing_path(config):
    from transport_core import config as native
    default = getattr(native, "PT_ROUTING_FILE", native.ROUTING_DIR / "pt_routing_input.pkl")
    if config["mode"] == "BIKE":
        default = native.BIKE_GRAPH_FILE
    path = Path(config.get("route_file") or default)
    return path if path.is_absolute() else _ROOT / path


def _recipe_contract(config):
    from transport_core import config as native
    recipe = {key: value for key, value in config.items() if key in (
        "kind", "mode", "station", "towards", "both_directions", "origin_station", "destination_station",
        "origin_coordinates", "destination_coordinates", "via_coordinates")}
    inputs = {"routing": _digest_path(_routing_path(config)), "skims": _digest_path(native.SKIM_PACKAGE_FILE),
              "zones": _digest_path(native.ZONES_FILE)}
    if config["kind"] == "bike_route" and any(config.get(key) for key in ("origin_station", "destination_station")):
        inputs["stations"] = _digest_path(_routing_path({"mode": "PT"}))
    if config["kind"] == "bike_route":
        inputs["cycling_speeds_kph"] = [float(native.BIKE_SPEED_CITY_KPH), float(native.BIKE_SPEED_CANTON_KPH)]
        inputs["city_speed_boundary"] = _digest_path(native.NETWORK_POLICY_CITY_FILE)
    return {"algorithm_version": _RECIPE_VERSION, "recipe": recipe, "inputs": inputs}


def _minimal_context():
    """Prepared inputs only; coverage generation does not run the transport model."""
    from types import SimpleNamespace
    from transport_core import config, zoning, travel_times
    zones = zoning.load_zones()
    labels = pd.Index(zones.grid_id.astype(str))
    times, _ = travel_times.load_travel_times(zones)
    return SimpleNamespace(zones=zones, baseline_od=pd.DataFrame(index=labels, columns=labels),
        travel_times=times, modules={"config": config})


def _ensure_recipe_coverage(config, context=None):
    contract = _recipe_contract(config)
    token = _hash(contract)
    path = _path(config) if config.get("coverage_file") else _ROOT / "cache" / "sections" / (token + ".npz")
    metadata = path.with_suffix(".json")
    if path.is_file() and metadata.is_file():
        try:
            if json.loads(metadata.read_text(encoding="utf-8")).get("contract") == contract:
                return path
        except (ValueError, OSError):
            pass
    _prepare_recipe(context or _minimal_context(), config, path, contract)
    return path


def _station_group(stops, graph, name):
    ids = stops.stop_id.astype(str)
    matched = stops.loc[stops.stop_name.map(_name).eq(_name(name)) | ids.eq(str(name))]
    if matched.empty:
        raise ValueError(f"Station {name!r} is absent from the routing input; use --list-stations.")
    parent = stops.get("parent_station", pd.Series("", index=stops.index)).fillna("").astype(str)
    parents = set(parent.loc[matched.index]) - {"", "nan", "None"}
    location = stops.get("location_type", pd.Series("", index=stops.index)).astype(str)
    parents |= set(ids.loc[matched.index[location.loc[matched.index].isin(["1", "1.0"])]] )
    if len(parents) > 1:
        raise ValueError(f"Station name {name!r} matches multiple station groups; use its parent stop ID.")
    selected = stops.loc[ids.isin(parents) | parent.isin(parents)] if parents else matched
    nodes = set(selected.stop_id.astype(str)) & set(map(str, graph))
    if not nodes:
        raise ValueError(f"Station {name!r} has no routed platforms.")
    return nodes, selected


def _ancestor_sum(values, predecessor):
    result = values.copy()
    rows = np.arange(len(predecessor))[:, None]
    parent = predecessor.copy()
    for _ in range(int(np.ceil(np.log2(max(predecessor.shape[1], 2)))) + 1):
        valid = parent >= 0
        result += np.where(valid, result[rows, np.maximum(parent, 0)], 0)
        parent = np.where(valid, parent[rows, np.maximum(parent, 0)], -9999)
        if not (parent >= 0).any():
            return result
    raise RuntimeError("Prepared route predecessor propagation did not terminate.")


def _trace_features(graph, nodes, endpoints, edge_features, station_nodes=None):
    """One deterministic shortest path per OD; bounded batches avoid a path store."""
    import networkx as nx
    from scipy.sparse.csgraph import dijkstra
    position = {node: i for i, node in enumerate(nodes)}
    size = len(nodes)
    adjacency = nx.to_scipy_sparse_array(graph, nodelist=nodes, weight="weight", format="csr", dtype=float)
    adjacency.indices = adjacency.indices.astype(np.int32)
    adjacency.indptr = adjacency.indptr.astype(np.int32)
    definitions = {}
    for name, edges in edge_features.items():
        encoded = sorted((position[u] * size + position[v], float(value)) for (u, v), value in edges.items())
        definitions[name] = (np.array([x[0] for x in encoded], dtype=np.int64), np.array([x[1] for x in encoded]))
    hub = np.array([str(node) in (station_nodes or set()) for node in nodes], dtype=bool)
    result, grouped = {}, {}
    for mode, endpoint in endpoints.items():
        shape = endpoint["origin"].shape
        result[mode] = {key: np.zeros(shape, dtype=float) for key in definitions}
        result[mode]["reachable"] = np.zeros(shape, dtype=bool)
        if station_nodes is not None:
            result[mode].update(access=(endpoint["origin"] >= 0) & hub[np.maximum(endpoint["origin"], 0)],
                                egress=(endpoint["destination"] >= 0) & hub[np.maximum(endpoint["destination"], 0)],
                                visits=np.zeros(shape, dtype=bool))
        flat = endpoint["origin"].ravel()
        unique, counts = np.unique(flat, return_counts=True)
        grouped[mode] = dict(zip(unique, np.split(np.argsort(flat, kind="stable"), np.cumsum(counts)[:-1])))
    origins = sorted({int(x) for endpoint in endpoints.values() for x in np.unique(endpoint["origin"]) if x >= 0})
    transit_keys = np.array([position[u] * size + position[v] for u, v, data in graph.edges(data=True)
                             if data.get("edge_type") == "transit"], dtype=np.int64)
    for offset in range(0, len(origins), 16):
        batch = origins[offset:offset + 16]
        distance, pred = dijkstra(adjacency, directed=True, indices=batch, return_predecessors=True)
        rows = np.arange(len(batch))[:, None]
        node_ids = np.broadcast_to(np.arange(size), pred.shape)
        encoded = pred.astype(np.int64) * size + node_ids
        totals = {}
        for name, (keys, values) in definitions.items():
            incoming = np.zeros(pred.shape)
            if len(keys):
                where = np.searchsorted(keys, encoded)
                bounded = np.minimum(where, len(keys) - 1)
                incoming = np.where((where < len(keys)) & (keys[bounded] == encoded), values[bounded], 0)
            if station_nodes is not None and name == "transfer_walk_minutes":
                transit = np.isin(encoded, transit_keys)
                used_transit = _ancestor_sum(transit.astype(float), pred) > 0
                incoming *= used_transit
                prefix = _ancestor_sum(incoming, pred)
                last = np.where(transit, node_ids, -1)
                parent = pred.copy()
                for _ in range(int(np.ceil(np.log2(max(size, 2)))) + 1):
                    last = np.where((last < 0) & (parent >= 0), last[rows, np.maximum(parent, 0)], last)
                    parent = np.where(parent >= 0, parent[rows, np.maximum(parent, 0)], -9999)
                    if not (parent >= 0).any():
                        break
                totals[name] = np.where(last >= 0, prefix[rows, np.maximum(last, 0)], 0)
            else:
                totals[name] = _ancestor_sum(incoming, pred)
        if station_nodes is not None:
            totals["visits"] = _ancestor_sum(np.broadcast_to(hub, pred.shape).astype(float), pred) > 0
        for row, origin in enumerate(batch):
            for mode, endpoint in endpoints.items():
                indices = grouped[mode].get(origin)
                if indices is None:
                    continue
                destinations = endpoint["destination"].ravel()[indices]
                valid = destinations >= 0
                indices, destinations = indices[valid], destinations[valid]
                reachable = np.isfinite(distance[row, destinations])
                result[mode]["reachable"].ravel()[indices] = reachable
                for name, values in totals.items():
                    result[mode][name].ravel()[indices] = np.where(reachable, values[row, destinations], 0)
    return result


def _prepare_recipe(context, config, target, contract):
    import pickle
    import networkx as nx
    from scipy.spatial import cKDTree
    from pyproj import Transformer
    print(f"Preparing {config['kind']} coverage from shared routing inputs; this result will be cached.", flush=True)
    graph, nodes, xy, endpoints, names, candidates, source = _route_inputs(context, config["mode"], _routing_path(config))
    positions = {node: i for i, node in enumerate(nodes)}
    labels = context.baseline_od.index.astype(str)
    kind = config["kind"]
    details, features, hub = {}, {}, None
    if kind in ("pt_approach", "pt_stop"):
        with Path(source).open("rb") as stream:
            stops = pickle.load(stream)["stops"]
        hub, _ = _station_group(stops, graph, config["station"])
        details["station_stop_ids"] = sorted(hub)
        if kind == "pt_approach":
            other, _ = _station_group(stops, graph, config["towards"])
            center = xy[[positions[node] for node in nodes if str(node) in hub]].mean(axis=0)
            toward = xy[[positions[node] for node in nodes if str(node) in other]].mean(axis=0) - center
            if np.linalg.norm(toward) < 1:
                raise ValueError("Station and towards must select distinct station locations.")
            selected = {}
            for u, v in candidates:
                inbound = str(v) in hub and str(u) not in hub
                outbound = str(u) in hub and str(v) not in hub
                if not (inbound or (config["both_directions"] and outbound)):
                    continue
                other_node = u if inbound else v
                if np.dot(xy[positions[other_node]] - center, toward) > 0:
                    selected[(u, v)] = 1.0
            if not selected:
                raise ValueError("No transit connections approach this station from the selected direction.")
            features["count"] = selected
            details["selected_edges"] = [{"origin": str(u), "destination": str(v),
                "origin_name": names.get(str(u)), "destination_name": names.get(str(v))} for u, v in selected]
            hub = None
        else:
            if not config["both_directions"]:
                raise ValueError("pt_stop counts access and egress together; use both_directions=True.")
            features["transfer_walk_minutes"] = {(u, v): float(a.get("transfer_physical_min", 0.0))
                for u, v, a in graph.edges(data=True) if a.get("edge_type") == "transfer"
                and str(u) in hub and str(v) in hub}
            if any("transfer_physical_min" not in a for u, v, a in graph.edges(data=True)
                   if a.get("edge_type") == "transfer" and str(u) in hub and str(v) in hub):
                raise ValueError("PT stop preparation requires physical transfer minutes in the shared routing input.")
    else:
        def endpoint_coordinates(side):
            coordinates = config.get(side + "_coordinates")
            station = config.get(side + "_station")
            if bool(coordinates is not None) == bool(station):
                raise ValueError(f"bike_route needs exactly one {side}_station or {side}_coordinates (LV95 metres).")
            if station:
                with _routing_path({"mode": "PT"}).open("rb") as stream:
                    data = pickle.load(stream)
                _, matched = _station_group(data["stops"], data["graph"], station)
                coordinates = Transformer.from_crs(4326, 2056, always_xy=True).transform(
                    float(matched.stop_lon.astype(float).median()), float(matched.stop_lat.astype(float).median()))
            value = np.asarray(coordinates, dtype=float)
            if value.shape != (2,) or not np.isfinite(value).all():
                raise ValueError("Cycling endpoint coordinates must be [easting, northing] in LV95 metres.")
            return value
        locations = [endpoint_coordinates("origin"), *config.get("via_coordinates", []), endpoint_coordinates("destination")]
        distances, snapped = cKDTree(xy).query(np.asarray(locations), k=1)
        route = []
        for start, end in zip(snapped[:-1], snapped[1:]):
            try:
                part = nx.shortest_path(graph, nodes[int(start)], nodes[int(end)], weight="weight")
            except nx.NetworkXNoPath as exc:
                raise ValueError("No cycling route connects these endpoints; choose connected locations.") from exc
            route.extend(part if not route else part[1:])
        route_edges = list(zip(route[:-1], route[1:]))
        if not route_edges:
            raise ValueError("Cycling route endpoints snap to the same graph node; choose distinct locations.")
        lengths = np.array([graph[u][v]["length_m"] for u, v in route_edges])
        total = float(lengths.sum())
        midpoint = route_edges[min(int(np.searchsorted(np.cumsum(lengths), total / 2)), len(route_edges) - 1)]
        route_fractions = {(u, v): graph[u][v]["length_m"] / total for u, v in route_edges}
        counter = {midpoint: 1.0}
        if config["both_directions"]:
            counter.update({(midpoint[1], midpoint[0]): 1.0} if graph.has_edge(midpoint[1], midpoint[0]) else {})
            route_fractions.update({(v, u): graph[v][u]["length_m"] / total for u, v in route_edges if graph.has_edge(v, u)})
        features.update(count=counter, saving_weight=route_fractions)
        details.update(route_length_m=total, route_time_minutes=float(sum(graph[u][v]["weight"] for u, v in route_edges)), route_nodes=list(map(str, route)),
            midpoint_link=list(map(str, midpoint)), endpoint_snap_distance_m=list(map(float, distances)))
    traced = _trace_features(graph, nodes, endpoints, features, hub)
    arrays = {"labels": np.asarray(labels, dtype=str)}
    keys, components = {}, {}
    for mode, values in traced.items():
        skim_key = "ivt_" + mode if mode.startswith("pt_") else mode
        times = _values(context.travel_times[skim_key], labels)
        valid = np.isfinite(times) & (times >= 0) & (times < _no_path_time(context)) & values["reachable"]
        if kind == "pt_stop":
            for key in ("access", "egress", "visits"):
                values[key] &= valid
            values["transfer_walk_minutes"] *= valid
            values["transfer"] = values["transfer_walk_minutes"] > 0
            values["count"] = values["access"] | values["egress"] | values["transfer"]
            values["saving_weight"] = np.zeros(valid.shape)
        else:
            values["count"] = (values["count"] > 0) & valid
            values["saving_weight"] = np.minimum(values.get("saving_weight", values["count"].astype(float)), 1.0) * valid
        keys[mode] = mode + "_count"
        arrays[keys[mode]] = values["count"]
        arrays[mode + "_saving_weight"] = values["saving_weight"]
        components[mode] = {}
        for key in ("access", "egress", "visits", "transfer", "transfer_walk_minutes"):
            if key in values:
                components[mode][key] = mode + "_" + key
                arrays[mode + "_" + key] = values[key]
    section_id = kind + "_" + _hash(contract)[:12]
    entry = {"id": section_id, "name": config.get("station", kind), "mode": config["mode"],
        "kind": kind, "masks": keys, "components": components,
        "saving_weights": {mode: mode + "_saving_weight" for mode in traced},
        "both_directions": config["both_directions"],
        "stations": [config[key] for key in ("station", "towards", "origin_station", "destination_station") if config.get(key)],
        "route_input_signature": route_input_signature(context, config["mode"]),
        "description": {"pt_approach": "Fixed PT paths approaching the station from the selected side.",
            "pt_stop": "Unique access, egress or represented interior walking-transfer users.",
            "bike_route": "Midpoint physical-link count; savings proportional to improved-route length used."}[kind], **details}
    metadata = {"schema_version": 3, "contract": contract, "default_section": section_id, "sections": [entry],
        "limitations": ["Fixed baseline shortest paths; counts are estimates, not route assignment.",
                        "Match observed mode, directions, time period and counting definition."]}
    if kind.startswith("pt_"):
        metadata["limitations"].append("Service connections are not railway geometries; same-platform changes are not identified.")
    target.parent.mkdir(parents=True, exist_ok=True)
    import tempfile
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, **arrays)
    temporary.replace(target)
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".json", mode="w", encoding="utf-8", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(target.with_suffix(".json"))


@lru_cache(maxsize=8)
def _read_coverage(path, stamp, metadata_stamp):
    with np.load(path, allow_pickle=False) as package:
        arrays = {key: package[key].copy() for key in package.files}
    catalog = json.loads(Path(path).with_suffix(".json").read_text(encoding="utf-8"))
    if catalog.get("contract"):
        labels = arrays.get("labels")
        if labels is None or labels.ndim != 1 or pd.Index(labels.astype(str)).has_duplicates:
            raise ValueError("Derived section cache has invalid OD labels.")
        sections = catalog.get("sections")
        if not isinstance(sections, list) or len(sections) != 1:
            raise ValueError("Derived section cache has invalid metadata.")
        section = sections[0]
        references = [(key, True) for key in section.get("masks", {}).values()]
        references += [(key, False) for key in section.get("saving_weights", {}).values()]
        references += [(key, name != "transfer_walk_minutes") for group in section.get("components", {}).values() for name, key in group.items()]
        if not references:
            raise ValueError("Derived section cache is missing its coverage arrays.")
        for key, boolean in references:
            array = arrays.get(key)
            if array is None or array.shape != (len(labels), len(labels)):
                raise ValueError(f"Derived section cache is missing or has malformed {key!r}.")
            if boolean and array.dtype != bool:
                raise ValueError(f"Derived coverage {key!r} must be boolean.")
            if not boolean and (not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all() or (array < 0).any()):
                raise ValueError(f"Derived coverage {key!r} must be finite nonnegative values.")
    for array in arrays.values():
        array.flags.writeable = False
    digest = _hash(catalog) if catalog.get("contract") else hashlib.sha256(
        Path(path).read_bytes() + json.dumps(catalog, sort_keys=True).encode()).hexdigest()
    return arrays, catalog, digest


def _coverage(config, context=None):
    path = _ensure_recipe_coverage(config, context) if config.get("kind") in ("pt_approach", "pt_stop", "bike_route") else _path(config)
    meta = path.with_suffix(".json")
    if not path.is_file() or not meta.is_file():
        raise FileNotFoundError(f"Prepare section coverage first: {path} and {meta}. Use code/additional/section_flows.py --prepare.")
    st, mt = path.stat(), meta.stat()
    try:
        arrays, catalog, digest = _read_coverage(str(path), (st.st_mtime_ns, st.st_size), (mt.st_mtime_ns, mt.st_size))
    except (ValueError, OSError, EOFError, KeyError, BadZipFile):
        if config.get("kind") not in ("pt_approach", "pt_stop", "bike_route"):
            raise
        _prepare_recipe(context or _minimal_context(), config, path, _recipe_contract(config))
        st, mt = path.stat(), meta.stat()
        arrays, catalog, digest = _read_coverage(str(path), (st.st_mtime_ns, st.st_size), (mt.st_mtime_ns, mt.st_size))
    override = None if config.get("kind") in ("pt_approach", "pt_stop", "bike_route") else config.get("section_override")
    candidates = [item for item in catalog["sections"] if _mode(item.get("mode", "PT")) == config["mode"]]
    if override is None:
        matches = [item for item in candidates if item["id"] == catalog["default_section"]]
    elif isinstance(override, str):
        matches = [item for item in candidates if _name(override) in {_name(alias) for alias in
            [item["id"], item.get("name", ""), *item.get("station_aliases", []),
             "-".join(item.get("stations", [])),
             *( ["-".join(reversed(item.get("stations", [])))] if item.get("both_directions", True) else [] ) ]}]
    elif isinstance(override, (list, tuple)) and len(override) == 2:
        pair = list(map(_name, override))
        matches = [item for item in candidates if
                   (sorted(map(_name, item.get("stations", []))) == sorted(pair)
                    if item.get("both_directions", True) else list(map(_name, item.get("stations", []))) == pair)]
    else:
        raise ValueError("section_override must be a registered name, station or station pair.")
    if len(matches) != 1:
        raise ValueError(f"Section {override!r} is not uniquely available for {config['mode']}. "
                         "Use --list-sections, or --prepare for a new section.")
    selected = matches[0]
    if bool(selected.get("both_directions", True)) != config["both_directions"]:
        raise ValueError("Coverage direction does not match SECTION; prepare the requested directions explicitly.")
    return arrays, selected, digest


def available_sections(coverage_file=None, config=None):
    """Show feasible registered names without running transport."""
    resolved = section_config(config)
    if coverage_file:
        path = _path({"coverage_file": coverage_file})
    elif not resolved["active"]:
        return pd.DataFrame(columns=["section_override", "mode", "station_pair", "description"])
    else:
        path = _ensure_recipe_coverage(resolved) if resolved.get("kind") in ("pt_approach", "pt_stop", "bike_route") else _path(resolved)
    catalog = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if catalog.get("contract"):
        return pd.DataFrame([{key: item.get(key) for key in ("kind", "mode", "stations", "both_directions", "description")}
            for item in catalog["sections"]])
    return pd.DataFrame([{"section_override": item["id"], "mode": item.get("mode", "PT"),
        "station_pair": item.get("stations", []), "both_directions": item.get("both_directions", True),
        "description": item.get("description", ""),
        "historical_daily_check": item.get("nominal_daily_passengers_check")}
        for item in catalog["sections"]])


def physical_signature(config=None):
    """Hash physical section choices, excluding cohort, capacity and prices."""
    config = section_config(config)
    if not config["active"]:
        return _hash({"schema": "general-section-v1", "active": False})
    _, section, digest = _coverage(config)
    return _hash({"schema": "general-section-v2", **{key: config[key] for key in
        ("active", "mode", "origin", "destination", "both_directions")},
        "coverage": digest, "section": section["id"]})


def appraisal_signature(params=None):
    """Identify economic/section settings used by cached notebook appraisals."""
    from stages import get_stages, package_appraisal_settings
    resolved = {**p.NOMINAL_PARAMS, **(params or {})}
    return _hash({"physical": physical_signature(section_config(params)), "parameters": resolved,
        "section": section_config(params), "external": external_flow_config(params),
        "stages": get_stages(resolved), **package_appraisal_settings(), "horizon": p.N_YEARS,
        "uncertainties": p.STRUCTURAL_UNCERTAINTIES})


def coverage_masks(labels, config=None):
    config = section_config(config)
    if not config["active"]:
        return {}
    labels = pd.Index(map(str, labels))
    if labels.has_duplicates:
        raise ValueError("Section coverage requires unique OD zone labels.")
    arrays, section, _ = _coverage(config)
    stored = pd.Index(arrays["labels"].astype(str))
    indexer = stored.get_indexer(labels)
    if stored.has_duplicates or (indexer < 0).any():
        raise ValueError("Section coverage does not match model zone labels; prepare it for these inputs.")
    keys = section.get("masks", {"pt_walk": section.get("walk_key"), "pt_bike": section.get("bike_key")})
    result = {}
    for mode in _MODES[config["mode"]]:
        matrix = arrays[keys[mode]]
        if matrix.dtype != bool or matrix.shape != (len(stored), len(stored)):
            raise ValueError(f"Invalid boolean coverage matrix for {mode}.")
        result[mode] = matrix[np.ix_(indexer, indexer)]
    return result


def component_masks(labels, config=None):
    """PT-stop access/egress/transfer incidence and local walking minutes."""
    config = section_config(config)
    if not config["active"] or config.get("kind") != "pt_stop":
        return {}
    arrays, section, _ = _coverage(config)
    stored = pd.Index(arrays["labels"].astype(str))
    indexer = stored.get_indexer(pd.Index(map(str, labels)))
    if (indexer < 0).any():
        raise ValueError("PT-stop coverage does not contain these OD labels.")
    return {mode: {name: arrays[key][np.ix_(indexer, indexer)] for name, key in components.items()}
            for mode, components in section["components"].items()}


def station_stop_ids(config):
    """Canonical routed platforms belonging to a configured PT station."""
    resolved = section_config(config)
    _, section, _ = _coverage(resolved)
    return tuple(section.get("station_stop_ids", []))


def section_saving_weights(labels, config=None):
    """Fraction of full section saving received by each fixed OD route."""
    config = section_config(config)
    if not config["active"]:
        return {}
    arrays, section, _ = _coverage(config)
    if not section.get("saving_weights"):
        return {mode: mask.astype(float) for mode, mask in coverage_masks(labels, config).items()}
    indexer = pd.Index(arrays["labels"].astype(str)).get_indexer(pd.Index(map(str, labels)))
    if (indexer < 0).any():
        raise ValueError("Section savings do not contain these OD labels.")
    return {mode: arrays[key][np.ix_(indexer, indexer)] for mode, key in section["saving_weights"].items()}


def intervention_masks(labels, config=None):
    """OD support needed for welfare, including partial cycling-route users."""
    config = section_config(config)
    if config.get("kind") == "bike_route" and config["active"]:
        return {mode: values > 0 for mode, values in section_saving_weights(labels, config).items()}
    return coverage_masks(labels, config)


def section_route_minutes(config=None):
    """Full cycling-route reference before e-bike adjustment, in minutes."""
    config = section_config(config)
    _, section, _ = _coverage(config)
    return float(section["route_time_minutes"])


def _bike_route_times(mode_result, config):
    from stages import get_stages
    from transport_core.travel_times import ebike_time_factor
    base = get_stages()[0]
    raw = section_route_minutes(config)
    baseline = raw * ebike_time_factor(float(base.get("ebike_share", 0)), float(base.get("EBIKE_SPEED_MULTIPLIER", 1.5)))
    current = max(raw * ebike_time_factor(float(mode_result.scenario.get("ebike_share", base.get("ebike_share", 0))),
        float(mode_result.scenario.get("EBIKE_SPEED_MULTIPLIER", base.get("EBIKE_SPEED_MULTIPLIER", 1.5)))) - _saving(mode_result.scenario), 0.0)
    return baseline, current


def _values(frame, labels):
    frame = frame.rename(index=str, columns=str)
    if frame.index.has_duplicates or frame.columns.has_duplicates:
        raise ValueError("Section skims require unique OD zone labels.")
    return frame.reindex(index=labels, columns=labels).to_numpy(dtype=float)


def route_input_signature(context, mode):
    """Identify prepared base skims used by the fixed coverage, cached per context."""
    cache = getattr(context, "_section_input_signatures", {})
    if mode in cache:
        return cache[mode]
    keys = list(_MODES[mode])
    if mode == "PT":
        keys += [f"{prefix}_{native}" for native in _MODES[mode]
                 for prefix in ("ivt", "chosen_origin_stop", "chosen_destination_stop")]
    digest = hashlib.sha256()
    for key in sorted(keys):
        frame = context.travel_times.get(key)
        if frame is None and key == "drive":
            frame = context.travel_times.get("car")
        if frame is None:
            raise ValueError(f"Coverage provenance requires prepared skim {key!r}.")
        digest.update(key.encode())
        digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
        digest.update(pd.util.hash_pandas_object(frame.columns).to_numpy().tobytes())
    cache[mode] = digest.hexdigest()
    context._section_input_signatures = cache
    return cache[mode]


def _validate_route_inputs(context, config):
    """Reject fixed masks from another prepared skim set with identical labels."""
    _, section, _ = _coverage(config, context)
    expected = section.get("route_input_signature")
    if expected is not None and expected != route_input_signature(context, config["mode"]):
        raise ValueError("Section coverage predates the prepared skims; regenerate/revalidate section coverage before running the model.")


def _saving(stage):
    value = float(stage.get("section_time_saving_min", 0.0))
    if not np.isfinite(value) or value < 0:
        raise ValueError("section_time_saving_min must be finite and nonnegative.")
    return value


def _no_path_time(context):
    return float(context.modules["config"].NO_PATH_TIME_MIN)


def apply_section_time_saving(context, travel_times, stage_spec, config=None):
    """Subtract fixed minutes once, before mode choice; stage values are cumulative."""
    config = section_config(config if config is not None else stage_spec.get("section_config"))
    saving = _saving(stage_spec)
    if saving > 0 and not config["active"]:
        raise ValueError("section_time_saving_min requires an active SECTION; remove the saving or configure a section.")
    if saving > 0 and config.get("kind") == "pt_stop":
        raise ValueError("PT-stop projects use station mobility_hubs component reductions, not section in-vehicle minutes.")
    if config["active"]:
        _coverage(config, context)
    metadata = {"section_config": config, "section_time_saving_min": saving,
                "section_signature": physical_signature(config)}
    output = dict(travel_times)
    if not config["active"] or saving == 0:
        if config["active"]:
            _validate_route_inputs(context, config)
        return output, metadata
    _validate_route_inputs(context, config)
    labels = context.baseline_od.index.astype(str)
    no_path = _no_path_time(context)
    for mode, weights in section_saving_weights(labels, config).items():
        mask = weights > 0
        key = f"ivt_{mode}" if mode.startswith("pt_") else mode
        values = _values(travel_times[key], labels).copy()
        valid = mask & np.isfinite(values) & (values >= 0) & (values < no_path)
        reduction = saving * weights[valid]
        if mode == "drive":
            physical = context.travel_times.get("drive", context.travel_times.get("car"))
            available = _values(physical, labels)
            # Perceived connector/parking constants cannot become a physical
            # time saving when the requested reduction exceeds actual driving.
            reduction = np.minimum(reduction, np.maximum(available[valid], 0.0))
        values[valid] = np.maximum(values[valid] - reduction, 0.0)
        output[key] = pd.DataFrame(values, index=labels, columns=labels)
        if mode.startswith("pt_"):
            output[mode] = output[key] + output[f"ovt_{mode}"]
    return output, metadata


def physical_car_times(context, mode_result):
    """Physical road minutes without perceived parking/connector penalties."""
    config = section_config(mode_result.scenario.get("section_config"))
    frame = context.travel_times.get("drive", context.travel_times.get("car"))
    if frame is None:
        raise ValueError("Section car appraisal requires a physical drive/car skim.")
    if not config["active"] or config["mode"] != "CAR" or _saving(mode_result.scenario) == 0:
        return frame
    # Use the same mask/minute operation on physical rather than perceived skim.
    times, _ = apply_section_time_saving(context, {"drive": frame}, mode_result.scenario, config)
    return times["drive"]


def _nominal(context):
    """One unassigned nominal baseline mode-choice call, reused for fixed weights."""
    from stages import get_stages
    baseline = get_stages()[0]
    nominal_key = _hash({key: value for key, value in baseline.items() if key not in ("section_config", "section_signature")})
    cached = getattr(context, "_section_nominal_cache", None)
    if cached is not None and cached.get("nominal_key", nominal_key) == nominal_key:
        return cached
    modules = context.modules
    times = modules["travel_times"].apply_perceived_time_policies(context.travel_times, context.zones)
    times, _ = modules["travel_times"].apply_ebike_share(times, float(baseline.get("ebike_share", 0)),
        speed_multiplier=float(baseline.get("EBIKE_SPEED_MULTIPLIER", 1.5)))
    native = modules["mode_choice_zurich"].load_mode_choice_parameters()
    native.update({key: value for key, value in baseline.items() if key in native})
    od, _ = modules["mode_choice_zurich"].mode_split_aggregated(times, context.lengths, context.baseline_od,
        walk_allowed_mask=modules["travel_times"].standalone_walk_mask(context.lengths, context.zones), parameters=native)
    cached = {"times": times, "quantities": od, "selections": {}, "nominal_key": nominal_key}
    context._section_nominal_cache = cached
    return cached


def _reference(context, mode_result, config):
    labels = context.baseline_od.index.astype(str)
    nominal = _nominal(context)
    signature = physical_signature(config)
    selection = nominal["selections"].get(signature)
    no_path = _no_path_time(context)
    if selection is None:
        pair = np.ones((len(labels), len(labels)), dtype=bool)
        if config.get("origin") and config.get("destination"):
            origin = _selected_zones(context, labels, config["origin"])
            destination = _selected_zones(context, labels, config["destination"])
            pair = origin[:, None] & destination[None, :]
            if config["both_directions"]:
                pair |= destination[:, None] & origin[None, :]
        masks = coverage_masks(labels, config)
        selection, total = {}, 0.0
        for mode in _MODES[config["mode"]]:
            key = f"ivt_{mode}" if mode.startswith("pt_") else mode
            raw_car = context.travel_times.get("drive", context.travel_times.get("car"))
            if mode == "drive" and raw_car is None:
                raise ValueError("Section car appraisal requires a physical drive/car skim.")
            t = _values(raw_car if mode == "drive" else nominal["times"][key], labels)
            if config.get("kind") == "pt_stop":
                t = np.zeros_like(t)
            q = _values(nominal["quantities"][mode], labels)
            # The representative pair must traverse the same selected section.
            valid = pair & masks[mode] & np.isfinite(q) & (q > 0) & np.isfinite(t) & (t >= 0) & (t < no_path)
            row, col = np.nonzero(valid)
            selection[mode] = (row, col, q[row, col], t[row, col])
            total += float(q[row, col].sum())
        if total <= 0:
            raise ValueError("Selected reference OD pair has no valid nominal passengers crossing this section.")
        selection = {mode: (row, col, q / total, t) for mode, (row, col, q, t) in selection.items()}
        nominal["selections"][signature] = selection
    baseline, current, delay = 0.0, 0.0, 0.0
    uncongested = getattr(mode_result, "uncongested_travel_times", None) or mode_result.travel_times
    for mode, (row, col, weights, old) in selection.items():
        key = f"ivt_{mode}" if mode.startswith("pt_") else mode
        frame = physical_car_times(context, mode_result) if mode == "drive" else uncongested[key]
        values = _values(frame, labels)[row, col] if config.get("kind") != "pt_stop" else np.zeros(len(row))
        if not (np.isfinite(values) & (values >= 0) & (values < no_path)).all():
            raise ValueError("Project skim invalid for fixed section reference cohort; do not redistribute its weights.")
        baseline += float(np.dot(weights, old))
        current += float(np.dot(weights, values))
        if mode == "drive":
            if getattr(mode_result, "uncongested_travel_times", None) is None:
                raise ValueError("External car time requires coupled assignment with uncongested skims.")
            actual = _values(mode_result.travel_times["drive"], labels)[row, col]
            free = _values(uncongested["drive"], labels)[row, col]
            if not np.isfinite(actual).all() or (actual >= no_path).any():
                raise ValueError("Assigned car skim invalid for the fixed section reference cohort.")
            delay += float(np.dot(weights, np.maximum(actual - free, 0)))
    if config.get("kind") == "bike_route":
        exposure = section_saving_weights(labels, config)
        fraction = sum(float(np.dot(weights, exposure[mode][row, col]))
            for mode, (row, col, weights, _) in selection.items())
        baseline, current = (value * fraction for value in _bike_route_times(mode_result, config))
    return baseline, current, delay


def section_reference(context, mode_result, config=None):
    """Fixed nominal OD/submode weights and reference minutes, read-only arrays.

    Each mode maps to (origin indices, destination indices, weights, minutes).
    Weights sum to one over the selected modes and OD pairs.
    """
    config = section_config(config if config is not None else mode_result.scenario.get("section_config"))
    if not config["active"]:
        return MappingProxyType({})
    _reference(context, mode_result, config)
    selection = _nominal(context)["selections"][physical_signature(config)]
    for terms in selection.values():
        for values in terms:
            values.flags.writeable = False
    return MappingProxyType(selection)


def section_welfare_times(context, mode_result, labels):
    config = section_config(mode_result.scenario.get("section_config"))
    labels = pd.Index(map(str, labels))
    result = {f"{mode}_section": np.zeros((len(labels), len(labels))) for mode in _MODES["PT"]}
    if not config["active"] or config["mode"] != "PT":
        return result
    _, reference, _ = _reference(context, mode_result, config)
    for mode, mask in coverage_masks(labels, config).items():
        values = _values(mode_result.travel_times[f"ivt_{mode}"], labels)
        valid = mask & np.isfinite(values) & (values >= 0) & (values < _no_path_time(context))
        result[f"{mode}_section"][valid] = np.minimum(values[valid], reference) / 60
    return result


def _hub_reference_components(context, mode_result, config):
    """Nominal cohort weights, valuing only components at the selected stop."""
    labels = context.baseline_od.index.astype(str)
    selected = section_reference(context, mode_result, config)
    components = component_masks(labels, config)
    from transport_core.interventions import station_transfer_walk_minutes
    transfer_minutes = station_transfer_walk_minutes(context, mode_result.travel_times, mode_result.scenario, config["station"], labels)
    totals = dict(access=0.0, egress=0.0, transfer_walk=0.0)
    for mode, (row, col, weights, _) in selected.items():
        for part in ("access", "egress"):
            values = _values(mode_result.travel_times[f"{part}_{mode}"], labels)[row, col]
            mask = components[mode][part][row, col]
            if not np.isfinite(values[mask]).all():
                raise ValueError(f"Invalid PT-stop {part} minutes.")
            totals[part] += float(np.dot(weights, np.where(mask, values, 0)))
        totals["transfer_walk"] += float(np.dot(weights, transfer_minutes[mode][row, col]))
    return {f"section_reference_{part}_minutes": value for part, value in totals.items()}


def section_metrics(context, mode_result):
    """Count covered person-trips over all retained ODs, independent of cordon."""
    config = section_config(mode_result.scenario.get("section_config"))
    result = {"section_active": config["active"], "section_mode": config["mode"],
        "section_signature": physical_signature(config), "section_modeled_trips_peak": 0.0,
        "section_modeled_person_hours_peak": 0.0, "section_reference_minutes0": 0.0,
        "section_reference_minutes": 0.0, "section_reference_delay_minutes": 0.0}
    if not config["active"]:
        return result
    _validate_route_inputs(context, config)
    labels = context.baseline_od.index.astype(str)
    old, current, delay = _reference(context, mode_result, config)
    result.update(section_reference_minutes0=old, section_reference_minutes=current, section_reference_delay_minutes=delay)
    if config.get("kind") == "pt_stop":
        result.update(_hub_reference_components(context, mode_result, config))
        components = component_masks(labels, config)
        for key, name in (("access", "access"), ("egress", "egress"), ("transfer", "transfer"), ("visits", "stop_visitors")):
            result[f"section_{name}_trips_peak"] = sum(float(_values(mode_result.od_by_mode[mode], labels)[values[key]].sum())
                for mode, values in components.items())
    if config.get("kind") == "bike_route":
        result["section_route_trips_peak"] = sum(float(_values(mode_result.od_by_mode[mode], labels)[mask].sum())
            for mode, mask in intervention_masks(labels, config).items())
    pt_hours = section_welfare_times(context, mode_result, labels) if config["mode"] == "PT" else {}
    for mode, mask in coverage_masks(labels, config).items():
        q = _values(mode_result.od_by_mode[mode], labels)
        if not np.isfinite(q[mask]).all() or (q[mask] < 0).any():
            raise ValueError("Section quantities must be finite nonnegative person-trips.")
        if mode.startswith("pt_"):
            hours = pt_hours[f"{mode}_section"]
        else:
            times = physical_car_times(context, mode_result) if mode == "drive" else mode_result.travel_times[mode]
            values = _values(times, labels)
            if not np.isfinite(values[mask & (q > 0)]).all():
                raise ValueError("Invalid section travel times.")
            hours = np.minimum(values, current) / 60
            if config.get("kind") == "bike_route":
                _, full_route = _bike_route_times(mode_result, config)
                exposure = section_saving_weights(labels, config)[mode]
                hours = np.minimum(values, exposure * full_route) / 60
        result["section_modeled_trips_peak"] += float(q[mask].sum())
        result["section_modeled_person_hours_peak"] += float(np.dot(q[mask], hours[mask]))
    return result


def calibrate_section(context, mode_result, observed_trips_daily=None, params=None):
    """A baseline diagnostic, never an automatic stage/year recalibration."""
    config = section_config(params if params is not None else mode_result.scenario.get("section_config"))
    if not config["active"]:
        raise ValueError("Activate a physical SECTION before counting it.")
    if "stage" not in mode_result.scenario or int(mode_result.scenario["stage"]) != 0:
        raise ValueError("Calibrate once against nominal Stage 0, not against a project stage.")
    assignment_metadata = getattr(mode_result, "assigned_metadata", None) or {}
    for key in ("road_freight_multiplier", "background_multiplier", "passenger_demand_multiplier"):
        if key in assignment_metadata and not np.isclose(float(assignment_metadata[key]), 1.0):
            raise ValueError("Calibration requires nominal passenger and road-background demand multipliers.")
    from stages import get_stages
    baseline = get_stages()[0]
    native = context.modules["mode_choice_zurich"].load_mode_choice_parameters()
    native.update({key: value for key, value in baseline.items() if key in native})
    required_preferences = {"ASC_PT_WALK", "ASC_PT_BIKE", "ASC_BIKE"}
    if not required_preferences.issubset(mode_result.scenario):
        raise ValueError("Calibration requires recorded nominal baseline mode preferences; rerun the baseline FSM.")
    # Native defaults need not all be duplicated in the scenario. Any explicit
    # override must match the known Stage-0 value, including time, cost, fare
    # and transfer coefficients as well as alternative-specific constants.
    for key, nominal in native.items():
        if key in mode_result.scenario and not np.isclose(
                float(mode_result.scenario[key]), float(nominal), rtol=1e-9, atol=1e-12):
            raise ValueError(f"Calibration requires nominal baseline mode-choice parameter {key}; rerun the baseline FSM.")
    for key in ("ebike_share", "EBIKE_SPEED_MULTIPLIER"):
        if not np.isclose(float(mode_result.scenario.get(key, baseline.get(key, 0))), float(baseline.get(key, 0))):
            raise ValueError("Calibration requires nominal baseline bicycle assumptions.")
    quantity = sum(_values(frame, context.baseline_od.index.astype(str)) for frame in mode_result.od_by_mode.values())
    if not np.allclose(quantity, context.baseline_od.to_numpy(float), rtol=1e-7, atol=1e-7):
        raise ValueError("Calibration requires nominal baseline OD demand, not a growth/dashboard demand scenario.")
    if _saving(mode_result.scenario) != 0:
        raise ValueError("Calibration requires zero baseline section saving.")
    original = mode_result.scenario
    mode_result.scenario = {**original, "section_config": config}
    try:
        metrics = section_metrics(context, mode_result)
    finally:
        mode_result.scenario = original
    values = {**p.NOMINAL_PARAMS, **(params or {})}
    share = float(values[config["mode"] + "_PEAK_SHARE"])
    if not 0 < share <= 1:
        raise ValueError("Peak share must lie in (0,1].")
    daily = metrics["section_modeled_trips_peak"] / share
    result = {**metrics, "modeled_trips_daily": daily, "peak_share": share,
        "units": "person-trips, total across the configured directions",
        "calibration_basis": "nominal baseline post-mode-choice; fixed prepared routes",
        "nominal_baseline_validation": "passed: stage, all explicit native mode-choice parameters, bicycle technology, every OD demand cell and supplied assignment demand multipliers",
        "observed_trips_daily": observed_trips_daily, "additional_trips_daily": None,
        "suggested_comfort_capacity_peak": None, "warning": None}
    labels = context.baseline_od.index.astype(str)
    if config.get("kind") == "pt_stop":
        components = component_masks(labels, config)
        counts = {key: 0.0 for key in ("access", "egress", "represented_transfer", "unique_endpoints", "unique_beneficiaries", "station_visitors")}
        for mode, masks in components.items():
            q = _values(mode_result.od_by_mode[mode], labels)
            selected = {"access": masks["access"], "egress": masks["egress"],
                "represented_transfer": masks["transfer"], "unique_endpoints": masks["access"] | masks["egress"],
                "unique_beneficiaries": masks["access"] | masks["egress"] | masks["transfer"], "station_visitors": masks["visits"]}
            for key, mask in selected.items():
                counts[key] += float(q[mask].sum()) / share
        result.update(component_counts_daily=counts,
            counting_basis="Unique access, egress or represented interior walking-transfer journeys; match observations to this union.",
            counting_limitations="Same-platform transfers are not identified; through visitors are reported separately and receive no hub benefit.")
    elif config.get("kind") == "bike_route":
        masks = intervention_masks(labels, config)
        users = sum(float(_values(mode_result.od_by_mode[mode], labels)[mask].sum()) for mode, mask in masks.items()) / share
        result.update(component_counts_daily={"midpoint_link": daily, "any_route_link": users},
            counting_basis="Direct cycling trips on the midpoint physical link; cycling access to PT is excluded.")
    else:
        result["counting_basis"] = "PT station approach towards the configured station; fixed inferred service paths." if config.get("kind") == "pt_approach" else "Configured fixed route coverage."
    metadata = assignment_metadata
    result["assignment_diagnostics"] = {key: value.item() if isinstance(value, np.generic) else value for key, value in metadata.items()
        if value is None or isinstance(value, (str, bool, int, float, np.integer, np.floating))}
    result["solver_basis"] = metadata.get("algorithm", "No assignment diagnostics supplied; unassigned/reference mode choice only")
    if observed_trips_daily is not None:
        observed = float(observed_trips_daily)
        if not np.isfinite(observed) or observed < 0:
            raise ValueError("Observed person-trips/day must be finite and nonnegative.")
        result.update(observed_trips_daily=observed, additional_trips_daily=max(observed - daily, 0),
            suggested_comfort_capacity_peak=observed * share if config["crowding_enabled"] else None)
        if observed < daily:
            result["warning"] = "Modeled count exceeds observation: no negative external cohort; investigate boundary/year/mode definitions."
    return result


def _route_inputs(context, mode, route_file=None):
    """Read existing routing inputs; never download/reprocess GTFS or OSM."""
    import pickle
    import networkx as nx
    from scipy.spatial import cKDTree
    labels = context.baseline_od.index.astype(str)
    if mode == "PT":
        path = Path(route_file or _routing_path({"mode": "PT"}))
        if not path.is_file():
            raise FileNotFoundError("New PT sections require a prepared route-only graph (--route-file). "
                                    "Scalar PT skims cannot recover intermediate routes.")
        with path.open("rb") as handle:
            data = pickle.load(handle)
        graph, stops = data["graph"], data["stops"]
        nodes = list(graph)
        positions = {str(node): i for i, node in enumerate(nodes)}
        endpoints = {}
        for native in _MODES[mode]:
            endpoint = {}
            for side in ("origin", "destination"):
                frame = context.travel_times[f"chosen_{side}_stop_{native}"].rename(index=str, columns=str)
                strings = frame.reindex(index=labels, columns=labels).to_numpy(dtype=str)
                endpoint[side] = pd.Series(strings.ravel()).map(positions).fillna(-1).to_numpy(dtype=np.int32).reshape(strings.shape)
            endpoints[native] = endpoint
        indexed = stops.set_index(stops.stop_id.astype(str)).reindex(list(map(str, nodes)))
        from pyproj import Transformer
        transformer = Transformer.from_crs(4326, 2056, always_xy=True)
        x, y = transformer.transform(indexed.stop_lon.to_numpy(float), indexed.stop_lat.to_numpy(float))
        coordinates = np.column_stack((x, y))
        names = {str(node): str(name) for node, name in zip(nodes, indexed.stop_name)}
        # Service edges can skip intermediate stations, so geometric screenlines
        # inspect all transit edges, not only services stopping at the override.
        candidates = [(u, v) for u, v, attributes in graph.edges(data=True)
                      if attributes.get("edge_type") == "transit"]
        return graph, nodes, coordinates, endpoints, names, candidates, str(path)
    native = _MODES[mode][0]
    import importlib
    network = context.modules.get("network") or importlib.import_module("transport_core.network")
    config = context.modules["config"]
    path = Path(route_file or getattr(config, {"CAR": "ROAD_GRAPH_FILE", "BIKE": "BIKE_GRAPH_FILE", "WALK": "PT_ACCESS_WALK_GRAPH_FILE"}[mode]))
    if not path.is_file():
        raise FileNotFoundError(f"Prepared {mode} routing graph is missing: {path}")
    with path.open("rb") as handle:
        original = pickle.load(handle)
    originals = list(original)
    coordinates = np.array([[original.nodes[node]["x"], original.nodes[node]["y"]] for node in originals], dtype=float)
    if mode == "WALK":
        # The all-canton walk skim uses the complete walk graph, bounded to5km;
        # city-only pairs are separately overridden below by their city graph.
        edges = []
        for u, v, attributes in original.edges(data=True):
            length = float(attributes.get("length", np.nan))
            if np.isfinite(length) and length > 0:
                edges.append((u, v, length / 1000 / float(config.WALK_SPEED_KPH) * 60))
                edges.append((v, u, length / 1000 / float(config.WALK_SPEED_KPH) * 60))
        graph = nx.DiGraph()
        graph.add_nodes_from(originals)
        for u, v, weight in edges:
            if not graph.has_edge(u, v) or graph[u][v]["weight"] > weight:
                graph.add_edge(u, v, weight=weight)
    else:
        edges = network._direct_mode_edges(original, native, network._network_policy_city_polygon())
        graph = nx.DiGraph()
        graph.add_nodes_from(originals)
        for edge in edges.itertuples(index=False):
            graph.add_edge(originals[edge.source], originals[edge.target], weight=float(edge.time_min), length_m=float(edge.length_m))
    zones = context.zones.set_index(context.zones.grid_id.astype(str)).reindex(labels)
    xy = np.column_stack((zones.centroid.x.to_numpy(float), zones.centroid.y.to_numpy(float)))
    _, snapped = cKDTree(coordinates).query(xy, k=1)
    origin = np.repeat(np.asarray(snapped)[:, None], len(labels), axis=1)
    destination = np.repeat(np.asarray(snapped)[None, :], len(labels), axis=0)
    endpoints = {native: {"origin": origin, "destination": destination}}
    return graph, originals, coordinates, endpoints, {}, list(graph.edges), str(path)


def available_stations(route_file=None):
    """Names accepted by new PT section preparation (registered aliases separate)."""
    import pickle
    path = Path(route_file or _routing_path({"mode": "PT"}))
    with path.open("rb") as handle:
        stops = pickle.load(handle)["stops"]
    parents = stops.get("parent_station", pd.Series("", index=stops.index)).fillna("").astype(str)
    identifiers = parents.where(~parents.isin(["", "nan", "None"]), stops.stop_id.astype(str))
    return pd.DataFrame({"station": stops.stop_name.astype(str), "station_id": identifiers}).drop_duplicates().sort_values(["station", "station_id"]).reset_index(drop=True)


def prepare_section_coverage(context, config, output=None, *, section_id=None, route_file=None,
                             station_override=None, half_width_m=None):
    """Generate disposable OD coverage from the named project recipe."""
    config = section_config(config)
    if not config["active"]:
        raise ValueError("Enable SECTION to prepare coverage.")
    if config.get("kind") not in ("pt_approach", "pt_stop", "bike_route"):
        raise ValueError("New coverage needs kind=pt_approach, pt_stop or bike_route; existing explicit coverage files can still be read.")
    if station_override is not None:
        values = [station_override] if isinstance(station_override, str) else list(station_override)
        if config["kind"] == "pt_approach" and len(values) == 2:
            config.update(station=values[1], towards=values[0])
        elif config["kind"] == "pt_stop" and len(values) == 1:
            config["station"] = values[0]
        else:
            raise ValueError("Use station/towards for a PT approach, station for a stop, or explicit cycling endpoints.")
    if route_file is not None:
        config["route_file"] = str(route_file)
    if output is not None:
        config["coverage_file"] = str(Path(output).resolve())
    contract = _recipe_contract(config)
    target = _path(config) if config.get("coverage_file") else _ROOT / "cache/sections" / (_hash(contract) + ".npz")
    _prepare_recipe(context, config, target, contract)
    return {**config, "coverage_file": str(target.resolve())}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list-sections", action="store_true", help="List registered station/section overrides.")
    parser.add_argument("--list-stations", action="store_true", help="List station names in the optional prepared PT route graph.")
    parser.add_argument("--prepare", action="store_true", help="Regenerate coverage from the selected recipe and prepared routing inputs.")
    parser.add_argument("--count", action="store_true", help="Run nominal native FSM/MSA and estimate a calibration residual.")
    parser.add_argument("--mode", choices=list(_MODES))
    parser.add_argument("--kind", choices=["pt_approach", "pt_stop", "bike_route"])
    parser.add_argument("--origin", help="Origin municipality name (or use --origin-zones).")
    parser.add_argument("--destination", help="Destination municipality name.")
    parser.add_argument("--origin-zones", nargs="+")
    parser.add_argument("--destination-zones", nargs="+")
    parser.add_argument("--one-direction", action="store_true")
    parser.add_argument("--section", help="Registered section name for counting; --list-sections shows choices.")
    parser.add_argument("--station", help="PT station name or parent stop ID.")
    parser.add_argument("--towards", help="Station indicating the side of a PT approach.")
    parser.add_argument("--origin-station", help="Cycling route start, located using a station name.")
    parser.add_argument("--destination-station", help="Cycling route end, located using a station name.")
    parser.add_argument("--origin-coordinates", type=float, nargs=2, metavar=("EASTING", "NORTHING"))
    parser.add_argument("--destination-coordinates", type=float, nargs=2, metavar=("EASTING", "NORTHING"))
    parser.add_argument("--coverage-file", help="Existing imported boolean OD coverage NPZ and accompanying JSON.")
    parser.add_argument("--route-file", help="Existing prepared mode route graph; never a GTFS feed.")
    parser.add_argument("--observed-daily", type=float, help="Observed existing person-trips/day across the same section/directions.")
    parser.add_argument("--iterations", type=int, help="MSA maximum iterations; defaults to ASSIGNMENT_SETTINGS.")
    parser.add_argument("--fixed-iterations", action="store_true", help="Use the same minimum and maximum MSA iterations for reproducible calibration.")
    parser.add_argument("--output", type=Path, default=Path("results/section_calibration"), help="Directory for the nominal count report.")
    args = parser.parse_args(argv)
    if args.list_sections:
        print(available_sections(args.coverage_file).to_string(index=False))
    if args.list_stations:
        print(available_stations(args.route_file).to_string(index=False))
    if not (args.prepare or args.count):
        if not (args.list_sections or args.list_stations):
            parser.print_help()
        return
    config = section_config()
    config["active"] = True
    if args.kind:
        config.update(kind=args.kind, mode="BIKE" if args.kind == "bike_route" else "PT", origin={}, destination={},
                      coverage_file=None, section_override=None, crowding_enabled=False)
    if args.mode:
        config["mode"] = args.mode
        config["crowding_enabled"] = bool(args.mode == "PT" and config["crowding_enabled"])
    for key in ("origin", "destination"):
        zones, municipality = getattr(args, key + "_zones"), getattr(args, key)
        if zones and municipality:
            parser.error(f"Choose --{key} or --{key}-zones, not both.")
        if zones or municipality:
            config[key] = {"zone_ids": zones} if zones else {"municipality_name": municipality}
    if args.coverage_file:
        config["coverage_file"] = args.coverage_file
    if args.section:
        config["kind"] = "screenline"
        config["section_override"] = args.section
    for key in ("station", "towards", "origin_station", "destination_station", "origin_coordinates", "destination_coordinates"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.route_file:
        config["route_file"] = args.route_file
    if args.one_direction:
        config["both_directions"] = False
    from transport_model_interface import load_transport_context, run_simulation, get_zone_ids_for_municipalities, ASSIGNMENT_SETTINGS
    from stages import get_stages
    context = load_transport_context(_ROOT) if args.count else _minimal_context()
    if args.prepare:
        config = prepare_section_coverage(context, config, route_file=args.route_file)
        print(f"Prepared coverage: {config['coverage_file']}")
    if args.count:
        specs = get_stages({"SECTION": config, "EXTERNAL_FLOW": {"enabled": False}})
        specs[0]["section_config"] = config
        specs[0]["section_time_saving_min"] = 0
        settings = dict(ASSIGNMENT_SETTINGS)
        try:
            ASSIGNMENT_SETTINGS.update(method="MSA", modal_feedback=True)
            if args.iterations is not None:
                if args.iterations <= 0:
                    parser.error("--iterations must be a positive integer.")
                ASSIGNMENT_SETTINGS["max_iterations"] = args.iterations
            if args.fixed_iterations:
                ASSIGNMENT_SETTINGS["min_iterations"] = ASSIGNMENT_SETTINGS["max_iterations"]
            result, _ = run_simulation(context, stage=0, stage_specs=specs,
                corridor_zone_ids=get_zone_ids_for_municipalities(context.zones, p.CORRIDOR_MUNICIPALITIES),
                corridor_municipalities=list(p.CORRIDOR_MUNICIPALITIES))
        finally:
            ASSIGNMENT_SETTINGS.clear()
            ASSIGNMENT_SETTINGS.update(settings)
        report = calibrate_section(context, result, args.observed_daily, {"SECTION": config})
        report["solver_settings"] = {**settings, "method": "MSA", "modal_feedback": True,
            "max_iterations": args.iterations or settings.get("max_iterations", 8),
            "min_iterations": (args.iterations or settings.get("max_iterations", 8)) if args.fixed_iterations else settings.get("min_iterations", 2)}
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "calibration.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
