"""Serve OpenStreetMap roads from the frozen Forch extract instead of Overpass.

Why: keeps all four group members on identical roads.

Usage, before the notebook builds the detailed network:
    import offline_osm
    offline_osm.install()
After install(), osmnx.graph_from_polygon cuts the requested polygon out of the
frozen graph, so the course code in transport_model_interface.py runs unchanged.
"""
from __future__ import annotations

import gzip
import json
import pickle
from functools import lru_cache
from pathlib import Path

import osmnx as ox
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
EXTRACT = ROOT / "data/transport/raw/osm/forch_2026-10-09_drive.pkl.gz"
PROVENANCE = EXTRACT.with_name("forch_2026-10-09_drive.provenance.json")


@lru_cache(maxsize=1)
def frozen_graph():
    """Load the extract once per session; it is about 100,000 nodes."""
    with gzip.open(EXTRACT, "rb") as handle:
        return pickle.load(handle)


def extract_bbox():
    west, south, east, north = json.loads(PROVENANCE.read_text())["bbox_west_south_east_north"]
    return box(west, south, east, north)


def graph_from_polygon(polygon, network_type="drive", simplify=True, retain_all=False, **_ignored):
    """Drop-in replacement for osmnx.graph_from_polygon on the frozen extract.

    The polygon must be in WGS84 and lie inside the extract, otherwise roads
    would silently go missing at the edge of the corridor.
    """
    if network_type != "drive":
        raise ValueError("The frozen extract only holds the drive network.")
    if not extract_bbox().contains(polygon):
        raise ValueError(f"Polygon {polygon.bounds} leaves the frozen OSM extract {extract_bbox().bounds}.")
    graph = ox.truncate.truncate_graph_polygon(frozen_graph(), polygon, truncate_by_edge=False)
    if not retain_all:
        graph = ox.truncate.largest_component(graph, strongly=False)
    if simplify:
        graph = ox.simplify_graph(graph)
    return graph


def install():
    """Point osmnx at the frozen extract for the rest of this Python session."""
    if not EXTRACT.is_file():
        raise FileNotFoundError(f"Frozen OSM extract missing: {EXTRACT}. Run tools/fetch_forch_osm.py on a machine with Overpass access.")
    ox.graph_from_polygon = graph_from_polygon
    print(f"[offline_osm] osmnx.graph_from_polygon now reads {EXTRACT.name}")
