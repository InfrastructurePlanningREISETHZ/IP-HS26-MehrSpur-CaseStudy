"""Helpers for the Forch version of Notebook 02, Part 1 (Sheet 2 §1).

The course notebook shows most §1 results inside clickable dashboards. These
functions produce the same results as plain tables and PNG figures, so the
notebook can run top to bottom without anyone clicking, and every number in
the report comes from a saved file.
"""
from __future__ import annotations

import time
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

ROOT = Path(__file__).resolve().parents[1]
S1 = ROOT / "deliverables/phase2/s1"
FIG = S1 / "figures"
TAB = S1 / "tables"
MAJOR_ROADS = ["motorway", "trunk", "primary", "secondary"]
REGION_COLOURS = {
    "Zürich centre": "#9ecae1", "Zürich east": "#4292c6", "Zollikerberg": "#fdae6b",
    "Zumikon": "#e6550d", "Forch hub area": "#a50f15", "Rest of Maur": "#74c476",
    "Rest of Egg": "#31a354", "Rest of Küsnacht": "#bcbddc", "Fällanden and Dübendorf": "#969696",
}


def d1_zone_ids() -> list[str]:
    """The v1 corridor D1 as built by tools/build_forch_corridor.py (kept reproducible after the v2 adoption)."""
    t = pd.read_csv(TAB / "corridor_zones_D1.csv", dtype={"grid_id": str})
    return t.loc[t["in_D1"], "grid_id"].tolist()


def save_table(df: pd.DataFrame, name: str, index: bool = False) -> Path:
    """Write a result table as CSV under deliverables/phase2/s1/tables."""
    TAB.mkdir(parents=True, exist_ok=True)
    path = TAB / f"{name}.csv"
    df.to_csv(path, index=index)
    return path


def save_figure(fig, name: str) -> Path:
    """Write a figure as a 200 dpi PNG under deliverables/phase2/s1/figures."""
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return path


def zone_regions(zones: pd.DataFrame, project_zones: dict) -> pd.Series:
    """Map every corridor zone ID to its D7 reporting region (the PROJECT_ZONES group)."""
    lookup = {z: region for region, ids in project_zones.items() for z in ids}
    return zones["grid_id"].astype(str).map(lookup)


# ---------------------------------------------------------------------------
# §1.3 Gates
# ---------------------------------------------------------------------------

def major_boundary_crossings(ctx, corridor, cluster_m: float = 400.0) -> gpd.GeoDataFrame:
    """Find where motorway, trunk, primary and secondary roads cross the cordon.

    A crossing is a road link with one end inside the cordon polygon and one
    outside. Links of one physical road (both carriageways, slip roads) lie
    close together, so links within cluster_m are merged into one crossing.
    """
    net = ctx.assignment_network
    edges, nodes = net["edges"], net["nodes"]
    inside = dict(zip(nodes["node_id"].astype(int), nodes.geometry.intersects(corridor.polygon)))
    src_in = edges["source"].map(inside).fillna(False).astype(bool)
    tgt_in = edges["target"].map(inside).fillna(False).astype(bool)
    road_class = edges["highway"].astype(str).str.replace("_link", "", regex=False)
    cross = edges[(src_in != tgt_in) & road_class.isin(MAJOR_ROADS)].copy()
    cross["road_class"] = road_class[cross.index]
    inner = np.where(src_in[cross.index], cross["source"], cross["target"]).astype(int)
    xy = dict(zip(nodes["node_id"].astype(int), zip(nodes.geometry.x, nodes.geometry.y)))
    cross["x"] = [xy[n][0] for n in inner]
    cross["y"] = [xy[n][1] for n in inner]
    cross["crossing"] = fcluster(linkage(cross[["x", "y"]].to_numpy(), "single"), cluster_m, "distance")
    # The best road class at a crossing decides how important it is.
    rank = {c: i for i, c in enumerate(MAJOR_ROADS)}
    out = cross.groupby("crossing").agg(
        x=("x", "mean"), y=("y", "mean"), links=("x", "size"),
        road_class=("road_class", lambda s: min(s, key=rank.get)),
        capacity_vph=("capacity_vph", "max"),
    ).reset_index()
    return gpd.GeoDataFrame(out, geometry=gpd.points_from_xy(out["x"], out["y"]), crs=ctx.zones.crs)


def gate_distance_to_crossings(corridor, crossings: gpd.GeoDataFrame) -> np.ndarray:
    """Distance in metres from each major crossing to the nearest selected gate."""
    g = np.c_[corridor.gates.geometry.x, corridor.gates.geometry.y]
    return np.hypot(crossings["x"].to_numpy()[:, None] - g[:, 0],
                    crossings["y"].to_numpy()[:, None] - g[:, 1]).min(axis=1)


def plot_model_loop():
    """Diagram of the calculation chain: fixed demand, coupled mode choice and assignment loop."""
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.set_axis_off()
    boxes = {
        "npvm": (0.02, 0.55, "NPVM OD matrix\n1,223 zones, 17-18 h\n(steps 1-2, fixed)"),
        "skims": (0.02, 0.10, "Skims: time, distance,\nPT waiting, access, transfers"),
        "logit": (0.30, 0.35, "Step 3: mode choice\n5-option logit\ncar, PT-walk, PT-bike,\nbike, walk"),
        "gates": (0.55, 0.62, "Car OD collapsed to\nzones + cordon gates\n(+5 % pass-through,\n+ fixed background)"),
        "msa": (0.55, 0.08, "Step 4: assignment\nAoN shortest paths,\nMSA step 1/k,\nBPR t = t0[1+0.15(v/c)^4]"),
        "out": (0.80, 0.35, "Indicators\nshares and PKM (step 3);\nflows, delay (step 4);\ntravel time, CO2"),
    }
    for x, y, text in boxes.values():
        ax.text(x + 0.085, y + 0.13, text, ha="center", va="center", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.5", fc="#eef3f8", ec="#4a6b8a"))
    arrows = [((0.19, 0.68), (0.30, 0.52)), ((0.19, 0.22), (0.30, 0.42)), ((0.47, 0.52), (0.55, 0.72)),
              ((0.64, 0.62), (0.64, 0.38)), ((0.55, 0.20), (0.47, 0.40)), ((0.73, 0.22), (0.80, 0.42))]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2))
    ax.text(0.47, 0.16, "congested car times\nfed back to the skim", fontsize=8, color="#a50f15",
            ha="center", transform=ax.transAxes)
    ax.text(0.80, 0.08, "loop until road relative gap <= 2 %\n(min 8, max 64 iterations, checked every 4)",
            fontsize=8, ha="left", transform=ax.transAxes)
    ax.set_title("Calculation chain of the course model (Sheet 2 §1.1); mode choice and assignment are coupled")
    return fig


def run_gate_tests(tmi, ctx, mode_result, configs, crossings, project_name="Forch",
                   assignment_options=None, covered_m: float = 400.0):
    """Build one corridor per gate configuration and run the coupled Stage-0 assignment on each.

    Returns the summary table and the corridor contexts. Retained demand alone
    cannot tell configurations apart (internal and boundary demand is kept by
    construction), so the table adds the Stage-0 road delay and whether every
    major road crossing has a gate within covered_m.
    """
    options = assignment_options or tmi.resolve_assignment_settings()
    rows, contexts, distances = [], {}, {}
    total = ctx.baseline_od.to_numpy(dtype=float).sum()
    for max_gates, separation in configs:
        corridor = tmi.build_corridor(ctx, project_name, max_gates=max_gates, gate_separation_m=separation)
        reduced, _ = tmi.collapse_od_to_gates(ctx.baseline_od, ctx, corridor)
        totals = tmi.gate_totals(reduced, corridor)
        started = time.perf_counter()
        result = tmi.run_coupled_corridor_assignment(
            ctx, corridor, mode_result, **tmi.assignment_solver_kwargs(options))
        d = gate_distance_to_crossings(corridor, crossings)
        diag = result.diagnostics
        links = result.links
        link_delay = links["flow_vehicles"] * links["delay_min"] / 60.0
        # Delay on links outside the city of Zürich: the part that concerns the Forch side of the corridor.
        city = ctx.zones.loc[ctx.zones["municipality_name"] == "Zürich"].geometry.union_all()
        outside_city = ~gpd.GeoSeries(links.geometry, crs=ctx.zones.crs).centroid.within(city)
        rows.append({
            "max_gates": max_gates, "separation_m": separation,
            "candidate_nodes": corridor.metadata["candidate_boundary_nodes"],
            "selected_gates": corridor.metadata["selected_gates"],
            "entry_gates": corridor.metadata["entry_gates"],
            "exit_gates": corridor.metadata["exit_gates"],
            "retained_trips_pct_of_canton": 100.0 * reduced.to_numpy(dtype=float).sum() / total,
            "largest_gate_share": totals["cordon_total"].max() / max(totals["cordon_total"].sum(), 1e-9),
            "major_crossings": len(crossings),
            "major_crossings_gated": int((d <= covered_m).sum()),
            "stage0_road_delay_veh_h": diag["total_delay_hours"],
            "stage0_delay_outside_city_veh_h": float(link_delay[outside_city.to_numpy()].sum()),
            "max_vc": float(links["volume_capacity_ratio"].max()),
            "stage0_vkt": diag["total_vkt"],
            "stage0_avg_speed_kph": diag["avg_speed_kph"],
            "final_road_gap": diag["final_road_relative_gap_feasible_averaged_od"],
            "iterations": diag["iterations"],
            "runtime_s": time.perf_counter() - started,
        })
        contexts[(max_gates, separation)] = corridor
        distances[(max_gates, separation)] = d
    table = pd.DataFrame(rows).sort_values("selected_gates").reset_index(drop=True)
    return table, contexts, distances


def choose_gate_configuration(table: pd.DataFrame, tolerance: float = 0.05) -> pd.Series:
    """Apply decision D3.

    A major crossing counts as gateable when the unrestricted configuration
    (all eligible candidates) has a gate within reach of it; crossings that no
    setting can gate are reported separately rather than failing every option.
    Keep configurations that gate every gateable crossing. Among them, take the
    smallest whose Stage-0 road delay is within `tolerance` of the next larger
    configuration in the table.
    """
    t = table.sort_values(["selected_gates", "max_gates"]).reset_index(drop=True)
    gateable = t["major_crossings_gated"].max()
    t["full_major_coverage"] = t["major_crossings_gated"] == gateable
    # Settings that select the same gates give the same result; compare each with the next larger distinct set.
    distinct = t.drop_duplicates("selected_gates").set_index("selected_gates")["stage0_road_delay_veh_h"]
    next_larger = distinct.shift(-1).reindex(t["selected_gates"]).to_numpy()
    t["delay_change_to_next_larger"] = (t["stage0_road_delay_veh_h"] - next_larger) / next_larger
    ok = t["full_major_coverage"] & (t["delay_change_to_next_larger"].abs() <= tolerance)
    # The largest configuration has no larger neighbour; it is the fallback.
    choice = t[ok].iloc[0] if ok.any() else t.iloc[-1]
    t["chosen"] = t.index == choice.name
    return t


# ---------------------------------------------------------------------------
# Maps
# ---------------------------------------------------------------------------

def _road_layer(ax, edges, crs):
    """Draw roads thicker for higher classes so the main axes stand out."""
    widths = {"motorway": 2.2, "trunk": 1.8, "primary": 1.4, "secondary": 1.0, "tertiary": 0.6}
    e = gpd.GeoDataFrame(edges, geometry="geometry", crs=crs)
    cls = e["highway"].astype(str).str.replace("_link", "", regex=False)
    for road_class, w in widths.items():
        part = e[cls == road_class]
        if len(part):
            part.plot(ax=ax, color="#555555", linewidth=w, zorder=2)
    e[~cls.isin(widths)].plot(ax=ax, color="#bbbbbb", linewidth=0.25, zorder=1)


def plot_corridor_map(ctx, corridor, project_zones, stations=None, title="Forch corridor (V-b: D1 plus Fällanden and Dübendorf)"):
    """Corridor zones coloured by reporting region, cordon, roads, gates and Forchbahn stations."""
    zones = ctx.zones.copy()
    zones["region"] = zone_regions(zones, project_zones)
    fig, ax = plt.subplots(figsize=(10, 9))
    xmin, ymin, xmax, ymax = corridor.polygon.buffer(1500).bounds
    context_zones = zones.cx[xmin:xmax, ymin:ymax]
    context_zones.boundary.plot(ax=ax, color="#dddddd", linewidth=0.4, zorder=0)
    for region, colour in REGION_COLOURS.items():
        part = zones[zones["region"] == region]
        if len(part):
            part.plot(ax=ax, color=colour, alpha=0.55, edgecolor="white", linewidth=0.3, zorder=1, label=region)
    _road_layer(ax, corridor.edges, zones.crs)
    gpd.GeoSeries([corridor.polygon], crs=zones.crs).boundary.plot(ax=ax, color="black", linewidth=1.0,
                                                                   linestyle="--", zorder=3)
    corridor.gates.plot(ax=ax, color="crimson", markersize=28, marker="^", zorder=5)
    if stations is not None:
        fb = stations[stations["forchbahn"]]
        ax.scatter(fb["x"], fb["y"], s=14, color="navy", zorder=6)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.55) for c in REGION_COLOURS.values()]
    labels = list(REGION_COLOURS)
    handles += [plt.Line2D([], [], color="black", linestyle="--"),
                plt.Line2D([], [], color="crimson", marker="^", linestyle=""),
                plt.Line2D([], [], color="navy", marker="o", linestyle="")]
    labels += ["Cordon (zones + 500 m buffer)", f"Gates ({len(corridor.gates)})", "Forchbahn stops (GTFS 2026)"]
    ax.legend(handles, labels, loc="lower left", fontsize=8, frameon=True)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(title)
    ax.set_xlabel("E (m, EPSG:2056)")
    ax.set_ylabel("N (m, EPSG:2056)")
    ax.set_aspect("equal")
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.grid(False)
    return fig


def plot_gate_configurations(ctx, contexts: dict, crossings, chosen_key):
    """Small multiples: gates of each tested configuration against the major road crossings."""
    keys = sorted(contexts, key=lambda k: len(contexts[k].gates))
    ncol = 3
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.6 * nrow))
    for ax, key in zip(np.ravel(axes), keys):
        corridor = contexts[key]
        gpd.GeoSeries([corridor.polygon], crs=ctx.zones.crs).boundary.plot(ax=ax, color="black", linewidth=0.7)
        crossings.plot(ax=ax, color="none", edgecolor="orange", markersize=90, linewidth=1.5)
        corridor.gates.plot(ax=ax, color="crimson", markersize=12, marker="^")
        flag = "  (chosen)" if key == chosen_key else ""
        ax.set_title(f"max {key[0]}, sep. {key[1]:.0f} m: {len(corridor.gates)} gates{flag}", fontsize=9)
        ax.set_axis_off()
        ax.set_aspect("equal")
    for ax in np.ravel(axes)[len(keys):]:
        ax.set_axis_off()
    fig.suptitle("Gate configurations tested (red triangles) against major road crossings of the cordon (orange rings)")
    return fig


# ---------------------------------------------------------------------------
# §1.4 Demand by reporting region
# ---------------------------------------------------------------------------

EXTERNAL_SECTORS = {  # bearing from Forch station, degrees clockwise from north
    "Ext. N: Glattal, Dübendorf, Winterthur": (315, 45),
    "Ext. E: Uster, Oberland": (45, 135),
    "Ext. S: Meilen, lake right bank, Rapperswil": (135, 225),
    "Ext. W: left lake bank, Limmattal, Sihltal": (225, 315),
}
REGION_ORDER = [
    "Zürich centre", "Zürich east", "Zollikerberg", "Zumikon", "Forch hub area", "Rest of Maur",
    "Rest of Egg", "Rest of Küsnacht", "Fällanden and Dübendorf", "Buffer: Zürich city", "Buffer: suburbs",
    "Ext. Zürich city (rest)", *EXTERNAL_SECTORS,
]
INTERNAL_REGIONS = REGION_ORDER[:11]


def forch_station_xy(stations: pd.DataFrame) -> tuple[float, float]:
    row = stations[stations["name"] == "Forch"].iloc[0]
    return float(row["x"]), float(row["y"])


def reporting_regions(ctx, corridor, project_zones: dict, stations: pd.DataFrame) -> pd.Series:
    """Label all 1,223 zones with a reporting region (decision D7).

    Configured corridor zones keep their PROJECT_ZONES group. Zones that are
    internal only because of the 500 m buffer are 'Buffer: ...'. External zones
    are Zürich city (rest) or one of four compass sectors seen from Forch
    station, so that 'Forch with the Glattal' is the northern sector.
    """
    zones = ctx.zones
    ids = zones["grid_id"].astype(str)
    region = zone_regions(zones, project_zones)
    internal = ids.isin(set(map(str, corridor.zone_ids)))
    city = zones["municipality_name"].eq("Zürich")
    region = region.where(region.notna() | ~internal,
                          np.where(city, "Buffer: Zürich city", "Buffer: suburbs"))
    fx, fy = forch_station_xy(stations)
    cent = gpd.GeoSeries(zones["centroid"], crs=zones.crs)
    bearing = (np.degrees(np.arctan2(cent.x - fx, cent.y - fy)) + 360) % 360
    sector = pd.Series(index=zones.index, dtype=object)
    for name, (lo, hi) in EXTERNAL_SECTORS.items():
        inside = (bearing >= lo) & (bearing < hi) if lo < hi else (bearing >= lo) | (bearing < hi)
        sector[inside] = name
    external = region.isna()
    region[external & city] = "Ext. Zürich city (rest)"
    region[external & ~city] = sector[external & ~city]
    return pd.Series(region.to_numpy(), index=ids, name="region")


def region_od(od: pd.DataFrame, regions: pd.Series) -> pd.DataFrame:
    """Aggregate a zone OD matrix to reporting regions, rows = origins."""
    m = od.copy()
    m.index, m.columns = m.index.astype(str), m.columns.astype(str)
    lab = regions.reindex(m.index)
    out = m.groupby(lab.to_numpy()).sum().T.groupby(lab.reindex(m.columns).to_numpy()).sum().T
    order = [r for r in REGION_ORDER if r in out.index]
    return out.reindex(index=order, columns=order, fill_value=0.0)


def corridor_relevant_od(region_matrix: pd.DataFrame) -> pd.DataFrame:
    """Keep rows and columns of internal regions plus external groups; external-external cells set to 0."""
    m = region_matrix.copy()
    ext = [r for r in m.index if r not in INTERNAL_REGIONS]
    m.loc[ext, ext] = 0.0
    return m


def demand_categories_by_region(region_matrix: pd.DataFrame) -> pd.DataFrame:
    """Internal, outbound and inbound peak-hour trips for every internal region."""
    internal = [r for r in INTERNAL_REGIONS if r in region_matrix.index]
    external = [r for r in region_matrix.index if r not in INTERNAL_REGIONS]
    rows = []
    for r in internal:
        rows.append({
            "region": r,
            "internal_from_region": region_matrix.loc[r, internal].sum(),
            "internal_to_region": region_matrix.loc[internal, r].sum(),
            "outbound_from_region": region_matrix.loc[r, external].sum(),
            "inbound_to_region": region_matrix.loc[external, r].sum(),
            "intra_region": region_matrix.loc[r, r],
        })
    t = pd.DataFrame(rows)
    t["total_trip_ends"] = t[["internal_from_region", "internal_to_region",
                              "outbound_from_region", "inbound_to_region"]].sum(axis=1) - t["intra_region"]
    return t


def key_relations(region_matrix: pd.DataFrame, pairs, n_largest: int = 10) -> pd.DataFrame:
    """Peak-hour trips for the D7 key relations in both directions, plus the largest relations between regions."""
    rows = []
    for a, b in pairs:
        for o, d in ((a, b), (b, a)):
            rows.append({"set": "D7 key relation", "origin": o, "destination": d,
                         "trips": float(region_matrix.loc[o, d])})
    rel = corridor_relevant_od(region_matrix).stack().rename("trips").reset_index()
    rel.columns = ["origin", "destination", "trips"]
    rel = rel[rel["origin"] != rel["destination"]].nlargest(n_largest, "trips")
    rel.insert(0, "set", f"{n_largest} largest relations")
    return pd.concat([pd.DataFrame(rows), rel], ignore_index=True)


def passthrough_test(tmi, ctx, corridor, fractions=(0.0, 0.05, 0.10)) -> pd.DataFrame:
    """Demand categories for several pass-through shares (decision D4)."""
    frames = []
    for f in fractions:
        _, br = tmi.collapse_od_to_gates(ctx.baseline_od, ctx, corridor, passthrough_fraction=f)
        frames.append(br.assign(passthrough_fraction=f))
    out = pd.concat(frames, ignore_index=True)
    return out.pivot(index="category", columns="passthrough_fraction", values="trips").reset_index()


def plot_region_heatmap(region_matrix: pd.DataFrame, title: str):
    """Region-to-region OD matrix as a heatmap with values written in each cell."""
    m = corridor_relevant_od(region_matrix)
    fig, ax = plt.subplots(figsize=(12, 9.5))
    im = ax.imshow(np.log10(m.to_numpy() + 1), cmap="Blues")
    ax.set_xticks(range(len(m.columns)), m.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(m.index)), m.index, fontsize=8)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.iat[i, j]
            if v >= 0.5:
                ax.text(j, i, f"{v:,.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if v > 2000 else "black")
    ax.set_xlabel("Destination region")
    ax.set_ylabel("Origin region")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.6, label="log10(peak-hour trips + 1)")
    ax.grid(False)
    return fig


def region_anchor_points(ctx, regions: pd.Series, stations: pd.DataFrame, corridor) -> pd.DataFrame:
    """One map point per region: population-weighted centroid inside, a point on the compass ring outside."""
    zones = ctx.zones.assign(region=regions.reindex(ctx.zones["grid_id"].astype(str)).to_numpy())
    cent = gpd.GeoSeries(zones["centroid"], crs=zones.crs)
    w = zones["baseline_population"].fillna(0) + zones["baseline_jobs"].fillna(0) + 1
    pts = pd.DataFrame({"x": cent.x * w, "y": cent.y * w, "w": w, "region": zones["region"]}).groupby("region").sum()
    pts["x"] /= pts["w"]
    pts["y"] /= pts["w"]
    fx, fy = forch_station_xy(stations)
    radius = 9000.0
    for name, (lo, hi) in EXTERNAL_SECTORS.items():
        mid = np.radians(((lo + hi + (360 if hi < lo else 0)) / 2) % 360)
        pts.loc[name, ["x", "y"]] = fx + radius * np.sin(mid), fy + radius * np.cos(mid)
    return pts[["x", "y"]]


PFANNENSTIEL = ["Zollikerberg", "Zumikon", "Forch hub area", "Rest of Maur", "Rest of Egg", "Rest of Küsnacht"]


def plot_desire_lines(ctx, corridor, region_matrix, anchors, title, min_trips=20.0, focus=None):
    """Desire lines between regions: width proportional to trips in both directions together.

    With `focus`, only relations with at least one end in those regions are drawn,
    so the small Pfannenstiel flows are not swamped by the city.
    """
    m = corridor_relevant_od(region_matrix)
    pairs = {}
    for o in m.index:
        for d in m.columns:
            if o < d and (focus is None or o in focus or d in focus):
                pairs[(o, d)] = m.loc[o, d] + m.loc[d, o]
    fig, ax = plt.subplots(figsize=(11, 9.5))
    ctx.zones.cx[2672000:2706000, 1230000:1258000].boundary.plot(ax=ax, color="#e3e3e3", linewidth=0.3)
    gpd.GeoSeries([corridor.polygon], crs=ctx.zones.crs).boundary.plot(ax=ax, color="black", linestyle="--", linewidth=0.8)
    top = max(pairs.values())
    for (o, d), v in sorted(pairs.items(), key=lambda kv: kv[1]):
        if v < min_trips:
            continue
        ax.plot([anchors.loc[o, "x"], anchors.loc[d, "x"]], [anchors.loc[o, "y"], anchors.loc[d, "y"]],
                color="#a50f15" if "Forch hub area" in (o, d) else "#3366aa",
                linewidth=0.5 + 12 * v / top, alpha=0.55, solid_capstyle="round", zorder=3)
    placed = []
    for name, row in anchors.sort_values("y", ascending=False).iterrows():
        intra = m.loc[name, name] if name in m.index else 0
        ax.scatter(row["x"], row["y"], s=20 + 400 * intra / max(np.diag(m.to_numpy()).max(), 1),
                   color="#333333", zorder=4)
        # Anchors closer than 1.2 km (Zürich centre and the city buffer) would print on top of each other.
        crowd = sum(np.hypot(row["x"] - x, row["y"] - y) < 1200 for x, y in placed)
        placed.append((row["x"], row["y"]))
        ax.annotate(name, (row["x"], row["y"]), xytext=(6, 4 - 11 * crowd), textcoords="offset points",
                    fontsize=7.5, zorder=5, bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.7))
    ax.plot([], [], color="#a50f15", linewidth=3, label="relation with the Forch hub area")
    ax.plot([], [], color="#3366aa", linewidth=3, label="other relation")
    ax.legend(loc="lower left", fontsize=8, title=f"width = trips both ways (max {top:,.0f}/h); dot = intra-region trips",
              title_fontsize=7.5)
    ax.set_xlim(anchors["x"].min() - 2500, anchors["x"].max() + 6000)
    ax.set_ylim(anchors["y"].min() - 2500, anchors["y"].max() + 2500)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.grid(False)
    ax.set_xlabel("E (m, EPSG:2056)")
    ax.set_ylabel("N (m, EPSG:2056)")
    return fig


# ---------------------------------------------------------------------------
# §1.5 Modal split
# ---------------------------------------------------------------------------

MODES = {"drive": "Car", "pt_walk": "PT-walk", "pt_bike": "PT-bike", "bike": "Bike", "walk": "Walk"}


def _aligned(frame: pd.DataFrame, labels) -> np.ndarray:
    v = frame.rename(index=str, columns=str).reindex(index=labels, columns=labels).to_numpy(dtype=float)
    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)


def mode_arrays(ctx, mode_result):
    """Person trips by mode and the car-road OD distance (km) on one common zone order."""
    labels = ctx.zones["grid_id"].astype(str).tolist()
    trips = {m: _aligned(mode_result.od_by_mode[m], labels) for m in MODES}
    km = _aligned(ctx.lengths["drive"], labels) / 1000.0
    return labels, trips, km


def od_masks(labels, corridor, regions: pd.Series | None = None) -> dict:
    """Boolean OD masks for the demand categories used in §1.4 and §1.5.

    'pass-through' marks external-to-external pairs whose entry and exit gates
    differ (the pairs collapse_od_to_gates scales by the pass-through share).
    """
    ids = np.asarray(labels)
    inside = np.isin(ids, np.asarray(list(corridor.zone_ids), dtype=str))
    entry = np.array([corridor.external_zone_to_entry_gate.get(z, "") for z in ids])
    exit_ = np.array([corridor.external_zone_to_exit_gate.get(z, "") for z in ids])
    ext = ~inside
    through = ext[:, None] & ext[None, :] & (entry[:, None] != exit_[None, :]) \
        & (entry[:, None] != "") & (exit_[None, :] != "")
    return {
        "corridor (at least one end inside)": inside[:, None] | inside[None, :],
        "internal -> internal": inside[:, None] & inside[None, :],
        "inbound (external -> internal)": ext[:, None] & inside[None, :],
        "outbound (internal -> external)": inside[:, None] & ext[None, :],
        "pass-through candidates (external -> external, different gates)": through,
    }


def mode_table(trips: dict, km: np.ndarray, mask: np.ndarray, min_km: float | None = None) -> pd.DataFrame:
    """Trips, trip shares, PKM and PKM shares for the five modes, PT aggregated, and the total.

    PKM uses the car-road OD distance for every mode, as the course's strategic
    indicator does; min_km drops OD pairs shorter than that distance from the PKM part only.
    """
    pkm_mask = mask & (km >= min_km) if min_km else mask
    rows = {MODES[m]: (trips[m][mask].sum(), (trips[m] * km)[pkm_mask].sum()) for m in MODES}
    rows["PT (all)"] = (rows["PT-walk"][0] + rows["PT-bike"][0], rows["PT-walk"][1] + rows["PT-bike"][1])
    order = ["Car", "PT (all)", "PT-walk", "PT-bike", "Bike", "Walk"]
    t = pd.DataFrame([(k, *rows[k]) for k in order], columns=["mode", "trips", "pkm"])
    total = t.loc[t["mode"].isin(["Car", "PT (all)", "Bike", "Walk"]), ["trips", "pkm"]].sum()
    t["trip_share"] = t["trips"] / total["trips"]
    t["pkm_share"] = t["pkm"] / total["pkm"] if total["pkm"] > 0 else np.nan
    t.loc[len(t)] = ["Total", total["trips"], total["pkm"], 1.0, 1.0 if total["pkm"] > 0 else np.nan]
    return t


def split_by_category(trips, km, masks: dict, min_km: float) -> pd.DataFrame:
    """Mode table for every demand category, stacked."""
    frames = []
    for name, mask in masks.items():
        frames.append(mode_table(trips, km, mask, min_km).assign(category=name))
    return pd.concat(frames, ignore_index=True)[["category", "mode", "trips", "trip_share", "pkm", "pkm_share"]]


def relation_mask(labels, regions: pd.Series, origin: str, destination: str) -> np.ndarray:
    lab = regions.reindex(labels).to_numpy()
    return (lab == origin)[:, None] & (lab == destination)[None, :]


def split_by_relation(trips, km, labels, regions, relations, min_km=None) -> pd.DataFrame:
    """Trip-based (and optionally PKM-based) shares for a list of (origin, destination) region pairs."""
    rows = []
    for o, d in relations:
        t = mode_table(trips, km, relation_mask(labels, regions, o, d), min_km).set_index("mode")
        mean_km = (sum(trips[m] for m in MODES) * km)[relation_mask(labels, regions, o, d)].sum() / max(t.loc["Total", "trips"], 1e-9)
        row = {"origin": o, "destination": d, "total_trips": t.loc["Total", "trips"], "mean_car_km": mean_km}
        for mode in ["Car", "PT (all)", "PT-walk", "PT-bike", "Bike", "Walk"]:
            row[f"{mode} trip share"] = t.loc[mode, "trip_share"]
        rows.append(row)
    return pd.DataFrame(rows)


def distance_band_test(trips, km, labels, regions, hub="Forch hub area",
                       partners=("Zumikon", "Rest of Egg", "Rest of Maur"), band=(3.0, 5.0)) -> pd.DataFrame:
    """Decision D5: share of the hub area's motorised trips that are 3-5 km trips with Zumikon, Egg or Maur."""
    lab = regions.reindex(labels).to_numpy()
    hub_end = (lab == hub)[:, None] | (lab == hub)[None, :]
    partner = np.isin(lab, partners)
    with_partner = ((lab == hub)[:, None] & partner[None, :]) | (partner[:, None] & (lab == hub)[None, :])
    motor = trips["drive"] + trips["pt_walk"] + trips["pt_bike"]
    in_band = (km >= band[0]) & (km < band[1])
    total = motor[hub_end].sum()
    rows = [
        ("motorised trips with an end in the hub area", total),
        (f"of which with {', '.join(partners)}", motor[hub_end & with_partner].sum()),
        (f"of which with those areas and {band[0]:g}-{band[1]:g} km", motor[hub_end & with_partner & in_band].sum()),
        (f"all hub-area motorised trips {band[0]:g}-{band[1]:g} km", motor[hub_end & in_band].sum()),
        (f"all hub-area motorised trips under {band[1]:g} km", motor[hub_end & (km < band[1])].sum()),
    ]
    t = pd.DataFrame(rows, columns=["quantity", "peak_hour_trips"])
    t["share_of_hub_motorised"] = t["peak_hour_trips"] / total
    return t


def plot_distance_histogram(trips, km, mask, title, bins=np.arange(0, 41, 1.0)):
    """Trips by car-road OD distance and mode (stacked), for one OD mask."""
    fig, ax = plt.subplots(figsize=(10, 4.2))
    bottom = np.zeros(len(bins) - 1)
    colours = {"drive": "#4c78a8", "pt_walk": "#f58518", "pt_bike": "#ffbf79", "bike": "#54a24b", "walk": "#b279a2"}
    for m, label in MODES.items():
        h, _ = np.histogram(km[mask], bins=bins, weights=trips[m][mask])
        ax.bar(bins[:-1], h, width=np.diff(bins), bottom=bottom, align="edge", color=colours[m], label=label,
               edgecolor="white", linewidth=0.3)
        bottom += h
    for x in (3, 5):
        ax.axvline(x, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Car-road OD distance (km), NPVM skim")
    ax.set_ylabel("Peak-hour person trips")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    return fig


def check_mode_choice(ctx, mode_result, tol: float = 1e-6) -> pd.DataFrame:
    """The four checks Sheet 2 §1.5 asks for, on every OD pair of the canton."""
    labels, trips, _ = mode_arrays(ctx, mode_result)
    demand = _aligned(ctx.baseline_od, labels) * float(mode_result.scenario.get("demand_multiplier", 1.0) or 1.0)
    total = sum(trips.values())
    positive = demand > 0
    probs = {m: np.divide(trips[m], demand, out=np.zeros_like(demand), where=positive) for m in MODES}
    psum = sum(probs.values())
    shares = np.array([trips[m].sum() for m in MODES]) / total.sum()
    rows = [
        ("probabilities sum to one for every OD pair with demand", float(np.abs(psum[positive] - 1).max()),
         bool(np.abs(psum[positive] - 1).max() < tol)),
        ("aggregate modal shares sum to one", float(abs(shares.sum() - 1)), bool(abs(shares.sum() - 1) < tol)),
        ("every probability lies in [0, 1]", float(min(p_[positive].min() for p_ in probs.values())),
         bool(all((p_[positive] >= -tol).all() and (p_[positive] <= 1 + tol).all() for p_ in probs.values()))),
        ("trips by mode sum to total demand", float(abs(total.sum() - demand.sum())),
         bool(abs(total.sum() - demand.sum()) < tol * max(demand.sum(), 1))),
    ]
    return pd.DataFrame(rows, columns=["check", "worst_value", "passed"])


def plot_relation_shares(relations: pd.DataFrame, corridor_split: pd.DataFrame):
    """Stacked trip shares for the key relations, with the corridor total as the last bar."""
    modes = ["Car", "PT-walk", "PT-bike", "Bike", "Walk"]
    colours = ["#4c78a8", "#f58518", "#ffbf79", "#54a24b", "#b279a2"]
    labels = [f"{o} -> {d}" for o, d in zip(relations["origin"], relations["destination"])] + ["Corridor total"]
    total = corridor_split.set_index("mode")["trip_share"]
    data = np.vstack([relations[[f"{m} trip share" for m in modes]].to_numpy(),
                      total.reindex(modes).to_numpy()[None, :]])
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(labels) + 1.5))
    left = np.zeros(len(labels))
    for k, (m, c) in enumerate(zip(modes, colours)):
        ax.barh(labels, data[:, k], left=left, color=c, label=m, edgecolor="white", linewidth=0.4)
        left += data[:, k]
    counts = list(relations["total_trips"]) + [corridor_split.set_index("mode").loc["Total", "trips"]]
    for y, n in enumerate(counts):
        ax.text(1.01, y, f"{n:,.0f} trips/h", va="center", fontsize=7.5)
    ax.set_xlim(0, 1.18)
    ax.invert_yaxis()
    ax.set_xlabel("Trip share (Stage 0, peak hour 17:00-18:00)")
    ax.legend(ncol=5, fontsize=8, loc="lower center", bbox_to_anchor=(0.45, 1.0))
    ax.tick_params(axis="y", labelsize=8)
    return fig


# ---------------------------------------------------------------------------
# Validation of demand and modal split (S1_validation.md)
# ---------------------------------------------------------------------------

# Published reference values (decision D8). Each has its source and access date.
REFERENCES = [
    # id, quantity, value, unit, source, url
    ("R1", "Trips per person per day, canton of Zürich residents (MZMV 2021)", 2.9, "trips/person/day",
     "Statistisches Amt Kanton Zürich (2023), MZMV 2021 report: 'knapp drei Wege' per day; 2.9 is our reading of 'just under three'",
     "https://www.web.statistik.zh.ch/ogd/daten/ressourcen/KTZH_00002342_00004583.pdf"),
    ("R2a", "Share of daily distance by car/motorbike (MIV), canton of Zürich 2021", 0.62, "share",
     "Statistisches Amt Kanton Zürich (2023), MZMV 2021, annex table 'Entwicklung des Modalsplits 1994-2021'",
     "https://www.web.statistik.zh.ch/ogd/daten/ressourcen/KTZH_00002342_00004583.pdf"),
    ("R2b", "Share of daily distance by public transport (ÖV), canton of Zürich 2021", 0.28, "share",
     "same table (text of the zh.ch page gives 27 %)",
     "https://www.web.statistik.zh.ch/ogd/daten/ressourcen/KTZH_00002342_00004583.pdf"),
    ("R2c", "Share of daily distance on foot and by bike/e-bike (FVV), canton of Zürich 2021", 0.10, "share",
     "same table", "https://www.web.statistik.zh.ch/ogd/daten/ressourcen/KTZH_00002342_00004583.pdf"),
    ("R3", "People in the 1.5 km catchment of Forch station (preliminary appraisal; read as residents, v2 assumption A26)", 5000.0, "people",
     "IP-HS2026-Report-Hub.pdf, §5.1, printed p. 11", "Assignments/IP-HS2026-Report-Hub.pdf (course document)"),
    ("R4", "Average trip length of catchment residents (preliminary appraisal)", 10.0, "km",
     "IP-HS2026-Report-Hub.pdf, §5.1", "Assignments/IP-HS2026-Report-Hub.pdf (course document)"),
    ("R5", "Forchbahn passengers per year (2024)", 6.2e6, "passengers/year",
     "Forchbahn AG, Geschäftsbericht 2024, p. 3-4 ('mehr als 6,2 Millionen')",
     "https://forchbahn.ch/wp-content/uploads/2025/09/Forchbahn_Geschaeftsbericht_2024.pdf"),
]
ACCESSED = "2026-10-09"


def daily_from_peak(trips: dict, mask: np.ndarray, peak_shares: dict) -> float:
    """Convert peak-hour trips by mode to weekday trips with the course's mode-specific peak shares."""
    share = {"drive": peak_shares["CAR_PEAK_SHARE"], "pt_walk": peak_shares["PT_PEAK_SHARE"],
             "pt_bike": peak_shares["PT_PEAK_SHARE"], "bike": peak_shares["BIKE_PEAK_SHARE"],
             "walk": peak_shares["WALK_PEAK_SHARE"]}
    return float(sum(trips[m][mask].sum() / share[m] for m in MODES))


def validation_comparators(ctx, mode_result, trips, km, labels, regions, masks, params) -> pd.DataFrame:
    """Model values that correspond to each published reference, computed from this run."""
    lab = regions.reindex(labels).to_numpy()
    zones = ctx.zones.set_index(ctx.zones["grid_id"].astype(str)).reindex(labels)
    pop = zones["baseline_population"].fillna(0).to_numpy()
    hub = lab == "Forch hub area"
    pfann = np.isin(lab, PFANNENSTIEL)
    corridor_zone = np.isin(np.asarray(labels), np.asarray(list(map(str, ctx.zones["grid_id"]))))
    origin_hub = hub[:, None] & np.ones_like(hub)[None, :]
    origin_pf = pfann[:, None] & np.ones_like(pfann)[None, :]
    motor = trips["drive"] + trips["pt_walk"] + trips["pt_bike"]
    hub_any = hub[:, None] | hub[None, :]
    pf_any = pfann[:, None] | pfann[None, :]
    nf = mode_table(trips, km, masks["corridor (at least one end inside)"], None).set_index("mode")
    pt_pf = {m: trips[m] * pf_any for m in ("pt_walk", "pt_bike")}
    pt_daily_pf = sum(pt_pf[m][pf_any].sum() for m in pt_pf) / params["PT_PEAK_SHARE"]
    rows = [
        ("R1", "Weekday trips starting in the Forch hub area per resident (peak hour converted with the course's peak shares)",
         daily_from_peak(trips, origin_hub, params) / pop[hub].sum(), "trips/resident/day"),
        ("R1", "Weekday trips starting in the six Pfannenstiel regions per resident",
         daily_from_peak(trips, origin_pf, params) / pop[pfann].sum(), "trips/resident/day"),
        ("R2a", "Car PKM share, corridor, peak hour, no distance filter (car-road OD distance for every mode); the reference is a daily share, so this row is context only (daily comparison: tables/s1_v2_validation.csv)", nf.loc["Car", "pkm_share"], "share"),
        ("R2b", "PT PKM share, corridor, peak hour, no distance filter; context only (daily comparison in s1_v2_validation.csv)", nf.loc["PT (all)", "pkm_share"], "share"),
        ("R2c", "Bike + walk PKM share, corridor, peak hour, no distance filter; context only (daily comparison in s1_v2_validation.csv)",
         nf.loc["Bike", "pkm_share"] + nf.loc["Walk", "pkm_share"], "share"),
        ("R3", "Residents of the four Forch-hub-area zones (zones.parquet baseline_population)", pop[hub].sum(), "residents"),
        ("R4", "Mean car-road OD distance of motorised peak-hour trips with an end in the hub area; the reference is a daily mean, so context only (daily comparison in s1_v2_validation.csv)",
         (motor * km)[hub_any].sum() / motor[hub_any].sum(), "km"),
        ("R5", "PT trips with an end in the six Pfannenstiel regions, per year (peak / PT peak share x equivalent days); OD trips against line boardings, an order-of-magnitude bracket only",
         pt_daily_pf * params["EQUIVALENT_DAYS_PER_YEAR"], "trips/year"),
    ]
    return pd.DataFrame(rows, columns=["ref_id", "model_quantity", "model_value", "model_unit"])


def validation_table(model: pd.DataFrame) -> pd.DataFrame:
    """Join model values to the references and compute the gap in percent of the reference."""
    ref = pd.DataFrame(REFERENCES, columns=["ref_id", "reference_quantity", "reference_value", "reference_unit",
                                            "source", "url"]).assign(accessed=ACCESSED)
    t = model.merge(ref, on="ref_id", how="left")
    t["gap_pct"] = 100 * (t["model_value"] - t["reference_value"]) / t["reference_value"]
    return t


def no_path_diagnostics(ctx, mode_result, labels, mask) -> pd.DataFrame:
    """Share of corridor demand whose skim says 'no path' (999 min) for each mode, and the worst origin zones."""
    od = _aligned(ctx.baseline_od, labels)
    rows = []
    for m in MODES:
        t = _aligned(mode_result.travel_times[m], labels)
        nop = mask & (t >= 999)
        worst = pd.Series((od * nop).sum(axis=1), index=labels).nlargest(3)
        rows.append({"mode": MODES[m], "demand_without_path_share": od[nop].sum() / od[mask].sum(),
                     "worst_origin_zones": ", ".join(f"{z} ({v:.0f})" for z, v in worst.items() if v > 0)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# §1.6 Assignment
# ---------------------------------------------------------------------------

def link_frame(ctx, links: pd.DataFrame, regions: pd.Series | None = None) -> gpd.GeoDataFrame:
    """Assigned links with delay in vehicle-hours, location (E/N, WGS84) and the municipality they lie in."""
    g = gpd.GeoDataFrame(links.copy(), geometry="geometry", crs=ctx.zones.crs)
    g["delay_veh_h"] = g["flow_vehicles"] * g["delay_min"] / 60.0
    mid = g.geometry.interpolate(0.5, normalized=True)
    g["x"], g["y"] = mid.x.round(0), mid.y.round(0)
    wgs = gpd.GeoSeries(mid, crs=ctx.zones.crs).to_crs(4326)
    g["lon"], g["lat"] = wgs.x.round(4), wgs.y.round(4)
    joined = gpd.sjoin(gpd.GeoDataFrame(geometry=mid, crs=ctx.zones.crs),
                       ctx.zones[["grid_id", "municipality_name", "city_quartier", "geometry"]], how="left",
                       predicate="within")
    joined = joined[~joined.index.duplicated()]
    g["municipality"] = joined["municipality_name"].reindex(g.index).to_numpy()
    g["zone"] = joined["grid_id"].astype(str).reindex(g.index).to_numpy()
    g["area"] = np.where(g["municipality"].eq("Zürich"), "Zürich city", "outside Zürich city")
    if regions is not None:
        g["region"] = regions.reindex(g["zone"]).to_numpy()
    return g


def assignment_summary(name: str, result, links: gpd.GeoDataFrame) -> dict:
    """One row of headline assignment results, total and split city / outside city."""
    d = result.diagnostics
    out = {"run": name, "iterations": d["iterations"], "termination": d.get("termination_reason"),
           "final_road_gap": d.get("final_road_relative_gap_feasible_averaged_od"),
           "vehicles_assigned": d.get("assignment_vehicle_trips"), "car_driver_vehicles": d.get("drive_vehicle_trips"),
           "vkt": d["total_vkt"], "vht": d["total_vht"], "delay_veh_h": d["total_delay_hours"],
           "mean_speed_kph": d["avg_speed_kph"], "max_vc": float(links["volume_capacity_ratio"].max()),
           "links_vc_over_1": int((links["volume_capacity_ratio"] > 1).sum()),
           "runtime_s": d.get("runtime_s")}
    for area, part in links.groupby("area"):
        tag = "city" if area == "Zürich city" else "outside_city"
        out[f"delay_veh_h_{tag}"] = part["delay_veh_h"].sum()
        vkt = (part["flow_vehicles"] * part["length_m"] / 1000).sum()
        vht = (part["flow_vehicles"] * part["time_min"] / 60).sum()
        out[f"mean_speed_kph_{tag}"] = vkt / vht if vht > 0 else np.nan
    return out


def bottlenecks(links: gpd.GeoDataFrame, n: int = 15, area: str | None = None) -> pd.DataFrame:
    """Links with the largest delay (vehicle-hours), optionally only in one area."""
    part = links if area is None else links[links["area"] == area]
    cols = ["edge_id", "highway", "municipality", "region", "lat", "lon", "length_m", "lanes", "capacity_vph",
            "flow_vehicles", "volume_capacity_ratio", "free_flow_time_min", "delay_min", "assigned_speed_kph",
            "delay_veh_h"]
    cols = [c for c in cols if c in part]
    return part.nlargest(n, "delay_veh_h")[cols].reset_index(drop=True)


def assignment_checks(result, links: pd.DataFrame, n_last: int = 3) -> pd.DataFrame:
    """Sheet 2 §1.6 checks: plausible flows, non-negative times and delays, stable last iterations."""
    h = result.history
    gap_col = "road_relative_gap_feasible_averaged_od"
    checked = h.loc[np.isfinite(h[gap_col])] if gap_col in h else h.iloc[0:0]
    last = checked.tail(n_last)
    share_cols = [c for c in ("car_share", "pt_share", "bike_share", "walk_share") if c in h]
    share_move = float(h[share_cols].tail(n_last).diff().abs().max().max()) if share_cols else np.nan
    rows = [
        ("no negative link flows", float(links["flow_vehicles"].min()), bool((links["flow_vehicles"] >= 0).all())),
        ("no negative congested times", float(links["time_min"].min()), bool((links["time_min"] >= 0).all())),
        ("no negative delays", float(links["delay_min"].min()), bool((links["delay_min"] >= -1e-9).all())),
        ("congested time >= free-flow time on every link",
         float((links["time_min"] - links["free_flow_time_min"]).min()),
         bool((links["time_min"] >= links["free_flow_time_min"] - 1e-9).all())),
        ("all OD demand served (unserved vehicles)", float(result.diagnostics.get("unserved_vehicles", 0.0)),
         float(result.diagnostics.get("unserved_vehicles", 0.0)) < 1.0),
        ("road gap at the last checks (values)", float(last[gap_col].iloc[-1]) if len(last) else np.nan,
         bool(len(last) and last[gap_col].iloc[-1] <= 0.02)),
        (f"largest change of a mode share over the last {n_last} iterations (share points)", share_move,
         bool(share_move < 0.005) if np.isfinite(share_move) else "not applicable (fixed demand)"),
        ("links with V/C > 2 (count; plausibility flag, not a failure)", float((links["volume_capacity_ratio"] > 2).sum()),
         True),
    ]
    return pd.DataFrame(rows, columns=["check", "value", "passed"])


def plot_link_map(ctx, corridor, links: gpd.GeoDataFrame, column: str, title: str, vmax=None, cmap="magma_r",
                  label=None, zoom=None):
    """Static map of one link attribute, line width by flow."""
    fig, ax = plt.subplots(figsize=(11, 9.5))
    xmin, ymin, xmax, ymax = zoom or corridor.polygon.buffer(800).bounds
    ctx.zones.cx[xmin:xmax, ymin:ymax].boundary.plot(ax=ax, color="#e6e6e6", linewidth=0.3)
    gpd.GeoSeries([corridor.polygon], crs=ctx.zones.crs).boundary.plot(ax=ax, color="black", linestyle="--", linewidth=0.8)
    order = links.sort_values(column)
    width = 0.3 + 3.5 * order["flow_vehicles"] / max(order["flow_vehicles"].max(), 1)
    order.plot(ax=ax, column=column, cmap=cmap, linewidth=width, vmin=0, vmax=vmax, legend=True,
               legend_kwds={"label": label or column, "shrink": 0.6})
    corridor.gates.plot(ax=ax, color="crimson", markersize=14, marker="^", zorder=5)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.grid(False)
    ax.set_title(title, fontsize=10)
    return fig


# ---------------------------------------------------------------------------
# §1.7 Forch-specific indicators
# ---------------------------------------------------------------------------

def year_pt_split(ctx, corridor, assignments: dict, min_km: float) -> pd.DataFrame:
    """Corridor trip and PKM shares per mode for each year run, with PT split by access mode (D6).

    The course's year tables only report PT as one mode; this uses the same corridor
    definition as §1.5 (at least one trip end inside the cordon).
    """
    frames = []
    for year, assignment in assignments.items():
        labels, trips, km = mode_arrays(ctx, assignment.mode_result)
        mask = od_masks(labels, corridor)["corridor (at least one end inside)"]
        frames.append(mode_table(trips, km, mask, min_km).assign(year=year))
    return pd.concat(frames, ignore_index=True)[["year", "mode", "trips", "trip_share", "pkm_share"]]


def forch_indicators(ctx, assignment, regions: pd.Series) -> dict:
    """Indicators that isolate the Forch side of the corridor from the city jam (see §1.6)."""
    labels, trips, km = mode_arrays(ctx, assignment.mode_result)
    lab = regions.reindex(labels).to_numpy()
    out = {}
    for name, members in (("Forch hub area", ["Forch hub area"]), ("Pfannenstiel regions", PFANNENSTIEL)):
        z = np.isin(lab, members)
        mask = z[:, None] | z[None, :]
        t = mode_table(trips, km, mask, None).set_index("mode")
        out[f"{name}: trips with an end in the area (peak hour)"] = t.loc["Total", "trips"]
        out[f"{name}: car trip share"] = t.loc["Car", "trip_share"]
        out[f"{name}: PT trip share"] = t.loc["PT (all)", "trip_share"]
        out[f"{name}: PT-walk trip share"] = t.loc["PT-walk", "trip_share"]
        out[f"{name}: PT-bike trip share"] = t.loc["PT-bike", "trip_share"]
    links = link_frame(ctx, assignment.links)
    out["Road delay outside Zürich city (veh-h, peak hour)"] = links.loc[links["area"] != "Zürich city", "delay_veh_h"].sum()
    out["Road delay in Zürich city part of the cordon (veh-h, peak hour)"] = links.loc[links["area"] == "Zürich city", "delay_veh_h"].sum()
    return out
