from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from config import (
    ASSIGNMENT_NETWORK_FILE,
    BASE_NETWORK_FILE,
    BIKE_GRAPH_FILE,
    DEMAND_PACKAGE_FILE,
    EXTERNAL_BACKGROUND_GATEWAYS_FILE,
    GTFS_FILE,
    INPUT_DIR,
    LOOKUP_PACKAGE_FILE,
    MUNICIPAL_BOUNDARIES_ARCHIVE,
    NETWORK_POLICY_CITY_FILE,
    NPVM_ACTIVE_DEMAND_ARCHIVE,
    NPVM_PT_DEMAND_ARCHIVE,
    NPVM_ROAD_DEMAND_ARCHIVE,
    NPVM_ZONES_ARCHIVE,
    PREPARED_DIR,
    PT_ACCESS_BIKE_GRAPH_FILE,
    PT_ACCESS_WALK_GRAPH_FILE,
    QUARTIER_BOUNDARIES_ARCHIVE,
    ROAD_GRAPH_FILE,
    SKIM_DIR,
    SKIM_PACKAGE_FILE,
    STATENT_ARCHIVE,
    STATPOP_ARCHIVE,
    WALK_GRAPH_FILE,
    WORK_DIR,
    WORK_GTFS_DIR,
    ZONES_FILE,
)
from input_packages import load_package, save_package
from travel_times import LENGTH_SKIMS, NUMERIC_SKIMS, STRING_SKIMS


RAW_INPUTS = (
    NPVM_ZONES_ARCHIVE,
    NPVM_ROAD_DEMAND_ARCHIVE,
    NPVM_PT_DEMAND_ARCHIVE,
    NPVM_ACTIVE_DEMAND_ARCHIVE,
    EXTERNAL_BACKGROUND_GATEWAYS_FILE,
    MUNICIPAL_BOUNDARIES_ARCHIVE,
    QUARTIER_BOUNDARIES_ARCHIVE,
    NETWORK_POLICY_CITY_FILE,
    STATPOP_ARCHIVE,
    STATENT_ARCHIVE,
    GTFS_FILE,
    BASE_NETWORK_FILE,
    ROAD_GRAPH_FILE,
    WALK_GRAPH_FILE,
    BIKE_GRAPH_FILE,
    PT_ACCESS_WALK_GRAPH_FILE,
    PT_ACCESS_BIKE_GRAPH_FILE,
)

PREPARED_INPUTS = (
    ZONES_FILE,
    DEMAND_PACKAGE_FILE,
    SKIM_PACKAGE_FILE,
    LOOKUP_PACKAGE_FILE,
    ASSIGNMENT_NETWORK_FILE,
)

LOOKUP_FILES = {
    "pt_zone_stop_access_walk": "pt_zone_stop_access_walk.parquet",
    "pt_zone_stop_access_bike": "pt_zone_stop_access_bike.parquet",
}


def _missing(paths) -> list[str]:
    return [str(path) for path in paths if not Path(path).exists()]


def validate_raw_inputs() -> None:
    missing = _missing(RAW_INPUTS)
    if missing:
        raise FileNotFoundError("Missing raw inputs:\n" + "\n".join(missing))
    print("Raw NPVM, boundary, population, GTFS and OSM inputs are present.")


def validate_prepared_inputs() -> None:
    missing = _missing(PREPARED_INPUTS)
    if missing:
        raise FileNotFoundError("Missing prepared inputs:\n" + "\n".join(missing))

    demand = load_package(DEMAND_PACKAGE_FILE, "demand")
    if not {"passenger_od", "road_background"}.issubset(demand):
        raise ValueError("The demand package is incomplete.")
    skims = load_package(SKIM_PACKAGE_FILE, "skims")
    if set(NUMERIC_SKIMS).difference(skims.get("travel_times", {})):
        raise ValueError("The numeric travel-time skims are incomplete.")
    if set(STRING_SKIMS).difference(skims.get("travel_times", {})):
        raise ValueError("The chosen-stop skims are incomplete.")
    if set(LENGTH_SKIMS).difference(skims.get("lengths", {})):
        raise ValueError("The distance skims are incomplete.")
    lookups = load_package(LOOKUP_PACKAGE_FILE, "lookups")
    if set(LOOKUP_FILES).difference(lookups):
        raise ValueError("The mobility-hub lookups are incomplete.")
    print("Prepared zones, demand, network, skim and lookup packages are ready.")


def package_skims() -> Path:
    if Path(SKIM_PACKAGE_FILE).exists():
        package = load_package(SKIM_PACKAGE_FILE, "skims")
        travel_times = dict(package.get("travel_times", {}))
        lengths = dict(package.get("lengths", {}))
    else:
        travel_times, lengths = {}, {}

    for key, filename in {**NUMERIC_SKIMS, **STRING_SKIMS}.items():
        path = Path(SKIM_DIR) / filename
        if path.exists():
            travel_times[key] = pd.read_parquet(path)
    for key, filename in LENGTH_SKIMS.items():
        path = Path(SKIM_DIR) / filename
        if path.exists():
            lengths[key] = pd.read_parquet(path)

    missing_times = set(NUMERIC_SKIMS).union(STRING_SKIMS).difference(travel_times)
    missing_lengths = set(LENGTH_SKIMS).difference(lengths)
    if missing_times or missing_lengths:
        raise FileNotFoundError(
            "Cannot package incomplete skims. "
            f"Missing times={sorted(missing_times)}, lengths={sorted(missing_lengths)}"
        )
    return save_package(
        SKIM_PACKAGE_FILE,
        "skims",
        {"travel_times": travel_times, "lengths": lengths},
    )


def package_lookups() -> Path:
    if Path(LOOKUP_PACKAGE_FILE).exists():
        lookups = load_package(LOOKUP_PACKAGE_FILE, "lookups")
    else:
        lookups = {}
    for key, filename in LOOKUP_FILES.items():
        path = Path(WORK_GTFS_DIR) / filename
        if path.exists():
            lookups[key] = pd.read_parquet(path)
    missing = set(LOOKUP_FILES).difference(lookups)
    if missing:
        raise FileNotFoundError(f"Cannot package missing lookup tables: {sorted(missing)}")
    return save_package(LOOKUP_PACKAGE_FILE, "lookups", lookups)


def clear_work_directory() -> None:
    work = Path(WORK_DIR).resolve()
    if work.parent != Path(INPUT_DIR).resolve():
        raise RuntimeError(f"Refusing to clear unexpected work directory: {work}")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    (work / "README.txt").write_text(
        "This is the disposable scratch directory for prepare_inputs.py.\n"
        "\n"
        "A rebuild may create:\n"
        "  npvm/   extracted OMX demand matrices\n"
        "  skims/  direct-mode and public-transport skim matrices\n"
        "  gtfs/   processed GTFS tables, diagnostics, and stop lookups\n"
        "\n"
        "These files are packaged into input_data/prepared/ and removed after a\n"
        "successful build. Use --keep-work to retain them for teaching or diagnostics.\n"
        "Main.py does not read this directory.\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the course model inputs.")
    parser.add_argument("--zones-demand", action="store_true", help="Rebuild zones and ASP demand.")
    parser.add_argument("--networks", action="store_true", help="Rebuild direct-mode skims and assignment network.")
    parser.add_argument("--gtfs", action="store_true", help="Rebuild public-transport skims and hub lookups.")
    parser.add_argument("--package", action="store_true", help="Package files already present in input_data/work.")
    parser.add_argument("--all", action="store_true", help="Rebuild all prepared inputs from raw data.")
    parser.add_argument("--check-runtime", action="store_true", help="Validate prepared runtime packages.")
    parser.add_argument("--check-raw", action="store_true", help="Validate raw preprocessing inputs.")
    parser.add_argument("--keep-work", action="store_true", help="Keep intermediate preprocessing files.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    requested_build = args.zones_demand or args.networks or args.gtfs or args.package or args.all
    if not requested_build:
        validate_raw_inputs() if args.check_raw else validate_prepared_inputs()
        raise SystemExit(0)

    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    if args.all:
        validate_raw_inputs()

    if args.zones_demand or args.all:
        from demand_preparation import prepare_demand
        from zoning import prepare_zones

        prepared_zones = prepare_zones()
        prepare_demand(prepared_zones)

    if args.networks or args.all:
        from network import process_networks
        from zoning import load_zones

        process_networks(load_zones())

    if args.gtfs or args.all:
        from gtfs_analysis import process_gtfs

        process_gtfs()

    if args.networks or args.gtfs or args.package or args.all:
        package_skims()
    if args.gtfs or args.package or args.all:
        package_lookups()

    validate_prepared_inputs()
    if not args.keep_work:
        clear_work_directory()
