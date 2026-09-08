from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from background_traffic import prepare_road_background
from config import (
    DEMAND_PACKAGE_FILE,
    EXTERNAL_BACKGROUND_GATEWAYS_FILE,
    NPVM_ACTIVE_DEMAND_ARCHIVE,
    NPVM_PT_DEMAND_ARCHIVE,
    NPVM_ROAD_DEMAND_ARCHIVE,
    WORK_DIR,
)
from input_packages import save_package


PASSENGER_MODES = (
    (NPVM_ROAD_DEMAND_ARCHIVE, "PW_Binnen.omx", 0.1049 * 1.14),
    (NPVM_ROAD_DEMAND_ARCHIVE, "PW_QZD.omx", 0.1232 * 1.14),
    (NPVM_PT_DEMAND_ARCHIVE, "OEV.omx", 0.0946),
    (NPVM_ACTIVE_DEMAND_ARCHIVE, "VELO_NPVM_2023.omx", 0.15),
    (NPVM_ACTIVE_DEMAND_ARCHIVE, "EBIKE_NPVM_2023.omx", 0.15),
    (NPVM_ACTIVE_DEMAND_ARCHIVE, "FGV_NPVM_2023.omx", 0.15),
)

ROAD_BACKGROUND_MODES = (
    (NPVM_ROAD_DEMAND_ARCHIVE, "LI_Binnen.omx", 0.1349),
    (NPVM_ROAD_DEMAND_ARCHIVE, "LI_QZD.omx", 0.1349),
    (NPVM_ROAD_DEMAND_ARCHIVE, "LW.omx", 0.1087),
    (NPVM_ROAD_DEMAND_ARCHIVE, "LZ.omx", 0.0908),
)


def _zone_id(value) -> str:
    return str(value).strip().removesuffix(".0")


def _extract_member(archive: Path, member: str, directory: Path) -> Path:
    if not archive.exists():
        raise FileNotFoundError(f"Missing raw NPVM archive: {archive}")
    output = directory / member
    if output.exists():
        return output
    directory.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        if member not in source.namelist():
            raise FileNotFoundError(f"{member} is absent from {archive}")
        with source.open(member) as source_handle, output.open("wb") as output_handle:
            while chunk := source_handle.read(1024 * 1024):
                output_handle.write(chunk)
    return output


def _read_omx_subset(path: Path, zone_ids: list[str]) -> np.ndarray:
    try:
        import openmatrix as omx
    except ImportError as error:
        raise ImportError("Raw NPVM processing requires openmatrix.") from error

    with omx.open_file(path, "r") as source:
        matrix_name = source.list_matrices()[0]
        mapping = {_zone_id(key): int(value) for key, value in source.mapping("NO").items()}
        missing = [zone_id for zone_id in zone_ids if zone_id not in mapping]
        if missing:
            raise ValueError(f"NPVM matrix {path.name} is missing {len(missing)} requested zones")
        positions = np.asarray([mapping[zone_id] for zone_id in zone_ids], dtype=np.int64)
        values = source[matrix_name][positions, :][:, positions]
    return np.asarray(values, dtype=float)


def _aggregate_modes(
    mode_specs,
    zone_ids: list[str],
    extraction_dir: Path,
) -> np.ndarray:
    total = np.zeros((len(zone_ids), len(zone_ids)), dtype=float)
    for archive, member, factor in mode_specs:
        path = _extract_member(Path(archive), member, extraction_dir)
        total += _read_omx_subset(path, zone_ids) * float(factor)
    return total


def _matrix_to_long(values: np.ndarray, zone_ids: list[str]) -> pd.DataFrame:
    rows, columns = np.nonzero(values)
    labels = np.asarray(zone_ids, dtype=object)
    return pd.DataFrame(
        {
            "origin": labels[rows],
            "destination": labels[columns],
            "trips": values[rows, columns],
        }
    )


def prepare_demand(
    zones: pd.DataFrame,
    output_path: str | Path = DEMAND_PACKAGE_FILE,
    work_dir: str | Path = WORK_DIR,
) -> tuple[dict, dict]:
    """Aggregate public NPVM matrices directly to the course model zones."""
    model_ids = zones.sort_values("IntIDs")["grid_id"].astype(str).tolist()
    gateways = pd.read_csv(EXTERNAL_BACKGROUND_GATEWAYS_FILE, dtype=str)
    external_ids = gateways["external_zone_id"].map(_zone_id).tolist()
    background_ids = list(dict.fromkeys([*model_ids, *external_ids]))
    extraction_dir = Path(work_dir) / "npvm"

    passenger_matrix = _aggregate_modes(PASSENGER_MODES, model_ids, extraction_dir)
    passenger_od = _matrix_to_long(passenger_matrix, model_ids)

    background_matrix = _aggregate_modes(
        ROAD_BACKGROUND_MODES,
        background_ids,
        extraction_dir,
    )
    source_background = _matrix_to_long(background_matrix, background_ids).rename(
        columns={"origin": "Origin", "destination": "Destination", "trips": "Trips"}
    )
    road_background, background_diagnostics = prepare_road_background(
        zones,
        source_background,
    )

    metadata = {
        "period": "ASP 17:00-18:00",
        "passenger_trips": float(passenger_od["trips"].sum()),
        "passenger_od_pairs": int(len(passenger_od)),
        **background_diagnostics,
    }
    package = {
        "passenger_od": passenger_od,
        "road_background": road_background,
        "metadata": metadata,
    }
    save_package(output_path, "demand", package)
    return package, metadata
