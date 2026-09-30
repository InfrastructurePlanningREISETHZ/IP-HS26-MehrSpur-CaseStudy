"""Save completed notebook figures and tables to stable, notebook-specific files."""
from __future__ import annotations

from pathlib import Path
import re
import unicodedata
import warnings

import pandas as pd

_directories: tuple[Path, Path] | None = None


def _name(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")
    if not text:
        raise ValueError("An export needs a nonempty filename.")
    return text


def configure(project_root: str | Path, notebook: str, *, enabled: bool = True) -> None:
    """Enable exports for this kernel; callers in unconfigured workers do nothing."""
    global _directories
    if not enabled:
        _directories = None
        return
    root = Path(project_root).resolve()
    folder = _name(notebook)
    _directories = (root / "figures" / folder, root / "results" / folder)


def _table(value):
    if not isinstance(value, (pd.DataFrame, pd.Series)):
        value = getattr(value, "data", value)  # Pandas Styler retains its original values.
    if isinstance(value, pd.Series):
        value = value.to_frame(name=value.name if value.name is not None else "value")
    if not isinstance(value, pd.DataFrame):
        raise TypeError("CSV exports require a pandas DataFrame, Series or Styler.")
    return value


def _write(path: Path, writer) -> bool:
    temporary = path.with_name(path.name + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        writer(temporary)
        temporary.replace(path)
        return True
    except OSError as exc:
        warnings.warn(f"Could not save notebook output {path}: {exc}", RuntimeWarning, stacklevel=3)
        return False
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def save_outputs(name: str, *, figure=None, png: bytes | None = None,
                 tables: dict | None = None) -> dict[str, Path]:
    """Save a PNG and complete numerical CSV tables; reruns replace the same files.

    Pass a Matplotlib figure or existing PNG bytes, plus named tables. An empty
    table suffix uses the base name. Meaningful row labels are retained; default
    row numbers are omitted. Widget objects and interactive maps are not saved.
    """
    if _directories is None:
        return {}
    if figure is not None and png is not None:
        raise ValueError("Pass either a Matplotlib figure or PNG bytes, not both.")
    stem = _name(name)
    figure_dir, table_dir = _directories
    saved = {}
    if figure is not None or png is not None:
        path = figure_dir / f"{stem}.png"
        if png is not None:
            success = _write(path, lambda target: target.write_bytes(png))
        else:
            success = _write(path, lambda target: figure.savefig(
                target, format="png", dpi=150, bbox_inches="tight"))
        if success:
            saved["figure"] = path
    for suffix, value in (tables or {}).items():
        frame = _table(value)
        filename = f"{stem}_{_name(suffix)}" if suffix else stem
        path = table_dir / f"{filename}.csv"
        keep_index = (not isinstance(frame.index, pd.RangeIndex)
                      or frame.index.start != 0 or frame.index.step != 1
                      or any(label is not None for label in frame.index.names))
        if _write(path, lambda target: frame.to_csv(
                target, index=keep_index, encoding="utf-8-sig")):
            saved[f"table:{suffix}"] = path
    return saved
