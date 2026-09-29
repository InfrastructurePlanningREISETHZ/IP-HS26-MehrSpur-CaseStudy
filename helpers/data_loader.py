from pathlib import Path
from typing import Dict
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def load_excel_file(filepath: Path) -> Dict[str, pd.DataFrame]:
    """Load all sheets from a single Excel file."""
    return pd.read_excel(filepath, sheet_name=None)


def load_all_excels(raw_dir: Path = RAW_DIR) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load every Excel file in the raw data folder."""
    files = sorted(raw_dir.glob("*.xlsx")) + sorted(raw_dir.glob("*.xls"))
    return {file.stem: load_excel_file(file) for file in files}


def save_dataframe(df: pd.DataFrame, name: str, processed_dir: Path = PROCESSED_DIR) -> Path:
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / f"{name}.csv"
    df.to_csv(output_path, index=False)
    return output_path
