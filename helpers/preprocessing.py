from pathlib import Path
import pandas as pd


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_column_names(df)
    # Add dataset-specific cleaning here.
    return df


def combine_dataframes(dfs: list[pd.DataFrame], key: str) -> pd.DataFrame:
    return pd.concat(dfs, ignore_index=True)
