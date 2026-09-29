import pandas as pd


def validate_modal_split(df: pd.DataFrame, mode_columns: list[str]) -> bool:
    """Confirm that the modal split probabilities sum to 1.0 for each row."""
    if not set(mode_columns).issubset(df.columns):
        raise ValueError("Some mode columns are missing from the dataframe.")
    row_sums = df[mode_columns].sum(axis=1)
    return row_sums.between(0.9999, 1.0001).all()
