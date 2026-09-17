from __future__ import annotations

import pandas as pd
from helpers.parsers import parse_distance_km, parse_height, parse_money_range, parse_percentage
from helpers.positioning import get_canonical_positions, get_primary_position, get_secondary_positions

PERCENTAGE_COLUMNS = [
    "Shots on Target Percentage", "Conv %", "Pass Completion Percentage",
    "Headers Won Percentage", "Crosses Completed Ratio",
    "Open Play Cross Completion Percentage", "Tackle Completion Percentage",
]


def load_and_clean(csv_path: str, *, require_rating: bool = False) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")

    if "Salary" in df:
        parsed = df["Salary"].apply(parse_money_range)
        df["salary_min_brl"] = parsed.map(lambda x: x[0])
        df["salary_max_brl"] = parsed.map(lambda x: x[1])
    if "Transfer Value" in df:
        parsed = df["Transfer Value"].apply(parse_money_range)
        df["transfer_value_min_brl"] = parsed.map(lambda x: x[0])
        df["transfer_value_max_brl"] = parsed.map(lambda x: x[1])
    if "Distance" in df:
        df["distance_km"] = df["Distance"].apply(parse_distance_km)
    if "Height" in df:
        df["height_cm"] = df["Height"].apply(parse_height)

    position_col = "Best Position" if "Best Position" in df else "Position" if "Position" in df else None
    if position_col:
        df["Primary Position"] = df[position_col].apply(get_primary_position)
        df["Secondary Positions"] = df[position_col].apply(lambda x: ",".join(get_secondary_positions(x)))
        df["All Positions"] = df[position_col].apply(lambda x: ",".join(get_canonical_positions(x)))

    for col in PERCENTAGE_COLUMNS:
        if col in df:
            df[col] = df[col].apply(parse_percentage)

    if "Rating" in df:
        df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
        if require_rating:
            df = df.dropna(subset=["Rating"])
    return df.reset_index(drop=True)