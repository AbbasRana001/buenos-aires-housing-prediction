import pandas as pd
from glob import glob

def merged_files(data):
    """Merge multiple CSV files matching a pattern into a single DataFrame."""
    return pd.concat([pd.read_csv(file) for file in glob(data)], ignore_index=True)

def filter_df(data):
    """Subset data: Apartments in "Capital Federal", less than $400,000"""
    return (
        data
        .loc[lambda x: x["place_with_parent_names"].str.contains("Capital Federal")]
        .query('property_type == "apartment"')
        .query('price_aprox_usd < 400_000')
    )

def outliers_df(data):
    """Remove outliers by surface_covered_in_m2 (keep middle 90%)"""
    return (
        data
        .loc[lambda x: x["surface_covered_in_m2"].between(
            x["surface_covered_in_m2"].quantile(0.1),
            x["surface_covered_in_m2"].quantile(0.9)
        )]
    )

def modify_cols(data):
    """Add and modify columns"""
    return (
        data
        .assign(
            lat=lambda x: x["lat-lon"].str.split(",", expand=True)[0].astype(float),
            lon=lambda x: x["lat-lon"].str.split(",", expand=True)[1].astype(float),
            neighborhood=lambda x: x["place_with_parent_names"].str.split("|", expand=True)[3]
        )
    )

def clean_files(file_path):
    """Final function to complete the cleaning process"""
    drop_cols = [
        "lat-lon", "place_with_parent_names",
        "floor", "expenses", "rooms", "price",
        "price_aprox_local_currency", "price_usd_per_m2", 
        "price_per_m2", "operation", "property_type", "currency", 
        "properati_url", "surface_total_in_m2"
    ]

    return (
        merged_files(file_path)
        .pipe(filter_df)
        .pipe(outliers_df)
        .pipe(modify_cols)
        .drop(columns=drop_cols)
        .dropna()
    )
