from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw")

def read_fred(path: Path, value_name: str) -> pd.DataFrame:
    """
    Reads FRED-style CSV with columns: observation_date, <VALUE_COL>
    Returns columns: date, <value_name>
    """
    df = pd.read_csv(path, parse_dates=["observation_date"])
    # second column name varies; use df.columns[1]
    df = df.rename(columns={"observation_date": "date", df.columns[1]: value_name})
    return df[["date", value_name]].sort_values("date")

def read_eia_exports(path: Path) -> pd.DataFrame:
    """
    Reads EIA exports CSV with 4 metadata lines, then header:
    Month,Liquefied U.S. Natural Gas Exports  Million Cubic Feet
    Returns columns: date, us_lng_exports_mcf
    """
    df = pd.read_csv(path, skiprows=4)
    # Standardize column names
    df = df.rename(columns={df.columns[0]: "date", df.columns[1]: "us_lng_exports_mcf"})

    # Parse "Oct 2025" -> 2025-10-01
    df["date"] = pd.to_datetime(df["date"], format="%b %Y", errors="coerce")

    # Numeric conversion
    df["us_lng_exports_mcf"] = pd.to_numeric(df["us_lng_exports_mcf"], errors="coerce")

    # Sort ascending
    df = df.dropna(subset=["date"]).sort_values("date")
    return df[["date", "us_lng_exports_mcf"]]

def to_month_start(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forces dates to month-start timestamps (YYYY-MM-01) for clean monthly joins.
    """
    out = df.copy()
    out["date"] = out["date"].dt.to_period("M").dt.to_timestamp()
    return out

def load_merged_monthly() -> pd.DataFrame:
    hh = read_fred(DATA_DIR / "MHHNGSP.csv", "henry_hub_price")
    eu = read_fred(DATA_DIR / "PNGASEUUSDM.csv", "eu_gas_price")
    jp = read_fred(DATA_DIR / "PNGASJPUSDM.csv", "japan_gas_price")
    ex = read_eia_exports(DATA_DIR / "Liquefied_U.S._Natural_Gas_Exports.csv")

    # Convert all to month-start to join cleanly
    hh = to_month_start(hh)
    eu = to_month_start(eu)
    jp = to_month_start(jp)
    ex = to_month_start(ex)

    # Outer joins preserve the full timeline (some series may start later)
    df = hh.merge(eu, on="date", how="outer") \
           .merge(jp, on="date", how="outer") \
           .merge(ex, on="date", how="outer") \
           .sort_values("date")

    return df
