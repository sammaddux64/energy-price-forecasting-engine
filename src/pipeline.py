# -----------------------------
# pipeline.py
# Monthly forecasting pipeline (Power BI-aligned)
#
# Uses src/modeling.py for:
#   - 12-month horizon forecasts
#   - walk-forward backtest metrics (out-of-sample)
#
# Includes:
#   - speed recs (n_estimators=100, n_jobs=-1, later min_train_size)
#   - de-dupe for stable Power BI outputs
# -----------------------------

import sys
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

# -------------------------------------------------
# Ensure project root is on Python import path
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.load_data import load_merged_monthly
from src.modeling import ModelConfig, forecast_horizon_months, walk_forward_backtest

# -------------------------------------------------
# Output locations
# -------------------------------------------------
FORECAST_OUT = Path("outputs/forecast_results.csv")
METRICS_OUT = Path("outputs/model_metrics.csv")

# -------------------------------------------------
# Model metadata / config (DEV SPEED DEFAULTS)
# -------------------------------------------------
CFG = ModelConfig(
    model_name="random_forest_lags_rollmean_v1",
    horizon_months=12,
    n_estimators=100,  # speed default
    random_state=42,
)

# Walk-forward starts later to reduce retrains (speed)
MIN_TRAIN_SIZE = 120

# Map internal merged columns -> EXACT names used in your PBIX tables
SERIES = {
    "henry_hub_price": "HenryHub_USDMMBtu",
    "eu_gas_price": "EU_Gas_USDMMBtu",
    "japan_gas_price": "Asia_LNG_USDMMBtu",
    "us_lng_exports_mcf": "US_LNG_Exports_MMCF",
}

# -------------------------------------------------
# Main orchestration
# -------------------------------------------------
def main():
    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    df = load_merged_monthly()

    forecast_rows = []
    metrics_rows = []

    for target_col, series_name in SERIES.items():
        # ---- Forecast horizon (12 months) ----
        try:
            f_df = forecast_horizon_months(df, target_col, CFG)  # returns Date, ForecastValue
        except Exception as e:
            print(f"[Forecast] Skipping {series_name}: {e}")
            continue

        f_df = f_df.copy()
        f_df["run_timestamp"] = run_ts
        f_df["series"] = series_name
        f_df["model_name"] = CFG.model_name
        f_df["horizon_months"] = CFG.horizon_months
        forecast_rows.extend(f_df.to_dict("records"))

        # ---- Walk-forward backtest metrics ----
        try:
            summary, _per_step = walk_forward_backtest(
                df=df,
                target_col=target_col,
                cfg=CFG,
                min_train_size=MIN_TRAIN_SIZE,
            )
        except Exception as e:
            print(f"[Backtest] Skipping metrics for {series_name}: {e}")
            continue

        for metric_name, metric_val in summary.items():
            metrics_rows.append({
                "run_timestamp": run_ts,
                "series": series_name,
                "metric": metric_name,
                "value": metric_val,
                "model_name": CFG.model_name,
                "horizon_months": CFG.horizon_months,
            })

    # -------------------------------------------------
    # Build outputs
    # -------------------------------------------------
    forecast_df = pd.DataFrame(forecast_rows)
    metrics_df = pd.DataFrame(metrics_rows)

    # Rename columns for Power BI friendliness (contract)
    if not forecast_df.empty:
        forecast_df = forecast_df.rename(columns={
            "series": "Series",
            "model_name": "ModelName",
            "horizon_months": "HorizonMonths",
        })

    if not metrics_df.empty:
        metrics_df = metrics_df.rename(columns={
            "series": "Series",
            "metric": "Metric",
            "value": "MetricValue",
            "model_name": "ModelName",
            "horizon_months": "HorizonMonths",
        })

    # -------------------------------------------------
    # De-dupe (safety)
    # -------------------------------------------------
    if not forecast_df.empty:
        forecast_df["Date"] = pd.to_datetime(forecast_df["Date"], errors="coerce")
        forecast_df = (
            forecast_df.drop_duplicates(
                subset=["run_timestamp", "Series", "ModelName", "Date"],
                keep="last"
            )
            .sort_values(["Series", "Date"])
            .reset_index(drop=True)
        )

    if not metrics_df.empty:
        metrics_df = (
            metrics_df.drop_duplicates(
                subset=["run_timestamp", "Series", "ModelName", "Metric"],
                keep="last"
            )
            .sort_values(["Series", "Metric"])
            .reset_index(drop=True)
        )

    # Write CSVs
    FORECAST_OUT.parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(FORECAST_OUT, index=False)
    metrics_df.to_csv(METRICS_OUT, index=False)

    # Print previews
    print("Wrote:", FORECAST_OUT)
    print("Wrote:", METRICS_OUT)

    print("\nForecast preview:")
    if forecast_df.empty:
        print("(empty)")
    else:
        print(forecast_df.head(30).to_string(index=False))

    print("\nMetrics preview:")
    if metrics_df.empty:
        print("(empty)")
    else:
        print(metrics_df.to_string(index=False))


# -------------------------------------------------
if __name__ == "__main__":
    main()
