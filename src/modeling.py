from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


FEATURE_COLS = ["lag_1", "lag_2", "roll_3_mean", "month"]


@dataclass
class ModelConfig:
    model_name: str = "random_forest_lags_rollmean_v1"
    n_estimators: int = 100
    random_state: int = 42
    horizon_months: int = 12


def make_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Input df must have columns: ["date", target_col]
    Output adds lag/rolling/month features.
    """
    d = df.copy()
    d["lag_1"] = d[target_col].shift(1)
    d["lag_2"] = d[target_col].shift(2)
    d["roll_3_mean"] = d[target_col].rolling(3).mean().shift(1)
    d["month"] = d["date"].dt.month
    return d


def train_rf(d_feat: pd.DataFrame, target_col: str, cfg: ModelConfig) -> RandomForestRegressor:
    """
    Train RandomForest on feature rows (assumes no NaNs in required columns).
    Speed rec:
      - n_jobs=-1 uses all cores
      - smaller n_estimators is fine for a portfolio project
    """
    X = d_feat[FEATURE_COLS]
    y = d_feat[target_col]

    model = RandomForestRegressor(
        n_estimators=cfg.n_estimators,
        random_state=cfg.random_state,
        n_jobs=-1,  # SPEED
    )
    model.fit(X, y)
    return model


def _rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(mean_squared_error(y_true, y_pred) ** 0.5)


def walk_forward_backtest(
    df: pd.DataFrame,
    target_col: str,
    cfg: ModelConfig,
    min_train_size: int = 120,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Walk-forward (forward chaining) validation.

    For each time step t:
      - Train on data up to t-1
      - Predict at t
    Collect predictions to compute out-of-sample metrics.

    Returns:
      (summary_metrics, per_step_df)
    per_step_df columns:
      date, y_true, y_pred, error, abs_error, ape_pct
    """
    sub = df[["date", target_col]].dropna().sort_values("date").reset_index(drop=True)

    if len(sub) < max(min_train_size, 12):
        raise ValueError(f"Not enough data for walk-forward backtest: {len(sub)} rows")

    rows: List[dict] = []

    for t in range(min_train_size, len(sub)):
        train_slice = sub.iloc[:t].copy()
        test_row = sub.iloc[t:t + 1].copy()

        train_feat = make_features(train_slice, target_col).dropna().reset_index(drop=True)
        if len(train_feat) < 20:
            continue

        model = train_rf(train_feat, target_col, cfg)

        combo = pd.concat([train_slice, test_row], ignore_index=True)
        combo_feat = make_features(combo, target_col)
        x_row = combo_feat.iloc[[-1]][FEATURE_COLS]

        if x_row.isna().any(axis=1).iloc[0]:
            continue

        y_pred = float(model.predict(x_row)[0])
        y_true = float(test_row[target_col].iloc[0])
        dt = pd.Timestamp(test_row["date"].iloc[0])

        err = y_true - y_pred
        abs_err = abs(err)
        ape = (abs_err / y_true * 100) if y_true != 0 else np.nan

        rows.append({
            "date": dt,
            "y_true": y_true,
            "y_pred": y_pred,
            "error": err,
            "abs_error": abs_err,
            "ape_pct": ape,
        })

    per_step = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if per_step.empty:
        raise ValueError("Walk-forward produced no rows (likely due to feature NaNs).")

    summary = {
        "MAE": float(mean_absolute_error(per_step["y_true"], per_step["y_pred"])),
        "RMSE": _rmse(per_step["y_true"], per_step["y_pred"].to_numpy()),
        "MAPE_pct": float(np.nanmean(per_step["ape_pct"])),
        "BacktestPoints": float(len(per_step)),
    }
    return summary, per_step


def forecast_horizon_months(
    df: pd.DataFrame,
    target_col: str,
    cfg: ModelConfig,
) -> pd.DataFrame:
    """
    Train on all available data (with feature rows) and forecast next cfg.horizon_months months.

    Returns a DataFrame with:
      Date (month-start), ForecastValue
    """
    sub = df[["date", target_col]].dropna().sort_values("date").reset_index(drop=True)
    if len(sub) < 12:
        raise ValueError(f"Not enough data to forecast: {len(sub)} rows")

    feat = make_features(sub, target_col).dropna().reset_index(drop=True)
    if len(feat) < 20:
        raise ValueError("Not enough feature rows to train model.")

    model = train_rf(feat, target_col, cfg)

    # Recursive multi-step forecast:
    # Append predictions month by month so future lags/rolling use predicted values.
    history = sub.copy()

    out_rows: List[dict] = []
    last_date = history["date"].max()
    next_date = (pd.Timestamp(last_date) + pd.offsets.MonthBegin(1)).normalize()

    for _ in range(cfg.horizon_months):
        tmp = pd.concat(
            [history, pd.DataFrame([{"date": next_date, target_col: np.nan}])],
            ignore_index=True
        )
        tmp_feat = make_features(tmp, target_col)
        x_next = tmp_feat.iloc[[-1]][FEATURE_COLS]

        if x_next.isna().any(axis=1).iloc[0]:
            raise ValueError("Feature row for forecast step has NaNs; insufficient history.")

        yhat = float(model.predict(x_next)[0])

        out_rows.append({
            "Date": next_date.strftime("%Y-%m-%d"),
            "ForecastValue": yhat,
        })

        history = pd.concat(
            [history, pd.DataFrame([{"date": next_date, target_col: yhat}])],
            ignore_index=True
        )

        next_date = (pd.Timestamp(next_date) + pd.offsets.MonthBegin(1)).normalize()

    return pd.DataFrame(out_rows)
