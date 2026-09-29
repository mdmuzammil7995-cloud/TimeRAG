"""
pipeline.py
Shared data-cleaning, feature-engineering, and recursive-forecasting logic
used by both train.py and app.py, so the two never drift out of sync.
"""

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "day_of_week",
    "month",
    "day_of_year",
    "lag_1",
    "lag_7",
    "rolling_mean_7",
    "rolling_mean_30",
]


def load_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean an uploaded/loaded time series dataframe.

    Requires exactly a `date` and `value` column (any others are ignored).
    Parses dates, sorts chronologically, fills small gaps, and flags
    outliers (kept, not dropped - just noted via an `is_outlier` column).
    Raises ValueError with a clear message on anything unrecoverable.
    """
    required = {"date", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input data is missing required column(s): {sorted(missing)}")

    data = df[["date", "value"]].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    if data["date"].isna().all():
        raise ValueError("Could not parse any values in the `date` column as dates.")
    data = data.dropna(subset=["date"])

    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data = data.sort_values("date").reset_index(drop=True)

    # Fill any missing numeric values via forward/backward fill (small gaps only).
    data["value"] = data["value"].ffill().bfill()
    if data["value"].isna().any():
        raise ValueError("The `value` column has no usable numeric data.")

    # Flag (not drop) statistical outliers using a simple z-score rule, so the
    # model can be aware of them without silently losing real data points.
    mean, std = data["value"].mean(), data["value"].std()
    if std and not np.isnan(std):
        data["is_outlier"] = (data["value"] - mean).abs() > 3 * std
    else:
        data["is_outlier"] = False

    if len(data) < 10:
        raise ValueError(f"Only {len(data)} valid rows after cleaning - need at least 10.")

    return data


def add_features(df: pd.DataFrame, value_col: str = "value") -> pd.DataFrame:
    """Add calendar, lag, and rolling-window features for the model.

    `value_col` controls which column the lag/rolling features are built
    from. Pass "detrended_value" (see compute_trend / add_detrended_target)
    so the lag/rolling inputs stay on the same bounded, stationary scale as
    a detrended target - otherwise these features keep growing with the
    series' trend just like the raw value would, and the tree model still
    ends up extrapolating past values it has ever seen.
    """
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear

    df["lag_1"] = df[value_col].shift(1)
    df["lag_7"] = df[value_col].shift(7)
    df["rolling_mean_7"] = df[value_col].shift(1).rolling(window=7, min_periods=1).mean()
    df["rolling_mean_30"] = df[value_col].shift(1).rolling(window=30, min_periods=1).mean()

    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].bfill().fillna(0)
    return df


def compute_trend(df: pd.DataFrame) -> tuple[float, float]:
    """Fit a simple linear trend (value ~ date) to the series.

    XGBoost (and tree models generally) cannot extrapolate beyond the value
    range they saw in training - on a strongly trending series this makes
    predictions systematically lag behind, hurting holdout R^2. Removing a
    linear trend before training, then adding it back after predicting,
    lets the tree model focus on the bounded residual pattern instead of
    the ever-growing absolute level.
    """
    x = df["date"].map(pd.Timestamp.toordinal).to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, df["value"].to_numpy(dtype=float), 1)
    return float(slope), float(intercept)


def trend_at(dates: pd.Series, slope: float, intercept: float) -> pd.Series:
    """Evaluate the fitted linear trend at the given dates."""
    x = dates.map(pd.Timestamp.toordinal).astype(float)
    return slope * x + intercept


def add_detrended_target(df: pd.DataFrame, slope: float, intercept: float) -> pd.DataFrame:
    """Add a `detrended_value` column: value minus the fitted linear trend."""
    df = df.copy()
    df["detrended_value"] = df["value"] - trend_at(df["date"], slope, intercept)
    return df


def infer_step(dates: pd.Series) -> pd.Timedelta:
    """Infer the typical spacing between consecutive timestamps."""
    diffs = dates.sort_values().diff().dropna()
    if diffs.empty:
        return pd.Timedelta(days=1)
    return diffs.mode().iloc[0]


def recursive_forecast(model_bundle: dict, history_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Forecast `horizon` future steps one at a time.

    `model_bundle` is a dict with keys "model", "trend_slope", "trend_intercept"
    (see train.py). The tree model predicts the detrended residual at each
    step; the fitted linear trend is added back to get the actual forecast
    value. This lets the model extrapolate a trending series sensibly,
    since the tree itself only ever predicts the bounded residual.

    Each prediction is fed back in as history so the next step's lag /
    rolling features reflect it. `history_df` must already be cleaned
    (i.e. the output of `load_and_clean`) with `date` and `value` columns.
    """
    model = model_bundle["model"]
    slope = model_bundle["trend_slope"]
    intercept = model_bundle["trend_intercept"]

    working = history_df[["date", "value"]].copy()
    step = infer_step(working["date"])
    future_rows = []

    for _ in range(horizon):
        next_date = working["date"].iloc[-1] + step
        feat_row = {
            "day_of_week": next_date.dayofweek,
            "month": next_date.month,
            "day_of_year": next_date.dayofyear,
            "lag_1": working["value"].iloc[-1],
            "lag_7": working["value"].iloc[-7] if len(working) >= 7 else working["value"].iloc[0],
            "rolling_mean_7": working["value"].tail(7).mean(),
            "rolling_mean_30": working["value"].tail(30).mean(),
        }
        X = pd.DataFrame([feat_row])[FEATURE_COLUMNS]
        detrended_pred = float(model.predict(X)[0])
        trend_val = slope * next_date.toordinal() + intercept
        pred = detrended_pred + trend_val

        future_rows.append({"date": next_date, "value": pred})
        working = pd.concat(
            [working, pd.DataFrame([{"date": next_date, "value": pred}])],
            ignore_index=True,
        )

    return pd.DataFrame(future_rows)
