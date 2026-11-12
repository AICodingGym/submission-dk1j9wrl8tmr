"""Quick baseline for the Ventilator Pressure Prediction competition.

Run from the repository root:
    python solution.py

Writes data/ventilator-pressure-prediction/predictions.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


DATA = Path(__file__).parent / "data" / "ventilator-pressure-prediction"
TARGET = "pressure"


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add within-breath history features available at prediction time."""
    df = frame.sort_values(["breath_id", "time_step"]).copy()
    groups = df.groupby("breath_id", sort=False)
    df["breath_pos"] = groups.cumcount()
    df["cum_u_in"] = groups["u_in"].cumsum()

    for col in ("u_in", "u_out", "time_step"):
        for lag in (1, 2, 3):
            df[f"{col}_lag{lag}"] = groups[col].shift(lag)
    df["u_in_lag1"] = df["u_in_lag1"].fillna(df["u_in"])
    df["u_in_lag2"] = df["u_in_lag2"].fillna(df["u_in_lag1"])
    df["u_in_lag3"] = df["u_in_lag3"].fillna(df["u_in_lag2"])
    df["u_out_lag1"] = df["u_out_lag1"].fillna(0)
    df["u_out_lag2"] = df["u_out_lag2"].fillna(0)
    df["u_out_lag3"] = df["u_out_lag3"].fillna(0)
    df["dt"] = df["time_step"] - df["time_step_lag1"]
    df["dt"] = df["dt"].fillna(0)

    # A short moving average smooths the noisy valve signal while retaining
    # the sharp transitions that determine pressure.
    df["u_in_ma5"] = groups["u_in"].transform(
        lambda s: s.rolling(5, min_periods=1).mean()
    )
    df["u_in_ma15"] = groups["u_in"].transform(
        lambda s: s.rolling(15, min_periods=1).mean()
    )
    return df


def main() -> None:
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    train_f = make_features(train)
    test_f = make_features(test)

    features = [
        "R", "C", "time_step", "u_in", "u_out", "breath_pos", "cum_u_in",
        "u_in_lag1", "u_in_lag2", "u_in_lag3", "u_out_lag1", "u_out_lag2",
        "u_out_lag3", "time_step_lag1", "time_step_lag2", "time_step_lag3",
        "dt", "u_in_ma5", "u_in_ma15",
    ]

    # Sampling keeps this quick and memory-friendly; set to None to use all rows.
    sample_size = 500_000
    if sample_size and len(train_f) > sample_size:
        rng = np.random.default_rng(42)
        chosen = rng.choice(len(train_f), size=sample_size, replace=False)
        fit_x = train_f.iloc[chosen][features]
        fit_y = train_f.iloc[chosen][TARGET]
    else:
        fit_x, fit_y = train_f[features], train_f[TARGET]

    model = HistGradientBoostingRegressor(
        max_iter=160,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        random_state=42,
    )
    model.fit(fit_x, fit_y)
    predictions = model.predict(test_f[features])

    submission = pd.DataFrame({"id": test["id"], "pressure": predictions})
    output = DATA / "predictions.csv"
    submission.to_csv(output, index=False)
    print(f"Wrote {output} ({len(submission):,} rows)")


if __name__ == "__main__":
    main()
