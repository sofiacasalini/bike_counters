
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import holidays
from geopy.distance import geodesic


problem_title = "Bike count prediction"
_target_column_name = "log_bike_count"
# A type (class) which will be used to create wrapper objects for y_pred


def get_cv(X, y, random_state=0):
    cv = TimeSeriesSplit(n_splits=6)
    rng = np.random.RandomState(random_state)

    for train_idx, test_idx in cv.split(X):
        # Take a random sampling on test_idx so it's that samples are not consecutives.
        yield train_idx, rng.choice(test_idx, size=len(test_idx) // 3, replace=False)


def get_train_data(path="data/train.parquet"):
    data = pd.read_parquet(path)
    # Sort by date first, so that time based cross-validation would produce correct results
    data = data.sort_values(["date", "counter_name"])
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name, "bike_count"], axis=1)
    return X_df, y_array


def train_test_split_temporal(X, y, delta_threshold="30 days"):

    cutoff_date = X["date"].max() - pd.Timedelta(delta_threshold)
    mask = X["date"] <= cutoff_date
    X_train, X_valid = X.loc[mask], X.loc[~mask]
    y_train, y_valid = y[mask], y[~mask]

    return X_train, y_train, X_valid, y_valid


holidays = holidays.CountryHoliday("France")


def is_holiday(date):  # 1: holiday, 0: not holiday
    return 1 if date in holidays else 0


def _encode_dates(X):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X["year"] = X["date"].dt.year
    X["quarter"] = X["date"].dt.quarter
    X["month"] = X["date"].dt.month
    X["day"] = X["date"].dt.day
    X["weekday"] = X["date"].dt.weekday
    X["hour"] = X["date"].dt.hour
    X["is_weekend"] = X["weekday"].apply(
        lambda x: 1 if x >= 5 else 0
    )  # 1: weekend, 0: weekday
    X["is_holiday"] = X["date"].apply(is_holiday)

    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["date"])
    # return X


def create_time_features(df):
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


def add_distance_to_most_trafficked(df):
    df = df.copy()
    most_trafficked = df[df["counter_name"] == "Totem 73 boulevard de Sébastopol S-N"]

    most_trafficked_location = (
        most_trafficked["latitude"].iloc[0],
        most_trafficked["longitude"].iloc[0],
    )

    df = df.dropna(subset=["latitude", "longitude"])

    df["distance_to_most_trafficked"] = df.apply(
        lambda row: geodesic(
            (row["latitude"], row["longitude"]), most_trafficked_location
        ).km,
        axis=1,
    )

    return df
