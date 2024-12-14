
from pathlib import Path

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import holidays
from geopy.distance import geodesic



def _merge_weather_data(X):
    weather_df = pd.read_csv(Path("external_data") / "external_data.csv", parse_dates=["date"])
    weather_df['date'] = pd.to_datetime(weather_df['date']).astype('datetime64[us]')
    
    #drop columns with >2000 nulls
    weather_df = weather_df.dropna(axis=1, how='all')
    columns_without_2000_non_nulls = weather_df.notnull().sum()[weather_df.notnull().sum() <= 2000].index
    weather_df.drop(columns=columns_without_2000_non_nulls, inplace=True)
    
    #drop dupplicates
    duplicate_dates = weather_df.index[weather_df.index.duplicated()]
    weather_df = weather_df[~weather_df.index.duplicated(keep='first')]

    #interpolate missing data
    weather_df.interpolate(method='linear', inplace=True)

    X = X.copy()
    # When using merge_asof left frame need to be sorted
    X["orig_index"] = np.arange(X.shape[0])
    X = pd.merge_asof(
        X.sort_values("date"), weather_df[['date','rr3','rr12','rr24']].sort_values("date"), on="date", direction ='nearest'
    )
    # Sort back to the original order
    X = X.sort_values("orig_index")
    del X["orig_index"]
    return X

def _merge_school_holiday_data(X):
    df_ext = pd.read_csv(Path("external_data") / "vacances-scolaires.csv", parse_dates=["date"])
    df_ext['date'] = pd.to_datetime(df_ext['date']).astype('datetime64[us]')
    X = X.copy()    
    # When using merge_asof left frame need to be sorted
    X["orig_index"] = np.arange(X.shape[0])
    X = pd.merge_asof(
        X.sort_values("date"), df_ext[['date','vacances_zone_c']].sort_values("date"), on="date", direction ='nearest'
    )
    X['vacances_zone_c'] = X['vacances_zone_c'].astype(int) 
    # Sort back to the original order
    X = X.sort_values("orig_index")
    del X["orig_index"]
    return X

def _merge_external_data(X) : 
     X = X.copy()
     X = _merge_weather_data(X)
     X = _merge_school_holiday_data(X)
     X_final = X.drop(columns=['counter_technical_id', 'coordinates','counter_name','site_name','counter_installation_date'])
     return X_final

