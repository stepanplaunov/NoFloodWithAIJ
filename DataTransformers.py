import statsmodels.api as sm
import pandas as pd
from rdm_helpers import *


def train_test_split(dframes, train_start, train_end, test_start, test_end):
    X_trains, X_tests, y_trains, y_tests = ([], [], [], [])
    for df in dframes:
        y = df['target']
        x = df.drop(['target'], axis=1)

        X_trains.append(x[train_start:train_end])
        X_tests.append(x[test_start:test_end])
        y_trains.append(y[train_start:train_end])
        y_tests.append(y[test_start:test_end])

    return X_trains, y_trains, X_tests, y_tests


class FinalTransformer:
    def __init__(self):
        self.is_fitted = False

    def fit(self, index, dframes):
        self.df = pd.concat([df.reindex(index) for df in dframes], axis=1)
        self.is_fitted = True
        return self

    def transform(self):
        if not self.is_fitted:
            raise Exception("not fitted")
        _lag = 10

        dframes = []
        for i in range(1, _lag + 1):
            df = self.df.copy()
            df["target"] = self.df['stage_max'].shift(-i)
            df["doy"] = self.df['doy'].shift(-i)
            df["seasonality"] = self.df['seasonality'].shift(-i)
            df.drop('stage_max', inplace=True, axis=1)
            dframes.append(df)

        return dframes

    def fit_transform(self, index, df):
        return self.fit(index, df).transform()


class HydroTransformer:
    def __init__(self):
        self.is_fitted = False

    def fit(self, df):
        self.df = df.copy()
        self.is_fitted = True
        return self

    def transform(self):
        if not self.is_fitted:
            raise Exception("not fitted")
        _lag = 10

        self.df.index = pd.to_datetime(self.df.index)
        self.df = interpolate_df(self.df, cols=['temp', 'stage_max'])
        self.df['stage_delta'] = (self.df['stage_max'] - self.df['stage_min'])
        self.df = self.df.drop(['water_code', 'station_id', 'stage_min', 'stage_avg'], axis=1)
        self.df['doy'] = self.df.index.dayofyear
        self.df['seasonality'] = sm.tsa.seasonal_decompose(self.df['stage_max'], period=365).seasonal
        self.df['stage_max_ma_2'] = self.df['stage_max'].shift(1).rolling(2).apply(lambda x: x[0] - x[1])

        ts_features = ['stage_delta', 'temp', 'stage_max']
        self.df = TimeSeriesExtracter(ts_features,
                                      [0, 0, 0],
                                      [4, 4, 10]).fit_transform(self.df)

        return self.df

    def fit_transform(self, df):
        return self.fit(df).transform()


class MeteoTransformer:
    def __init__(self):
        self.is_fitted = False

    def fit(self, df):
        self.df = df.copy()
        self.is_fitted = True
        return self

    def transform(self):
        if not self.is_fitted:
            raise Exception("not fitted")
        _lag = 10

        self.df.index = pd.to_datetime(self.df.index)
        variebles = ['humidity',
                     'temperature_air',
                     'temperature_ground',
                     'precipitation_amount',
                     'wind_speed_max',
                     'wind_direction']

        for var in variebles:
            self.df.loc[self.df[var + '_quality'] > 2, var] = np.nan

        self.df = self.df.resample('D') \
            .agg({
            'precipitation_amount': 'sum',
            'temperature_air': 'mean',
            'temperature_ground': 'mean',
            'humidity': 'mean'
        })

        self.df['humidity'] = self.df['humidity'].shift(7).rolling(10).sum()
        self.df['humidity'] = self.df['humidity'].shift(-max(7, _lag))

        self.df = TimeSeriesExtracter(['precipitation_amount', 'temperature_ground'],
                                      [0, 0],
                                      [10, 8]).fit_transform(self.df)

        return self.df

    def fit_transform(self, df):
        return self.fit(df).transform()


class IceTransformer:
    def __init__(self):
        self.is_fitted = False

    def fit(self, df):
        self.df = df.copy()
        self.is_fitted = True
        return self

    def transform(self):
        if not self.is_fitted:
            raise Exception("not fitted")
        _lag = 10

        idx = pd.date_range('1984-01-01', '2019-01-04')
        ice_df = self.df.groupby('date').mean()

        ice_df.index = pd.DatetimeIndex(ice_df.index)
        ice_df = ice_df.drop(["place", 'station_id'], axis=1)
        ice_df = ice_df.reindex(idx, fill_value=np.nan)
        ice_df = interpolate_df(ice_df, cols=['ice_thickness', 'snow_height'])

        ice_df.loc[ice_df.index.month.isin(range(4, 11))] = 0

        back = 30
        period = 270

        ice_df[['snow_height_ma', 'ice_thickness_ma']] = ice_df[['snow_height', 'ice_thickness']].shift(back).rolling(
            period).sum()

        return ice_df[['snow_height_ma', 'ice_thickness_ma']]

    def fit_transform(self, df):
        return self.fit(df).transform()
