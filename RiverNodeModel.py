import pandas as pd
from gbmForecastModel import GBForecastModel
from DataTransformers import *
from sklearn.metrics import mean_absolute_error as mae
from rdnhelpers import get_meteo, nse
import pickle
import datetime
import numpy as np
from datetime import datetime, timedelta

def mat_nse(simulations, evaluation):
    result = np.array(simulations.shape[0], dtype=np.float64)
    for i in range(simulations.shape[0]):
        result[i] = nse(simulations[i, :], evaluation)
    return result.mean()


class RiverNodeModel:
    def __load__(self):
        self.hydro_df = pd.read_csv(self.data_path + 'hydro/0' + str(self.post_id) + '_daily.csv', index_col='date',
                                    engine='python')
        self.meteo_df, _ = get_meteo(self.post_id, './datasets/')

        try:
            self.ice_df = pd.read_csv(self.data_path + 'hydro/0' + str(self.post_id) + '_ice.csv', index_col='date', engine='python')
        except:
            self.ice_df = None

        self.left_bound = self.hydro_df.index.min()
        self.right_bound = self.hydro_df.index.max()

    def __init__(self, post_id, data_path, models_path, is_load=False):
        self.post_id = post_id
        self.data_path = data_path
        self.models_path = models_path
        self.__load__()
        if is_load:
            self.load_model()
        else:
            self.create_model()

    def create_model(self):
        _lag = 10

        parameters = {
            'objective': 'rmse',
            'random_state': 42,
            'metrics': ['l2', 'mae', 'mse'],
            'drop_rate': 0.5,
            'learning_rate': 0.004,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.4,
            #     'bagging_freq': 10,
            "n_estimators": 1500,
            "max_depth": 8,
            "num_leaves": 32,
            "max_bin": 64,
            "n_jobs": 4,
            "verbosity ": -1,
            "early_stopping_rounds": 100,
        }

        self.f_model = GBForecastModel(10)
        self.f_model.set_params([parameters] * _lag)

    def save_model(self):
        with open(f'{self.models_path}f_model_{self.post_id}.pkl', 'wb') as file:
            pickle.dump(self.f_model, file)

    def load_model(self):
        with open(f'{self.models_path}f_model_{self.post_id}.pkl', 'rb') as file:
            self.f_model = pickle.load(file)
        return self

    def fit(self, end_callbacks=None):
        if end_callbacks is None:
            end_callbacks = []

        _hdata = HydroTransformer().fit_transform(self.hydro_df) if self.hydro_df is not None else None
        _mdata = MeteoTransformer().fit_transform(self.meteo_df) if self.meteo_df is not None else None
        _idata = IceTransformer().fit_transform(self.ice_df) if self.ice_df is not None else None

        data = FinalTransformer().fit_transform(_hdata.index, filter(lambda x: x is not None, [_hdata, _mdata, _idata]))

        train_start = '2000-01-01'
        train_end = '2014-12-31'
        test_start = '2015-01-01'
        test_end = '2016-12-30'

        X_trains, y_trains, X_tests, y_tests \
            = train_test_split(data, train_start, train_end, test_start, test_end)

        self.f_model.fit(X_trains, y_trains, X_tests, y_tests)

        frame = {}
        for i, y in enumerate(y_tests):
            frame['%s%d'%("stage_max", i)] = y

        d = pd.DataFrame(frame)

        print('mae: %f'%(mae(self.f_model.predict(X_tests), d.values)))
        # print('nse: %f'%(mat_nse(self.f_model.predict(X_tests), d.values)))

        for cb in end_callbacks:
            cb(self.f_model, (X_trains, y_trains, X_tests, y_tests))
        self.hydro_df = None

        return self

    def predict(self, X_start, X_end, sigma_X=None):
        if sigma_X is not None:
            raise Exception("Not implemented")

        self.__load__()
        _hdata = HydroTransformer().fit_transform(self.hydro_df)[X_start: X_end]
        _mdata = MeteoTransformer().fit_transform(self.meteo_df)[X_start: X_end]
        _idata = IceTransformer().fit_transform(self.ice_df)[X_start: X_end]
        data = FinalTransformer().fit_transform(_hdata.index, [_hdata, _mdata, _idata])

        X_list = [df.drop('target', axis=1) for df in data]
        prediction = self.f_model.predict(X_list)

        return prediction
