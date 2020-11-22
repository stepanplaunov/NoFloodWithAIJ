import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgb
import sys

from datetime import datetime, timedelta

from rdm_helpers import get_meteo, helper_save, split_df_by, na_heat_map, interpolate_df, nse
import rdm_helpers


class GBForecastModel:
    INAPPROPRIATE_LEN = "The number of parameters passed does not correspond with a number of models"

    def __init__(self, n=-1, models=[]):
        if n == -1:
            self.models = models
        else:
            self.models = []
            for i in range(0, n):
                self.models.append(lgb.LGBMRegressor())

    def set_params(self, parameters):
        if len(parameters) == len(self.models):
            for i in range(0, len(self.models)):
                self.models[i].set_params(**(parameters[i]))
        else:
            print(self.INAPPROPRIATE_LEN, file=sys.stderr)

    def fit(self, list_x_train, list_y_train, list_x_test, list_y_test):
        for model, x_train, x_test, y_train, y_test in zip(self.models, list_x_train, list_x_test, list_y_train, list_y_test):
            model.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=True)

    def predict(self, list_x_test):
        predictions = np.empty((0, len(list_x_test[0])), float)
        for model, x_test in zip(self.models, list_x_test):
            predictions = np.concatenate((predictions, np.array([model.predict(x_test)])))
        return predictions.T
