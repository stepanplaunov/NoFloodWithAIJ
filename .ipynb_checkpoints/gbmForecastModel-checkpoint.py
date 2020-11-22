import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgb

from datetime import datetime, timedelta

from rdm_helpers import get_meteo, helper_save, split_df_by, na_heat_map, interpolate_df, nse
import rdm_helpers

class GBForecastModel():
    INAPPROPRIATE_LEN="The number of parameters passed does not correspond with a number of models"
    
    def __init__(self,n=-1,moedls=[]):
        if n==-1:
            self.models=models
        else:                
            self.models=[]
            for i in range(0,n):
                self.models.append(lgb.LGBMRegressor())
    
    def set_params(self,parameters):
        if len(parameters)==len(self.models):
            for i in range(0,len(self.models)):
                self.models[i].set_params(**(parameters[i]))
        else:
            print(self.INAPPROPRIATE_LEN,file=sys.stderr)
    
    def fit(self,x_train,y_train,x_test,y_test,labels):
        for model,label in zip(self.models,labels):
            model.fit(x_train,y_train[label],eval_set=[(x_test,y_test[label])], verbose=False)
    
    def predict(self,x_test):
        predictions=np.empty((0,len(x_test)),float)
        for model in self.models:
            predictions=np.concatenate((predictions,np.array([model.predict(x_test)])))
        return predictions.T
        