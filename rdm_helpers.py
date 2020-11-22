import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def split_df_by(df, col):
    return df.groupby(col)

def helper_save(lst, path, name, index = 'date'):
    for grp, df in lst:
        df.set_index('date')
        df.to_csv(path + '0' + str(grp) + name)

def get_meteo(station_id, data_path):
    x = pd.read_pickle(data_path + "processed_data/s2m.pkl")
    return pd.read_csv(data_path + 'meteo/' + str(int(x.loc[station_id]['meteo_id'])) + '.csv', sep = ';', index_col = 'time'), x.loc[station_id]['dist']

def na_heat_map(df, figsize=(20,12), cmap='viridis'):
    fig, ax = plt.subplots(figsize=figsize)
    sns_heatmap = sns.heatmap(df.isna(), yticklabels = False, cbar = False, cmap = cmap)
    
def interpolate_df(df, method = 'time', cols = []):
    if len(cols) == 0:
        return df.interpolate(method = method)
    else:
        res = df.copy()
        res[cols] =  res[cols].interpolate(method = method)
        return res
    
def nse(simulations, evaluation):
    nse = 1 - (np.sum((evaluation - simulations) ** 2, axis=0, dtype=np.float64)
        / np.sum((evaluation - np.mean(evaluation)) ** 2, dtype=np.float64))
    return nse

def plot_time_series(df, grouping_field, gf_vals, cols, start_date, end_date):
    _df = df.copy()
    if start_date != None and end_date != None:
        _df = _df.loc[start_date : end_date]
    elif start_date == None:
        _df = _df.loc[ : end_date]
    elif end_date == None:
        _df = _df.loc[start_date : ]
    
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (20,10)) 
    for key, grp in _df[_df[grouping_field].isin(gf_vals)].loc[start_date:end_date].groupby(grouping_field): 
        for col in cols:
            ax.plot(grp.index, grp[col], label = key)
            
def plot_series(df, cols, start_date = None, end_date = None, figsize = (20,10), **plot_params):
    _df = df.copy()
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = figsize)
    if start_date != None and end_date != None:
        _df = _df.loc[start_date : end_date]
    elif start_date == None:
        _df = _df.loc[ : end_date]
    elif end_date == None:
        _df = _df.loc[start_date : ]
    
    for col in cols:
        ax.plot(_df.index, _df[col], *plot_params, label = col)     
        
class TimeSeriesExtracter():
    def __init__(self, x_cols, deltas, lags):
        self.x_cols = x_cols
        self.deltas = deltas
        self.lags = lags
        pass
    
    def fit(self, df, **fit_params):
        self.df = df.copy()
        return self
    
    def transform(self, **transform_params):
        for x, d, lag, in zip(self.x_cols, self.deltas, self.lags):
            for i in range(d, d + lag + 1):
                self.df[x + "_t_" + str(i)] = self.df[x].shift(i)
                
        return self.df
    
    def fit_transform(self, df):
        return self.fit(df).transform()
    
def substract_seasonality(series, period):
    return series - sm.tsa.seasonal_decompose(series, period = period).seasonal

def extract_seasonality(series, period):
    return sm.tsa.seasonal_decompose(series, period = period).seasonal
    
class RiverNodeModel:
    def __load__(self):
        self.hydro_df = pd.read_csv(self.data_path + 'hydro/0' + str(self.post_id) + '_daily.csv', sep=';', engine='python')
        self.ice_df = pd.read_csv(self.data_path + 'hydro/0' + str(self.post_id) + '_ice.csv', sep=';', engine='python')
        self.meteo_df = get_meteo(self.post_id, path_to_data)
        try:
            self.disch_df = pd.read_csv(self.data_path + 'hydro/0' + str(self.post_id) + '_disch_d.csv', sep=';', engine='python')
        except FileNotFoundError:
            self.disch_df = None
            
        
        
    def __init__(self, post_id, data_path):
        self.post_id = post_id
        self.data_path = data_path
        self.__load__()