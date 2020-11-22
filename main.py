# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from RiverNodeModel import RiverNodeModel

data_path = './data/'
models_path = './models/'

targets = ('06005', '06022', '06296', '06027',
           '05004', '05012', '05024', '05805')

if __name__ == '__main__':
    nodeModel = RiverNodeModel(5012, data_path, models_path, is_load=True)
    # nodeModel.fit()
    # nodeModel.save_model()
    print(nodeModel.predict('2017-04-01', '2017-04-01'))
