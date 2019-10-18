import pandas as pd
import os

class Reader:
    def __init__(self, folder_name, data_name, test_name):
        self.folder_name = folder_name
        self.data_name = data_name
        self.test_name = test_name

    def load(self):
        x = pd.read_csv(os.path.join(self.folder_name, self.data_name))
        y = pd.read_pickle('../demo/credit_data/train_target.pkl')
        test_x = pd.read_csv(os.path.join(self.folder_name, self.test_name))
        return x, y['target'], test_x