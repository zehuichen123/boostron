import pandas as pd
import os

class Reader:
    def __init__(self, folder_name, data_name, target_name, test_name):
        self.folder_name = folder_name
        self.data_name = data_name
        self.target_name = target_name
        self.test_name = test_name

    def load(self):
        x = pd.read_pickle(os.path.join(self.folder_name, self.data_name))
        y = pd.read_pickle(os.path.join(self.folder_name, self.target_name))
        test_x = pd.read_pickle(os.path.join(self.folder_name, self.test_name))
        return x.drop(['id'], axis=1), y['target'], test_x.drop(['id'], axis=1)