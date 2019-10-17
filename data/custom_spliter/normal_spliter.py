import numpy as np
from sklearn.model_selection import train_test_split

class Spliter:
    def __init__(self):
        pass

    def split(self, data_x, data_y, test_size, seed):
        return train_test_split(data_x, data_y, test_size=test_size, seed=seed)