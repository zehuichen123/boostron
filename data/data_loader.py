import numpy as np
import pandas as pd

class DataLoader():
    def __init__(self, custom_reader, custom_split):
        self.reader = custom_reader
        self.spliter = custom_split
        self.x = None
        self.y = None
        self.test_x = None

    def load(self):
        x, y, test_x = self.reader.load()
        self.x, self.y, self.test_x = x.values, y.values, test_x.values

    def split(self, train_idx, val_idx):
        self.train_x, self.val_x, self.train_y, self.val_y = \
                        self.x[train_idx], self.x[val_idx], self.y[train_idx], self.y[val_idx]

    def get_split_data(self, test_size=0.2, seed=0):
        self.train_x, self.val_x, self.train_y, self.val_y = self.spliter.split(
            self.x, self.y, test_size=test_size, seed=seed)
        return self.train_x, self.val_x, self.train_y, self.val_y
