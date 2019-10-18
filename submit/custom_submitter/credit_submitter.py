import numpy as np
import pandas as pd
import os

def minmax_normalization(data):
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    return (data - min_val) / (max_val - min_val + 1e-10)

class Submitter:
    def __init__(self, submit_file_path, save_path, file_name):
        self.submit_file_path = submit_file_path
        self.save_path = save_path
        self.file_name = file_name

    def submit(self, model_list, data):
        pred_y_list = []
        for model in model_list:
            pred_y = model.predict(data.test_x)
            pred_y = minmax_normalization(pred_y)
            pred_y_list.append(pred_y.reshape(-1, 1))
        pred_y_arr = np.hstack(pred_y_list)
        pred_y = np.mean(pred_y_arr, axis=1)
        sample_submission = pd.read_csv(self.submit_file_path)
        sample_submission['target'] = pred_y
        sample_submission.to_csv(os.path.join(self.save_path, self.file_name), index=False)

