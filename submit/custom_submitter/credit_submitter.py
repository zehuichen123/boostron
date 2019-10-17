import numpy as np
import pandas as pd
import os

class Submitter:
    def __init__(self, submit_file_path, save_path, file_name):
        self.submit_file_path = submit_file_path
        self.save_path = save_path
        self.file_name = file_name

    def submit(self, model, data):
        pred_y = model.predict(data.test_x)
        sample_submission = pd.read_csv(self.submit_file_path)
        sample_submission['target'] = pred_y
        sample_submission.to_csv(os.path.join(self.save_path, self.file_name), index=False)

