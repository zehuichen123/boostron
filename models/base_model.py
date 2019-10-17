import numpy as np

class Model:
    def __init__(self, custom_model):
        self.model = custom_model

    def train(self, data):
        self.model.train(data)

    def predict_prob(self, pred_data):
        return self.model.predict_prob(pred_data)

    def measure(self, evaler, data):
        metric_eval = evaler.eval(self.model, data)
        return metric_eval


