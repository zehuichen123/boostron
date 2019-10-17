from sklearn.metrics import roc_auc_score

class Evaler:
    def __init__(self):
        pass

    def eval(self, target, pred_y):
        return roc_auc_score(target, pred_y)

    def model_eval(self, model, data, target):
        pred = model.predict_prob(data)
        return roc_auc_score(target, pred)