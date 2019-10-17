from sklearn.model_selection import StratifiedKFold
import numpy as np
from common.log import LogHelper
import copy

class KFoldEnsemble:
    def __init__(self, base_model, evaler, nfold=5, seed=0, nni_log=False):
        self.model_list = []
        self.base_model = base_model
        self.evaler = evaler
        self.nfold = nfold
        self.seed = seed
        self.logger = LogHelper(nni_log)

    def fit(self, data):
        X = data.x
        y = data.y.reshape((-1, 1))
        print("[KFold Ensemble Input] X: ", X.shape, "y: ", y.shape)

        train_pred_summary = np.zeros((X.shape[0], 1))

        sfolder = StratifiedKFold(n_splits=self.nfold, random_state=self.seed, shuffle=True)
        for idx, (train_idx, vali_idx) in enumerate(sfolder.split(X, y)):
            data.split(train_idx, vali_idx)

            self.base_model.train(data)
            self.model_list.append(copy.deepcopy(self.base_model))

            train_pred = self.base_model.predict_prob(data.train_x)
            vali_pred = self.base_model.predict_prob(data.val_x)

            train_loss = self.evaler.eval(data.train_y, train_pred)
            vali_loss = self.evaler.eval(data.val_y, vali_pred)

            train_pred_summary[vali_idx] = vali_pred.reshape((-1, 1))

            self.logger.kfold_single(idx, train_loss, vali_loss)

        summary_loss = self.evaler.eval(y, train_pred_summary)
        self.logger.kfold_summary(summary_loss)

    def predict(self, test_data):
        assert (len(self.model_list) > 0)

        test_pred_summary = np.zeros((test_data.shape[0], len(self.model_list)))
        for idx, m in enumerate(self.model_list):
            test_pred_summary[:, idx] = m.predict_prob(test_data)

        return np.mean(test_pred_summary, axis=1)