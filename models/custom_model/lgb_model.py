import lightgbm as lgb

class LGB:
    def __init__(self, config):
        self.config = config
        self.model = lgb.LGBMClassifier(**config['param'])

    def train(self, data):
        config = self.config
        self.model.fit(data.train_x, data.train_y, eval_set=[(data.train_x, data.train_y),\
                        (data.val_x, data.val_y)], eval_metric='auc',verbose=config['print_every'])

    def predict_prob(self, data):
        return self.model.predict_proba(data)[:,1]