import sys
import warnings
warnings.simplefilter('ignore')
sys.path.append('..')

from data.custom_reader.credit_reader import Reader
from data.custom_spliter.normal_spliter import Spliter
from data.data_loader import DataLoader

from models.base_model import Model
from models.custom_model.xgb_model import XGB

from methods.kfold import KFoldEnsemble

from eval.custom_evaler.auc_evaler import Evaler
from submit.custom_submitter.credit_submitter import Submitter

config = {
    "print_every": 50,
    "param": {
        'learning_rate': 0.05,
        'max_depth': 9,
        'num_leaves': 24,
        'feature_fraction': 0.6,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'objective': 'rank:pairwise',
        'save_binary': True,
        'metric': 'auc',
        'verbose': -1,
        'seed': 0,
        'num_iterations': 300,
        # 'subsample_freq': 5,
        'early_stopping_round' : 60,
        # 'num_threads': 12,
    }
}

# load data
custom_reader = Reader('../demo/credit_data', 'train.pkl', 'train_target.pkl', 'test.pkl')
custom_spliter = Spliter()
data = DataLoader(custom_reader, custom_spliter)
data.load()

# initialize model
lgb_custom = XGB(config)
base_model = Model(lgb_custom)

# initialize metric
evaler = Evaler()

# intialize method
kfoldEnsemble = KFoldEnsemble(base_model=base_model, evaler=evaler, nfold=5, seed=0, nni_log=False)

# start training
kfoldEnsemble.fit(data)

# initialize submitter
submitter = Submitter(submit_file_path='../demo/credit_data/submit.csv', save_path='../demo', file_name='lgb_base.csv')

# submit your prediction
submitter.submit(kfoldEnsemble, data)