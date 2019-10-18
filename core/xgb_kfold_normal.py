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
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'n_estimators': 1000,
        'max_depth': 5,
        'learning_rate': 0.01,
        # 'subsample': 0.7,
        # 'colsample_bytree': 0.7,
        'random_state': 0,
        'tree_method': 'gpu_hist', 
        'gpu_id': 0,
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

kfoldEnsemble = KFoldEnsemble(base_model=base_model, evaler=evaler, nfold=5, seed=ii, nni_log=False)
kfoldEnsemble.fit(data)

# initialize submitter
submitter = Submitter(submit_file_path='../demo/credit_data/submit.csv', save_path='../demo', file_name='xgb_base.csv')

# submit your prediction
submitter.submit([kfoldEnsemble], data)