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

from concurrent.futures import ProcessPoolExecutor

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

def single_run(index):
    custom_reader = Reader('../demo/credit_data', 'train.pkl', 'train_target.pkl', 'test.pkl')
    custom_spliter = Spliter()
    data = DataLoader(custom_reader, custom_spliter)
    data.load()

    config['param']['gpu_id'] = index

    xgb_custom = XGB(config)
    base_model = Model(xgb_custom)

    evaler = Evaler()

    print("[KFold Time] Num: %d" % (index+1))
    kfoldEnsemble = KFoldEnsemble(base_model=base_model, evaler=evaler, nfold=5, seed=index, nni_log=False)
    kfoldEnsemble.fit(data)

    return kfoldEnsemble

sum_res = 0
kfold_time = 5
index_list = [i for i in range(kfold_time)]
model_list = []
with ProcessPoolExecutor(max_workers=kfold_time) as executor:
    for index, kfold_model in enumerate(executor.map(single_run, index_list)):
        model_list.append(kfold_model)
        sum_res += kfold_model.eval_res
# start training
print("[Overall Summary] Train Loss: %g" % (sum_res/kfold_time))

custom_reader = Reader('../demo/credit_data', 'train.pkl', 'train_target.pkl', 'test.pkl')
custom_spliter = Spliter()
data = DataLoader(custom_reader, custom_spliter)
data.load()
# initialize submitter
submitter = Submitter(submit_file_path='../demo/credit_data/submit.csv', save_path='../demo', file_name='xgb_base.csv')

# submit your prediction
submitter.submit(model_list, data)