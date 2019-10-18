import sys
import warnings
warnings.simplefilter('ignore')
sys.path.append('..')

from data.custom_reader.credit_reader import Reader
from data.custom_spliter.normal_spliter import Spliter
from data.data_loader import DataLoader

from models.base_model import Model
from models.custom_model.lgb_model import LGB

from methods.kfold import KFoldEnsemble

from eval.custom_evaler.auc_evaler import Evaler
from submit.custom_submitter.credit_submitter import Submitter

from concurrent.futures import ProcessPoolExecutor

config = {
    "print_every": 50,
    "param": {
        'learning_rate': 0.01,
        # max_depth': 6,
        # 'num_leaves': 24,
        # 'feature_fraction': 0.7,
        # 'bagging_fraction': 0.7,
        # 'bagging_freq': 1,
        'objective': 'binary',
        'save_binary': True,
        'metric': 'auc',
        'verbose': -1,
        'seed': 0,
        'num_iterations': 1000,
        # 'subsample_freq': 5,
        'early_stopping_round' : 300,
        'num_threads': 10,
    }
}

def single_run(index):
    custom_reader = Reader('../demo/credit_data', 'train.pkl', 'train_target.pkl', 'test.pkl')
    custom_spliter = Spliter()
    data = DataLoader(custom_reader, custom_spliter)
    data.load()

    lgb_custom = LGB(config)
    base_model = Model(lgb_custom)

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
submitter = Submitter(submit_file_path='../demo/credit_data/submit.csv', save_path='../demo', file_name='lgb_base.csv')

# submit your prediction
submitter.submit(model_list, data)