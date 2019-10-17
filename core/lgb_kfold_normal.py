from data.custom_reader.credit_reader import Reader
from data.custom_spliter.normal_spliter import Spliter
from data.data_loader import DataLoader

from models.base_model import Model
from models.custom_model.lgb_model import LGB

from methods.kfold import KFoldEnsemble

from eval.custom_evaler.auc_evaler import Evaler

def main(config):
    # load data
    custom_reader = Reader('demo', 'train.pkl', 'train_target.pkl')
    custom_spliter = Spliter()
    data = DataLoader(custom_reader, custom_spliter)

    # initialize model
    lgb_custom = LGB(config)
    base_model = Model(lgb_custom)

    # initialize metric
    evaler = Evaler()

    # intialize method
    kfoldEnsemble = KFoldEnsemble(base_model=base_model, evaler=evaler, nfold=5, seed=0, nni_log=False)

    # start training
    kfoldEnsemble.fit(data)

if __name__ == '__main__':
    config = {

    }
    main(config)