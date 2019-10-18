# Boostron -- A Simple and Efficient Framework for Classification and Regression

> Start your tabular data competition quickly!

## Major Features
- XGBoost, LightGBM base models (MLP, DAE, Catboost, FM, FFM to be added)
- Support Multi processes training
- Visitor Design Pattern, allows you to customize your whole framework efficiently
- NNI AutoML for search parameters (to be added)
- Feature Visualization, powered by Facets (to be added)

## How to start
### 1. Customize dataset reader
Create `new_reader.py` at `data/custom_reader` to customize the way to load dataset. Refer: `data/custom_reader/credit_reader.py`. You may peform feature engineering here.
### 2. Customize dataset spliter
Create `new_spliter.py` at `data/custom_spliter` to customize the way to split dataset. Refer: `data/custom_spliter/normal_spliter.py`.
### 3. Customize your ML model
You may directly use `XGBoost` or `LightGBM` from this library. If you want  to design your own model, please implement `train()` and `predict_prob()`.
### 4. Customize your ML training ways
You may directly use `KFoldEnsembles` from this library. If you want to customize your own way, like undersampling, please implement `fit()` and `predict()` according to `methods/kfold.py`
### 5. Customize your eval metrics
You may directly use `auc_evaler` from this library. If you want to customize your own way, like undersampling, please implement `eval()` and `model_eval()` according to `eval/custom_evaler/auc_evaler.py`
### 6. Customize your submit metrics
Create `new_submitter.py` at `submit/custom_submitter` to customize the way to split dataset. Refer: `submit/custom_submitter/credit_submitter.py`.
### 7. Start Training
Here is an example code on how to ensembles all these modules together.
```python
# import what you need
from data.custom_reader.credit_reader import Reader
from data.custom_spliter.normal_spliter import Spliter
from data.data_loader import DataLoader

from models.base_model import Model
from models.custom_model.xgb_model import XGB

from methods.kfold import KFoldEnsemble

from eval.custom_evaler.auc_evaler import Evaler
from submit.custom_submitter.credit_submitter import Submitter

# custom config for model
config = {
    "print_every": 50,
    "param": {
        ...
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

# training model
kfoldEnsemble.fit(data)

# initialize submitter
submitter = Submitter(submit_file_path='../demo/credit_data/submit.csv', save_path='../demo', file_name='xgb_base.csv')

# submit your prediction
submitter.submit([kfoldEnsemble], data)
```
For more example codes you may refer to `core/`.


