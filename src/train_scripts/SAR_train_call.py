import yaml
from src.train_scripts.bncde_train_wrapper import bncde_train
from src.train_scripts.tecde_train_wrapper import tecde_train
from itertools import product
import numpy as np
np.random.seed(0)

# i) First train BNCDE
# Load hyperparams
with open('Code/config/bncde_SAR_hyperparam.yml', 'r') as stream:
    data = yaml.safe_load(stream)
keys, values = zip(*data.items())
values_tolist = [[p] if not isinstance(p, list) else p for p in values]
hyperparam_combinations = [dict(zip(keys, p)) for p in product(*values_tolist)]

# training
for i, hyperparams in enumerate(hyperparam_combinations):
    bncde_train(hyperparams=hyperparams,
                data_path='./data/cancer_sim/cancer_pickle_map_irregular_SAR.pkl',
                load_pickle=False,
                informative_sampling = True,
                experiment_name='SAR_setting',
                run_name='BNCDE_config_normal_data'+str(i+1),
                tags=None,
                log_model=True,
                tracking_uri='http://localhost:5002',
                device='cuda',
                accelerator='cuda',
                num_workers=0)


# ii) Second train TESAR-CDE
# Load hyperparams
with open('Code/config/tecde_SAR_hyperparam.yml', 'r') as stream:
    data = yaml.safe_load(stream)
keys, values = zip(*data.items())
values_tolist = [[p] if not isinstance(p, list) else p for p in values]
hyperparam_combinations = [dict(zip(keys, p)) for p in product(*values_tolist)]

# training
for i, hyperparams in enumerate(hyperparam_combinations):
    tecde_train(hyperparams=hyperparams,
                data_path='./data/cancer_sim/cancer_pickle_map_irregular_SAR.pkl',
                load_pickle=True,
                informative_sampling=True,
                experiment_name='SAR_setting',
                run_name='TECDE_config_normal_data'+str(i+1),
                tags=None,
                log_model=True,
                tracking_uri='http://localhost:5002',
                device='cuda',
                accelerator='cuda',
                num_workers=0)


