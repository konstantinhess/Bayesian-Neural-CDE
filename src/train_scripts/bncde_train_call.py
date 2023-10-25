import yaml
from src.train_scripts.bncde_train_wrapper import bncde_train
from itertools import product
import numpy as np
np.random.seed(0)
# Load hyperparams
with open('Code/config/bncde_hyperparam.yml', 'r') as stream:
    data = yaml.safe_load(stream)
keys, values = zip(*data.items())
values_tolist = [[p] if not isinstance(p, list) else p for p in values]
hyperparam_combinations = [dict(zip(keys, p)) for p in product(*values_tolist)]

# training
for i, hyperparams in enumerate(hyperparam_combinations):
    bncde_train(hyperparams=hyperparams,
                data_path='./data/cancer_sim/cancer_pickle_map_irregular.pkl',
                load_pickle=False,
                experiment_name='BNCDE',
                run_name='BNCDE_config_'+str(i+1),
                tags=None,
                log_model=True,
                tracking_uri='http://localhost:5002',
                device='cuda',
                accelerator='cuda',
                num_workers=0)


