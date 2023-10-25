import pickle
import torch
import pytorch_lightning as pl
from src.cancer_sim.cancer_simulation import get_cancer_sim_data
from src.cancer_sim.process_irregular_data import *
from src.cancer_sim.cancer_data_module import CancerDataloader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from src.models.bncde_model import BNCDE
from pytorch_lightning.loggers.mlflow import MLFlowLogger


def bncde_train(hyperparams: dict,
                data_path,
                load_pickle,
                experiment_name,
                run_name,
                informative_sampling=False,
                tags=None,
                log_model=True,
                tracking_uri='sqlite:///mlflow.db',
                device='cuda',
                accelerator='cuda',
                num_workers=0):

    torch.set_default_device("cuda")

    # 1) Instantiate ML flow logger
    mlflow_logger = MLFlowLogger(experiment_name=experiment_name, run_name=run_name,
                                 tracking_uri=tracking_uri, tags=tags,
                                 save_dir='./mlruns', log_model=log_model)
    mlflow_logger.log_hyperparams(hyperparams)

    # 2) Instantiate Bayesian Neural Controlled Differential Equation
    model = BNCDE(mc_samples=hyperparams['mc_samples'],hidden_size=hyperparams['hidden_size'],
                  sd_diffusion=hyperparams['sd_diffusion'], drift_layers=hyperparams['drift_layers'],
                  intensity_weighting=hyperparams['intensity_weighting'],
                  control_size=hyperparams['control_size'], treatment_size=hyperparams['treatment_size'],
                  prediction_window=hyperparams['prediction_window'],
                  interpolation_method=hyperparams['interpolation_method'],
                  learning_rate=hyperparams['learning_rate'], method=hyperparams['method'])

    # 3) Load or generate data
    if load_pickle:
        with open(data_path, 'rb') as f:
            pickle_map = pickle.load(f)

    else:
        if informative_sampling:
            obs_coeff = 2.
        else:
            obs_coeff = 1.

        # Simulate. Param specs adapted from: https://github.com/seedatnabeel/TE-CDE
        args = {"chemo_coeff": 0,  # Not used (confounding)
                "radio_coeff": 0,  # Not used (confounding)
                "obs_coeff": obs_coeff,  # 6.,
                "intensity_cov": 0,  # (outcome unrelated sampling intensity)
                "intensity_cov_only": False,  # True for outcome unrelated sampling
                "max_intensity": 1.,  # Max intensity (1 / S_\lambda in paper)
                "num_patients": 10000,
                "results_dir": "results",
                "model_name": "BTE-CDE",
                "load_dataset": False,  # True to skip data generation
                "use_transformed": False,  # True to skip data transformation
                "experiment": "default",  # Add other experiments as yml files
                "data_path": "../../data/new_cancer_sim.p",
                "kappa": 5,  # Not used
                "max_samples": 1,  # Not used
                "strategy": "all",  # Not used
                "save_raw_datapath": "../../data/raw",
                "save_transformed_datapath": "../../data/transformed"
                }

        # Generate or load raw data -- latent paths of X_t, A_t, Y_t, lambda_t
        pickle_map = get_cancer_sim_data(
            chemo_coeff=args["chemo_coeff"],
            radio_coeff=args["radio_coeff"],
            obs_coeff=args["obs_coeff"],
            intensity_cov=args["intensity_cov"],
            intensity_cov_only=bool(args["intensity_cov_only"]),
            max_intensity=args["max_intensity"],
            num_patients=args["num_patients"],
            b_load=False,
            b_save=False,
            model_root=args["results_dir"],
        )

        # Transformed data (or load) -- apply observation process
        pickle_map = transform_data(
            data=pickle_map,
            interpolate=False,
            strategy=args["strategy"],
            sample_prop=1,
            kappa=args["kappa"],
            max_samples=args["max_samples"],
        )
        # Save new dataset
        with open(data_path, 'wb') as f:
            pickle.dump(pickle_map, f)

    data = CancerDataloader(pickle_map=pickle_map, batch_size=hyperparams['batch_size'],
                            device=device, num_workers=num_workers)


    # 4) Instantiate callback and trainer
    earlystopping_callback = EarlyStopping(monitor="val_loss", mode="min", patience=hyperparams['patience'])

    trainer = pl.Trainer(accelerator=accelerator, min_epochs=1, max_epochs=hyperparams['max_epochs'],
                         gradient_clip_val=hyperparams['clip_grad'],
                         logger=mlflow_logger, devices=[0],
                         callbacks=[earlystopping_callback])

    # 5) Fit model
    trainer.fit(model, data)
