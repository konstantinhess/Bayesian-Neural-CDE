# Adapted from: https://github.com/toonvds/TESAR-CDE

from src.cancer_sim.cancer_simulation import get_cancer_sim_data
from src.cancer_sim.process_irregular_data import *
import pickle


args = {"chemo_coeff":0,               # Not used (confounding)
    "radio_coeff":0,                   # Not used (confounding)
    "obs_coeff":2,#6.,
    "intensity_cov":0,                 # (outcome unrelated sampling intensity)
    "intensity_cov_only":False,        # True for outcome unrelated sampling
    "max_intensity":1.,                # Max intensity (1 / S_\lambda in paper)
    "num_patients":10000,
    "results_dir":"results",
    "model_name":"BNCDE",
    "load_dataset":False,               # True to skip data generation
    "use_transformed":False,            # True to skip data transformation
    "experiment":"default",             # Add other experiments as yml files
    "data_path":"../../data/new_cancer_sim.p",
    "kappa":5,                          # Not used
    "max_samples":1,                    # Not used
    "strategy":"all",                   # Not used
    "save_raw_datapath":"../../data/raw",
    "save_transformed_datapath":"../../data/transformed"
}



# Generate raw data
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
    noise_level=0.01
)

# Transformed data (or load) -- apply observation processs
pickle_map = transform_data(
    data=pickle_map,
    interpolate=False,
    strategy=args["strategy"],
    sample_prop=1,
    kappa=args["kappa"],
    max_samples=args["max_samples"],
)

# Save new dataset
with open('./data/cancer_sim/cancer_pickle_map_irregular_informative_sampling.pkl', 'wb') as f:
    pickle.dump(pickle_map, f)

