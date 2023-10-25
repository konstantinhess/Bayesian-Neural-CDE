# Run tests
import torch
import pytorch_lightning as pl
import os
from src.cancer_sim.cancer_data_module import CancerDataloader
from src.models.tecde_dropout_model import TE_CDE_dropout_head
from src.models.bncde_model import BNCDE
import pandas as pd
import pickle
import numpy as np


# helper function
def unpack_preds(pred_out):
    combined_data = {}
    losses = [batch['loss'] for batch in pred_out]
    y_preds = [batch['y_pred'] for batch in pred_out]
    if 'sigma_pred' in pred_out[0].keys():
        sigma_pred = [batch['sigma_pred'] for batch in pred_out]
    coverage = pred_out[-1]['coverage']
    size = pred_out[-1]['size']
    covariates = [batch['x'] for batch in pred_out]

    ys = [batch['y'] for batch in pred_out]
    combined_data['losses'] = torch.tensor(losses)
    combined_data['y_pred'] = torch.concatenate(y_preds, dim=-2).squeeze(dim=-1)
    if 'sigma_preds' in pred_out[0].keys():
        combined_data['sigma_pred'] = torch.concatenate(sigma_pred, dim=-2).squeeze(dim=-1)
    combined_data['y'] = torch.concatenate(ys, dim=-2).squeeze(dim=-1)
    combined_data['coverage'] = coverage
    combined_data['size'] = size
    combined_data['covariates'] = torch.concatenate(covariates, dim=0)

    return combined_data




def eval_predictions(model_instance, model_path, data_path, batch_size=64,  informative_sampling=False,
                     device='cuda', num_workers=0, only_cf=False):

    # Load data
    with open(data_path, 'rb') as f:
        pickle_map = pickle.load(f)

    # Load state dict
    state_dict = torch.load(os.path.join(model_path))['state_dict']
    if informative_sampling:
        rm_keys = [key for key in state_dict.keys() if (key.startswith('z_field'))]
    else:
        rm_keys = [key for key in state_dict.keys() if (key.startswith('z_field') | key.startswith('obs_net'))]
    [state_dict.pop(key, None) for key in rm_keys]

    model_instance.load_state_dict(state_dict)
    trainer = pl.Trainer(accelerator=device, devices=[0])

    # Counterfactual data
    data_cf = CancerDataloader(pickle_map=pickle_map, batch_size=batch_size, device=device, num_workers=num_workers,
                               test_set='cf')
    pred_cf = trainer.predict(model_instance, data_cf)
    if only_cf:
        unpacked_cf = unpack_preds(pred_cf)
        y_cf = unpacked_cf['y']
        mu_cf = unpacked_cf['y_pred']
        return None, None, None, None, unpacked_cf['coverage'], unpacked_cf['size'], None, mu_cf, None, y_cf

    else:
        model_instance.coverage_metric.reset()

        # Factual data
        data_f = CancerDataloader(pickle_map=pickle_map, batch_size=batch_size,device=device, num_workers=num_workers,
                                  test_set='f')
        pred_f = trainer.predict(model_instance, data_f)


        # Unpack predictions
        unpacked_f = unpack_preds(pred_f)
        unpacked_cf = unpack_preds(pred_cf)

        # Get last treatment decision
        last_treatment_f = unpacked_f['covariates'][:, -1, -4:]
        last_treatment_cf = unpacked_cf['covariates'][:, -1, -4:]

        # Determine whether last treatment is in factual or counterfactual path (only relevant for augmented test data)
        treatment_1 = (last_treatment_f[:, 0] < last_treatment_cf[:, 0]) * 1  # 0 = CF, 1 = F

        # True CATE
        y_f = unpacked_f['y']
        y_cf = unpacked_cf['y']
        CATE = (y_f - y_cf) ** (-1) ** treatment_1
        observed = ~torch.isnan(CATE)
        CATE = CATE[observed]

        # Point estimates
        mu_f = unpacked_f['y_pred'][:, observed]
        mu_cf = unpacked_cf['y_pred'][:, observed]
        # For save
        y_f = y_f[observed]
        y_cf = y_cf[observed]


        # Estimated CATE and epistemic uncertainty of CATE estimate
        estimated_CATE = torch.mean((mu_f - mu_cf) * (-1) ** treatment_1[observed], dim=0)
        epistemic_var_CATE = torch.var((mu_f - mu_cf) * (-1) ** treatment_1[observed], dim=0)
        indexes_sorted = torch.argsort(epistemic_var_CATE, descending=True)


        # Cate error
        cate_error = (CATE - estimated_CATE) ** 2

        cate_error_sorted = cate_error[indexes_sorted]
        cate_error_sum = torch.sum(cate_error)
        abs_mean_error_sorted = []
        rel_mean_error_sorted = []

        # Compute MSEs for ascending number of samples withheld
        for i in range(len(CATE)):
            abs_mean_error_sorted.append(torch.mean(cate_error_sorted[i:]))
            rel_mean_error_sorted.append(torch.sum(cate_error_sorted[i:]) / cate_error_sum)
        abs_mean_error_sorted = torch.tensor(abs_mean_error_sorted)
        rel_mean_error_sorted = torch.tensor(rel_mean_error_sorted)


        return abs_mean_error_sorted, rel_mean_error_sorted, unpacked_f['coverage'], unpacked_f['size'], unpacked_cf['coverage'], unpacked_cf['size'], mu_f, mu_cf, y_f, y_cf






def run_tests(bncde_path,
              tecde_path,
              baseline='dropout_head', # ['dropout_head', 'dropout_vectorfield']
              coverage_vs_quantile = True, # if True: compute BNCDE coverage vs quantiles q on q=[0.9,0.905,...,0.995]
              mc_samples=50,
              only_cf = False,
              window='one_step',
              informative_sampling=False,
              information_level ='',
              data_path='./data/cancer_sim/cancer_pickle_map_irregular.pkl',
              save=True,
              save_path='./data/results/',
              extension='',
              device='cuda',
              num_workers=0
              ):
    torch.set_default_device(device)

    if isinstance(baseline, str):
        baseline = [baseline]
    results = {}
    batch_size = 5000 # does not impact prediction, only for computation speed
    window_dict = {'one_step':1, 'two_step':2, 'three_step':3, 'four_step':4, 'five_step':5}
    if informative_sampling:
        extension = extension+'_informative_sampling'

    ####################################################################################################################
    # TE-CDE: MC dropout head
    if 'dropout_head' in baseline:
        # Iterate over multiple mc dropout configurations
        dropout_head_results = {}

        for dropout_prob in [0.1]:
            print(window)
            print('dropout prob:'+str(dropout_prob))
            model_instance = TE_CDE_dropout_head(hidden_size=8, control_size=7, treatment_size=4, hidden_layers=[128, 128],
                                                    mc_samples=mc_samples, dropout_probability=dropout_prob,
                                                 intensity_weighting=informative_sampling,
                                                    prediction_window=window_dict[window],
                                                    interpolation_method='spline', method='euler')

            instance_path = tecde_path

            if only_cf:
                abs_errors, rel_errors, tecde_coverage_f, tecde_size_f, tecde_coverage_cf, tecde_size_cf, mu_f, mu_cf, y_f, y_cf = eval_predictions(
                    model_instance, instance_path, data_path, batch_size, only_cf=only_cf, informative_sampling=informative_sampling, device=device,
                    num_workers=num_workers)
            else:
                abs_errors, rel_errors, tecde_coverage_f, tecde_size_f, tecde_coverage_cf, tecde_size_cf, mu_f, mu_cf, y_f, y_cf = eval_predictions(
                    model_instance, instance_path, data_path, batch_size, informative_sampling=informative_sampling, device=device, num_workers=num_workers)

            dropout_head_results[str(dropout_prob)] = {
                'abs_errors': abs_errors,
                'rel_errors': rel_errors,
                'coverage_f': tecde_coverage_f, 'size_f': tecde_size_f,
                'coverage_cf': tecde_coverage_cf, 'size_cf': tecde_size_cf,
                'mu_f': mu_f, 'mu_cf': mu_cf, 'y_f': y_f, 'y_cf': y_cf
            }

        # output structure
        results['tecde_dropout_head'] = dropout_head_results

    ####################################################################################################################
    # BNCDE

    # Iterate over multiple mc dropout configurations
    bncde_results = {}

    model_instance = BNCDE(mc_samples=mc_samples, hidden_size=8,
                  sd_diffusion=0.001, drift_layers=[16,64,64,64,16],
                  intensity_weighting=informative_sampling,
                  control_size=7, treatment_size=4,
                  prediction_window=window_dict[window],
                  interpolation_method='spline', method='euler')
    instance_path = bncde_path
    if only_cf:
        abs_errors, rel_errors, bncde_coverage_f, bncde_size_f, bncde_coverage_cf, bncde_size_cf, mu_f, mu_cf, y_f, y_cf = eval_predictions(
            model_instance, instance_path, data_path, batch_size, only_cf=only_cf, informative_sampling=informative_sampling,device=device, num_workers=num_workers)
    else:
        abs_errors, rel_errors, bncde_coverage_f, bncde_size_f, bncde_coverage_cf, bncde_size_cf, mu_f, mu_cf, y_f, y_cf = eval_predictions(
            model_instance, instance_path, data_path, batch_size, informative_sampling=informative_sampling, device=device, num_workers=num_workers)
    bncde_results['None'] = { # 'None' = dropout_prob key level
        'abs_errors': abs_errors,
        'rel_errors': rel_errors,
        'coverage_f': bncde_coverage_f, 'size_f': bncde_size_f,
        'coverage_cf': bncde_coverage_cf, 'size_cf': bncde_size_cf,
        'mu_f': mu_f, 'mu_cf': mu_cf, 'y_f': y_f, 'y_cf': y_cf
    }

    # output structure
    results['bncde'] = bncde_results

    ####################################################################################################################

    # Extract predictions and outcomes into dictionary
    outcomes_dict = {}
    # Iterate through the dictionary and extract the errors
    for model, dropout_probs in results.items():
        outcomes = {}
        for dropout_prob, values in dropout_probs.items():
            outcomes[dropout_prob] = {}
            outcomes[dropout_prob]['mu_f'] = values['mu_f']
            outcomes[dropout_prob]['mu_cf'] = values['mu_cf']
            outcomes[dropout_prob]['y_f'] = values['y_f']
            outcomes[dropout_prob]['y_cf'] = values['y_cf']

        outcomes_dict[model] = outcomes


    # Extract abs deferral errors into errors dictionary
    abs_errors_dict = {}
    # Iterate through the dictionary and extract the errors
    for model, dropout_probs in results.items():
        model_errors = {}
        for dropout_prob, values in dropout_probs.items():
            model_errors[dropout_prob] = values['abs_errors']
        abs_errors_dict[model] = model_errors

    # Extract abs deferral errors into errors dictionary
    rel_errors_dict = {}
    # Iterate through the dictionary and extract the errors
    for model, dropout_probs in results.items():
        model_errors = {}
        for dropout_prob, values in dropout_probs.items():
            model_errors[dropout_prob] = values['rel_errors']
        rel_errors_dict[model] = model_errors

    # Exract coverage and size into pandas data frame
    # Create empty lists to store data
    model_list = []
    dropout_prob_list = []
    coverage_f_list = []
    size_f_list = []
    coverage_cf_list = []
    size_cf_list = []

    # Iterate through the dictionary and extract the values
    for model, dropout_probs in results.items():
        for dropout_prob, values in dropout_probs.items():
            model_list.append(model)
            dropout_prob_list.append(dropout_prob)
            coverage_f_list.append(values['coverage_f'])
            size_f_list.append(values['size_f'])
            coverage_cf_list.append(values['coverage_cf'])
            size_cf_list.append(values['size_cf'])

    # Create a pandas DataFrame
    coverage_df = pd.DataFrame({
                    'model': model_list,
                    'dropout_prob': dropout_prob_list,
                    'coverage_f': coverage_f_list,
                    'size_f': size_f_list,
                    'coverage_cf': coverage_cf_list,
                    'size_cf': size_cf_list
                })


    # OPTIONAL: Run additional quantiles vs coverage
    if coverage_vs_quantile:

        quantile_results = {}
        quantile_list = np.linspace(0.95, 0.99, 5)  # iterate over quantiles
        #quantile_list = [0.81, 0.83, 0.85, 0.87, 0.89, 0.91, 0.93, 0.95, 0.97, 0.99]
        ####################################################################################################################
        # TE-CDE

        quantile_dict = {}
        for q in quantile_list:  # iterate over quantiles q = 0.9, ..., 0.995
            print(window)
            print('TE-CDE: '+str(q))
            model_instance = TE_CDE_dropout_head(hidden_size=8, control_size=7, treatment_size=4,
                                                 hidden_layers=[128, 128],
                                                 mc_samples=mc_samples, dropout_probability=0.1,
                                                 prediction_window=window_dict[window],
                                                 interpolation_method='spline', method='euler',
                                                 coverage_metric_confidence=q)

            instance_path = tecde_path

            _, _, _, _, tecde_coverage_cf, tecde_size_cf, _, _, _, _ = eval_predictions(model_instance, instance_path, data_path,
                                                                                  batch_size, informative_sampling=informative_sampling,
                                                                                  device=device,
                                                                                  num_workers=num_workers, only_cf=True)

            quantile_dict[str(q)] = {
                'coverage_cf': tecde_coverage_cf, 'size_cf': tecde_size_cf
            }

        quantile_results['tecde_dropout_head'] = quantile_dict

        ####################################################################################################################
        # BNCDE
        quantile_dict = {}
        for q in quantile_list:  # iterate over quantiles q = 0.9, ..., 0.995
            print(window)
            print('BNCDE: ' + str(q))
            model_instance = BNCDE(mc_samples=mc_samples, hidden_size=8,
                                   sd_diffusion=0.001, drift_layers=[16, 64, 64, 64, 16],
                                   control_size=7, treatment_size=4,
                                   prediction_window=window_dict[window],
                                   interpolation_method='spline', method='euler',
                                   coverage_metric_confidence=q)

            instance_path = bncde_path
            _, _, _, _, tecde_coverage_cf, tecde_size_cf, _, _, _, _ = eval_predictions(model_instance, instance_path, data_path,
                                                                      batch_size,  informative_sampling=informative_sampling, device=device,
                                                                      num_workers=num_workers, only_cf=True)

            quantile_dict[str(q)] = {
                                'coverage_cf': bncde_coverage_cf, 'size_cf': bncde_size_cf
                            }

        quantile_results['bncde'] = quantile_dict


        # Exract coverage and size into pandas data frame
        # Create empty lists to store data
        model_list = []
        quantile_list = []
        coverage_cf_list = []
        size_cf_list = []

        # Iterate through the dictionary and extract the values
        for model, qs in quantile_results.items():
            for q, values in qs.items():
                model_list.append(model)
                quantile_list.append(q)
                coverage_cf_list.append(values['coverage_cf'])
                size_cf_list.append(values['size_cf'])

        # Create a pandas DataFrame
        quantile_df = pd.DataFrame({
            'model': model_list,
            'quantile': quantile_list,
            'coverage_cf': coverage_cf_list,
            'size_cf': size_cf_list
        })


    if save:

        coverage_df.to_csv(save_path+'/coverage_'+str(mc_samples)+'_'+window+extension+'.csv')
        torch.save(abs_errors_dict, save_path+'/abs_errors_'+str(mc_samples)+'_'+window+extension+'.pkl')
        torch.save(rel_errors_dict, save_path+'/rel_errors_'+str(mc_samples)+'_'+window+extension+'.pkl')

        torch.save(outcomes_dict, save_path + '/outcomes_' + str(mc_samples) + '_' + window +extension+'.pkl')
        if coverage_vs_quantile:
            quantile_df.to_csv(save_path + '/quantiles_' + str(mc_samples) + '_' + window +extension+'.csv')



