import torch
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns


def format_df(df):
    # Convert to float32
    #columns_to_convert = ['coverage_f', 'coverage_cf', 'size_f', 'size_cf']
    columns_to_convert = ['coverage_cf', 'size_cf']

    def convert_to_float32(s):
        match = re.search(r'\d+\.\d+', s)
        if match:
            return float(match.group())
        else:
            return 1.0

    df[columns_to_convert] = df[columns_to_convert].applymap(convert_to_float32)

    if 'model' in df.columns:
        # Rename
        df['model'] = df['model'].map({'tecde_dropout_head': 'TE-CDE', 'bncde': 'BNCDE'})

    return df



#################################################################################
# Errors vs deferral rate:

def plot_deferral_rate(path, mc_samples, window, informative_sampling=False, save=False, save_dir=None, log_scale=False, error_type='rel_errors',
                       extension=''):

    if informative_sampling:
        sampling_ext = '_informative_sampling'
        baseline_name = 'TESAR-CDE'

    else:
        sampling_ext = ''
        baseline_name = 'TE-CDE'

    for seed in [1, 2, 3, 4, 5]:
        errors_path = path + '/' + error_type + '_' + str(mc_samples) + '_' + window + extension + '_seed' + str(
            seed) + sampling_ext +'.pkl'
        errors = torch.load(errors_path, map_location=torch.device('cpu'))
        if seed == 1:
            bncde_error = errors['bncde']['None'][None, ...]
            tecde_error = errors['tecde_dropout_head']['0.1'][None, ...]

        else:
            bncde_error = np.concatenate((bncde_error, errors['bncde']['None'][None, ...]), axis=0)
            tecde_error = np.concatenate((tecde_error, errors['tecde_dropout_head']['0.1'][None, ...]), axis=0)

    plt.figure(figsize=(5, 5))
    # sns.set(font_scale=1.15, style="whitegrid")
    sns.set(style="whitegrid",font_scale=1.15)
    rc('font', **{'family': 'libertine'})
    rc('text', usetex=True)
    rc('text.latex', preamble=r'\usepackage{amsfonts}')


    sns.lineplot(x=np.linspace(0., 1., tecde_error.shape[1]), y=tecde_error.mean(axis=0), color='tab:red',
                 label=baseline_name)
    plt.fill_between(x=np.linspace(0., 1., tecde_error.shape[1]), y1=tecde_error.mean(axis=0) - tecde_error.std(axis=0),
                     y2=tecde_error.mean(axis=0) + tecde_error.std(axis=0), color='tab:red', alpha=0.3)
    sns.lineplot(x=np.linspace(0., 1., bncde_error.shape[1]), y=bncde_error.mean(axis=0), color='tab:blue',
                 label='BNCDE (ours)')
    plt.fill_between(x=np.linspace(0., 1., bncde_error.shape[1]), y1=bncde_error.mean(axis=0) - bncde_error.std(axis=0),
                     y2=bncde_error.mean(axis=0) + bncde_error.std(axis=0), color='tab:blue', alpha=0.3)

    sns.lineplot(x=np.linspace(0., 1., tecde_error.shape[1]),
                 y=np.linspace(1, 0., tecde_error.shape[1]), color='grey', label='Random deferral', linestyle=':')

    # Add labels and title
    plt.xlabel('Proportion of most uncertain samples withheld')

    if error_type == 'rel_errors':
        y_label = 'Normalized total MSE'
    elif error_type == 'abs_errors':
        y_label = 'Average MSE'
    plt.ylabel(y_label)

    if log_scale:
        plt.yscale('log')

    plt.xlim([0., 1.])
    plt.ylim([1e-7, 1])

    legend = plt.legend(frameon=1)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    plt.subplots_adjust(left=0.17)  # Increase the left margin to make more space for the label

    if save:
        plt.savefig(save_dir + '/deferral_rate_' + str(mc_samples) + '_' + window + sampling_ext + '.pdf')

    plt.show()




########################################################################################################################

def plot_coverage_quantile(path, mc_samples, window,  informative_sampling=False,
                           lower=0.95, save=False, save_dir='./figures', extension=''):

    if informative_sampling:
        sampling_ext = '_informative_sampling'
        baseline_name = 'TESAR-CDE'
    else:
        sampling_ext = ''
        baseline_name = 'TE-CDE'

    # load results from all seeds and merge
    for seed in [1, 2, 3, 4, 5]:
        quantiles_path = path + '/quantiles_' + str(mc_samples) + '_' + window + extension + '_seed' + str(
            seed) + sampling_ext + '.csv'
        q_results = pd.read_csv(quantiles_path)
        q_results = format_df(q_results)
        q_results['seed'] = seed

        if seed == 1:
            bncde_results = q_results[q_results['model'] == 'BNCDE']
            tecde_results = q_results[q_results['model'] == 'TE-CDE']
        else:
            bncde_results =  pd.concat([bncde_results, q_results[q_results['model'] == 'BNCDE']], ignore_index=True)
            tecde_results = pd.concat([tecde_results, q_results[q_results['model'] == 'TE-CDE']], ignore_index=True)

    bncde_results = bncde_results[['quantile', 'coverage_cf', 'size_cf', 'seed']][(bncde_results['quantile'] < 1.0)&(bncde_results['quantile'] >= lower)]
    tecde_results = tecde_results[['quantile', 'coverage_cf', 'size_cf', 'seed']][(tecde_results['quantile'] < 1.0)&(tecde_results['quantile'] >= lower)]
    print(tecde_results)
    # Compute mean and standard deviation of coverage error
    bncde_results['error'] = bncde_results['quantile'] - bncde_results['coverage_cf']
    bncde_mean = bncde_results.groupby('quantile')['error'].mean().reset_index()
    bncde_std = bncde_results.groupby('quantile')['error'].std().reset_index()
    bncde_max = bncde_results.groupby('quantile')['error'].max().reset_index()
    bncde_min = bncde_results.groupby('quantile')['error'].min().reset_index()

    tecde_results['error'] = tecde_results['quantile'] - tecde_results['coverage_cf']
    tecde_mean = tecde_results.groupby('quantile')['error'].mean().reset_index()
    tecde_std = tecde_results.groupby('quantile')['error'].std().reset_index()
    tecde_max = tecde_results.groupby('quantile')['error'].max().reset_index()
    tecde_min = tecde_results.groupby('quantile')['error'].min().reset_index()


    # Plot results
    plt.figure(figsize=(5, 5))
    sns.set(style="whitegrid",font_scale=1.15)
    rc('font', **{'family': 'libertine'})
    rc('text', usetex=True)
    rc('text.latex', preamble=r'\usepackage{amsfonts}')

    sns.lineplot(x=tecde_mean['quantile'], y=tecde_mean['error'],
                 color='tab:red', linestyle='-', label=baseline_name, marker='s')

    plt.fill_between(x=tecde_mean['quantile'], y1=tecde_mean['error'] - tecde_std['error'],
                      y2=tecde_mean['error'] + tecde_std['error'], color='tab:red', alpha=0.7)

    sns.lineplot(x=bncde_mean['quantile'], y=bncde_mean['error'],
                 color='tab:blue', linestyle='-', label='BNCDE (ours)', marker='D')

    plt.fill_between(x=bncde_mean['quantile'], y1=bncde_mean['error'] - bncde_std['error'],
                     y2=bncde_mean['error'] + bncde_std['error'], color='tab:blue', alpha=0.7)

    plt.fill_between(x=[lower, 0.99], y1=[0, 0], y2=[1, 1], color='tab:red', alpha=0.1,
                     label='Overconfident')
    plt.fill_between(x=[lower, 0.99], y1=[0, 0], y2=[-1, -1], color='tab:green', alpha=0.1,
                     label='Conservative')

    plt.ylim([-0.07, 0.07])
    plt.xlim([lower, 0.99])
    plt.xlabel('Posterior predictive credible interval')
    plt.ylabel('CrI minus empirical coverage')
    legend = plt.legend(frameon=1,loc='lower right')

    frame = legend.get_frame()
    frame.set_facecolor('white')
    # Add labels and legend
    plt.subplots_adjust(left=0.17)  # Increase the left margin to make more space for the label
    if save:
        plt.savefig(save_dir + '/coverage_' + str(mc_samples) + '_' + window + extension + sampling_ext + '.pdf')

    plt.show()


########################################################################################################################


def plot_size_quantile(path, mc_samples, window, informative_sampling=False,
                       lower=0.95, save=False, save_dir='./figures', extension=''):

    if informative_sampling:
        sampling_ext = '_informative_sampling'
        baseline_name = 'TESAR-CDE'

    else:
        sampling_ext = ''
        baseline_name = 'TE-CDE'
    # load results from all seeds and merge
    for seed in [1, 2, 3, 4, 5]:
        quantiles_path = path + '/quantiles_' + str(mc_samples) + '_' + window + extension + '_seed' + str(
            seed) + sampling_ext + '.csv'
        q_results = pd.read_csv(quantiles_path)
        q_results = format_df(q_results)
        q_results['seed'] = seed

        if seed == 1:
            bncde_results = q_results[q_results['model'] == 'BNCDE']
            tecde_results = q_results[q_results['model'] == 'TE-CDE']
        else:
            bncde_results =  pd.concat([bncde_results, q_results[q_results['model'] == 'BNCDE']], ignore_index=True)
            tecde_results = pd.concat([tecde_results, q_results[q_results['model'] == 'TE-CDE']], ignore_index=True)

    bncde_results = bncde_results[['quantile', 'coverage_cf', 'size_cf', 'seed']][(bncde_results['quantile'] < 1.0)&(bncde_results['quantile'] >= lower)]
    tecde_results = tecde_results[['quantile', 'coverage_cf', 'size_cf', 'seed']][(tecde_results['quantile'] < 1.0)&(tecde_results['quantile'] >= lower)]

    # Compute mean and standard deviation of size
    bncde_mean = bncde_results.groupby('quantile')['size_cf'].mean().reset_index()
    bncde_std = bncde_results.groupby('quantile')['size_cf'].std().reset_index()

    tecde_mean = tecde_results.groupby('quantile')['size_cf'].mean().reset_index()
    tecde_std = tecde_results.groupby('quantile')['size_cf'].std().reset_index()

    # Plot results
    plt.figure(figsize=(5, 5))
    sns.set(style="whitegrid", font_scale=1.15)
    rc('font', **{'family': 'libertine'})
    rc('text', usetex=True)
    rc('text.latex', preamble=r'\usepackage{amsfonts}')

    sns.lineplot(x=tecde_mean['quantile'], y=tecde_mean['size_cf'],
                 color='tab:red', linestyle='-', label=baseline_name, marker='s')#, linewidth=0.5)
    #plt.fill_between(x=tecde_mean['quantile'], y1=tecde_min['size_cf'],
    #                 y2=tecde_max['size_cf'], color='tab:red', alpha=0.7)
    plt.fill_between(x=tecde_mean['quantile'], y1=tecde_mean['size_cf'] - tecde_std['size_cf'],
                     y2=tecde_mean['size_cf'] + tecde_std['size_cf'], color='tab:red', alpha=0.7)


    sns.lineplot(x=bncde_mean['quantile'], y=bncde_mean['size_cf'],
                 color='tab:blue', linestyle='-', label='BNCDE (ours)', marker='D')#, linewidth=0.5)
    #plt.fill_between(x=bncde_mean['quantile'], y1=bncde_min['size_cf'],
    #                 y2=bncde_max['size_cf'], color='tab:blue', alpha=0.7)
    plt.fill_between(x=bncde_mean['quantile'], y1=bncde_mean['size_cf'] - bncde_std['size_cf'],
                      y2=bncde_mean['size_cf'] + bncde_std['size_cf'], color='tab:blue', alpha=0.7)


    plt.xlim([lower, 0.99])
    plt.ylim([0., 0.25])

    plt.xlabel('Posterior predictive credible interval')
    plt.ylabel('Credible interval width')
    legend = plt.legend(frameon=1, loc='center right')
    frame = legend.get_frame()
    frame.set_facecolor('white')
    plt.subplots_adjust(left=0.17)  # Increase the left margin to make more space for the label
    if save:
        plt.savefig(save_dir + '/size_' + str(mc_samples) + '_' + window + sampling_ext +'.pdf')

    plt.show()


def plot_error_vs_corruption(path='./data/results', mc_samples=100, window='five_step', informative_sampling=False,
                             save=False, save_dir='./figures', extension='_final_results',
                             error_type='median'):
    all_bncde_medians = []
    all_tecde_medians = []
    all_bncde_means = []
    all_tecde_means = []

    if informative_sampling:
        sampling_ext = '_informative_sampling'
        baseline_name = 'TESAR-CDE'
    else:
        sampling_ext = ''
        baseline_name = 'TE-CDE'

    for seed in ['_seed1', '_seed2', '_seed3', '_seed4', '_seed5']:
        bncde_medians = []
        tecde_medians = []
        bncde_means = []
        tecde_means = []
        for noise in ['_noise_01', '_noise_02', '_noise_03', '_noise_04', '_noise_05',
                      '_noise_06', '_noise_07', '_noise_08', '_noise_09', '_noise_10']:
            outcomes = torch.load(
                path + '/outcomes_' + str(mc_samples) + '_' + window + noise + extension + seed + sampling_ext + '.pkl',
                map_location=torch.device('cpu'))


            bncde_estimate_cf = outcomes['bncde']['None']['mu_cf'].mean(dim=0)  # quantile(0.5, dim=0)
            tecde_estimate_cf = outcomes['tecde_dropout_head']['0.1']['mu_cf'].mean(dim=0)  # .quantile(0.5, dim=0)
            y_cf = outcomes['bncde']['None']['y_cf']
            observed = ~torch.isnan(y_cf)
            y_cf = y_cf[observed]
            tecde_estimate_cf = tecde_estimate_cf[observed]
            bncde_estimate_cf = bncde_estimate_cf[observed]

            tecde_error_mean = torch.abs(tecde_estimate_cf - y_cf).mean()
            bncde_error_mean = torch.abs(bncde_estimate_cf - y_cf).mean()
            tecde_means.append(tecde_error_mean)
            bncde_means.append(bncde_error_mean)

            tecde_error_med = torch.abs(tecde_estimate_cf - y_cf).median()
            bncde_error_med = torch.abs(bncde_estimate_cf - y_cf).median()
            tecde_medians.append(tecde_error_med)
            bncde_medians.append(bncde_error_med)

        all_tecde_means.append(np.array(tecde_means))
        all_bncde_means.append(np.array(bncde_means))
        all_tecde_medians.append(np.array(tecde_medians))
        all_bncde_medians.append(np.array(bncde_medians))

    all_bncde_medians = np.array(all_bncde_medians)
    all_tecde_medians = np.array(all_tecde_medians)
    all_bncde_means = np.array(all_bncde_means)
    all_tecde_means = np.array(all_tecde_means)

    # Plot results
    plt.figure(figsize=(5, 5))
    sns.set(style="whitegrid", font_scale=1.15)
    rc('font', **{'family': 'libertine'})
    rc('text', usetex=True)
    rc('text.latex', preamble=r'\usepackage{amsfonts}')

    if error_type == 'mean':

        noise_levels = np.linspace(0.01, 0.1, 10)

        sns.lineplot(x=noise_levels, y=all_tecde_means.mean(axis=0), color='tab:red', marker='s', label=baseline_name)
        plt.fill_between(x=noise_levels, y1=all_tecde_means.mean(axis=0) - all_tecde_means.std(axis=0),
                         y2=all_tecde_means.mean(axis=0) + all_tecde_means.std(axis=0),
                         color='tab:red', alpha=0.3)
        sns.lineplot(x=noise_levels, y=all_bncde_means.mean(axis=0), color='tab:blue', marker='D', label='BNCDE (ours)')
        plt.fill_between(x=noise_levels, y1=all_bncde_means.mean(axis=0) - all_bncde_means.std(axis=0),
                         y2=all_bncde_means.mean(axis=0) + all_bncde_means.std(axis=0),
                         color='tab:blue', alpha=0.3)
    if error_type == 'median':

        noise_levels = np.linspace(0.01, 0.1, 10)

        sns.lineplot(x=noise_levels, y=all_tecde_medians.mean(axis=0), color='tab:red', marker='s', label=baseline_name)
        plt.fill_between(x=noise_levels, y1=all_tecde_medians.mean(axis=0) - all_tecde_medians.std(axis=0),
                         y2=all_tecde_medians.mean(axis=0) + all_tecde_medians.std(axis=0),
                         color='tab:red', alpha=0.3)
        sns.lineplot(x=noise_levels, y=all_bncde_medians.mean(axis=0), color='tab:blue', marker='D', label='BNCDE (ours)')
        plt.fill_between(x=noise_levels, y1=all_bncde_medians.mean(axis=0) - all_bncde_medians.std(axis=0),
                         y2=all_bncde_medians.mean(axis=0) + all_bncde_medians.std(axis=0),
                         color='tab:blue', alpha=0.3)
    plt.xlabel(r'$\sqrt{\mathrm{Var}(\epsilon_t)}$')
    plt.ylabel('MSE of Monte Carlo mean estimate')
    plt.xlim([0.01, 0.1])

    legend = plt.legend(frameon=1, loc='center right')
    frame = legend.get_frame()
    frame.set_facecolor('white')
    plt.subplots_adjust(left=0.17)

    if save:
        plt.savefig(save_dir + '/corruption_' + str(mc_samples) + '_' + error_type + '_' + window + sampling_ext +'.pdf')

    plt.show()

########################################################################################################################
# Call

for window in ['one_step', 'two_step', 'three_step', 'four_step', 'five_step']:
     plot_deferral_rate(path='./data/results', mc_samples=100, window=window, informative_sampling=False,
                       save=False, save_dir='./figures',
                       extension='', log_scale=True)

     plot_coverage_quantile(path='./data/results/', mc_samples=100, window=window, informative_sampling=False,
                            save=False, save_dir='./figures',
                           extension='', lower=0.95)

     plot_size_quantile(path='./data/results/', mc_samples=100, window=window, informative_sampling=False,
                        save=False, save_dir='./figures',
                        extension='', lower=0.95)

     plot_error_vs_corruption(path='./data/results', mc_samples=100, window=window, informative_sampling=False,
                              save=False, save_dir='./figures', extension='',
                              error_type='median')
