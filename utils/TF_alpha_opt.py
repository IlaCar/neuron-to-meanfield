import numpy as np
import os
import json
import pandas as pd
from scipy.optimize import minimize
from utils.TF_helper import *


# -----------------------------------------------------
def run_fits(alpha = None,
             df_data = None,
             mu_V = None,
             sig_V = None,
             tau_V = None,
             tau_V_norm = None,
             params_SI = None,
             saving_json = False, out_file_name = None):
    
    """
    Run est_thresh -> res_1_func -> res_2_func for a given alpha
    """
    
    try:
        # Step 1: Estimate threshold
        est_V_th = est_thresh(data=df_data, mu_V=mu_V, sig_V=sig_V, tau_V=tau_V, alpha=alpha)

        # Step 2: First fit
        poly_params_init = -np.ones(10) * 1e-3
        poly_params_init[0] = np.mean(est_V_th)
        fit_1 = minimize(
            res_1_func,
            poly_params_init,
            args=(mu_V, sig_V, tau_V_norm, est_V_th),
            method='SLSQP',
            options={'ftol': 1e-17, 'disp': False, 'maxiter': 3000}
        )
        poly_params_1 = fit_1['x']

        # Step 3: Second fit
        fit_2 = minimize(
            res_2_func,
            poly_params_1,
            args=(df_data, params_SI, alpha),
            method='Nelder-Mead',
            options={'disp': False, 'maxiter': 10000}
        )
        poly_params_2 = fit_2['x']

        # Step 4: Compute mean error
        mean_error = res_2_func(poly_params_2, data=df_data, params=params_SI, alpha=alpha)

        # Step 5: Optionally save
        if saving_json and out_file_name is not None:
            data_structure_to_save = {
                'alpha': float(alpha),
                'mean_error': float(mean_error),
                'polynomial_params': poly_params_2.tolist()
            }

            if os.path.exists(out_file_name):
                with open(out_file_name, 'r') as f:
                    data = json.load(f)
            else:
                data = []

            data.append(data_structure_to_save)

            with open(out_file_name, 'w') as f:
                json.dump(data, f, indent=4)

        return mean_error, poly_params_2

    except Exception as e:
        print(f"Warning: alpha={alpha} caused an exception: {e}")
        return np.inf, None


def discrete_alpha_search(df_data = None,
                          mu_V = None,
                          sig_V = None,
                          tau_V = None,
                          tau_V_norm = None,
                          params_SI = None,
                          alpha_min=0.1, alpha_max=2.0, alpha_step=0.001,
                          saving_json = False, out_file_name = None):
    """
    Search over alpha values and return best fit
    """
    
    alpha_candidates = np.round(np.arange(alpha_min, alpha_max + alpha_step, alpha_step), 3)

    best_alpha = None
    best_error = np.inf
    best_poly = None

    for alpha in alpha_candidates:
        mean_error, poly_params_2 = run_fits(alpha, df_data, mu_V, sig_V, tau_V, tau_V_norm,
                                             params_SI, saving_json=saving_json, out_file_name=out_file_name)
        print(f'Tested alpha: {alpha:.3f}, mean_error: {mean_error:.6f}')

        if mean_error < best_error:
            best_error = mean_error
            best_alpha = alpha
            best_poly = poly_params_2

    print(f'\nBest alpha: {best_alpha:.3f}, mean_error: {best_error:.6f}')
    return best_alpha, best_error, best_poly

'''
# -----------------------------------------------------
# --- Main function
# -----------------------------------------------------
def main():
    data_folder = '../../AdEx_models'
    idx = 0
    neuron_model = 'RS'

    params_SI = get_params_model_SI(
        neuron_model=neuron_model,
        json_file_name=os.path.join(data_folder, neuron_model + '.json')
    )

    network_config_folder = '../../neural_network/network_config_files/'
    network_file_name = 'network_config_file_v0.json'
    network_config = get_network_config(
        json_file_name=os.path.join(network_config_folder, network_file_name)
    )

    params_SI = adding_K_params(
        neuron_params=params_SI,
        network_config=network_config
    )

    simulation_folder = '../data_extraction/simulations/TF_RS_v0_delay'
    folder_name = os.path.basename(simulation_folder)
    df_data = pd.read_csv(
        os.path.join(simulation_folder, 'testing_TF_data_RS.dat'),
        delim_whitespace=True,
        header=0
    )

    mu_V, sig_V, tau_V, tau_V_norm = membrane_potential_fluctuations(data=df_data, params=params_SI)

    best_alpha, best_error, best_poly = discrete_alpha_search(
        df_data, mu_V, sig_V, tau_V, tau_V_norm, params_SI,
        alpha_min=0.1, alpha_max=2.0, alpha_step=0.001,
        saving_json=True, out_file_name = f'alpha_results_{folder_name}.json'
        )

    print("Finished search.")
    print(f"Best alpha = {best_alpha}, error = {best_error}")


# -----------------------------------------------------
# --- Script entry point
# -----------------------------------------------------
if __name__ == "__main__":
    main()
'''