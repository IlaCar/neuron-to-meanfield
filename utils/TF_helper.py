import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import erfc, erfcinv
import pandas as pd
import json
import seaborn as sns
import scipy.special as sp_spec
import warnings

implemented_neuron_models = ['FS', 'RS', 'RS_no_adapt']

# ---------------------------------------------------
def get_params_model_SI(idx = None, neuron_model = None, json_file_name = None, sim_info = False):
    if idx == None:
        idx = 0
    if neuron_model == None:
        raise ValueError("Plese, specify the neuron_model you wish to simulate")
    if neuron_model not in implemented_neuron_models:
        raise ValueError(f"neuron_model must be one of {implemented_neuron_models}, but got '{neuron_model}'.")
    if json_file_name == None:
        raise ValueError("Plese, specify the json_file_name containing the model parameters")    
    
    with open(json_file_name, 'r') as file:
        data = json.load(file)
        
        params_SI={}
        
        params_SI['C_m'] = data[0][0]['model']['C_m'] * 10**-9
        params_SI['g_L'] = data[0][0]['model']['g_L'] * 10**-6
        params_SI['E_L'] = data[0][0]['model']['E_L'] * 10**-3

        params_SI['a'] = data[0][0]['model']['a'] * 10**-9
        params_SI['b'] = data[0][0]['model']['b'] * 10**-9
        params_SI['tau_w'] = data[0][0]['model']['tau_w'] * 10**-3

        
        params_SI['V_th'] = data[0][0]['model']['V_th'] * 10**-3
        params_SI['Delta_T'] = data[0][0]['model']['Delta_T'] * 10**-3
        params_SI['V_reset'] = data[0][0]['model']['V_reset'] * 10**-3
        params_SI['V_peak'] = data[0][0]['model']['V_peak'] * 10**-3

        params_SI['t_ref'] = data[0][0]['model']['t_ref'] * 10**-3

        params_SI['E_e'] = data[0][0]['model']['E_e'] * 10**-3
        params_SI['Q_e'] = data[0][0]['model']['Q_e'] * 10**-9
        params_SI['E_i'] = data[0][0]['model']['E_i'] * 10**-3
        params_SI['Q_i'] = data[0][0]['model']['Q_i'] * 10**-9
        params_SI['tau_syn'] = data[0][0]['model']['tau_syn'] * 10**-3

        return params_SI
        
# ---------------------------------------------------
def get_network_config(idx = None, json_file_name = None, sim_info = False):
    if idx == None:
        idx = 0

    if json_file_name == None:
        raise ValueError("Plese, specify the json_file_name containing the model parameters")    
    
    with open(json_file_name, 'r') as file:
        data = json.load(file)

        return data[0]

# ---------------------------------------------------
def adding_K_params(neuron_params = None, network_config = None):
    neuron_params['K_e'] =  network_config['external_input']['N_external_exc'] * network_config['external_input']['conn_prob']
    neuron_params['K_i'] = network_config['external_input']['N_external_inh'] * network_config['external_input']['conn_prob']

    return neuron_params

# ---------------------------------------------------
def membrane_potential_fluctuations(neuron_model = None, data = None, params = None):
    """
    This function computes the mean, standard deviation and autocorrelation time constant
    of the membrane potential fluctuations as defined in
    Zerlaut, Y., Chemla, S., Chavane, F., & Destexhe, A. (2018) 
    with the addition of adaptation (which can also be zero) as in 
    Di Volo, M., Romagnoni, A., Capone, C. and Destexhe, A., (2019).
    """

    f_e = data['input_exc'].to_numpy()
    f_i = data['input_inh'].to_numpy()

    # Add small epsilon to avoid exact zeros but preserve scale
    f_e = np.maximum(f_e, 1e-9)
    f_i = np.maximum(f_i, 1e-9)    
    
    w_ad = np.zeros(len(f_i))

    ### eq. 5 in Zerlaut et al., (2018)
    mu_Ge = f_e * params['K_e'] * params['tau_syn'] * params['Q_e']
    sig_Ge = np.sqrt(0.5 * f_e * params['K_e'] * params['tau_syn']) * params['Q_e']
    mu_Gi = f_i * params['K_i'] * params['tau_syn'] * params['Q_i']
    sig_Gi = np.sqrt(0.5 * f_i * params['K_i'] * params['tau_syn']) * params['Q_i']  

    ### eq. 6 in Zerlaut et al., (2018)
    mu_G = mu_Ge + mu_Gi + params['g_L']
    tau_m = params['C_m'] / mu_G

    ### eq. 7 in Zerlaut et al., (2018) + adapt
    mu_V = (mu_Ge * params['E_e'] + mu_Gi * params['E_i'] + params['g_L'] * params['E_L'] - w_ad) / mu_G

    ### defined between eq. 9 and 10 in Zerlaut et al., (2018)
    U_e = params['Q_e'] / mu_G * (params['E_e'] - mu_V)
    U_i = params['Q_i'] / mu_G * (params['E_i'] - mu_V)

    ### eq. 15 in Zerlaut et al., (2018)    
    sig_V = np.sqrt(
        params['K_e'] * f_e * (U_e * params['tau_syn'])**2 / (2 * (tau_m + params['tau_syn'])) +
        params['K_i'] * f_i * (U_i * params['tau_syn'])**2 / (2 * (tau_m + params['tau_syn']))
    )
    
    ### eq. 17 in Zerlaut et al., (2018)    
    tau_V = (
            (params['K_e'] * f_e * (U_e * params['tau_syn'])**2 +
             params['K_i'] * f_i * (U_i * params['tau_syn'])**2
            ) / 
            (params['K_e'] * f_e * (U_e * params['tau_syn'])**2 / (tau_m + params['tau_syn']) +
             params['K_i'] * f_i * (U_i * params['tau_syn'])**2 / (tau_m + params['tau_syn']))
            )
    tau_V_norm = tau_V * params['g_L'] / params['C_m']
    
    return mu_V, sig_V, tau_V, tau_V_norm

# ---------------------------------------------------
def membrane_potential_fluctuations_sim(neuron_model=None, f_e=0.0, f_i=0.0, params=None, w_ad=0.0):
    """
    Compute mu_V, sig_V, tau_V, tau_V_norm for the current mean-field state.
    This version is scalar and optimized for dynamic network simulation.
    Based on Zerlaut et al., 2018 with the addition of adaptation based on Di Volo et al., 2019.
    """

    # avoid exact zeros
    f_e = max(f_e, 1e-9)
    f_i = max(f_i, 1e-9)

    # eq. 5 Zerlaut et al., 2018
    mu_Ge = f_e * params['K_e'] * params['tau_syn'] * params['Q_e']
    sig_Ge = np.sqrt(0.5 * f_e * params['K_e'] * params['tau_syn']) * params['Q_e']
    mu_Gi = f_i * params['K_i'] * params['tau_syn'] * params['Q_i']
    sig_Gi = np.sqrt(0.5 * f_i * params['K_i'] * params['tau_syn']) * params['Q_i']

    # eq. 6
    mu_G = mu_Ge + mu_Gi + params['g_L']
    tau_m = params['C_m'] / mu_G

    # eq. 7
    mu_V = (
        mu_Ge * params['E_e']
        + mu_Gi * params['E_i']
        + params['g_L'] * params['E_L']
        - w_ad
    ) / mu_G

    # eq. 9–10
    U_e = params['Q_e'] / mu_G * (params['E_e'] - mu_V)
    U_i = params['Q_i'] / mu_G * (params['E_i'] - mu_V)

    # eq. 15
    sig_V = np.sqrt(
        params['K_e'] * f_e * (U_e * params['tau_syn'])**2 / (2 * (tau_m + params['tau_syn']))
        + params['K_i'] * f_i * (U_i * params['tau_syn'])**2 / (2 * (tau_m + params['tau_syn']))
    )

    # eq. 17
    tau_V = (
        (params['K_e'] * f_e * (U_e * params['tau_syn'])**2
         + params['K_i'] * f_i * (U_i * params['tau_syn'])**2)
        / (params['K_e'] * f_e * (U_e * params['tau_syn'])**2 / (tau_m + params['tau_syn'])
           + params['K_i'] * f_i * (U_i * params['tau_syn'])**2 / (tau_m + params['tau_syn']))
    )
    tau_V_norm = tau_V * params['g_L'] / params['C_m']

    return mu_V, sig_V, tau_V, tau_V_norm
    
# ---------------------------------------------------
def plot_membrane_potential_fluctuations(data=None, mu_V=None, sig_V=None, tau_V=None):
    fig, axs = plt.subplots(4, 1, figsize=(8, 8))

    inh_vals = sorted(data['input_inh'].unique())
    exc_vals = sorted(data['input_exc'].unique())

    n_inh = len(inh_vals)
    n_exc = len(exc_vals)

    out_freq = data['avg_f_out'].to_numpy().reshape(n_inh, n_exc)
    mu_V = mu_V.reshape(n_inh, n_exc)
    sig_V = sig_V.reshape(n_inh, n_exc)
    tau_V = tau_V.reshape(n_inh, n_exc)

    heatmap_kwargs = dict(
        cmap='viridis',
        xticklabels=exc_vals,
        yticklabels=inh_vals
    )

    datasets = [
        (mu_V, 'Mean Membrane Potential [V]'),
        (sig_V, 'Membrane Potential Std [V]'),
        (tau_V, 'Autocorrelation Time [s]'),
        (out_freq, 'Output Frequency [Hz]')
    ]

    for ax, (data_map, title) in zip(axs, datasets):
        sns.heatmap(data_map, ax=ax, **heatmap_kwargs)
        ax.set_title(title)
        ax.set_xlabel('Freq. exc [Hz]')
        ax.set_ylabel('Freq. inh [Hz]')

        # Reduce label density
        step_x = max(1, len(exc_vals)//10)
        step_y = max(1, len(inh_vals)//5)
        ax.set_xticks(np.arange(0.5, n_exc, step_x))
        ax.set_xticklabels(np.round(exc_vals[::step_x], 1), rotation=60)
        ax.set_yticks(np.arange(0.5, n_inh, step_y))
        ax.set_yticklabels(np.round(inh_vals[::step_y], 1), rotation=0)

    plt.tight_layout(h_pad=1.5)
    return fig

# ---------------------------------------------------
def est_thresh(data=None, mu_V=None, sig_V=None, tau_V=None, alpha=None, clip_arg=True):
    F_out = np.asarray(data['avg_f_out'].to_numpy()) + 1e-12
    mu_V = np.asarray(mu_V)
    sig_V = np.asarray(sig_V)
    tau_V = np.asarray(tau_V)

    # safety floors
    sig_V_safe = np.maximum(sig_V, 1e-9)
    tau_V_safe = np.maximum(tau_V, 1e-9)

    arg = (1.0 / alpha) * (F_out * 2.0 * tau_V_safe)  # should be in [0, 2]
    # diagnostic counts
    n_invalid = np.sum((arg <= 0.0) | (arg >= 2.0))
    if n_invalid > 0:
        warnings.warn(
            f"est_thresh: {n_invalid} element(s) of 'arg' outside (0,2). "
            "They will be clipped for erfcinv; consider increasing alpha or checking tau_V."
        )

    if clip_arg:
        # clip into a safe open interval for erfcinv
        arg_clipped = np.clip(arg, 1e-12, 2.0 - 1e-12)
    else:
        arg_clipped = arg

    est_V_th = mu_V + np.sqrt(2.0) * sig_V_safe * sp_spec.erfcinv(arg_clipped)

    return est_V_th

# ---------------------------------------------------
def eff_thresh(mu_V = None, sig_V = None, tau_V_norm = None, poly_params = None):

    ### eq. 4 in Zerlaut et al., (2018)
    P_0, P_mu, P_sig, P_tau, P_mu2, P_sig2, P_tau2, P_mu_sig, P_mu_tau, P_sig_tau = poly_params
    
    # initial conditions
    mu_0  = -60e-3  # V
    sig_0 = 4e-3    # V 
    tau_0 = 0.5     # s
     
    mu_d  = 10e-3   # V
    sig_d = 6e-3    # V
    tau_d = 1.      # s
    
    V_0 = P_0

    V_1 = (P_mu * (mu_V - mu_0) / mu_d +
          P_sig * (sig_V - sig_0) / sig_d +
          P_tau * (tau_V_norm - tau_0) / tau_d)
    
    V_2 = (P_mu2 * ((mu_V - mu_0) / mu_d)**2 +
          P_sig2 * ((sig_V - sig_0) / sig_d)**2 +
          P_tau2 * ((tau_V_norm - tau_0) / tau_d)**2 +
          P_mu_sig * ((mu_V - mu_0) / mu_d) * ((sig_V - sig_0) / sig_d) +
          P_mu_tau * ((mu_V - mu_0) / mu_d) * ((tau_V_norm - tau_0) / tau_d) +
          P_sig_tau * ((sig_V - sig_0) / sig_d) * ((tau_V_norm - tau_0) / tau_d))
    
    return V_0 + V_1 + V_2

# ---------------------------------------------------
def res_1_func(poly_params, mu_V = None, sig_V = None, tau_V_norm = None, est_V_th = None):
    
    eff_V_th = eff_thresh(mu_V = mu_V, sig_V = sig_V, tau_V_norm = tau_V_norm, poly_params = poly_params)
    
    res = np.mean((est_V_th - eff_V_th)**2)
    return res

# ---------------------------------------------------
def res_2_func(poly_params,
               data = None, 
               params = None, 
               alpha = None):
    
    # importing firing frequency from data 
    F_out = data['avg_f_out'].to_numpy()

    f_e = data['input_exc'].to_numpy()
    f_i = data['input_inh'].to_numpy()
  
    res = np.mean((F_out - TF_template(data = data, params = params, poly_params = poly_params, alpha = alpha))**2)
    return res

# ---------------------------------------------------
def TF_template(neuron_model = None,
                data = None,
                params = None,
                poly_params = None,
                alpha = None):

    # compute mu_V, sig_V, tau_V (tau_V in seconds), tau_V_norm
    mu_V, sig_V, tau_V, tau_V_norm = membrane_potential_fluctuations(neuron_model = neuron_model, data=data, params=params)
    
    # ensure arrays and shapes match
    mu_V = np.asarray(mu_V)
    sig_V = np.asarray(sig_V)
    tau_V = np.asarray(tau_V)

    # protect against zero/very small sigma and tau
    sig_V_safe = np.maximum(sig_V, 1e-9)   # 1e-9 V -> 1 uV, tune as needed
    tau_V_safe = np.maximum(tau_V, 1e-9)   # seconds

    eff_V_th = eff_thresh(mu_V = mu_V, sig_V = sig_V_safe, tau_V_norm = tau_V_norm, poly_params=poly_params)

    z = (eff_V_th - mu_V) / (np.sqrt(2.0) * sig_V_safe)  # dimensionless
    # compute erfc; it returns in [0,2]
    F_out_th = alpha * sp_spec.erfc(z) / (2.0 * tau_V_safe)

    # numerical safety: rates cannot be negative
    F_out_th = np.maximum(F_out_th, 0.0)

    return F_out_th


# ---------------------------------------------------
def TF_template_sim(neuron_model = None,
                    f_e = 0.0,
                    f_i = 0.0,
                    params = None,
                    poly_params = None,
                    alpha = None,
                    w_ad = 0.0
                    ):
    """
    Compute the output rate for one population given 'frequency inputs'.
    """

    # compute membrane potential statistics
    mu_V, sig_V, tau_V, tau_V_norm = membrane_potential_fluctuations_sim(
                                     neuron_model = neuron_model,
                                     f_e = f_e,
                                     f_i = f_i,
                                     params = params,
                                     w_ad = w_ad
                                     )

    # protect against numerical issues
    sig_V_safe = max(sig_V, 1e-9)
    tau_V_safe = max(tau_V, 1e-9)

    # effective threshold
    eff_V_th = eff_thresh(
        mu_V = mu_V,
        sig_V = sig_V_safe,
        tau_V_norm = tau_V_norm,
        poly_params = poly_params
    )

    # transfer function (Zerlaut et al. 2018)
    z = (eff_V_th - mu_V) / (np.sqrt(2.0) * sig_V_safe)
    F_out_th = alpha * sp_spec.erfc(z) / (2.0 * tau_V_safe)

    return max(F_out_th, 0.0)
    #return max(F_out_th, 0.0), mu_v # when adaptation !=0, work in progress
