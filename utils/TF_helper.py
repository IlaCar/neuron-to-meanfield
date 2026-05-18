import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import erfc, erfcinv
import pandas as pd
import json
import seaborn as sns
import scipy.special as sp_spec
import warnings
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# ---------------------------------------------------

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
def membrane_potential_fluctuations(neuron_model = None, data = None, params = None, w_ad = None):
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
    
    # w_ad can be passed, if not, by default is set to an array of zeros
    if w_ad is None:
        w_ad = np.zeros(len(f_i))
    else:
        w_ad = np.asarray(w_ad)
        
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
                alpha = None,
                w_ad = None):

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

    return max(F_out_th, 0.0), mu_v

# -------------------- #
def get_mean_error_distribution(neuron_model, df_data, poly_params_2, params_SI, alpha, unique_inh, alpha_idx = None):
    
    distr_mean_error = np.zeros(len(unique_inh))
    idx = 0
    for fixed_inh in unique_inh:

        # Select a fixed inhibitory input
        tol = 1e-6  # tolerance in case of floating point noise
        mask = np.isclose(df_data['input_inh'], fixed_inh, atol=tol)
        
        inp_exc = df_data.loc[mask, 'input_exc'].to_numpy()
        out_rate = df_data.loc[mask, 'avg_f_out'].to_numpy()

        if alpha_idx == None:
            fit_rate = df_data.loc[mask,'fit_rate']
        else:
            fit_rate = df_data.loc[mask,f'fit_rate_alpha_{alpha_idx}']
        mean_error = res_2_func(poly_params_2, data=df_data.loc[mask], params=params_SI, alpha=alpha)

        distr_mean_error[idx] = mean_error
        idx += 1
                    
    return distr_mean_error

# ---------------------------------------------------
def simulate_MF_FS_RS(time=None,
                      neuron_models=None,
                      params=None, 
                      poly_params=None,
                      alphas=None,
                      network_config=None,
                      driving_input=None):
    """
    Simulate a 2-population mean-field network (FS, RS)
    """

    dt = time[1] - time[0]
    n_steps = len(time)
    pops = neuron_models

    # Initialize firing rates
    rates = {pop: np.zeros(n_steps) for pop in pops}
    rates['FS'][0] = 0
    rates['RS'][0] = 0
    tau_f = 0.01  # 10 ms population time constant

    # --- network structure ---
    N_FS = network_config['network_composition']['FS_neuron']
    N_RS = network_config['network_composition']['RS_neuron']

    p_ext = network_config['external_input']['conn_prob']
    N_exc = network_config['external_input']['N_external_exc']
    N_inh = network_config['external_input']['N_external_inh']
    K_ext_exc = N_exc * p_ext   # 400
    K_ext_inh = N_inh * p_ext   # 100

    p = network_config['network_composition']['conn_prob']
    K_RS_to_RS = int(p * N_RS)  # 400
    K_RS_to_FS = int(p * N_RS)  # 400
    K_FS_to_RS = int(p * N_FS)  # 100
    K_FS_to_FS = int(p * N_FS)  # 100

    K_ref_exc = K_RS_to_RS
    K_ref_inh = K_FS_to_FS
    
    # --- quantal conductances ---
    # - external inputs -
    Qe_ext = network_config['external_input']['Q_e']* 10**-9   # nS * e-9-> S
    Qi_ext = network_config['external_input']['Q_i']* 10**-9   # nS * e-9-> S
  
    # - RS quantal conductances -
    Qe_RS_RS = params['RS']['Q_e']  
    Qe_RS_FS = params['FS']['Q_e'] 
    
    # - FS quantal conductances -
    Qi_FS_FS = params['FS']['Q_i']  # S 
    Qi_FS_RS = params['RS']['Q_i']

    Qe_RS_ref_exc = Qe_RS_RS
    Qi_RS_ref_inh = Qi_FS_RS
    
    Qe_FS_ref_exc = Qe_RS_FS
    Qi_FS_ref_inh = Qi_FS_FS
    
    # Loop over time
    for t in range(1, n_steps):
        current_rates = {pop: rates[pop][t-1] for pop in pops}

        # external drives
        nu_ext_exc_FS = driving_input['excitatory']['FS'][t]
        nu_ext_exc_RS = driving_input['excitatory']['RS'][t]
        nu_ext_inh_FS = driving_input['inhibitory']['FS'][t]
        nu_ext_inh_RS = driving_input['inhibitory']['RS'][t]
        
        # --- effective excitatory and inhibitory rates ---
        nu_eff_exc_RS = (
            K_ext_exc * Qe_ext * nu_ext_exc_RS +
            K_RS_to_RS * Qe_RS_RS * current_rates['RS']
        ) / (K_ref_exc * Qe_RS_ref_exc)
        
        ## TO DO !!! ##
        #it is important to notice that in case tau_ext != tau_e != tau_ref then they need to be written
        ''' 
        nu_eff_exc_RS = (
            K_ext_exc * Qe_ext * tau_ext * nu_ext_exc_RS +
            K_RS_to_RS * Qe_RS_RS * tau_e * current_rates['RS']
        ) / (K_ref_exc * Qe * tau_ref)
        '''
        
        nu_eff_inh_RS = (
            K_ext_inh * Qi_ext * nu_ext_inh_RS +
            K_FS_to_RS * Qi_FS_RS * current_rates['FS']
        ) / (K_ref_inh * Qi_RS_ref_inh)

        nu_eff_exc_FS = (
            K_ext_exc * Qe_ext * nu_ext_exc_FS +
            K_RS_to_FS * Qe_RS_FS * current_rates['RS']
        ) / (K_ref_exc * Qe_FS_ref_exc)

        nu_eff_inh_FS = (
            K_ext_inh * Qi_ext * nu_ext_inh_FS +
            K_FS_to_FS * Qi_FS_FS * current_rates['FS']
        ) / (K_ref_inh * Qi_FS_ref_inh)

        # --- evaluate transfer functions ---
        F_RS, mu_RS = TF_template_sim(
            neuron_model='RS',
            f_e=nu_eff_exc_RS,
            f_i=nu_eff_inh_RS,
            params=params['RS'],
            poly_params=poly_params['RS'],
            alpha=alphas['RS'],
            w_ad=0.0
        )

        F_FS, mu_FS = TF_template_sim(
            neuron_model='FS',
            f_e=nu_eff_exc_FS,
            f_i=nu_eff_inh_FS,
            params=params['FS'],
            poly_params=poly_params['FS'],
            alpha=alphas['FS'],
            w_ad=0.0
        )

        # --- integrate rate dynamics ---
        rates['RS'][t] = rates['RS'][t-1] + dt / tau_f * (F_RS - rates['RS'][t-1])
        rates['FS'][t] = rates['FS'][t-1] + dt / tau_f * (F_FS - rates['FS'][t-1])

    return rates

# ---------------------------------------------------
def simulate_MF_FS_RS_solve_ivp(time=None,
                      neuron_models=None,
                      params=None, 
                      poly_params=None,
                      alphas=None,
                      network_config=None,
                      driving_input=None):
    """
    Simulate a 2-population mean-field network (FS, RS) using SciPy's solve_ivp.
    """
    
    tau_f = 0.01  # 10 ms population time constant

    # --- network structure ---
    N_FS = network_config['network_composition']['FS_neuron']
    N_RS = network_config['network_composition']['RS_neuron']

    p_ext = network_config['external_input']['conn_prob']
    N_exc = network_config['external_input']['N_external_exc']
    N_inh = network_config['external_input']['N_external_inh']
    K_ext_exc = N_exc * p_ext
    K_ext_inh = N_inh * p_ext

    p = network_config['network_composition']['conn_prob']
    K_RS_to_RS = int(p * N_RS)
    K_RS_to_FS = int(p * N_RS)
    K_FS_to_RS = int(p * N_FS)
    K_FS_to_FS = int(p * N_FS)

    K_ref_exc = K_RS_to_RS
    K_ref_inh = K_FS_to_FS
    
    # --- quantal conductances ---
    Qe_ext = network_config['external_input']['Q_e'] * 10**-9  # nS to S
    Qi_ext = network_config['external_input']['Q_i'] * 10**-9  # nS to S
  
    # Using the passed 'params' dict instead of global variables
    Qe_RS_RS = params['RS']['Q_e']  
    Qe_RS_FS = params['FS']['Q_e'] 
    Qi_FS_FS = params['FS']['Q_i']  
    Qi_FS_RS = params['RS']['Q_i']

    Qe_RS_ref_exc = Qe_RS_RS
    Qi_RS_ref_inh = Qi_FS_RS
    
    Qe_FS_ref_exc = Qe_RS_FS
    Qi_FS_ref_inh = Qi_FS_FS

    # --- Setup Continuous Driving Inputs ---
    # solve_ivp evaluates at arbitrary time points, so we interpolate the discrete driving inputs.
    drive_exc_FS = interp1d(time, driving_input['excitatory']['FS'], bounds_error=False, fill_value="extrapolate")
    drive_exc_RS = interp1d(time, driving_input['excitatory']['RS'], bounds_error=False, fill_value="extrapolate")
    drive_inh_FS = interp1d(time, driving_input['inhibitory']['FS'], bounds_error=False, fill_value="extrapolate")
    drive_inh_RS = interp1d(time, driving_input['inhibitory']['RS'], bounds_error=False, fill_value="extrapolate")

    # --- Define the ODE system ---
    def mean_field_derivatives(t, y):
        # y state vector: [rate_RS, rate_FS]
        rate_RS, rate_FS = y

        # Interpolate external drives at current integration time 't'
        nu_ext_exc_FS = drive_exc_FS(t)
        nu_ext_exc_RS = drive_exc_RS(t)
        nu_ext_inh_FS = drive_inh_FS(t)
        nu_ext_inh_RS = drive_inh_RS(t)
        
        # Effective rates
        nu_eff_exc_RS = (
            K_ext_exc * Qe_ext * nu_ext_exc_RS +
            K_RS_to_RS * Qe_RS_RS * rate_RS
        ) / (K_ref_exc * Qe_RS_ref_exc)
        
        nu_eff_inh_RS = (
            K_ext_inh * Qi_ext * nu_ext_inh_RS +
            K_FS_to_RS * Qi_FS_RS * rate_FS
        ) / (K_ref_inh * Qi_RS_ref_inh)

        nu_eff_exc_FS = (
            K_ext_exc * Qe_ext * nu_ext_exc_FS +
            K_RS_to_FS * Qe_RS_FS * rate_RS
        ) / (K_ref_exc * Qe_FS_ref_exc)

        nu_eff_inh_FS = (
            K_ext_inh * Qi_ext * nu_ext_inh_FS +
            K_FS_to_FS * Qi_FS_FS * rate_FS
        ) / (K_ref_inh * Qi_FS_ref_inh)

        # Transfer functions
        F_RS, mu_RS = TF_template_sim(
            neuron_model='RS', f_e=nu_eff_exc_RS, f_i=nu_eff_inh_RS,
            params=params['RS'], poly_params=poly_params['RS'],
            alpha=alphas['RS'], w_ad=0.0
        )

        F_FS, mu_FS = TF_template_sim(
            neuron_model='FS', f_e=nu_eff_exc_FS, f_i=nu_eff_inh_FS,
            params=params['FS'], poly_params=poly_params['FS'],
            alpha=alphas['FS'], w_ad=0.0
        )

        # Calculate derivatives (d/dt)
        dRS_dt = (F_RS - rate_RS) / tau_f
        dFS_dt = (F_FS - rate_FS) / tau_f

        return [dRS_dt, dFS_dt]

    # --- Integrate ---
    # Initial conditions: [RS_init, FS_init]
    y0 = [0.0, 0.0] 
    
    # solve_ivp handles the time stepping automatically
    solution = solve_ivp(
        fun=mean_field_derivatives,
        t_span=(time[0], time[-1]),
        y0=y0,
        t_eval=time,         # Forces the solver to output results exactly matching the time array
        method='RK45'        # Standard Runge-Kutta. Switch to 'LSODA' if the system becomes stiff
    )

    # --- Format Output ---
    # Reconstruct the rates dictionary
    rates = {
        'RS': solution.y[0],
        'FS': solution.y[1]
    }

    return rates

# ---------------------------------------------------
def generate_ou_process(time, dt, mu, tau, sigma, x0=None):
    x = np.zeros_like(time)
    x[0] = x0 if x0 is not None else mu
    np.random.seed(0)
    
    # Pre-generate Gaussian noise for efficiency
    noise = np.random.normal(0, 1, len(time))
    
    for i in range(1, len(time)):
        dx = ((mu - x[i-1]) / tau) * dt + sigma * np.sqrt(dt) * noise[i]
        x[i] = x[i-1] + dx
        
    # Rectify to prevent negative firing rates
    return np.maximum(0, x)
   
# ---------------------------------------------------
def simulate_MF_FS_RS_adapt_solve_ivp(time=None,
                      neuron_models=None,
                      params=None, 
                      poly_params=None,
                      alphas=None,
                      network_config=None,
                      driving_input=None):
    """
    Simulate a 2-population mean-field network (FS, RS) using SciPy's solve_ivp,
    including the dynamics of the adaptation variable (w_ad).
    """
    
    tau_f = 0.01  # 10 ms population time constant

    # --- network structure ---
    N_FS = network_config['network_composition']['FS_neuron']
    N_RS = network_config['network_composition']['RS_neuron']

    p_ext = network_config['external_input']['conn_prob']
    N_exc = network_config['external_input']['N_external_exc']
    N_inh = network_config['external_input']['N_external_inh']
    K_ext_exc = N_exc * p_ext
    K_ext_inh = N_inh * p_ext

    p = network_config['network_composition']['conn_prob']
    K_RS_to_RS = int(p * N_RS)
    K_RS_to_FS = int(p * N_RS)
    K_FS_to_RS = int(p * N_FS)
    K_FS_to_FS = int(p * N_FS)

    K_ref_exc = K_RS_to_RS
    K_ref_inh = K_FS_to_FS
    
    # --- quantal conductances ---
    Qe_ext = network_config['external_input']['Q_e'] * 10**-9
    Qi_ext = network_config['external_input']['Q_i'] * 10**-9
  
    Qe_RS_RS = params['RS']['Q_e']  
    Qe_RS_FS = params['FS']['Q_e'] 
    Qi_FS_FS = params['FS']['Q_i']  
    Qi_FS_RS = params['RS']['Q_i']

    Qe_RS_ref_exc = Qe_RS_RS
    Qi_RS_ref_inh = Qi_FS_RS
    
    Qe_FS_ref_exc = Qe_RS_FS
    Qi_FS_ref_inh = Qi_FS_FS

    # --- Setup Continuous Driving Inputs ---
    drive_exc_FS = interp1d(time, driving_input['excitatory']['FS'], bounds_error=False, fill_value="extrapolate")
    drive_exc_RS = interp1d(time, driving_input['excitatory']['RS'], bounds_error=False, fill_value="extrapolate")
    drive_inh_FS = interp1d(time, driving_input['inhibitory']['FS'], bounds_error=False, fill_value="extrapolate")
    drive_inh_RS = interp1d(time, driving_input['inhibitory']['RS'], bounds_error=False, fill_value="extrapolate")

    # --- Define the ODE system ---
    def mean_field_derivatives(t, y):
        # y state vector: [rate_RS, rate_FS, w_RS, w_FS]
        rate_RS, rate_FS, w_RS, w_FS = y

        # Interpolate external drives at current integration time 't'
        nu_ext_exc_FS = drive_exc_FS(t)
        nu_ext_exc_RS = drive_exc_RS(t)
        nu_ext_inh_FS = drive_inh_FS(t)
        nu_ext_inh_RS = drive_inh_RS(t)
        
        # Effective rates
        nu_eff_exc_RS = (
            K_ext_exc * Qe_ext * nu_ext_exc_RS +
            K_RS_to_RS * Qe_RS_RS * rate_RS
        ) / (K_ref_exc * Qe_RS_ref_exc)
        
        nu_eff_inh_RS = (
            K_ext_inh * Qi_ext * nu_ext_inh_RS +
            K_FS_to_RS * Qi_FS_RS * rate_FS
        ) / (K_ref_inh * Qi_RS_ref_inh)

        nu_eff_exc_FS = (
            K_ext_exc * Qe_ext * nu_ext_exc_FS +
            K_RS_to_FS * Qe_RS_FS * rate_RS
        ) / (K_ref_exc * Qe_FS_ref_exc)

        nu_eff_inh_FS = (
            K_ext_inh * Qi_ext * nu_ext_inh_FS +
            K_FS_to_FS * Qi_FS_FS * rate_FS
        ) / (K_ref_inh * Qi_FS_ref_inh)

        # Transfer functions (Pass the current adaptation state w_RS / w_FS)
        ### adding mu_RS
        F_RS, mu_RS = TF_template_sim(
            neuron_model='RS', f_e=nu_eff_exc_RS, f_i=nu_eff_inh_RS,
            params=params['RS'], poly_params=poly_params['RS'],
            alpha=alphas['RS'], w_ad=w_RS
        )
        ### adding mu_FS
        F_FS, mu_FS = TF_template_sim(
            neuron_model='FS', f_e=nu_eff_exc_FS, f_i=nu_eff_inh_FS,
            params=params['FS'], poly_params=poly_params['FS'],
            alpha=alphas['FS'], w_ad=w_FS
        )

        # 1. Rate derivatives (d/dt)
        dRS_dt = (F_RS - rate_RS) / tau_f
        dFS_dt = (F_FS - rate_FS) / tau_f

        # 2. Adaptation derivatives (dW/dt = -W/tau_w + b * rate)
        # We use .get() so it safely defaults to 0 if the cell type doesn't have adaptation parameters
        tau_w_RS = params['RS'].get('tau_w', 1.0) # avoid div by zero, default tau to 1.0 if missing
        b_RS = params['RS'].get('b', 0.0)
        a_RS = params['RS'].get('a', 0.0)
        E_L_RS = params['RS']['E_L']
        
        dW_RS_dt = (
            -w_RS / tau_w_RS
            + b_RS * rate_RS
            + a_RS * (mu_RS - E_L_RS) / tau_w_RS
        )
                
        tau_w_FS = params['FS'].get('tau_w', 1.0)
        b_FS = params['FS'].get('b', 0.0)
        a_FS = params['FS'].get('a', 0.0)
        E_L_FS = params['FS']['E_L']
        
        dW_FS_dt = (
            -w_FS / tau_w_FS
            + b_FS * rate_FS
            + a_FS * (mu_FS - E_L_FS) / tau_w_FS
        )
        return [dRS_dt, dFS_dt, dW_RS_dt, dW_FS_dt]

    # --- Integrate ---
    # Initial conditions: [RS_init, FS_init, W_RS_init, W_FS_init]
    y0 = [0.0, 0.0, 0.0, 0.0] 
    
    solution = solve_ivp(
        fun=mean_field_derivatives,
        t_span=(time[0], time[-1]),
        y0=y0,
        t_eval=time,         
        method='RK45'        
    )

    # --- Format Output ---
    rates = {
        'RS': solution.y[0],
        'FS': solution.y[1]
    }
    
    # Returning also the adaptation traces
    adaptation = {
        'RS': solution.y[2],
        'FS': solution.y[3]
    }

    return rates, adaptation
