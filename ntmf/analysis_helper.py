"""Statistical helpers for analysing ensembles of fitted transfer-function parameters.

These are pure-analysis routines used by the
``Analysing_params_distribution_{FS,RS,RS_no_adapt}`` notebooks.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# -------------------- #
def coefficient_stats(poly_params, params_name):
    """Summarise an ensemble of fitted polynomial coefficients.

    Parameters
    ----------
    poly_params : ndarray of shape (n_models, n_params)
        Fitted polynomial coefficients, one row per retained model.
    params_name : list of str
        Coefficient names, ordered as the columns of `poly_params`.

    Returns
    -------
    pandas.DataFrame
        Per-coefficient mean, standard deviation and coefficient of
        variation (|std / mean|), sorted from stiff (low CV) to sloppy
        (high CV).
    """
    means = poly_params.mean(axis=0)
    stds = poly_params.std(axis=0, ddof=1)

    df = pd.DataFrame({
        "param_id": np.arange(len(params_name)),
        "params_name": params_name,
        "mean": means,
        "std": stds,
        "cv": np.abs(stds / means),
    })
    return df.sort_values("cv").reset_index(drop=True)

# -------------------- #
def alpha_trend(alphas, poly_params, params_name, deg=2):
    """Quantify how much of each coefficient's variance is a function of alpha.

    A polynomial of degree `deg` is fitted to coefficient(alpha) and the
    fraction of variance it explains is reported. A high R^2 means the
    coefficient is not an independent degree of freedom but is slaved to
    alpha through the fit.

    Parameters
    ----------
    alphas : ndarray of shape (n_models,)
        Fitted alpha of each retained model.
    poly_params : ndarray of shape (n_models, n_params)
    params_name : list of str
    deg : int, optional
        Degree of the trend fitted in alpha. Default 2.

    Returns
    -------
    pandas.DataFrame
        Per-coefficient Pearson r with alpha (and its p-value) and the
        R^2 of the degree-`deg` trend in alpha.
    """
    rows = []
    for i, name in enumerate(params_name):
        y = poly_params[:, i]
        r, p = pearsonr(alphas, y)
        resid = y - np.polyval(np.polyfit(alphas, y, deg), alphas)
        r2 = 1.0 - resid.var() / y.var()
        rows.append([name, r, p, r2])

    return pd.DataFrame(rows, columns=["params_name", "r_alpha", "p_alpha", f"R2_deg{deg}_alpha"])

# -------------------- #
def detrend_alpha(alphas, values, deg=1):
    """Remove the trend in alpha from one or several variables.

    Parameters
    ----------
    alphas : ndarray of shape (n_models,)
    values : ndarray of shape (n_models,) or (n_models, n_params)
    deg : int, optional
        Degree of the trend removed. Default 1.

    Returns
    -------
    ndarray
        Residuals, same shape as `values`.
    """
    values = np.asarray(values)
    if values.ndim == 1:
        return values - np.polyval(np.polyfit(alphas, values, deg), alphas)

    return np.column_stack([
        values[:, i] - np.polyval(np.polyfit(alphas, values[:, i], deg), alphas)
        for i in range(values.shape[1])
    ])

# -------------------- #
def correlation_with_error(poly_params, errors, alphas, params_name, deg=1):
    """Correlate each coefficient with the fitting error, raw and controlling for alpha.

    The raw correlation is confounded whenever the coefficients are slaved
    to alpha: both quantities are then functions of the same variable. The
    partial correlation removes the alpha trend from both sides first.

    Parameters
    ----------
    poly_params : ndarray of shape (n_models, n_params)
    errors : ndarray of shape (n_models,)
    alphas : ndarray of shape (n_models,)
    params_name : list of str
    deg : int, optional
        Degree of the alpha trend removed. Default 1.

    Returns
    -------
    pandas.DataFrame
        Raw and alpha-controlled Pearson r with the mean error, and their
        p-values.
    """
    err_resid = detrend_alpha(alphas, errors, deg=deg)
    par_resid = detrend_alpha(alphas, poly_params, deg=deg)

    rows = []
    for i, name in enumerate(params_name):
        r_raw, p_raw = pearsonr(poly_params[:, i], errors)
        r_par, p_par = pearsonr(par_resid[:, i], err_resid)
        rows.append([name, r_raw, p_raw, r_par, p_par])

    return pd.DataFrame(rows, columns=["params_name", "r_raw", "p_raw", "r_partial", "p_partial"])

# -------------------- #
def collinearity_summary(corr_matrix):
    """Summarise the off-diagonal structure of a coefficient correlation matrix.

    Parameters
    ----------
    corr_matrix : ndarray of shape (n_params, n_params)

    Returns
    -------
    dict
        Median, minimum and maximum |r| off the diagonal, and the fraction
        of pairs with |r| > 0.8.
    """
    off = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
    return {
        "median_abs_r": float(np.median(np.abs(off))),
        "min_abs_r": float(np.abs(off).min()),
        "max_abs_r": float(np.abs(off).max()),
        "frac_abs_r_above_0.8": float((np.abs(off) > 0.8).mean()),
    }

# -------------------- #
def detect_clusters(poly_params, n_clusters='auto', max_clusters=4, min_silhouette=0.55):
    """Detect discrete solution clusters in an ensemble of fitted coefficients.

    A fitting search may converge to several distinct basins rather than to a
    single solution sampled repeatedly. When that happens the rows of
    `poly_params` are not independent draws: the effective sample size is the
    number of clusters, not the number of rows, and any statistic computed with
    the latter is invalid.

    Clusters are found by Ward clustering of the z-scored coefficients. The
    number of clusters is chosen by silhouette score unless fixed by the caller;
    if no partition reaches `min_silhouette` the ensemble is reported as a single
    cluster.

    Parameters
    ----------
    poly_params : ndarray of shape (n_models, n_params)
    n_clusters : int or 'auto', optional
        Number of clusters. If 'auto', selected by silhouette score.
    max_clusters : int, optional
        Largest number of clusters considered when `n_clusters` is 'auto'.
    min_silhouette : float, optional
        Silhouette below which the ensemble is declared unbranched.

    Returns
    -------
    labels : ndarray of shape (n_models,)
        Cluster index of each model, ordered by increasing mean PC1 score so that
        the labelling is reproducible across runs.
    diagnostics : pandas.DataFrame
        Silhouette score for each candidate number of clusters.
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA

    n_models = poly_params.shape[0]
    Z = (poly_params - poly_params.mean(axis=0)) / poly_params.std(axis=0, ddof=1)

    k_max = min(max_clusters, n_models - 1)
    rows = []
    for k in range(2, k_max + 1):
        lab = AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(Z)
        rows.append([k, silhouette_score(Z, lab)])
    diagnostics = pd.DataFrame(rows, columns=["n_clusters", "silhouette"])

    if n_clusters == 'auto':
        best = diagnostics.loc[diagnostics["silhouette"].idxmax()]
        k = int(best["n_clusters"]) if best["silhouette"] >= min_silhouette else 1
    else:
        k = int(n_clusters)

    if k == 1:
        return np.zeros(n_models, dtype=int), diagnostics

    labels = AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(Z)

    # Relabel by increasing mean PC1 score, so the labels do not depend on the
    # arbitrary order returned by the clustering.
    pc1 = PCA(n_components=1).fit_transform(Z).ravel()
    order = np.argsort([pc1[labels == j].mean() for j in range(k)])
    remap = {old: new for new, old in enumerate(order)}

    return np.array([remap[l] for l in labels]), diagnostics

# -------------------- #
def cluster_summary(labels, alphas, errors, poly_params, params_name):
    """Summarise each detected cluster and the spread within it.

    Parameters
    ----------
    labels : ndarray of shape (n_models,)
        Cluster index per model, from `detect_clusters`.
    alphas, errors : ndarray of shape (n_models,)
    poly_params : ndarray of shape (n_models, n_params)
    params_name : list of str

    Returns
    -------
    summary : pandas.DataFrame
        One row per cluster: size, alpha range, error mean and spread, and the
        within-cluster correlation between alpha and the error.
    coeff_means : pandas.DataFrame
        Mean of each coefficient per cluster, with a flag marking the
        coefficients that change sign between clusters.
    """
    rows = []
    for j in np.unique(labels):
        m = labels == j
        if m.sum() > 2:
            r, p = pearsonr(alphas[m], errors[m])
        else:
            r, p = np.nan, np.nan
        rows.append([j, int(m.sum()), alphas[m].min(), alphas[m].max(),
                     errors[m].mean(), errors[m].std(ddof=1) if m.sum() > 1 else np.nan,
                     r, p])
    summary = pd.DataFrame(rows, columns=["cluster", "n", "alpha_min", "alpha_max",
                                          "error_mean", "error_std",
                                          "r_alpha_error_within", "p_alpha_error_within"])

    coeff_means = pd.DataFrame({"params_name": params_name})
    for j in np.unique(labels):
        coeff_means[f"cluster_{j}"] = poly_params[labels == j].mean(axis=0)

    cluster_cols = [c for c in coeff_means.columns if c.startswith("cluster_")]
    signs = np.sign(coeff_means[cluster_cols].to_numpy())
    coeff_means["sign_flip"] = np.where(
        (signs != signs[:, [0]]).any(axis=1), "SIGN FLIP", "")

    return summary, coeff_means

# -------------------- #
def align_pca_signs(components, reference=None):
    """Fix the arbitrary sign of PCA components.

    The sign of a principal component is not determined by the decomposition, so
    loadings cannot be compared across neuron models until a convention is
    imposed. Each component is oriented so that its largest-magnitude loading is
    positive, or so that it correlates positively with a reference.

    Parameters
    ----------
    components : ndarray of shape (n_components, n_params)
        `pca.components_`.
    reference : ndarray of shape (n_components, n_params), optional
        Components to align against, typically those of a chosen neuron model.

    Returns
    -------
    ndarray
        `components` with signs fixed, same shape.
    """
    out = np.array(components, dtype=float, copy=True)
    for k in range(out.shape[0]):
        if reference is None:
            flip = out[k, np.argmax(np.abs(out[k]))] < 0
        else:
            flip = np.dot(out[k], reference[k]) < 0
        if flip:
            out[k] *= -1.0
    return out

# -------------------- #
def pooled_pca(param_dict, n_components=3):
    """Fit one PCA on the pooled ensembles so that models share a basis.

    Scores from separate PCAs live in different bases and cannot be plotted on
    shared axes. This fits a single decomposition on all ensembles at once and
    projects each into that common basis.

    Coefficients are standardised using the pooled mean and standard deviation,
    so that the scores of the different neuron models remain comparable.

    Parameters
    ----------
    param_dict : dict of {str: ndarray of shape (n_models, n_params)}
        Fitted coefficients per neuron model.
    n_components : int, optional

    Returns
    -------
    pca : sklearn.decomposition.PCA
        The decomposition fitted on the pooled, standardised data.
    scores : dict of {str: ndarray of shape (n_models, n_components)}
        Scores of each neuron model in the common basis.
    """
    from sklearn.decomposition import PCA

    keys = list(param_dict.keys())
    stacked = np.vstack([param_dict[k] for k in keys])
    mu, sd = stacked.mean(axis=0), stacked.std(axis=0, ddof=1)
    Z = (stacked - mu) / sd

    pca = PCA(n_components=min(n_components, min(Z.shape)))
    pca.fit(Z)

    scores, i = {}, 0
    for k in keys:
        n = param_dict[k].shape[0]
        scores[k] = pca.transform(Z[i:i + n])
        i += n

    return pca, scores

# -------------------- #
def diagnostic_report(neuron_model, poly_params, alphas, errors, params_name,
                      labels=None, expl_var=None, envelope_frac=None):
    """Print a compact identifiability report for one fitted ensemble.

    Intended as the summary a pipeline user reads before deciding whether the
    fitted coefficients can be interpreted individually.

    Parameters
    ----------
    neuron_model : str
    poly_params : ndarray of shape (n_models, n_params)
    alphas, errors : ndarray of shape (n_models,)
    params_name : list of str
    labels : ndarray of shape (n_models,), optional
        Cluster labels from `detect_clusters`.
    expl_var : ndarray, optional
        `pca.explained_variance_ratio_`.
    envelope_frac : float, optional
        Fraction of grid points where the ensemble spread falls below the
        simulated dispersion.

    Returns
    -------
    None
    """
    n_models = poly_params.shape[0]
    stats = coefficient_stats(poly_params, params_name)
    C = np.corrcoef(((poly_params - poly_params.mean(0)) / poly_params.std(0, ddof=1)).T)
    coll = collinearity_summary(C)

    line = "=" * 62
    print(line)
    print(f"  IDENTIFIABILITY REPORT  --  {neuron_model}   ({n_models} retained fits)")
    print(line)

    print(f"\n  Error range          : {errors.min():.3f} - {errors.max():.3f} "
          f"({100 * np.ptp(errors) / errors.min():.1f}% spread)")
    print(f"  Alpha range          : {alphas.min():.3f} - {alphas.max():.3f}")

    print(f"\n  Stiffest coefficient : {stats.iloc[0].params_name} "
          f"(CV = {100 * stats.iloc[0].cv:.1f}%)")
    print(f"  Sloppiest            : {stats.iloc[-1].params_name} "
          f"(CV = {100 * stats.iloc[-1].cv:.0f}%)")
    print(f"  Collinearity         : median |r| = {coll['median_abs_r']:.2f}, "
          f"{100 * coll['frac_abs_r_above_0.8']:.0f}% of pairs above 0.8")

    if expl_var is not None:
        n95 = int(np.searchsorted(np.cumsum(expl_var), 0.95) + 1)
        print(f"  Effective dimension  : {n95} PC(s) reach 95% of the variance "
              f"(PC1 alone = {100 * expl_var[0]:.1f}%)")

    if labels is not None:
        k = len(np.unique(labels))
        print(f"\n  Clusters detected    : {k}")
        if k > 1:
            sizes = ", ".join(str(int((labels == j).sum())) for j in np.unique(labels))
            print(f"  Cluster sizes         : {sizes}")
            print(f"  >> EFFECTIVE SAMPLE SIZE IS {k}, NOT {n_models}.")
            print( "     Correlations and p-values over the rows are not valid.")
        else:
            print( "     The ensemble is a single basin sampled repeatedly.")

    if envelope_frac is not None:
        print(f"\n  Fits indistinguishable on {100 * envelope_frac:.1f}% of the input grid")
        if envelope_frac < 0.9:
            print( "     >> Check WHERE they diverge: that region bounds the")
            print( "        domain of validity of the fitted transfer function.")

    print(f"\n  {line[:58]}")
    print( "  Reading: high collinearity + one dominant PC means the fitted")
    print( "  coefficients are not individually identifiable. Report the fitted")
    print( "  SURFACE and its domain of validity, not the coefficient values.")
    print(line)