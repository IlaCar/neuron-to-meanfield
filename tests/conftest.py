"""Shared test fixtures for the neuron-to-meanfield test suite."""

from pathlib import Path

import json
import numpy as np
import pandas as pd
import pytest

from ntmf.config import (
    adding_K_params,
    get_network_config,
    get_params_model_SI,
)

# ---- paths ----------------------------------------------------------------

@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def neuron_models_dir(project_root: Path) -> Path:
    return project_root / "neuron_models" / "AdEx"


@pytest.fixture(scope="session")
def config_dir(project_root: Path) -> Path:
    return project_root / "config"


@pytest.fixture(scope="session")
def tf_data_dir(project_root: Path) -> Path:
    return project_root / "transfer_function" / "data_extraction" / "simulations"


@pytest.fixture(scope="session")
def validation_dir(project_root: Path) -> Path:
    return project_root / "transfer_function" / "validation"


# ---- neuron params (SI) ---------------------------------------------------

@pytest.fixture(scope="session")
def params_FS(neuron_models_dir: Path) -> dict:
    return get_params_model_SI("FS", neuron_models_dir / "FS.json")


@pytest.fixture(scope="session")
def params_RS(neuron_models_dir: Path) -> dict:
    return get_params_model_SI("RS", neuron_models_dir / "RS.json")


@pytest.fixture(scope="session")
def params_RS_na(neuron_models_dir: Path) -> dict:
    return get_params_model_SI("RS_no_adapt", neuron_models_dir / "RS_no_adapt.json")


# ---- network config + K params --------------------------------------------

@pytest.fixture(scope="session")
def network_config(config_dir: Path) -> dict:
    return get_network_config(config_dir / "network_config_file_v0.json")


@pytest.fixture(scope="session")
def params_FS_with_K(params_FS: dict, network_config: dict) -> dict:
    return adding_K_params(params_FS.copy(), network_config)


@pytest.fixture(scope="session")
def params_RS_with_K(params_RS: dict, network_config: dict) -> dict:
    return adding_K_params(params_RS.copy(), network_config)


@pytest.fixture(scope="session")
def params_RS_na_with_K(params_RS_na: dict, network_config: dict) -> dict:
    return adding_K_params(params_RS_na.copy(), network_config)


# ---- existing simulation data (.dat files) --------------------------------

@pytest.fixture(scope="session")
def df_FS(tf_data_dir: Path) -> pd.DataFrame:
    path = tf_data_dir / "TF_FS_v1" / "testing_TF_data_FS.dat"
    if not path.exists():
        pytest.skip("FS transfer-function data file not found")
    return pd.read_csv(path, sep=r"\s+", header=0)


@pytest.fixture(scope="session")
def df_RS(tf_data_dir: Path) -> pd.DataFrame:
    path = tf_data_dir / "TF_RS_delay_v1" / "testing_TF_data_RS.dat"
    if not path.exists():
        pytest.skip("RS transfer-function data file not found")
    return pd.read_csv(path, sep=r"\s+", header=0)


@pytest.fixture(scope="session")
def df_RS_na(tf_data_dir: Path) -> pd.DataFrame:
    path = tf_data_dir / "TF_RS_no_adapt_v1" / "testing_TF_data_RS.dat"
    if not path.exists():
        pytest.skip("RS_no_adapt transfer-function data file not found")
    return pd.read_csv(path, sep=r"\s+", header=0)


# ---- saved TF fitting results (validation JSON) ---------------------------

@pytest.fixture(scope="session")
def FS_MF_params(validation_dir: Path) -> list[dict]:
    path = validation_dir / "FS_MF_params.json"
    if not path.exists():
        pytest.skip("FS_MF_params.json not found")
    with open(path) as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def RS_MF_params(validation_dir: Path) -> list[dict]:
    path = validation_dir / "RS_MF_params.json"
    if not path.exists():
        pytest.skip("RS_MF_params.json not found")
    with open(path) as fh:
        return json.load(fh)


# ---- small synthetic data for optimisation tests ---------------------------

@pytest.fixture(scope="session")
def synthetic_RS_tf_data(params_RS_with_K: dict, RS_MF_params: list[dict]) -> pd.DataFrame:
    """Generate a small (7×7) synthetic TF grid from known RS params + noise."""
    best = RS_MF_params[0]
    alpha = best["alpha"]
    poly = np.array(best["polynomial_params"])

    from ntmf.transfer_function import TF_template

    exc_vals = np.arange(0, 31, 5, dtype=float)  # 0, 5, 10, ..., 30
    inh_vals = np.arange(0, 31, 5, dtype=float)

    rows = []
    for inh in inh_vals:
        for exc in exc_vals:
            df_row = pd.DataFrame({"input_exc": [exc], "input_inh": [inh],
                                    "avg_f_out": [0.0], "std_f_out": [0.0]})
            f_out = TF_template(data=df_row, params=params_RS_with_K,
                                poly_params=poly, alpha=alpha)
            # add 1% noise but keep non-negative
            f_noisy = max(0.0, float(f_out[0]) * (1.0 + np.random.default_rng(42).normal(0, 0.01)))
            rows.append({"input_exc": exc, "input_inh": inh,
                         "avg_f_out": f_noisy, "std_f_out": 1.0})

    return pd.DataFrame(rows)
