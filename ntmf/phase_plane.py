"""Phase-plane analysis adapter for the 2-population mean-field model."""

import numpy as np
from ntmf.phase_plane_widget.models import BaseModel
from ntmf.meanfield import MFModel
from ntmf.transfer_function import TF_template_sim


class NTMFMeanField(BaseModel):
    """BaseModel adapter for the 2-population (FS, RS) mean-field model.

    Parameters exposed to widget sliders:
      - External input rates (Hz) to FS and RS populations
      - Population time constant tau_f
      - TF scaling factors alpha_FS and alpha_RS
    """

    name = "ntmf_meanfield"
    display_name = "NTMFMeanField"
    dim = 2
    state_names = ["nu_FS", "nu_RS"]
    state_labels = ["\u03bd_FS", "\u03bd_RS"]
    state_units = ["Hz", "Hz"]
    default_xlim = [0.0, 100.0]
    default_ylim = [0.0, 100.0]

    def __init__(
        self,
        params: dict[str, dict[str, float]],
        poly_params: dict[str, list | np.ndarray],
        alphas: dict[str, float],
        network_config: dict,
        tau_f: float = 0.01,
    ):
        self.mf = MFModel(
            params=params,
            poly_params=poly_params,
            alphas=alphas,
            network_config=network_config,
            tau_f=tau_f,
        )
        self._last_key = None

        # Initial external-input rates come from the network config's ``rates``
        # block, so the sliders start at the same operating point used to build
        # the network (freq_exc_FS = 5 Hz, etc.) rather than at zero.  Each maps
        # to one slider; a missing key falls back to 0.0.
        rates = network_config.get("rates", {})

        def _rate(key):
            return float(rates.get(key, 0.0))

        # Parameters exposed to widget sliders.  Insertion order is the
        # fallback order used when ``param_layout`` is not honoured.
        self.param_info = {
            "nu_ext_exc_FS": (0.0, 200.0, _rate("freq_exc_FS"), "Ext. exc. → FS (Hz)"),
            "nu_ext_exc_RS": (0.0, 200.0, _rate("freq_exc_RS"), "Ext. exc. → RS (Hz)"),
            "nu_ext_inh_FS": (0.0, 200.0, _rate("freq_inh_FS"), "Ext. inh. → FS (Hz)"),
            "nu_ext_inh_RS": (0.0, 200.0, _rate("freq_inh_RS"), "Ext. inh. → RS (Hz)"),
            "alpha_FS": (0.5, 2.0, alphas["FS"], "TF scale FS"),
            "alpha_RS": (0.5, 2.0, alphas["RS"], "TF scale RS"),
            "tau_f": (0.001, 0.1, tau_f, "τ_f (s)"),
        }

        # Slider grid: column 1 is FS, column 2 is RS, tau_f is shared by both
        # populations and sits on its own row.
        self.param_layout = [
            ["nu_ext_exc_FS", "nu_ext_exc_RS"],
            ["nu_ext_inh_FS", "nu_ext_inh_RS"],
            ["alpha_FS", "alpha_RS"],
            ["tau_f"],
        ]
        self.default_params = {k: v[2] for k, v in self.param_info.items()}

    def f(self, t, state, params):
        """Evaluate RHS, updating MFModel only when slider params change."""
        p = {**self.default_params, **params}

        key = (
            float(p["nu_ext_exc_FS"]),
            float(p["nu_ext_exc_RS"]),
            float(p["nu_ext_inh_FS"]),
            float(p["nu_ext_inh_RS"]),
            float(p["tau_f"]),
            float(p["alpha_FS"]),
            float(p["alpha_RS"]),
        )
        if key != self._last_key:
            self._last_key = key
            # Use a long time span so interp1d covers any integration window.
            # t_max beyond 1000 s will hit fill_value=0.0 in interp1d and produce
            # incorrect zero-input trajectories.  This is unlikely for the default
            # widget setting (t_max = 100 s) but worth noting.
            time = np.array([0.0, 1000.0])
            driving_input = {
                "excitatory": {
                    "FS": np.full_like(time, p["nu_ext_exc_FS"]),
                    "RS": np.full_like(time, p["nu_ext_exc_RS"]),
                },
                "inhibitory": {
                    "FS": np.full_like(time, p["nu_ext_inh_FS"]),
                    "RS": np.full_like(time, p["nu_ext_inh_RS"]),
                },
            }
            self.mf.set_driving_input(time, driving_input)
            self.mf.tau_f = float(p["tau_f"])
            self.mf.alphas["FS"] = float(p["alpha_FS"])
            self.mf.alphas["RS"] = float(p["alpha_RS"])

        dydt = self.mf.rhs(float(t), np.asarray(state, dtype=float))
        return [float(dydt[0]), float(dydt[1])]
