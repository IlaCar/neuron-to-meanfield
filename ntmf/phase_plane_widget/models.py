"""Neural mass model definitions for phase plane analysis."""

from collections import Counter

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve


class BaseModel:
    """Base class for neural mass models."""

    name = "base"
    dim = 2
    state_names = ["x", "y"]
    default_params = {}
    param_info = {}
    default_xlim = [-1.0, 1.0]
    default_ylim = [-1.0, 1.0]

    @property
    def state_vars(self):
        """Ordered list of all state variable names (alias for ``state_names``)."""
        return list(self.state_names)

    def to_model_spec(self):
        """Return a dict compatible with :class:`ModelSpec` and the JS widget.

        Built-in models use this to populate the ``model_spec`` traitlet when
        the user switches to a custom-model view.
        """
        from .model_spec import ModelSpec

        # Build state_vars dict: name -> (default, (min, max))
        sv = {}
        for i, name in enumerate(self.state_names):
            # Use default xlim/ylim for ranges
            if i == 0:
                lo, hi = self.default_xlim
            elif i == 1:
                lo, hi = self.default_ylim
            else:
                lo, hi = -5.0, 5.0
            sv[name] = (0.0, (lo, hi))

        # Build parameters dict: name -> (default, (min, max), step)
        params = {}
        for name, spec in self.param_info.items():
            lo, hi, default, _desc = spec
            step = (hi - lo) / 500.0
            params[name] = (default, (lo, hi), step)

        # Equations as strings — for built-in models the JS already has the
        # compiled functions, so we return empty equations and rely on the
        # ``model_name`` traitlet to select the hard-coded JS model.
        return {
            "name": self.name,
            "state_vars": {
                n: {"default": d, "range": [lo, hi]}
                for n, (d, (lo, hi)) in sv.items()
            },
            "parameters": {
                n: {"default": d, "range": [lo, hi], "step": step}
                for n, (d, (lo, hi), step) in params.items()
            },
            "equations": {},
            "display": list(range(min(2, len(self.state_names)))),
            "custom_functions": {},
            "integrator": "rk4",
            "noise_per_var": None,
        }

    def f(self, t, state, params):
        """Compute derivatives. Returns list of length dim."""
        raise NotImplementedError

    def compute_vector_field(self, params, xlim, ylim, n_grid=12):
        """Compute a sparse grid of derivative vectors."""
        x = np.linspace(xlim[0], xlim[1], n_grid)
        y = np.linspace(ylim[0], ylim[1], n_grid)
        vectors = []
        for xi in x:
            for yi in y:
                d = self.f(0, [xi, yi], params)
                vectors.append([float(xi), float(yi), float(d[0]), float(d[1])])
        return vectors

    def compute_nullclines(self, params, xlim, ylim, n_grid=60):
        """Compute nullclines by finding zero crossings on a grid."""
        x = np.linspace(xlim[0], xlim[1], n_grid)
        y = np.linspace(ylim[0], ylim[1], n_grid)
        X, Y = np.meshgrid(x, y)

        dx = np.zeros_like(X)
        dy = np.zeros_like(Y)
        for i in range(n_grid):
            for j in range(n_grid):
                d = self.f(0, [X[i, j], Y[i, j]], params)
                dx[i, j] = d[0]
                dy[i, j] = d[1]

        nc_x = self._find_zero_crossings(X, Y, dx)
        nc_y = self._find_zero_crossings(X, Y, dy)
        return nc_x, nc_y

    def _find_zero_crossings(self, X, Y, Z):
        """Find points where Z=0 using linear interpolation."""
        points = []
        n = X.shape[0]

        for i in range(n):
            for j in range(n - 1):
                if Z[i, j] == 0:
                    points.append([float(X[i, j]), float(Y[i, j])])
                elif Z[i, j] * Z[i, j + 1] < 0:
                    t = abs(Z[i, j]) / (abs(Z[i, j]) + abs(Z[i, j + 1]))
                    px = X[i, j] + t * (X[i, j + 1] - X[i, j])
                    py = Y[i, j]
                    points.append([float(px), float(py)])

        for i in range(n - 1):
            for j in range(n):
                if Z[i, j] == 0:
                    points.append([float(X[i, j]), float(Y[i, j])])
                elif Z[i, j] * Z[i + 1, j] < 0:
                    t = abs(Z[i, j]) / (abs(Z[i, j]) + abs(Z[i + 1, j]))
                    px = X[i, j]
                    py = Y[i, j] + t * (Y[i + 1, j] - Y[i, j])
                    points.append([float(px), float(py)])

        return points

    def find_fixed_points(self, params, xlim, ylim, n_grid=25):
        """Find fixed points by grid search + numerical refinement."""
        x = np.linspace(xlim[0], xlim[1], n_grid)
        y = np.linspace(ylim[0], ylim[1], n_grid)
        fixed_points = []
        tol = 0.08

        for xi in x:
            for yi in y:
                try:
                    sol = fsolve(lambda s: self.f(0, s, params), [xi, yi], full_output=True)
                    if sol[2] == 1:
                        fp = sol[0]
                        if (xlim[0] - 0.5 <= fp[0] <= xlim[1] + 0.5 and
                                ylim[0] - 0.5 <= fp[1] <= ylim[1] + 0.5):
                            # Verify it's actually a fixed point
                            residual = np.linalg.norm(self.f(0, fp, params))
                            if residual > 0.1:
                                continue
                            # Check for duplicates
                            is_new = True
                            for existing in fixed_points:
                                if np.linalg.norm(np.array(fp) - np.array(existing[:2])) < tol:
                                    is_new = False
                                    break
                            if is_new:
                                J = self.jacobian(fp, params)
                                ev = np.linalg.eigvals(J)
                                stability = self._classify_fixed_point(ev)
                                fixed_points.append([float(fp[0]), float(fp[1]), stability])
                except Exception:
                    pass

        return fixed_points

    def jacobian(self, state, params, eps=1e-6):
        """Numerical Jacobian."""
        n = len(state)
        J = np.zeros((n, n))
        f0 = np.array(self.f(0, state, params))
        for i in range(n):
            s_plus = np.array(state, dtype=float)
            s_plus[i] += eps
            f_plus = np.array(self.f(0, s_plus.tolist(), params))
            J[:, i] = (f_plus - f0) / eps
        return J

    def _classify_fixed_point(self, eigenvalues):
        """Classify fixed point based on eigenvalues."""
        real = np.real(eigenvalues)
        imag = np.imag(eigenvalues)

        if all(r < -1e-6 for r in real):
            return "stable_focus" if any(abs(im) > 1e-6 for im in imag) else "stable_node"
        elif all(r > 1e-6 for r in real):
            return "unstable_focus" if any(abs(im) > 1e-6 for im in imag) else "unstable_node"
        return "saddle"

    def compute_trajectory(self, initial_state, params, t_span, dt=0.01):
        """Compute trajectory using solve_ivp."""
        try:
            t_eval = np.arange(t_span[0], t_span[1], dt)
            sol = solve_ivp(
                lambda t, y: self.f(t, y, params),
                [t_span[0], t_span[1]],
                initial_state,
                method="RK45",
                t_eval=t_eval,
                max_step=dt * 5,
            )
            trajectory = []
            for i in range(len(sol.t)):
                row = [float(sol.t[i])]
                for j in range(self.dim):
                    row.append(float(sol.y[j, i]))
                trajectory.append(row)
            return trajectory
        except Exception:
            return []

    def detect_regime(self, params, xlim, ylim, t_total=120.0, dt=0.05):
        """Detect dynamical regime: fixed_point, limit_cycle, or other."""
        ics = [
            [xlim[0] * 0.6, ylim[0] * 0.6],
            [xlim[1] * 0.6, ylim[1] * 0.6],
            [(xlim[0] + xlim[1]) * 0.5, (ylim[0] + ylim[1]) * 0.5],
            [xlim[0] + 0.1 * (xlim[1] - xlim[0]), ylim[0] + 0.1 * (ylim[1] - ylim[0])],
        ]

        regimes = []
        for ic in ics:
            try:
                traj = self.compute_trajectory(ic, params, [0, t_total], dt=dt)
                if not traj:
                    regimes.append("other")
                    continue

                n = len(traj)
                n_check = min(200, n // 4)
                if n_check < 10:
                    regimes.append("other")
                    continue

                last = np.array(traj[-n_check:])
                dx = np.std(last[:, 1])
                dy = np.std(last[:, 2])

                if dx < 0.025 and dy < 0.025:
                    regimes.append("fixed_point")
                else:
                    # Check amplitude stability for limit cycle
                    mid = len(last) // 2
                    x_vals = last[:, 1]
                    y_vals = last[:, 2]
                    amp1_x = np.max(x_vals[:mid]) - np.min(x_vals[:mid])
                    amp2_x = np.max(x_vals[mid:]) - np.min(x_vals[mid:])
                    amp1_y = np.max(y_vals[:mid]) - np.min(y_vals[:mid])
                    amp2_y = np.max(y_vals[mid:]) - np.min(y_vals[mid:])

                    x_stable = abs(amp1_x - amp2_x) < 0.15 * max(amp1_x, 0.01)
                    y_stable = abs(amp1_y - amp2_y) < 0.15 * max(amp1_y, 0.01)

                    if x_stable and y_stable:
                        regimes.append("limit_cycle")
                    else:
                        regimes.append("other")
            except Exception:
                regimes.append("other")

        return Counter(regimes).most_common(1)[0][0]


class WilsonCowan(BaseModel):
    """Wilson-Cowan model of excitatory and inhibitory neural populations."""

    name = "wilson_cowan"
    dim = 2
    state_names = ["E", "I"]
    default_params = {
        "aee": 10.0,
        "aei": 10.0,
        "aie": 10.0,
        "aii": 2.0,
        "Pe": -2.0,
        "Pi": -8.0,
        "ke": 1.0,
        "ki": 1.0,
        "thetae": 4.0,
        "thetai": 4.0,
    }
    param_info = {
        "aee": (0.0, 20.0, 10.0, "E→E coupling"),
        "aei": (0.0, 20.0, 10.0, "I→E coupling"),
        "aie": (0.0, 20.0, 10.0, "E→I coupling"),
        "aii": (0.0, 20.0, 2.0, "I→I coupling"),
        "Pe": (-10.0, 10.0, -2.0, "External E input"),
        "Pi": (-10.0, 10.0, -8.0, "External I input"),
        "ke": (0.1, 5.0, 1.0, "E sigmoid gain"),
        "ki": (0.1, 5.0, 1.0, "I sigmoid gain"),
        "thetae": (0.0, 10.0, 4.0, "E sigmoid threshold"),
        "thetai": (0.0, 10.0, 4.0, "I sigmoid threshold"),
    }
    default_xlim = [-0.2, 1.2]
    default_ylim = [-0.2, 1.2]

    def f(self, t, state, params):
        E, I = state
        p = {**self.default_params, **params}

        def _sigmoid(x, k, theta):
            arg = -k * (x - theta)
            # Clip to prevent overflow in exp (ln(max_float64) ~ 709)
            arg = np.clip(arg, -709, 709)
            return 1.0 / (1.0 + np.exp(arg))

        return [
            -E + _sigmoid(p["aee"] * E - p["aei"] * I + p["Pe"], p["ke"], p["thetae"]),
            -I + _sigmoid(p["aie"] * E - p["aii"] * I + p["Pi"], p["ki"], p["thetai"]),
        ]


class HindmarshRose(BaseModel):
    """Hindmarsh–Rose model of neuronal bursting.

    A three-variable phenomenological model producing spiking and bursting
    oscillations.  The fast subsystem (x, y) drives action potentials, while
    the slow adaptation current z modulates the burst period.

    Equations
    ---------
    dx/dt = y – a·x³ + b·x² – z + I
    dy/dt = c – d·x² – y
    dz/dt = r·(s·(x – x_r) – z)

    References
    ----------
    Hindmarsh J. L.; Rose R. M. (1984).
    "A model of neuronal bursting using three coupled first order
    differential equations". Proc. R. Soc. Lond. B 221, 87–102.
    """

    name = "hindmarsh_rose"
    dim = 3
    state_names = ["x", "y", "z"]
    default_params = {
        "a": 1.0,
        "b": 3.0,
        "c": 1.0,
        "d": 5.0,
        "I": 3.0,
        "r": 0.001,
        "s": 4.0,
        "x_r": -1.6,
    }
    param_info = {
        "a": (0.5, 2.0, 1.0, "Cubic coefficient a"),
        "b": (1.0, 5.0, 3.0, "Quadratic coefficient b"),
        "c": (0.0, 2.0, 1.0, "Constant drive c"),
        "d": (1.0, 10.0, 5.0, "Quadratic loss d"),
        "I": (-2.0, 6.0, 3.0, "Applied current I"),
        "r": (0.0001, 0.01, 0.001, "Slow time scale r"),
        "s": (1.0, 8.0, 4.0, "Slow gain s"),
        "x_r": (-3.0, 0.0, -1.6, "Resting offset x_r"),
    }
    default_xlim = [-3.0, 3.0]
    default_ylim = [-20.0, 10.0]

    def f(self, t, state, params):
        x, y, z = state
        p = {**self.default_params, **params}
        return [
            y - p["a"] * x ** 3 + p["b"] * x ** 2 - z + p["I"],
            p["c"] - p["d"] * x ** 2 - y,
            p["r"] * (p["s"] * (x - p["x_r"]) - z),
        ]


class FitzHughNagumo(BaseModel):
    """FitzHugh-Nagumo model of excitable neuron dynamics."""

    name = "fitzhugh_nagumo"
    dim = 2
    state_names = ["v", "w"]
    default_params = {"a": 0.7, "b": 0.8, "epsilon": 0.08, "I": 0.5}
    param_info = {
        "a": (-1.0, 2.0, 0.7, "Recovery offset"),
        "b": (0.0, 2.0, 0.8, "Recovery gain"),
        "epsilon": (0.001, 1.0, 0.08, "Time scale (ε)"),
        "I": (-2.0, 2.0, 0.5, "External current"),
    }
    default_xlim = [-3.0, 3.0]
    default_ylim = [-1.5, 2.0]

    def f(self, t, state, params):
        v, w = state
        p = {**self.default_params, **params}
        return [
            v - v ** 3 / 3.0 - w + p["I"],
            p["epsilon"] * (v + p["a"] - p["b"] * w),
        ]


class MPRModel(BaseModel):
    """Montbrió-Pazó-Roxin exact firing-rate equations for QIF neurons.

    The macroscopic variables are firing rate (r) and mean membrane potential (v).
    See: Phys. Rev. X 5, 021028 (2015).
    """

    name = "mpr"
    dim = 2
    state_names = ["r", "v"]
    default_params = {
        "delta": 1.0,
        "eta_bar": -5.0,
        "J": 15.0,
        "I": 0.0,
    }
    param_info = {
        "delta": (0.01, 5.0, 1.0, "Lorentzian half-width Δ"),
        "eta_bar": (-20.0, 10.0, -5.0, "Mean excitability η̄"),
        "J": (-20.0, 30.0, 15.0, "Synaptic coupling J"),
        "I": (-10.0, 10.0, 0.0, "External input I"),
    }
    default_xlim = [0.0, 2.0]
    default_ylim = [-4.0, 2.0]

    def f(self, t, state, params):
        r, v = state
        p = {**self.default_params, **params}
        # Clip r to avoid numerical issues at r ≈ 0
        r_eff = max(r, 1e-10)
        dr = p["delta"] / np.pi + 2 * r_eff * v
        dv = v**2 + p["eta_bar"] + p["J"] * r_eff + p["I"] - (np.pi * r_eff) ** 2
        return [dr, dv]


MODEL_REGISTRY = {
    "wilson_cowan": WilsonCowan,
    "fitzhugh_nagumo": FitzHughNagumo,
    "mpr": MPRModel,
    "hindmarsh_rose": HindmarshRose,
}
