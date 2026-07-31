"""Interactive phase plane widget for neural mass models.

Two usage modes:
  1. Jupyter / VS Code  – anywidget wrapper; JS computes everything client-side.
  2. Standalone HTML    – export via ``to_standalone_html()`` for mkdocs,
     GitHub Pages, or any static site (no kernel, no Python runtime).
"""

import json
import os
import pathlib
import tempfile

import anywidget
import numpy as np
import traitlets

_STATIC_DIR = pathlib.Path(__file__).parent / "static"


class PhasePlaneWidget(anywidget.AnyWidget):
    """Interactive phase plane widget.

    In Jupyter / VS Code the widget is backed by anywidget.  All numerical
    work (ODE integration, fixed-point search, nullclines, sweeps) is done
    in the browser by the JS front-end, so interactivity is instantaneous.

    For static sites use :meth:`to_standalone_html` to obtain a self-contained
    ``.html`` file that can be dropped into mkdocs, GitHub Pages, etc.
    """

    # Inline JS/CSS so the widget is self-contained and standalone HTML exports work.
    _esm = (_STATIC_DIR / "widget.js").read_text()
    _css = (_STATIC_DIR / "widget.css").read_text()

    # ── Initial state (synced to JS front-end) ──
    model_name = traitlets.Unicode("wilson_cowan").tag(sync=True)
    # Human-readable label for the model selector.  Models supplied from
    # Python are absent from the JS registry, so without this the selector
    # would fall back to showing "Wilson-Cowan".
    model_label = traitlets.Unicode("").tag(sync=True)
    params = traitlets.Dict({}).tag(sync=True)
    param_info = traitlets.Dict({}).tag(sync=True)
    # Optional slider arrangement: a list of rows, each row a list of
    # parameter names.  Empty -> the front-end falls back to a flowing list.
    param_layout = traitlets.List([]).tag(sync=True)
    state_names = traitlets.List(["x", "y"]).tag(sync=True)
    state_labels = traitlets.List([]).tag(sync=True)
    state_units = traitlets.List([]).tag(sync=True)

    x0 = traitlets.Float(0.1).tag(sync=True)
    y0 = traitlets.Float(0.1).tag(sync=True)

    xlim = traitlets.List([-0.5, 1.5]).tag(sync=True)
    ylim = traitlets.List([-0.5, 1.5]).tag(sync=True)
    t_max = traitlets.Float(100.0).tag(sync=True)
    # Output step for the trajectory, in the model's time unit.  0.0 means
    # "auto": t_max / 2000, so the transient is resolved regardless of t_max.
    # It used to be hard-coded at 0.01 s, i.e. exactly one tau_f, which left
    # the relaxation of this model covered by about four points.
    dt = traitlets.Float(0.0).tag(sync=True)
    # Label only -- the unit the model's time variable is expressed in.  The
    # widget does not convert; it just annotates the time-series axis.
    time_unit = traitlets.Unicode("s").tag(sync=True)

    # Pre-computed data (populated by JS, kept for inspection / export)
    nullcline_x = traitlets.List([]).tag(sync=True)
    nullcline_y = traitlets.List([]).tag(sync=True)
    vector_field = traitlets.List([]).tag(sync=True)
    fixed_points = traitlets.List([]).tag(sync=True)
    trajectory = traitlets.List([]).tag(sync=True)

    sweep_results = traitlets.List([]).tag(sync=True)
    sweep_fixed_points = traitlets.List([]).tag(sync=True)
    sweep_running = traitlets.Bool(False).tag(sync=True)
    # The JS front-end reads and writes this; it was missing from the trait
    # list, so the sweep panel could never label its own x axis.
    sweep_param = traitlets.Unicode("").tag(sync=True)
    # Cost controls for the Python-side sweep.  Regime detection integrates
    # four trajectories per sweep point, which is expensive for models whose
    # RHS is not cheap; it is opt-in.
    sweep_options = traitlets.Dict(
        {"n_grid": 12, "detect_regime": False}
    ).tag(sync=True)

    # Display toggles
    show_nullclines = traitlets.Bool(True).tag(sync=True)
    show_vector_field = traitlets.Bool(True).tag(sync=True)
    show_trajectory = traitlets.Bool(True).tag(sync=True)
    show_fixed_points = traitlets.Bool(True).tag(sync=True)

    # Integrator / noise
    integrator = traitlets.Unicode("rk4").tag(sync=True)
    noise_enable = traitlets.Bool(False).tag(sync=True)
    noise_sigma = traitlets.List([]).tag(sync=True)

    # Custom model specification (JSON-serialisable dict for JS)
    model_spec = traitlets.Dict(allow_none=True, default_value=None).tag(sync=True)

    # Display indices for multi-variable projections
    display = traitlets.List([0, 1]).tag(sync=True)

    # Clamped values for non-displayed state variables
    clamped = traitlets.List(default_value=None, allow_none=True).tag(sync=True)

    # Layout mode: "full" shows controls + phase plane + time series + sweep
    # "phase_plane" shows only the phase plane canvas (useful in notebooks / iframes)
    display_mode = traitlets.Unicode("full").tag(sync=True)

    # When True, JS skips its own computation and renders data
    # that is supplied by the Python kernel (dumb-renderer mode).
    python_compute = traitlets.Bool(False).tag(sync=True)

    _DISPLAY_MODES = frozenset({"full", "phase_plane"})

    @traitlets.validate("display_mode")
    def _validate_display_mode(self, proposal):
        v = proposal["value"]
        if v not in self._DISPLAY_MODES:
            raise traitlets.TraitError(
                f"display_mode must be one of {self._DISPLAY_MODES}, got {v!r}"
            )
        return v

    _model_instance = None
    # Guards the model_name observer, which traitlets fires from inside
    # HasTraits.__init__ -- before __init__ has had a chance to apply the
    # caller's xlim/ylim.  Without it, constructor arguments are lost.
    _initialised = False

    def __init__(self, model=None, **kwargs):
        if model is not None:
            self._model_instance = model
            kwargs.setdefault("model_name", model.name)
            kwargs.setdefault(
                "model_label",
                getattr(model, "display_name", None) or type(model).__name__,
            )
        super().__init__(**kwargs)
        # Only fall back to the model's defaults for view state the caller did
        # NOT supply.  Previously _update_model() ran unconditionally and
        # silently overwrote xlim/ylim/params passed to the constructor.
        self._update_model(preserve=set(kwargs))
        self._initialised = True
        # Register handler for custom JS → Python messages (e.g. TikZ export)
        self.on_msg(self._on_custom_msg)
        if self.python_compute and self._model_instance is not None:
            self._run_python_compute()

    def _get_model(self):
        from .models import MODEL_REGISTRY

        if self._model_instance is not None:
            return self._model_instance
        cls = MODEL_REGISTRY.get(self.model_name, MODEL_REGISTRY["wilson_cowan"])
        return cls()

    def _update_model(self, preserve=frozenset()):
        """Push model metadata to the JS front-end.

        Parameters
        ----------
        preserve : set of str
            Trait names the caller set explicitly.  These are left untouched
            so constructor arguments are not clobbered by model defaults.
            Switching models later calls this with an empty set, which resets
            the view as intended.
        """
        model = self._get_model()
        self.param_info = model.param_info
        self.state_names = model.state_names
        self.state_labels = list(getattr(model, "state_labels", None) or [])
        self.state_units = list(getattr(model, "state_units", None) or [])
        self.param_layout = list(getattr(model, "param_layout", None) or [])
        if "params" not in preserve:
            self.params = {k: v[2] for k, v in model.param_info.items()}
        if "xlim" not in preserve:
            self.xlim = model.default_xlim
        if "ylim" not in preserve:
            self.ylim = model.default_ylim

    @traitlets.observe("model_name")
    def _on_model_change(self, change):
        # During construction __init__ pushes the metadata itself, preserving
        # whatever the caller passed.  Only a genuine later model switch
        # should reset the view limits.
        if not self._initialised:
            return
        if self.model_name != "custom":
            self._update_model()

    @traitlets.observe("params", "x0", "y0", "xlim", "ylim", "t_max", "dt", "display")
    def _on_state_change(self, change):
        """Trigger Python-side recomputation when interactive state changes."""
        if not self.python_compute or self._model_instance is None:
            return
        self._run_python_compute()

    def _run_python_compute(self):
        """Compute nullclines, VF, FPs, and trajectory using the Python model."""
        import numpy as np

        model = self._get_model()
        params = self.params
        xlim = list(self.xlim)
        ylim = list(self.ylim)

        # Nullclines & vector field
        nc_x, nc_y = model.compute_nullclines(params, xlim, ylim, n_grid=60)
        vf = model.compute_vector_field(params, xlim, ylim, n_grid=12)

        # Fixed points
        fps = model.find_fixed_points(params, xlim, ylim, n_grid=25)

        # Trajectory
        state0 = [0.0] * model.dim
        display = list(self.display) if self.display else [0, 1]
        if self.clamped:
            for i, v in enumerate(self.clamped):
                if v is not None and i < model.dim:
                    state0[i] = v
        state0[display[0]] = self.x0
        if len(display) > 1:
            state0[display[1]] = self.y0

        dt = float(self.dt) if self.dt > 0 else max(self.t_max / 2000.0, 1e-9)
        traj = model.compute_trajectory(state0, params, [0, self.t_max], dt=dt)

        # Write back to synced traits (triggers JS re-render)
        self.nullcline_x = nc_x
        self.nullcline_y = nc_y
        self.vector_field = vf
        self.fixed_points = fps

        if traj:
            step = max(1, len(traj) // 2000)
            self.trajectory = traj[::step]
        else:
            self.trajectory = []

    def set_model_spec(self, spec: dict):
        """Load a custom model from a ``ModelSpec`` dict.

        Parameters
        ----------
        spec : dict
            JSON-serialisable model specification (see
            :meth:`ModelSpec.to_widget_state`).
        """
        self.model_spec = spec
        self.model_name = "custom"
        # Derive initial params / limits from the spec so the widget
        # has sensible defaults before JS takes over.
        params = {n: v["default"] for n, v in spec.get("parameters", {}).items()}
        self.params = params
        # Sync param_info for the existing slider infrastructure
        param_info = {}
        for n, v in spec.get("parameters", {}).items():
            lo, hi = v["range"]
            step = v.get("step", (hi - lo) / 500)
            param_info[n] = [lo, hi, v["default"], f"Parameter {n}"]
        self.param_info = param_info
        state_names = list(spec.get("state_vars", {}).keys())
        self.state_names = state_names
        # Sync display indices
        display = spec.get("display", [0, min(1, len(state_names) - 1)])
        self.display = display
        # Set default display limits from state variable ranges
        state_vars = spec.get("state_vars", {})
        if state_names:
            first = state_names[0]
            lo, hi = state_vars[first]["range"]
            self.xlim = [lo, hi]
            self.x0 = state_vars[first]["default"]
        if len(state_names) > 1:
            second = state_names[display[1]] if len(display) > 1 else state_names[1]
            lo, hi = state_vars[second]["range"]
            self.ylim = [lo, hi]
            self.y0 = state_vars[second]["default"]
        # Initialize clamped values for non-displayed vars
        n = len(state_names)
        clamped = []
        for i, name in enumerate(state_names):
            if i in display:
                clamped.append(None)  # displayed vars are not clamped
            else:
                lo, hi = state_vars[name]["range"]
                clamped.append((lo + hi) / 2.0)
        self.clamped = clamped

    # ── Python-side helpers (for programmatic use / validation) ──

    def run_sweep(self, param_name: str, values, *, progress=None):
        """Sweep ``param_name`` over ``values`` using the Python model.

        For models defined in Python (``python_compute=True``) this is the
        only correct implementation: the JS front-end has no compiled RHS for
        them and its own sweep path resolves the model to ``undefined``.

        Fixed points found at one sweep step are reused as Newton seeds for
        the next, so branches are followed by continuation rather than
        rediscovered from scratch on every step.

        Parameters
        ----------
        param_name : str
            Parameter to vary.  Must appear in ``param_info``.
        values : sequence of float
            Values to evaluate, in order.
        progress : callable, optional
            Called as ``progress(i, n)`` after each sweep step.

        Returns
        -------
        (results, fixed_points)
            ``results`` is a list of dicts with ``param_value``, ``regime``
            and ``num_fixed_points``.  ``fixed_points`` is a flat list of
            ``[param_value, x, y, stability]``.
        """
        model = self._get_model()
        if param_name not in self.param_info:
            raise KeyError(
                f"{param_name!r} is not a sweep parameter; "
                f"expected one of {sorted(self.param_info)}"
            )

        opts = dict(self.sweep_options or {})
        n_grid = int(opts.get("n_grid", 12))
        want_regime = bool(opts.get("detect_regime", False))

        xlim = list(self.xlim)
        ylim = list(self.ylim)
        values = [float(v) for v in values]

        results = []
        all_fps = []
        seeds = None

        for i, val in enumerate(values):
            p = {**self.params, param_name: val}
            try:
                fps = model.find_fixed_points(
                    p, xlim, ylim, n_grid=n_grid, seeds=seeds
                )
            except TypeError:  # models.py predating the ``seeds`` argument
                fps = model.find_fixed_points(p, xlim, ylim, n_grid=n_grid)
            seeds = [[fp[0], fp[1]] for fp in fps] or None

            if want_regime:
                regime = model.detect_regime(p, xlim, ylim)
            else:
                regime = "other"

            results.append(
                {
                    "param_value": val,
                    "regime": regime,
                    "num_fixed_points": len(fps),
                }
            )
            for fp in fps:
                all_fps.append([val, float(fp[0]), float(fp[1]), fp[2]])

            if progress is not None:
                progress(i + 1, len(values))

        self.sweep_param = param_name
        self.sweep_results = results
        self.sweep_fixed_points = all_fps
        return results, all_fps

    # ── Standalone HTML export ──

    def to_standalone_html(
        self,
        filename: str | pathlib.Path,
        title: str = "Phase Plane Widget",
        *,
        on_render_js: str = "",
    ):
        """Export the widget to a self-contained HTML file.

        The resulting ``.html`` file contains the full JS computation engine,
        all model definitions, the CSS, and the current widget state.  It works
        in any modern browser with **no Python runtime and no Jupyter kernel**.

        Parameters
        ----------
        filename : str or pathlib.Path
            Output path (e.g. ``"widget.html"``).
        title : str
            Page ``<title>``.
        on_render_js : str
            Optional JavaScript snippet executed after the widget renders.
            Useful for auto-opening UI panels (e.g. the live editor).
        """
        js_code = self._esm
        css_code = self._css

        state = {
            "model_name": self.model_name,
            "model_label": self.model_label,
            "params": self.params,
            "param_layout": list(self.param_layout),
            "param_info": self.param_info,
            "state_names": self.state_names,
            "x0": self.x0,
            "y0": self.y0,
            "xlim": self.xlim,
            "ylim": self.ylim,
            "t_max": self.t_max,
            "dt": self.dt,
            "time_unit": self.time_unit,
            "state_labels": list(self.state_labels),
            "state_units": list(self.state_units),
            "show_nullclines": self.show_nullclines,
            "show_vector_field": self.show_vector_field,
            "show_trajectory": self.show_trajectory,
            "show_fixed_points": self.show_fixed_points,
            "nullcline_x": self.nullcline_x,
            "nullcline_y": self.nullcline_y,
            "vector_field": self.vector_field,
            "fixed_points": self.fixed_points,
            "trajectory": self.trajectory,
            "sweep_results": self.sweep_results,
            "sweep_fixed_points": self.sweep_fixed_points,
            "sweep_param": "",
            "sweep_running": False,
            "model_spec": self.model_spec,
            "display": list(self.display) if self.display else [0, 1],
            "clamped": list(self.clamped) if self.clamped else None,
            "integrator": self.integrator,
            "noise_enable": self.noise_enable,
            "noise_sigma": self.noise_sigma,
            "display_mode": self.display_mode,
            "python_compute": False,
        }

        extra_js = f"\n{on_render_js}\n" if on_render_js else ""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
{css_code}
    </style>
</head>
<body>
<div id="ppw-root"></div>
<script type="module">
{js_code}

const initialState = {json.dumps(state, indent=2)};

const mockModel = {{
  _isMock: true,
  _data: initialState,
  _callbacks: {{}},
  get(name) {{ return this._data[name]; }},
  set(name, value) {{ this._data[name] = value; }},
  save_changes() {{}},
  on(event, cb) {{
    if (!this._callbacks[event]) this._callbacks[event] = [];
    this._callbacks[event].push(cb);
  }},
  send() {{}},
}};

render({{ model: mockModel, el: document.getElementById('ppw-root') }});
{extra_js}
</script>
</body>
</html>"""

        pathlib.Path(filename).write_text(html, encoding="utf-8")

    # ── TikZ / PGFPlots export ──

    # ── NTMF colour scheme (mirrors NTMF_PALETTE in static/widget.js) ──
    # Warm hues encode inhibition, cool hues encode excitation.
    NTMF_COLORS = ["f46d43", "225ea5", "41b6c4", "8c6bb1"]   # FS, RS, RS_no_adapt, extra
    TRAJ_COLOR = "2f2f2f"
    IC_COLOR = "f46d43"

    # Stability is carried by marker SHAPE; colour stays achromatic so the
    # three population hues remain exclusive to populations.
    STABILITY_MARKERS = {
        "stable_node": ("circle", "ntmfFP", "ntmfFP"),
        "stable_focus": ("circle", "ntmfFP", "ntmfFP"),
        "unstable_node": ("circle", "ntmfFPopen", "white"),
        "unstable_focus": ("circle", "ntmfFPopen", "white"),
        "saddle": ("diamond", "ntmfSaddle", "ntmfSaddle"),
    }

    def _compute_vector_field_for_tikz(self, nx=15, ny=15):
        r"""Compute a normalized vector-field grid for embedding in TikZ.

        Returns a list of arrow endpoints ``[x1, y1, x2, y2]`` suitable for
        ``\draw[-stealth] (axis cs:x1,y1) -- (axis cs:x2,y2);``.
        """
        model = self._get_model()
        x = np.linspace(self.xlim[0], self.xlim[1], nx)
        y = np.linspace(self.ylim[0], self.ylim[1], ny)
        dx = self.xlim[1] - self.xlim[0]
        dy = self.ylim[1] - self.ylim[0]
        scale = 0.03 * min(dx, dy)
        arrows = []
        for xi in x:
            for yi in y:
                state = [0.0] * model.dim
                if self.display and len(self.display) >= 2:
                    state[self.display[0]] = xi
                    state[self.display[1]] = yi
                else:
                    state[0] = xi
                    if model.dim > 1:
                        state[1] = yi
                if self.clamped:
                    for i, val in enumerate(self.clamped):
                        if val is not None and i < model.dim:
                            state[i] = val
                d = model.f(0, state, self.params)
                norm = np.sqrt(d[0] ** 2 + d[1] ** 2)
                if norm > 1e-12:
                    x2 = xi + scale * d[0] / norm
                    y2 = yi + scale * d[1] / norm
                    arrows.append([float(xi), float(yi), float(x2), float(y2)])
        return arrows

    def _axis_title(self, index: int, fallback: str = "x") -> str:
        """LaTeX axis title for state variable ``index``, including units."""
        names = list(self.state_names)
        labels = list(self.state_labels)
        units = list(self.state_units)
        if index < len(labels) and labels[index]:
            body = self._tex_label_from_display(labels[index])
        elif index < len(names):
            body = self._tex_label(names[index])
        else:
            body = f"${fallback}$"
        unit = units[index] if index < len(units) else ""
        if not unit:
            return body
        # body is already wrapped in $...$; splice the unit inside so the
        # brackets and \mathrm stay in math mode.
        if body.startswith("$") and body.endswith("$"):
            return f"{body[:-1]}~[\\mathrm{{{unit}}}]$"
        return f"${body}~[\\mathrm{{{unit}}}]$"

    @staticmethod
    def _tex_label_from_display(label: str) -> str:
        """``\u03bd_FS`` -> ``$\\nu_{\\mathrm{FS}}$`` (already-typeset names)."""
        greek = {
            "\u03bd": "nu", "\u03bc": "mu", "\u03c3": "sigma", "\u03c4": "tau",
            "\u03b1": "alpha", "\u03b2": "beta", "\u03b3": "gamma",
            "\u03bb": "lambda", "\u03c9": "omega", "\u03c1": "rho",
        }
        head, _, sub_ = label.partition("_")
        base = f"\\{greek[head]}" if head in greek else f"\\mathrm{{{head}}}"
        if sub_:
            return f"${base}_{{\\mathrm{{{sub_}}}}}$"
        return f"${base}$"

    @staticmethod
    def _tex_label(name: str) -> str:
        """Turn a state-variable name into a LaTeX math label.

        ``nu_FS`` -> ``$\\nu_{\\mathrm{FS}}$``.  Without this, the raw name is
        dropped into math mode and ``nu_FS`` typesets as *nuF S*.
        """
        greek = {
            "nu", "mu", "sigma", "tau", "alpha", "beta", "gamma", "delta",
            "lambda", "phi", "psi", "theta", "omega", "rho", "eta", "kappa",
        }
        head, _, sub = name.partition("_")
        base = f"\\{head}" if head in greek else f"\\mathrm{{{head}}}"
        if sub:
            return f"${base}_{{\\mathrm{{{sub.replace('_', '')}}}}}$"
        return f"${base}$"

    @staticmethod
    def _fmt_coords(data):
        """Format a list of [x, y] pairs as PGFPlots ``coordinates`` block."""
        if not data:
            return "coordinates {(0,0)}"
        pairs = " ".join(f"({row[0]:.6f},{row[1]:.6f})" for row in data)
        return f"coordinates {{\n  {pairs}\n}}"

    def export_tikz(self, filename: str | pathlib.Path = "phase_plane.tex"):
        """Generate a self-contained ``.tex`` file.

        The resulting file can be compiled with ``pdflatex`` or ``lualatex``
        (requires the ``pgfplots`` package).

        Parameters
        ----------
        filename : str or pathlib.Path
            Output path for the ``.tex`` file.
        """
        filename = pathlib.Path(filename)

        ix = self.display[0] if self.display else 0
        iy = self.display[1] if self.display and len(self.display) > 1 else 1
        x_label = self._axis_title(ix, "x")
        y_label = self._axis_title(iy, "y")
        xlim = self.xlim
        ylim = self.ylim

        # ── vector field ──
        vfield_data = []
        if self.show_vector_field:
            vfield_data = self._compute_vector_field_for_tikz()

        # ── trajectories ──
        traj_data = []
        if self.show_trajectory and self.trajectory:
            idx_x = 1 + self.display[0]
            idx_y = 1 + self.display[1]
            traj_data = [
                [float(row[idx_x]), float(row[idx_y])] for row in self.trajectory
            ]

        # ── nullclines ──
        nc_x = self.nullcline_x if self.show_nullclines else []
        nc_y = self.nullcline_y if self.show_nullclines else []

        # ── fixed points ──
        fps = self.fixed_points if self.show_fixed_points else []

        # ── build plot commands ──
        plots = []

        if vfield_data:
            lines = "\n".join(
                f"            \\draw[-stealth, gray] (axis cs:{x1:.6f},{y1:.6f}) -- (axis cs:{x2:.6f},{y2:.6f});"
                for x1, y1, x2, y2 in vfield_data
            )
            plots.append(lines)

        # Nullclines take the colour of the population they belong to.
        idx_cx = self.display[0] if self.display else 0
        idx_cy = self.display[1] if self.display and len(self.display) > 1 else 1
        col_x = f"ntmfState{idx_cx % len(self.NTMF_COLORS)}"
        col_y = f"ntmfState{idx_cy % len(self.NTMF_COLORS)}"

        if nc_x:
            plots.append(
                f"            \\addplot[{col_x}, thick, no marks, smooth] {self._fmt_coords(nc_x)};"
            )

        if nc_y:
            plots.append(
                f"            \\addplot[{col_y}, thick, no marks, smooth] {self._fmt_coords(nc_y)};"
            )

        if traj_data:
            plots.append(
                f"            \\addplot[ntmfTraj, thick, no marks, smooth] {self._fmt_coords(traj_data)};"
            )

        plots_block = "\n".join(plots)

        # ── fixed-point nodes ──
        fp_nodes = []
        for fp in fps:
            if len(fp) < 3:
                continue
            x_fp, y_fp, stability = float(fp[0]), float(fp[1]), fp[2]
            shape, color, fill = self.STABILITY_MARKERS.get(
                stability, ("circle", "black", "white")
            )
            inner = ""
            if stability == "stable_focus":
                inner = (
                    f"\\node[fill={color}, circle, inner sep=0.8pt] "
                    f"at (axis cs:{x_fp:.6f},{y_fp:.6f}) {{}};"
                )
            elif stability == "unstable_focus":
                inner = (
                    f"\\node[draw={color}, fill={color}, circle, inner sep=0.8pt] "
                    f"at (axis cs:{x_fp:.6f},{y_fp:.6f}) {{}};"
                )
            opts = f"draw={color}"
            if fill == "white":
                opts += ", fill=white"
            elif fill != color:
                opts += f", fill={fill}"
            node_tex = (
                f"\\node[{opts}, {shape}, inner sep=1.5pt] "
                f"at (axis cs:{x_fp:.6f},{y_fp:.6f}) {{}};"
            )
            if inner:
                node_tex += "\n            " + inner
            fp_nodes.append(node_tex)
        fp_block = "\n            ".join(fp_nodes)

        # ── initial-condition marker (matches the on-screen legend) ──
        ic_node = ""
        if traj_data:
            ic_x, ic_y = traj_data[0]
            ic_node = (
                f"\\node[draw=black, fill=ntmfIC, circle, inner sep=1.8pt] "
                f"at (axis cs:{ic_x:.6f},{ic_y:.6f}) {{}};"
            )

        # ── parameter annotation ──
        # NB: this is plain concatenation, not an f-string, so braces must not
        # be doubled.  The previous version emitted "{{...}}};" (unbalanced).
        param_lines = ", ".join(
            (k.replace("_", r"\_") + f"={v:.4g}") for k, v in self.params.items()
        )
        label = (self.model_label or self.model_name).replace("_", r"\_")
        param_node = (
            r"\path (rel axis cs:0.02,0.98) node[anchor=north west, font=\tiny, align=left] "
            "{" + label + r"\\ " + param_lines + "};"
        )

        tex = (
            r"\documentclass[border=5pt]{standalone}" + "\n"
            r"\usepackage{pgfplots}" + "\n"
            r"\usepackage{tikz}" + "\n"
            r"\pgfplotsset{compat=1.17}" + "\n"
            r"\usetikzlibrary{shapes.geometric}" + "\n"
            + "".join(
                f"\\definecolor{{ntmfState{i}}}{{HTML}}{{{c.upper()}}}\n"
                for i, c in enumerate(self.NTMF_COLORS)
            )
            + f"\\definecolor{{ntmfTraj}}{{HTML}}{{{self.TRAJ_COLOR.upper()}}}\n"
            + f"\\definecolor{{ntmfIC}}{{HTML}}{{{self.IC_COLOR.upper()}}}\n"
            + "\\definecolor{ntmfFP}{HTML}{1A1A1A}\n"
            + "\\definecolor{ntmfFPopen}{HTML}{8A8A8A}\n"
            + "\\definecolor{ntmfSaddle}{HTML}{7B3294}\n"
            r"\begin{document}" + "\n"
            r"\begin{tikzpicture}" + "\n"
            r"\begin{axis}[" + "\n"
            "    width=10cm, height=10cm,\n"
            f"    xmin={xlim[0]}, xmax={xlim[1]}, ymin={ylim[0]}, ymax={ylim[1]},\n"
            f"    xlabel={x_label}, ylabel={y_label},\n"
            "    axis lines=box,\n"
            "    tick align=outside,\n"
            "    enlargelimits=true,\n"
            "]\n"
            + plots_block + "\n"
            + (fp_block + "\n" if fp_block else "")
            + ("            " + ic_node + "\n" if ic_node else "")
            + "            " + param_node + "\n"
            r"\end{axis}" + "\n"
            r"\end{tikzpicture}" + "\n"
            r"\end{document}" + "\n"
        )
        filename.write_text(tex, encoding="utf-8")
        return str(filename)

    # ── Custom message handler (JS → Python) ──

    def _handle_sweep_request(self, content):
        """Run a sweep requested by the JS front-end and report progress."""
        import numpy as np

        param = content.get("param")
        try:
            lo = float(content["min"])
            hi = float(content["max"])
            n = max(2, int(content["n"]))
        except (KeyError, TypeError, ValueError) as exc:
            self.send({"type": "sweep_error", "message": f"bad sweep range: {exc}"})
            return

        values = np.linspace(lo, hi, n).tolist()
        self.sweep_running = True

        def _progress(i, total):
            self.send({"type": "sweep_progress", "pct": round(100 * i / total)})

        try:
            self.run_sweep(param, values, progress=_progress)
        except Exception as exc:  # surface it in the widget, not just stderr
            self.send({"type": "sweep_error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            self.sweep_running = False
            self.send({"type": "sweep_done"})

    def _on_custom_msg(self, _widget, content, buffers):
        """Handle messages from the JS front-end."""
        msg_type = content.get("type")
        if msg_type == "run_sweep":
            self._handle_sweep_request(content)
            return
        if msg_type == "export_tikz":
            fd, tmppath = tempfile.mkstemp(suffix=".tex", prefix="phase_plane_")
            os.close(fd)
            try:
                self.export_tikz(tmppath)
                tex_content = pathlib.Path(tmppath).read_text(encoding="utf-8")
                self.send(
                    {
                        "type": "tikz_data",
                        "content": tex_content,
                        "filename": "phase_plane.tex",
                    }
                )
            finally:
                os.unlink(tmppath)
