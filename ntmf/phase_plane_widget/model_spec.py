"""Model specification and validation for custom phase-plane models.

A ModelSpec describes an N-dimensional ODE system, validates the
specification, and produces a JSON-serialisable dict that the JS widget
consumes.
"""
from __future__ import annotations

import json
import signal
from typing import Any, Callable

import sympy as sp
from sympy import Expr, Symbol, Matrix


class ModelSpec:
    """Validated model specification for the phase-plane widget.

    Parameters
    ----------
    equations : dict[str, Expr]
        Mapping from state-variable name → SymPy expression for its
        derivative.  All expressions must share the same symbols.
    state_vars : dict[str, tuple[float, float]]
        Mapping from state-variable name → (default_value, (min, max)).
        Order matters: the widget uses insertion order for indexing.
    parameters : dict[str, tuple[float, float, float]]
        Mapping from parameter name → (default_value, (min, max), step).
        Step is optional and defaults to ``(max-min)/500``.
    display : list[str] | None
        Names of state variables shown on the X / Y axes.  For a
        2-D projection this must contain exactly two distinct names
        that exist in ``state_vars``.  ``None`` defaults to the first
        two (or the first one for a 1-D system).
    custom_functions : dict[str, tuple[list[str], Expr]] | None
        Optional user-defined helper functions.  Each entry is
        ``name → (argument_names, body_expression)``.
    integrator : str
        ``"rk4"`` or ``"heun"``.  Defaults to ``"rk4"``.
    noise_per_var : list[float] | None
        Per-variable noise strengths for the stochastic Heun
        integrator.  ``None`` means deterministic.
    name : str
        Human-readable model name.

    Examples
    --------
    >>> from tvb_phaseplane.model_spec import ModelSpec
    >>> m = ModelSpec.from_strings(
    ...     equations=['a*x - x**3 - y', 'x - b*y'],
    ...     state_vars={'x': (-3, 3), 'y': (-3, 3)},
    ...     params={'a': (0.7, 0, 2), 'b': (0.8, 0, 2)},
    ... )
    """

    def __init__(
        self,
        *,
        equations: dict[str, Expr],
        state_vars: dict[str, tuple[float, tuple[float, float]]],
        parameters: dict[str, tuple[float, tuple[float, float], float]],
        display: list[str] | None = None,
        custom_functions: dict[str, tuple[list[str], Expr]] | None = None,
        integrator: str = "rk4",
        noise_per_var: list[float] | None = None,
        name: str = "Custom",
    ):
        self.name = name
        self.equations = dict(equations)
        self.state_vars = dict(state_vars)
        self.parameters = dict(parameters)
        self.custom_functions = dict(custom_functions or {})
        self.integrator = integrator
        self.noise_per_var = (
            list(noise_per_var) if noise_per_var is not None else None
        )

        # ── Validation ──
        state_names = list(self.state_vars.keys())
        param_names = list(self.parameters.keys())

        for eq_name in self.equations:
            if eq_name not in state_names:
                raise ValueError(
                    f"Equation key '{eq_name}' is not a declared state variable. "
                    f"Declared: {state_names}"
                )

        # Resolve display
        if display is None:
            if len(state_names) == 1:
                self.display = [state_names[0]]
            else:
                self.display = state_names[:2]
        else:
            self.display = list(display)

        if len(self.display) == 0:
            raise ValueError("display must contain at least one state variable")
        if len(self.display) > 2:
            raise ValueError(
                "Phase-plane projection supports at most 2 displayed variables; "
                f"got {len(self.display)}: {self.display}"
            )
        if len(set(self.display)) != len(self.display):
            raise ValueError(
                f"display contains duplicates: {self.display}"
            )
        for d in self.display:
            if d not in state_names:
                raise ValueError(
                    f"display variable '{d}' is not in state_vars"
                )

        if integrator not in ("rk4", "heun"):
            raise ValueError(f"integrator must be 'rk4' or 'heun', got {integrator!r}")

        if noise_per_var is not None:
            if len(noise_per_var) != len(state_names):
                raise ValueError(
                    f"noise_per_var length ({len(noise_per_var)}) must match "
                    f"number of state variables ({len(state_names)})"
                )

    # ------------------------------------------------------------------
    #  Factory methods
    # ------------------------------------------------------------------
    @classmethod
    def from_strings(
        cls,
        equations: list[str] | dict[str, str],
        state_vars: dict[str, tuple[float, float]],
        params: dict[str, tuple[float, float, float]] | dict[str, Any],
        *,
        display: list[str] | None = None,
        custom_functions: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> ModelSpec:
        """Build a ModelSpec from string equations.

        ``state_vars`` values are ``(min, max)`` pairs; defaults are
        taken as the midpoint.  ``params`` values can be either
        ``(default, min, max)`` triples or ``(default, min, max, step)``
        4-tuples.
        """
        # Normalise equations dict
        if isinstance(equations, list):
            eq_names = list(state_vars.keys())
            if len(equations) != len(eq_names):
                raise ValueError(
                    f"{len(equations)} equations but {len(eq_names)} state variables"
                )
            equations = dict(zip(eq_names, equations))

        # Build symbols
        all_names = list(state_vars.keys()) + list(params.keys())
        sym_map = {n: sp.Symbol(n) for n in all_names}

        # Parse equations
        eq_exprs: dict[str, Expr] = {}
        for name, expr_str in equations.items():
            try:
                eq_exprs[name] = sp.parse_expr(expr_str, local_dict=sym_map)
            except Exception as exc:
                raise ValueError(f"Failed to parse equation for '{name}': {expr_str}") from exc

        # Convert state_vars to (default, (min, max))
        state_vars_full: dict[str, tuple[float, tuple[float, float]]] = {}
        for name, lims in state_vars.items():
            lo, hi = float(lims[0]), float(lims[1])
            default = (lo + hi) / 2.0
            state_vars_full[name] = (default, (lo, hi))

        # Convert params to (default, (min, max), step)
        params_full: dict[str, tuple[float, tuple[float, float], float]] = {}
        for name, spec in params.items():
            vals = list(spec)
            if len(vals) < 3:
                raise ValueError(
                    f"Parameter '{name}' spec must be (default, min, max) or "
                    f"(default, min, max, step); got {spec!r}"
                )
            default, lo, hi = float(vals[0]), float(vals[1]), float(vals[2])
            step = float(vals[3]) if len(vals) >= 4 else (hi - lo) / 500.0
            params_full[name] = (default, (lo, hi), step)

        # Parse custom functions
        cf_parsed: dict[str, tuple[list[str], Expr]] | None = None
        if custom_functions:
            cf_parsed = {}
            for fname, fbody in custom_functions.items():
                # fbody can be a string expression; we need arg names.
                # For now, assume no nested custom functions in the
                # string-based API.
                cf_parsed[fname] = ([], sp.parse_expr(fbody, local_dict=sym_map))

        return cls(
            equations=eq_exprs,
            state_vars=state_vars_full,
            parameters=params_full,
            display=display,
            custom_functions=cf_parsed,
            **kwargs,
        )

    @classmethod
    def from_sympy(
        cls,
        equations: list[Expr] | dict[str, Expr],
        variables: list[Symbol],
        parameters: dict[Symbol, Any],
        *,
        display: list[Symbol] | None = None,
        **kwargs: Any,
    ) -> ModelSpec:
        """Build a ModelSpec from pre-constructed SymPy objects."""
        var_names = [str(v) for v in variables]

        if isinstance(equations, list):
            if len(equations) != len(var_names):
                raise ValueError(
                    f"{len(equations)} equations but {len(var_names)} variables"
                )
            equations = dict(zip(var_names, equations))

        # Convert parameters
        params_full: dict[str, tuple[float, tuple[float, float], float]] = {}
        for sym, spec in parameters.items():
            name = str(sym)
            if isinstance(spec, tuple):
                default, lo, hi = float(spec[0]), float(spec[1]), float(spec[2])
                step = float(spec[3]) if len(spec) >= 4 else (hi - lo) / 500.0
                params_full[name] = (default, (lo, hi), step)
            else:
                params_full[name] = (float(spec), (-10.0, 10.0), 0.1)

        # Default state_vars from variable names with dummy ranges
        # (caller should provide ranges via kwargs if needed)
        state_vars_full: dict[str, tuple[float, tuple[float, float]]] = {
            name: (0.0, (-5.0, 5.0)) for name in var_names
        }

        display_names = None
        if display is not None:
            display_names = [str(d) for d in display]

        return cls(
            equations=equations,
            state_vars=state_vars_full,
            parameters=params_full,
            display=display_names,
            **kwargs,
        )

    # ------------------------------------------------------------------
    #  Python-side utilities
    # ------------------------------------------------------------------
    def rhs_function(self, param_values: dict[str, float]) -> Callable:
        """Return a Python callable ``f(t, state) -> list[float]``.

        Suitable for use with ``scipy.integrate.solve_ivp``.
        """
        all_syms = list(self.state_vars.keys()) + list(self.parameters.keys())
        sym_map = {n: sp.Symbol(n) for n in all_syms}
        eq_list = [self.equations[n] for n in self.state_vars]
        lambdas = [sp.lambdify(all_syms, eq, "numpy") for eq in eq_list]

        def f(t: float, state: list[float]) -> list[float]:
            args = list(state) + [param_values.get(p, 0.0) for p in self.parameters]
            return [float(fn(*args)) for fn in lambdas]

        return f

    def jacobian_symbolic(self) -> Matrix:
        """Return the symbolic Jacobian matrix ``J[i,j] = d(eq_i)/d(var_j)``."""
        vars = [sp.Symbol(n) for n in self.state_vars]
        eqs = [self.equations[n] for n in self.state_vars]
        return Matrix(eqs).jacobian(vars)

    def fixed_points_symbolic(
        self,
        param_values: dict[str, float],
        *,
        timeout: float = 5.0,
    ) -> list[dict[str, float]] | None:
        """Attempt to solve for fixed points analytically.

        Returns ``None`` on timeout or if ``sympy.solve`` fails.
        """
        # Build equations: eq_i = 0
        vars = [sp.Symbol(n) for n in self.state_vars]
        subs = {sp.Symbol(p): float(v) for p, v in param_values.items()}
        eqs = [self.equations[n].subs(subs) for n in self.state_vars]

        def _solve():
            try:
                sols = sp.solve(eqs, vars, dict=True)
                if not sols:
                    return []
                result = []
                for sol in sols:
                    result.append({str(k): float(v.evalf()) for k, v in sol.items()})
                return result
            except Exception:
                return None

        # Use alarm for timeout (Unix only)
        old_handler = None
        try:
            old_handler = signal.signal(signal.SIGALRM, lambda _sig, _frm: (_ for _ in ()).throw(TimeoutError()))
            signal.setitimer(signal.ITIMER_REAL, timeout)
            result = _solve()
            signal.setitimer(signal.ITIMER_REAL, 0)
        except TimeoutError:
            result = None
        finally:
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)
                signal.setitimer(signal.ITIMER_REAL, 0)

        return result

    # ------------------------------------------------------------------
    #  JS widget state
    # ------------------------------------------------------------------
    def to_widget_state(self) -> dict[str, Any]:
        """JSON-serialisable dict consumed by the JS widget."""
        state_names = list(self.state_vars.keys())

        # Display as indices
        display_indices = [state_names.index(d) for d in self.display]

        # Equations as strings (Nerdamer parses these)
        eq_strs = {}
        for name, expr in self.equations.items():
            eq_strs[name] = str(expr)

        # Custom functions as strings
        cf_strs = {}
        for fname, (args, body) in self.custom_functions.items():
            cf_strs[fname] = {
                "vars": args,
                "expr": str(body),
            }

        return {
            "name": self.name,
            "state_vars": {
                n: {"default": d, "range": [lo, hi]}
                for n, (d, (lo, hi)) in self.state_vars.items()
            },
            "parameters": {
                n: {"default": d, "range": [lo, hi], "step": step}
                for n, (d, (lo, hi), step) in self.parameters.items()
            },
            "equations": eq_strs,
            "display": display_indices,
            "custom_functions": cf_strs,
            "integrator": self.integrator,
            "noise_per_var": self.noise_per_var,
        }

    def to_json(self) -> str:
        """Pretty-printed JSON for debugging / copy-paste."""
        return json.dumps(self.to_widget_state(), indent=2)
