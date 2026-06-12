"""Interactive phase plane widget for neural mass models.

Vendored from ``tvb-phaseplane`` (github.com/maedoc/trajecturtle) with
``python_compute`` patches so the widget can render data computed in
Python (dumb-renderer mode).
"""

from .models import BaseModel, FitzHughNagumo, MPRModel, WilsonCowan, MODEL_REGISTRY
from .model_spec import ModelSpec

__all__ = [
    "BaseModel",
    "WilsonCowan",
    "FitzHughNagumo",
    "MPRModel",
    "MODEL_REGISTRY",
    "ModelSpec",
]

# Lazy import of widget-dependent symbols so the rest of the package
# can be used even when anywidget / ipywidgets are not installed.
try:
    from .widget import PhasePlaneWidget
    __all__.append("PhasePlaneWidget")
except ImportError:  # pragma: no cover
    PhasePlaneWidget = None  # type: ignore

__version__ = "0.1.0"


def phase_plane(
    equations,
    state_vars=None,
    params=None,
    display=None,
    custom_functions=None,
    integrator="rk4",
    noise_per_var=None,
    **kwargs,
):
    """Create a PhasePlaneWidget from user-supplied ODE equations.

    Requires ``ntmf[widget]`` dependencies (anywidget, traitlets, sympy).
    """
    if PhasePlaneWidget is None:
        raise ImportError(
            "PhasePlaneWidget is not available. "
            "Install widget dependencies: pip install ntmf[widget]"
        )
    spec = ModelSpec.from_strings(
        equations=equations,
        state_vars=state_vars or {},
        params=params or {},
        display=display,
        custom_functions=custom_functions,
        integrator=integrator,
        noise_per_var=noise_per_var,
        **kwargs,
    )
    widget = PhasePlaneWidget()
    widget.set_model_spec(spec.to_widget_state())
    return widget
