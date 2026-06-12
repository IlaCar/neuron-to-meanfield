"""SymPy-to-JavaScript expression transpiler.

Provides symbolic-to-JS translation for use in the phase plane widget's
custom model feature.  Full FUNCTION_MAP coverage is required; any SymPy
function not in the map raises a clear error.
"""

import sympy as sp
from sympy import Number, Float, Integer, Symbol, Add, Mul, Pow
from sympy.core.numbers import Pi as SymPyPi


FUNCTION_MAP = {
    'Add': lambda args: ' + '.join(args),
    'Mul': lambda args: ' * '.join(args),
    'Pow': lambda args: f'Math.pow({args[0]}, {args[1]})',
    'exp': 'Math.exp',
    'log': 'Math.log',
    'sin': 'Math.sin',
    'cos': 'Math.cos',
    'tan': 'Math.tan',
    'sqrt': 'Math.sqrt',
    'Abs': 'Math.abs',
    'sign': 'Math.sign',
    'tanh': 'Math.tanh',
    'Min': 'Math.min',
    'Max': 'Math.max',
    'pi': 'Math.PI',
    'Heaviside': '(x => x > 0 ? 1 : (x < 0 ? 0 : 0.5))',
}


def sympy_to_js(expr, symbol_map):
    """Recursively convert a SymPy expression to a JavaScript expression string.

    Parameters
    ----------
    expr : sympy.Expr
        The SymPy expression to transpile.
    symbol_map : dict[str, str]
        Mapping from SymPy symbol names to JavaScript variable names.

    Returns
    -------
    str
        Valid JavaScript expression string.
    """
    # ── Atoms ──
    if isinstance(expr, (Number, Float, Integer)):
        return _format_number(float(expr))

    if isinstance(expr, Symbol):
        try:
            return symbol_map[str(expr)]
        except KeyError:
            raise ValueError(f"Unknown symbol '{expr}' — not in symbol_map {set(symbol_map.keys())}")

    # Special constants
    if isinstance(expr, SymPyPi):
        return 'Math.PI'
    if isinstance(expr, Pow):
        import sys as _sys
        print(f'DEBUG Pow: {expr}, exp_val={expr.args[1]}, type={type(expr.args[1])}', file=_sys.stderr)
        base = sympy_to_js(expr.args[0], symbol_map)
        exp_val = expr.args[1]
        # x^0.5 → Math.sqrt(x)
        if isinstance(exp_val, (Number, Float, Integer)) and abs(float(exp_val) - 0.5) < 1e-12:
            return f'Math.sqrt({base})'
        # x^-1 → 1/x
        if isinstance(exp_val, (Number, Float, Integer)) and float(exp_val) == -1.0:
            return f'1/{base}'
        exponent = sympy_to_js(exp_val, symbol_map)
        return f'Math.pow({base}, {exponent})'

    # ── Mul ──
    if isinstance(expr, Mul):
        return _handle_mul(expr, symbol_map)

    # ── Add ──
    if isinstance(expr, Add):
        return _handle_add(expr, symbol_map)

    # ── Function applications ──
    func_name = expr.func.__name__
    if func_name in FUNCTION_MAP:
        entry = FUNCTION_MAP[func_name]
        if func_name == 'Heaviside':
            arg = sympy_to_js(expr.args[0], symbol_map)
            return f'({arg} > 0 ? 1 : ({arg} < 0 ? 0 : 0.5))'
        if callable(entry):
            # e.g. Add, Mul, Pow — not reached here (handled above)
            args = [sympy_to_js(a, symbol_map) for a in expr.args]
            return entry(args)
        # Plain function name: entry = 'Math.exp' etc.
        args = ', '.join(sympy_to_js(a, symbol_map) for a in expr.args)
        return f'{entry}({args})'

    raise ValueError(
        f"Unsupported SymPy function '{func_name}'. "
        f"Supported: {', '.join(sorted(FUNCTION_MAP.keys()))}. "
        "Consider defining a custom function in the model spec."
    )


def transpile_custom_function(name, vars, expr, symbol_map):
    """Convert a custom function definition to a JavaScript function string.

    Parameters
    ----------
    name : str
        Function name (e.g. 'sigmoid').
    vars : list of str
        Parameter names of the custom function.
    expr : str or sympy.Expr
        Body expression (SymPy expression or string).
    symbol_map : dict[str, str]
        Mapping from symbol names to JS variable names (must include *vars*).

    Returns
    -------
    str
        JavaScript function definition, e.g.::

            function sigmoid(x, k, theta) { return 1/(1+Math.exp(-k*(x-theta))); }
    """
    if isinstance(expr, str):
        expr = sp.parse_expr(expr)
    body = sympy_to_js(expr, symbol_map)
    params = ', '.join(vars)
    return f'function {name}({params}) {{ return {body}; }}'


# ── Internal helpers ──

def _format_number(val):
    """Format a float as a JS-compatible number string."""
    s = repr(float(val))
    if s == 'inf':
        return 'Infinity'
    if s == '-inf':
        return '-Infinity'
    if s == 'nan':
        return 'NaN'
    # Strip trailing '.0' for integer-looking values
    if '.' in s and s.rstrip('0').endswith('.'):
        s = s.rstrip('0').rstrip('.')
    return s


def _handle_add(expr, symbol_map):
    """Convert an Add expression, handling subtraction via sign detection."""
    args = list(expr.args)
    parts = []
    for i, arg in enumerate(args):
        arg_str = sympy_to_js(arg, symbol_map)
        if i == 0:
            parts.append(arg_str)
        elif arg_str.startswith('-'):
            # e.g. '-Math.pow(x, 3)' → ' - Math.pow(x, 3)'
            parts.append(f' - {arg_str[1:]}')
        else:
            parts.append(f' + {arg_str}')
    return ''.join(parts)


def _handle_mul(expr, symbol_map):
    """Convert a Mul expression, handling negative coefficients and division."""
    args = list(expr.args)

    # Separate numeric coefficients from the rest
    nums = [float(a) for a in args if isinstance(a, (Number, Float, Integer))]
    non_nums = [a for a in args if not isinstance(a, (Number, Float, Integer))]

    num_coeff = 1.0
    for n in nums:
        num_coeff *= n

    # Check for division (Pow with negative exponent)
    numerators = []
    denominators = []
    for arg in args:
        if isinstance(arg, Pow) and isinstance(arg.args[1], (Number, Float, Integer)) and float(arg.args[1]) < 0:
            # This is a denominator
            denom_base_expr = arg.args[0]
            denom_exp = -float(arg.args[1])
            base_js = sympy_to_js(denom_base_expr, symbol_map)
            if denom_exp == 1.0:
                denominators.append(base_js)
            else:
                denominators.append(f'Math.pow({base_js}, {_format_number(denom_exp)})')
        elif isinstance(arg, (Number, Float, Integer)):
            continue  # handled via num_coeff
        else:
            numerators.append(sympy_to_js(arg, symbol_map))

    if denominators:
        # Build a fraction
        num_str = ' * '.join(numerators) if numerators else '1'
        den_str = ' * '.join(denominators)
        # Handle the numeric coefficient
        if num_coeff != 1.0:
            num_str = f'{_format_number(num_coeff)} * {num_str}' if num_str != '1' else _format_number(num_coeff)
        # Parenthesise the whole numerator if it's a sum/difference
        if num_coeff < 0 and (' + ' in num_str or ' - ' in num_str):
            result = f'({num_str}) / ({den_str})'
        else:
            result = f'{num_str} / ({den_str})'
        return result

    # Simple multiplication (no division)
    if not non_nums:
        return _format_number(num_coeff)

    rest_strs = numerators  # these were already built above
    if not rest_strs:
        rest_strs = non_nums and [sympy_to_js(a, symbol_map) for a in non_nums]

    if num_coeff == 1.0:
        return ' * '.join(rest_strs)
    elif num_coeff == -1.0:
        return '-' + ' * '.join(rest_strs)
    else:
        coeff_str = _format_number(num_coeff)
        return ' * '.join([coeff_str] + rest_strs)


# ── Self-test ──

if __name__ == '__main__':
    import re
    x, y, a = sp.symbols('x y a')
    symbol_map = {'x': 'x', 'y': 'y', 'a': 'a'}

    def _commutative_eq(result, expected):
        """Check commutative equality: split on +/-, strip leading operator, compare sorted sets."""
        def _token_set(s):
            s = s.replace(' ', '')
            # Split on + or - but keep delimiters to distinguish +/- terms
            tokens = re.split(r'(?=[+-])', s)
            # Strip leading +/- and sort
            cleaned = []
            for t in tokens:
                if t:
                    t = t.lstrip('+-')
                    if t:
                        cleaned.append(t)
            return sorted(cleaned)
        return _token_set(result) == _token_set(expected)

    tests = [
        (x + y, 'x + y'),
        (x * y, 'x * y'),
        (x**2, 'Math.pow(x, 2)'),
        (sp.exp(x), 'Math.exp(x)'),
        (sp.sin(x) + sp.cos(y), 'Math.sin(x)+Math.cos(y)'),
        (a * x - x**3 - y, 'a*x-Math.pow(x,3)-y'),
        (sp.sqrt(x), 'Math.sqrt(x)'),
        (sp.Abs(x), 'Math.abs(x)'),
        (sp.tanh(x), 'Math.tanh(x)'),
        (sp.log(x), 'Math.log(x)'),
        (sp.pi, 'Math.PI'),
    ]

    for expr, expected in tests:
        result = sympy_to_js(expr, symbol_map)
        ok = _commutative_eq(result, expected)
        status = '✓' if ok else '✗'
        print(f"  {status} {expr} -> {result}")
        assert ok, f"FAIL: {expr} -> '{result}' != '{expected}'"

    # Test Heaviside
    h = sp.Heaviside(x)
    result = sympy_to_js(h, symbol_map)
    assert 'x > 0' in result
    print(f"  ✓ {h} -> {result}")

    # Test custom function transpilation
    sigmoid_expr = 1 / (1 + sp.exp(-x))
    result = transpile_custom_function('sigmoid', ['x'], sigmoid_expr, {'x': 'x'})
    assert 'function sigmoid' in result
    assert 'Math.exp' in result
    print(f"  ✓ sigmoid custom function -> {result}")

    print("\nAll transpiler tests passed")
