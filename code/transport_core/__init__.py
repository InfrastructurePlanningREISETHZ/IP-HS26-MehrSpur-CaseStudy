"""Prepared transport model used through transport_model_interface.

Modules are imported explicitly so optional routing dependencies remain lazy.
"""


def _alias_numpy_core_for_pickle_compat() -> None:
    """Read prepared NumPy 2 arrays when the course environment uses NumPy 1.26."""
    import importlib
    import sys
    try:
        importlib.import_module("numpy._core.numeric")
    except ImportError:
        sys.modules["numpy._core.numeric"] = importlib.import_module("numpy.core.numeric")


_alias_numpy_core_for_pickle_compat()
