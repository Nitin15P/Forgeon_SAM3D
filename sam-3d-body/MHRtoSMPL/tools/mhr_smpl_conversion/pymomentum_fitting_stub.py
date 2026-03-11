"""
Stub for pymomentum_fitting that provides only the constants and a dummy class
needed by pytorch_fitting.py and conversion.py when running the PyTorch-only path.
This avoids importing pymomentum.solver and sklearn which conflict in the same process.
"""

_NUM_RIG_PARAMETERS = 204
_NUM_BODY_BLENDSHAPES = 20
_NUM_HEAD_BLENDSHAPES = 20
_NUM_HAND_BLENDSHAPES = 5


class PyMomentumModelFitting:
    """Dummy class - not used when method='pytorch'."""
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "PyMomentumModelFitting is not available in stub mode. "
            "Use method='pytorch' instead."
        )
