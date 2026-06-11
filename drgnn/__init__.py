import warnings

try:
    from drgnn.model import DRGNN
except ImportError:
    warnings.warn("DRGNN model import failed (missing dgl/torch)")
    DRGNN = None

try:
    from drgnn.data import DataControl
except ImportError:
    DataControl = None

try:
    from drgnn.utils.config import BASE_DIR, DATA_DIR, RESULTS_DIR  # noqa: F401
except ImportError:
    pass

__version__ = "1.0.0"
