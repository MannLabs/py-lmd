import os

import matplotlib
import matplotlib.pyplot

# Disable jit so that the coverage is correctly reported
os.environ["NUMBA_DISABLE_JIT"] = "1"

matplotlib.use("Agg")
matplotlib.pyplot.ioff()
