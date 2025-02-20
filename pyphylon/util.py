"""
General utility functions 
"""

import numpy as np
import pandas as pd

# NMF normalization #


def _get_normalization_diagonals(W):
    # Generate normalization diagonal matrices
    normalization_vals = [1 / np.quantile(W[col], q=0.99) for col in W.columns]
    recipricol_vals = [1 / x for x in normalization_vals]

    D1 = np.diag(normalization_vals)
    D2 = np.diag(recipricol_vals)

    return D1, D2
