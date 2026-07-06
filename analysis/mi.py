import numpy as np
from sklearn.metrics import mutual_info_score


def compute_symbol_mi(symbols):
    symbols = np.asarray(symbols)
    unique_symbols = np.unique(symbols)
    n = len(unique_symbols)
    mi = np.zeros((n, n), dtype=np.float32)

    for i, left_symbol in enumerate(unique_symbols):
        for j, right_symbol in enumerate(unique_symbols):
            mi[i, j] = mutual_info_score(
                symbols == left_symbol, symbols == right_symbol
            )

    return mi
