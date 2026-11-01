import numpy as np


def calculate_psi(reference, current):
    bins = np.histogram_bin_edges(reference, bins=10)

    bins[0] = -np.inf
    bins[-1] = np.inf

    reference_counts = np.histogram(reference, bins=bins)[0]
    current_counts = np.histogram(current, bins=bins)[0]

    reference_percent = reference_counts / len(reference)
    current_percent = current_counts / len(current)

    reference_percent = np.maximum(reference_percent, 0.0001)
    current_percent = np.maximum(current_percent, 0.0001)

    psi = np.sum(
        (current_percent - reference_percent)
        * np.log(current_percent / reference_percent)
    )

    return psi