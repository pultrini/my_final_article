import numpy as np
def normalize_data(data):
    """ Normalize the data to the range [0, 1]. """
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val - min_val == 0:
        return np.zeros_like(data)
    return (data - min_val) / (max_val - min_val)

def calculate_probabilities(data, num_bins):
    """ Calculate the probability distribution of the data. """
    counts, _ = np.histogram(data, bins=num_bins, density=True)
    probabilities = counts/np.sum(counts)
    return probabilities[probabilities > 0]  # Remove zero probabilities

def shannon_entropy(probabilities):
    """ Calculate the Shannon entropy. """
    return -np.sum(probabilities * np.log(probabilities))

def disequilibrium(probabilities, num_bins):
    """ Calculate the disequilibrium. """
    equi_prob = 1.0 / num_bins
    return np.sqrt(np.sum((probabilities - equi_prob)**2))

def lmc_complexity(data, num_bins=100):
    """ Calculate the LMC complexity of the data. """
    normalized_data = normalize_data(data)
    probabilities = calculate_probabilities(normalized_data, num_bins)
    H = shannon_entropy(probabilities)
    D = disequilibrium(probabilities, num_bins)
    C = H * D
    return H, D, C