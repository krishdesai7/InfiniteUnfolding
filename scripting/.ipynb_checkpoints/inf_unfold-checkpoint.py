from sklearn.neighbors import KernelDensity
import numpy as np

def main(x_det_train, x_part_train, y_train, x_part_test, y_test):
    # truth = x_part_test[y_test == 1]
    # gen = x_part_test[y_test == 0]
    # # Fit KDE to 'truth' data
    # kde_truth = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(truth)
    # log_dens_truth = kde_truth.score_samples(gen)
    
    # # Fit KDE to 'gen' data
    # kde_gen = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(gen)
    # log_dens_gen = kde_gen.score_samples(gen)
    
    # # Compute per-event weights as the ratio of densities
    # per_event_weights = np.exp(log_dens_truth - log_dens_gen)
    
    # return per_event_weights
    # Separate the data into 'truth' and 'gen' based on the labels
    truth = x_part_test[y_test == 1]
    gen = x_part_test[y_test == 0]
    
    bins = np.linspace(-3.5, 3.5, 50)
    t_counts, bin_edges = np.histogram(truth, bins=bins)
    gen_counts, _ = np.histogram(gen, bins=bin_edges)
    
    gen_counts = np.where(gen_counts == 0, 1e-8, gen_counts)
    
    bin_weights = t_counts / gen_counts
    
    indices = np.digitize(gen, bins=bin_edges) - 1  # Subtract 1 for zero-based indexing
    
    indices = np.clip(indices, 0, len(bin_weights) - 1)
    
    per_event_weights = bin_weights[indices]
    
    return per_event_weights