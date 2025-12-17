import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'figure.figsize': (4, 4),
    'figure.dpi': 120,
    'font.family': 'serif',
    'font.size': 20,
    'axes.grid': True,
    'errorbar.capsize': 5,
    'lines.linewidth': 2,
    'lines.linestyle': 'dashed',
    'lines.markerfacecolor': 'none',
    'lines.markersize': 10,
})


# ----------------------------
# Helper: Load and aggregate results from all pickle files
# ----------------------------
pkl_dir = "of_preds"
# List all .pkl files in the directory and sort them numerically.
pkl_files = sorted([f for f in os.listdir(pkl_dir) if f.endswith(".pkl")],
                    key=lambda x: int(os.path.splitext(x)[0]))

aggregate = dict()
for file in pkl_files:
    file_path = os.path.join(pkl_dir, file)
    with open(file_path, "rb") as f:
        of_preds = pickle.load(f)
    # Each file is a dict: { smearing: triplet }
    for smearing, triplet in of_preds.items():
        aggregate.setdefault(smearing, []).append(triplet)

def triangular_discriminator(p, q, bin_widths):
    """
    Triangular discriminator:
      0.5 * sum_i [((p_i - q_i)^2)/(p_i + q_i)] * bin_width_i * 1e3
    Bins where p_i + q_i == 0 contribute zero.
    """
    denom = p + q
    mask = denom > 0
    val = np.zeros_like(p, dtype=float)
    val[mask] = (p[mask] - q[mask])**2 / denom[mask]
    return 0.5 * np.sum(val * bin_widths) * 1e3

deltas_gen = dict()
deltas_of  = dict()
for smearing in sorted(aggregate.keys()):
    delta_gen_list = []
    delta_of_list  = []
    for triplet in aggregate[smearing]:
        truth_test, gen_test, of_weights = triplet

        # Build histograms using density=True (so the result is a PDF)
        counts_truth, bin_edges = np.histogram(truth_test, density=True)
        counts_gen,   _         = np.histogram(gen_test, density=True)
        counts_of,    _         = np.histogram(gen_test,
                                                weights=np.atleast_2d(of_weights).T,
                                                density=True)
        bin_widths = np.diff(bin_edges)

        delta_gen = triangular_discriminator(counts_gen, counts_truth, bin_widths)
        delta_of  = triangular_discriminator(counts_of, counts_truth, bin_widths)

        delta_gen_list.append(delta_gen)
        delta_of_list.append(delta_of)

    # Compute mean and standard error for each smearing value.
    # Standard error is computed as std/√(N-1)
    N = len(delta_gen_list)
    deltas_gen[smearing] = (np.mean(delta_gen_list),
                            np.std(delta_gen_list) / np.sqrt(N - 1) if N > 1 else 0)
    deltas_of[smearing]  = (np.mean(delta_of_list),
                            np.std(delta_of_list)  / np.sqrt(N - 1) if N > 1 else 0)

# ----------------------------
# Plotting: Errorbar Scatter Plot
# ----------------------------
sorted_keys = sorted([float(k) for k in deltas_gen.keys()])
gen_means = [deltas_gen[k][0] for k in sorted_keys]
gen_errs  = [deltas_gen[k][1] for k in sorted_keys]
of_means  = [deltas_of[k][0] for k in sorted_keys]
of_errs   = [deltas_of[k][1] for k in sorted_keys]

plt.errorbar(sorted_keys, gen_means, yerr=gen_errs, fmt='o', color='r', label='Delta (Gen vs Truth)')
plt.errorbar(sorted_keys, of_means,  yerr=of_errs,  fmt='o', color='g', label='Delta (OF vs Truth)')
plt.xlabel("Smearing")
plt.ylabel("Triangular Discriminator")
plt.legend()
plt.savefig("OF_delta.pdf", bbox_inches='tight')