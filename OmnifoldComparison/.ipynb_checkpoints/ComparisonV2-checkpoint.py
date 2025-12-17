import numpy as np
from omnifold import DataLoader, MLP, MultiFold
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import concurrent.futures

data_size = 10**4
sim_size = 10**5
mu_true, mu_gen = 0, -1

iterations = 5
dims = 1

rng = np.random.default_rng(1042)
truth = rng.normal(mu_true, 1, (data_size, 1))
gen = rng.normal(mu_gen, 1, (sim_size, 1))

with open("of_preds.pkl", "rb") as f:
    of_preds = pickle.load(f)

deltas_gen = dict()
deltas_of = dict()

def triangular_discriminator(p, q, bin_widths):
    """
    Triangular discriminator:
      0.5 * sum_i [((p_i - q_i)^2)/(p_i + q_i)] * bin_width_i * 1e3
    Bins where p_i + q_i = 0 contribute zero.
    """
    denom = p + q
    mask = denom > 0
    val = np.zeros_like(p)
    val[mask] = (p[mask] - q[mask])**2 / denom[mask]
    return 0.5 * np.sum(val * bin_widths) * 1e3

def process_smearing(smearing):
    """
    Process one smearing iteration: perform train/test split, build loaders, 
    instantiate models and MultiFold, then run unfolding and return the results.
    
    Returns:
        smearing, truth_test, gen_test, unfolded_weights
    """
    # Create smeared versions:
    data = truth * smearing
    sim  = gen   * smearing

    # For truth/data and gen/sim we need to split separately since they differ in size.
    from sklearn.model_selection import train_test_split
    truth_train, truth_test, data_train, data_test = train_test_split(
        truth, data, test_size=0.2
    )
    gen_train, gen_test, sim_train, sim_test = train_test_split(
        gen, sim, test_size=0.2, random_state=random_state
    )

    # Build data loaders (you may need to import or define these classes)
    nature_loader = DataLoader(reco=data_train, normalize=True)
    mc_loader     = DataLoader(reco=sim_train, gen=gen_train, normalize=True)
    
    # Create models
    reco_model = MLP(dims)
    gen_model  = MLP(dims)
    
    # Create and run MultiFold
    omnifold = MultiFold(
        "RAN Comparison",
        reco_model,
        gen_model,
        nature_loader,
        mc_loader,
        batch_size = 512,
        niter = 5,
        epochs = 10,
        verbose = False,
        lr = 5e-5,
    )
    omnifold.Unfold()
    unfolded_weights = omnifold.reweight(gen_test, omnifold.model2, batch_size=1000)
    return smearing, truth_test, gen_test, unfolded_weights

results = dict() 
with concurrent.futures.ProcessPoolExecutor() as executor:
    futures = [executor.submit(process_smearing, s) for s in smearings]
    for fut in concurrent.futures.as_completed(futures):
        try:
            smearing, truth_test, gen_test, unfolded_weights = fut.result()
            results.setdefault(smearing, []).append([truth_test, gen_test, unfolded_weights])
            print(f"Finished smearing={smearing}")
        except Exception as e:
            print(f"Error in a smearing iteration: {e}")
of_preds = results

with open("of_preds.pkl", "wb") as f:
    pickle.dump(of_preds, f)

sorted_keys = sorted([float(k) for k in of_preds.keys()])
print([f"{k:.3f}" for k in sorted_keys])

for smearing in sorted_keys:
    delta_gen_list = []
    delta_of_list = []
    for triplet in of_preds[smearing]:
        truth_test, gen_test, of_weights = triplet

        # Build histograms with density=True
        counts_truth, bin_edges = np.histogram(truth_test, density=True)
        counts_gen,   _         = np.histogram(gen_test, density=True)
        # Use the event-level weights when making the histogram for the reweighted gen.
        counts_of,    _         = np.histogram(gen_test, weights=np.atleast_2d(of_weights).T, density=True)
        
        bin_widths = np.diff(bin_edges)
    
        # Compute the triangular discriminators for this triplet
        delta_gen = triangular_discriminator(counts_gen, counts_truth, bin_widths)
        delta_of  = triangular_discriminator(counts_of,  counts_truth, bin_widths)
        
        # Append the individual delta values
        delta_gen_list.append(delta_gen)
        delta_of_list.append(delta_of)
        
    # Average the deltas over the K triplets for this smearing value
    deltas_gen[smearing] = (np.mean(delta_gen_list), np.std(delta_gen_list)/np.sqrt(len(delta_gen_list) - 1))
    deltas_of[smearing] = (np.mean(delta_of_list), np.std(delta_of_list)/np.sqrt(len(delta_of_list) - 1))

gen_means = [deltas_gen[k][0] for k in sorted_keys]
gen_errs  = [deltas_gen[k][1] for k in sorted_keys]

# Extract means and errors for the OF vs Truth data
of_means  = [deltas_of[k][0] for k in sorted_keys]
of_errs   = [deltas_of[k][1] for k in sorted_keys]

# Plot using error bars
plt.errorbar(gen_keys, gen_means, yerr=gen_errs, fmt='o', color='r', label='delta_gen')
plt.errorbar(of_keys, of_means, yerr=of_errs, fmt='o', color='g', label='delta_of')

# Add labels and legend
plt.xlabel("Smearing")
plt.ylabel("Triangular Discriminator")
plt.legend()
plt.show()

# Plot
plt.scatter(gen_x, gen_y, color='r', label='delta_gen')
plt.scatter(of_x, of_y, color='g', label='delta_of')

# Add labels and legend
plt.xlabel("Smearings")
plt.ylabel(r"$\Delta_{VL}$")
plt.legend()
plt.savefig("OF_delta.pdf", bbox_inches = 'tight')