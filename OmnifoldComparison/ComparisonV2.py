#!/usr/bin/env python
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import numpy as np
import pickle
import concurrent.futures
import matplotlib.pyplot as plt
from omnifold import DataLoader, MLP, MultiFold
from sklearn.model_selection import train_test_split
import tensorflow as tf
import sys

# ----------------------------
# TensorFlow GPU Setup
# ----------------------------
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Enabled memory growth for {len(gpus)} GPU(s).")
    except RuntimeError as e:
        print(e)
else:
    print("No GPUs found. Running on CPU.")

# ----------------------------
# Global Parameters and Data
# ----------------------------
data_size = 10**4
sim_size = 10**5
mu_true, mu_gen = 0, -1
iterations = 10   # Number of outer iterations to run
dims = 1

rng = np.random.default_rng()
truth = rng.normal(mu_true, 1, (data_size, 1))
gen   = rng.normal(mu_gen, 1, (sim_size, 1))

smearings = np.linspace(1, 20)  # smearing factors to try

# ----------------------------
# Function to Process One Smearing Iteration
# ----------------------------
def process_smearing(smearing, gpu_id=None, random_state=42):
    """
    Process one smearing iteration: perform train/test split, build loaders, 
    instantiate models and MultiFold, then run unfolding and return the results.
    
    Returns:
        (smearing, truth_test, gen_test, unfolded_weights)
    """
    # Restrict visible GPUs in this process.
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"Process for smearing {smearing} assigned to GPU {gpu_id}")
    else:
        print(f"Process for smearing {smearing} running on CPU")
    
    # Create smeared versions:
    data = truth * smearing
    sim  = gen   * smearing

    # Split truth/data and gen/sim separately (they have different sizes)
    truth_train, truth_test, data_train, data_test = train_test_split(
        truth, data, test_size=0.2, random_state=random_state
    )
    gen_train, gen_test, sim_train, sim_test = train_test_split(
        gen, sim, test_size=0.2, random_state=random_state
    )

    # Build data loaders
    nature_loader = DataLoader(reco=data_train, normalize=True)
    mc_loader     = DataLoader(reco=sim_train, gen=gen_train, normalize=True)
    
    # Instantiate the models (using TensorFlow under the hood)
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

# ----------------------------
# Main Processing Block: Outer Loop Over Iterations
# ----------------------------
if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    # Make sure output directory exists.
    output_dir = "of_preds"
    os.makedirs(output_dir, exist_ok=True)

    num_gpus = len(gpus)
    max_workers = num_gpus if num_gpus > 0 else None

    # Run iterations sequentially.
    for j in range(5, iterations):
        print(f"Starting iteration {j}")
        results = dict()
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, s in enumerate(smearings):
                gpu_id = i % num_gpus if num_gpus > 0 else None
                futures.append(executor.submit(process_smearing, s, gpu_id))
            
            for fut in concurrent.futures.as_completed(futures):
                try:
                    smearing, truth_test, gen_test, unfolded_weights = fut.result()
                    # Save one triplet per smearing.
                    results[smearing] = [truth_test, gen_test, unfolded_weights]
                    print(f"Iteration {j}: Finished smearing = {smearing}")
                except Exception as e:
                    print(f"Iteration {j}: Error in a smearing iteration: {e}")
        
        # Save the dictionary for this iteration to a file "j.pkl"
        outfile = os.path.join(output_dir, f"{j}.pkl")
        with open(outfile, "wb") as f:
            pickle.dump(results, f)
        print(f"Saved iteration {j} results to {outfile}")