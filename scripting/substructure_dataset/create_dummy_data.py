import numpy as np

# Load the data from the .npz file
data = np.load('/global/home/users/krishdesai/InfiniteUnfolding/rawdata_omnifold.npz')

# Inspect the keys in the loaded data to see what you have
print(data.keys())

substructure_variables = ['m', 'M', 'w', 'tau21', 'zg', 'sdm']
data_streams = ['_true', '_true_alt', '_reco', '_reco_alt']
n_variables = len(substructure_variables)
N = 2**19



normalize = True
    
for var_name in data.files:
    globals()[var_name] = data[var_name][:N]
    
if normalize:
    for var_name in substructure_variables:
        mu = np.mean(globals()[var_name+data_streams[0]])
        sig = np.std(globals()[var_name + data_streams[0]])
        for stream in data_streams:
            globals()[var_name+stream] = (globals()[var_name+stream] - mu)/sig
            
from sklearn.model_selection import train_test_split

# Prepare your inputs and labels
xvals_truth = np.array([np.concatenate([globals()[f"{var}_true_alt"],
                                        globals()[f"{var}_true"]]) for var in substructure_variables],
                       dtype = np.float32).T
xvals_reco = np.array([np.concatenate([globals()[f"{var}_reco_alt"],
                                       globals()[f"{var}_reco"]]) for var in substructure_variables],
                     dtype = np.float32).T
yvals = np.concatenate([np.zeros(len(globals()[f"{substructure_variables[0]}_true_alt"])),
                        np.ones(len(globals()[f"{substructure_variables[0]}_true"]))],
                      )

# Double-check the shapes to make sure everything is consistent
print(f"xvals_truth shape: {xvals_truth.shape}")
print(f"xvals_reco shape: {xvals_reco.shape}")
print(f"yvals shape: {yvals.shape}")

# Split the data into training and temporary sets (temp will be further split into val and test)
X_train_truth, X_temp_truth, X_train_reco, X_temp_reco, Y_train, Y_temp = train_test_split(
    xvals_truth, xvals_reco, yvals, test_size=0.25, random_state=42)

# Split the temporary set into validation and test sets
X_val_truth, X_test_truth, X_val_reco, X_test_reco, Y_val, Y_test = train_test_split(
    X_temp_truth, X_temp_reco, Y_temp, test_size=0.5, random_state=42)

# Print shapes to verify the splits
print(f"Training set: {X_train_truth.shape}, {X_train_reco.shape}, {Y_train.shape}")
print(f"Validation set: {X_val_truth.shape}, {X_val_reco.shape}, {Y_val.shape}")
print(f"Test set: {X_test_truth.shape}, {X_test_reco.shape}, {Y_test.shape}")

# Save the training data
np.savez('train_data.npz', xvals_truth=X_train_truth, xvals_reco=X_train_reco, yvals=Y_train)

# Save the validation data
np.savez('val_data.npz', xvals_truth=X_val_truth, xvals_reco=X_val_reco, yvals=Y_val)

# Save the test data
np.savez('test_data.npz', xvals_truth=X_test_truth, xvals_reco=X_test_reco, yvals=Y_test)