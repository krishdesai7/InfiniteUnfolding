import numpy as np

def normalize_data(data, substructure_variables, data_streams):
    for var_name in substructure_variables:
        mu = np.mean(data[var_name + data_streams[0]])
        sig = np.std(data[var_name + data_streams[0]])
        for stream in data_streams:
            data[var_name + stream] = (data[var_name + stream] - mu) / sig
    return data