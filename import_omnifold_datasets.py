import numpy as np
import energyflow as ef

datasets = {
    'Pythia26': ef.zjets_delphes.load(
        'Pythia26', 
        num_data=-1,
        which='all',
        exclude_keys=['particles']
    ),
    'Herwig': ef.zjets_delphes.load(
        'Herwig', 
        num_data=-1,
        which='all',
        exclude_keys=['particles']
    )
}

synthetic = datasets['Pythia26']  
nature    = datasets['Herwig']    

obs_multifold = ['m', 'M', 'w', 'tau21', 'zg', 'sdm']

def get_var(dset, var, ptype):
    if var == 'm':
        return dset[ptype + '_jets'][:, 3]  
    elif var == 'M':
        return dset[ptype + '_mults']
    elif var == 'w':
        return dset[ptype + '_widths']
    elif var == 'tau21':
        return dset[ptype + '_tau2s'] / (dset[ptype + '_widths'] + 1e-50)
    elif var == 'zg':
        return dset[ptype + '_zgs']
    elif var == 'sdm':
        jet_sq = dset[ptype + '_jets'][:, 0]**2
        eps = 1e-12 * np.mean(jet_sq)
        return np.log(dset[ptype + '_sdms']**2 / np.maximum(jet_sq, eps) + eps)
    else:
        raise ValueError(f"Unknown variable '{var}'")

out_data = {}
for var in obs_multifold:
    out_data[var + '_true']      = get_var(synthetic, var, 'gen')
    out_data[var + '_reco']      = get_var(synthetic, var, 'sim')
    out_data[var + '_true_alt']  = get_var(nature, var, 'gen')
    out_data[var + '_reco_alt']  = get_var(nature, var, 'sim')

np.savez('rawdata_omnifold.npz', **out_data)