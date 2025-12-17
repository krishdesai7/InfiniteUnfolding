import tensorflow as tf
import numpy as np
from config.model_config import batch_size

def load_dataset():
    """Loads the datasets from npz files and prepares them for training."""

    datasets = {}
    
    for split in ['train', 'val', 'test']:
        data = np.load(f'substructure_dataset/{split}_data.npz')
        dataset = tf.data.Dataset.from_tensor_slices((data['xvals_truth'], data['xvals_reco'], data['yvals']))
        
        # Shuffle, batch, and prefetch the dataset
        if split == 'train':
            dataset = dataset.shuffle(buffer_size=10000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        else:
            dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)       
        datasets[split] = dataset

    return datasets['train'], datasets['val'], datasets['test']