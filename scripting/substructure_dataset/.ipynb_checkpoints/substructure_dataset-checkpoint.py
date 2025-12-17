import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf

class SubstructureDataset(tfds.core.GeneratorBasedBuilder):
    """Dataset for substructure variables."""

    VERSION = tfds.core.Version('1.0.0')

    def _info(self):
        """Returns the dataset metadata (description, features, etc.)."""
        return tfds.core.DatasetInfo(
            builder=self,
            description="Dataset of substructure variables for ML models.",
            features=tfds.features.FeaturesDict({
                'x_true': tfds.features.Tensor(shape=(None,), dtype=tf.float32),
                'x_reco': tfds.features.Tensor(shape=(None,), dtype=tf.float32),
                'label': tfds.features.ClassLabel(num_classes=2),
            }),
            supervised_keys=('x_true', 'x_reco', 'label'),
            homepage='https://your-dataset-homepage',
            citation=r"""@article{my_citation}...""",
        )
    def _split_generators(self, dl_manager):
        """Returns the splits of the dataset (train, val, test)."""
        # Paths to the data files
        return {
            'train': self._generate_examples(split='train'),
            'val': self._generate_examples(split='val'),
            'test': self._generate_examples(split='test'),
        }

    def _generate_examples(self, split):
        """Yields examples as (key, example) tuples."""
        # Load the data based on the split
        if split == 'train':
            data = np.load('dummy_data/train_data.npz')
        elif split == 'val':
            data = np.load('dummy_data/val_data.npz')
        else:
            data = np.load('dummy_data/test_data.npz')

        # Generate examples
        for i in range(len(data['xvals_truth'])):
            yield i, {
                'x_true': data['xvals_truth'][i],
                'x_reco': data['xvals_reco'][i],
                'label': int(data['yvals'][i]),
            }