import numpy as np
from pathlib import Path


def binarized_mnist(path=Path('./datasets/binarized_mnist')):
    def lines_to_np_array(lines):
        return np.array([[int(i) for i in line.split()] for line in lines])

    data = {}
    for split in ['train', 'valid', 'test']:
        with open(path / 'binarized_mnist_{}.amat'.format(split)) as f:
            lines = f.readlines()
            data[split] = lines_to_np_array(lines).astype('float32')
            idxs = list(range(data[split].shape[0]))
            np.random.shuffle(idxs)
            data[split] = data[split][idxs]

    return data
