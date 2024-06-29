from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections.abc import Sequence
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class MyTensorDataset(Dataset[Dict[str, Tensor]]):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Args:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    tensors: Dict[str, Tensor]

    @staticmethod
    def convert_to_tensor(item):
        key, tensor = item
        if isinstance(tensor, np.ndarray):
            return key, torch.from_numpy(tensor).float()
        return key, tensor

    def __init__(self, tensors: Dict[str, Tensor] | Dict[str, np.ndarray], dataset_id: str = ""):
        # create a {key: tensor} from {key: numpy} dict.
        with ThreadPoolExecutor() as executor:
            tensors = dict(executor.map(self.convert_to_tensor, tensors.items()))

        any_tensor = next(iter(tensors.values()))
        self.dataset_size = any_tensor.size(0)
        for k, value in tensors.items():
            if torch.is_tensor(value):
                assert value.size(0) == self.dataset_size, "Size mismatch between tensors"
            elif isinstance(value, Sequence):
                assert (
                    len(value) == self.dataset_size
                ), f"Size mismatch between list ``{k}`` of length {len(value)} and tensors {self.dataset_size}"
            else:
                raise TypeError(f"Invalid type for tensor {k}: {type(value)}")

        self.tensors = tensors
        self.dataset_id = dataset_id

    def __getitem__(self, index):
        # TODO: debug printing - REMOVE
        print(f"--> MyTensorDataset.__getitem__({index})")
        A = {key: tensor[index] for key, tensor in self.tensors.items()}

        # TODO: remove this.
        # save a plot of the index sample.
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 5, figsize=(12, 5))
        plt.rcParams.update({"font.size": 8})
        axs[0].imshow(A["dynamics"][0, 0, :, :])
        axs[1].imshow(A["dynamics"][2, 0, :, :])
        axs[2].imshow(A["dynamics"][4, 0, :, :])
        axs[3].imshow(A["dynamics"][6, 0, :, :])
        axs[4].imshow(A["dynamics"][8, 0, :, :])
        axs[4].set_title("x0+h (h=8)")
        axs[0].set_title("x0")
        plt.savefig(f"misc_images/interp_input_seq_{index}.png")

        return A
        # return {key: tensor[index] for key, tensor in self.tensors.items()}

    def __len__(self):
        return self.dataset_size


def get_tensor_dataset_from_numpy(*ndarrays, dataset_id="", dataset_class=MyTensorDataset, **kwargs):
    tensors = [torch.from_numpy(ndarray.copy()).float() for ndarray in ndarrays]
    return dataset_class(*tensors, dataset_id=dataset_id, **kwargs)


class AutoregressiveDynamicsTensorDataset(Dataset[Tuple[Tensor, ...]]):
    data: Tensor

    def __init__(self, data, horizon: int = 1, **kwargs):
        assert horizon > 0, f"horizon must be > 0, but is {horizon}"
        self.data = data
        self.horizon = horizon

    def __getitem__(self, index):
        # input: index time step
        # output: index + horizon time-steps ahead
        return self.data[index], self.data[index + self.horizon]

    def __len__(self):
        return len(self.data) - self.horizon
