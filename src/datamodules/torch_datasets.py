from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections.abc import Sequence
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from src.utilities.utils import get_logger

log = get_logger(__name__)


class MyTensorDataset(Dataset[Dict[str, Tensor]]):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Args:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    tensors: Dict[str, Tensor]

    @staticmethod
    def log_linear_norm(
        arr: Tensor, c: float, d: float, a: float = 0, b: float = 1
    ) -> Tensor:
        """
        Apply log +1 to the linear norm of a tensor/arr. The linear normalisation is:

        arr_lin_norm = (arr_in - c).(b-a / d-c) + a

        where a and b are the lower and upper limit of the
        resulting range and c and d are the lower and upper
        values of the input range. By default, the resulting range of
        values [a, b] is chosen to be between 0 and 1.

        arr_log_lin_norm = np.log(arr_lin_norm + 1)
        """

        _min = 0.0
        _max = 3.230964096613462

        scale = (b - a) / (d - c)
        return ((np.log1p(((arr - c) * scale) + a)) - _min) / (_max - _min)

    @staticmethod
    def reverse_log_linear_norm(
        arr: Tensor, c: float, d: float, a: float = 0, b: float = 1
    ) -> Tensor:
        """
        Reverse the log_linear_normalisation.
        arr_in = (exp(arr_out - 1) - a).(d-c / b-a) + c)
        """
        scale = (d - c) / (b - a)
        return ((np.expm1(arr) - a) * scale) + c
    

    @staticmethod
    def min_max_norm(
        arr: Tensor, _min: float, _max: float) -> Tensor:
        """"""
        return (arr - _min) / (_max - _min)
    
    @staticmethod
    def reverse_min_max_norm(
        arr: Tensor, _min: float, _max: float) -> Tensor:
        """"""
        return (arr * (_max - _min)) + _min

    @staticmethod
    def convert_to_tensor(item):
        key, tensor = item
        if isinstance(tensor, np.ndarray):
            return key, torch.from_numpy(tensor).float()
        return key, tensor

    @staticmethod
    def _validate_tensors(tensors, dataset_size):
        """Check that the number of tensors == dataset size."""
        for k, value in tensors.items():
            if torch.is_tensor(value):
                assert value.size(0) == dataset_size, f"Size mismatch for tensor {k}"
            elif isinstance(value, Sequence):
                assert (
                    len(value) == dataset_size
                ), f"Size mismatch between list ``{k}`` of length {len(value)} and tensors {dataset_size}"
            else:
                raise TypeError(f"Invalid type for tensor {k}: {type(value)}")

    def __init__(
        self,
        tensors: Dict[str, Tensor] | Dict[str, np.ndarray],
        dataset_id: str = "",
        **kwargs,
    ):
        """initialisation."""
        log.info(f"creating {dataset_id} tensor dataset.")
        self.normalize = kwargs.get("norm", False)
        self.train_percentiles = kwargs.get("percentiles", None)
        self.minmax = kwargs.get("min_max", None)

        with ThreadPoolExecutor() as executor:
            # create a {key: tensor} from {key: numpy} dict.
            tensors = dict(executor.map(self.convert_to_tensor, tensors.items()))

        any_tensor = next(iter(tensors.values()))
        self.dataset_size = any_tensor.size(0)
        self._validate_tensors(tensors, self.dataset_size)

        self.tensors = tensors
        self.dataset_id = dataset_id

        if self.normalize:
            if self.train_percentiles is None:
                raise ValueError(
                    "Percentiles must be provided in kwargs when normalize is True."
                )
            log.info("normalizing data using 1st and 99th training data percentiles.")
            log.info(f"1st: {self.train_percentiles['1']}, 99th: {self.train_percentiles['99']}.")
            log.info(f"applying min max norm w/ min={self.minmax[0]}, max={self.minmax[-1]}.")
            self._normalize_tensors()

    def _normalize_tensors(self):
        for key, tensor in self.tensors.items():
            self.tensors[key] = self.log_linear_norm(
                arr=tensor,
                c=self.train_percentiles["1"],
                d=self.train_percentiles["99"],
            )
            
            # # applying min max to the linear log norm.
            # self.tensors[key] = self.min_max_norm(
            #     arr=self.log_linear_norm(arr=tensor, c=self.train_percentiles["1"], d=self.train_percentiles["99"]),
            #     _min=self.minmax[0], 
            #     _max=self.minmax[1],
            # )

    def __getitem__(self, index):
        return {key: tensor[index] for key, tensor in self.tensors.items()}

    def __len__(self):
        return self.dataset_size


def get_tensor_dataset_from_numpy(
    *ndarrays, dataset_id="", dataset_class=MyTensorDataset, **kwargs
):
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
