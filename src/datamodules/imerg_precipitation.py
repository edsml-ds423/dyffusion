from __future__ import annotations

import datetime
import os
from os.path import join
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import dask
import numpy as np
import xarray as xr
from src.datamodules.abstract_datamodule import BaseDataModule
from src.datamodules.datasets.configs.imerg_data import IMERGEarlyRunDataConfig
from src.datamodules.torch_datasets import MyTensorDataset
from src.utilities.utils import get_logger, raise_error_if_invalid_type, raise_error_if_invalid_value

log = get_logger(__name__)


class IMERGPrecipitationDataModule(BaseDataModule):
    """"""
    def __init__(
        self,
        data_dir: str,
        boxes: Union[List, str] = "all",
        # validation_boxes: Union[List, str] = "all",  # splitting data based on t not boxes.
        # predict_boxes: Union[List, str] = "all",     # ^^
        # predict_slice: Optional[slice] = slice("2020-12-01", "2020-12-31"),  #TODO: is this just  = test?
        box_size: int = 256,
        window: int = 1,
        horizon: int = 1,
        prediction_horizon: int = None,  # None means use horizon
        multi_horizon: bool = False,
        pixelwise_normalization: bool = True,
        save_and_load_as_numpy: bool = False,
        **kwargs,
    ):
        """initialisation."""
        raise_error_if_invalid_type(data_dir, possible_types=[str], name="data_dir")
        raise_error_if_invalid_value(pixelwise_normalization, [True], name="pixelwise_normalization")
        raise_error_if_invalid_value(box_size, [256], name="box_size")

        if "imerg-precipitation" not in data_dir:
            for name in ["imerg-precipitation"]:
                if os.path.isdir(join(data_dir, name)):
                    data_dir = join(data_dir, name)
                    break

        super().__init__(data_dir=data_dir, **kwargs)
        self.save_hyperparameters()
        
        self.normalization_hparams = {
            "norm": True,
            "percentiles": {"1": 0.0,
                            "99": 4.569999694824219},
            }

        # Set the temporal slices for the train, val, and test sets. lims (not inclusive).
        self.train_lims = [None, "2023-01-01 00:00:00"]  # jan-20 - may-23
        self.val_lims = ["2023-01-01 00:00:00", "2023-07-01 00:00:00"]  # may-23 - sep-23
        self.test_lims = ["2023-07-01 00:00:00", "2024-01-01 00:00:00"]  # sep-23 - jan-24

        # convert lims to index key (specific to IMERG).
        self.train_index = [self.convert_str_dt_to_imerg_int_dt(t) for t in self.train_lims]
        self.val_index = [self.convert_str_dt_to_imerg_int_dt(t) for t in self.val_lims]
        self.test_index = [self.convert_str_dt_to_imerg_int_dt(t) for t in self.test_lims]

        # set the temporal slices for the train, val, and test sets.
        self.train_slice = slice(*self.train_index)
        self.val_slice = slice(*self.val_index)
        self.test_slice = slice(*self.test_index)

        self.stage_to_slice = {
            "fit": slice(self.train_slice.start, self.val_slice.stop),
            "validate": self.val_slice,
            "test": self.test_slice,
            "predict": self.test_slice,
            None: None,
        }
    
    @staticmethod
    def convert_str_to_datetime(str_dt: str, dt_format: str = "%Y-%m-%d %H:%M:%S"):
        return datetime.datetime.strptime(str_dt, dt_format)

    @staticmethod
    def convert_str_dt_to_imerg_int_dt(str_dt: Optional[str]):
        """"""
        if str_dt is not None:
            datetime_dt = IMERGPrecipitationDataModule.convert_str_to_datetime(str_dt)
            y, m, d, hh, mm = IMERGPrecipitationDataModule.get_year_month_day_hour_min_from_datetime(
                datetime_dt, zero_padding=2
            )
            hhmm_in_dt_s = IMERGEarlyRunDataConfig().download_file_path_info["hhmm_to_dt_secs"].get(f"{hh}{mm}")
            str_dt = int(f"{y}{m}{d}{hhmm_in_dt_s}")
        return str_dt

    @staticmethod
    def get_year_month_day_hour_min_from_datetime(dt: datetime.datetime, zero_padding: int):
        """"""
        _year = str(dt.year)
        _month = str(dt.month).zfill(zero_padding)
        _day = str(dt.day).zfill(zero_padding)
        _hour = str(dt.hour).zfill(zero_padding)
        _min = str(dt.minute).zfill(zero_padding)

        return _year, _month, _day, _hour, _min

    # TODO: consider adding this to BaseDataModule
    def get_horizon(self, split: str):
        if split in ["predict", "test"]:
            return self.hparams.prediction_horizon or self.hparams.horizon
        else:
            return self.hparams.horizon

    # TODO: Create a GriddedDataBaseModel(BaseDataModule) and move this there.
    def _check_args(self):
        boxes = self.hparams.boxes
        h = self.hparams.horizon
        w = self.hparams.window
        assert isinstance(h, list) or h > 0, f"horizon must be > 0 or a list, but is {h}"
        assert w > 0, f"window must be > 0, but is {w}"
        assert self.hparams.box_size > 0, f"box_size must be > 0, but is {self.hparams.box_size}"
        assert isinstance(boxes, Sequence) or boxes in [
            "all"
        ], f"boxes must be a list or 'all', but is {self.hparams.boxes}"

    # TODO: move to GriddedDataBaseModel(BaseDataModule).
    def load_xarray_ds(self, stage: str) -> bool:
        b1 = not self.hparams.save_and_load_as_numpy
        return b1 or self._get_numpy_filename(stage) is None or stage == "predict"

    def _create_time_index_mask(self, index_to_slice, _slice: slice):
        """
        create a mask for the t_index in the imerg .nc files.
        """
        if (_slice.start is None) and (_slice.stop is not None):
            mask = index_to_slice < _slice.stop
        elif (_slice.stop is None) and (_slice.start is not None):
            mask = index_to_slice >= _slice.start
        elif (_slice.stop is not None) and (_slice.start is not None):
            mask = (index_to_slice >= _slice.start) & (index_to_slice < _slice.stop)
        else:
            mask = None
        return mask

    def get_glob_pattern(self, boxes: Union[List, str] = "all"):
        """"""
        ddir = Path(self.hparams.data_dir)
        if isinstance(boxes, Sequence) and boxes != "all":
            self.n_boxes = len(boxes)
            log.info(f"training & validation using {self.n_boxes} (i, j) boxes: {boxes}.")
            return [ddir / f"imerg.box.{b.split(',')[0]}.{b.split(',')[1]}.sequenced.nc" for b in boxes]
        elif boxes == "all":
            # compute the number of boxes
            log.info(f"training & validation using 'all' (i, j) boxes: {boxes}.")
            self.n_boxes = len(list(ddir.glob("imerg.box.*.*.sequenced.nc")))
            return str(ddir / "imerg.box.*.*.sequenced.nc")
        else:
            raise ValueError(f"Unknown value for boxes: {boxes}")

    def _get_numpy_filename(self, stage: str):
        """"""
        split = "train" if stage in ["fit", None] else stage
        if stage == "predict":
            return None
        fname = join(self.numpy_dir, f"{self.dataset_identifier}_{split}")
        if os.path.isfile(fname + ".npy"):
            return fname + ".npy"
        elif os.path.isfile(fname + ".npz"):
            return fname + ".npz"
        return None

    def get_ds_xarray_or_numpy(self, split: str, time_slice) -> Union[xr.DataArray, Dict[str, np.ndarray]]:
        """"""
        if self.load_xarray_ds(split):
            glob_pattern = self.get_glob_pattern(self.hparams.boxes)
            log.info(f"Using data from {self.n_boxes} boxes for ``{split}`` split.")
            log.info(f"{split} data split: [{time_slice.start}, {time_slice.stop}]")
            with dask.config.set(**{"array.slicing.split_large_chunks": False}):
                try:
                    data = xr.open_mfdataset(
                        paths=glob_pattern,
                        combine="nested",
                        concat_dim="t_index",
                    )
                    # TODO: maybe create .nc with t_index being an int to avoid this conversion.
                    # TODO: change t_index to <time>.
                    data.coords["t_index"] = data.coords["t_index"].astype(int)  # convert to int to allow < > ops.
                    mask = self._create_time_index_mask(index_to_slice=data.t_index, _slice=time_slice)
                    masked_data = data.where(mask, drop=True)

                except OSError as e:
                    raise ValueError(
                        f"Could not open imerg-precipitation data files from {glob_pattern}. "
                        f"Check that the data directory is correct: {self.hparams.data_dir}"
                    ) from e

            return masked_data.__xarray_dataarray_variable__

        else:
            log.info(f"Loading data from numpy file {self._get_numpy_filename(split)}")
            fname = self._get_numpy_filename(split)
            assert fname is not None, f"Could not find numpy file for split {split}"
            npz_file = np.load(fname, allow_pickle=False)
            # print(f'Keys in npz file: {list(npz_file.keys())}, files: {npz_file.files}')
            return {k: npz_file[k] for k in npz_file.files}

    def create_and_set_dataset(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        """Create a torch dataset from the given xarray DataArray and return it."""
        # TODO: can multi & single horizon be joined into single self.create_ function?
        if self.hparams.multi_horizon:
            return self.create_and_set_dataset(*args, **kwargs)
            # return self.create_and_set_dataset_multi_horizon(*args, **kwargs)
        else:
            return self.create_and_set_dataset(*args, **kwargs)
            # return self.create_and_set_dataset_single_horizon(*args, **kwargs)

    def create_and_set_dataset(self, split: str, dataset: xr.DataArray) -> Dict[str, np.ndarray]:
        """Create a torch dataset from the given xarray DataArray and return it."""
        window, horizon = self.hparams.window, self.get_horizon(split)

        #TODO: try and speed this up.
        X = dataset.to_numpy()  # dims: (N, S, H, W)
        X = np.expand_dims(X, 2)  # dims: (N, S, C, H, W)

        assert X.shape == (dataset.shape[0], window + horizon, 1, self.hparams["box_size"], self.hparams["box_size"])
        return {"dynamics": X}

    def setup(self, stage: Optional[str] = None):
        """Load data. Set internal variables: self._data_train, self._data_val, self._data_test."""
        ds_train = self.get_ds_xarray_or_numpy("fit", self.train_slice) if stage in ["fit", None] else None
        ds_val = (
            self.get_ds_xarray_or_numpy("validate", self.val_slice) if stage in ["fit", "validate", None] else None
        )
        ds_test = self.get_ds_xarray_or_numpy("test", self.test_slice) if stage in ["test", None] else None
        ds_predict = (
            self.get_ds_xarray_or_numpy("predict", self.stage_to_slice["predict"]) if stage == "predict" else None
        )
        ds_splits = {"train": ds_train, "val": ds_val, "test": ds_test, "predict": ds_predict}        
        for split, split_ds in ds_splits.items():
            if split_ds is None:
                continue
            if isinstance(split_ds, xr.DataArray):
                # create the numpy arrays from the xarray dataset.
                numpy_tensors = self.create_and_set_dataset(split, split_ds)

                # save the numpy tensors to disk (if requested).
                if self.hparams.save_and_load_as_numpy:
                    self.save_numpy_arrays(numpy_tensors, split)
            else:
                # Alternatively, load the numpy arrays from disk (if requested).
                numpy_tensors = split_ds

            # Create the pytorch tensor dataset
            tensor_ds = MyTensorDataset(numpy_tensors, dataset_id=split, **self.normalization_hparams)

            # Save the tensor dataset to self._data_{split}
            setattr(self, f"_data_{split}", tensor_ds)
            assert getattr(self, f"_data_{split}") is not None, f"Could not create {split} dataset"

        # print sizes of the datasets (how many examples).
        self.print_data_sizes(stage)
