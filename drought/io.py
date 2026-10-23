"""MAT, array and tabular input with explicit variable selection."""
from __future__ import annotations

from pathlib import Path
import numpy as np


def load_mat(path: str | Path, variable_names=None) -> dict:
    from scipy.io import loadmat
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        return {k: v for k, v in loadmat(path, variable_names=variable_names,
                                         squeeze_me=False, struct_as_record=False).items()
                if not k.startswith('__')}
    except (NotImplementedError, ValueError) as error:
        import h5py
        if not h5py.is_hdf5(path):
            raise error
        result = {}
        with h5py.File(path, 'r') as handle:
            for name, dataset in handle.items():
                if name.startswith('#') or (variable_names and name not in variable_names):
                    continue
                if isinstance(dataset, h5py.Dataset) and dataset.dtype.kind != 'O':
                    value = np.asarray(dataset)
                    result[name] = value.transpose(tuple(range(value.ndim - 1, -1, -1)))
        return result


def find_array(mat: dict, names: str | list[str] | tuple[str, ...]) -> np.ndarray:
    names = [names] if isinstance(names, str) else list(names)
    for name in names:
        if name in mat:
            return np.asarray(mat[name])
    raise KeyError(f"Required variable {names}; available variables: {list(mat)}")


def read_series(path: str | Path, variable: str | list[str], expected=None) -> np.ndarray:
    values = np.asarray(find_array(load_mat(path), variable), dtype=float).squeeze()
    if expected is not None:
        shape = (expected,) if isinstance(expected, int) else tuple(expected)
        if values.shape != shape:
            raise ValueError(f'{path}: expected {shape}, found {values.shape}')
    return values


def load_array(path: str | Path, variable: str | None = None) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == '.mat':
        if variable is None:
            raise ValueError('Specify the MAT variable.')
        return find_array(load_mat(path), variable)
    if path.suffix.lower() == '.npy':
        return np.load(path, allow_pickle=False)
    if path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as data:
            if variable is None:
                if len(data.files) != 1:
                    raise ValueError('Specify the NPZ variable.')
                variable = data.files[0]
            return np.asarray(data[variable])
    raise ValueError(f'Unsupported array format: {path.suffix}')
