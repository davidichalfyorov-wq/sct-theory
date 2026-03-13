"""
SCT Theory — Data I/O utilities for experimental/observational data.

Provides unified interface for reading physics data formats:
    - HDF5 (Planck, LIGO via h5py)
    - ROOT (CERN/LHC via uproot)
    - FITS (astronomy via astropy)
    - CSV/TSV (general tabular data)

Also provides result serialization for reproducibility.
"""

import json
from pathlib import Path

import numpy as np

# =============================================================================
# HDF5 I/O (Planck, LIGO)
# =============================================================================

def read_hdf5(filepath, dataset_name=None):
    """Read data from HDF5 file.

    Parameters:
        filepath: path to .h5 or .hdf5 file
        dataset_name: specific dataset to read (None = list available)

    Returns:
        numpy array if dataset_name given, else dict of dataset names.
    """
    import h5py
    with h5py.File(filepath, 'r') as f:
        if dataset_name is None:
            return _h5_tree(f)
        if dataset_name not in f:
            available = list(f.keys())
            raise KeyError(
                f"Dataset '{dataset_name}' not found in {filepath}. "
                f"Available top-level keys: {available}"
            )
        return f[dataset_name][:]


def _h5_tree(group, prefix=''):
    """Recursively list HDF5 groups and datasets."""
    import h5py
    tree = {}
    for key in group:
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(group[key], h5py.Group):
            tree[path] = _h5_tree(group[key], path)
        else:
            ds = group[key]
            tree[path] = {'shape': ds.shape, 'dtype': str(ds.dtype)}
    return tree


# =============================================================================
# ROOT I/O (CERN/LHC data via uproot)
# =============================================================================

def read_root(filepath, tree_name=None, branches=None):
    """Read data from ROOT file.

    Parameters:
        filepath: path to .root file
        tree_name: TTree name (None = list available trees)
        branches: list of branch names to read (None = all)

    Returns:
        dict of {branch_name: numpy_array} if tree_name given,
        else list of tree names.
    """
    import uproot
    with uproot.open(filepath) as f:
        if tree_name is None:
            return list(f.keys())
        tree = f[tree_name]
        if branches is None:
            branches = tree.keys()
        return {b: tree[b].array(library="np") for b in branches}


# =============================================================================
# FITS I/O (astronomy)
# =============================================================================

def read_fits(filepath, hdu=1):
    """Read data from FITS file.

    Parameters:
        filepath: path to .fits file
        hdu: HDU index (default: 1 = first data extension)

    Returns:
        astropy Table or numpy array depending on content.
    """
    from astropy.io import fits
    with fits.open(filepath, memmap=False) as hdul:
        data = hdul[hdu].data
        # Copy to detach from internal FITS buffers (compressed HDUs
        # and FITS_rec objects can hold dangling references after close)
        if data is None:
            raise ValueError(f"read_fits: HDU {hdu} contains no data")
        return np.array(data)


def read_fits_header(filepath, hdu=0):
    """Read FITS header metadata.

    Parameters
    ----------
    filepath : str or Path
        Path to the FITS file.
    hdu : int, optional
        HDU index to read (default 0 = primary).

    Returns
    -------
    dict
        Header cards as {keyword: value} mapping.
    """
    from astropy.io import fits
    with fits.open(filepath) as hdul:
        return dict(hdul[hdu].header)


# =============================================================================
# CSV/TSV I/O (Pantheon+, DESI BAO, general tabular data)
# =============================================================================

def read_csv(filepath, comment='#', delimiter=None, columns=None, skip_header=0,
             has_header=True):
    """Read tabular data from CSV/TSV/whitespace-delimited file.

    Handles the common formats used in astrophysics data releases
    (Pantheon+, DESI, Fermi catalogs, etc.).

    Parameters:
        filepath: path to data file
        comment: comment character (default '#')
        delimiter: column delimiter (None = auto-detect: comma for .csv, whitespace otherwise)
        columns: list of column names (overrides header row)
        skip_header: number of header lines to skip before data
        has_header: whether the first data row is a header (default True).
                    Set False for headerless files to avoid losing the first row.

    Returns:
        dict with:
            'data': numpy structured array or dict of arrays
            'columns': list of column names
            'n_rows': number of data rows
    """
    import pandas as pd
    filepath = str(filepath)
    if not isinstance(skip_header, int) or skip_header < 0:
        raise ValueError(
            f"read_csv: skip_header must be a non-negative integer, got {skip_header}"
        )
    if delimiter is None:
        if filepath.endswith('.csv'):
            delimiter = ','
        else:
            import warnings
            delimiter = r'\s+'
            warnings.warn(
                f"read_csv: delimiter auto-detected as {delimiter!r} for non-CSV "
                f"file. Use delimiter= to override.",
                stacklevel=2,
            )
    header_arg = 0 if (has_header and columns is None) else None
    df = pd.read_csv(
        filepath, comment=comment, delimiter=delimiter,
        header=header_arg, names=columns, skiprows=skip_header, engine='python',
    )
    result = {col: df[col].values for col in df.columns}
    return {
        'data': result,
        'columns': list(df.columns),
        'n_rows': len(df),
    }


# =============================================================================
# RESULT SERIALIZATION
# =============================================================================

def save_results(filepath, results, metadata=None):
    """Save computation results as JSON (reproducibility).

    Parameters:
        filepath: output path (.json)
        results: dict of {name: value} (values must be JSON-serializable)
        metadata: optional dict with computation metadata
    """
    output = {
        'results': _serialize(results),
        'metadata': metadata if metadata is not None else {},
    }
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=_json_default)


def _json_default(obj):
    """Fallback JSON serializer — raises on unrecognized types."""
    raise TypeError(
        f"Object of type {type(obj).__name__} is not JSON-serializable. "
        f"Update _serialize() to handle this type. Value: {obj!r}"
    )


def load_results(filepath):
    """Load saved results from JSON."""
    with open(filepath, encoding='utf-8') as f:
        data = json.load(f)
    if 'results' not in data:
        raise ValueError(
            f"load_results: file {filepath} does not contain a 'results' key. "
            f"Available keys: {list(data.keys())}"
        )
    return _deserialize(data['results']), data.get('metadata', {})


def _deserialize(obj):
    """Reconstruct complex/ndarray types from JSON-serialized form."""
    if isinstance(obj, dict):
        if obj.get('__complex__'):
            return complex(obj['real'], obj['imag'])
        if obj.get('__ndarray_complex__'):
            flat = [_deserialize(v) for v in obj['data']]
            return np.array(flat).reshape(obj['shape'])
        return {k: _deserialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deserialize(v) for v in obj]
    return obj


def _serialize(obj):
    """Convert numpy/mpmath types to JSON-serializable."""
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        if np.issubdtype(obj.dtype, np.complexfloating):
            return {
                '__ndarray_complex__': True,
                'shape': list(obj.shape),
                'data': [_serialize(v) for v in obj.flat],
            }
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        val = float(obj)
        if not np.isfinite(val):
            import warnings
            warnings.warn(
                f"_serialize: non-finite float value ({val}) converted to null "
                f"for JSON compliance. Use save_results only for finite data.",
                stacklevel=3,
            )
            return None
        return val
    if isinstance(obj, (np.complexfloating, complex)):
        return {'__complex__': True, 'real': float(obj.real), 'imag': float(obj.imag)}
    # mpmath types (high-precision form factor results)
    try:
        import mpmath
        from mpmath.ctx_mp_python import _mpf as _mpf_base
        if isinstance(obj, _mpf_base):  # catches mpf, constant (e.g. mp.pi)
            return float(obj)
        if isinstance(obj, mpmath.mpc):
            return {'__complex__': True, 'real': float(obj.real), 'imag': float(obj.imag)}
    except ImportError:
        pass
    return obj
