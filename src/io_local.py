"""
Functions for saving and reading data. User should not use them directly and instead use the appropriate saving method for the class related to the dataset, unless a special dataset unrelated to classes is created by user.

Requires modules: xarray
                  sys
                  os
"""
import xarray as xr
import sys
import os


def save_xarray(ds, path, name):
    """Saves a dataset to a netCDF file.

    Args:
        ds (xarray.Dataset): Dataset to be saved.
        path (str): Directory in which the netCDF file will be written.
        name (str): Name of the created netCDF file. Extension will be added automatically.
    """
    if path[-1] != '/':
        path += '/'
    make_dir(path)
    name = path + name + '.nc'
    ds.to_netcdf(name)

def read_xarray(path, name):
    """Creates a dataset from a netCDF file.

    Args:
        path (str): Directory in which the netCDF file can be found.
        name (str): Name of the netCDF file.

    Returns:
        xarray Dataset: Dataset created from the netCDF file
    """
    if path[-1] != '/':
        path += '/'
    ds = xr.open_dataset(path + name)
    return ds

def read_dataarray(path, name):
    """Creates a dataset from a netCDF file.

    Args:
        path (str): Directory in which the netCDF file can be found.
        name (str): Name of the netCDF file.

    Returns:
        xarray Dataset: Dataset created from the netCDF file
    """
    if path[-1] != '/':
        path += '/'
    ds = xr.open_dataarray(path + name)
    return ds

def make_dir(directory):
    """Makes a directory if none can be found of that name.

    Args:
        directory (str): Name of the directory which will be created. If directory already exists, no action is taken.
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
