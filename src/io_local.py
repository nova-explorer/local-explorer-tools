"""[summary]

Returns:
    [type]: [description]
"""
import xarray as xr
import sys
import os


def save_dataset(ds, path, name):
    """[summary]

    Args:
        ds ([type]): [description]
        path ([type]): [description]
        name ([type]): [description]
    """
    if path[-1] != '/':
        path += '/'
    make_dir(path)
    name = path + name + '.nc'
    ds.to_netcdf(name)

def read_dataset(path, name):
    """[summary]

    Args:
        path ([type]): [description]
        name ([type]): [description]

    Returns:
        [type]: [description]
    """
    if path[-1] != '/':
        path += '/'
    ds = xr.open_dataset(path + name)
    return ds

def make_dir(directory):
    """[summary]

    Args:
        directory ([type]): [description]
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
