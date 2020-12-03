"""
Functions for saving and reading data. User should not use them directly and instead use the appropriate saving method for the class related to the dataset, unless a special dataset unrelated to classes is created by user.

Requires modules: xarray
                  sys
                  os
"""
import xarray as xr
import sys
import os
from glob import glob


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

def read_dataset(file):
    """Creates a dataset from a netCDF file.

    Args:
        path (str): Directory in which the netCDF file can be found.
        name (str): Name of the netCDF file.

    Returns:
        xarray Dataset: Dataset created from the netCDF file
    """
    ds = xr.open_dataset(file)
    return ds

def read_dataarray(file):
    """Creates a dataset from a netCDF file.

    Args:
        path (str): Directory in which the netCDF file can be found.
        name (str): Name of the netCDF file.

    Returns:
        xarray Dataset: Dataset created from the netCDF file
    """
    da = xr.open_dataarray(file)
    return da

def create_file_list(path, file_pattern):
    """Creates the file list from the files matching file_pattern in path

    Args:
        - path (str): Directory of the trajectory files to read. Needs to end with "/"
        - file_pattern (str): Pattern to match for finding the trajectory files. Works with the Unix style pathname pattern expansion.

    Raises:
        - EnvironmentError: Script doesn't continue if no file is found.

    TODO:
        - EnvironmentError is a placeholder. Check if a more appropriate exists/can be created.
    """
    if path[-1] != '/':
        path += '/'
    make_dir(path)
    file_list = glob(path + file_pattern)
    print("Found", len(file_list), "matching", file_pattern, "in", path, "...")
    if len(file_list) == 0:
        raise EnvironmentError
    return file_list

def is_valid_name(name):
    ## not sure raise and then return work well together
    flag = False
    try:
        str(name)
        flag = True
    except:
        raise ValueError('The name of the file needs to be a string or a string compatible container')
    return flag

def export_ovito(atoms, labels, mono_type=3, name='cluster.dump', path='save/'):
    if is_valid_name(name):
        if path[-1] != '/':
            path += '/'
        make_dir(path)

        atoms = atoms.where(atoms.type==mono_type, drop=True)
        atoms = atoms.drop_vars(['bounds', 'type'])
        data_vars = [ i for i in atoms.data_vars]
        atoms['labels'] = xr.DataArray(labels, coords=[atoms.id.values], dims=['id'])

        f = open(path+name, 'wt')
        f.write('ITEM: TIMESTEP\n')
        f.write(''.join(map(str,atoms.ts.values)) + '\n')
        f.write('ITEM: NUMBER OF ATOMS\n')
        f.write(str(len(atoms.id)) + '\n')

        f.write('ITEM: ATOMS id type ' + ' '.join(map(str, data_vars)) + '\n')
        for i in atoms.id.values:
            line = str(i) + ' ' # id
            line += str(atoms.labels.sel(id=i).values)  + ' '# type
            for j in data_vars:
                line += str(atoms[j].sel(id=i).values[0]) + ' '
            f.write(line + '\n')

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
