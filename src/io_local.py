import os
from glob import glob
import gzip as gz
import re
import xarray as xr
import ast
import numpy as np

def create_file_list(path, file_pattern) -> list:
    """Creates a list of files matching file_pattern.

    Args:
        path (str): Path to the files.
        file_pattern (str): Pattern matching the files.

    Raises:
        EnvironmentError: Catches of no files are found that match pattern.

    Returns:
        list: List of files that match pattern.
    """
    if path[-1] != '/':
        path += '/'
    file_list = glob(path + file_pattern)
    if len(file_list) == 0:
        raise EnvironmentError("No file matching found.")
    return file_list

def check_file_list(file_list, name, accepted):
    """Check if file_list contains only files that can be restored

    Args:
        file_list (list of str): File_list previously generated with pattern and path.
        name (str): Prefix name pattern of the file_list files.
        accepted (list of str): Suffixes of the different accepted files.

    Raises:
            EnvironmentError: If a file in file_list (created with path and pattern) doesn't match a restore possibility, it will raise an error. This error could be removed if user names some files the same way as save_trajectory does. Still, it would be better pratice if user moves saved trajectories to a different directory so this doesn't trigger.
    """
    accepted = [ name[:-1]+i for i in accepted ]
    file_list = [i.split("/")[-1].split("\\")[-1] for i in file_list]
    for i in file_list:
        if i not in accepted:
            raise EnvironmentError("Restore not implemented for this file : " + i)


def make_dir(directory) -> None:
    """Create directory if not existing.

    Args:
        directory (str): Name of the directory to create.
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError as error: ## idk if print error is the best choice. Should just let error happen.
        print ('Error: Could not create directory:', directory)

def read_file(filename) -> list:
    """Reads a file and exports the content to a list of strings where each line is an element.

    Args:
        filename (str): path and name of file.

    Returns:
        list of str: Content of the file.
    """
    if re.search('.gz', filename):
        opener = gz.open
    else:
        opener = open

    with opener(filename, 'rt') as f:
        lines = f.readlines()
    return lines

def read_dataset(filename) -> xr.Dataset:
    """Reads filename from netCDF format and turns it into a dataset.

    Args:
        filename (str): Filename to be read. Contains the saved dataset.

    Returns:
        xr.Dataset: Dataset contained in the file.
    """
    return xr.open_dataset(filename)

def read_dataarray(filename) -> xr.DataArray:
    """Reads filename from netCDF format and turns it into a dataarray.


    Args:
        filename (str): Filename to be read. Contains the saved dataarray.

    Returns:
        xr.DataArray: Dataarray contained in the file.
    """
    return xr.open_dataarray(filename)

def read_dict(filename) -> dict:
    """Reads filename from text and turns it into a dict.

    Args:
        filename (str): Filename to be read. Contains the saved dict.

    Returns:
        dict: Dict contained in the file.
    """
    with open(filename, 'rt') as f:
        dict = f.read()
    return ast.literal_eval(dict)

def is_valid_name(name) -> bool:
    """Checks if name is a suitable filename. Could add restrictions if needed.

    Args:
        name (str): name to be tested. Will pass unless it is not string compatible.

    Returns:
        bool: Flag. True if name is valid.
    """
    flag = False
    try:
        str(name)
        flag = True
    except:
        ## raise error maybe
        print('Error: The name of the file needs to be a string or a string compatible container')
        flag = False
    return flag

def save_xarray(ds, path, name, format_type=np.float32) -> None:
    """Saves an xarray Dataset or DataArray object to a netCDF file.

    Args:
        ds (xr.Dataset): Dataset(or dataarray) to be saved.
        path (str): Path where file will be saved
        name (str): Name of the saved file.
    """
    if path[-1] != '/':
        path += '/'
    make_dir(path)
    name = path + name
    ds = ds.astype(dtype=format_type)
    ds.to_netcdf(name)

def save_dict(dict, path, name) -> None:
    """Saves a dictionnary to a text file.

    Args:
        dict (dict): Dictionnary to be saved.
        path (str): Path where file will be saved
        name (str): Name of the saved file.
    """
    if path[-1] != '/':
        path += '/'
    make_dir(path)

    with open(path + name, 'wt') as f:
        f.write(str(dict))

def frame_to_dump(data, name = "cluster", path = "save/clusters/") -> None:
    """Exports a frame from a clustered trajectory and saves it in a LAMMPS dump file format.

    Args:
        data (xr.Dataset): Clustraj dataset but selected for a single timestep and clustering parameter (n_clusters).
        name (str, optional): Name with which the dump file will be saved as. Defaults to "cluster".
        path (str, optional): Path where the dump file will be saved. Defaults to "save/clusters/".
    """

    if is_valid_name(name):
        if path[-1] != '/':
            path += '/'
        make_dir(path)

    data_vars = [ i for i in list(data.keys()) if 'id' in data[i].coords and i != 'labels' ]

    with open(path+name, 'wt') as f:
        f.write( 'ITEM: TIMESTEP\n' )
        f.write( str(data.ts.values) + '\n' )

        f.write( 'ITEM: N_CLUSTER\n' )
        f.write( str(data.n_clusters.values) + '\n' )

        f.write( 'ITEM: NUMBER OF ATOMS\n' )
        f.write( str(len(data.id)) + '\n' )

        f.write( 'ITEM: BOX BOUNDS pp pp pp\n' )
        for comp in data.comp:
            f.write( '0 ' + str(data.bounds.sel(comp = comp).values) + '\n' )

        f.write( 'ITEM: ATOMS id type ' + ' '.join(map(str, data_vars)) + '\n')
        for i in data.id.values:
            line = str(i) + ' ' # id
            line += str(data.labels.sel(id = i).values) + ' ' # type
            for j in data_vars:
                line += str(data[j].sel(id = i).values) + ' ' # atom properties
            f.write(line + '\n')
        f.write('\n')