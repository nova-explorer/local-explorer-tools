import os
from glob import glob
import gzip as gz
import re

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
    if re.search('.gz', filename): # if the file is compressed with gz
        f = gz.open(filename, 'rt')
    else:
        f = open(filename, 'rt')
    file = f.readlines()
    f.close()
    return file

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

def save_xarray(ds, path, name) -> None:
    """Saves an xarray Dataset or DataArray object to a netCDF file.

    Args:
        ds (xr.Dataset): Dataset(or dataarray) to be saved.
        path (str): Path where file will be saved
        name (str): Name of the saved file.
    """
    if path[-1] != '/':
        path += '/'
    make_dir(path)
    name = path + name + '.nc'
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

    name = path + name + '.dict'
    f = open(name, "rt")
    for i in dict:
        f.write( str(i) + " : " + str(dict[i]) )
    f.close()