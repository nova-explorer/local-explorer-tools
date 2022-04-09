import os
from glob import glob
import gzip as gz
import re

def create_file_list(path, file_pattern) -> list:
    if path[-1] != '/':
        path += '/'
    file_list = glob(path + file_pattern)
    if len(file_list) == 0:
        raise EnvironmentError
    return file_list

def make_dir(directory) -> None:
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError as error:
        print ('Error: Could not create directory:', directory)

def read_file(filename) -> list:
    if re.search('.gz', filename): # if the file is compressed with gz
        f = gz.open(filename, 'rt')
    else:
        f = open(filename, 'rt')
    file = f.readlines()
    f.close()
    return file

def is_valid_name(name):
    flag = False
    try:
        str(name)
        flag = True
    except:
        print('Error: The name of the file needs to be a string or a string compatible container')
        flag = False
    return flag

def save_xarray(ds, path, name):
    if path[-1] != '/':
        path += '/'
    make_dir(path)
    name = path + name + '.nc'
    ds.to_netcdf(name)

def save_dict(dict, path, name):
    if path[-1] != '/':
        path += '/'
    make_dir(path)

    name = path + name + '.dict'
    f = open(name, "rt")
    for i in dict:
        f.write( str(i) + " : " + str(dict[i]) )
    f.close()