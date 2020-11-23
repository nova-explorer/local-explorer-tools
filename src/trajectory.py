"""
Classes for importing LAMMPS atom trajectory into a xarray dataset. The trajectory class also has a dataset with ellipsoid vectors.
traj_options contains all the inputs for generating the trajectory class.

Usage: traj_opt = t.trajectory_options(path = "./trajectories/",
                                       file_pattern = "sma.dump.gz",
                                       exclude_types = [1],
                                       monomer_types = 3)
       traj = t.trajectory(traj_opt)
       traj.save_trajectory(path = "./save/")
Accessing data:
       traj.vectors.cm
       traj.vectors.coord
       traj.atoms.xu

Requires modules: xarray
                  numpy
                  glob
                  re
                  pandas
                  gzip

Requires scripts: compute_op
                  io_local

TODO: * save options
      * optimize reading file - > dataset process
      * more verbosity
      * test/remove trim_to_types method
      * add nb_atoms to the dataset
"""
import xarray as xr
import numpy as np
from glob import glob
import re
import pandas as pd
import gzip as gz

import compute_op as cop
import io_local as io

class trajectory():
    """Generates trajectory

    Args:
        - options (trajectory_options): Contains options for reading the trajectory files

    Attributes:
        - Arguments
        - atoms (xarray.Dataset): Contains the trajectory information of the atoms as written in the LAMMPS files.
        - vectors (xarray.Dataset): Contains the trajectory of the vectors of the monomers.
    """
    def __init__(self, options):
        """Creates the class. Creates the atoms and vectors Datasets.

        Args:
            - options (trajectory_options): Contains the options for reading the trajectory files
        """
        self.options = options
        if self.options.restore:
            self.restore_trajectory()
        else:
            self.atoms = self.read_traj()
            self.vectors = self.add_vectors()
            self.vectors = cop.qmatrix(self.vectors)
        # v_traj = self.trim_to_types(self.full_traj, self.options.monomer_types['atom']) ## needs testing
        # delimiters = self.trim_to_types(self.full_traj, self.options.monomer_types['del']) ## needs testing

    def read_traj(self):
        """Creates the atoms dataset by reading the LAMMPS trajectory files.

        Options:
            - file_list (list of strings): Contains the files that will be read and converted to a Dataset
            - exclude_types (list of ints): Atoms with type that matches any in this list will not be imported into the atoms Dataset

        Returns:
            - xarray.Dataset: Dataset that contains the trajectory of the atoms contained in the files. Classed by the coordinates ts(current timestep of the atoms data), id(atom id of the current atom) and comp(x, y and z coordinates when needed).
        """
        bounds = {'x':[], 'y':[], 'z':[]}
        bounds_list = []
        bounds_array = []
        comp = ['x', 'y', 'z']

        nb_atoms = None
        step = None
        timestep = []
        data = []
        for filename in self.options.file_list: # For each file in file_list
            if re.search('.gz', filename): # if the file is compressed with gz
                f = gz.open(filename, 'rt')
            else:
                f = open(filename, 'rt')
            file = f.readlines()
            f.close()

            cnt = 0 # serves as the line counter
            # scans for the line with the properties of the atoms
            while not re.search('ITEM: ATOMS', str(file[cnt])): # str() only serves to suppress pyright overload error.
                cnt += 1
            coordinates = file[cnt].strip().split() [2:] # Dataset variables. The 1st 3 are skipped

            for i, line in enumerate(file): # loop over the lines in the current file

                if line.strip() == 'ITEM: TIMESTEP':
                    step = int(file[i+1])

                elif line.strip() == 'ITEM: NUMBER OF ATOMS':
                    nb_atoms = int(file[i+1])

                elif line.strip() == 'ITEM: BOX BOUNDS pp pp pp':
                    bounds['x'] = [float(val) for val in file[i+1].strip().split() ]
                    bounds['y'] = [float(val) for val in file[i+2].strip().split() ]
                    bounds['z'] = [float(val) for val in file[i+3].strip().split() ]

                elif re.search('ITEM: ATOMS', str(line)): # str() only serves to suppress pyright overload error.
                    data.append(self.lines_to_df(file[i+1:i+nb_atoms-1], coordinates)) # will put the lines with all the atoms property in a pandas dataframe which is appended in a list for all timesteps
                    timestep.append(step) # list with all the timesteps

                    bounds_list = []
                    for i in bounds:
                        bounds_list.append(bounds[i][1] - bounds[i][0]) # just reformats the bounds to a good format for the atoms dataset
                    bounds_array.append(bounds_list)

        data = self.dfs_to_ds(data, timestep)
        data['bounds'] = xr.DataArray(bounds_array, coords = [data.ts, comp], dims = ['ts', 'comp'])

        return data

    def lines_to_df(self, lines, column_names):
        """Takes the lines containing atoms properties for a single timestep and imports them in a dataframe

        Args:
            - lines (list of strings): Contains the properties of the atoms. Each string is one atom.
            - column_names (list of strings): Contains the names of the atoms properties.

        Options:
            - exclude_types (list of ints): Atoms with type that matches any in this list will not be imported into the atoms Dataset

        Returns:
            pandas DataFrame: Dataframe with the properties of the atoms. Each column is a property and each index is an atom.
        """
        data = []
        type_id = None

        # Checks the position of the property: type.
        for i, name in enumerate(column_names):
            if name == 'type':
                type_id = i

        for i in lines:
            line = i.strip().split()
            if self.options.exclude_types: # will not skip types if exclude_types is None or False
                for j in self.options.exclude_types:
                    if int(line[type_id]) != int(j):
                        data.append(line)
            else:
                data.append(line)

        df = pd.DataFrame(data, columns = column_names)
        df = df.apply(pd.to_numeric)
        df = df.set_index(['id'])
        df = df.sort_index()

        return df

    def dfs_to_ds(self, dfs, list_ids):
        """Turns a list of dataframes into a dataset

        Args:
            - dfs (list of dataframes): Contains the dataframes that needs to be converted to a dataset
            - list_ids (list of ints): Contains the coordinate of each item in dfs

        Returns:
            Dataset: Dataset with each variables matching the dataframes' columns. The coordinates match list_ids and the dataframes' index
        """
        ds = []
        for i, df in enumerate(dfs):
            ds.append(xr.Dataset())
            for j in df:
                ds[i][j] = df[j].to_xarray()
            ds[i].update( {'ts': ( 'ts', [list_ids[i]] )} )
        return xr.concat(ds, dim = 'ts')

    def trim_to_types(self, traj, types):
        """removes all but those types
            Need to check if relevant
            TODO: Need to test many subsequent trims
        """
        traj = traj.where( traj.type.isin(types), drop = True )
        return traj

    def add_vectors(self):
        """Creates a Dataset containing vectors information based on the atoms dataset
        Options:
            - monomer_types
        Returns:
            - xarray.Dataset: Contains vector trajectories for specified monomer(s). Has same coordinates as the atoms dataset

        TODO:
            - streamline the vector extremities determination to match a general behavior and not just the ellispoid+pseudoatom template
        """
        data = self.atoms # just because it's shorter
        mono_type = self.options.monomer_types
        droppers = []
        coords = ['x', 'y', 'z']

        # creates a list containing all the columns that don't are not positions
        for i in data.data_vars:
            if i not in ['xu', 'yu', 'zu']:
                droppers.append(i)

        vectors = []
        for i in range(len(data.id)):
            if int(data.isel(id = i).type[0]) == mono_type: # type doesnt change during simulation
                # since ids are ordered, the atom before and after the ellipsoid type are pseudo atoms and are used as vector extremities
                atom_0 = data.drop_vars(droppers).isel(id = i-1)
                atom_1 = data.drop_vars(droppers).isel(id = i+1)

                # we take advantage of the arithmetic properties of datasets
                v = atom_1 - atom_0
                norm = np.sqrt(v.xu**2 + v.yu**2 + v.zu**2)
                v = v / norm
                v['coord'] = xr.DataArray( np.transpose([v.xu, v.yu, v.zu]), coords = [v.ts, coords], dims = ['ts', 'comp'] )
                v['norm'] = norm

                alpha = np.arccos(v.xu)
                beta = np.arccos(v.yu)
                gamma = np.arccos(v.zu)
                v['angle'] = xr.DataArray( np.transpose([alpha, beta, gamma]), coords = [v.ts, coords], dims = ['ts', 'comp'] )

                x_cm = data.isel(id = i).xu
                y_cm = data.isel(id = i).yu
                z_cm = data.isel(id = i).zu
                v['cm'] = xr.DataArray( np.transpose([x_cm, y_cm, z_cm]), coords = [v.ts, coords], dims = ['ts', 'comp'] )

                v = v.drop_vars(['xu', 'yu', 'zu'])
                v.update( {'id': ( 'id', [i] )} )

                vectors.append(v)

        return xr.concat(vectors, dim = 'id')

    def save_trajectory(self, path = 'save/'):
        """Saves the trajectory class in a nc (netCDF) file. Much faster than read_trajectory.

        Args:
            - path (str, optional): Directory where the trajectory will be saved. Defaults to 'save/'.

        TODO:
            - Check integration with traj_options
            - Allow custom naming of files
            - Estimate size of files?
            - The try: except: to see if the datasets exist are a little rough. See if there's a better way to do that.
            - Integrate for the cluster class
            - Integrate a way to save options as well
        """
        try:
            print('saving atoms...')
            io.save_xarray(self.atoms, path, 'atoms_io')
        except:
            print('No trajectory for atoms; not saving dataset...')

        try:
            print('saving vectors...')
            io.save_xarray(self.vectors, path, 'vectors_io')
        except:
            print('No trajectory for vectors; not saving dataset...')

    def restore_trajectory(self, include = 'all'):
        """Restores a saved trajectory from netCDF files to regenerate the trajectory class.

        Args:
            - include (str, optional): Which saved datasets to restore. Defaults to 'all'.

        TODO:
            - Damn this part is rough a bit.
            - Allow custom naming. Maybe recognition of files based on a pattern not defined by user, eg. atoms_io.nc -> name.aio.nc
            - Remove if include ==.... section, it's clunky and pattern recognition would be more versatile.
            - Integrate option restoration.
        """
        path = self.options.path
        DO_ATOMS = False
        DO_VECTORS = False
        DO_OPTIONS = False

        if include == 'all':
            DO_ATOMS = True
            DO_VECTORS = True
            DO_OPTIONS = True
        elif include == 'atoms':
            DO_ATOMS = True
        elif include == 'vectors':
            DO_VECTORS = True
        elif include == 'options':
            DO_OPTIONS = True
        else:
            print('argument for include :', include, 'is not recognized!')

        if DO_ATOMS:
            self.atoms = io.read_xarray(path, name = 'atoms_io.nc')
        if DO_VECTORS:
            self.vectors = io.read_xarray(path, name = 'vectors_io.nc')

class trajectory_options():
    """Generates options for the trajectory. See Args for description of the options.

    Args:
        - path (str, optional): Directory of the trajectory files to read, be it LAMMPS dumps or netCDF previously saved. Defaults to "./", the current directory.
        - file_pattern (str, optional): Pattern to match for finding the trajectory files. As of current version, it only works for restore=False. Defaults to "ellipsoid.*".
        - exclude_types ([int], optional): Types that match the ones in this list will not be imported in the trajectory datasets. As of current version, it only works for restore=False. Defaults to None.
        - monomer_types ([int], optional): Types that correspond to the ellipsoid of the monomer. Vectors extremities will be chosen as the particles one before and one after this type in a id ordered dataset. Defaults to None.
        - restore (bool, optional): If True, the trajectory class will be generated from restoring the datasets from netCDF files. Defaults to False.

    Attributes:
        - Arguments
        - file_list ([str]): List of the files that match pattern in path. Corresponds to the files that will be read for creating the trajectory class.

    TODO:
        - monomer_types is rough and need work for a more general approach
    """
    def __init__( self, path = "./", file_pattern = "ellipsoid.*", exclude_types = None, monomer_types = None, restore = False):
        """Creates the class. Initializes the arguments attributes and creates file_list.

        Args:
            - path (str, optional): Directory of the trajectory files to read, be it LAMMPS dumps or netCDF previously saved. Needs to end with "/". Defaults to "./", the current directory.
            - file_pattern (str, optional): Pattern to match for finding the trajectory files. Works with the Unix style pathname pattern expansion. As of current version, it only works for restore=False. Defaults to "ellipsoid.*".
            - exclude_types ([int], optional): Types that match the ones in this list will not be imported in the trajectory datasets. As of current version, it only works for restore=False. Defaults to None.
            - monomer_types ([int], optional): Types that correspond to the ellipsoid of the monomer. Vectors extremities will be chosen as the particles one before and one after this type in a id ordered dataset. Defaults to None.
            - restore (bool, optional): If True, the trajectory class will be generated from restoring the datasets from netCDF files. Defaults to False.

        """
        self.path = path
        self.exclude_types = exclude_types
        self.monomer_types = monomer_types
        self.restore = restore
        if not self.restore:
            self.create_file_list(path, file_pattern)

    def create_file_list(self, path, file_pattern):
        """Creates the file list from the files matching file_pattern in path

        Args:
            - path (str): Directory of the trajectory files to read. Needs to end with "/"
            - file_pattern (str): Pattern to match for finding the trajectory files. Works with the Unix style pathname pattern expansion.

        Raises:
            - EnvironmentError: Script doesn't continue if no file is found.

        TODO:
            - EnvironmentError is a placeholder. Check if a more appropriate exists/can be created.
        """
        self.file_list = glob(path + file_pattern)
        print("Found", len(self.file_list), "matching", file_pattern, "in", path, "...")
        if len(self.file_list) == 0:
            raise EnvironmentError
