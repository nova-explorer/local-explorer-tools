import xarray as xr
import numpy as np
import re
import pandas as pd

import io_local as iol
import compute_structure as cs

class trajectory( object ): ## need object here?

    def __init__(self,
                 path="./", pattern="ellipsoid.*", exclude=None, vector_types=None, restore=False, updates=True
                 ) -> None:
        """LAMMPS trajectory converted to an xarray dataset. Additionally ellipsoidal particles are defined as vectors and stored in another dataset for further computation.

        Args:
            path (str, optional): Path to input files. Input files can be either LAMMPS trajectory dumps or previously saved trajectory using the save_trajectory method. Defaults to "./".
            pattern (str, optional): Pattern matching input files. Input files can be either LAMMPS trajectory dumps or previously saved trajectory using the save_trajectory method. Defaults to "ellipsoid.*".
            exclude (list of int or None/False, optional): Types to exclude from the trajectory. Defaults to None.
            vector_types (list of int or None/False, optional): Types that match the ellipsoidal particles vector. The vector will be defined as the particles with ids just before and after this ellipsoidal particle. Defaults to None. TODO Should not default to None since code will fail, should find a way to match any particle sequence.
            restore (bool, optional): If True, the input files will be read as a restore of the trajectory class. Those input files need to have been created by the save_trajectory method. Defaults to False.
            updates (bool, optional): If True, prints will update the user of current progress. Defaults to True.
        TODO: Attributes
        """

        self.updates = updates

        self.path = path
        self.pattern = pattern
        self.file_list = iol.create_file_list(path, pattern)

        self._print( "\tFile list created and containing {} files\n".format(len(self.file_list)) )

        if restore:
            self._print("\tRestoring trajectory...\n")

            self.__restore_trajectory()
        else:
            self._print("\tReading trajectory...\n")

            self.exclude = exclude ## could be not stored in the object and just passed as argument to __read_dumps() but wouldn't be able to save it with save_trajectory() afterwards.
            self.vector_types = vector_types

            self._atoms = self.__read_dumps()

            self._vectors = self.__compute_vectors()
            self._vectors['op'] = cs.global_onsager(self._vectors)

        self._print("\tTrajectory done!\n")

    def __read_dumps(self) -> xr.Dataset:
        """Reads the LAMMPS trajectory dump files and turns them in a dataset array.

        Returns:
            xr.Dataset: Trajectory dataset. It will have every properties found in the dump files in function of all particle ids and timesteps.
        """
        comp = ['x', 'y', 'z']
        nb_atoms = None
        step = None
        bounds = []
        bounds_array = []
        timestep = []
        data = []

        for filename in self.file_list:
            file = iol.read_file(filename)

        coordinates = self.__get_variable_names(file)

        for i, line in enumerate(file):
            line = line.strip()
            if line == "ITEM: TIMESTEP":
                step = int(file[i+1])
            elif line == "ITEM: NUMBER OF ATOMS":
                nb_atoms = int(file[i+1])
            elif line == "ITEM: BOX BOUNDS pp pp pp":
                for j in range(len(comp)):
                    bounds.append([ float(val) for val in file[i+j+1].strip().split() ])
                bounds_array.append([ j[1] - j[0] for j in bounds ])
            elif re.search("ITEM: ATOMS", str(line)):
                # it's easier to convert numeric strings in dataframe than to a dataset (from my knowledge). So here the text for 1 timestep is sent to a dataframe, which we make a list of. Then later we transform those dataframes to a dataset (conversion is builtin the dataset object)
                data.append(self.__lines_to_df(file[i+1:i+nb_atoms+1], coordinates))
                timestep.append(step)

        data = self.__dfs_to_ds(data, timestep)
        data['bounds'] = xr.DataArray(bounds_array, coords = [data.ts, comp], dims = ['ts', 'comp'])

        return data

    def __get_variable_names(self, file) -> list:
        """Retrieves the names of the trajectory properties in the dump file. They need to be consistent accross dump files.

        Args:
            file (list of str): Content of the dump file.

        Returns:
            list of str: names of the trajectory properties.
        TODO: Naming for those "properties" isnt consistent in code/documentation
        """
        counter = 0
        # scans for the line with the properties of the atoms
        while not re.search("ITEM: ATOMS", str(file[counter])):
            counter += 1
        return file[counter].strip().split() [2:]

    def __lines_to_df(self, lines, column_names) -> pd.DataFrame:
        """Converts a list of strings to a dataframe.

        Args:
            lines (list of str): Lines in the input files corresponding to 1 timestep.
            column_names (list of str): Names of the particle properties of the trajectory.

        Returns:
            pd.DataFrame: Dataframe containing the trajectory properties for all particles of 1 timestep.
        """
        data = []
        type_id = None

        # Checks the position of the property: type.
        for i, name in enumerate(column_names):
            if name == 'type':
                type_id = i

        if not self.exclude:
            for i in lines:
                data.append(i.strip().split())
        else:
            for i in lines:
                line = i.strip().split()
                if self.exclude:
                    for j in self.exclude:
                        if int(line[type_id]) != int(j):
                            data.append(line)

        df = pd.DataFrame(data, columns = column_names)
        df = df.apply(pd.to_numeric)
        df = df.set_index(['id'])
        df = df.sort_index()

        return df

    def __dfs_to_ds(self, dfs, timesteps) -> xr.Dataset:
        """Converts a list of dataframes to a dataset

        Args:
            dfs (list of dataframes): Dataframes of particle properties for all the timesteps.
            timesteps (list of ints): List with the timesteps of the dataframes in dfs. Both dfs and timesteps need to be in same order.

        Returns:
            xr.Dataset: Trajectory dataset. It will have every properties found in the dump files in function of all particle ids and timesteps.
        """
        ds = []
        for i, df in enumerate(dfs):
            ds.append(xr.Dataset())
            for j in df:
                ds[i][j] = df[j].to_xarray()
            ds[i].update( {'ts': ( 'ts', [timesteps[i]] )} )
        return xr.concat(ds, dim = 'ts')

    def __compute_vectors(self) -> xr.Dataset:
        """Computes the trajectory of vectors. Vectors are particles defined by vector_types. Will also compute some properties of said vectors and add them to the trajectory properties.

        Returns:
            xr.Dataset: Trajectory of vectors. Depends of timesteps, xyz coordinate and particle ids. It's properties are center of mass (cm), norm of vector (norm), angle with respect to xyz coordinates (euler angles?) (angle) and vector coordinates (coord).
        """
        data = self._atoms
        coords = ['x', 'y', 'z']
        # creates a list containing all the columns that are not positions
        droppers = []
        for i in data.data_vars:
            if i not in ['xu', 'yu', 'zu']:
                droppers.append(i)

        vectors = []
        total_ids = len(data.id)
        for i in range(total_ids):

            self._print( "\r\tComputing vectors... {}/{}".format(i+1,total_ids) )

            if data.isel(id=i).type in self.vector_types:
                atom0 = data.drop_vars(droppers).isel(id=i-1)
                atom1 = data.drop_vars(droppers).isel(id=i+1)

                v = atom1 - atom0
                norm = np.sqrt(v.xu**2 + v.yu**2 + v.zu**2)
                v /= norm
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
        self._print("\n")
        return xr.concat(vectors, dim = 'id')

    def __restore_trajectory(self) -> None:
        """Restores the trajectory object from a previously saved trajectory object (with save_trajectory method).

        Raises:
            EnvironmentError: If a file in file_list (created with path and pattern) doesn't match a restore possibility, it will raise an error. This error could be removed if user names some files the same way as save_trajectory does. Still, it would be better pratice if user moves saved trajectories to a different directory so this doesn't trigger.
        TODO: a way to make sure dataset integrity is good and a way to make sure all of the class is properly restored.
        """
        ## Could do dataset checkup to verify integrity
        for i in self.file_list:
            if "atoms.nc" in i:
                self._atoms = iol.read_dataset(i)
            elif "vectors.nc" in i:
                self._vectors = iol.read_dataset(i)
            elif "t-options.dict" in i:
                options = iol.read_dict(i)
                self.exclude = options["exclude"]
                self.vector_types = options["vector_types"]
                self.restore = True
                self.file_list = options["file_list"] ## already exists ?
            else:
                raise EnvironmentError("Restore not implemented for this file")

    def _print(self, text): ## __print()?
        """A method to print if updates is enabled (True) and not print if not enabled (False). Will be used a lot by child classes too.

        Args:
            text (str): Text to print.
        """
        if self.updates:
            print(text, end="\r")

    def save_trajectory(self, name, path="save/") -> None:
        """Saves trajectory object in a netcdf file format. Will create 3 files: _atoms trajectory dataset, _vectors trajectory dataset and _t-options dictionary of the object options.

        Args:
            name (str): Name used to identify the saved files. It needs to be str convertible.
            path (str, optional): Path where the saved files will be written. Defaults to "save/".
        """
        options = {"exclude":self.exclude,
                   "vector_types":self.vector_types,
                   "file_list":self.file_list}
        if iol.is_valid_name(name):
            iol.save_xarray(self._atoms, path, name+"_atoms")
            iol.save_xarray(self._vectors, path, name+"_vectors")
            iol.save_dict(options, path, name+"_t-options")

    def get_atoms_ds(self) -> xr.Dataset:
        """Getter for the _atoms dataset.

        Returns:
            xr.Dataset: Trajectory of all the atoms.
        """
        return self._atoms

    def get_vectors_ds(self) -> xr.Dataset:
        """Getter for the _vectors dataset.

        Returns:
            xr.Dataset: Trajectory of all the vectors.
        """
        return self._vectors