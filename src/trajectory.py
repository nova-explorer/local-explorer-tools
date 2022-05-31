import xarray as xr
import numpy as np
import re
import pandas as pd

import io_local as iol
import compute_structure as cs

class trajectory( object ): ## need object here?

    def __init__(self, 
                 path = "./", pattern = "ellipsoid.*", exclude = None, vector_patterns = [[2, 3, 2]], restore = False, updates = True
                 ) -> None:
        """LAMMPS trajectory converted to an xarray dataset. Additionally ellipsoidal particles are defined as vectors and stored in another dataset for further computation.

        Args:
            path (str, optional): Path to input files. Input files can be either LAMMPS trajectory dumps or previously saved trajectory using the save_trajectory method. Defaults to "./".
            pattern (str, optional): Pattern matching input files. Input files can be either LAMMPS trajectory dumps or previously saved trajectory using the save_trajectory method. Defaults to "ellipsoid.*".
            exclude (list of int or None/False, optional): Types to exclude from the trajectory. Defaults to None.
            vector_patterns (nested list of int, optional): Patterns of types for defining vectors. Each element of the mother list is a vector pattern. Defaults to [[2, 3, 2]].
            restore (bool, optional): If True, the input files will be read as a restore of the trajectory class. Those input files need to have been created by the save_trajectory method. Defaults to False.
            updates (bool, optional): If True, prints will update the user of current progress. Defaults to True.
        TODO: -Attributes
              -Check if timesteps repeat
              -Check if no vectors are found
              -IO for save
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
            self.vector_patterns = vector_patterns

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
                bounds = []
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
                    if int(line[type_id]) not in self.exclude:
                    # for j in self.exclude:
                    #     if int(line[type_id]) != int(j):
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
        """Computes the trajectory of vectors. Vectors are sequence of particles defined by vector_patterns. Will also compute some properties of said vectors and add them to the trajectory properties.

        Returns:
            xr.Dataset: Trajectory of vectors. Depends of timesteps, xyz coordinate and particle ids. It's properties are center of mass (cm), norm of vector (norm), angle with respect to xyz coordinates (euler angles?) (angle) and vector coordinates (coord). Center of mass is computed with the 2 extremum particles. Id is that of the center particle.
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

            self._print( "\r\tComputing vectors... {:.0%}".format((i+1)/total_ids) ) ## change to percent
            for pattern in self.vector_patterns:

                if i <= total_ids-len(pattern) and self.__is_fit_pattern( data.isel( id = range(i, i+len(pattern)) ).type, pattern ):

                    atom0 = data.drop_vars(droppers).isel(id = i)
                    atom1 = data.drop_vars(droppers).isel(id = i+len(pattern)-1)

                    v = atom1 - atom0
                    norm = np.sqrt(v.xu**2 + v.yu**2 + v.zu**2)
                    v /= norm
                    v['coord'] = xr.DataArray( np.transpose([v.xu, v.yu, v.zu]), coords = [v.ts, coords], dims = ['ts', 'comp'] )
                    v['norm'] = norm

                    alpha = np.arccos(v.xu)
                    beta = np.arccos(v.yu)
                    gamma = np.arccos(v.zu)
                    v['angle'] = xr.DataArray( np.transpose([alpha, beta, gamma]), coords = [v.ts, coords], dims = ['ts', 'comp'] )

                    mid = (atom1 + atom0) / 2 ## should take into account all atoms in pattern and their mass
                    x_cm = mid.xu
                    y_cm = mid.yu
                    z_cm = mid.zu
                    v['cm'] = xr.DataArray( np.transpose([x_cm, y_cm, z_cm]), coords = [v.ts, coords], dims = ['ts', 'comp'] )

                    v = v.drop_vars(['xu', 'yu', 'zu'])
                    v.update( {'id': ( 'id', [data.id[ i+int(len(pattern)/2) ]] )} )
                    vectors.append(v)
        self._print("\n")
        return xr.concat(vectors, dim = 'id')

    def __is_fit_pattern(self, types, pattern) -> bool:
        """Checks if a sequence matches the pattern. If the particle sequence has exactly the same type sequence as the pattern, True will be returned, False otherwise.

        Args:
            types (xr.dataset): Dataset containing only the types of the selected particles.
            ## may be dataArray
            pattern (list of int): Type pattern defined as a vector.

        Returns:
            bool: True if particle sequence fits pattern, False otherwise.
        """
        flag = True
        for i in range(len(pattern)):
            self.__check_type_consistency(types.isel(id = i))
            if types.isel(id = i, ts = 0) != pattern[i]:
                flag = False
        return flag

    def __check_type_consistency(self, types):
        type_0 = types.isel(ts = 0).values
        for i in types.ts:
            if type_0 != types.sel(ts = i).values:
                raise IndexError("type is not consistent across simulation for id {id}. ts 0: {type0}, ts {ts1}: {type1}.".format(id = types.id,
                                                                                                                                  type0 = type_0,
                                                                                                                                  ts1 = i,
                                                                                                                                  type1 = types.sel(ts = i).values,
                                                                                                                                  )
                                 )

    def __restore_trajectory(self) -> None:
        """Restores the trajectory object from a previously saved trajectory object (with save_trajectory method).

        TODO: a way to make sure dataset integrity is good and a way to make sure all of the class is properly restored.
        """
        ## Could do dataset checkup to verify integrity
        files_imported = {'atoms':None, 'vectors':None, 'options':None}

        for i in self.file_list:
            if '.tnc' in i or '.tdc' in i:
                if '.atoms.' in i:
                    if not files_imported['atoms']:
                        self._atoms = iol.read_dataset(i)
                        files_imported['atoms'] = i
                    else:
                        raise EnvironmentError("Already imported the atoms dataset with " + files_imported['atoms'] + ", current file : " + i)

                elif '.vectors.' in i:
                    if not files_imported['vectors']:
                        self._vectors = iol.read_dataset(i)
                        files_imported['vectors'] = i
                    else:
                        raise EnvironmentError("Already imported the vectors dataset with " + files_imported['vectors'] + ", current file : " + i)

                elif '.options.' in i:
                    if not files_imported['options']:
                        options = iol.read_dict(i)
                        self.exclude = options["exclude"]
                        self.vector_patterns = options["vector_patterns"]
                        self.restore = True

                        files_imported['options'] = i
                    else:
                        raise EnvironmentError("Already imported the options dict with " + files_imported['options'] + ", current file : " + i)

                else:
                    raise EnvironmentError("Restore not implemented for this file : " + i)

    def _print(self, text): ## __print()?
        """A method to print if updates is enabled (True) and not print if not enabled (False). Will be used a lot by child classes too.

        Args:
            text (str): Text to print.
        """
        if self.updates:
            print(text, end = "\r")

    def save_trajectory(self, name, path = "save/") -> None:
        """Saves trajectory object in a netcdf file format. Will create 3 files: _atoms trajectory dataset, _vectors trajectory dataset and _t-options dictionary of the object options.

        Args:
            name (str): Name used to identify the saved files. It needs to be str convertible.
            path (str, optional): Path where the saved files will be written. Defaults to "save/".
        """
        options = {"exclude":self.exclude,
                   "vector_patterns":self.vector_patterns,
                   "file_list":self.file_list}
        if iol.is_valid_name(name):
            iol.save_xarray(self._atoms, path, name+".atoms.tnc")
            iol.save_xarray(self._vectors, path, name+".vectors.tnc")
            iol.save_dict(options, path, name+".options.tdc")

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