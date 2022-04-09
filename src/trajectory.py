from typing import Pattern #?
import xarray as xr
import numpy as np
import re
import pandas as pd

import io_local as iol
import compute_structure as cs

class trajectory( object ):

    def __init__(self,
                 path="./", pattern="ellipsoid.*", exclude=None, vector_types=None, restore=False, updates=True
                 ) -> None:

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

            self.exclude = exclude
            self.vector_types = vector_types

            self._atoms = self.__read_dumps()

            self._vectors = self.__compute_vectors()
            self._vectors['op'] = cs.global_onsager_op(self._vectors)

        self._print("\tTrajectory done!\n")

    def __read_dumps(self) -> xr.Dataset:
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
                data.append(self.__lines_to_df(file[i+1:i+nb_atoms+1], coordinates))
                timestep.append(step)

        data = self.__dfs_to_ds(data, timestep)
        data['bounds'] = xr.DataArray(bounds_array, coords = [data.ts, comp], dims = ['ts', 'comp'])

        return data

    def __get_variable_names(self, file) -> list:
        counter = 0
        # scans for the line with the properties of the atoms
        while not re.search("ITEM: ATOMS", str(file[counter])):
            counter += 1
        return file[counter].strip().split() [2:]

    def __lines_to_df(self, lines, column_names) -> pd.DataFrame:
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

    def __dfs_to_ds(self, dfs, list_ids):
        ds = []
        for i, df in enumerate(dfs):
            ds.append(xr.Dataset())
            for j in df:
                ds[i][j] = df[j].to_xarray()
            ds[i].update( {'ts': ( 'ts', [list_ids[i]] )} )
        return xr.concat(ds, dim = 'ts')

    def __compute_vectors(self) -> xr.Dataset:
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

    def get_atoms_ds(self) -> xr.Dataset:
        return self._atoms
    def get_vectors_ds(self) -> xr.Dataset:
        return self._vectors

    def save_trajectory(self, name, path="save/") -> None:
        options = {"exclude":self.exclude,
                   "vector_types":self.vector_types,
                   "file_list":self.file_list}
        if iol.is_valid_name(name):
            iol.save_xarray(self._atoms, path, name+"_atoms")
            iol.save_xarray(self._vectors, path, name+"_vectors")
            iol.save_dict(options, path, name+"_t-options")

    def restore_trajectory(self) -> None:
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
                self.file_list = options["file_list"] ## already exist
            else:
                raise EnvironmentError("Restore not implemented for this file")

    def _print(self, text):
        if self.updates:
            print(text, end="\r")