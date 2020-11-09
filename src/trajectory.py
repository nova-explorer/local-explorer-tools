import xarray as xr
import numpy as np
from glob import glob
#import matplotlib.pyplot as plt
import re
import pandas as pd
import gzip as gz

import op
import io_local as io

class trajectory():
    def __init__(self, options):
        self.options = options
        if self.options.restore == True:
            self.restore_trajectory()
        else:
            self.atoms = self.read_traj()
            self.vectors = self.add_vectors()
            self.vectors = op.legendre_poly(self.vectors)
        # v_traj = self.trim_to_types(self.full_traj, self.options.monomer_types['atom']) ## needs testing
        # delimiters = self.trim_to_types(self.full_traj, self.options.monomer_types['del']) ## needs testing

    def read_traj(self):
        bounds = {'x':[], 'y':[], 'z':[]}
        bounds_list = []
        bounds_array = []
        comp = ['x', 'y', 'z']

        nb_atoms = None
        timestep = []
        data = []
        for filename in self.options.file_list:
            if re.search('.gz', filename):
                f = gz.open(filename, 'rt')
            else:
                f = open(filename, 'rt')
            file = f.readlines()
            f.close()

            cnt = 0
            while not re.search('ITEM: ATOMS', file[cnt]):
                cnt += 1
            coordinates = file[cnt].strip().split() [2:]

            for i, line in enumerate(file):

                if line.strip() == 'ITEM: TIMESTEP':
                    step = int(file[i+1])

                elif line.strip() == 'ITEM: NUMBER OF ATOMS':
                    nb_atoms = int(file[i+1])

                elif line.strip() == 'ITEM: BOX BOUNDS pp pp pp':
                    bounds['x'] = [float(val) for val in file[i+1].strip().split() ]
                    bounds['y'] = [float(val) for val in file[i+2].strip().split() ]
                    bounds['z'] = [float(val) for val in file[i+3].strip().split() ]

                elif re.search('ITEM: ATOMS', line):
                    data.append(self.lines_to_df(file[i+1:i+nb_atoms-1], coordinates))
                    timestep.append(step)

                    bounds_list = []
                    for i in bounds:
                        bounds_list.append(bounds[i][1] - bounds[i][0])
                    bounds_array.append(bounds_list)

        data = self.dfs_to_ds(data, timestep)
        data['bounds'] = xr.DataArray(bounds_array, coords=[data.ts, comp], dims=['ts', 'comp'])

        return data

    def lines_to_df(self, lines, column_names):
        data = []

        for i,name in enumerate(column_names):
            if name == 'type':
                type_id = i

        for i in lines:
            line = i.strip().split()
            if self.options.exclude_types:
                for j in self.options.exclude_types:
                    if int(line[type_id]) != int(j):
                        data.append(line)
            else:
                data.append(line)

        df = pd.DataFrame(data, columns=column_names)
        df = df.apply(pd.to_numeric)
        df = df.set_index(['id'])
        df = df.sort_index()

        return df

    def dfs_to_ds(self, dfs, step):
        ds = []
        for i, df in enumerate(dfs):
            ds.append(xr.Dataset())
            for j in df:
                ds[i][j] = df[j].to_xarray()
            ds[i].update( {'ts': ( 'ts', [step[i]] )} )
        return xr.concat(ds, dim='ts')

    def trim_to_types(self, traj, types):
        """removes all but those types
            Need to test many subsequent trims
        """
        traj = traj.where( traj.type.isin(types), drop=True )
        return traj

    def add_vectors(self):
        """append vectors
        """
        data = self.atoms
        mono_type = self.options.monomer_types
        droppers = []
        coords = ['x', 'y', 'z']
        for i in data.data_vars:
            if i not in ['xu', 'yu', 'zu']:
                droppers.append(i)

        vectors = []
        for i in range(len(data.id)):
            if int(data.isel(id=i).type[0]) == mono_type: # type doesnt change during simulation
                atom_0 = data.drop_vars(droppers).isel(id=i-1)
                atom_1 = data.drop_vars(droppers).isel(id=i+1)

                v = atom_1 - atom_0
                norm = np.sqrt(v.xu**2 + v.yu**2 + v.zu**2)
                v = v / norm
                v['coord'] = xr.DataArray( np.transpose([v.xu, v.yu, v.zu]), coords=[v.ts, coords], dims=['ts', 'comp'] )
                v['norm'] = norm

                alpha = np.arccos(v.xu)
                beta = np.arccos(v.yu)
                gamma = np.arccos(v.zu)
                v['angle'] = xr.DataArray( np.transpose([alpha, beta, gamma]), coords=[v.ts, coords], dims=['ts', 'comp'] )

                x_cm = data.isel(id=i).xu
                y_cm = data.isel(id=i).yu
                z_cm = data.isel(id=i).zu
                v['cm'] = xr.DataArray( np.transpose([x_cm, y_cm, z_cm]), coords=[v.ts, coords], dims=['ts', 'comp'] )

                v = v.drop_vars(['xu', 'yu', 'zu'])
                v.update( {'id': ( 'id', [i] )} )

                vectors.append(v)

        return xr.concat(vectors, dim='id')

    def save_trajectory(self, path='save/'):
        try:
            print('saving atoms...')
            io.save_dataset(self.atoms, path, 'atoms_io')
        except:
            print('No trajectory for atoms; not saving dataset...')

        try:
            print('saving vectors...')
            io.save_dataset(self.vectors, path, 'vectors_io')
        except:
            print('No trajectory for vectors; not saving dataset...')

        #save options

    def restore_trajectory(self, include='all'):
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
            self.atoms = io.read_dataset(path, name='atoms_io.nc')
        if DO_VECTORS:
            self.vectors = io.read_dataset(path, name='vectors_io.nc')

        #read  options

class trajectory_options():
    def __init__( self, path="./", file_pattern="ellipsoid.*", exclude_types=None, monomer_types=None, restore=False):
        self.path = path
        self.exclude_types = exclude_types
        self.monomer_types = monomer_types
        self.restore = restore
        if not self.restore:
            self.create_file_list(path, file_pattern)
        # if not monomer_types:
        #     self.monomer_types = {'atom':[6, 4], 'del':[ [5,5], [2,3] ]}

    def create_file_list(self, path, file_pattern):
        self.file_list = glob(path + file_pattern)
        print("Found",len(self.file_list),"matching",file_pattern,"in", path, "...")
        if len(self.file_list) == 0:
            raise EnvironmentError # Edit to real error
