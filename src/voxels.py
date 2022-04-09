from numpy.lib.function_base import append #?
import xarray as xr
import numpy as np
from scipy.spatial.distance import pdist, squareform
from xarray.core.indexes import remove_unused_levels_categories #?

import compute_structure as cs
import io_local as iol
from trajectory import trajectory

class locals( trajectory ):

    def __init__(self,
                 path="./", pattern="ellipsoid.*", exclude=None, vector_types=None, restore_trajectory=False, updates=True,
                 neighbors=10, restore_locals=False
                 ) -> None:

        super().__init__(path, pattern, exclude, vector_types, restore_trajectory, updates)

        if restore_locals:
            self._print("\tRestoring voxels...\n")

            self.restore_voxels()
        else:
            self.neighbors = neighbors + 1 # particle belongs to its own voxel
            self.timesteps = self._vectors.ts.values
            self.ids = self._vectors.id.values
            self.nb_atoms = len(self.ids)
            self.comps = self._vectors.comp.values

            self._distance_matrix = self.__compute_distance_matrices()

            self._print("\tComputing voxels...\n")

            self._voxels = self.__compute_voxels_trajectory()
    def __compute_distance_matrices(self):

        distance_array=[]
        total_ts = len(self.timesteps)
        for cnt, i in enumerate(self.timesteps):

            self._print( "\r\tComputing distance matrix for timestep {}/{}".format(cnt+1,total_ts) )

            distance_matrix = np.empty([3, self.nb_atoms, self.nb_atoms])
            bounds = self._atoms.bounds.sel(ts=i).values
            positions = self._vectors.cm.sel(ts=i).values
            ## should check if bounds and positions have same dimensionality but in trajectory; not here.

            for j in range(len(self.comps)):
                comp_distance = squareform(pdist( positions[:,[j]] ))
                comp_distance = self.__apply_pbc(comp_distance, bounds[j])
                distance_matrix[j] = comp_distance #** 2
            # distance_matrix = np.sqrt( np.sum(distance_matrix, axis=0) )
            # this modification allows distance matrix to still have separate xyz components, thus being useful for clustering.
            distance_array.append(distance_matrix)

            self._print( "\n" )

        distance_array = xr.DataArray(distance_array, coords=[self.timesteps, self.comps, self.ids, self.ids], dims=['ts', 'comp','id', 'id_n'])
        return distance_array

    def __apply_pbc(self, positions, cutoff):
        positions = np.where( positions >= cutoff,
                         positions - cutoff,
                         positions)
        return positions

    def __compute_voxels_trajectory(self, properties = ['cm', 'angle', 'coord']):
        # Creating voxel ds using distance matrix
        voxels_array = []
        for i in self.timesteps:
            voxel = []
            for j in self.ids:
                distances = self._distance_matrix.sel(ts=i, id=j)
                distances = np.sqrt( distances.sel(comp='x')**2 + distances.sel(comp='y')**2 + distances.sel(comp='z')**2 )
                distances = distances.sortby(distances) [0:self.neighbors].id_n.values
                voxel.append(distances)
            voxels_array.append(voxel)

        data = xr.Dataset({"voxel":
            xr.DataArray(voxels_array,
                         coords = [self.timesteps, self.ids, range(self.neighbors)],
                         dims = ['ts', 'id', 'id_2']
                         )
            })

        ## cool but fits whole DA to the ordering of DA[0][0]
        # #ordering
        # distances = self._distance_matrix.sortby(self._distance_matrix[0][0])
        # #trimming to neighbors
        # distances = distances [dict(id_2=slice(None,self.neighbors))]

        # adding properties to the voxel ds
        total_props = len(properties)
        total_ts = len(self.timesteps)

        for cnt_p, var in enumerate(properties):
            data_traj = []
            for cnt_i, i in enumerate(self.timesteps):

                self._print( "\r\tAdding property {}/{} on timestep {}/{}".format(cnt_p+1, total_props,
                                                                                cnt_i+1, total_ts)
                                                                                )

                data_array = []
                for j in self.ids:
                    data_voxel = []
                    for k in data.voxel.sel(ts=i, id=j).values:
                        data_voxel.append(self._vectors[var].sel(ts=i, id=k))
                    data_array.append(data_voxel)
                data_traj.append(data_array)
            data[var] = xr.DataArray(data_traj, coords = [data.ts, data.id, data.id_2, self.comps], dims = ['ts', 'id', 'id_2', 'comp'])

        self._print("\n")

        data["bounds"] = self._atoms.bounds
        return data

            ## cool but doesnt take timesetep into account.
            ## When code is working, should test which is faster.
            # for i in self.__ids:
            #     vox_ids = data.voxels.sel(id=i).id_2.values
            #     data_var = self.__vectors[var].sel(id=vox_ids)
            #     array.append(data_var)
            # recombine array in da

    def add_local_op(self, op_type="onsager", nb_ave=1):
        op_traj = []
        data_name = op_type + "_" + str(nb_ave)

        if op_type == "onsager":
            op_func = cs.voxel_onsager
        elif op_type == "common_neigh":
            op_func = cs.voxel_common_neigh
        elif op_type == "pred_neigh":
            op_func = cs.voxel_pred_neigh
        elif op_type == "another_neigh":
            op_func = cs.voxel_another_neigh
        else:
            raise ValueError("Specified OP not implemented yet:" + op_type)

        total_ts = len(self.timesteps)
        for cnt_i, i in enumerate(self.timesteps):
            op_array = []
            for cnt_j, j in enumerate(self.ids):
                self._print( "\r\tComputing local {} on timestep {}/{} for id {}/{}".format(data_name,
                                                                                             cnt_i+1, total_ts,
                                                                                             cnt_j+1, self.nb_atoms
                                                                                            )
                            )
                op_array.append( self.__get_voxel_op(i, j, op_func, nb_ave) )
            op_traj.append(op_array)
        self._print("\n")
        self._voxels[data_name] = xr.DataArray( op_traj, coords=[self.timesteps, self.ids], dims=["ts", "id"] )

    def __get_voxel_op(self, ts_, id_, op_func, nb_ave):
        op = 0

        if nb_ave == 1:
            op += op_func( self._voxels.sel(ts=ts_, id=id_) )
        elif nb_ave > 1:
            for i in self._voxels.voxel.sel(ts=ts_, id=id_).values[1:]:
                op += self.__get_voxel_op(ts_, i, op_func, nb_ave-1) / self.neighbors
        elif nb_ave < 1:
            raise ValueError("nb_ave cannot be lower than one")
        else:
            raise ValueError("Unexpected error towards value of nb_ave:" + str(nb_ave))
        return op

    def get_distances_ds(self) -> xr.Dataset:
        return self._distance_matrix

    def get_voxel_ds(self) -> xr.Dataset:
        return self._voxels

    def save_locals(self, name, path="save/") -> None:
        options = {"neighbors":self.neighbors
                   }
        if iol.is_valid_name(name):
            iol.save_xarray(self._distance_matrix, name+"_distances")
            iol.save_xarray(self._voxels, name+"_voxels")
            iol.save_dict(options, path, name+"_l-options")

    def restore_locals(self):
        for i in self.file_list:
            if "distances.nc" in i:
                self._distance_matrix = iol.read_dataset(i)
            elif "voxels.nc" in i:
                self._voxels = iol.read_dataset(i)
            elif "l-options.dict" in i:
                options = iol.read_dict(i)
                self.neighbors = options["neighbors"]
            else:
                raise EnvironmentError("Restore not implemented for this file")