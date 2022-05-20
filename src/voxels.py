import xarray as xr
import numpy as np
from scipy.spatial.distance import pdist, squareform

import compute_structure as cs
import io_local as iol
from trajectory import trajectory

class local( trajectory ):

    def __init__(self,
                 path="./", pattern="ellipsoid.*", exclude=None, vector_patterns=[[2,3,2]], restore_trajectory=False, updates=True,
                 neighbors=10, restore_locals=False
                 ) -> None:
        """Computes voxels from neighbor list of vectors using LAMMPS trajectory. Can also compute local properties of said voxels.

        Args:
        ## beginning of copied from trajectory.py
            path (str, optional): Path to input files. Input files can be either LAMMPS trajectory dumps or previously saved trajectory using the save_trajectory method. Defaults to "./".
            pattern (str, optional): Pattern matching input files. Input files can be either LAMMPS trajectory dumps or previously saved trajectory using the save_trajectory method. Defaults to "ellipsoid.*".
            exclude (list of int or None/False, optional): Types to exclude from the trajectory. Defaults to None.
            vector_patterns (nested list of int, optional): Patterns of types for defining vectors. Each element of the mother list is a vector pattern. Defaults to [[2,3,2]].
            restore_trajectory (bool, optional): If True, the input files will be read as a restore of the trajectory class. Those input files need to have been created by the save_trajectory method. Defaults to False.
            updates (bool, optional): If True, prints will update the user of current progress. Defaults to True.
        ## end of copied from trajectory.py
            neighbors (int, optional): Number of neighbors to form voxels. A particle counts in it's own voxel, so you will see voxels of size +1 what you specify here. Defaults to 10.
            restore_locals (bool, optional): _If True, the input files will be read as a restore of the local class. Those input files need to have been created by the save_locals method. Defaults to False.
        """

        super().__init__(path, pattern, exclude, vector_patterns, restore_trajectory, updates)

        if restore_locals:
            self._print("\tRestoring voxels...\n")

            self.__restore_local()

            self.timesteps = self._vectors.ts.values
            self.ids = self._vectors.id.values
            self.nb_atoms = len(self.ids)
            self.comps = self._vectors.comp.values

        else:
            self.neighbors = neighbors + 1 # a particle belongs to its own voxel
            self.timesteps = self._vectors.ts.values
            self.ids = self._vectors.id.values
            self.nb_atoms = len(self.ids)
            self.comps = self._vectors.comp.values

            self._distance_matrix = self.__compute_distance_matrices()

            self._print("\tComputing voxels...\n")

            self._voxels = self.__compute_voxels_trajectory()

    def __compute_distance_matrices(self) -> xr.DataArray:
        """Computes, for each trajectory timestep, the euclidian pairwise distance between each vector center of mass. Periodic boundary conditions are applied.

        Returns:
            xr.DataArray: Pairwise distance dataarray. In function of timestep and xyz component.id and id_n correspond to particle i an j.
        TODO: Should make a uniform thing with features.
        """
        distance_array = []
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

    def __apply_pbc(self, distances, cutoff) -> np.ndarray:
        """Apply periodic boundary conditions on the distances. If the distance is longer than the cutoff, the actual distance corresponds to the distance - cutoff.

        Args:
            distances (numpy.ndarray): Redundant pairwise distance matrix for one xyz component.
            cutoff (float): Size of the simulation box in that xyz component.

        Returns:
            np.ndarray: Distance matrix with PBC applied.
        """
        distances = np.where( distances >= cutoff,
                         distances - cutoff,
                         distances)
        return distances

    def __compute_voxels_trajectory(self, properties = ['cm', 'angle', 'coord']) -> xr.Dataset:
        """Constructs the voxel dataset using the nearest neighbors according to the distance matrix. Also adds properties from the vectors dataset to the voxels.

        Args:
            properties (list of str, optional): Properties from the vectors dataset that will be imported to the voxels. Defaults to ['cm', 'angle', 'coord'].

        Returns:
            xr.Dataset: Voxels dataset, in function of timestep, id and xyz components. The id_2 coordinate is use to list the elements of voxel i, where voxel i corresponds to the voxel of nearest neighbors of particle i.
        """
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

        ## cool but doesnt take timesetep into account.
        ## When code is working, should test which is faster.
        # for i in self.__ids:
        #     vox_ids = data.voxels.sel(id=i).id_2.values
        #     data_var = self.__vectors[var].sel(id=vox_ids)
        #     array.append(data_var)
        # recombine array in da

        return data

    def __get_voxel_op(self, ts_, id_, op_func, nb_ave) -> float:
        """Computes the order parameter of voxel i. Passing the id in this function was made so it could be used recusively if nb_ave > 1.

        Args:
            ts_ (int): Current timestep.
            id_ (int): Current id (for identifying voxel).
            op_func (func): Function that differs with the order parameter. Check the compute_structure file for details.
            nb_ave (int): Number of averages to go over when computing the order parameter. nb_ave=1 will compute the order parameter for just the voxel. nb_ave=2 will compute the order parameter of all the voxels corresponding to the particle in voxel i and average them. nb_ave can thus increase until it matches the whole system but it also increases compute time a lot.

        Raises:
            ValueError: If nb_ave gets lower than 1 (0 or negative) or is inputed as such, this error will catch it.
            ValueError: Error to catch any weird and unexpected behavior with nb_ave. Technically, should not trigger.
            TODO: check if proper error types

        Returns:
            float: Order parameter of voxel i.
        """
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

    def __restore_local(self) -> None:
        """Restores the local object from a previously saved local object (with save_local method).

        TODO: a way to make sure dataset integrity is good and a way to make sure all of the class is properly restored.
        """
        for i in self.file_list:
            if "distances.nc" in i:
                self._distance_matrix = iol.read_dataarray(i)
            elif "voxels.nc" in i:
                self._voxels = iol.read_dataset(i)
            elif "l-options.dict" in i:
                options = iol.read_dict(i)
                self.neighbors = options["neighbors"]

    def add_local_op(self, op_type="onsager", nb_ave=1) -> None:
        """Will add a local order parameter to the voxels dataset. Local order parameters are computed over each voxel.

        Args:
            op_type (str, optional): Specifies the order parameter to be computed. Choices are: onsager, common_neigh, pred_neigh, another_neigh. Thus far, only onsager has been implemented properly. Defaults to "onsager".
            nb_ave (int, optional): Number of averages to go over when computing the order parameter. nb_ave=1 will compute the order parameter for just the voxel. nb_ave=2 will compute the order parameter of all the voxels corresponding to the particle in voxel i and average them. nb_ave can thus increase until it matches the whole system but it also increases compute time a lot. Defaults to 1.

        Raises:
            ValueError: Catches the case where order parameter isn't recognized
            TODO: proper error type
        """
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

    def save_local(self, name, path="save/") -> None:
        """Saves local object in a netcdf file format. Will create 3 files: _distances trajectory dataarray, _voxels trajectory dataset and _t-options dictionary of the object options.

        Args:
            name (str): Name used to identify the saved files. It needs to be str convertible.
            path (str, optional): Path where the saved files will be written. Defaults to "save/".
        """
        options = {"neighbors":self.neighbors,
                   }
        if iol.is_valid_name(name):
            iol.save_xarray(self._distance_matrix, path, name+"_distances")
            iol.save_xarray(self._voxels, path, name+"_voxels")
            iol.save_dict(options, path, name+"_l-options")

    def get_distances_da(self) -> xr.DataArray:
        """Getter for the _distance_matrix dataarray

        Returns:
            xr.DataArray: Redundant pairwise distance matrix of the vectors' center of mass trajectory
        """
        return self._distance_matrix

    def get_voxel_ds(self) -> xr.Dataset:
        """Getter for the _voxels dataset

        Returns:
            xr.Dataset: Trajectory of the voxels in function of timestep, id and xyz components. The id_2 coordinate is use to list the elements of voxel i, where voxel i corresponds to the voxel of nearest neighbors of particle i.
        """
        return self._voxels