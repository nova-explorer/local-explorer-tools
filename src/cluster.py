"""
Classes that process the trajectory. It creates a DataArray with the distance matrix of all monomers and a dataset of monomers voxels and voxels computed order parameters. cluster_options contains all the inputs for generating the cluster class, with the exception of trajectory.

Usage: c_opt = c.cluster_options()
       clust = c.cluster(traj, c_opt)
       clust.save_cluster(path='./save/')

Accessing data:
       clust.dist.isel(id=10, id_2=20)
       clust.voxels.onsager_1

Requires modules: xarray
                  numpy
                  scipy (only pdist and squareform from scipy.spatial.distance)

Requires scripts: compute_op
                  io_local

TODO: * save options
      * better save/restore
"""
import xarray as xr
import numpy as np
from scipy.spatial.distance import pdist, squareform
from glob import glob

import compute_op as cop
import io_local as io

class cluster():
    """Generates cluster

    Args:
        - trajectory (trajectory class): Trajectory information created by the trajectory class.
        - options: Contains the options for creating the clusters. More details available in the cluster_options class

    Attributes:
        - Arguments
        - dist (xarray.DataArray): Distance matrix of the monomers center of mass.
        - voxels (xarray.Dataset): Contains the information and data regarding voxels and voxels computed order parameters.
    """
    def __init__(self, trajectory, options):
        """Creates the class. Creates the dist and voxels Dataset. An advanced user could put voxels op here instead of the main, however for versatility it is better to keep them in the main.

        Args:
            - trajectory (trajectory class): Trajectory information created by the trajectory class.
            - options: Contains the options for creating the clusters
        """
        self.options = options
        self.traj = trajectory
        if self.options.restore:
            self.restore_cluster()
        else:
            self.dist = self.compute_distance_matrices()
            self.voxels = self.create_voxel_trajectory()

    def compute_distance_matrices(self):
        """ Computes the distance matrix for each timestep of the trajectory. The distance matrix is like so that the (i,j)th element corresponds to the distance between particle i and j. Periodic Boundary Conditions are taken into account and if distance in one direction in larger than half the length of the simulation box, this box length will be substracted from the distance.

        Returns:
            xarray.DataArray: Distance matrices. Classed by the coordinates ts(current timestep), id(atom id of the particle i) and id_2(atom id of the particle j). Units are in Angstrom or whatever center of mass units are.

        TODO: * check if bounds and positions have same dimensions.
        """
        distance_array = []
        distance_matrix = []
        for i in range(len(self.traj.atoms.ts)):
            # get data from boundaries and atom positions
            ##check if bounds and positions have same dimensions.
            bounds = self.traj.atoms.bounds.isel(ts = i).values
            positions = self.traj.vectors.cm.isel(ts = i).values

            for j in range(len(self.traj.atoms.comp)):
                dist_dim = squareform(pdist(positions [ :, [j] ]))
                # PBC implementation
                dist_dim = np.where(dist_dim >= bounds[j]/2,
                                    dist_dim - bounds[j],
                                    dist_dim)
                # will create distance_matrix if it's the first iteration of j. Allows to reset between iterations of i
                if j == 0:
                    distance_matrix = dist_dim ** 2
                else:
                    distance_matrix += dist_dim ** 2
            distance_array.append(np.sqrt(distance_matrix))
        distance_array = xr.DataArray(distance_array, coords = [self.traj.vectors.ts, self.traj.vectors.id, self.traj.vectors.id], dims = ['ts', 'id', 'id_2'])
        return distance_array

    def get_voxels(self):
        """Generates id voxels from the distance matrices. The voxel of particle i is generated from the N nearest particle to i, where N is options.nb_neigh

        Returns:
            xarray.DataArray: Contains the data of the voxels' ids (Nb(i)). For a particle i, it's voxel is an array of the id of j which corresponds to the N nearest neighbors. Classed by the coordinates ts(current timestep), id(atom id of the particle i) and id_2(atom id of the particles j).
        """
        vox_size = range(0, self.options.nb_neigh + 1) # +1 is for the particle which the voxel belongs to
        voxels_array = []
        for i in self.dist.ts:
            voxel = []
            for j in self.dist.id:
                distances = self.dist.sel(ts = i, id = j)
                distances = distances.sortby(distances)
                ids = distances[0:self.options.nb_neigh+1].id_2.values # distances[0] is the particle for id = id_2
                voxel.append(ids)
            voxels_array.append(voxel)
        voxels_array = xr.DataArray(voxels_array, coords = [self.traj.vectors.ts, self.traj.vectors.id, vox_size], dims = ['ts', 'id', 'id_2'])
        return voxels_array

    def create_voxel_trajectory(self):
        """Generates the voxels dataset. It takes the voxel DataArray from get_voxels and adds the properties cm, angle and coord from the vectors trajectory Dataset. Will also get the bounds from trajectory as they can be used later.

        Returns:
            xarray.Dataset: Contains the voxels Dataset. Now it contains only the ids and the properties imported from the trajectory datasets. Classed by ts(current timestep), id(atom id of the particle i), id_2(atom id of the particle j) and comp(x,y and z coordinates when needed).

        TODO: * Could be in 1 function with get_voxels
              * Also adding trajectory properties should be a method by itself which can me modulated with options
        """
        data = xr.Dataset({'voxel':self.get_voxels()})

        # getting trajectory properties from vectors dataset. They are in function of ts, id and comp
        vector_properties = ['cm', 'angle', 'coord']
        for var in vector_properties:
            data_traj = []
            for i in range(len(self.traj.vectors.ts)):
                data_array = []
                for j in range(len(self.traj.vectors.id)):
                    data_voxel = []
                    for k in data.voxel.isel(ts = 0, id = j).values:
                        data_voxel.append(self.traj.vectors[var].isel(ts = i).sel(id = k))
                    data_array.append(data_voxel)
                data_traj.append(data_array)
            data[var] = xr.DataArray(data_traj, coords = [data.ts, data.id, data.id_2, ['x', 'y', 'z']], dims = ['ts', 'id', 'id_2', 'comp'])

        # getting boundaries from atoms dataset. They are in function of ts and comp.
        data['bounds'] = self.traj.atoms.bounds
        return data

    def add_local_op(self, op_type='onsager', nb_ave=1):
        """Adds a local order parameter computed on voxels to the voxels Dataset. If the number of averagess is 1, the order parameter of i corresponds to the a=1 formula of this OP. If the number of averages is higher than 1, then the OP of i corresponds to the average of the OP of j with a-1. The data variable name of this OP will be it's <op_type> + <nb_ave>. For example, the default name would be 'onsager_1'. This method is just a loop on ts and id and actual computation is done in the recursive get_voxel_op.

        Args:
            - op_type (str, optional): Type of the order parameter computed. See Types or their respective functions in compute_op for more information. Defaults to 'onsager'.
            - nb_ave (int, optional): Number of averages done on the order parameter. For more details on this, see get_voxel_op. Warning: As of current version, higher number of averagings take considerably more time. Defaults to 1.

        Usage: cluster.add_local_op(op_type='onsager', nb_ave=2)
            * Accessing data:
                cluster.voxels.onsager_2 or cluster.voxels['onsager_2']

        Types:
            - Onsager (op_type='onsager'): Orientational order parameter, also known as P2. Computes how much particles j are oriented towards i. Pretty fast computing.
            - Common Neighborhood (op_type='common_neigh'): STILL IN TESTING

        TODO: * Better verbosity
        """
        op_traj = []
        data_name = op_type + '_' + str(nb_ave)
        print('currently computing', data_name, '...')
        for i in self.voxels.ts.values:
            op_array = []
            for j in self.voxels.id.values:
                op_array.append(self.get_voxel_op(i, j, op_type, nb_ave))
            op_traj.append(op_array)
        self.voxels[data_name] = xr.DataArray(op_traj, coords=[self.voxels.ts, self.voxels.id], dims=['ts', 'id'])
        print(data_name,'is done computing!')

    def get_voxel_op(self,ts_, id_, op_type, nb_ave):
        """Computes the op for a single particle at a single timestep. Function is recursive if nb_ave > 1 and nb_ave decreases by 1 each recursion until it reaches 1. All arguments are passed from add_local_op or recursion.

        Args:
            - ts_ (int): Timestep of the current computation
            - id_ (int): atom id of the current computation
            - op_type (str, optional): Type of the order parameter computed. See Types or their respective functions in compute_op for more information.
            - nb_ave (int, optional): Number of averages done on the order parameter. For more details on this, see get_voxel_op. Warning: As of current version, higher number of averagings take considerably more time.

        Types:
            - Onsager (op_type='onsager'): Orientational order parameter, also known as P2. Computes how much particles j are oriented towards i. Pretty fast computing.
            - Common Neighborhood (op_type='common_neigh'): STILL IN TESTING

        Raises:
            - ValueError: nb_ave cannot be lower than 1. If it is from user input, it will catch it here and if not from user input, it comes from a bug and should be reported.

        Returns:
            float: value of the computed order parameter
        """
        r"""
        Formula:
            $$X_i^{(N,a,...)} = \frac{1}{\tilde{n}_b} \sum_{j\in\tilde{\bold{N}}_b(i)} X_j^{(N,a-1,...)}$$
        """
        op = 0
        nb_neigh = len(self.voxels.id_2)

        if nb_ave == 1:
            if op_type == 'onsager':
                op = cop.voxel_onsager(self.voxels.sel(ts=ts_, id=id_))
            elif op_type == 'common_neigh':
                op = cop.voxel_common_neigh(self.voxels.sel(ts=ts_), self.dist.sel(ts=ts_), id_)
            elif op_type == 'pred_neigh':
                op = cop.voxel_pred_neigh(self.voxels.sel(ts=ts_), self.dist.sel(ts=ts_), id_)
            elif op_type == 'another_neigh':
                op = cop.voxel_another_neigh(self.voxels.sel(ts=ts_), self.dist.sel(ts=ts_), id_)

        elif nb_ave > 1:
            for i in self.voxels.voxel.sel(ts=ts_, id=id_).values[1:]:
                op += self.get_voxel_op(ts_=ts_, id_=i, op_type=op_type, nb_ave=nb_ave-1) / nb_neigh
        else:
            raise ValueError('nb_ave cannot be lower than 1.')
        return op

    def save_cluster(self, name, path = 'save/'):
        """Saves the cluster classs in a nc (netCDF) file. Much faster than computing OPs each time.

        Args:
            - path (str, optional): Directory where the trajectory will be saved. Defaults to 'save/'.

        TODO: * Same as save_trajectory
        """
        if io.is_valid_name(name):
            try:
                print('saving distances...')
                io.save_xarray(self.dist, path, name+'_dist')
            except:
                print('No data for distance; not saving dataArray...')

            try:
                print('saving voxels...')
                io.save_xarray(self.voxels, path, name+'_voxels')
            except:
                print('No data for voxels; not saving dataset...')

    def restore_cluster(self):
        """Restores a saved cluster from netCDF files to regenerate the cluster class.

        Args:
            include (str, optional): Which saved datasets to restore. Defaults to 'all'. Defaults to 'all'.

        TODO: * same as restore_trajectory
        """
        for i in self.options.file_list:
            if 'dist.nc' in i:
                self.dist = io.read_dataset(i)
                print('trajectory.atoms restored!')
            elif 'voxels.nc' in i:
                self.voxels = io.read_dataset(i)
                print('trajectory.vectors restored!')
            elif options.txt in i:
                print('not implemented yet')

class cluster_options():
    """Generates options for the cluster class. See Args for description of the options.

    Args:
        - nb_neigh (int, optional): Number of closest neighbors used to define the voxels. If value is 5, it means the voxel of particle i will contain i itself + the 5 closest particles to i.Defaults to 10.
        - restore (bool, optional): If True, the trajectory class will be generated from restoring the voxels dataset and the dist dataarray from netCDF files. Defaults to False.
        - path (str, optional): Directory of the previously saved netCDF files which will be used to restore the class. Defaults to "./", the current directory.

    Attributes:
        - Arguments

    TODO:
    """
    def __init__(self, nb_neigh = 10, restore = False, path = './', file_pattern='data*.nc'):
        """Creates the class. Initializes the argument attributes.

        Args:
            - nb_neigh (int, optional): Number of closest neighbors used to define the voxels. If value is 5, it means the voxel of particle i will contain i itself + the 5 closest particles to i.Defaults to 10.
            - restore (bool, optional): If True, the trajectory class will be generated from restoring the voxels dataset and the dist dataarray from netCDF files. Defaults to False.
            - path (str, optional): Directory of the previously saved netCDF files which will be used to restore the class. Defaults to "./", the current directory.
        """
        self.nb_neigh = nb_neigh
        self.restore = restore
        if restore:
            self.file_list = io.create_file_list(path, file_pattern)