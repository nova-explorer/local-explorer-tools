"""[summary]

Returns:
    [type]: [description]
"""
import xarray as xr
import numpy as np
from scipy.spatial.distance import pdist, squareform
import compute_op as cop

class cluster_map():
    """[summary]
    """
    def __init__(self, trajectory, options):
        """[summary]

        Args:
            trajectory ([type]): [description]
            options ([type]): [description]
        """
        self.options = options
        self.traj = trajectory
        self.dist = self.compute_distance_matrices()
        self.voxels = self.create_voxel_trajectory()
        self.add_local_op()

    def compute_distance_matrices(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        distance_array = []
        distance_matrix = [] # just to initialize variable
        # Iterate over timesteps
        for i in range(len(self.traj.atoms.ts)):
            # get data from boundaries and atom positions
            ##check if bounds and positions have same dimensions.
            bounds = self.traj.atoms.bounds.isel(ts = i).values
            positions = self.traj.vectors.cm.isel(ts = i).values

            for j in range(len(self.traj.atoms.comp)):
                dist_dim = squareform(pdist(positions [ :, [j] ]))
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
        """[summary]

        Returns:
            [type]: [description]
        """
        voxels_array = []
        for i in self.dist.ts:
            voxel = []
            for j in self.dist.id:
                distances = self.dist.sel(ts = i, id = j)
                distances = distances.sortby(distances)
                ids = distances[0:11].id_2.values # distances[0] is the particle for id = id_2
                voxel.append(ids)
            voxels_array.append(voxel)
        voxels_array = xr.DataArray(voxels_array, coords = [self.traj.vectors.ts, self.traj.vectors.id, range(0, 11)], dims = ['ts', 'id', 'id_2'])
        return voxels_array

    def create_voxel_trajectory(self):
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
        op_traj = []
        for i in self.voxels.ts.values:
            op_array = []
            for j in self.voxels.id.values:
                op_array.append(self.get_voxel_op(i, j, op_type, nb_ave))
            op_traj.append(op_array)
        self.voxels[op_type] = xr.DataArray(op_traj, coords=[self.voxels.ts, self.voxels.id], dims=['ts', 'id'])

    def get_voxel_op(self,ts_, id_, op_type, nb_ave):
        op = 0
        nb_neigh = len(self.voxels.id_2)

        if nb_ave == 1:
            if op_type == 'onsager':
                op = cop.voxel_onsager(self.voxels.sel(ts=ts_, id=id_))
        else: ## do an error case if nb_ave is lower than 1
            for i in self.voxels.voxel.sel(ts=ts_, id=id_).values[1:]:
                op += self.get_voxel_op(ts_, i, op_type, nb_ave-1) / nb_neigh
        return op

class cluster_options():
    """[summary]
    """
    def __init__(self, nb_neigh = 10):
        """[summary]

        Args:
            nb_neigh (int, optional): [description]. Defaults to 10.
        """
        self.nb_neigh = nb_neigh