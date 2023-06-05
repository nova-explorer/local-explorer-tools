from numpy import NaN
import sklearn.cluster as sk_c
import sklearn_extra.cluster as ske_c
import sklearn.metrics as sk_m
import xarray as xr
import numpy as np

from features import features
import io_local as iol
import compute_structure as cs

class cluster( features ):

    def __init__(self, path = "./", pattern = "ellipsoid.*", exclude = None, vector_patterns = [[2, 3, 2]], restore_trajectory = False, updates = True,
                 neighbors = 10, restore_locals = False,
                 vector_descriptors = ["cm", "angle"], voxel_descriptors = ["cm", "angle"], distance_descriptor = True, director = False, normalization = "max"
                 ) -> None:
        """Computes clusters and saves them in the clustraj dataset. Contains many methods related to clustering.

        Args:
        ## beginning of copied from features.py
            path (str, optional): Path to input files. Input files can be either LAMMPS trajectory dumps or previously saved trajectory using the save_trajectory method. Defaults to "./".
            pattern (str, optional): Pattern matching input files. Input files can be either LAMMPS trajectory dumps or previously saved trajectory using the save_trajectory method. Defaults to "ellipsoid.*".
            exclude (list of int or None/False, optional): Types to exclude from the trajectory. Defaults to None.
            vector_patterns (nested list of int, optional): Patterns of types for defining vectors. Each element of the mother list is a vector pattern. Defaults to [[2, 3, 2]].
            restore_trajectory (bool, optional): If True, the input files will be read as a restore of the trajectory class. Those input files need to have been created by the save_trajectory method. Defaults to False.
            updates (bool, optional): If True, prints will update the user of current progress. Defaults to True.
            neighbors (int, optional): Number of neighbors to form voxels. A particle counts in it's own voxel, so you will see voxels of size +1 what you specify here. Defaults to 10.
            restore_locals (bool, optional): If True, the input files will be read as a restore of the local class. Those input files need to have been created by the save_locals method. Defaults to False.
            vector_descriptors (list of str, optional): List of variables to take in from the trajectory._vectors dataset. Defaults to ["cm", "angle"].
            voxel_descriptors (list of str, optional): List of variables to take in from the local._voxels dataset. Defaults to ["cm", "angle"].
            distance_descriptor (bool, optional): Whether or not to take in the distance matrix from local._distance_matrix. Defaults to True.
            director (bool, optional): Whether or not only one xyz component should be taken into account. If false, all xyz components are used. Defaults to False.
            normalization (str, optional): Normalization technique. Choices are: min-max, max, zscores_abs and zscores_std. See methods for more details. Defaults to "max".
        ## end of copied from features.py
        TODO: Generalisation of n_clusters to generic clustering parameters
        """

        super().__init__(path, pattern, exclude, vector_patterns, restore_trajectory, updates, neighbors, restore_locals, vector_descriptors, voxel_descriptors, distance_descriptor, director, normalization)

        self._clustraj = self.__generate_clustered_trajectory(["xu", "yu", "zu", "c_orient[*]", "c_shape[*]", "bounds"])

    def __generate_clustered_trajectory(self, vars) -> xr.Dataset:
        """Generates the clustraj dataset from the atoms dataset.

        Args:
            vars (list of str): Lists the variables to transfer from atoms to clustraj. For variables containing [*] in their name, every instance of said names matching without the [*] will be added instead.

        Returns:
            xr.Dataset: Clustraj dataset. Coordinates are the same as atoms (generally timestep and id) and data_variables are those imported.
        """
        ds = xr.Dataset()
        data_vars = list(self._atoms.keys())
        for i in vars:
            if "[*]" in i:
                for j in [ name for name in data_vars if i[:-3] in name ]:
                    ds [j] = self._atoms[j].where(self._atoms.id == self._vectors.id)
            elif "id" not in self._atoms[i].coords:
                ds [i] = self._atoms[i]
            else:
                ds [i] = self._atoms[i].where(self._atoms.id == self._vectors.id)
        return ds

    def __compute_silhouette(self, features, labels) -> float:
        """Computes the silhoutette coefficient of a single clustering. In case there is only one cluster, silhouette coefficient is NaN instead.
        The silhouette coefficient is a measure of how different clusters are and ranges from 1 to -1.

        Args:
            features (xr.Dataset): Features used for clustering. Need to be selected for a single timestep.
            labels (np.ndarray 1D): Clustering labels. Need to be selected for a single timestep and clustering parameter (n_clusters).

        Returns:
            float: Silhouette coefficient
        """
        labels = labels.where(labels != 0, drop=True)
        if (labels[0] == labels).all() : # if there are on
            silhouette = NaN ## could be 0 or -1
        else: # if there more than 1 cluster, in which case silhouette_score raises an error
            silhouette = sk_m.silhouette_score(features, labels, metric = 'precomputed')
        self._print('\t\tSilhouette : {:.3f}\n'.format(silhouette))

        return silhouette

    def __compute_dopcev(self, data) -> float:
        """Computes the Domain Order Parameter Coefficient for External Validation (DOPCEV). DOPCEV is used for representing how well a cluster represents domain using the local and global onsager order parameter.

        Formula:
            TODO

        Args:
            data (xr.Dataset): Clustered clustraj dataset for a single timestep and clustering parameter (n_clusters). The vectors and voxels dataset will also be needed.

        Returns:
            float: DOPCEV value for the current clustering.
        """
        data = data.where(data.labels != 0, drop=True)
        dopcev = []
        for i in range(data.n_clusters.values+1)[1:]:
            self._print('\t\tDOPCEV for cluster {}\n'.format(i))
            current_cluster = data.where(data.labels == i, drop = True).id.values ##maybe not just ids?
            current_cluster_indices = [ i for i, id in enumerate(self.ids) if id in current_cluster ]

            particles_fraction = len(current_cluster) / len(self.ids)
            if particles_fraction == 0:
                dopcev.append(0)
                self._print('\t\t\t0 atoms in cluster, DOPCEV value set to 0\n')
            else:
                op_cluster = float( cs.global_onsager(self._vectors.sel(ts = data.ts).isel(id = current_cluster_indices)) )
                op_local = float( self._voxels.sel(ts = data.ts).isel(id = current_cluster_indices).onsager_1.values.mean() )## could take onsager 2

                ratio = (op_cluster/op_local - 1) ** 2
                dopcev.append( particles_fraction * ratio )
                self._print( '\t\t\tCluster OP: {:.3f}, local OP: {:.3f}, fraction: {:.3f}, DOPCEV: {:.3f}\n'.format(op_cluster,
                                                                                                                     op_local,
                                                                                                                     particles_fraction,
                                                                                                                     1-ratio)
                            )
        self._print('\t\tDOPCEV: {:.3f}\n'.format(1-sum(dopcev)))
        return 1 - sum(dopcev)

    def __compute_dopcev_1_1(self, data) -> float:
        """Computes the Domain Order Parameter Coefficient for External Validation (DOPCEV). DOPCEV is used for representing how well a cluster represents domain using the local and global onsager order parameter.

        Formula:
            TODO

        Args:
            data (xr.Dataset): Clustered clustraj dataset for a single timestep and clustering parameter (n_clusters). The vectors and voxels dataset will also be needed.

        Returns:
            float: DOPCEV value for the current clustering.
        """
        data = data.where(data.labels != 0, drop=True)
        dopcev = []
        for i in range(data.n_clusters.values+1)[1:]:
            self._print('\t\tDOPCEV for cluster {}\n'.format(i))
            current_cluster = data.where(data.labels == i, drop = True).id.values ##maybe not just ids?
            current_cluster_indices = [ i for i, id in enumerate(self.ids) if id in current_cluster ]

            particles_fraction = len(current_cluster) / len(self.ids)
            if particles_fraction == 0:
                dopcev.append(0)
                self._print('\t\t\t0 atoms in cluster, DOPCEV value set to 0\n')
            else:
                op_cluster = float( cs.global_onsager(self._vectors.sel(ts = data.ts).isel(id = current_cluster_indices)) )
                op_local = float( self._voxels.sel(ts = data.ts).isel(id = current_cluster_indices).onsager_1.values.mean() )## could take onsager 2

                ratio = abs(op_cluster - op_local) ** 1
                dopcev.append( particles_fraction * ratio )
                self._print( '\t\t\tCluster OP: {:.3f}, local OP: {:.3f}, fraction: {:.3f}, DOPCEV: {:.3f}\n'.format(op_cluster,
                                                                                                                     op_local,
                                                                                                                     particles_fraction,
                                                                                                                     1-ratio)
                            )
        self._print('\t\tDOPCEV: {:.3f}\n'.format(1-sum(dopcev)))
        return 1 - sum(dopcev)

    def __compute_dopcev_1_2(self, data) -> float:
        """Computes the Domain Order Parameter Coefficient for External Validation (DOPCEV). DOPCEV is used for representing how well a cluster represents domain using the local and global onsager order parameter.

        Formula:
            TODO

        Args:
            data (xr.Dataset): Clustered clustraj dataset for a single timestep and clustering parameter (n_clusters). The vectors and voxels dataset will also be needed.

        Returns:
            float: DOPCEV value for the current clustering.
        """
        data = data.where(data.labels != 0, drop=True)
        dopcev = []
        for i in range(data.n_clusters.values+1)[1:]:
            self._print('\t\tDOPCEV for cluster {}\n'.format(i))
            current_cluster = data.where(data.labels == i, drop = True).id.values ##maybe not just ids?
            current_cluster_indices = [ i for i, id in enumerate(self.ids) if id in current_cluster ]

            particles_fraction = len(current_cluster) / len(self.ids)
            if particles_fraction == 0:
                dopcev.append(0)
                self._print('\t\t\t0 atoms in cluster, DOPCEV value set to 0\n')
            else:
                op_cluster = float( cs.global_onsager(self._vectors.sel(ts = data.ts).isel(id = current_cluster_indices)) )
                op_local = float( self._voxels.sel(ts = data.ts).isel(id = current_cluster_indices).onsager_1.values.mean() )## could take onsager 2

                ratio = abs(op_cluster - op_local) ** 2
                self._print( '\t\t\tCluster OP: {:.3f}, local OP: {:.3f}, fraction: {:.3f}, DOPCEV: {:.3f}\n'.format(op_cluster,
                                                                                                                     op_local,
                                                                                                                     particles_fraction,
                                                                                                                     1-ratio)
                            )
        self._print('\t\tDOPCEV: {:.3f}\n'.format(1-sum(dopcev)))
        return 1 - sum(dopcev)

    def __compute_dopcev_3_1(self, data) -> float:
        """Computes the Domain Order Parameter Coefficient for External Validation (DOPCEV). DOPCEV is used for representing how well a cluster represents domain using the local and global onsager order parameter.

        Formula:
            TODO

        Args:
            data (xr.Dataset): Clustered clustraj dataset for a single timestep and clustering parameter (n_clusters). The vectors and voxels dataset will also be needed.

        Returns:
            float: DOPCEV value for the current clustering.
        """
        data = data.where(data.labels != 0, drop=True)
        dopcev = []
        for i in range(data.n_clusters.values+1)[1:]:
            self._print('\t\tDOPCEV for cluster {}\n'.format(i))
            current_cluster = data.where(data.labels == i, drop = True).id.values ##maybe not just ids?
            current_cluster_indices = [ i for i, id in enumerate(self.ids) if id in current_cluster ]

            particles_fraction = len(current_cluster) / len(self.ids)
            if particles_fraction == 0:
                dopcev.append(0)
                self._print('\t\t\t0 atoms in cluster, DOPCEV value set to 0\n')
            else:
                op_cluster = float( cs.global_onsager(self._vectors.sel(ts = data.ts).isel(id = current_cluster_indices)) )
                op_local = float( self._voxels.sel(ts = data.ts).isel(id = current_cluster_indices).onsager_1.values.mean() )## could take onsager 2

                ratio = abs((op_cluster - op_local)/(op_cluster + op_local)) ** 3
                dopcev.append( particles_fraction * ratio )
                self._print( '\t\t\tCluster OP: {:.3f}, local OP: {:.3f}, fraction: {:.3f}, DOPCEV: {:.3f}\n'.format(op_cluster,
                                                                                                                     op_local,
                                                                                                                     particles_fraction,
                                                                                                                     1-ratio)
                            )
        self._print('\t\tDOPCEV: {:.3f}\n'.format(1-sum(dopcev)))
        return 1 - sum(dopcev)

    def __compute_dopcev_3_2(self, data) -> float:
        """Computes the Domain Order Parameter Coefficient for External Validation (DOPCEV). DOPCEV is used for representing how well a cluster represents domain using the local and global onsager order parameter.

        Formula:
            TODO

        Args:
            data (xr.Dataset): Clustered clustraj dataset for a single timestep and clustering parameter (n_clusters). The vectors and voxels dataset will also be needed.

        Returns:
            float: DOPCEV value for the current clustering.
        """
        data = data.where(data.labels != 0, drop=True)
        dopcev = []
        for i in range(data.n_clusters.values+1)[1:]:
            self._print('\t\tDOPCEV for cluster {}\n'.format(i))
            current_cluster = data.where(data.labels == i, drop = True).id.values ##maybe not just ids?
            current_cluster_indices = [ i for i, id in enumerate(self.ids) if id in current_cluster ]

            particles_fraction = len(current_cluster) / len(self.ids)
            if particles_fraction == 0:
                dopcev.append(0)
                self._print('\t\t\t0 atoms in cluster, DOPCEV value set to 0\n')
            else:
                op_cluster = float( cs.global_onsager(self._vectors.sel(ts = data.ts).isel(id = current_cluster_indices)) )
                op_local = float( self._voxels.sel(ts = data.ts).isel(id = current_cluster_indices).onsager_1.values.mean() )## could take onsager 2

                ratio = abs((op_cluster - op_local)/(op_cluster + op_local)) ** 2
                dopcev.append( particles_fraction * ratio )
                self._print( '\t\t\tCluster OP: {:.3f}, local OP: {:.3f}, fraction: {:.3f}, DOPCEV: {:.3f}\n'.format(op_cluster,
                                                                                                                     op_local,
                                                                                                                     particles_fraction,
                                                                                                                     1-ratio)
                            )
        self._print('\t\tDOPCEV: {:.3f}\n'.format(1-sum(dopcev)))
        return 1 - sum(dopcev)

    def clusterize(self, features, algorithm = "kmedoids", n_clusters = [1],  **kwargs) -> xr.Dataset:
        """Performs the clustering of features.

        Args:
            features (xr.Dataset): Features dataset. Must be a precomputed kernel matrix. Using the combine_features method is recommended.
            algorithm (str, optional): Chooses between the different clustering algorithm. Choices are: KMedoids, AffinityPropagation and commonNN. Defaults to "KMedoids".
            n_clusters (int or list of int, optional): Number of clusters used. In case n_clusters is a list, each number of clusters is done. Defaults to None.

        Raises:
            ValueError: Catches the error where n_clusters is neither an int or a list.

        Returns:
            xr.Dataset: New dataset of clustraj but with labels corresponding to the clustering.
        """

        if algorithm == "kmedoids":
            if isinstance(n_clusters, int):
                n_clusters = [n_clusters]
            elif isinstance(n_clusters, list):
                pass
            else:
                raise ValueError("n_clusters bad format") ## do a proper error

            clust_alg = [ ske_c.KMedoids(metric = 'precomputed', n_clusters = i, **kwargs) for i in n_clusters ]

        elif algorithm == "agg":
            if isinstance(n_clusters, int):
                n_clusters = [n_clusters]
            elif isinstance(n_clusters, list):
                pass
            else:
                raise ValueError("n_clusters bad format") ## do a proper error

            clust_alg = [ sk_c.AgglomerativeClustering(affinity = 'precomputed', n_clusters = i, **kwargs) for i in n_clusters ]

        elif algorithm == "birch":
            if isinstance(n_clusters, int):
                n_clusters = [n_clusters]
            elif isinstance(n_clusters, list):
                pass
            else:
                raise ValueError("n_clusters bad format") ## do a proper error

            clust_alg = [ sk_c.Birch(n_clusters = i, **kwargs) for i in n_clusters ]

        else:
            raise ValueError("No selected clustering")## do a proper error

        labels_array = []
        total_ts = len(self.timesteps)
        total_param = len(clust_alg)
        for cnt_ts, ts in enumerate(self.timesteps):
            labels = []
            for i, cluster in enumerate(clust_alg):
                self._print( '\r\tClusterizing timestep {}/{} with parameter {}/{}'.format(cnt_ts+1, total_ts,
                                                                                           i+1, total_param)
                            )
                cluster.fit(features.sel(ts = ts).values.copy(order='C'))
                labels.append(cluster.labels_ + 1)
            labels_array.append(labels)
        self._print("\n")
        data = self._clustraj.copy()
        data['labels'] = xr.DataArray(labels_array, coords = [self.timesteps, n_clusters, features.id], dims = ['ts', 'n_clusters', 'id'])
        return data.fillna({'labels':0})
        # return data

    def compute_coefficients(self, data, features, dopcev_type = 1) -> xr.Dataset:
        """Method that deals with computing the clustering validation coefficients.

        Args:
            data (xr.Dataset): Clustraj dataset with clustering labels.
            features (xr.Dataset): Features used for clustering.
            dopcev_type (int, optional): Just a quick way for me to change the DOPCEV formula for testing. Read the different ones or stick to default. Defaults to 1.

        Raises:
            ValueError: Catches the error where normalization technique isn't recognized.

        Returns:
            xr.Dataset: Clustraj dataset with added validation coefficients for each timestep and clustering parameters (n_clusters).
        """

        if dopcev_type == 1:
            dopcev_func = self.__compute_dopcev
        elif dopcev_type == "1_1":
            dopcev_func = self.__compute_dopcev_1_1
        elif dopcev_type == "1_2":
            dopcev_func = self.__compute_dopcev_1_2
        elif dopcev_type == "3_1":
            dopcev_func = self.__compute_dopcev_3_1
        elif dopcev_type == "3_2":
            dopcev_func = self.__compute_dopcev_3_2
        else:
            raise ValueError("Specified DOPCEV type not implemented: " + str(dopcev_type))

        silhouettes_arr = []
        dopcevs_arr = []
        for i in data.ts.values:
            silhouettes = []
            dopcevs = []

            for j in data.n_clusters.values:
                self._print("\tComputing coefficients on timestep {} for {} clusters\n".format(i, j))
                silhouettes.append(self.__compute_silhouette( features.sel(ts = i), data.labels.sel(ts = i, n_clusters = j) ))
                dopcevs.append(dopcev_func( data.sel(ts = i, n_clusters = j) ))

            silhouettes_arr.append(silhouettes)
            dopcevs_arr.append(dopcevs)

        data['silhouette'] = xr.DataArray(silhouettes_arr, coords = [self.timesteps, data.n_clusters], dims = ['ts', 'n_clusters'])
        data['dopcev'] = xr.DataArray(dopcevs_arr, coords = [self.timesteps, data.n_clusters], dims = ['ts', 'n_clusters'])

        return data

    def find_best_cluster(self, data):
        droppers = []
        for i in data.data_vars:
            if i not in ['dopcev', 'silhouette']:
                droppers.append(i)
        data = data.drop_vars(droppers)

        best_array = []
        for ts in data.ts:
            best_val = data.sel(ts=ts).isel(n_clusters=0).dopcev.values
            best_clust = data.n_clusters.values[0] # in case n_cluster start with 2s
            for i in data.n_clusters.values[1:]:
                current = ( data.sel(ts=ts, n_clusters=i).dopcev + data.sel(ts=ts, n_clusters=i).silhouette ) / 2
                if current > best_val:
                    best_val = current
                    best_clust = i
            best_array.append(best_clust)
        return best_array

    def find_better_cluster(self, data):
        droppers = []
        for i in data.data_vars:
            if i not in ['dopcev', 'silhouette']:
                droppers.append(i)
        data = data.drop_vars(droppers)

        best_array = []
        for ts in data.ts:
            dopcev = data.sel(ts=ts).dopcev.values
            silhouette = data.sel(ts=ts).silhouette.values
            silhouette[0] = max(silhouette[1:])

            coefficients = dopcev/max(dopcev) + silhouette/max(silhouette)
            best_val = 0
            best_clust = 0
            for i in range(len(data.n_clusters.values)):
                if coefficients[i] > best_val:
                    best_val = coefficients[i]
                    best_clust = data.n_clusters.values[i]
            best_array.append(best_clust)
        return best_array

    def export_to_ovito(self, data, name, path) -> None:
        """Allows exporting a clustered clustraj dataset to LAMMPS dump format. A trajectory dump file will be created for each timestep and clustering parameter (n_clusters).

        Args:
            data (xr.Dataset): _description_
            name (str): Base name with which the dump files will be saved as.
            path (str): Path where the dump files will be saved.
        """
        total_ts = len(self.timesteps)
        total_param = len(data.n_clusters.values)

        for cnt_ts, ts in enumerate(data.ts.values): ## add for cluster param scan instead of n_clusters specifically
            for cnt_i, i in enumerate(data.n_clusters.values):
                self._print("\r\tExporting frame on timestep {}/{} for parameter {}/{}\n".format(cnt_ts+1, total_ts,
                                                                                                 cnt_i+1, total_param)
                            )
                name_ij = "cluster." + name + "_" + str(ts) + "_" + str(i) + ".dump"
                iol.frame_to_dump(data.sel(n_clusters = i, ts = ts), name_ij, path)

    def coefficients_to_csv(self, data, name, path, series_name, write_style = 'wt') -> None:
        """_summary_

        Args:
            data (_type_): _description_
            name (_type_): _description_
            path (_type_): _description_
            series_name (_type_): _description_
            write_style (str, optional): _description_. Defaults to 'wt'.
        TODO: user io
        """
        ## see if python coding convention hecks with the csv format
        with open(path+name, write_style) as f:
            # Header
            f.write(series_name + ', ')
            f.write('timestep, n_clusters')
            for i in data.n_clusters.values:
                f.write(', ' + str(i))
            f.write(', best_cluster')
            f.write('\n')

            for ts in data.ts.values:
                f.write(', ' + str(ts))

                # Silhouette
                f.write(', silhouette')
                for i in data.n_clusters:
                    f.write(', ' + str(data.silhouette.sel(ts = ts, n_clusters = i).values) )
                f.write(', ' + str(self.find_best_cluster(data)[0]))
                f.write('\n')

                # DOPCEV
                f.write(', , DOPCEV')
                for i in data.n_clusters:
                    f.write(', ' + str(data.dopcev.sel(ts = ts, n_clusters = i).values) )
                f.write(', ' + str(self.find_better_cluster(data)[0]))
                f.write('\n')

    def get_clustraj_ds(self) -> xr.Dataset:
        """Getter for the clustraj dataset.

        Returns:
            xr.Dataset: Trajectory of to be clustered particles.
        """
        return self._clustraj