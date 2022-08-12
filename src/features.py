import numpy as np
import xarray as xr
from pandas.plotting import scatter_matrix
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import fnmatch
from skbio.stats.ordination import pcoa ## NEED TO IMPLEMENT PROPERLY
from sklearn.decomposition import PCA

from voxels import local
import compute_structure as cs

class features( local ):

    ## missing much communication to user

    def __init__(self, path = "./", pattern = "ellipsoid.*", exclude = None, vector_patterns = [[2, 3, 2]], restore_trajectory = False, updates = True,
                 neighbors = 10, restore_locals = False,
                 vector_descriptors = ["cm", "angle"], voxel_descriptors = ["cm", "angle"], distance_descriptor = True, director = False, normalization = "max",
                 ) -> None:
        """Computes features as a kernel matrix from the voxels dataset and vectors dataset from a LAMMPS trajectory. Pairwise distance is used to create the kernel matrix and features and symmetrized and normalized.

        Args:
        ## beginning of copied from voxels.py
            path (str, optional): Path to input files. Input files can be either LAMMPS trajectory dumps or previously saved trajectory using the save_trajectory method. Defaults to "./".
            pattern (str, optional): Pattern matching input files. Input files can be either LAMMPS trajectory dumps or previously saved trajectory using the save_trajectory method. Defaults to "ellipsoid.*".
            exclude (list of int or None/False, optional): Types to exclude from the trajectory. Defaults to None.
            vector_patterns (nested list of int, optional): Patterns of types for defining vectors. Each element of the mother list is a vector pattern. Defaults to [[2, 3, 2]].
            restore_trajectory (bool, optional): If True, the input files will be read as a restore of the trajectory class. Those input files need to have been created by the save_trajectory method. Defaults to False.
            updates (bool, optional): If True, prints will update the user of current progress. Defaults to True.
            neighbors (int, optional): Number of neighbors to form voxels. A particle counts in it's own voxel, so you will see voxels of size +1 what you specify here. Defaults to 10.
            restore_locals (bool, optional): _If True, the input files will be read as a restore of the local class. Those input files need to have been created by the save_locals method. Defaults to False.
        ## end of copied from voxels.py
            vector_descriptors (list of str, optional): List of variables to take in from the trajectory._vectors dataset. Defaults to ["cm", "angle"].
            voxel_descriptors (list of str, optional): List of variables to take in from the local._voxels dataset. Defaults to ["cm", "angle"].
            distance_descriptor (bool, optional): Whether or not to take in the distance matrix from local._distance_matrix. Defaults to True.
            director (list of str or 'auto' or False, optional): Whether or not only one xyz component should be taken into account. If list of str, should be like ['x', 'y']. If 'auto', director will be choosen using the longuest dimension. TODO: more doc on that. If false, all xyz components are used.Defaults to False.
            normalization (str, optional): Normalization technique. Choices are: min-max, max, zscores_abs and zscores_std. See methods for more details. Defaults to "max".
        TODO: save and restore features
        """
        super().__init__(path, pattern, exclude, vector_patterns, restore_trajectory, updates, neighbors, restore_locals)

        self._features = self.__generate_raw_features(vector_descriptors, voxel_descriptors, distance_descriptor, director)
        self.__apply_symmetries()
        self.__compute_distances()

        self.__normalize(normalization)
        ## public self.add_features

    def __generate_raw_features(self, vector, voxel, distance, director) -> xr.Dataset:
        """Generates the raw features dataset by specifying which properties from which dataset to use.

        Args:
            vector (list of str): List of variables to take in from the trajectory._vectors dataset.
            voxel (list of str): List of variables to take in from the local._voxels dataset.
            distance (bool): Whether or not to take in the distance matrix from local._distance_matrix.
            director (str or bool): Whether or not only one xyz component should be taken into account. If false, all xyz components are used.
        Returns:
            xr.Dataset: Raw features dataset. xyz coordinates are removed and if multiple are take into account, _x (or _y, _z) will be added to the feature's name.
        """
        features = xr.Dataset()

        if not director: # director is False or None
            director = ['x', 'y', 'z']
        elif director == 'auto':
            director = self.__auto_director(0.5) # could be argument
        elif isinstance(director, list): # director is specified by user and already formated, we do nothing
            for i in director:
                if i not in ['x', 'y', 'z']:
                    raise ValueError('User specified director but it isnt either x, y or z:' + i)
        else:
            raise ValueError('Specified director is not valid :' + director)

        data = [self._vectors, self._voxels, xr.Dataset({'distance':self._distance_matrix})]
        to_import = [ vector, voxel, ['distance'] if distance else False ]

        for i, current_ds in enumerate(data):
            if to_import[i]:
                for name in to_import[i]:
                    if not "comp" in current_ds[name].coords:
                        features[name] = current_ds[name]
                    else:
                        for comp in director:
                            features[name + '_' + comp] = current_ds[name].sel(comp = comp)

        return features.drop_vars('comp')

    def __auto_director(self, threshold):
        bounds = self._atoms.bounds.mean(axis = self._atoms.bounds.get_axis_num('ts'))
        longuest = max(bounds)

        ratios = bounds / longuest
        director = []
        for i in ratios:
            if i > threshold:
                director.append(str(i.comp.values))
        return director

    def __apply_symmetries(self) -> None:
        """Applies symmetries to the features. These symmetries are extremely system dependant. Check the methods with symmetry in the name for details.
        """
        for name in list(self._features.keys()):
            if "angle" in name:
                self._features[name] = self.__angle_symmetry(self._features[name]) # check if ok with different dimensionality
            if "distance" in name:
                self.__check_distance_symmetry(self._features[name])

    def __check_distance_symmetry(self, data):
        ## TODO
        # check if sparse and no distance is larger than bounds
        pass

    def __angle_symmetry(self, data) -> xr.DataArray:
        """Applies the symmetry for a reversible particle's vector angle. The angle needs to not be on a degree (or radian) scale. Since it's reversible we then don't consider difference between negative and positive values.

        Args:
            data (xr.dataArray): Raw angle data.

        Returns:
            xr.DataArray: Symmetric angle data. Applying the symmetry doesn't modify the coords of the dataArray.
        """
        return abs(np.cos(data)) # could be squared instead of abs

    def __compute_distances(self, **pdist_kwargs) -> None:
        """Computes pairwise distances for each feature on each timestep separately. Overwrites _features.
        TODO different distances, check if kwargs works properly
        """
        for name in list(self._features.keys()):
            if not 'distance' in name:

                distance_array = []
                total_ts = len(self.timesteps)
                for cnt, i in enumerate(self.timesteps):

                    self._print( "\r\tComputing distance on feature {} for timestep {}/{}".format(name, cnt+1, total_ts) )
                    data = [ [i] for i in self._features[name].sel(ts = i).values ]
                    distance_array.append(squareform(pdist( data )))

                self._print("\n")
                self._features[name] = xr.DataArray(distance_array, coords = [self.timesteps, self.ids, self.ids], dims = ['ts', 'id', 'id_n'])

    def __normalize(self, normalization) -> None:
        """Applies normalization on the features. Overwrites _features.

        Args:
            normalization (str): Normalization technique. Choices are: min-max, max and standardize. See methods for more details.

        Raises:
            ValueError: Catches the error where normalization technique isn't recognized.
        """
        # need to check if standardize affects 0s
        # if so need to check if squareform before and after normalization
        ## check if normalization techniques give right range
        ## might be useful to go back to data parameter instead of _features since we could use __normalize from outside script (same with symmetries and distances)

        if normalization:
            if normalization == "min-max":
                norm_func = self.__normalize_min_max
            elif normalization == "max":
                norm_func = self.__normalize_max
            elif normalization == "zscores_abs":
                norm_func = self.__zscores_abs_dev
            elif normalization == "zscores_std":
                norm_func = self.__zscores_std_dev
            else:
                raise ValueError("Specified normalization not implemented yet:" + normalization)

            for name in self._features:
                data = []
                for ts in self.timesteps:
                    if self._features[name].sel(ts = ts).values.any() == 0:
                        data.append(self._features[name].sel(ts = ts))
                        self._print("Data for {} at timestep {} is all 0. Skipping normalization.\n".format(name, ts))
                    else:
                        data.append( norm_func( self._features[name].sel(ts = ts) ) )
                self._features[name] = xr.DataArray(data, coords = [self.timesteps, self.ids, self.ids], dims = ['ts', 'id', 'id_n'])

    def __rescale_positive(self, data):
        if (data < 0).any():
            min_ = abs( np.min(data) )
            data += min_
        return data

    def __normalize_max(self, data) -> xr.DataArray:
        """Normalization technique. Divide the whole feature by it's maximum value.

        Formula:
            x_i = x_i / max(x)

        Args:
            data (xr.DataArray): Unnormalized data.

        Returns:
            xr.DataArray: Normalized data. Values range from 0 to 1.
        """
        self.__check_dims(data, n = 2) ## idk if necessary
        return data / np.max(data)

    def __normalize_min_max(self, data) -> xr.DataArray:
        """Normalization technique. Substracts the minimum value and divide the whole feature by the biggest difference.

        Formula:
            x_i = ( x_i - min(x) ) / ( max(x) - min(x) )

        Args:
            data (xr.DataArray): Unnormalized data.

        Returns:
            xr.DataArray: Normalized data. Values range from 0 to 1.
        """
        self.__check_dims(data, n = 2)
        return ( data - np.min(data) ) / ( np.max(data) - np.min(data) )

    def __zscores_abs_dev(self, data) -> xr.DataArray: ## not working for some reason
        """Normalization technique. Subsracts the average value and and divides by the average difference.

        Formula:
            z_if = (x_if - avg(x_f))/s_f
            s_f = 1/n { sum_i^n (x_if - m_f) }
            TODO: All formulas should be latex compatible and have refs

        Args:
            data (xr.DataArray): Unnormalized data.

        Returns:
            xr.DataArray: Normalized data. Values range from -4 to 4.
        """
        self.__check_dims(data, n = 2)
        data = squareform(data)
        data = ( data - data.mean() ) / np.mean( abs(data-data.mean()) )
        return squareform(data)

    def __zscores_std_dev(self, data) -> xr.DataArray: ## not working for some reason
        """Normalization technique. Subsracts the average value and and divides by the average difference.

        Formula:
            z_if = (x_if - avg(x_f))/s_f
            s_f = 1/n { sum_i^n (x_if - m_f) }
            TODO: All formulas should be latex compatible and have refs

        Args:
            data (xr.DataArray): Unnormalized data.

        Returns:
            xr.DataArray: Normalized data. Values range from -4 to 4.
        """
        self.__check_dims(data, n = 2)
        data = squareform(data)
        data = ( data - data.mean() ) / data.std() ##?
        return squareform(data)

    def __check_dims(self, data, n) -> None:
        """Checks if data is has the right dimmensionality.

        Args:
            data (xr.DataArray): Data.
            n (int): Number of dimensions needed.

        Raises:
            ValueError: Catches if data doesn't have the right amount of dimensions.
            ValueError: Catches if data isn't a DataArray.
        """
        if isinstance(data, xr.core.dataarray.DataArray):
            if data.ndim == n:
                pass
            else:
                raise ValueError("shit isnt good in the right dimensionality (should write proper error later)")
        else:
            raise ValueError("havent written this part properly yet")

    def set_weights(self, features, method):
        """_summary_

        Args:
            features (_type_): _description_
            method (_type_): _description_

        Returns:
            _type_: _description_
        """
        weights = {}
        if not method:
            for i in features:
                weights[i] = 1
        elif method == 'auto':
            bounds = self._atoms.bounds.mean(axis = self._atoms.bounds.get_axis_num('ts'))
            bounds = bounds / max(bounds)

            for i in features:
                dir = i.split('_')[-1]
                if dir in self.comps:
                    weights[i] = float(bounds.sel(comp=dir).values)
        elif isinstance(method, dict):
            features_name = [i for i in features]
            for i in method:
                if '*' in i:
                    for j in fnmatch.filter(features_name, i):
                        weights[j] = method[i]
        return weights

    def combine_weights(self, w1, w2, method='product'):
        """_summary_

        Args:
            w1 (_type_): _description_
            w2 (_type_): _description_
            method (str, optional): _description_. Defaults to 'product'.

        Returns:
            _type_: _description_
        """
        weights = {}
        if method == 'sum':
            for i in w1:
                weights[i] = w1[i] + w2[i]
        elif method == 'product':
            for i in w1:
                weights[i] = w1[i] * w2[i]
        elif method == 'euclidian':
            for i in w1:
                weights[i] = np.sqrt(w1[i]**2 + w2[i]**2)
        return weights

    def combine_features(self, features, weights, method = "sum") -> xr.Dataset:
        """Combines the different kernel matrices of all features into 1 kernel matrix. Different combination methods are implemented

        Args:
            method (str, optional): Method of combination. Choices are sum, product and euclidean. Defaults to "sum".

        Returns:
            xr.Dataset: Combined features. Should have no coordinates except timestep, id and id_n.
        ## might be dataArray
        TODO: check if weights name match features
        """

        if method == "sum":
            for cnt, name in enumerate(features):
                if cnt == 0:
                    data = weights[name] * features[name]
                else:
                    data += weights[name] * features[name]
        elif method == "product": # not sure how to implement weights here.
            for cnt, name in enumerate(features):
                if cnt == 0:
                    data = features[name]
                else:
                    data *= features[name]
        elif method == "euclidean":
            for cnt, name in enumerate(features):
                if cnt == 0:
                    data = weights[name] * features[name] ** 2
                else:
                    data += weights[name] * features[name] ** 2
            data = np.sqrt(data)
        return data

    def pairwise_plot(self, data, ts=0):
        plt.imshow(data.isel(ts=ts), interpolation='nearest')
        plt.show()
    ## difference between pairwise_plot and scatter_matrix?

    def scatter_matrix(self, ts=0, id=0):
        data = self._features.isel(ts = ts, id = id).drop_vars(['ts', 'id'])

        scatter_matrix(data.to_dataframe())
        plt.show()
        return data.to_dataframe()

    def decomposition(self, features, method='pca', **kwargs):
        """http://scikit-bio.org/docs/0.5.4/generated/generated/skbio.stats.ordination.pcoa.html

        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA

        Args:
            features (_type_): _description_
            method (str, optional): _description_. Defaults to 'pca'.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        data = []
        if method == 'pca':
            for ts in self.timesteps:
                data.append( PCA(**kwargs).fit(features.sel(ts=ts)).transform(features.sel(ts=ts)) )
        elif method == 'pcoa':
            for ts in self.timesteps:
                data.append( pcoa(features.sel(ts), **kwargs).samples )
        else:
            raise ValueError("smtn")

        # return xr.DataArray(data)
        return data

        # pcoa_data = pcoa(final_features.isel(ts=0))
        # plt.scatter(pcoa_data.samples['PC1'], pcoa_data.samples['PC2'])
        # plt.show()
        # return pcoa_data

    def filter_features(self, onsager_thresh = 0.3, rdf_dist_thresh = 2, rdf_value_tresh = 0.1):
        ## take into account timesteps
        mean_rdf = self._rdf.mean(axis = self._rdf.get_axis_num('id'))
        peak_mean_rdf = mean_rdf.idxmax('distance')
        max_mean_rdf = mean_rdf.max('distance')

        min_ = peak_mean_rdf - rdf_dist_thresh
        max_ = peak_mean_rdf + rdf_dist_thresh

        is_rdf_peak = np.amax(self._rdf, axis = self._rdf.get_axis_num('distance')) >= max_mean_rdf * rdf_value_tresh
        is_ordered_angle = self._voxels.onsager_1 >= onsager_thresh

        is_greater_or_eq_rdf_peak = self._rdf.max('distance') >= min_
        is_lower_rdf_peak = self._rdf.max('distance') < max_

        is_rdf = np.logical_and(is_rdf_peak, is_greater_or_eq_rdf_peak)
        is_rdf = np.logical_and(is_rdf, is_lower_rdf_peak)
        is_good = np.logical_or(is_ordered_angle, is_rdf)

        filtered = self._features.where(is_good, drop = True)

        filtered_ids = []
        for i in filtered.id_n.values:
            if i in filtered.id.values:
                filtered_ids.append(True)
            else:
                filtered_ids.append(False)
        return filtered.where(filtered_ids).dropna('id_n')

    def to_dissimilarities(self, ds) -> xr.Dataset:
        return ds / ds.max(['id', 'id_n'])

    def to_similarities(self, ds) -> xr.Dataset:
        return 1 - self.to_dissimilarities(ds)

    def get_features_ds(self) -> xr.Dataset:
        """Getter for the _features dataset.

        Returns:
            xr.Dataset: Trajectory of features.
        """
        return self._features