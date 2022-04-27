import numpy as np
import xarray as xr
from pandas.plotting import scatter_matrix
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

from voxels import local

class features( local ):

    ## missing much communication to user

    def __init__(self, path="./", pattern="ellipsoid.*", exclude=None, vector_patterns=[[2,3,2]], restore_trajectory=False, updates=True,
                 neighbors=10, restore_locals=False,
                 vector_descriptors=["cm","angle"], voxel_descriptors=["cm", "angle"], distance_descriptor=True, director=False, normalization="standardize" ## proximity matrix, distance function?
                 ) -> None:
        """Computes features as a kernel matrix from the voxels dataset and vectors dataset from a LAMMPS trajectory. Pairwise distance is used to create the kernel matrix and features and symmetrized and normalized.

        Args:
        ## beginning of copied from voxels.py
            path (str, optional): Path to input files. Input files can be either LAMMPS trajectory dumps or previously saved trajectory using the save_trajectory method. Defaults to "./".
            pattern (str, optional): Pattern matching input files. Input files can be either LAMMPS trajectory dumps or previously saved trajectory using the save_trajectory method. Defaults to "ellipsoid.*".
            exclude (list of int or None/False, optional): Types to exclude from the trajectory. Defaults to None.
            vector_patterns (nested list of int, optional): Patterns of types for defining vectors. Each element of the mother list is a vector pattern. Defaults to [[2,3,2]].
            restore_trajectory (bool, optional): If True, the input files will be read as a restore of the trajectory class. Those input files need to have been created by the save_trajectory method. Defaults to False.
            updates (bool, optional): If True, prints will update the user of current progress. Defaults to True.
            neighbors (int, optional): Number of neighbors to form voxels. A particle counts in it's own voxel, so you will see voxels of size +1 what you specify here. Defaults to 10.
            restore_locals (bool, optional): _If True, the input files will be read as a restore of the local class. Those input files need to have been created by the save_locals method. Defaults to False.
        ## end of copied from voxels.py
            vector_descriptors (list of str, optional): List of variables to take in from the trajectory._vectors dataset. Defaults to ["cm","angle"].
            voxel_descriptors (list of str, optional): List of variables to take in from the local._voxels dataset. Defaults to ["cm", "angle"].
            distance_descriptor (bool, optional): Whether or not to take in the distance matrix from local._distance_matrix. Defaults to True.
            director (bool, optional):Whether or not only one xyz component should be taken into account. If false, all xyz components are used. Defaults to False.
            normalization (str, optional): Normalization technique. Choices are: min-max, max and standardize. See methods for more details. Defaults to "standardize".
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
            TODO: multiple xyz component
        Returns:
            xr.Dataset: Raw features dataset. xyz coordinates are removed and if multiple are take into account, _x (or _y, _z) will be added to the feature's name.
        """
        ## same dimensions?
        data = xr.Dataset()

        if vector:
            for name in vector:
                if "comp" in self._vectors[name].coords:
                    if director:
                        data[name] = self._vectors[name].sel(comp=director)
                    else:
                        for i in self.comps:
                            data[name + "_" + i] = self._vectors[name].sel(comp=i)
                else:
                    data[name] = self._vectors[name]

        if voxel:
            for name in voxel:
                if "comp" in self._voxels[name].coords:
                    if director:
                        data[name] = self._voxels[name].sel(comp=director)
                    else:
                        for i in self.comps:
                            data[name + "_" + i] = self._voxels[name].sel(comp=i)
                else:
                    data[name] = self._voxels[name]

        if distance:
            name = "distance"
            if "comp" in self._distance_matrix.coords:
                if director:
                    data[name] = self._distance_matrix.sel(comp=director) # vectors dont necessarily have director
                else:
                    for i in self.comps:
                        data[name + "_" + i] = self._distance_matrix.sel(comp=i)
            else:
                data[name] = self._distance_matrix

        return data.drop_vars('comp')

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

                distance_array=[]
                total_ts = len(self.timesteps)
                for cnt, i in enumerate(self.timesteps):

                    self._print( "\r\tComputing distance on feature {} for timestep {}/{}".format(name, cnt+1,total_ts) )
                    data = [ [i] for i in self._features[name].sel(ts=i).values ]
                    distance_array.append(squareform(pdist( data )))

                    self._print("\n")
                self._features[name] = xr.DataArray(distance_array, coords=[self.timesteps, self.ids, self.ids], dims=['ts', 'id', 'id_n'])

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

        if normalization == "min-max":
            norm_func = self.__normalize_min_max
        elif normalization == "max":
            norm_func = self.__normalize_max
        elif normalization == "standardize":
            norm_func = self.__standardize
        else:
            raise ValueError("Specified normalization not implemented yet:" + normalization)

        for name in self._features:
            data = []
            for ts in self.timesteps:
                data.append( norm_func( self._features[name].sel(ts=ts) ) )
            self._features[name] = xr.DataArray(data, coords=[self.timesteps, self.ids, self.ids], dims=['ts', 'id', 'id_n'])

    def __normalize_max(self, data) -> xr.DataArray:
        """Normalization technique. Divide the whole feature by it's maximum value.

        Formula:
            x_i = x_i / max(x)

        Args:
            data (xr.DataArray): Unnormalized data.

        Returns:
            xr.DataArray: Normalized data. Values range from 0 to 1.
        """
        self.__check_dims(data, n=2) ## idk if necessary
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
        self.__check_dims(data, n=2)
        return ( data - np.min(data) ) / ( np.max(data) - np.min(data) )

    def __standardize(self, data) -> xr.DataArray: ## not working for some reason
        """Normalization technique. Subsracts the average value and and divides by the average difference.

        Formula:
            z_if = (x_if - avg(x_f))/s_f
            s_f = 1/n { sum_i^n (x_if - m_f) }
            TODO: All formulas should be latex compatible and have refs

        Args:
            data (xr.DataArray): Unnormalized data.

        Returns:
            xr.DataArray: Normalized data. Values range from -1 to 1.
        """
        self.__check_dims(data, n=2)
        data = squareform(data)
        data = ( data - data.mean() ) / np.mean( abs(data-data.mean()) )
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

    def combine_features(self, method="sum") -> xr.Dataset:
        """Combines the different kernel matrices of all features into 1 kernel matrix. Different combination methods are implemented

        Args:
            method (str, optional): Method of combination. Choices are sum, product and euclidean. Defaults to "sum".

        Returns:
            xr.Dataset: Combined features. Should have no coordinates except timestep, id and id_n.
        ## might be dataArray
        """
        if method == "sum":
            for cnt, name in enumerate(self._features):
                if cnt == 0:
                    data = self._features[name]
                else:
                    data += self._features[name]
        elif method == "product":
            for cnt, name in enumerate(self._features):
                if cnt == 0:
                    data = self._features[name]
                else:
                    data *= self._features[name]
        elif method == "euclidean":
            for cnt, name in enumerate(self._features):
                if cnt == 0:
                    data = self._features[name] ** 2
                else:
                    data += self._features[name] ** 2
            data = np.sqrt(data)
        return data

    def analyze_features(self):
        """Still working on that

        Returns:
            _type_: _description_
        """
        data = self._features.isel(ts=0,id=0).drop_vars(['ts','id'])

        scatter_matrix(data.to_dataframe())
        plt.show()

        # fig,ax = plt.subplots()
        # scatter = ax.scatter( self.ids, self.ids, c=self._features.isel(ts=0) )
        # bar = fig.colorbar(scatter)
        # plt.show()

        # draw system xyz in 3d with angle as color -> the same as trajectory tho
        # draw feature space but what is feature space?

        return data.to_dataframe()

    def get_features_ds(self) -> xr.Dataset:
        """Getter for the _features dataset.

        Returns:
            xr.Dataset: Trajectory of features.
        """
        return self._features