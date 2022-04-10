import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import xarray as xr
from scipy.spatial.distance import pdist, squareform

from voxels import locals
"""
kmeans : https://stackoverflow.com/questions/5529625/is-it-possible-to-specify-your-own-distance-function-using-scikit-learn-k-means

affinity propagation : https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html#sklearn.cluster.AffinityPropagation

kmedoids : https://scikit-learn-extra.readthedocs.io/en/stable/generated/sklearn_extra.cluster.KMedoids.html#sklearn_extra.cluster.KMedoids

common nearest neighbors clustering : https://scikit-learn-extra.readthedocs.io/en/stable/generated/sklearn_extra.cluster.CommonNNClustering.html#sklearn_extra.cluster.CommonNNClustering

principal component analysis : https://scikit-learn.org/stable/modules/decomposition.html#pca
"""

class cluster( locals ):

    ## missing much communication to user

    def __init__(self, path="./", pattern="ellipsoid.*", exclude=None, vector_types=None, restore_trajectory=False, updates=True,
                 neighbors=10, restore_locals=False,
                 vector_descriptors=["cm","angle"], voxel_descriptors=["cm", "angle"], distance_descriptor=True, director=False, normalization="standardize" ## proximity matrix, distance function?
                 ) -> None:

        super().__init__(path, pattern, exclude, vector_types, restore_trajectory, updates, neighbors, restore_locals)

        self._features = self.__generate_raw_features(vector_descriptors, voxel_descriptors, distance_descriptor, director)
        self.__apply_symmetries()
        self.__compute_distances()

        self.__normalize(normalization)
        ## public self.add_features

    def __generate_raw_features(self, vector, voxel, distance, director):
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
        # if voxel_descriptors:
        #     for name in voxel_descriptors:
        #         if director:
        #             self._descriptors[name + "_voxel"] = self._voxels[name].sel(comp=director)
        #         else:
        #             self._descriptors[name + "_voxel"] = self._voxels[name]
        # if distance_descriptor:
        #     self._descriptors["distance"] = self._distance_matrix

    def __compute_distances(self):
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

    def __normalize(self, normalization):
        """_summary_

        Args:
            normalization (_type_): min-max, max, standardize, --dissimilarities--

        Returns:
            _type_: _description_

        data = np.array[id  [prop1 prop2 prop3 prop4] ]
                       [id                            ]
                       [id                            ]
                       [id                            ]

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

    def __apply_symmetries(self):
        for name in list(self._features.keys()):
            if "angle" in name:
                self._features[name] = self.__angle_symmetry(self._features[name]) # check if ok with different dimensionality
            if "distance" in name:
                self.__check_distance_symmetry(self._features[name])

    def __check_distance_symmetry(self, data):
        # check if sparse and no distance is larger than bounds
        pass

    def __angle_symmetry(self, data):
        return abs(np.cos(data))

    def combine_features(self):
        ## add more combining methods like euclidean and product
        for cnt, name in enumerate(self._features):
            if cnt == 0:
                data = self._features[name]
            else:
                data += self._features[name]
        return data

    def __clusterize(self):
        pass
    def __compute_silhouette(self):
        pass
    def __compute_dopcev(self):
        pass

    def __normalize_max(self, data):
        """
        x_i = x_i / max(x)
        """
        self.__check_dims(data, n=2)
        return data / np.max(data)

    def __normalize_min_max(self, data):
        """
        x_i = ( x_i - min(x) ) / ( max(x) - min(x) )
        """
        self.__check_dims(data, n=2)
        return ( data - np.min(data) ) / ( np.max(data) - np.min(data) )

    def __standardize(self, data): ## not working for some reason
        """
        z_if = (x_if - avg(x_f))/s_f

        s_f = 1/n { sum_i^n (x_if - m_f) }
        """
        self.__check_dims(data, n=2)
        return ( data - data.mean() ) / np.mean( abs(data-data.mean()) )

    def __check_dims(self, data, n):
        if isinstance(data, xr.core.dataarray.DataArray):
            if data.ndim == n:
                pass
            else:
                print(data)
                raise ValueError("shit isnt good in the right dimensionality (should write proper error later)")
        else:
            raise ValueError("havent written this part properly yet")

    def get_descriptor_ds(self) -> xr.Dataset:
        return self._descriptors