import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import xarray as xr

from voxels import locals
"""
kmeans : https://stackoverflow.com/questions/5529625/is-it-possible-to-specify-your-own-distance-function-using-scikit-learn-k-means

affinity propagation : https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html#sklearn.cluster.AffinityPropagation

kmedoids : https://scikit-learn-extra.readthedocs.io/en/stable/generated/sklearn_extra.cluster.KMedoids.html#sklearn_extra.cluster.KMedoids

common nearest neighbors clustering : https://scikit-learn-extra.readthedocs.io/en/stable/generated/sklearn_extra.cluster.CommonNNClustering.html#sklearn_extra.cluster.CommonNNClustering

principal component analysis : https://scikit-learn.org/stable/modules/decomposition.html#pca
"""

class cluster( locals ):

    def __init__(self, path="./", pattern="ellipsoid.*", exclude=None, vector_types=None, restore_trajectory=False, updates=True,
                 neighbors=10, restore_locals=False
                #  normalization="standardize" ## proximity matrix, distance function?
                 ) -> None:

        super().__init__(path, pattern, exclude, vector_types, restore_trajectory, updates, neighbors, restore_locals)

        ### Deal with local ops as descriptors
        # self._descriptors = self.__generate_descriptors(vector_descriptors, voxel_descriptors, distance_descriptor, director)
        # self._features = self.compute_features(normalize, standardize)

    def generate_descriptors(self, vector_descriptors=["cm","angle"], voxel_descriptors=["cm", "angle"], distance_descriptor=True, director=False):
        ## same dimensions?
        self._descriptors = xr.Dataset()
        if vector_descriptors:
            for name in vector_descriptors:
                if director:
                    self._descriptors[name] = self._vectors[name].sel(comp=director)
                else:
                    self._descriptors[name] = self._vectors[name]
        if voxel_descriptors:
            for name in voxel_descriptors:
                if director:
                    self._descriptors[name + "_voxel"] = self._voxels[name].sel(comp=director)
                else:
                    self._descriptors[name + "_voxel"] = self._voxels[name]
        if distance_descriptor:
            self._descriptors["distance"] = self._distance_matrix

    def compute_features(self, normalization):
        """_summary_

        Args:
            normalization (_type_): min-max, max, standardize, --dissimilarities--

        Returns:
            _type_: _description_
        """
        features = xr.Dataset()

        # symmetries
        # distance euclidean
        # normalize
            # normalizing distances vs normalizing values ?? normalizing values should be more meaningful
        # combine nd in 2d
        # is sparse ?
        """
        data = np.array[id  [id id id id ] ]
                       [id                 ]
                       [id                 ]
                       [id                 ]
        """

        return features

    def __normalize(self, normalization, data):
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
        if normalization == "min-max":
            norm_func = self.__normalize_min_max
        elif normalization == "max":
            norm_func = self.__normalize_max
        elif normalization == "standardize":
            norm_func = self.__standardize
        # elif normalization == "dissimilarities":
        #     norm_func = self.__dissimilarities
        else:
            raise ValueError("Specified normalization not implemented yet:" + normalization)

        for prop in range(len(data[0])):
            data[:,prop] = norm_func( data[:,prop] )
        return data

    def __apply_symmetries(self, data):
        for name in list(data.keys()):
            if "angle" in name:
                data[name] = self.__angle_symmetry(data[name]) # check if ok with different dimensionality
            if "distance" in name:
                self.__check_distance_symmetry(data[name])

        return data

    def __check_distance_symmetry(self, data):
        pass

    def __angle_symmetry(self, data):
        return abs(np.cos(data))

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
        self.__check_dims(data, n=1)
        ## not needed if data is np.ndarray
        # factor = 1 / max(data)
        # for i in range(len(data)):
        #     data[i] = data[i] * factor
        return data / max(data)

    def __normalize_min_max(self, data):
        """
        x_i = ( x_i - min(x) ) / ( max(x) - min(x) )
        """
        self.__check_dims(data, n=1)
        ## not needed if data is np.ndarray
        # factor = 1 / ( max(data) - min(data) )
        # min_ = min(data)
        # for i in range(len(data)):
        #     data[i] = ( data[i] - min_ ) * factor
        return ( data - min(data) ) / ( max(data) - min(data) )

    def __standardize(self, data):
        """
        z_if = (x_if - avg(x_f))/s_f

        s_f = 1/n { sum_i^n (x_if - m_f) }
        """
        self.__check_dims(data, n=1)
        ## not needed if data is np.ndarray
        # avg = data.mean()
        # length = len(data)
        # factor = 1 / ( 1/length * sum([ data[i]-avg for i in range(length) ]) )
        # for i in range(length):
        #     data[i] = ( data[i] - avg ) * factor
        return len(data) * ( data - data.mean() ) / sum( abs(data-data.mean()) )

    # def __dissimilarities(self, data):
    #     self.__check_dims(data, n=1)
    #     pass

    def __check_dims(self, data, n):
        if isinstance(data, np.ndarray):
            if data.ndim == n:
                pass
            else:
                raise ValueError("shit isnt good in the right dimensionality (should write proper error later)")
        else:
            raise ValueError("havent written this part properly yet")

    def get_descriptor_ds(self) -> xr.Dataset:
        return self._descriptors