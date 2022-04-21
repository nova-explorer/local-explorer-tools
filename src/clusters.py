import sklearn as sk
import sklearn_extra as ske
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from features import features

"""
kmeans : https://stackoverflow.com/questions/5529625/is-it-possible-to-specify-your-own-distance-function-using-scikit-learn-k-means

affinity propagation : https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html#sklearn.cluster.AffinityPropagation

kmedoids : https://scikit-learn-extra.readthedocs.io/en/stable/generated/sklearn_extra.cluster.KMedoids.html#sklearn_extra.cluster.KMedoids

common nearest neighbors clustering : https://scikit-learn-extra.readthedocs.io/en/stable/generated/sklearn_extra.cluster.CommonNNClustering.html#sklearn_extra.cluster.CommonNNClustering

principal component analysis : https://scikit-learn.org/stable/modules/decomposition.html#pca
"""

class cluster( features ):

    def __init__(self, path="./", pattern="ellipsoid.*", exclude=None, vector_patterns=[[2,3,2]], restore_trajectory=False, updates=True,
                 neighbors=10, restore_locals=False,
                 vector_descriptors=["cm","angle"], voxel_descriptors=["cm", "angle"], distance_descriptor=True, director=False, normalization="standardize"
                 ) -> None:

        super().__init__(path, pattern, exclude, vector_patterns, restore_trajectory, updates, neighbors, restore_locals, vector_descriptors, voxel_descriptors, distance_descriptor, director, normalization)

    def clusterize(self, features, algorithm, **kwargs):

        if algorithm == "KMedoids":
            clust_alg = ske.cluster.KMedoids(kwargs, metric='precomputed')
        elif algorithm == "AffinityPropagation":
            clust_alg = sk.cluster.AffinityPropagation(kwargs, metric='precomputed')
        elif algorithm == "commonNN":
            clust_alg = ske.cluster.CommonNNClustering(kwargs, metric='precomputed')

        clust_alg.fit(features)
        ## do loop if n_clusters is range

    def __compute_silhouette(self):
        pass
    def __compute_dopcev(self):
        pass