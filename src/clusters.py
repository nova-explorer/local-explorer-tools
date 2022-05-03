import sklearn.cluster as sk_c
import sklearn_extra.cluster as ske_c
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import xarray as xr

from features import features
import io_local as iol

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

        self._clustraj = self.__generate_clustered_trajectory(["xu", "yu", "zu", "c_orient[*]", "c_shape[*]", "bounds"])

    def clusterize(self, features, algorithm="KMedoids", n_clusters=None, name="cluster.dump", **kwargs):

        if algorithm == "KMedoids":
            if isinstance(n_clusters, int):
                n_clusters = [n_clusters]
                clust_alg = [ ske_c.KMedoids(metric='precomputed', n_clusters=i, **kwargs) for i in n_clusters ]
            else:
                raise ValueError("n_clusters bad format") ## do a proper error

        elif algorithm == "AffinityPropagation":
            clust_alg = sk_c.AffinityPropagation(metric='precomputed', **kwargs)

        elif algorithm == "commonNN":
            clust_alg = ske_c.CommonNNClustering(metric='precomputed', **kwargs)

        labels_array = []
        for ts in self.timesteps:
            labels = []
            for i, cluster in enumerate(clust_alg):
                cluster.fit(features.sel(ts=ts))
                labels.append(cluster.labels_)
            labels_array.append(labels)

        self._clustraj['labels'] = xr.DataArray(labels_array, coords=[self.timesteps, n_clusters, self.ids], dims=['ts', 'n_clusters', 'id'])


    def __generate_clustered_trajectory(self, vars):
        ds = xr.Dataset()
        data_vars = list(self._atoms.keys())
        for i in vars:
            if "[*]" in i:
                for j in [ name for name in data_vars if i[:-3] in name ]:
                    ds [j] = self._atoms[j].where(self._atoms.id==self._vectors.id)
            elif "id" not in self._atoms[i].coords:
                ds [i] = self._atoms[i]
            else:
                ds [i] = self._atoms[i].where(self._atoms.id==self._vectors.id)
        return ds

    def __compute_silhouette(self):
        pass
    def __compute_dopcev(self):
        pass

    def get_clustraj_ds(self):
        return self._clustraj

    def export_to_ovito(self, name, path):
        for i in self._clustraj.n_clusters.values: ## add for cluster param scan instead of n_clusters specifically
            for j in self._clustraj.ts.values:
                name_ij = "cluster." + name + "_" + str(j) + "_" + str(i) + ".dump"
                iol.frame_to_dump(self._clustraj.sel(n_clusters=i, ts=j), name_ij, path)