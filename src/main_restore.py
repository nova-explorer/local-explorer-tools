#!/bin/python

from clusters import cluster
from time import time

start = time()

try:
    shell = get_ipython().__class__.__name__
    prefix = "../"
except NameError:
    shell = "standard Python"
    prefix = ""

print("Running in:", shell)

testing = cluster(path=prefix+"save/restores/",
                   pattern="testing_*",
                   exclude=[1],
                   vector_patterns=[ [5,6,5] ],
                   restore_trajectory=True,
                   updates=True,
                   neighbors=10,
                   restore_locals=True,
                   vector_descriptors=["angle"],
                   voxel_descriptors=False,
                   distance_descriptor=True,
                   director=False,
                   normalization="max"
                   )

features_final = testing.combine_features()

data = testing.clusterize(features_final, "KMedoids", n_clusters=[1,2,3,4,5])
data = testing.compute_coefficients(data, features_final, dopcev_type=1)
testing.export_to_ovito(data, name='testing_KMedoids', path=prefix+"save/clusters/")
testing.coefficients_to_csv(data, name='testing_KMedoids.csv', path=prefix+'save/', series_name='test', write_style='at')

# a=testing.analyze_features()
# get feature space graph
## do ~10 reference configs by hand

stop = time()
print("\n\nRUNTIME:", stop-start)