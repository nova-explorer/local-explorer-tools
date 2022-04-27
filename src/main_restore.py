#!/bin/python

from clusters import cluster
from time import time

start = time()

import matplotlib.pyplot as plt

try:
    shell = get_ipython().__class__.__name__
    prefix = "../"
except NameError:
    shell = "standard Python"
    prefix = ""

print("Running in:", shell)

testing = cluster(path=prefix+"save/",
                   pattern="testing_*",
                   exclude=[1],
                   vector_patterns=[ [2,3,2] ],
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

# testing.clusterize(features_final, "KMedoids", n_clusters=2)

# a=testing.analyze_features()
# get feature space graph
## do ~10 reference configs by hand

stop = time()
print("\n\nRUNTIME:", stop-start)