#!/bin/python

from clusters import cluster
from time import time

start = time()

import matplotlib.pyplot as plt

# testing = t.trajectory(path="local-explorer-tools/trajectories/",
#                        pattern="poly.dump",
#                        exclude=[1],
#                        vector_types=[3],
#                        restore=False)

# testing = v.locals(path="../../local-explorer-tools/trajectories/",
#                    pattern="sma.dump.gz",
#                    exclude=[1],
#                    vector_types=[3],
#                    restore_trajectory=False,
#                    updates=True,
#                    neighbors=10,
#                    restore_locals=False)

try:
    shell = get_ipython().__class__.__name__
    prefix = "../"
except NameError:
    shell = "standard Python"
    prefix = ""

print("Running in:", shell)

testing = cluster(path=prefix+"trajectories/",
                   pattern="sma.dump.gz",
                   exclude=[1],
                   vector_patterns=[ [2,3,2] ],
                   restore_trajectory=False,
                   updates=True,
                   neighbors=10,
                   restore_locals=False,
                   vector_descriptors=["angle"],
                   voxel_descriptors=False,
                   distance_descriptor=True,
                   director=False,
                   normalization="max"
                   )

features_final = testing.combine_features()

# a=testing.analyze_features()
# get feature space graph
## do ~10 reference configs by hand

stop = time()
print("\n\nRUNTIME:", stop-start)