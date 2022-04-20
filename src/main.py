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

testing = cluster(path="trajectories/",
                   pattern="sma.dump.gz",
                   exclude=[1],
                   vector_types=[3],
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

a=testing.analyze_features()
# get feature space graph
## do ~10 reference configs by hand

stop = time()
print("\n\nRUNTIME:", stop-start)