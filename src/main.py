#!/bin/python

from clusters import cluster
from time import time

start = time()

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

print("1:", testing._features)

features_final = testing.combine_features()

print("2:", testing._features)
print("3:", features_final)

stop = time()
print("\n\nRUNTIME:", stop-start)