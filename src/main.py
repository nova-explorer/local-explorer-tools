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

testing = cluster(path="../../local-explorer-tools/trajectories/",
                   pattern="sma.dump.gz",
                   exclude=[1],
                   vector_types=[3],
                   restore_trajectory=False,
                   updates=True,
                   neighbors=10,
                   restore_locals=False)

testing.generate_descriptors(vector_descriptors=["angle"], voxel_descriptors=False, distance_descriptor=True, director=False)

print(testing.get_voxel_ds())

# testing.add_local_op()

# testing.compute_features()
# testing.__clusterize()

stop = time()
print("\n\nRUNTIME:", stop-start)