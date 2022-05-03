#!/bin/python

from cgi import test
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

testing.save_trajectory("testing", prefix+"save/")
testing.save_local("testing", prefix+"save/")

stop = time()
print("\n\nRUNTIME:", stop-start)