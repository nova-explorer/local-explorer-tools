# !/bin/python
from time import time
import os, psutil
from glob import glob

from clusters import cluster

process = psutil.Process(os.getpid())
start = time()

try:
    shell = get_ipython().__class__.__name__
    prefix = "../"
except NameError:
    shell = "standard Python"
    prefix = ""

print("Running in:", shell)

testing = cluster(path = prefix+'save/restores/',
                  pattern = 'testing.*',
                  exclude = None,
                  vector_patterns = [ [5, 6, 5] ],
                  restore_trajectory = True,
                  updates = True,
                  neighbors = 10,
                  restore_locals = True,
                  vector_descriptors = ['angle'],
                  voxel_descriptors = False,
                  distance_descriptor = True,
                  director = False,
                  normalization = "max"
                  )

## Vanilla modifications
# a = testing.get_features_ds()
##

## Full modifications
a = testing.filter_features(0.3, 2, 0.1)
w1 = testing.set_weights(a, method='auto')
w2 = testing.set_weights(a, method={'distance*':2, 'angle*':1})
weights = testing.combine_weights(w1, w2, method='product')
##

features_final = testing.combine_features(a, weights, method='sum')

data = testing.clusterize(features_final, "KMedoids", n_clusters = [1,2,3,4,5])
data = testing.compute_coefficients(data, features_final, dopcev_type = 4)

name = "test"

testing.export_to_ovito(data, name, path = prefix+'save/clusters/')
testing.coefficients_to_csv(data, name+'.csv', path = prefix+'save/', series_name = 'testing', write_style = 'wt')

stop = time()
print("\n\nRUNTIME:", stop-start)
print("END MEMORY USAGE:", process.memory_info().rss / 1024**2, "MB")