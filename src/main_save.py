# !/bin/python
from time import time
import os, psutil
from glob import glob

from voxels import local

process = psutil.Process(os.getpid())
start = time()

try:
    shell = get_ipython().__class__.__name__
    prefix = "../"
except NameError:
    shell = "standard Python"
    prefix = ""

print("Running in:", shell)

testing = local(path = prefix+'trajectories/',
                    pattern = 'poly.dump',
                    exclude = [1],
                    vector_patterns = [ [5, 6, 5] ],
                    restore_trajectory = False,
                    updates = True,
                    neighbors = 10,
                    restore_locals = False,
                    )

testing.add_local_op(op_type = 'onsager', nb_ave = 1)
testing.add_local_op(op_type = 'onsager', nb_ave = 2)
testing.add_rdf(20, 200)


testing.save_trajectory('testing', prefix+'save/restores')
testing.save_local('testing', prefix+'save/restores')

stop = time()
print("\n\nRUNTIME:", stop-start)
print("END MEMORY USAGE:", process.memory_info().rss / 1024**2, "MB")