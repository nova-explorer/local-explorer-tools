"""
Example script for restoring the trajectory and cluster classes from saved datasets. As of current version, the script (this file) needs to be executed from inside the src/ directory (since src is not yet a package).
You may also need to fix path in traj_opt and clust_opt if you run from inside src/ or from the parent directory. You may also need to edit paths so that they match yours.
You may want to run this in ipython to make sure everything is restored properly at first or play with the datasets
"""

import trajectory as t
import cluster as c

# How to generate and save datasets (skip/comment if you already have saved datasets)

traj_opt = t.trajectory_options(path="./trajectories/",
                                file_pattern="sma.dump.gz",
                                exclude_types=[1],
                                monomer_types=3)
c_opt = c.cluster_options()

traj = t.trajectory(traj_opt)
clust = c.cluster(traj, c_opt)

traj.save_trajectory(path='./save/')
clust.save_cluster(path='./save/')

# How to restore datasets

traj_opt = t.trajectory_options(path='./save/', restore=True)
c_opt = c.cluster_options(path='./save/', restore=True)

traj = t.trajectory(traj_opt)
clust = c.cluster(traj, c_opt)