"""[summary]
"""
import trajectory as t
import cluster as c

traj_opt = t.trajectory_options(path="./trajectories/",
                                file_pattern="sma.dump.gz",
                                exclude_types=[1],
                                monomer_types=3)
c_opt = c.cluster_options()

traj = t.trajectory(traj_opt)
cluster = c.cluster_map(traj, c_opt)
print('done')
# traj.save_trajectory(path="./save/")