import trajectory as t
#import cluster as c
import op
import time

start = time.time()

traj_opt = t.trajectory_options(path="./trajectories/",
                                file_pattern="sma.dump.gz",
                                exclude_types=[1],
                                monomer_types=3)
traj = t.trajectory(traj_opt)

traj.save_trajectory()