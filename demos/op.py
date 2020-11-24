"""
Example script for getting and drawing the local onsager order parameter for a trajectory with a single timestep. At the end, a comparison is made between the global and local order parameter. As of current version, the script (this file) needs to be executed from inside the src/ directory (since src/ is not yet a package).
You may also need to fix path in traj_opt if you run from inside src/ or from the larger directory. You may want to edit this script or run main.py to have saved datasets and restore them here instead of generating them each time you execute this. See other demos for that.
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import xarray as xr

import trajectory as t
import cluster as c
import compute_op as cop

traj_opt = t.trajectory_options(path="./trajectories/",
                                file_pattern="sma.dump.gz",
                                exclude_types=[1],
                                monomer_types=3)
c_opt = c.cluster_options()

traj = t.trajectory(traj_opt)
clust = c.cluster(traj, c_opt)
clust.add_local_op()

########################################################################################3

op = clust.voxels.onsager_1.isel(ts=0).values
pos = traj.vectors.isel(ts=0).cm.values

fig = plt.figure(1)
ax = Axes3D(fig)
onsager_scatter = ax.scatter(pos[:,0], pos[:,1], pos[:,2], c=op, edgecolors='k', cmap='jet')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
bar = fig.colorbar(onsager_scatter)
bar.ax.set_ylabel("Onsager OP")

plt.show()

print('Global op :', traj.vectors.p2_qm.values[0])
print('Local op average :',op.mean())
