"""
Example script for getting and drawing the local rdf distribution for a trajectory with a single timestep. As of current version, the script (this file) needs to be executed from inside the src/ directory (since src/ is not yet a package).
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
rdf = cop.rdf(clust, cutoff_val=15, nb_bins=50)

spd_threshold = 1 # This is the treshold for which a peak around the mean maximum peak is considered SmA
###############################################################

mean = rdf.mean(dim='id').isel(ts=0).values.tolist() # we compute the average RDF
cutoff = rdf.distance.values.tolist() # distance bins
rdf = rdf.isel(ts=0).values

sma_pair_dist = cutoff[ mean.index(max(mean)) ] # gives the distance value of the max average peak
spd_min = sma_pair_dist - spd_threshold
spd_max = sma_pair_dist + spd_threshold

sma = []
iso = []
for i,rdf_i in enumerate(rdf): # loop over particles
    if spd_min <= cutoff[ rdf_i.tolist().index(max(rdf_i)) ] <= spd_max: # if maximum peak for this particle is within the thresholf
        sma.append(i) # then it is considered SmA
    else:
        iso.append(i) # or it is considered isotropic (we assume here only SmA and isotropic exist in our system). If nematic is also needed, one could check the local op value of this particle.

# getting positions
sma = traj.vectors.isel(ts=0, id=sma).cm.values
iso = traj.vectors.isel(ts=0, id=iso).cm.values

# Plot the mean RDF profile
fig = plt.figure(1)
plt.plot(cutoff,mean)

# Plot the distribution of mesophase based on rdf distribution
fig = plt.figure(2)
ax = Axes3D(fig)
sma_plot = ax.scatter(sma[:,0], sma[:,1], sma[:,2], label='SmA')
iso_plot = ax.scatter(iso[:,0], iso[:,1], iso[:,2], label='Isotropic')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.legend()

plt.show()