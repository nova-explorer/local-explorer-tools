"""[summary]
"""
import trajectory as t
import cluster as c
import compute_op as cop

traj_opt = t.trajectory_options(path="./trajectories/test/",
                                file_pattern="sma.dump.gz",
                                exclude_types=[1],
                                monomer_types=3)
c_opt = c.cluster_options()

traj = t.trajectory(traj_opt)
clust = c.cluster(traj, c_opt)

print('datasets done')

clust.add_local_op(op_type='onsager', nb_ave=1)
clust.add_local_op(op_type='onsager', nb_ave=2)
clust.add_local_op(op_type='common_neigh', nb_ave=1)
clust.add_local_op(op_type='common_neigh', nb_ave=2)
print('voxel OPs done')

# rdf = cop.rdf(clust, cutoff_val=15, nb_bins=50)
# print('rdf done')

traj.save_trajectory(name='sma', path='./save/')
clust.save_cluster(name='sma', path='./save/')