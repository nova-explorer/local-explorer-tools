import numpy as np
import xarray as xr

def global_onsager(ds) -> xr.Dataset :
    """Computes the Onsager order parameter of the whole system. The director of the system is the mean orientation of all vector orientations.

    Args:
        ds (xr.Dataset): vectors dataset

    Returns:
        xr.Dataset: Dataset containing the order parameter for all timesteps.
        TODO: not sure if return is a dataset or a dataarray.
    """
    director = ds.coord.mean(dim='id')
    director = director / np.sqrt(np.square(director).sum(dim=['comp']))

    data = ds.coord
    op = np.square(data.dot(director, dims=['comp'])).mean(dim=['id'])
    op=1.5 * op - 0.5
    return op

def rdf():
    ## should do soon
    return 0

def voxel_onsager(ds) -> float:
    """Computes the Onsager order parameter of voxel i for a single timestep. the director is considered to be particle i.

    Args:
        ds (xr.Dataset): voxels dataset for timestep ts_ and id_.

    Returns:
        float: Local Onsager order parameter for that voxel.
    TODO: Check consistency between voxel_onsager and global_onsager_op.
    """
    data = ds.coord
    op = 0
    neigh = data.id_2[1:].values
    nb_neigh = len(neigh)

    for i in neigh:
        op += ( 3 * np.dot(data[0],data[i])**2 - 1 ) / 2 / nb_neigh
    return op

## not the most useful local order parameters
def voxel_common_neigh():
    return 0

def voxel_pred_neigh():
    return 0

def voxel_another_neigh():
    return 0

def voxel_neigh_dist():
    return 0