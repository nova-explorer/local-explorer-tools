import numpy as np
import xarray as xr

def global_onsager_op(ds) -> xr.Dataset :
    director = ds.coord.mean(dim='id')
    director = director / np.sqrt(np.square(director).sum(dim=['comp']))

    data = ds.coord
    op = np.square(data.dot(director, dims=['comp'])).mean(dim=['id'])
    op=1.5 * op - 0.5
    return op

def rdf():
    return 0

def voxel_onsager(ds):
    data = ds.coord
    op = 0
    neigh = data.id_2[1:].values
    nb_neigh = len(neigh)

    for i in neigh:
        op += ( 3 * np.dot(data[0],data[i])**2 - 1 ) / 2 / nb_neigh

    return op

def voxel_common_neigh():
    return 0

def voxel_pred_neigh():
    return 0

def voxel_another_neigh():
    return 0

def voxel_neigh_dist():
    return 0