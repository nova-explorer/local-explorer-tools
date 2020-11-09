## TODO doc

import numpy as np
import xarray as xr
#import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

def legendre_poly(ds, l=2, director='z'):
    if l == 2:
        op = np.cos(ds.angle.sel(comp=director)) ** 2
        op = op.mean(dim='id')

        op = (3*op - 1) / 2
        ds['p2'] = op
    elif l == 4:
        cos2 = np.cos(ds.angle.sel(comp=director)) ** 2
        cos2 = cos2.mean(dim='id')
        cos4 = np.cos(ds.angle.sel(comp=director)) ** 4
        cos4 = cos4.mean(dim='id')

        op = (35 * cos4 - 30 * cos2 + 3) / 8
        ds['p4'] = op

    return ds

def qmatrix(ds):
    op_list = []
    for i in range(len(ds.ts)):
        q_matrix = 0
        length = len(ds.id)
        for j in range(length):
            current = ds.coord[j][i] # id comes first, then timestep
            q_matrix += ( 3 * np.outer(current, current) - np.identity(3) ) / 2 / length

        eig_vals, eig_vecs = np.linalg.eig(q_matrix)
        idx = eig_vals.argsort()[::-1]
        eig_vals = eig_vals[idx]
        eig_vecs = eig_vecs[:, idx]
        op = eig_vals[1] * -2.
        #director = eig_vecs[:, 0] # if needed could be added

        op_list.append(op)
    ds['p2_qm'] = xr.DataArray(op_list, coords=[ds.ts], dims=['ts'])
    return op
    # return ds

def rdf(traj, cutoff_val=25, nb_bins=10):
    bounds = traj.atoms.bounds.isel(ts=0).values
    vol_cell = bounds[0] * bounds[1] * bounds[2]

    # données des centres de masses des ellipsoides
    positions = traj.vectors.cm.isel(ts=0).values
    # matrices de distance en x y et z
    dx = squareform(pdist(positions[ :,[0] ]))
    dy = squareform(pdist(positions[ :,[1] ]))
    dz = squareform(pdist(positions[ :,[2] ]))

    # applique les PBC
    dx = np.where(dx >= bounds[0]/2, dx - bounds[0], dx)
    dy = np.where(dy >= bounds[1]/2, dy - bounds[1], dy)
    dz = np.where(dz >= bounds[2]/2, dz - bounds[2], dz)
    # combine les distances en x,y et z
    distance = np.sqrt(dx**2 + dy**2 + dz**2)

    cutoff = np.linspace(0, cutoff_val, nb_bins)
    diff = cutoff[1] - cutoff[0]
    N_ATOMS = 1800
    CONSTANT = 4/3 * np.pi / vol_cell * N_ATOMS

    rdf = []
    mean_rdf = [0] * len(cutoff)
    for i in distance:
        bins = [0] * len(cutoff)
        i = i[i<=cutoff_val]
        i = i[i!=0]
        for j in i:
            for idd,d in enumerate(cutoff):
                d_min = d - diff
                d_max = d + diff
                if d_min <= j < d_max:
                    vfrac = CONSTANT * (d_max**3 - d_min**3)
                    bins [idd] += 1 / vfrac
                    mean_rdf [idd] += 1 / vfrac / N_ATOMS
        rdf.append(bins)

    return rdf, mean_rdf, cutoff