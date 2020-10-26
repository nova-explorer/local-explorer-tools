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

    return ds

# def mcmillan(ds):
#     return 0
# def gz12(ds, ini_layer=35, direction='z'):
#     return 0

def rdf(ds, cutoff=25, nb_bins=100): ## Some infos are relevant to clusterings
    bounds_x = ds.atoms.b_x1 - ds.atoms.b_x0
    bounds_y = ds.atoms.b_y1 - ds.atoms.b_y0
    bounds_z = ds.atoms.b_z1 - ds.atoms.b_z0
    bounds = [ bounds_x.values[0], bounds_y.values[0], bounds_z.values[0] ]

    positions = ds.vectors.cm.isel(ts=0).values
    dx = squareform(pdist(positions[ :,[0] ]))
    dy = squareform(pdist(positions[ :,[1] ]))
    dz = squareform(pdist(positions[ :,[2] ]))

    dx = np.where(dx >= bounds[0]/2, dx - bounds[0], dx)
    dy = np.where(dy >= bounds[1]/2, dy - bounds[1], dy)
    dz = np.where(dz > bounds[2]/2, dz - bounds[2], dz)

    distance = np.sqrt(dx**2 + dy **2 + dz**2)

    rdf = []
    for d in distance:
        d = d[d<=cutoff]
        d = d[d!=0]
        freq, bins= np.histogram(d, nb_bins)
        rdf.append(freq)

    return rdf