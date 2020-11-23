"""
Functions that compute order parameters or structure/dynamic factors.

Usage: It is recommended to use these functions within their associated class if possible. See the Usage section for directions on each one.

Requires modules: xarray
                  numpy

References (More details on references are available in the readme):
        1. Quinquephenyl: The Simplest Rigid-Rod-Like Nematic Liquid Crystal, or is it? An Atomistic Simulation (10.1002/cphc.201301126)
        2. Machine learning-aided analysis for complex local structure of liquid crystal polymers (10.1038/s41598-019-51238-1)
        3. RDF
"""
import numpy as np
import xarray as xr

def legendre_poly(ds, l = 2, director = 'z'):
    """Computes the orientational order parameter with the legendre polynomial method. The orientational order parameter gives information about how oriented particles are towards a director vector. Formula 1 is the general P_L equation while 2 and 3 are the solved P_L equations for respectively L = 2 and L = 4. P_6 and beyond are rarely considered useful.

    Args:
        ds (xarray.Dataset): Vector dataset from the trajectory class
        l (int, optional): Rank of the polynomial. A higher value gives more importance to angle difference between particles. Only P2 and P4 are implemented as of now. Defaults to 2.
        director (str, optional): Defines the director. Can be x, y or z. Defaults to 'z'.

    Returns:
        xarray.Dataset: Vector dataset but now with the orientational order parameter added to it as a new Data variable name p2 or p4.

    Usage: result = legendre_poly(vectors)
        * Accessing data:
            result['p2'] or result.p2
            result['p4'] or result.p4
        * In a more concrete example:
            traj.vectors = cop.legendre_poly(traj.vectors, l=2, director='z')

    TODO: * Find a way to compute for any L.
          * Find a way to compute with a custom director. This is not urgent as this method is not considered as the best for P2.
          * Better way to pass vectors (like in cpp)
    """
    r"""Formula: reference 1.
            (1) $$\langle P_L\rangle = \int_0^\pi \ d\beta sin\beta P_l (cos\beta)Pcos\beta , L=2,4,...$$
            (2) $$\langle P_2 \rangle = \langle \sum_{i=1}^N (3cos^2\beta_i(t) - 1)\rangle / 2N$$
            (3) $$\langle P_4 \rangle = \langle \sum_{i=1}^N (35cos^4\beta_i(t) - 30cos^2\beta_i(t) +3)\rangle / 8N$$
    """
    if l == 2:
        op = np.cos(ds.angle.sel(comp = director)) ** 2
        op = op.mean(dim = 'id') ## mean is inside the summation, need to check if it's ok
        op = (3*op - 1) / 2
        ds['p2'] = op

    elif l == 4:
        cos2 = np.cos(ds.angle.sel(comp = director)) ** 2
        cos2 = cos2.mean(dim = 'id')
        cos4 = np.cos(ds.angle.sel(comp = director)) ** 4
        cos4 = cos4.mean(dim = 'id')
        op = (35 * cos4 - 30 * cos2 + 3) / 8
        ds['p4'] = op
    return ds

def qmatrix(ds):
    """Computes the P2 orientational order parameter with the Q ordering matrix method. The orientational order parameter gives information about how oriented particles are towards each other. The Q matrix method doesn't need the director so it is prefered to the legendre polynomial method. Equation 1 is how we compute the Q ordering matrix. The eigenvalues of the Q matrix gives us 3 values which are shown in equation 2. Equation 3 shows how from the middle eigenvalue we get P2.

    Args:
        ds (xarray.Dataset): Vector dataset from the trajectory class

    Returns:
        xarray.Dataset: Vector dataset but now with the orientational order parameter added to it as a new Data variable named p2_qm.

    Usage: result = qmatrix(vectors)
        * Accessing data:
            result['p2_qm'] or result.p2_qm
        * In a more concrete example:
            traj.vectors = cop.qmatrix(traj.vectors)
    """
    r"""
    Formula: reference 1
        (1) $$\bold{Q}(t) = \sum_{j=1}^N[3\bold{u}_j(t)\bigotimes\bold{u}_j - I]/2N$$
        (2) $$\lambda_-(t) < \lambda_0(t) < \lambda_+(t)$$
        (3) s$$\langle P_2(t)\rangle = -2\langle\lambda_0(t)\rangle$$
    """
    op = None
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
    ds['p2_qm'] = xr.DataArray(op_list, coords = [ds.ts], dims = ['ts'])
    return ds

def rdf(cluster, cutoff_val = 15, nb_bins = 50):
    """Computes the distribution of radial distribution functions for each particle. The RDF gives information about the density surrounding a particle. A strong peak at certain distances may be a sign of a certain arrangment in the system. For example, a strong peak near 5 Angstrom for our LCP systems is characteristic of a Smectic A mesophase. Equation 1 is the density function. While the mathematical equation is quite simple, it's translation to code is less straigthforward than other order parameters. RDF works by categorizing in bins the number of particles within a spherical volume. Then we plot the number of particles (with a normalizing factor) in each bin vs the radius of the sphere.

    Args:
        cluster (cluster custom class): Object generated from the cluster class. While not all information are necessary, we still need the distance dataarray and the bounds from the traj dataset. There may be a simpler way to convey both without passing the whole class.
        cutoff_val (int, optional): Cutoff value for which the sphere's radius will not go higher. The higher the value, the less precise the results since bins are larger. Defaults to 15.
        nb_bins (int, optional): Number of bins created. More bins means more precision but more compute time. Defaults to 50.

    Returns:
        xarray.DataArray: Distribution of RDF for each particle. It can then be averaged to have the global RDF.

    Usage: result = rdf(cluster)
        * Accessing data:
            rdf or rdf.mean(dim='id')
        * In a more concrete example:
            rdf = cop.rdf(cluster, cutoff_val=15, nb_bins=50)
    """
    r"""
    Formula:
        $$g(r) = \frac{V_{cell}}{V_rN}\sum_{i=1}^N\sum_{i>j}^N\delta(r-r_{ij})$$

    """
    rdf_array = []
    cutoff = []
    for step in cluster.dist.ts.values:
        bounds = cluster.traj.atoms.bounds.sel(ts = step).values
        vol_cell = bounds[0] * bounds[1] * bounds[2]

        distances = cluster.dist.sel(ts=step).values

        cutoff = np.linspace(0, cutoff_val, nb_bins)
        diff = cutoff[1] - cutoff[0]
        CONSTANT = 4/3 * np.pi / vol_cell * len(cluster.dist.id)

        rdf = []
        for i in distances:
            bins = [0] * len(cutoff)
            i = i[i <= cutoff_val]
            i = i[i != 0]
            for j in i:
                for idd, d in enumerate(cutoff):
                    d_min = d - diff
                    d_max = d + diff
                    if d_min <= j < d_max:
                        vfrac = CONSTANT * (d_max**3 - d_min**3)
                        bins [idd] += 1 / vfrac
            rdf.append(bins)
        rdf_array.append(rdf)
    return xr.DataArray(rdf_array, coords=[cluster.dist.ts, cluster.dist.id, cutoff], dims=['ts', 'id', 'distance'])

def voxel_onsager(data):
    """Computes the P2 orientational order parameter on a voxel using the Onsager method. The voxel P2 gives information on wether the neighbor particles (j) surrounding a certain particle(i) are oriented towards i. This function is part of the voxels dataset from the cluster class and should not be used as is. Voxel size is changed throught the cluster class (option: nb_neigh). Equation 1 corresponds to the definition of instantenous Onsager order parameter.

    Args:
        data (xarray.Dataset): Voxels dataset from the cluster class. Timestep and particle id are limited to those computed in this single operation.

    Returns:
        float: order parameter computed from this function for the selected timestep and particle id.

    Usage: cluster.add_local_op()
        *Accessing data:
            cluster.voxels['onsager'] or cluster.voxels.onsager
        * In a more concrete example:
            cluster.add_local_op(op_type='onsager')
    """
    r"""
    Formula: reference 2
        (1) $$S_i^{(N,1)} = \sum_{j\in\bold{N}_b(i)} \{3(\bold{u}_i . \bold{u}_j)^2 - 1\} / 2n_b$$
    """
    data = data.coord
    op = 0
    neigh = data.id_2[1:].values
    for i in neigh:
        op += (3*np.dot(data[0], data[i])**2 - 1) / 2 / len(neigh)
    return op

def voxel_common_neigh(data, dist, i):
    """STILL IN TESTING

    Args:
         data (xarray.Dataset): Voxels dataset from the cluster class. Timestep is limited to the current one.
        dist (xarray.DataArray): Distance dataarray from the cluster class. Timestep is limited to the current one.
        i (int): particle id of the current particle.

    Returns:
        float: order parameter computed from this function for the selected timestep and particle id.

    Usage: cluster.add_local_op(op_type='common_neigh')
        *Accessing data:
            cluster.voxels['common_neigh'] or cluster.voxels.common_neigh
    """
    r"""
    Formula: reference 2
        (1) $$A_i^{(N,1,m_b,o_i,o_j,o_k)} = \frac{1}{n_b}\sum_{j\in\bold{N}_b(i)} |\sum_{k\in\bold{N}_b(i,j)}(\bold{r}_{ik}^{(o_i,o_k)} - \bold{r}_{jk}^{(o_j,o_k)}) |^2$$
    """
    op = 0
    vox_i = data.voxel.sel(id=i)[1:].values.tolist()
    for j in vox_i:
        op_l = 0
        vox_j = data.voxel.sel(id=j)[1:].values.tolist()
        for k in vox_i + vox_j:
            op_l += ( dist.sel(id=i, id_2=k) + dist.sel(id=j, id_2=k) ).values
        op += op_l ** 2
    return op / len(vox_i)

def voxel_pred_neigh(data, dist, i):
    r"""[summary]
    check if relevant. Error in article with formula

    Formula:
        $$P_i^{(N,1,m_b,o_i,o_j,o_k)} = \frac{1}{n_b}\sum_{j\in\bold{N}_b(i)} |\sum_{k\in\bold{N}_b(i,j)}(\bold{r}_{ij}^{(o_i,o_k)} - \bold{r}_{kj}^{(o_k,o_j)}) |^2$$
    Args:
        data ([type]): [description]
        dist ([type]): [description]
        i ([type]): [description]

    Returns:
        [type]: [description]
    """
    op = 0
    vox_i = data.voxel.sel(id=i)[1:].values.tolist()
    for j in vox_i:
        op_l = 0
        vox_j = data.voxel.sel(id=j)[1:].values.tolist()
        for k in vox_i + vox_j:
            op_l += ( dist.sel(id=i, id_2=j) + dist.sel(id=k, id_2=j) ).values
        op += op_l ** 2
    return op / len(vox_i)

def voxel_another_neigh(data, dist, i):
    r"""[summary]

    Formula:
        $$M_i^{(N,1,m_b,o_i,o_j,o_k)} = \frac{1}{n_b}|\sum_{j\in\bold{N}_b(i)} \sum_{k\in\bold{N}_b(i,j)}(\bold{r}_{ik}^{(o_i,o_k)} - \bold{r}_{kj}^{(o_k,o_j)}) |^2$$

    Args:
        data ([type]): [description]
        dist ([type]): [description]
        i ([type]): [description]

    Returns:
        [type]: [description]
    """
    op = 0
    vox_i = data.voxel.sel(id=i)[1:].values.tolist()
    for j in vox_i:
        op_l = 0
        vox_j = data.voxel.sel(id=j)[1:].values.tolist()
        for k in vox_i + vox_j:
            op_l += ( dist.sel(id=i, id_2=j) + dist.sel(id=k, id_2=j) ).values
        op += op_l
    return op ** 2 / len(vox_i)

def voxel_neigh_dist(data, dist, i):
    op = 0
    vox = data.voxel.sel(id=i)[1:].values.tolist()
    for j in vox:
        for k in vox:
            if j != k:
                op += ( dist.sel(id=i, id_2=j) * dist.sel(id=i, id_2=j) * dist.sel(id=j, id_2=k) ).values
    return op / (len(vox)**2 - len(vox))