# local-explorer-tools

## Description
Local Explorer Tools is a set of python scripts that has been designed to reveal the local behavior of liquid crystalline polymer systems (LCP or SCLCP) from [**LAMMPS**](https://lammps.sandia.gov) dump files.
LET was developped during my master in the group of Armand Soldera, the [**LPCM**](https://lpcm.recherche.usherbrooke.ca/fr/).

The goal is to be able to separate different poly-domains using clustering techniques. Many tools will be needed in order to do that such as global and local onsager order parameters and RDF.

## Requirements
LET has been developped using python 3.9.0.
The following modules are also used:
 - numpy
 - sklearn
 - sklearn_extra
 - matplotlib
 - xarray
 - os
 - glob
 - gzip
 - re
 - ast
 - pandas
 - scipy

The main scripts also use these although they could be removed if needed by user:
 - time
 - psutil

## Usage
The main_save.py and main_restore.py files are a great way to get to know the basic interface. The objects introduced in the scripts have been built using inheritance so if needed by the user, the trajectory class could be used instead of the cluster one without much change to the front end of the code.
The save/restore feature allows the user to generate a bunch of trajectory and voxels datasets (which takes quite a long time) and then run the clustering multiple times on them without generating those datasets from scratch everytime.

Typically, one would do
```
    python3 src/main_save.py
```
Then once everything is done running
```
    python3 src/main_restore.py
```

Here is a transcript of the arguments available when generating the cluster object. Note that it is a direct copy from the code doc and I should eventually do something more in depth and readable.
```
            path (str, optional): Path to input files. Input files can be either LAMMPS trajectory dumps or previously saved trajectory using the save_trajectory method. Defaults to "./".
            pattern (str, optional): Pattern matching input files. Input files can be either LAMMPS trajectory dumps or previously saved trajectory using the save_trajectory method. Defaults to "ellipsoid.*".
            exclude (list of int or None/False, optional): Types to exclude from the trajectory. Defaults to None.
            vector_patterns (nested list of int, optional): Patterns of types for defining vectors. Each element of the mother list is a vector pattern. Defaults to [[2, 3, 2]].
            restore_trajectory (bool, optional): If True, the input files will be read as a restore of the trajectory class. Those input files need to have been created by the save_trajectory method. Defaults to False.
            updates (bool, optional): If True, prints will update the user of current progress. Defaults to True.
            neighbors (int, optional): Number of neighbors to form voxels. A particle counts in it's own voxel, so you will see voxels of size +1 what you specify here. Defaults to 10.
            restore_locals (bool, optional): If True, the input files will be read as a restore of the local class. Those input files need to have been created by the save_locals method. Defaults to False.
            vector_descriptors (list of str, optional): List of variables to take in from the trajectory._vectors dataset. Defaults to ["cm", "angle"].
            voxel_descriptors (list of str, optional): List of variables to take in from the local._voxels dataset. Defaults to ["cm", "angle"].
            distance_descriptor (bool, optional): Whether or not to take in the distance matrix from local._distance_matrix. Defaults to True.
            director (bool, optional): Whether or not only one xyz component should be taken into account. If false, all xyz components are used. Defaults to False.
            normalization (str, optional): Normalization technique. Choices are: min-max, max, zscores_abs and zscores_std. See methods for more details. Defaults to "max".
```

## Todos
### General
 - Missing some formulas with reference here and there.
 - Documentation in clusters.py and features.py isn't quite finished.
 - Terminal messages to user for coefficients_to_csv, save functions.
 - No documentation on object attributes.
 - Should do check on dataset/dataarray integrity after restore and the class is wholly restored.
 - Terminal messages to user do not erase last character when going to a number with one less decimal than last number.
 - Proper error types
### cluster.py
 - For now iteration over clustering parameter (such as done for clustering coefficients) only works for n_clusters. This should be generalized.
### compute_structure.py
 - Check if voxel_onsager and global_onsager give same result for same set of particles.
### features.py
 - Features could be saved/restored but they are very quick to compute.
 - __check_distance_symmetry
### io_local.py
 - with open() should be used instead of open().
### trajectory.py
 - Should do checks if timesteps repeat and if no vectors are found since both will create errors.
 - Name used for trajectory properties isn't consistent in code.
### voxels.py
 - __compute_distance_matrices isn't uniform with features.py


## Citation
  so far nothing, just cite this github page and/or [contact me](couo2506@usherbrooke.ca).

## References
1. Regarding the systems studied here :
  Cuierrier, É., Ebrahimi, S., Couture, O., Soldera, A. (2021). Simulation of main chain liquid crystalline polymers using a Gay-Berne/Lennard-Jones hybrid model. Computational Materials Science. 186(110041). doi: [10.1016/j.commatsci.2020.110041](https://doi.org/10.1016/j.commatsci.2020.110041)
2. Regarding global order parameter :
  Olivier, Y., Muccioli, L., & Zannoni, C. (2014). Quinquephenyl: The Simplest Rigid-Rod-Like Nematic Liquid Crystal, or is it? An Atomistic Simulation. Chemphyschem, 15(7), 1345-1355. doi: [10.1002/cphc.201301126](10.1002/cphc.201301126)
3. Regarding local order parameter :
  Doi, H., Takahashi, K.Z., Tagashira, K., Fukuda, J. & Aoyagi, T. (2019). Machine learning-aided analysis for complex local structure of liquid crystal polymers. Scientific Reports, 9(1). doi: [10.1038/s41598-019-51238-1](10.1038/s41598-019-51238-1)
4. Regarding radial distribution function :
  [Ovito source code](https://gitlab.com/stuko/ovito)
