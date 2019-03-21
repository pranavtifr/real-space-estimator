#! /usr/bin/env python
"""Compute and Plot Coorelation Functions from FITS file."""
import numpy as np
from read_data import check_sorted, get_rcs, read_parameters_diff_file
from scipy.spatial import cKDTree
mindec = 0
BATCHSIZE = 100


def find_coorelation_fast(tree, maxdist, binsize,
                          parameter1, parameter2, weights=None,
                          cores=1, batchnumber=None):
    """
    Find the coorelation faster using the query() function.

    Parameters
    ----------
    tree: CKDTree
        Coordinates in the CKDTree format
    dist: Float
        Find coorelation of a distance x
    parameter1, parameter2:
        Parameters as Numpy arrays
    weights: Numpy Array
        weights for the pixel value
    cores: int
        Number of cores to use

    Returns
    -------
    Coorelation at a particular point x

    Raises
    ------
    ValueError:
        If the distances from query is not sorted

    See Also
    --------
    find_coorelation()
    find_indices()

    Notes
    -----
    None

    """
    if weights is None:
        weights = np.ones(len(parameter2))

    def temp_func(pair):
        import itertools
        ans = 0
        if pair is None:
            return ans
        for i, j in itertools.combinations_with_replacement(pair, 2):
            ans += parameter1[i]*parameter2[j]*weights[j]
        print(pair)
        return ans/sum(weights[pair])

    itersize = 0
    batchstart = 0
    if not batchnumber:
        batchstart = 0
        itersize = SIZE
        batchend = 0
    else:
        itersize = BATCHSIZE
        batchstart = itersize*(batchnumber-1)
        batchend = itersize*(batchnumber)
    ans = np.zeros(len(np.arange(0, maxdist, binsize)))
    for point in range(batchstart, batchend, 1):
        pairs = find_indices_bin(tree, point, maxdist, binsize)
        tempans = []
        for pair in pairs:
            tempans.append(temp_func(pair))
        ans = ans + np.array(tempans)
    return ans


def find_indices_bin(tree, pointindex, maxdist, binsize, cores=1):
    """
    Find the indices of the tree at a particular distance from a point.

    Parameters
    ----------
    tree: CKDTree
        Coordinates in the CKDTree format
    dist: Float
        Find coorelation of a distance x, This will be the midpoint of
        the bin
    pointindex: int
        Index of the point around which you want the neighbours
    cores: int
        Number of cores to use


    Returns
    -------
    Coorelation at a particular point x

    Raises
    ------
    ValueError:
        If the distances from query is not sorted
    RuntimeWarning:
        If there is a list of size zero

    See Also
    --------
    check_sorted()
    find_coorelation_fast()

    Notes
    -----
    None

    """
    print("Pointindex", pointindex)
    print("Value at pointindex", tree.data[pointindex])
    print("SIZE", SIZE)
    nearbypts = cKDTree.query(tree, tree.data[pointindex], k=int(SIZE),
                              distance_upper_bound=maxdist, n_jobs=cores)
    if not check_sorted(nearbypts[0]):
        raise ValueError

    bins = np.arange(0, maxdist, binsize)
    indices = []
    for ll in range(len(bins)-1):
        tempindices = []
        for kk in range(len(nearbypts[0])):
            if nearbypts[0][kk] > bins[ll] and nearbypts[0][kk] < bins[ll+1]:
                tempindices.append(nearbypts[1][kk])
        print("Tempindices", tempindices)
        indices.append(set(tempindices))
    print("Indices", indices)
    return indices


def find_indices(tree, pointindex, dist, binsize, cores=-1):
    """
    Find the indices of the tree at a particular distance from a point.

    Parameters
    ----------
    tree: CKDTree
        Coordinates in the CKDTree format
    dist: Float
        Find coorelation of a distance x, This will be the midpoint of
        the bin
    pointindex: int
        Index of the point around which you want the neighbours
    cores: int
        Number of cores to use


    Returns
    -------
    Coorelation at a particular point x

    Raises
    ------
    ValueError:
        If the distances from query is not sorted
    RuntimeWarning:
        If there is a list of size zero

    See Also
    --------
    check_sorted()
    find_coorelation_fast()

    Notes
    -----
    None

    """
    upper = dist + binsize/2
    lower = dist - binsize/2
    nearbypts = cKDTree.query(tree, tree.data[pointindex], k=int(SIZE),
                              distance_upper_bound=upper, n_jobs=cores)
    if not check_sorted(nearbypts[0]):
        raise ValueError

    lowfil = False  # Use this to filter out the values < lower

    # Filtering out the indices
    for k in range(len(nearbypts[0])):
        if lower <= nearbypts[0][k]:
            lowfil = k
            break

    upfil = SIZE - 1
    for k in range(len(nearbypts[0])):
        if SIZE <= nearbypts[1][k] and nearbypts[0][k] == np.inf:
            upfil = k
            break
    if not lowfil or upfil == 0:
        return None

    distances = nearbypts[0][lowfil:upfil]
    indices = nearbypts[1][lowfil:upfil]
    if indices.size == 0:
        import warnings
        warnings.warn("List Size Zero", RuntimeWarning)
    return indices, distances


def coorelation_function(tree, parameter1, parameter2, binwidth, maxsize):
    """
    Compute Coorelation function by repeating find_coorelation().

    Compute the coorelation function by using the coorelation computed
    using find_coorelation() for different r

    Parameters
    ----------
    tree: CKD Tree
            The Tree Structure containing the Galaxy Coordinates
    parameter1: Numpy Array
    parameter2: Numpy Array
            The Parameters read using read_parameters()
    binwidth: Float
            Binwidth when trying to compute the coorelation
    maxsize:
            Max size upto which the coorelation is computed

    Returns
    -------
    coorel: Numpy Array
            The computed coorelation function as an array
    Raises
    ------
    None

    See Also
    --------
    find_coorelation()
    plot_coorel()

    """
    coorel = []
    for dist in np.arange(0, maxsize, binwidth):
        # ans = find_coorelation(tree, dist, parameter1, parameter2)
        ans = find_coorelation_fast(tree, dist, binwidth, parameter1,
                                    parameter2)
        if len(coorel) == 0:
            coorel.append(ans)
        else:
            coorel.append(ans)  # - coorel[len(coorel) - 1])
    return coorel


def plot_coorel(coorel, binwidth, maxsize):
    """
    Plot the Coorelation functions.

    Parameters
    ----------
    coorel: Numpy Array
            Array containing xi(r) for various r computed using
            coorelation_function()
    binwidth: Float
            Binwidth used for computing th coorelation function
    maxsize: Float
            Maxsize used for computing the coorelation function
    To be used with the same parameters given to coorelation_function()

    Returns
    -------
    None It Plots, Saves the fig, Shows and Exits
    Raises
    ------
    None

    See Also
    --------
    coorelation_function()
    find_coorelation()

    """
    import matplotlib.pyplot as plt
    plt.plot(np.arange(0, maxsize, binwidth), coorel)
    plt.savefig("plot_fast_rcs.png")
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", "--number", help="Batch Number", type=int)
    args = parser.parse_args()
    BATCHNUMBER = args.number
    import time
    binsize = 100
    maxxx = 3.
    binnn = maxxx/binsize
    print("Read galaxies")
    start = time.time()
    # ctree = galaxy_positions()
    ctree, param1, param2, weights = get_rcs()
    print("Time taken to read galaxies", time.time() - start)
    print(ctree.data)
    print("Read Parameters")
    start = time.time()
    # param1, param2, weights = read_parameters()
    param3 = read_parameters_diff_file(ctree.data)
    print("Time taken to read parameters", time.time() - start)
    print("Testing find_coorelation_fast")
    start = time.time()
    from read_data import SIZE
    corel = find_coorelation_fast(ctree, 5, 0.1, param1, param3,
                                  weights=weights,
                                  cores=1, batchnumber=BATCHNUMBER)
    print("time taken to compute coorelation function", time.time() - start)
    print(corel)
    print("Done")
    print("Calculating the coorel function")
    # start = time.time()
    # coorel = coorelation_function(ctree, param1, param3, binnn, maxxx)
    # print(time.time() - start)
    np.savetxt("coorel_rcs_"+str(BATCHNUMBER)+".csv", corel)
    print(corel)
    # plot_coorel(coorel, binnn, maxxx)

# ans = manual_real_space_estimator(coords, param1, param2)
# plt.plot(ans)
