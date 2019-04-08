#! /usr/bin/env python
"""Compute and Plot Coorelation Functions from FITS file."""
import numpy as np
from read_data import check_sorted, read_parameters_diff_file
from read_data2 import make_healpix_coord_tree, make_map, make_finaldata
from scipy.spatial import cKDTree
mindec = 0
BATCHSIZE = 1000


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
        if len(pair) == 0 or pair is None:
            return ans
        for i, j in itertools.combinations_with_replacement(pair, 2):
            ans += parameter1[i]*parameter2[j]*weights[j]
        return ans/sum(weights[pair])

    def corel_point(pairs):
        ans = np.zeros(len(np.arange(0, maxdist, binsize))-1)
        tempans = []
        for pair in pairs:
            tempans.append(temp_func(pair))  # Try to parallelize this step
        ans = ans + np.array(tempans)
        return ans
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
    allpairs = find_indices_bin(tree, batchstart, batchend, maxdist, binsize)
    print(len(allpairs))
    print("Finding Coorelations")
    start = time.time()
    # param1, param2, weights = read_parameters()
    final_ans = list(map(corel_point, allpairs))
    print("Time taken to find coorelations", time.time() - start)
    print(final_ans)
    return np.sum(np.array(final_ans), axis=0)


def find_indices_bin(tree, batchstart, batchend, maxdist, binsize, cores=1):
    """
    Find the indices of the tree at a particular distance from a point.

    Parameters
    ----------
    tree: CKDTree
        Coordinates in the CKDTree format
    dist: Float
        Find coorelation of a distance x, This will be the midpoint of
        the bin
    batchstart: Int
        starting index of the points of which you want the neighbours
    batchend: Int
        ending index of the points of which you want the neighbours
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
    batchdata = tree.data[batchstart:batchend]
    dist, nearbypts = cKDTree.query(tree, batchdata,
                                    k=int(SIZE),
                                    distance_upper_bound=maxdist, n_jobs=cores)
    print("Batch data queried")

    def point_indices(blah):  # Try to parallize this loop too
        if not check_sorted(dist[blah]):
            raise ValueError

        bins = np.arange(0, maxdist, binsize)
        indices = []
        for ll in range(len(bins)-1):
            tempindices = []
            for kk in range(len(dist[blah])):
                if dist[blah][kk] > bins[ll] and dist[blah][kk] < bins[ll+1]:
                    tempindices.append(nearbypts[blah][kk])
            indices.append(tempindices)
        return indices
    print("Finding Indices")
    start = time.time()
    # param1, param2, weights = read_parameters()
    allindices = list(map(point_indices, range(len(batchdata))))
    print("Time taken to find indices", time.time() - start)
    return allindices  # The binned pair for every point


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
    parser.add_argument("-T", "--threads", help="Number of parallel threads",
                        type=int, default=1)
    args = parser.parse_args()
    BATCHNUMBER = args.number
    THREADS = args.threads
    import time
    binsize = 100
    maxxx = 30.
    binnn = maxxx/binsize
    print("Read galaxies")
    start = time.time()
    # ctree = galaxy_positions()
    param1, param2, weights, shearcalibmap = make_map(make_finaldata())
    ctree = make_healpix_coord_tree()

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
    print("SIZE", SIZE)
    # from concurrent.futures import ProcessPoolExecutor
    # with ProcessPoolExecutor(max_workers=THREADS) as p:
    BATCHSIZE = len(ctree.data)
    corel = find_coorelation_fast(ctree, maxxx, binnn, param1, param3,
                                  cores=THREADS, batchnumber=BATCHNUMBER)
    print("time taken to compute coorelation function", time.time() - start)
    print(corel)
    print("Done")
    print("Calculating the coorel function")
    # start = time.time()
    # coorel = coorelation_function(ctree, param1, param3, binnn, maxxx)
    # print(time.time() - start)
    np.savetxt("coorel_rcs_"+str(BATCHNUMBER)+".csv", corel)
    print(corel)
    plot_coorel(corel, binnn, maxxx)

# ans = manual_real_space_estimator(coords, param1, param2)
# plt.plot(ans)
