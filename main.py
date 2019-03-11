#! /usr/bin/env python
"""Compute and Plot Coorelation Functions from FITS file."""
import scipy.linalg as LA
import astropy.io.fits as pf
import numpy as np
import tqdm as tqdm
import healpy as hp
from scipy.spatial import cKDTree
SIZE = 0
maxra = 0
minra = 0
maxdec = 0
mindec = 0
bsize = abs(max(maxra, maxdec) - min(mindec, minra))
NSIDE = 2048
source = '/user1/pranav/msc_codes/'
sample = 10000


def get_rcs():
    """
    Get the RCSLens Data.

    Parameters
    ----------
    None

    Returns
    -------
    coords: CKD Tree
            Coordinates of the Galaxies on the CKDTree format
    params: As Numpy arrays

    Raises
    ------
    None

    See Also
    --------
    galaxy_positions()

    Notes
    -----
    None

    """
    kk = np.loadtxt("rcslens.csv", delimiter=",",
                    usecols=(1, 2, 3, 4, 5), skiprows=1)
    global maxra
    maxra = max(kk[:, 0])
    global minra
    minra = min(kk[:, 0])
    global maxdec
    maxdec = max(kk[:, 1])
    global mindec
    mindec = min(kk[:, 1])
    global bsize
    bsize = abs(max(maxra, maxdec) - min(mindec, minra))
    coords = np.column_stack([kk[:, 0], kk[:, 1]])
    global SIZE
    SIZE = len(coords)
    print(maxra, maxdec, minra, mindec, SIZE)
    ctree = cKDTree(coords)
    # gamma_shear = -k[:,2]*np.cos
    return ctree, kk[:, 2]


def check_sorted(thelist):
    """
    Check if the array is sorted.

    Parameters
    ----------
    thelist: List of Numbers

    Returns
    -------
    Bool:
        True if sorted, False if not

    Raises
    ------
    None

    See Also
    --------
    None

    Notes
    -----
    None

    """
    it = iter(thelist)
    next(it, None)
    return all(b >= a for a, b in zip(thelist, it))


def DeclRaToIndex(decl, RA):
    """
    Return the corresponding index in the Healpy array.

    Parameters
    ----------
    decl: Float
            Declination
    RA: Float
            Right Ascention
    Returns
    -------
    index: Int
            The index in the Healpy Array
    Raises
    ------
    None
    See Also
    --------
    None

    """
    return hp.pixelfunc.ang2pix(NSIDE, np.radians(90. - decl), np.radians(RA))


def galaxy_positions():
    """
    Read Galaxy Positions.

    Read the Positions of the Galaxy from the FITS file
    and then stack it as coordinates and then arrange
    it into a tree using cKDTree.

    Parameters
    ----------
    None

    Returns
    -------
    coords: CKD Tree
            Coordinates of the Galaxies on the CKDTree format

    Raises
    ------
    None

    See Also
    --------
    read_parameters()

    Notes
    -----
    It also sets the global coordinates such as maxdec, mindec, maxra,
    minra, bsize and SIZE.

    """
    hdulist1 = pf.open(source+'/kids_data/KiDS_DR3.1_G9_ugri_shear.fits')
    '''
    hdulist2 = pf.open('../kids_data/KiDS_DR3.1_G12_ugri_shear.fits')
    hdulist3 = pf.open('../kids_data/KiDS_DR3.1_G15_ugri_shear.fits')
    hdulist4 = pf.open('../kids_data/KiDS_DR3.1_G23_ugri_shear.fits')
    hdulist5 = pf.open('../kids_data/KiDS_DR3.1_GS_ugri_shear.fits')
    '''
    ra = hdulist1[1].data['RAJ2000'][:sample]
    dec = hdulist1[1].data['DECJ2000'][:sample]
    global maxra
    maxra = max(ra)
    global minra
    minra = min(ra)
    global maxdec
    maxdec = max(dec)
    global mindec
    mindec = min(dec)
    global bsize
    bsize = abs(max(maxra, maxdec) - min(mindec, minra))
    coords = np.column_stack([ra, dec])
    global SIZE
    SIZE = len(coords)
    print(maxra, maxdec, minra, mindec, SIZE)
    ctree = cKDTree(coords)
    return ctree


def read_parameters_diff_file(coords):
    """
    Read Params from a different FITS file and order it according to coords.

    Parameters
    ----------
    coords: Numpy Array
            Coordinates in the list format
    Returns
    -------
    params: Numpy Array
            Parameters as a Numpy Array
    Raises
    ------
    None
    See Also
    --------
    read_parameters()

    """
    param_map = hp.read_map(source +
                            "kids_data/"
                            "COM_CompMap_Compton-SZMap-milca-"
                            "ymaps_2048_R2.00.fits")
    params = []
    for point in coords:
        ra, dec = point
        index = DeclRaToIndex(dec, ra)
        params.append(param_map[index])
    return params


def decompose_shear(coords, gamma1, gamma2):
    """
    Decompose the shear to tangential components.

    Parameters
    ----------
    coords: Numpy array
        Coordinates of the galaxies
    gamma1: Shear1
    gamma2: Shear2

    Returns
    -------
    tangential shear as a numpy array

    Raises
    ------
    None

    See Also
    --------
    None

    Notes
    -----
    None

    """


def read_parameters():
    """
    Read the Parameters to do the coorelations.

    To be used only on the same file as the coordinates,
    Since order is retained.

    Parameters
    ----------
    None

    Returns
    -------
    param1: Numpy Array
            Array of Parameter 1
    param2: Numpy Array
            Array of Parameter 2
    Raises
    ------
    None

    See Also
    --------
    galaxy_positions()

    """
    hdulist1 = pf.open(source+'/kids_data/KiDS_DR3.1_G9_ugri_shear.fits')
    param1 = hdulist1[1].data['e1'][:sample]
    param2 = hdulist1[1].data['e2'][:sample]
    weights = hdulist1[1].data['weight'][:sample]
    return param1, param2, weights


def random_positions(mini, maxi):
    """
    Create a random bunch of positions within a box.

    Parameters
    ----------
    mini: Float
    maxi: Float
            Minimum and maximum coordinates for the random_positions

    Returns
    -------
    coords: Numpy Array
            A numpy array containing the random coordinates

    Raises
    ------
    None

    See Also
    --------
    None

    """
    x_cord = (maxi - mini)*np.random.random(SIZE) + mini
    y_cord = (maxi - mini)*np.random.random(SIZE) + mini
    return np.column_stack([x_cord, y_cord])


def get_index(pos1, pos2, maxdist, bins):
    """
    Get the index for the binning.

    Parameters
    ----------
    pos1: Numpy Array
          Position of the First Point
    pos2: Numpy Array
          Position of the Second Point
    maxdist: Float
            Maximum Distance for the coorelation function hist
    bins: Float
            Number of Bins

    Returns
    -------
    index: Integer
            Index for the Histogram

    Raises
    ------
    None

    See Also
    --------
    manual_real_space_estimator()


    """
    distance = LA.norm(pos1 - pos2)
    factor = bins/maxdist
    return int(distance*factor)


def find_coorelation_fast(tree, dist, binsize,
                          parameter1, parameter2, weights=None, cores=1):
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

    def temp_func(i):
        ans = 0
        if pairs[i] is None:
            return ans
        for j in pairs[i]:
            ans += parameter1[i]*parameter2[j]*weights[j]
        return ans

    pairs = list(map(lambda point: find_indices(tree, point, dist, binsize),
                     range(SIZE)))
    coorel = list(map(temp_func, range(SIZE)))
    suum = 0
    for i in range(len(pairs)):
        if pairs[i] is None:
            continue
        for j in range(len(pairs[i])):
            suum += weights[j]
    if suum == 0:
        suum = 1
    ans = sum(coorel)
    return ans/suum


def find_indices(tree, pointindex, dist, binsize, cores=1):
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

    indices = nearbypts[1][lowfil:upfil]
    if indices.size == 0:
        import warnings
        warnings.warn("List Size Zero", RuntimeWarning)
    return indices


def manual_real_space_estimator(coords, parameter1, parameter2):
    """
    Do real space estimation by looping over the coordinates.

    Parameters
    ----------
    coords: CKD Tree
            Coordinates from galaxy_positions()
    parameter1: Numpy Array
    parameter2: Numpy Array
            parameters from read_parameters()
    Returns
    -------
    ans: Numpy Array
            The coorelation function as a numpy array

    Raises
    ------
    None

    See Also
    --------
    get_index()

    """
    SIZE = len(coords)
    bins = 10000
    ans = np.zeros((bins))
    for k in tqdm.tqdm(range(SIZE)):
        for l in range(SIZE):
            index = get_index(coords[k], coords[l], 250, bins)
            ans[index] += parameter1[k]*parameter2[l]
    return ans
    '''
    nbh = cKDTree.sparse_distance_matrix(ctree,
                                        ctree,
                                        max_distance=200)
    # ,output_type='ndarray',p=2)
    '''


def find_coorelation(tree, maxdist, parameter1, parameter2):
    """
    Find Coorelation upto r.

    Find Coorelation by summing over all the points are seperated
    upto a distance of maxdist

    Parameters
    ----------
    tree: CKD Tree
            The Tree Structure containing the Galaxy Coordinates
    maxdist: Float
            The max distance upto which the points are seperated
    parameter1: Numpy Array
    parameter2: Numpy Array
            The Parameters read using read_parameters()
    Returns
    -------
    ans: Float
            The computed coorelation upto maxdist
    Raises
    ------
    None

    See Also
    --------
    coorelation_function()
    plot_coorel()

    """
    pairs = cKDTree.query_pairs(tree, maxdist)
    ans = 0
    # print("Before:", ans, maxdist)
    for apair in pairs:
        i, j = apair
        ans += parameter1[i] * parameter2[j]
    # print("After:", ans, len(pairs))
    return ans


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
    plt.savefig("plot_fast.png")
    plt.show()


if __name__ == "__main__":
    import time
    binsize = 100
    maxxx = 1.
    binnn = maxxx/binsize
    print("Read galaxies")
    start = time.time()
    # ctree = galaxy_positions()
    ctree, param1 = get_rcs()
    print(time.time() - start)
    print(ctree.data)
    print("Read Parameters")
    start = time.time()
    param1, param2, weights = read_parameters()
    param3 = read_parameters_diff_file(ctree.data)
    print(time.time() - start)
    print("Testing find_coorelation_fast")
    start = time.time()
    corel = find_coorelation_fast(ctree, 0.5, 0.1, param1, param3,
                                  weights=weights)
    print(time.time() - start)
    print(corel)
    print("Done")
    print("Calculating the coorel function")
    start = time.time()
    coorel = coorelation_function(ctree, param1, param3, binnn, maxxx)
    print(time.time() - start)
    np.savetxt("coorel_kids.csv", coorel)
    print(coorel)
    # plot_coorel(coorel, binnn, maxxx)

# ans = manual_real_space_estimator(coords, param1, param2)
# plt.plot(ans)
