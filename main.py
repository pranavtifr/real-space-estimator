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
source = '/home/pranav/masters_code/'
sample = 1000


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


def galaxy_positions(tree=True):
    """
    Read Galaxy Positions.

    Read the Positions of the Galaxy from the FITS file
    and then stack it as coordinates and then arrange
    it into a tree using cKDTree.

    Parameters
    ----------
    tree: Bool
            Send it as a cKDTree

    Returns
    -------
    coords: CKD Tree or List
            Coordinates of the Galaxies on the Tree format or List
            format based on whether tree=True or False

    Raises
    ------
    None

    See Also
    --------
    read_parameters()

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
    if tree:
        ctree = cKDTree(coords)
        return ctree
    else:
        return coords


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
    source = '/home/pranav/masters_code/'
    hdulist1 = pf.open(source+'/kids_data/KiDS_DR3.1_G9_ugri_shear.fits')
    param1 = hdulist1[1].data['e1'][:sample]
    param2 = hdulist1[1].data['e2'][:sample]
    return param1, param2


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
        ans = find_coorelation(tree, dist, parameter1, parameter2)
#        coorel.append(ans)
        if not len(coorel) == 0:
            coorel.append(ans - coorel[len(coorel) - 1])
        else:
            coorel.append(ans)
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
    plt.savefig("plot.png")
    plt.show()


binsize = 100
maxxx = 1.
binnn = maxxx/binsize
ctree = galaxy_positions()
param1, param2 = read_parameters()
param3 = read_parameters_diff_file(galaxy_positions(False))
coorel = coorelation_function(ctree, param1, param3, binnn, maxxx)
print(coorel)
plot_coorel(coorel, binnn, maxxx)


# ans = manual_real_space_estimator(coords, param1, param2)
# plt.plot(ans)
