#! /usr/bin/env python
"""Import data and convert to healpix map."""
import astropy.io.fits as pf
import healpy as hp
import numpy as np
from scipy.spatial import cKDTree
NSIDE = 2048
sample = 10000000
source = '/home/pranav/masters_code/'
SIZE = 0
maxra = 0
minra = 0
maxdec = 0
mindec = 0


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
    kk = np.loadtxt(source+"/kids_data/rcslens2.csv", delimiter=",",
                    skiprows=1, max_rows=sample)
    global maxra
    maxra = max(kk[:sample, 0])
    global minra
    minra = min(kk[:sample, 0])
    global maxdec
    maxdec = max(kk[:sample, 1])
    global mindec
    mindec = min(kk[:sample, 1])
    global bsize
    bsize = abs(max(maxra, maxdec) - min(mindec, minra))
    coords = np.column_stack([kk[:sample, 0], kk[:sample, 1]])
    global SIZE
    SIZE = len(coords)
    print(maxra, maxdec, minra, mindec, SIZE)
    ctree = cKDTree(coords)
    # gamma_shear = -k[:,2]*np.cos
    return ctree, kk[:sample, 2], kk[:sample,
                                     3], kk[:sample, 4], kk[:sample, 5]


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


def declratoindex(decl, ra, nside=NSIDE):
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
    return hp.pixelfunc.ang2pix(nside, np.radians(90. - decl), np.radians(ra))


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
        index = declratoindex(dec, ra)
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
