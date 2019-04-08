#! /usr/bin/env python
"""Import data and convert to a Healpix map."""
import numpy as np
import healpy as hp
from read_data import get_rcs, declratoindex
from scipy.spatial import cKDTree
NSIDE = 1024


def make_finaldata():
    """
    Make the finaldata array needed for make_map().

    Parameters
    ----------
    None

    Returns
    -------
    None

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
    tree, e1, e2, w, m = get_rcs()
    finaldata = np.column_stack([tree.data, e1, e2, w, m])
    return finaldata


def make_map(finaldata):
    """
    Make healpix map from data.

    Parameters
    ----------
    finaldata: Numpy array
    The array needs to have
    coords, e1, e2, weightmap, shearcalibmap
    shearcalibmap in that order

    Returns
    -------
    Healpix Map

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
    e1map = np.full(hp.nside2npix(NSIDE), hp.UNSEEN, dtype=np.float)
    e2map = np.full(hp.nside2npix(NSIDE), hp.UNSEEN, dtype=np.float)
    weightmap = np.full(hp.nside2npix(NSIDE), hp.UNSEEN, dtype=np.float)
    shearcalibmap = np.full(hp.nside2npix(NSIDE), hp.UNSEEN, dtype=np.float)
    existance = np.full(hp.nside2npix(NSIDE), False, dtype=np.bool)
    for k in finaldata:
        index = declratoindex(k[0], k[1], NSIDE)
        if not existance[index]:
            e1map[index] = 0
            e2map[index] = 0
            weightmap[index] = 0
            shearcalibmap[index] = 0
            existance[index] = True
        e1map[index] += k[2]
        e2map[index] += k[3]
        weightmap[index] += k[4]
        shearcalibmap[index] += k[5]

    return e1map, e2map, weightmap, shearcalibmap


def indextodeclra(index):
    """
    Convert index to angles.

    Parameters
    ----------
    index: Int
        Healpix pixel index

    Returns
    -------
    Decl, RA: Float
        Declination and right ascention

    Raises
    ------
    None

    See Also
    --------
    DeclRaToIndex()

    Notes
    -----
    None

    """
    ra, decl = np.degrees(hp.pix2ang(NSIDE, index))
    return [90. - ra, decl]


def make_healpix_coord_tree():
    """
    Make a CKDTree for healpix pixels.

    Parameters
    ----------
    None

    Returns
    -------
    CKDTree

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
    length = hp.nside2npix(NSIDE)
    coords = []
    for i in range(length):
        coords.append(indextodeclra(i))
    ctree = cKDTree(coords)
    return ctree


if __name__ == '__main__':
    print("Making coord data")
    ctree = make_healpix_coord_tree()
    print("Write coords")
    hp.write_map('coordmap_ra.fits', ctree.data[:, 0])
    hp.write_map('coordmap_dec.fits', ctree.data[:, 1])
    del ctree
    print("Reading data")
    e1, e2, w, m = make_map(make_finaldata())
    print("writing data")
    hp.write_map('e1.fits', e1)
    hp.write_map('e2.fits', e2)
    hp.write_map('w.fits', w)
    hp.write_map('m.fits', m)
