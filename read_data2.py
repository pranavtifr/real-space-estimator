#! /usr/bin/env python
"""Import data and convert to a Healpix map."""
import numpy as np
import healpy as hp
NSIDE = 2048


def make_map(finaldata):
    """
    Make healpix map from data.

    Parameters
    ----------
    finaldata: Numpy array
    The array needs to have
    e1map, e2map, indices, weightmap, temperature,
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
    temperature = np.full(hp.nside2npix(NSIDE), hp.UNSEEN, dtype=np.float)
    shearcalibmap = np.full(hp.nside2npix(NSIDE), hp.UNSEEN, dtype=np.float)
    existance = np.full(hp.nside2npix(NSIDE), False, dtype=np.bool)
    for k in finaldata:
        if temperature[int(k[2])] == hp.UNSEEN:
            temperature[int(k[2])] = 1
            e1map[int(k[2])] = 0
            e2map[int(k[2])] = 0
            weightmap[int(k[2])] = 0
            shearcalibmap[int(k[2])] = 0
            temperature[int(k[2])] = 1
            existance[int(k[2])] = True
        e1map[int(k[2])] += k[0]
        e2map[int(k[2])] += k[1]
        weightmap[int(k[2])] += k[3]
        shearcalibmap[int(k[2])] += k[4]

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
    ra_, decl = np.degrees(hp.pix2ang(NSIDE, index))
    return [90. - ra_, decl]
