#! /usr/bin/env python

from scipy.spatial import cKDTree
import astropy.io.fits as pf
import numpy as np
maxra = 0
minra = 0
maxdec = 0
mindec = 0
bsize = abs(max(maxra, maxdec) - min(mindec, minra))


def galaxy_positions():
    source = '/home/pranav/masters_code/'
    hdulist1 = pf.open(source+'/kids_data/KiDS_DR3.1_G9_ugri_shear.fits')
    # hdulist2 = pf.open('../kids_data/KiDS_DR3.1_G12_ugri_shear.fits')
    # hdulist3 = pf.open('../kids_data/KiDS_DR3.1_G15_ugri_shear.fits')
    # hdulist4 = pf.open('../kids_data/KiDS_DR3.1_G23_ugri_shear.fits')
    # hdulist5 = pf.open('../kids_data/KiDS_DR3.1_GS_ugri_shear.fits')
    ra = hdulist1[1].data['RAJ2000']
    dec = hdulist1[1].data['DECJ2000']
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
    return np.column_stack([ra, dec])


def random_positions(mini, maxi):
    x_cord = (maxi - mini)*np.random.random(SIZE) + mini
    y_cord = (maxi - mini)*np.random.random(SIZE) + mini
    return np.column_stack([x_cord, y_cord])


coords = galaxy_positions()
print(maxra, maxdec, minra, mindec)
SIZE = len(coords)
ctree = cKDTree(coords)
nbh = cKDTree.sparse_distance_matrix(ctree,
                                     ctree,
                                     max_distance=200)
# ,output_type='ndarray',p=2)
print(nbh.shape)
print(nbh)
