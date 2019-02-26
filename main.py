#! /usr/bin/env python

import scipy.linalg as LA
import astropy.io.fits as pf
import numpy as np
import tqdm as tqdm
from scipy.spatial import KDTree as cKDTree
SIZE = 0
maxra = 0
minra = 0
maxdec = 0
mindec = 0
bsize = abs(max(maxra, maxdec) - min(mindec, minra))


def galaxy_positions():
    source = '/home/pranav/masters_code/'
    hdulist1 = pf.open(source+'/kids_data/KiDS_DR3.1_G9_ugri_shear.fits')
    '''
    hdulist2 = pf.open('../kids_data/KiDS_DR3.1_G12_ugri_shear.fits')
    hdulist3 = pf.open('../kids_data/KiDS_DR3.1_G15_ugri_shear.fits')
    hdulist4 = pf.open('../kids_data/KiDS_DR3.1_G23_ugri_shear.fits')
    hdulist5 = pf.open('../kids_data/KiDS_DR3.1_GS_ugri_shear.fits')
    '''
    ra = hdulist1[1].data['RAJ2000']
    dec = hdulist1[1].data['DECJ2000']
    '''
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
    '''
    coords = np.column_stack([ra, dec])
    global SIZE
    SIZE = len(coords)
    print(maxra, maxdec, minra, mindec)
    ctree = cKDTree(coords)
    return ctree


def read_parameters():
    source = '/home/pranav/masters_code/'
    hdulist1 = pf.open(source+'/kids_data/KiDS_DR3.1_G9_ugri_shear.fits')
    param1 = hdulist1[1].data['e1']
    param2 = hdulist1[1].data['e2']
    return param1, param2


def random_positions(mini, maxi):
    x_cord = (maxi - mini)*np.random.random(SIZE) + mini
    y_cord = (maxi - mini)*np.random.random(SIZE) + mini
    return np.column_stack([x_cord, y_cord])


def get_index(pos1, pos2, maxdist, bins):
    distance = LA.norm(pos1 - pos2)
    factor = bins/maxdist
    return int(distance*factor)


def manual_real_space_estimator(coords, parameter1, parameter2):
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
    pairs = cKDTree.query_pairs(tree, maxdist)
    ans = 0
    for apair in pairs:
        i, j = apair
        ans += parameter1[i] * parameter2[j]

    return ans


def plot_coorelation_function(tree, parameter1, parameter2, binwidth, maxsize):
    import matplotlib.pyplot as plt
    coorel = []
    for dist in np.arange(0, maxsize, binwidth):
        ans = find_coorelation(tree, maxsize, parameter1, parameter2)
        coorel.append(ans - coorel[len(coorel) - 1])
    plt.plot(np.arange(0, maxsize, binwidth), coorel)
    plt.show()


ctree = galaxy_positions()
param1, param2 = read_parameters()
plot_coorelation_function(ctree, param1, param2, 0.1, 250)
# ans = manual_real_space_estimator(coords, param1, param2)
# plt.plot(ans)
