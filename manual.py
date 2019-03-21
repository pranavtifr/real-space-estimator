import scipy.linalg as LA
import numpy as np
import tqdm as tqdm


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
