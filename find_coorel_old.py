
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


