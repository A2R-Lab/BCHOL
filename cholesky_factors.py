import math


class NdLqrCholeskyFactors():
    def __init__(self, depth, nhorizon):
        if depth <= 0:
            return None
        if nhorizon <= 0:
            return None

        num_leaf_factors = 2 * nhorizon
        num_S_factors = 0

        for level in len(depth):
            numleaves = math.pow(depth - level - 1, 2)
            num_S_factors += numleaves

        numfacts = num_leaf_factors + num_S_factors

        self.depth = depth
        self.nhorizon = nhorizon
        self.numfacts = numfacts


# cholfacts is an object of class NdLqrCholeskyFactors
def ndlqr_GetQFactorizon(cholfacts, index):
    if cholfacts is None:
        return -1
    if index < 0 or index >= cholfacts.nhorizon - 1:
        return -1
    # do I need this method?
    return 0


def ndlqr_GetRFactorizon(cholfacts, index):
    if cholfacts is None:
        return -1
    if index < 0 or index >= cholfacts.nhorizon - 1:
        return -1
    # do I need this method?
    return 0


def ndlqr_GetSFactorization(cholfacts, leaf, level):
    numleaves = math.pow(cholfacts.depth - level - 1, 2)
    num_leaf_factors = 2 * cholfacts.nhorizon #?? used for cholinfo

    if cholfacts is None:
        return -1

    if level < 0 or level >= cholfacts.depth:
        return -1

    if leaf < 0 or leaf >= numleaves:
        return -1

    leaf_index = 0

    for lvl in range(level):
        numleaves = math.pow(cholfacts.depth - lvl - 1, 2)
        leaf_index += numleaves

    leaf_index += leaf

    return 0