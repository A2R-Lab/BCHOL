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

        self.cholinfo = []
        for i in range(numfacts):
            self.cholinfo.append('None') #??

        self.depth = depth
        self.nhorizon = nhorizon
        self.numfacts = numfacts


def ndlqr_GetQFactorizon(cholfacts, index):
    if cholfacts is None:
        return None
    if index < 0 or index >= cholfacts.nhorizon - 1:
        return None

    return cholfacts.cholinfo[2 * index]


def ndlqr_GetRFactorizon(cholfacts, index):
    if cholfacts is None:
        return None
    if index < 0 or index >= cholfacts.nhorizon - 1:
        return None

    return cholfacts.cholinfo[2 * index]


def ndlqr_GetSFactorization(cholfacts, leaf, level):
    numleaves = math.pow(cholfacts.depth - level - 1, 2)
    num_leaf_factors = 2 * cholfacts.nhorizon

    if cholfacts is None:
        return None
    if level < 0 or level >= cholfacts.depth:
        return None
    if leaf < 0 or leaf >= numleaves:
        return None

    leaf_index = 0

    for lvl in range(level):
        numleaves = math.pow(cholfacts.depth - lvl - 1, 2)
        leaf_index += numleaves

    leaf_index += leaf

    return cholfacts.cholinfo[num_leaf_factors + leaf_index]