from search_space_iterator import *
"""
from collections import defaultdict
from line import *
import random
"""
from relevance_functions import *

########################## start: activation and random selection functions

'''
a sample relevance function to help demonstrate the work of CenterResplat.

return:
- function(vector) ->
        True if all values in (point % modulo) fall within moduloPercentileRange
'''
def relevance_func_2(modulo, moduloPercentileRange):
    assert modulo >= 0.0, "invalid modulo"
    assert is_valid_point(moduloPercentileRange)
    assert min(moduloPercentileRange) >= 0.0 and max(moduloPercentileRange) <= 1.0
    assert moduloPercentileRange[0] <= moduloPercentileRange[1]

    minumum,maximum = moduloPercentileRange[0] * modulo,moduloPercentileRange[1] * modulo

    def f(p):
        p_ = p % modulo
        return np.all(p_ >= minumum) and np.all(p_ <= maximum)

    return f

########################## end: activation and random selection functions

'''
has capabilities of a polynomial-like function constructor
'''
class KDimProjector:

    '''
    s := vector, starting point
    d := float, degree
    h := float, > 0,
    '''
    def __init__(self, s, d, h,axes = None):
        assert is_vector(s), "invalid vector"
        assert type(d) is float and d >= 0.0 and d <= 360.0, "invalid degree"
        assert h > 0.0, "invalid hop"
        self.s = s
        self.d = d
        self.h = h
        self.ax = axes
        self.l = l # vector, delta

    def change_axes(self,ax1):
        assert is_valid_point(ax1),"invalid point"
        self.ax = ax1
        return

    '''
    nd := float, new degree
    '''
    def move_with_delta_degree(self,nd,t = 1.0):
        if type(self.ax) == type(None):
            iss = [i for i in range(len(s))]
            q = random_select_k_unique_from_sequence(iss,2)
        else:
            q = np.copy(self.ax)

        p2 = ResplattingSearchSpaceIterator.point_at_info(self.s,self.d,self.h,q)
        self.l = p2 - self.s
        self.s = self.s + (self.l * t)
        return p2

    # could make a normalized version of this
    # such that p2 - p == d2
    @staticmethod
    def point_at_degree_and_distance(p,d,d2):
        assert is_vector(p)

        '''
        '''
        def value_at_index(refPair, targetV0):
            h_ = hypotenuse_from_point(refPair,d2,d)

            l = Line(np.array([refPair,h_]))

            # determine point at x
            q = l.y_given_x(targetV0,False)
            return q

        # get start
        s = p[:2]
        h = hypotenuse_from_point(s, d2, d)
        rp = np.copy(h)

        for i in range(1,len(p) -1):
            rp = np.copy(p[i:i + 2])
            q = value_at_index(rp, h[-1])
            h.append(q)
        return np.array(h)

    # TODO: inverse here
    @staticmethod
    def point_at_info(p,degree,distance,axes):
        assert is_vector(p), "invalid point"
        assert degree >= 0.0 and degree < 360, "invalid degree"
        assert is_valid_point(axes), "invalid axes"

        # collect the two
        x = p[axes]
        h = hypotenuse_from_point(x,distance,degree)
        p_ = np.copy(p)
        p_[axes] = h
        return p_

    '''
    '''
    def continue_path(self, t):
        delta = t * self.l
        self.s = self.s + delta

'''
RZoom is a map-like class that
- uses a set `rf` of relevance functions to determine if a point
  satisfies.
'''
class RZoom:

    def __init__(self, relevanceFunctions):
        self.rf = relevanceFunctions
        self.activationRanges = [] # each element is a bound
        self.activationRange = None
        return

    def score(self,p):
        return True in [rf(p) for rf in self.rf]

    def output(self,p):
        s = self.score(p)

        # do update on activation ranges
        if s:
            if type(self.activationRange) == type(None):
                self.activationRange = np.copy(p)
            else: # assume size 1
                if len(self.activationRange.shape) != 2:
                    self.activationRange = np.vstack((self.activationRange,p)).T
                else:
                    self.activationRange[:,1] = p
        else:
            if type(self.activationRange) == type(None): return

            # case: 0-size bounds
            if len(self.activationRange.shape) == 1:
                self.activationRange = np.vstack((self.activationRange,\
                                    self.activationRange)).T

            self.activationRanges.append(self.activationRange)
            self.activationRange = None
        return

## TODO: more advanced
'''
class<CenterResplat> declares a <KDimProjector> for each
center.
'''

# TODO:
'''
add relevance function as arg. to center resplat.
'''
class CenterResplat:

    def __init__(self,centers,centerBounds,cfunc):
        assert is_2dmatrix(centers), "invalid center points"
        assert is_proper_bounds_vector(centerBounds), "invalid bounds"
        self.centers = centers
        self.centerBounds = centerBounds
        self.cfunc = cfunc
        self.radius = self.default_radius()

    def output(self,p):
        return self.cfunc(p)

    # TODO: add here.
    def default_radius(self):
        d = euclidean_point_distance(self.centerBounds[:,1],self.centerBounds[:,0])
        p = self.centers.shape[0] * 2
        return d / float(p)

    def __next__(self):

        # choose random center
        i = random.randrange(self.centers.shape[0])
        c = np.copy(self.centers[i])

        # choose random distance from center
        randDist = random.uniform(0.0,self.radius)
        randDeg = random.uniform(0.0,360.0)
        iss = [i for i in range(self.centers.shape[1])]
        randAx = random_select_k_unique_from_sequence(iss,2)
        p2 = KDimProjector.point_at_info(c,randDeg,randDist,randAx)
        h = p2 - c
        return vector_hop_in_bounds(c,h,self.centerBounds)

'''
class provides resplatting instructions for ResplattingSearchSpaceIterator.

If mode is "prg", then will output random point from CenterResplat.
If this option is chosen

'''
class ResplattingInstructor:

    def __init__(self,rzoom,centerResplat):
        self.rzoom = rzoom
        self.rzoomBoundsCache = [] # pop(0), push(-1)
        self.centerResplat = centerResplat
        self.relevantPointsCache = []

    def check_args(self):
        return type(self.rzoom) == type(None) or type(self.centerResplat) == type(None)

    def output(self,p):
        assert self.check_args()
        if type(self.centerResplat) != type(None):
            o = self.centerResplat.output(p)
            if o:
                self.relevantPointsCache.append(p)
        else:
            self.rzoom.output(p)

    '''
    if mode == relevance zoom:
    '''
    def __next__(self):
        assert self.check_args()
        if type(self.centerResplat) != type(None):
            return next(self.centerResplat)
        return self.next_activation_range()

    def next_activation_range(self):
        if len(self.rzoomBoundsCache) == 0: return None
        return self.rzoomBoundsCache.pop(0)
