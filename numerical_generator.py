'''
(a(X_n) + c) modulo m
'''
import random
from math import ceil
from matrix_methods import *
from variance_works import *
from collections import OrderedDict

# TODO:

####--------------------------------------------------------------------------
#### basic linear congruential generator
'''
linear congruential generator
'''
class LCG:

    def __init__(self, startingInteger, modulo):
        assert type(modulo) is int and modulo > 128, "invalid modulo"
        assert type(startingInteger) is int and startingInteger < modulo, "invalid starting integer"

        self.startingInteger = startingInteger
        self.vInt = startingInteger
        self.modulo = modulo
        self.fetch_arguments()

    # TODO: mod here?
    def fetch_arguments(self):
        self.multiplier = self.fetch_random_integer_in_range((0, self.modulo))
        self.increment = self.fetch_random_integer_in_range((0, self.modulo))

    def fetch_random_integer_in_range(self, ranje):
        assert type(ranje[0]) is int and type(ranje[1]) == type(ranje[0]), "invalid range"
        assert ranje[0] < ranje[1], "invalid range [1]"

        q = random.random()
        c = ceil(ranje[0] + q * (ranje[1] - ranje[0]))
        return int(c)

    def calculate(self, v):
        assert type(v) is int, "invalid value v"
        return int((self.multiplier * v + self.increment) % self.modulo)

    def __next__(self):
        q = self.vInt
        self.vInt = self.calculate(q)
        return q

#############------------------------------------------------------------

class CycleMap:

    '''
    '''
    def __init__(self, cRange:'int'):
        assert type(cRange) is int, "invalid type for cycle range"
        self.cRange = cRange
        self.mahp = OrderedDict()
        self.head = None
        self.phead = None
        return

    def set_map(self, m):
        assert type(m) is OrderedDict, "invalid type for map"
        assert CycleMap.is_valid_map(m), "invalid cycle map"
        assert len(m) == self.cRange, "invalid length for map"
        self.mahp = m

    @staticmethod
    def random_cycle_map(vortexRange):
        ks = np.arange(vortexRange)
        np.random.shuffle(ks)

        # select the head
        h = ks[0]
        h_ = h
        l = OrderedDict()
        rem = np.arange(vortexRange)
        for i in range(vortexRange - 1):
            possible = [r for r in rem if r != h]

            # choose random
            lp = len(possible)
            ri = random.randrange(lp)
            l[h] = possible[ri]

            rem = np.array(possible)
            h = l[h]

        l[h] = h_

        return l

    @staticmethod
    def is_valid_map(m):
        # fetch the first element in the map
        q = None
        for k in m.keys():
            q = k
            break

        # check for cycle
        x = [q]
        l = len(m)
        c = 0
        while c < l:
            r = m[x[-1]]
            x.append(r)
            c += 1

        # check that last element is first key
        if not (x[-1] == x[0]): return False
        # check that number of unique elements is l
        if len(set(x)) != l: return False
        return True

    def head_(self):
        for k,v in self.mahp.items():
            self.head = k
            self.c = k
            break

    def __next__(self):
        if type(self.c) == type(None): self.head_()
        q = self.mahp[self.c]
        self.c = q
        return q

    def v(self,k):
        return self.mahp[k]

    #### TODO: make generators for non-cycles

##########################################################################################

DELTA_ADD = lambda x,c: x + c
DELTA_MULT = lambda x,c: x * c
# TODO: future
DELTA_POLY = lambda x, poly: 1

'''
- rangesGanges: 2-tuple
- divider: 0.0|(float >= 1.0)

constant: 0 or 1
delta: divider n^-1
'''
def FloatDeltaGenerator_pattern1(rangesGanges, divider):
    assert is_valid_point(rangesGanges), "ranges ganges ranges ganges"
    assert divider >= 0.0, "invalid divider"

    # no delta
    if divider != 0.0:
        dividingLengthon = abs(rangesGanges[1] - rangesGanges[0]) / divider
        directororios = 1 if rangesGanges[1] > rangesGanges[0] else -1
        lengthonLengthon = dividingLengthon * directororios
        divider = lengthonLengthon
    return FloatDeltaGenerator(rangesGanges[0], DELTA_ADD, divider, rangesGanges[1])

#------------------------------------------------------------------
#### binary sequence generators

def generate_possible_binary_sequences(vecOrder, thisNumber, elements = [0,1]):

    if len(thisNumber) == vecOrder:
        yield thisNumber
        return

    q1, q2 = np.copy(thisNumber), np.copy(thisNumber)
    q1, q2 = np.hstack((q1,[elements[0]])), np.hstack((q1,[elements[1]]))
    yield from generate_possible_binary_sequences(vecOrder, q1,elements)
    yield from generate_possible_binary_sequences(vecOrder, q2,elements)

def generate_random_binary_sequence(vecOrder):
    assert type(vecOrder) is int, "invalid vec. order"
    return np.random.randint(0, 2, (vecOrder,))

####---------------------------------------------------------------------
#### uniform dist. numerical generators

"""
"""
def generate_uniform_sequence_in_bounds(vecOrder, bounds):
    assert is_2dmatrix(bounds), "invalid bounds {}".format(bounds)
    assert vecOrder == len(bounds) or len(bounds) == 1, "invalid bounds"

    if len(bounds) == 1:
        return rng.uniform(bounds[0,0], bounds[0,1], (vecOrder,))
    else:
        q = np.zeros((vecOrder,))
        for i in range(vecOrder):
            q[i] = rng.uniform(bounds[i,0], bounds[i,1])#, (vecOrder,))
        return q

def random_bounds_edge(bounds):
    assert is_bounds_vector(bounds), "invalid bounds"
    bs = generate_random_binary_sequence(bounds.shape[0])
    r = [i for i in range(bounds.shape[0])]
    return bounds[r,bs]

################################# TODO: noise methods need to be checked.

'''
a method that generates points in bounds by random.random()
'''
def k_random_points_in_bounds(minVec,maxVec,k):
    assert np.all(maxVec - minVec >= 0.0), "invalid arguments min.,max. vec"

    d = maxVec - minVec
    for i in range(k):
        x = random.random()
        yield minVec + (d * x)

###------------------------

"""
noiseRange := 2dmatrix, len is 1 or len(point)
"""
def one_random_noise(bounds,noiseRange):
    assert is_proper_bounds_vector(bounds), "invalid bounds"

    # set max distance for each dim.
    q = bounds[:,1] - bounds[:,0]
    us = generate_uniform_sequence_in_bounds(bounds.shape[0], noiseRange)
    q = q * us
    return q

def random_npoint_from_point(p1,l,roundDepth=5):
    p2 = np.copy(p1)
    r = np.arange(p1.shape[0])
    np.random.shuffle(r)
    r,x = r[:-1],r[-1]
    d = 0.0

    def delta_extreme():
        return math.sqrt(l ** 2 - d)

    def random_delta(j,randomize = True):
        delta = delta_extreme()
        delta = delta if random.random() > 0.5 else -delta

        if randomize:
            delta = random.uniform(0.0,delta)
        return delta

    for r_ in r:
        rd = random_delta(r_)
        p2[r_] = p2[r_] + rd
        d += (rd **2)
    p2[x] = p2[x] + random_delta(x,False)
    return np.round(p2,5)

# TODO: refactor duplicate code from above
"""
calculates a random n-dimensional point p2 in bounds `b` of distance `l`
from `p1`.
"""
def random_npoint_from_point_in_bounds_(b,p1,l,roundDepth = 5):
    p2 = np.copy(p1)
    r = np.arange(p1.shape[0])
    np.random.shuffle(r)
    r,x = r[:-1],r[-1]
    d = 0.0

    def delta_extreme():
        return math.sqrt(l ** 2 - d)

    def possible_extreme(j):
        return [b[j,0] - p1[j],b[j,1] - p1[j]]

    def random_delta(j,randomize = True):
        delta = delta_extreme()
        delta = delta if random.random() > 0.5 else -delta

        if randomize:
            delta = random.uniform(0.0,delta)

        pe = possible_extreme(j)
        pe = pe[1] if delta > 0 else pe[0]
        sols = [pe,delta]
        k = np.argmin(np.abs(sols))
        return sols[k]

    for r_ in r:
        rd = random_delta(r_)
        p2[r_] = p2[r_] + rd
        d += (rd **2)
    p2[x] = p2[x] + random_delta(x,False)
    return np.round(p2,5)

def random_npoint_from_point_in_bounds(b,p1,l,attempts = 15):
    t = False

    while not t and attempts > 0:
        p2 = random_npoint_from_point_in_bounds_(b,p1,l)
        #print("DISTANCE ", euclidean_point_distance(p1,p2))
        t = abs(euclidean_point_distance(p1,p2) - l) < 10 ** -5
        attempts -= 1
    return p2 if t else None

'''
adds noise to points in restricted bounds by noise range,
set to (0.25,0.65)

each output in yield is a point that may not lie within the bounds set by minVec
and maxVec

minVec,maxVec := vectors that do not need to be sorted
'''
def add_noise_to_points_restricted_bounds(minVec,maxVec, points, noiseRange = np.array([[0.25,0.65]]), boundsRestriction = False):

    b = np.array([minVec,maxVec]).T
    for p in points:
        yield p + one_random_noise(b,noiseRange)

def random_noise_sequence(s,b,noiseRange):
    rn = []
    for i in range(s):
        n = one_random_noise(b,noiseRange)
        rn.append(n)
    return np.array(rn)

# TODO:
def generate_gaussian_sequence_in_bounds(mean, var):
    rng.normal()
    return -1
