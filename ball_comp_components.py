from numerical_generator import *
from point_weight_function import * 
from collections import Counter

'''
similar to the frequent item-set mining algorithm, Apriori algorithm.

Conducts merging of sets based on their likeness

'''
class SetMerger:

    '''
    sv := list(set(int))
    '''
    def __init__(self, sv):
        self.sv = sv
        self.iterationMem = []
        self.newSets = []

    def save_new_sets(self):
        self.sv = deepcopy(self.newSets)
        self.newSets = []

    '''
    non-commutative operation b/t two equally-sized sets
    '''
    @staticmethod
    def set_difference_score(s1,s2):
        assert type(s1) == type(s2) and type(s1) is set, "[0] invalid sets"
        assert len(s1) == len(s2), "[1] invalid sets"
        return len(s1 - s2)

    # TODO: test this.
    '''
    a closed implication is a container of sets that satisfy the following:
    (1)  for every pair of elements s1,s2 in s, distance(s1,s2) = d.
    (2) for every unique value v in s, v appears in s a minumum of `fr` times.
    '''
    @staticmethod
    def is_closed_implication_for_merge(s,d = 1, fr = 2):
        l = len(s)

        # case: empty
        if l == 0: return False

        c = Counter(s[l - 1])
        for i in range(l - 1):
            c2 = Counter(s[i])
            c = c + c2
            for j in range(i + 1,l):
                d2 = SetMerger.set_difference_score(s[i],s[j])
                if d2 != d:
                    return False

        x = np.array(list(c.values()))

        # case: fr is None, check for equal freq.
        if type(fr) == type(None) and len(x) > 0:
            return np.all(x == x[0])

        return np.all(x >= fr)

    '''
    '''
    @staticmethod
    def perform_merge(s,d,f = None):
        q = SetMerger.is_closed_implication_for_merge(s,d,f)
        if q:
            x = set()
            for q_ in s:
                x |= q_
            return x
        return None

    '''
    '''
    @staticmethod
    def is_set_at_distance_to_others(s,others,wantedDistance):
        assert type(wantedDistance) is int and wantedDistance >= 0, "invalid wanted distance"
        for o in others:
            d = SetMerger.set_difference_score(s,o)
            if d != wantedDistance: return False
        return True

    def merges_at_index(self,i,d = 1):
        pm = self.possible_merges_at_index(i,d)
        merges = []
        for pm_ in pm:
            pm_.append(self.sv[i])
            pm2 = SetMerger.perform_merge(pm_,d,None)
            if type(pm2) != type(None):
                merges.append(pm2)
        return merges

    def possible_merges_at_index(self,i, d = 1):

        l = len(self.sv)
        if i >= len(self.sv) - 1: return []

        possibleMerges = [] # list(list(set))
        ref = self.sv[i]

        # iterate through the sets and if s == 1,
        # determines which index in possibleMerges to add to
        for j in range(i + 1, l):
            s = SetMerger.set_difference_score(ref,self.sv[j])
            if s != d: continue

            # find the list(set) in possibleMerges to merge to
            if len(possibleMerges) != 0:
                pm = -1
                for (i2,p2) in enumerate(possibleMerges):
                    stat = SetMerger.is_set_at_distance_to_others(ref,p2,d)
                    if stat:
                        p2.append(self.sv[j])
                        pm = 1

                # case: pm == -1
                if pm == -1:
                    possibleMerges.append([self.sv[j]])
            # make a new element to add to possible merges
            else:
                possibleMerges.append([self.sv[j]])
        return possibleMerges

    def merge_by_implication(self):
        l = len(self.sv)
        if l == 0: return

        ## TODO: or delete?
        '''
        # calculate endpoint
        e = l -
        '''
        m = []
        for i in range(l):
            m_ = self.merges_at_index(i)
            m.extend(m_)
        return m

ball_area = lambda r,k: math.pi * float(r) **  k

'''
'''
class Ball:

    def __init__(self, center):
        assert is_vector(center), "invalid center point for Ball"
        self.center = center
        self.data = np.empty((0,self.center.shape[0]))
        self.data = PointSorter(self.data)
        self.radius = 0.0
        self.radialRef = None

        # container that holds the most recent delta
        self.clear_delta() # (point,radius delta)
        self.neighbors = set() # of ball neighbor identifiers
        return

    '''
    '''
    @staticmethod
    def dataless_copy(b):
        b_ = Ball(np.copy(b.center))
        b_.radius = b.radius
        b_.radialRef = np.copy(b.center)
        b_.neighbors = set(b.neighbors)
        return b_

    '''
    '''
    def is_neighbor(self,b):
        m = max([self.radius,b.radius])
        return euclidean_point_distance(self.center,b.center) < m

    def area(self):
        b = ball_area(self.radius, self.center.shape[0])
        return round(b,5)

    @staticmethod
    def one_ball(center, points):
        b = Ball(center)

        for p in points:
            b.add_element(p)
        return b

    def point_in_data(self,p):
        return self.data.vector_exists(p)

    def point_in_ball(self,p):
        ed = euclidean_point_distance(p,self.center)
        return ed <= self.radius

    '''
    adds single point to Ball
    and updates mean
    '''
    def add_element(self, p):
        ##self.data = np.vstack((self.data,p))
        self.data.insert_point(p)

        # update radius
        r = euclidean_point_distance(self.center,p)

        if r > self.radius:
            self.radialRef = np.copy(p)
            self.radius = r
            self.radiusDelta = (p, r - self.radius)

    # TODO:
    '''
    '''
    def revert_delta(self):
        if None in self.radiusDelta:
            return

        self.data.delete_point(self.radiusDelta[0])
        self.radius = self.radius - self.radiusDelta[1]
        return

    def clear_delta(self):
        self.radiusDelta = (None,None)

    '''
    determines if two balls intersect
    '''
    @staticmethod
    def does_intersect(b1,b2):
        epd = euclidean_point_distance(b1.center,b2.center)
        epd = epd - b1.radius - b2.radius
        return epd <= 0

    '''
    upper-bound estimate on the k-dimensional sub-area
    of b2 that intersects b1.

    return:
    - ratio:float, w.r.t. b2 area
    '''
    @staticmethod
    def area_intersection_estimation_(b1, b2):
        d2 = euclidean_point_distance(b2.center,b1.center)
        d3 = d2 - b1.radius + b2.radius

        # case: b2 proper subspace of b1
        if d3 <= 0:
            return 1.0

        # case: b2 not proper subspace
        r = 1.0 - (d3 / (2.0 * b2.radius))
        return r

    '''
    upper-bound estimate on the k-dimensional sub-area of
    intersection b/t b1 and b2.
    '''
    @staticmethod
    def area_intersection_estimation(b1,b2):
        a1 = Ball.area_intersection_estimation_(b1,b2)
        a2 = Ball.area_intersection_estimation_(b2,b1)
        x1 = min([b2.area() * a1, b1.area() * a2])
        return max([0.0,x1])
    '''
    '''
    @staticmethod
    def threeway_area_intersection_estimation(b1,b2,b3):
        a13 = Ball.area_intersection_estimation_(b1,b3)
        a23 = Ball.area_intersection_estimation_(b2,b3)

        # case: no three way intersection
        if a13 < 0 or a23 < 0: return 0.0

        a13_ = Ball.area_intersection_estimation(b1,b3)
        a23_ = Ball.area_intersection_estimation_(b2,b3)

        av = ((a13_ * a23) + (a23_ * a13)) / 2.0
        return av

    '''
    * note: for use w/ k-dim.
    '''
    @staticmethod
    def radial_endpoint_for_point(m,p,r,dir):
        rd = Ball.radial_delta_for_point(m,p,r,dir)
        return m + rd

    @staticmethod
    def radial_delta_for_point(m,p,r,dir):
        assert is_vector(m), "invalid slope"
        assert m.shape[0] == p.shape[0], "slope and point not same shape"
        assert type(r) is float, "invalid type for r"
        assert dir in [1,-1], "invalid dir"

        # get distance of point p
        z = np.zeros((m.shape[0],))
        return float(dir) * r / euclidean_point_distance(m,z)

    '''
    merges two balls
    '''
    def __add__(self, b2):

        # get the radius of each
        r1,r2 = self.radius,b2.radius
        diff = b2.center - self.center
        d1 = Ball.radial_endpoint_for_point(diff, self.center, r1,-1)
        d2 = Ball.radial_endpoint_for_point(diff, b2.center, r2,1)

        # calculate midpoint
        m1 = (d1 + d2) / 2.0

        #
        b3 = Ball(m1)

        # push all points
        dr = np.vstack((self.data,b2.data))
        b3.data = dr
        return b3
