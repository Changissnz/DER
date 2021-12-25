from ball_comp_components import *
from set_merger import *
#from itertools import combination

# move volume methods from ball_comp_components to this file

def estimate_n_intersection(intersectionAreas):
    return min(intersectionAreas) * 0.5

"""
BallComp will update each Ball's volume after change
"""
class DisjunctionVolumeEstimator:

    def __init__(self):

        # single-ball volumes: int->float
        self.ballVolumes = {}

        # all 2-intersection volumes
        self.d = {}

        # ball-set intersection volumes: str->float
        self.bv = {}

        self.cache1 = [] # single-ball volume
        self.cache2 = [] # 2-intersection volume

    def log_ball_volume(self,b1):
        prev = self.ballVolumes[b1.idn] if b1.idn in self.ballVolumes else None
        self.ballVolumes[b1.idn] = ball_area(b1.radius,b1.center.shape[0])
        self.cache1.append([b1.idn,prev])

    def log_ball_volume_2intersection(self,b1,b2,updateValue=True):
        k = vector_to_string(sorted([b1.idn,b2.idn]))

        # case: do not update
        if not updateValue and k in self.d:
            return

        # log previous value into cache
        x = None if k not in self.d else self.d[k]
        self.cache2.append([k,x])

        est = volume_2intersection_estimate(b1,b2)
        self.d[k] = est

    def clear_cache(self):
        self.cache1 = []
        self.cache2 = []
        return

    def revert_cache_delta(self,cacheId):
        if cacheId == 1:
            c = self.cache1
            d = self.ballVolumes
        else:
            c = self.cache2
            d = self.d

        while len(c) > 0:
            p = c.pop(0)
            if type(p[1]) == type(None):
                del d[p[0]]
            else:
                d[p[0]] = p[1]

    def revert_cache2_delta(self):
        while len(self.cache2) > 0:
            p = self.cache2.pop(0)
            if type(p[1]) == type(None):
                del self.d[p[0]]
            else:
                self.d[p[0]] = p[1]

    def two_intersection_ball_volume(self,k):
        if k not in self.d: return None
        return self.d[k]

    def target_ball_neighbors(self,bIdn):
        s = set()
        for x in self.d.keys():
            q = string_to_vector(x)
            if bIdn in q:
                s = s | {q[0] if q[0] != bIdn else q[1]}
        return s

    def relevant_2intersections_for_ballset(self,bs):
        twoIntersections = []
        for x in self.d.keys():
            q = string_to_vector(x)
            if q[0] in bs and q[1] in bs:
                twoIntersections.append(set(q))
        return twoIntersections

    def estimate_disjunction_at_target_ball(self,bIdn):
        # get 1-intersection volume
        tan = self.target_ball_neighbors(bIdn) | {bIdn}
        q = sum([self.ballVolumes[k] for k in tan])

        # get 2-intersection volumes
        ti = self.relevant_2intersections_for_ballset(tan)
        # case: no intersections
        if len(ti) == 0:
            return q

        # minus two-intersection volumes
        q2 = np.sum([self.d[vector_to_string(sorted(t))] for t in ti])
        q -= (q2 * 2)
        j = 3
        c = 1.0
        self.sm = SetMerger(ti)

        # alternately add and minus j'th intersection volumes
        while True:
            # estimate the j'th disjunction value
            a = self.estimate_disjunction_at_target_ball_()
            if a == 0.0:
                break
            q += (a * j * c)

            # increment the coefficients
            c = -1 * c
            j += 1
        return q

    """
    Performs a `SetMerger.merge_one` operation and estimate volumes of new
    intersection sets.
    """
    def estimate_disjunction_at_target_ball_(self):
        # merge one and collect the new merges and their predecessors
        r1,r2 = self.sm.merge_one(True,True)

        if len(r1) == 0:
            return 0.0

        # calculate the intersection estimate of each predecessor
        vs = []
        for r in r2:
            iv = self.estimate_int_value(r)
            vs.append(iv)

        self.bv = {}
        q = 0.0
        for (r1_,vs_) in zip(r1,vs):
            k = vector_to_string(ordered(r1_))
            self.bv[k] = vs_
            q += vs_
        return q

    """
    Calculates the intersection-related value for disjunction estimation

    iSet := list(set), closed implication
    """
    def estimate_int_value(self,iSet):

        # collect volumes
        v = []
        for x in iSet:
            q = vector_to_string(sorted(x))
            v.append(self.bv[q])
        return estimate_n_intersection(v)

    """
    iterates through keyset and deletes all keys found in keyset from
    `ballVolumes` and `bv`
    """
    def delete_keyset(self,keySet):

        # delete ball volume
        for k in keySet:
            del self.ballVolumes[k]

        # delete 2-intersections
        ks = list(self.bv.keys())

        for k in ks:
            q = string_to_vector(k)
            if q[0] in keySet or q[1] in keySet:
                del self.bv[k]

        return
