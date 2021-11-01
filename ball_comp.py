'''
BallComp is an approximator class, similar to KMeans,
It is designed to use less resources than a typical
KMeans algorithm to provide approximation values.
'''
from ball_comp_components import *

def ball_comp_score_function_1(ballSet):
    return -1

## TODO: refactor this ####################################
'''
description:
- calculates label for point based on minumum distance

return:
- label, point distance
'''
def label_for_point__min_distance(point, centroids):
    dist = [round(ndim_point_distance(point, c),5) for c in centroids]
    dist = np.array(dist)
    minumum = round(np.min(dist),5)

    indices = np.argwhere(dist == minumum).flatten()

    # choose random in case of tie-breaker
    return np.random.choice(indices), minumum

## END TODO: ##############################################

'''
u := unique area
t := total area
p := potential area
'''
def dfunc1(u,t,p):
    return (float(u) / t + (1.0 - float(t) / p))  / 2.0

'''
u := unique area
t := total area
p := potential area
b1 := existing number of balls
b2 := total number of balls
'''
def dfunc2(u,t,p,b1,b2):
    return (float(u) / t + (1.0 - float(t) / p) +\
        (1.0 - float(b1)/ b2)) / 3.0

'''
'''
class BallComp:

    MIN_LENGTH_RATIO = 0.01

    def __init__(self, maxBalls,maxRadius,k):
        assert maxRadius > 0.0 and maxBalls > 0, "invalid args. for BallComp"
        self.maxBalls = maxBalls
        self.maxRadius = maxRadius
        self.k = k
        self.balls = [] # each element is Ball

        # potentialSpace is staticvar
        self.at,self.au,self.ap = None,None,self.potential_space()
        self.updateCache =

    def new_ball(self,p):
        b = Ball.one_ball(p,[p])
        b.radius = BallComp.MIN_LENGTH_RATIO * self.maxRadius
        b.radiusDelta = (np.copy(p),b.radius)
        return b

    def potential_space(self):
        return ball_area(self.maxRadius,self.k) * self.maxBalls

    def potential_net_ball_space(self):
        return sum([b.area() for b in self.balls])

    def add_point(self, p):

        # case: make new ball, maxBalls not filled yet
        if len(self.balls) < self.maxBalls:
            b = self.new_ball(p)
            self.balls.append(b)
            return

        #
        return -1

    ####################### START: ball neighbor methods #####

    '''
    '''
    def update_ball_neighbors(self):
        for i in range(len(self.balls)):
            self.update_neighbors_of_ball(i)
        return

    '''
    '''
    def update_neighbors_of_ball(self,i):
        # delete all previous neighbors' pointers
        self.change_neighbor_ptrs(i,-1)
        # update neighbors of ball
        self.balls[i].neighbors = self.neighbors_of_ball(i)
        # update neighbor pointers
        self.change_neighbor_ptrs(i,1)
        return

    def change_neighbor_ptrs(self,i,change):
        assert change in [-1,1], "invalid change {}".format(change)

        for j in range(len(self.balls)):
            if i == j: continue
            if change == -1:
                self.balls[j].neighbors.remove(i)
            else:
                self.balls[j].neighbors.add(i)
        return

    '''
    '''
    def neighbors_of_ball(self,i):
        b = self.balls[i]
        n = set()
        for (j,b_) in enumerate(self.balls):
            if j == i: continue
            if b.is_neighbor(b_): n.add(j)
        return n

    ####################### END: ball neighbor methods #####

    ####################### START: decision scores #####

    def move_it(self,p):
        d = self.new_point_decision(p)
        self.conduct_decision(p,d)
        return

    # TODO: code this
    def check_termination(self, score):
        return -1

    '''
    return:
    '''
    def new_point_decision(self,p):
        d1 = self.decision_1_score(p)
        d2 = self.decision_2_score(p)

        if d1[1] < d2:
            return d1
        return (2, d2)

    '''
    '''
    def conduct_decision(self, p, decision):
        assert decision[0] in [True,False,2]

        # decision 1
        if decision[0] in [True,False]:
            l = self.ball_label_for_point(p)
            #
            b = self.balls[l]
            b.add_point(p)
            self.update_neighbors_of_ball(l)

            # merge balls
            if decision[0] == True:

                obs = list(b.neighbors | {l})
                obs_ = set([self.balls[o] for o in obs])
                newB = self.merge_ball_set(obs_)

                # iterate through and delete all
                self.delete_balls(obs)
                self.balls.append(newB)
                self.update_neighbors_after_merge(len(self.balls) - 1, set(obs))
                return

        # make new ball
        else:
            self.add_new_ball(p)
            return
        return

    def add_new_ball(self,p):
        b = self.new_ball(p)
        l = len(self.balls)
        self.balls.append(b)
        self.update_neighbors_of_ball(l)

    def delete_balls(self, ballIndices):
        q = [b for (i,b) in enumerate(self.balls) if not (i in ballIndices)]
        self.balls = []
        self.balls = q

    '''
    iterates through each ball's neighbors and replaces `deletedBallIndices` w/
    `newBallIndex`

    args:
    -
    '''
    def update_neighbors_after_merge(self, newBallIndex, deletedBallIndices):
        assert type(newBallIndex) is int, "invalid new ball index"
        assert type(deletedBallIndices) is set, "invalid deleted ball indices"

        def index_func(i):
            b = self.balls[i]
            q = b.neighbors -  deletedBallIndices
            if len(q) == len(b.neighbors):
                return
            b.neighbors |= {newBallIndex}
            return

        for j in range(len(self.balls)):
            index_func(j)
        return

    ##
    '''
    calculates a score to add point to present ball

    return:
    - (bool::(merge neighbors), float::(score))
    '''
    def decision_1_score(self,p):

        # old score
            # determine ball to add to
        l = self.ball_label_for_point(p)

            # add point to ball
        b = self.balls[l]
        b_ = deepcopy(b)
        b.add_point(p)

            # determine neighbors of ball before adding point
        n0 = set(b.neighbors)
        self.update_neighbors_of_ball(l)

            # merge all neighbors into bx, make {bx} the newBs
        n1 = set(b.neighbors)

            # determine area delta from modification
        # decision for non-uniting balls
        oldBs = self.dataless_ball_copies(n0)
        oldBs = set([self.balls[n0_] for n0_ in n0]) + {b_}
        newBs = [set([self.balls[n1_] for n1_ in n1])] + {b}
        d = BallComp.unique_area_delta_from_ballset_mod(oldBs,newBs)
            # apply variables `at`,`ap`,`au` for score
        au_ = self.au + d
        df1 = dfunc1(au_,self.at,self.ap)

        # decision for merging balls
            # try mergin
        newBs1 = [self.merge_ball_set(list(oldBs))]
        d_ = BallComp.unique_area_delta_from_ballset_mod(newBs,newBs1)
        au_ = self.au + d_
        df2 = dfunc1(au_,self.at,self.ap)

        # revert all changes made involving ball l
        b.revert_delta()
        b.update_neighbors_of_ball()

        if df1 < df2:
            return (False,df1)
        return (True,df2)

    '''
    return:
    - set(Balls)
    '''
    def dataless_ball_copies(self,indices):
        return set([self.b[i].dataless_copy() for i in indices])

    '''
    calculates a score to make new ball for point
    '''
    def decision_2_score(self,p):

        # make new ball for p
        b = self.new_ball(p)
        self.balls.append(b)
        i = len(self.balls) - 1
        self.update_neighbors_of_ball(i)

        # calculate new au and at
            # case: b has neighbors
        if len(b.neighbors) > 0:
            # modify au
            obs = self.dataless_ball_copies(b.neighbors)
            nbs = obs + {b}
            aud = self.unique_area_delta_from_ballset_mod(obs,nbs)
            au_ = self.au + aud
        else:
            au_ = au + b.area()
        at_ = self.at + b.area()

        # get score
        dfs = dfunc2(u,t,p,b1,len(self.balls) + 1, self.maxBalls)

        # delete b
        self.change_neighbor_ptrs(i, -1)
        self.balls.pop(-1)
        return dfs

    ####################### END: decision scores #####

    '''
    function logs the changes that will occur
    after the addition of p.
    - dict<ball index -> radiusDelta>
    - set<ball int>: indices for balls to be merged.

    return:
    -
    '''
    ###
    """
    def log_add_point_delta(self, p):

        # perform score for adding p to
        # old ball versus new ball
        oldBallScore =
        newBallScore =

        # new score
        '''
            # make new ball for p

            # determine neighbors of ball

            # determine area delta from mod, oldBs := {neighbors}, newBs := {neighbors + ball(p)}

            # apply variables `at`,`ap`,`au` for score
        '''
        b =
        return -1
    """
    ####


    '''
    determines the balls that intersect b after delta
    '''
    ###
    """
    def ball_intersections_from_ball_delta(self,b):
        return -1
    """
    ###

    def update_ball_info(self):

        # case: initial
        if type(self.at) == type(None):
            self.at = self.potential_net_ball_space()
            self.au = BallComp.unique_area_of_ballset_estimation(self.bs)

        # case: update
        else:

        return -1

    '''
    updates `at` and `au`
    '''
    def update_ball_info_cache(self):

        for b in self.balls:
            q1 = b.radiusDelta[]


        return -1

    '''
    determines the ball label for point based on minumum
    euclidean point distance
    '''
    def ball_label_for_point(self, p):

        # TODO: find function that performs distance on all
        #       iterables.
        bc = np.array([euclidean_point_distance(b.center,p) for b in self.balls])

        if len(bc) == 0:
            return -1

        return np.argmin(bc)

    '''
    rounds ball delta so that its radius does not exceed `maxRadius`
    '''
    def round_ball_delta(self):
        return -1

    '''
    determines the label for point based on
    bool::(point in ball)
    '''
    def ball_label_set_for_point(self, p):
        ls = []
        for (i,b) in enumerate(self.balls):
            if b.point_in_ball(p):
                ls.append(i)
        return ls

    def new_point_decision(self,p):

        return -1


    """
    def unique_area_delta_from_ballset_mod_(self,oldBsIndices,newBsIndices):
        obs = set([self.balls[obi for obi in oldBsIndices])

        return -1
    """

    '''
    oldBs := <set::ball>
    newBs := list(<set::ball>)
    '''
    @staticmethod
    def unique_area_delta_from_ballset_mod(oldBs,newBs):
        oldBsA = BallComp.unique_area_of_ballset_estimation(oldBs)
        newBsA = sum([BallComp.unique_area_of_ballset_estimation(newBs_)\
                for newBs_ in newBs])
        return newBsA - oldBsA

    '''
    '''
    @staticmethod
    def unique_area_of_ballset_estimation(bs):

        def args_for_threeway_intersection_estimation(bs_):
            i = argmax([b.area() for b in bs_])
            q = [bs[j] for j in range(len(bs)) if j != i]
            return q[0],q[1] bs[i]

        # determine area of each ball
        a = sum([b.area() for b in bs])

        # determine 2-intersections b/t balls
        pi = BallComp.pairwise_intersections_in_ballset(bs)
        pik = list([vector_to_string(k) for k in pi.keys()])
        a2 = sum([v for v in pi.values()])

        # determine 3-intersections b/t balls
        sm = SetMerger(pik)

            # determine elements of 3-intersections
        x = sm.merge_by_implication()

            # calculate area
        a3 = 0.0
        for x_ in x:
            b1,b2,b3 = args_for_threeway_intersection_estimation(x_)
            a3 += Ball.threeway_area_intersection_estimation(b1,b2,b3)

        return a - a2 + a3

    '''
    determines pairwise intersections in ball-set.

    return:
    - Dict, str::pairs -> area
    '''
    @staticmethod
    def pairwise_intersections_in_ballset(bs):
        pi = defaultdict(int)
        l = len(bs) - 1
        for i in range(l -1):
            for j in range(i + 1,l):
                a = Ball.area_intersection_estimation(bs[i],bs[j])

                # case: no intersection
                if not (a > 0): continue

                # case: intersection
                v = np.array([i,j])
                s = vector_to_string(v,int)
                pi[s] = a
        return pi

    '''
    '''
    def merge_intersections(ballSet,intersectionSet):
        return -1

    '''
    '''
    def merge_two(self):
        return -1

    '''
    '''
    def find_merge_sets(self):
        return -1

    '''
    assumes all balls are neighbors
    '''
    @staticmethod
    def merge_ball_set(bs):

        # sort ball set in ascending distance from ball-set mean
        q = np.array([bs_.center for bs_ in bs])
        m = np.mean(q,axis = 0)
        d = [euclidean_point_distance(bs_.center) for bs_ in bs]
        i = np.argmin(d)
        b_ = bs[i]

        for (b1,j) in enumerate(bs):
            if j == i: continue
            b_ = b_ + b1
        return b_

    '''
    '''
    def merge_ball_set_(self,bsi):
        return -1

    '''
    '''
    def point_in_bounds(self):
        return -1

    ######################
