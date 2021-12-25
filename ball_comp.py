from ball_volume_estimators import *

def dec1(u,t,p):
    x1 = 1 - u / t
    x2 = t / p
    return (x1 + x2) / 2.0

def dec2(b1,b2):
    return b1/b2

def v1_update():
    return -1

def v2_update():
    return -1


# do violation recommendation
# verbose
# arg. history (max balls, max radius)
class BallComp:

    MIN_LENGTH_RATIO = 0.01

    def __init__(self, maxBalls,maxRadius,k,verbose = False):
        assert maxRadius > 0.0 and maxBalls > 0, "invalid args. for BallComp"
        self.maxBalls = maxBalls
        self.maxRadius = maxRadius
        self.k = k
        self.balls = {} # int:idn -> Ball
        self.pointMem = None # (label of last point added, last point added)
        self.ballNeighborsUpdate = None

        assert verbose in [True,False]
        self.verbose = verbose
        self.dve = DisjunctionVolumeEstimator()
        self.ballCounter = 0

        self.violation = None

    #### start: main function --------------------------------------------------

    """
    """
    def conduct_decision(self,p):
        self.summarize_volume_measures()
        d1 = self.decision_1_score(p)
        d2 = self.decision_2_score()

        # case: make new ball
        if d2 < d1[2]:
            print("-- choose decision 2")
            b = Ball(p,self.ballCounter)
            b.add_element(p)
            b.radius = BallComp.MIN_LENGTH_RATIO * self.maxRadius
            self.ballCounter += 1

            ## TODO: add ball here
            self.add_ball(b)
            print("-------------------------------------------")
            return

        # case: add to present ball
        self.balls[d1[1]].add_element(p)
        print("--- decision 1 ball")
        print(self.balls[d1[1]])
        print("----")
            # update neighbors
        self.update_neighbors_of_ball(d1[1])

            # subcase: no merge
            ## update ball volume and its neighbor 2-intersection volumes
        if d1[0] == 1:
            print("-- choose decision 1-no merge")
            self.update_target_ball_volume(d1[1])
            self.update_target_ball_2int(d1[1])
            print("-------------------------------------------")
            return

            # subcase: merge
            ## merge balls into one
        print("-- choose decision 1-merge")
            ##bs = self.dataless_ball_copies(self.balls[d1[1]].neighbors | {d1[1]})
        x = self.balls[d1[1]].neighbors | {d1[1]}
        bs = [self.balls[x_] for x_ in x]
        ball0 = BallComp.merge_ball_list(bs)
        ball0.idn = int(self.ballCounter)
        self.ballCounter += 1
            ## delete all values in self.balls[d1[1]].neighbros | {d1[1]}
        self.remove_ballset(self.balls[d1[1]].neighbors | {d1[1]})
            ## add ball0
        self.add_ball(ball0)

        print("-------------------------------------------")

    def summarize_volume_measures(self):
        print("\t* pre-decision measures")
        print("---")
        print(self.dve.ballVolumes)
        print("---")
        print(self.dve.d)
        print("\t********")

    #### end: main function --------------------------------------------------

    #### start: method requirements for decision 1
    ########### start: point add functions

    '''
    calculates the new neighbor set N1 of the bl'th ball that used to have the neighbor
    set N0. Then update the `neighbors` variable for all affected neighbors of the
    bl'th ball.
    '''
    def update_neighbors_of_ball(self,idn):

        q = self.balls[idn].neighbors
        self.balls[idn].neighbors = self.neighbors_of_ball(idn)

        self.ballNeighborsUpdate = (idn,q,deepcopy(self.balls[idn].neighbors))
        self.update_ball_neighbors_var(self.ballNeighborsUpdate[0],\
            self.ballNeighborsUpdate[1],self.ballNeighborsUpdate[2])

    '''
    - positive difference set N1 - N0: adds bl to these balls' neighbors.
    - negative difference set N0 - N1: subtracts bl from these balls' neighbors.
    '''
    def update_ball_neighbors_var(self,idn,n0,n1):
        pd = n1 - n0
        nd = n0 - n1
        for p in pd:
            self.balls[p].neighbors = self.balls[p].neighbors | {idn}
        for n in nd:
            self.balls[n].neighbors = self.balls[n].neighbors - {idn}
        return

    def revert_update_neighbors_of_ball(self):
        self.update_ball_neighbors_var(self.ballNeighborsUpdate[0],\
            self.ballNeighborsUpdate[2],self.ballNeighborsUpdate[1])
        self.ballNeighborsUpdate = None
        return

    #@
    def neighbors_of_ball(self,idn):
        b = self.balls[idn]
        n = set()
        for k,v in self.balls.items():
            if idn == k: continue
            if v.is_neighbor(b): n.add(k)
        return n

    """
    """
    def add_point_to_ball(self,p,idn):
        self.balls[idn].add_element(p)
        return

    '''
    determines the ball idn for point based on minumum
    euclidean point distance
    '''
    def ball_label_for_point(self, p):
        bc = np.array([(euclidean_point_distance(b.center,p),b.idn)\
            for b in self.balls.values()])

        if len(bc) == 0:
            return -1
        i = np.argmin(bc[:,0])
        return int(bc[i,1])

    ########### end: point add functions

    '''
    return:
    - list(Balls)
    '''
    def dataless_ball_copies(self,indices):
        return [Ball.dataless_copy(self.balls[i]) for i in indices]

    '''
    assumes all balls are neighbors
    '''
    @staticmethod
    def merge_ball_list(bs):
        if len(bs) == 0: return None

        # sort ball set in ascending distance from ball-set mean
        q = np.array([bs_.center for bs_ in bs])
        m = np.mean(q,axis = 0)
        d = [euclidean_point_distance(bs_.center,m) for bs_ in bs]
        indices = list(np.argsort(d))

        # merge in that order
        i = indices.pop(0)
        b_ = bs[i]

        while len(indices) > 0:
            i = indices.pop(0)
            b_ = b_ + bs[i]
        return b_

    #### end: method requirements for decision 1

    #### start: decision function 1

    def pre_decision_1_(self,p,idn):
        # add point to ball
        self.add_point_to_ball(p,idn)

        # update its neighbors
        if self.verbose:
            print("\t\tprevious neighbors: ",vector_to_string(sorted(self.balls[idn].neighbors)))

        self.update_neighbors_of_ball(idn)

        if self.verbose:
            print("\t\tnew neighbors: ",vector_to_string(sorted(self.balls[idn].neighbors)))

        ##print("BEFORE VOLUME UPDATE")
        ##self.summarize_volume_measures()

        # update target ball volume
        self.update_target_ball_volume(idn)

        # update target ball 2-int
        self.update_target_ball_2int(idn)

        ##print("AFTER VOLUME UPDATE")
        ##self.summarize_volume_measures()

        return

    def update_target_ball_volume(self,idn):
        self.dve.log_ball_volume(self.balls[idn])

    def update_target_ball_2int(self,idn):
        q = self.balls[idn].neighbors
        for q_ in q:
            b2 = self.balls[q_]
            self.dve.log_ball_volume_2intersection(self.balls[idn],b2)
        return

    def post_decision_1_(self,idn):
        # revert all changes made
            # target ball add point
        self.balls[idn].revert_add_point()
            # target ball neighbors
        self.revert_update_neighbors_of_ball()

            # target ball volume and its 2-intersection volumes
        self.dve.revert_cache_delta(1)
        self.dve.revert_cache_delta(2)

    def decision_1_score(self,p):
        bl = self.ball_label_for_point(p)

        if self.verbose:
            print("\tSIM:\n\tdecision 1")
            print("\t\tadding point to: ",bl)

        # case: no balls to choose
        if bl == -1:
            return (1,bl,2.0)

        self.pre_decision_1_(p,bl)

        bs1 = self.balls[bl].neighbors | {bl}
        # simulate 1: no merge
        vu = self.dve.estimate_disjunction_at_target_ball(self.balls[bl].idn)
        vt = np.sum([ball_area(self.balls[i].radius,p.shape[0]) for i in bs1])
        vp = ball_area(self.maxRadius,p.shape[0]) * len(bs1)
        d1 = dec1(vu,vt,vp)
        if self.verbose:
            print("\t\t no-merge volume measures: ",vu,vt,vp)
            print("\t\t no-merge score: ", d1)

        # simulate 2: merge target ball w/ its neighbors
        ballSet = self.dataless_ball_copies(bs1)

        ball0 = BallComp.merge_ball_list(ballSet)
        vu = ball_area(ball0.radius,p.shape[0])
        vt = vu
        vp = ball_area(self.maxRadius,p.shape[0])
        d2 = dec1(vu,vt,vp)
        if self.verbose:
            print("\t\t merge volume measures: ",vu,vt,vp)
            print("\t\t merge score: ", d2)

        # choose the better option
        option = (1,bl,d1) if d1 <= d2 else (2,bl,d2)

        if self.verbose:
            print("\t\t decision 1 option: ",option)

        # revert changes
        self.post_decision_1_(bl)
        return option

    #### end: decision function 1

    #### start: decision function 2

    def decision_2_score(self):
        score = dec2(len(self.balls) + 1, self.maxBalls)

        if self.verbose:
            print("\tdecision 2 option:\n\t\tballs {} max balls {} score {}".format(len(self.balls),self.maxBalls,score))
        return score

    #### end: decision function 2

    #### start: decision 1 merge- ballset removal from neighbors

    def remove_ballset(self,idns):

        s = set()
        # get affected neighbors from ballset removal
        for idn in idns:
            s = s | self.balls[idn].neighbors

            # delete ball
            del self.balls[idn]

        # filter out all balls found in idns
        s = s - idns

        # remove labels of ballset from affected neighbors
        self.delete_balls_from_affected_neighbors(idns,s)

        # remove all volume and intersection values in `dve` that contain
        # any label
        self.dve.delete_keyset(idns)

        return

    def delete_balls_from_affected_neighbors(self,bs,neighbors):
        for n in neighbors:
            self.balls[n].neighbors = self.balls[n] - bs
        return

    #### start: adjustment

    def add_ball(self,b):

        print("\t * adding ball ", b.idn)

        self.balls[b.idn] = b
        self.update_neighbors_of_ball(b.idn)
        self.update_target_ball_volume(b.idn)
        self.update_target_ball_2int(b.idn)
        self.dve.clear_cache()
        return

    def check_v1(self):
        return -1

    def check_v2(self):

        return -1

    def violation_recommendation(self,t,p):
        return -1

    def load_recommendation(self):
        return -1

    ####

"""
test data:

- n points spaced far apart
- '           '   near
- ball clump data generator
"""
