'''
this is a re-write of RSSI
'''
from rssi_components import *
# use for adding noise
from numerical_generator import *

DEFAULT_RSSI__CR__NOISE_ADDER = np.array([[0.01,0.15]])

##

### TODO: test this
'''
method used by <ResplattingSearchSpaceIterator>
'''
def update_rch_by_path(rch,largs,updatePath):

    def collect_largs(indices):
        return [l for (i,l) in enumerate(largs) if i in indices]

    for k,v in updatePath.items():
        q = collect_largs(v)
        rch[k].update_args(q)

"""
class<ResplattingSearchSpaceIterator> is a data-structure
that relies on class<SearchSpaceIterator>.

There are two modes for this class when it resplats on
the original bounds:
(1) relevance zoom: requires an RCH to determine relevance of
                    each point
(2) prg: pseudo-random generator
"""

class ResplattingSearchSpaceIterator:

    def __init__(self,bounds, startPoint, columnOrder = None, SSIHop = 7, resplattingMode = ("relevance zoom",None), additionalUpdateArgs = ()):
        assert is_proper_bounds_vector(bounds), "bounds "
        assert is_vector(startPoint), "invalid start point"

        ## TODO: delete
        #assert resplattingMode[0] in {"relevance zoom", "relevance zoom noise", "prg"}
        assert resplattingMode[0] in {"relevance zoom", "prg"}
        assert type(resplattingMode[1]) == RChainHead, "invalid argument for resplatting mode"

        self.bounds = bounds
        self.startPoint = startPoint
        self.referencePoint = np.copy(self.startPoint)
        self.columnOrder = columnOrder
        self.SSIHop = SSIHop
        self.rm = resplattingMode

        self.iteratedOnce = False # if SSI iterated over bounds once
        self.iterateIt = False
        self.terminated = False
        self.rprv = None # relevant point ratio vector
        self.rangeHistory = [np.copy(self.bounds)]

        self.declare_new_ssi(np.copy(self.bounds), np.copy(self.startPoint))

        self.ri = None
        assert type(additionalUpdateArgs) == tuple, "invalid additionalUpdateArgs"
        self.aua = additionalUpdateArgs
        self.update_resplatting_instructor()

        self.ic = [] # iteration cache
        return

    @staticmethod
    def iterate_one_bound(rssi):

        if rssi.terminated:
            yield None
            return

        if not rssi.iteratedOnce:
            while not rssi.iteratedOnce:
                yield next(rssi)
            yield rssi.ssi.close_cycle()
            rssi.iterateIt = False

        else:
            rssi.ssi.referencePoint = rssi.ssi.de_start()

            while not rssi.iterateIt:
                yield next(rssi)
            yield rssi.ssi.close_cycle()
            rssi.iterateIt = False

    '''
    declares either a class<SearchSpaceIterator> or class<SkewedSearchSpaceIterator>
    instance for the given bounds
    '''
    def declare_new_ssi(self,bounds, startPoint):

        print("[d]eclaring new ssi w/ bounds")
        print(bounds)
        print("##$#")

        # if no specified order, default is descending
        if type(self.columnOrder) == type(None):
            self.columnOrder =ResplattingSearchSpaceIterator.column_order(bounds.shape[0],"descending")

        if is_proper_bounds_vector(bounds):
            print("making reg")
            self.ssi = SearchSpaceIterator(bounds, startPoint, self.columnOrder, self.SSIHop,cycleOn = True)
        else: # make SkewedSearchSpaceIterator
            print("making skew")
            self.ssi = SkewedSearchSpaceIterator(bounds,self.bounds,startPoint,self.columnOrder,self.SSIHop,cycleOn = True)

    # CAUTION: only rm[0] == `relevance zoom` has rch update
    def update_resplatting_instructor(self):

        if type(self.ri) == type(None):
            if self.rm[0] == "relevance zoom":
                rz = RZoom(self.rm[1])
                q = (rz,None)
            else:
                cs = CenterResplat(np.copy(self.bounds), self.rm[1], DEFAULT_RSSI__CR__NOISE_ADDER)
                q = (None,cs)

            self.ri = ResplattingInstructor(q[0],q[1])
            return False

        if self.rm[0] == "relevance zoom":
            print("X declaring new")
            nb = self.save_rzoom_bounds_info()
            if type(nb) == type(None): return True

            if self.check_duplicate_range(nb):
                return True

            self.declare_new_ssi(nb,np.copy(nb[:,0]))

            # log point into range history
            self.rangeHistory.append(nb)

            # update rch here
            s = [self.bounds, nb, self.SSIHop] + list(self.aua)
            self.rm[1].load_update_vars(s)
            self.rm[1].update_rch()
            self.ri.rzoom = RZoom(self.rm[1])

        # TODO: optional, add update func. for png here

        return False

    def check_duplicate_range(self,d):
        for d_ in self.rangeHistory:
            if equal_iterables(d_,d): return True
        return False

    '''
    stores activation ranges of rzoom pertaining to iteration of
    the most recent relevant bounds
    '''
    def save_rzoom_bounds_info(self):
        # case: current rzoom a.r. not saved
        if type(self.ri.rzoom.activationRange) != type(None):
            # case: make 0-bounds
            if len(self.ri.rzoom.activationRange.shape) == 1:
                 self.ri.rzoom.activationRange = np.vstack((self.ri.rzoom.activationRange,\
                    self.ri.rzoom.activationRange)).T
            self.ri.rzoom.activationRanges.append(self.ri.rzoom.activationRange)
        print("\tLR")
        self.load_activation_ranges()

        # make the next rzoom
        nb = next(self.ri)

        # case: done
        if type(nb) == type(None): return nb

        # case: fix 0-bounds
        if equal_iterables(nb[:,0],nb[:,1]):
            nb = self.fix_zero_size_activation_range(nb)
        return nb



    '''
    used for rm[0] == "relevance rezoom"
    '''
    ###
    def load_activation_ranges(self):

        # TODO: verbose
        ##print("loading activation ranges ", len(self.ri.rzoom.activationRanges))
        additions = []
        print("LEN ", len(self.ri.rzoom.activationRanges))
        while len(self.ri.rzoom.activationRanges) > 0:
            # pop the activation range
            ar = self.ri.rzoom.activationRanges.pop(0)

            # case: 0-size, modify activation range
            if equal_iterables(ar[:,0],ar[:,1]):
                ar = self.fix_zero_size_activation_range(ar)
            self.ri.rzoomBoundsCache.append(ar)


    """
    new activation range is
    ar[0], midpoint(ar[0],next(ar[0]))
    """
    def fix_zero_size_activation_range(self, ar):
        assert equal_iterables(ar[:,0],ar[:,1]), "not zero-size"

        # save ssi location
        q = self.ssi.de_value()
        #
        self.ssi.set_value(ar[:,0])

        e1 = next(self.ssi)
        self.ssi.rev__next__()
        e2 = self.ssi.rev__next__()
        print("E1 ", e1)
        print("E2 ", e2)
        e3 = np.array([e1,e2]).T

        rv = np.ones((e3.shape[0],)) / 2.0
        p = point_on_improper_bounds_by_ratio_vector(\
            self.bounds,e3,rv)

        e3[:,1] = p
        return e3


    @staticmethod
    def column_order(k,mode = "random"):
        assert mode in ["random","ascending","descending"]
        s = [i for i in range(k)]
        if mode == "random":
            random.shuffle(s)
        elif mode == "descending":
            s = s[::-1]
        return np.array(s)

    def __next__(self):
        return self.get_next()

    '''
    '''
    def get_next(self):
        if self.terminated: return None
        #
        q = self.pre_next()

        # case: prg terminates
        if type(q) == type(None):
            self.terminated = True
            return None

        # log point into ResplattingInstructor
        #$%$
        self.ri.output(q)

        # set a new ssi based on resplatting mode
        self.post_next()
        return q

    def pre_next(self):
        # case: RSSI switched to prg
        if self.iteratedOnce and self.rm[0] == "prg":
            return next(self.ri)
        return next(self.ssi)

    def post_next(self):
        if self.ssi.reached_end():
            self.iteratedOnce = True
            self.iterateIt = True
            self.terminated = self.update_resplatting_instructor()
            # close cycle here.
        else:
            self.iterateIt = False

    def display_range_history(self):
        for (i,r) in enumerate(self.rangeHistory):
            print("{}:\t{}".format(i,r))

    def _summary(self):
        print("BOUND SUMMARY")
        print("parent bounds")
        print(self.bounds)
        print()
        print("start point")
        print(self.startPoint)
        print()
        print("reference point")
        print(self.referencePoint)
        print()

        #### do range history
        print("range history")
        self.display_range_history()

        #### do

def rssi__display_n_bounds(rssi, n):
    for i in range(n):
        print("iterating bound ",i)
        q = ResplattingSearchSpaceIterator.iterate_one_bound(rssi)
        for q_ in q:
            print(q_)
        # summary
        rssi._summary()
        print("\n--------------------------------------------------")
    return -1

###
## def vector_ratio_improper(parentBounds,bounds,point):
###

"""
def fix_zero_size_activation_range(self, ar):
    assert equal_iterables(ar[:,0],ar[:,1]), "not zero-size"

    # save ssi location
    q = self.ssi.de_value()

    #
    self.ssi.set_value(ar[:,0])

    # TODO: below (e1,e2) can be revised to other values
    e1 = next(self.ssi)
    self.ssi.rev__next__()
    e2 = self.ssi.rev__next__()

    return np.array([e1,e2]).T
"""
