from rssi_components import *

# use for adding noise
from numerical_generator import *

ITERATION_CACHE_SIZE_LIMIT = 1232#1
DEFAULT_NOISE = np.array([(0.05,0.2)])

# TODO: logging enabled

## objectives
'''
rz
? rz w/ noise

by default autocreate-rf
by custom rchain
'''

class ResplattingSearchSpaceIterator:

    '''
    performs functions that extend that of SearchSpaceIterator,
    such as:

    - pseudo-random generation of point
        * after iteration over bounds, outputs random points
          centered around points of relevance

    - relevance zoom
        * after iteration over a <SearchSpaceIterator> (next and then close cycle),
          loops over a subbound of it based on relevance measures

    -------
    resplattingMode := ("relevance zoom",None|(center points)) |
                       ("prg",function(point)->bool)
    '''
    def __init__(self,bounds, startPoint, columnOrder = None, SSIHop = 7, resplattingMode = ("relevance zoom",None), activationThreshold = None): #,cycleOn = False, cycleIs = 0):
        assert is_proper_bounds_vector(bounds), "bounds "
        assert is_vector(startPoint), "invalid start point"

        ## TODO: delete
        #assert resplattingMode[0] in {"relevance zoom", "relevance zoom noise", "prg"}
        assert resplattingMode[0] in {"relevance zoom", "prg"}


        self.bounds = bounds
        self.startPoint = startPoint
        ##self.startPointRatio = self.startPoint / (self.bounds[1] - self.bounds[0])
        self.referencePoint = np.copy(self.startPoint)
        self.columnOrder = columnOrder
        self.SSIHop = SSIHop
        self.rm = resplattingMode

        # default activation threshold is half the hop length
        if type(activationThreshold) == type(None):
            activationThreshold = 1.0#(self.SSIHop ** -1) / 2.0# * 0.75# 2.0
        assert activationThreshold >= 0.0 and activationThreshold <= 1.0, "invalid activation threshold"
        self.activationThreshold = activationThreshold

        self.iteratedOnce = False # if SSI iterated over bounds once
        self.iterateIt = False
        self.terminated = False

        self.rprv = None # relevant point ratio vector
        self.rangeHistory = [np.copy(self.bounds)]

        self.declare_new_ssi(np.copy(self.bounds), np.copy(self.startPoint))


        self.ri = None

        self.update_resplatting_instructor()
        self.ic = [] # iteration cache
        return

    def check_resplatting_mode(self,rm):
        if rm[0] ==  "relevance zoom":
            assert type(rm[1]) == RChainHead, "invalid type for relevance function"
        else:
            assert is_2dmatrix(rm[1]),"invalid"
            ####
            """
            return -1
            for v in rm[1]:
                assert type(v) == RChainHead,"invalid type for relevance function"
            """
            ####
    ##@!
    '''
    used for rm[0] == "relevance rezoom"
    '''
    def load_activation_ranges(self):

        # TODO: verbose
        ##print("loading activation ranges ", len(self.ri.rzoom.activationRanges))
        additions = []
        while len(self.ri.rzoom.activationRanges) > 0:
            # pop the activation range
            ar = self.ri.rzoom.activationRanges.pop(0)

            # case: 0-size, modify activation range
            if equal_iterables(ar[:,0],ar[:,1]):
                ar = self.fix_zero_size_activation_range(ar)

            self.ri.rzoomBoundsCache.append(ar)

    @staticmethod
    def column_order(k,mode = "random"):
        assert mode in ["random","ascending","descending"]
        s = [i for i in range(k)]
        if mode == "random":
            random.shuffle(s)
        elif mode == "descending":
            s = s[::-1]
        return np.array(s)

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
        print("SUB-BOUND SUMMARY")
        print("bounds")
        print(self.ssi.de_bounds())
        print()
        print("start")
        print(self.ssi.de_start())
        print()
        print("end")
        print(self.ssi.de_end())
        print()

    # TODO: bug, fix this.
    @staticmethod
    def iterate_one_bound(rssi):
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


    @staticmethod
    def iterate_one_batch(rssi,primedRssi = 1):
        if rssi.rm[0] == "relevance zoom":
            return ResplattingSearchSpaceIterator.iterate_one_bound(rssi,primedRssi)

        assert type(primedRssi) is int and primedRssi >= 0, "invalid primed rssi"
        for i in range(primedRssi):
            yield next(rssi)

    ##
    # TODO
    '''
    make duplicate_point_with_noise(numDuplicates)
    '''

    def __next__(self):
        return self.get_next()

    '''
    '''
    def get_next(self):
        if self.terminated: return None
        #
        q = self.pre_next()

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

    def display_stat(self):
        print("\nreference point:\n{}".format(self.ssi.de_value()))
        print("\nendpoint:\n{}".format(self.ssi.de_end()))
        print("\nbounds:\n{}".format(self.ssi.de_bounds()))
        print("-----------")

    ######### start: initialization methods for different modes

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

    '''
    an auto-create method for relevance zoom functions
    '''
    @staticmethod
    def relevance_zoom_functions_autocreate(bounds,k, activationThreshold,partition = None):

        # make the values
        ## TODO: test this
        if type(partition) == type(None):
            cs = ResplattingSearchSpaceIterator.resplat_partition(bounds,k)
        else:
            cs = partition

        # make the zoom functions
        bd = euclidean_point_distance(bounds[:,0],bounds[:,1])

        rfs = []
        for i in range(cs.shape[0]):
            rf = relevance_zoom_func_1(cs[i],bd,activationThreshold)
            rfs.append(rf)
        return rfs



    def add_noise_to_samples(self,b,s):
        ns = random_noise_sequence(s.shape[0],s.shape[1],b,DEFAULT_NOISE)
        for i,(ns_,cs_) in enumerate(zip(ns,s)):
            cs_ = vector_hop_in_bounds(cs_,ns_,b)
            s[i] = cs_
        return s

    """
    stores initial points and relevant info. for `relevance zoom`
    """
    def preproc_relevant_points(self,rp):
        if type(self.rprv) != type(None):
            return

        # store ratio vector data
        self.rprv = []
        for (i,cs_) in enumerate(rp):
            rv = cs_ / (self.bounds[:,1] - self.bounds[:,0])
            self.rprv.append(rv)
        self.rprv = np.array(self.rprv)

    '''
    makes relevance functions and stores the ratio values into rprv
    '''
    def relevance_zoom_functions_autocreate_(self,bounds,k,activationThreshold):

        l = "relevance zoom"
        if self.rm[0][:len(l)] == "relevance zoom":

            # case: make new random samples, initialize rprv
            if type(self.rm[1]) == type(None):
                # default random points at k
                cs = ResplattingSearchSpaceIterator.resplat_partition(bounds,k)
                self.preproc_relevant_points(cs)
            else:
                if type(self.rprv) == type(None):
                    self.preproc_relevant_points(self.rm[1])

                # map the new points from old to new bound by rprv
                nups = []
                for (i,x) in enumerate(self.rprv):
                    nup = point_on_bounds_by_ratio_vector(bounds,x)
                    nups.append(nup)
                cs = np.copy(nups)

            # add noise to each sample
            if len(self.rm[0]) > len(l):
                print("adding noise to samples")
                cs = self.add_noise_to_samples(bounds,cs)

            return ResplattingSearchSpaceIterator.relevance_zoom_functions_autocreate(\
                bounds,k, activationThreshold,cs)

        raise ValueError("RSSI not in mode<relevance zoom>")

    @staticmethod
    def resplat_partition(bounds,k):
        newB = n_partition_for_bound(bounds,k)
        return central_sequence_of_sequence(newB)

    '''
    rfunc := func(<k-dim vector>) -> bool
    '''
    def make_relevance_zoom(self, bounds, ssih):
        rfs = self.relevance_zoom_functions_autocreate_(bounds,ssih,self.activationThreshold)
        return RZoom(rfs)

    # TODO: use this method to generate relevance functions for non-auto mode
    def rf_generator(self,bounds,ssih):
        return -1

    # TODO: test this method
    '''
    '''
    def set_relevance_zoom_functions_by_RCH(self,rchs):

        rfs = []
        for r in rchs:
            assert type(r) is RChainHead, "invalid rch"
            rfs.append(r.apply)

        return RZoom(rfs)



    ###

    def make_prg_centers(self,bounds):
        return ResplattingSearchSpaceIterator.resplat_partition(bounds,self.SSIHop)

    # use SSI, set ar as point,do next and revnext
    """
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

    '''
    call this initially and after each ssi completed

    return:
    - bool::(finished splatting)
    '''
    def update_resplatting_instructor(self):
        # case:
        ##print("[u]pdating ri")
        l = "relevance zoom"
        if type(self.ri) == type(None):
            ##print("\t[u]pdate default")
            if self.rm[0][:len(l)] == l:
                rz = self.make_relevance_zoom(np.copy(self.bounds),self.SSIHop)#,self.rm[1])
                q = (rz,None)
            else:
                cs = self.make_prg_centers(np.copy(self.bounds))
                cs = CenterResplat(cs, np.copy(self.bounds),self.rm[1])
                q = (None,cs)
            self.ri = ResplattingInstructor(q[0],q[1])
            return False

        # case: relevance zoom
        if self.rm[0][:len(l)] == "relevance zoom":
            ## case: append the last activation range of bounds
            if type(self.ri.rzoom.activationRange) != type(None):
                # case: make 0-bounds
                if len(self.ri.rzoom.activationRange.shape) == 1:
                     self.ri.rzoom.activationRange = np.vstack((self.ri.rzoom.activationRange,\
                        self.ri.rzoom.activationRange)).T
                self.ri.rzoom.activationRanges.append(self.ri.rzoom.activationRange)

            self.load_activation_ranges()

            # make the next rzoom
            nb = next(self.ri)

            # case: done
            if type(nb) == type(None): return True

            # case: fix 0-bounds
            if equal_iterables(nb[:,0],nb[:,1]):
                nb = self.fix_zero_size_activation_range(nb)

            # default start point is bound [0]
            sp = np.copy(nb[:,0])

            print("X declaring new")
            self.declare_new_ssi(nb,sp)

            # log point into range history
            self.rangeHistory.append(nb)

            # make the new relevance zoom
            self.ri.rzoom = self.make_relevance_zoom(nb,self.SSIHop)

        # case: center resplat does not need update after initialization
        return False

    ######### end: initialization methods for different modes
