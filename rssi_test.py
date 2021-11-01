from rssi_test_cases import *
import unittest
from unittest.mock import MagicMock

class TestResplattingSearchSpaceIteratorClass(unittest.TestCase):

    ##def test__ResplattingSearchSpaceIterator__
    #TODO: clean this up
    """
    - prg
    """
    def test__ResplattingSearchSpaceIterator__sample_2(self):
        rssi = sample_rssi_2()

        print("* DISPLAY TEST FOR SAMPLE 2")
        # iterate until ssi has iterated once
        # 9 ** 5 = 59049
        print("ITERATING CYCLE")
        i = 0
        while not rssi.iteratedOnce:
            print(i, "\t",next(rssi))
            i += 1
            ##next(rssi)

        print("ITERATING BATCH")
        b = ResplattingSearchSpaceIterator.iterate_one_batch(rssi,50)
        for b_ in b:
            print(b_)

    """
    iterate through bounds once and display relevant bounds
    reference points are default.
    """
    def test__relevance_zoom_func_1_2(self):

        q = sample_rssi_args_2()
        columnOrder = np.arange(q["bounds"].shape[0])[::-1]

        # case: activation is 1/6
        """
        ssi = SearchSpaceIterator(q["bounds"], np.copy(q["bounds"][:,0]),columnOrder,\
            SSIHop = 0.5,cycleOn = True, cycleIs = 0)
        """

        act = 0.5#round(1/6,5) #0.5

        rssi = ResplattingSearchSpaceIterator(q["bounds"],\
            np.copy(q["bounds"][:,0]), columnOrder = None, SSIHop = 3,\
                resplattingMode = ("relevance zoom",None), activationThreshold = act)

        while not rssi.iteratedOnce:
            q = next(rssi)

        # display rzoom bounds cache
        print("displaying {} activation ranges ".format(len(rssi.ri.rzoomBoundsCache)))
        for (i,z) in enumerate(rssi.ri.rzoomBoundsCache):
            print("{} : {}".format(i,z))

        c = check_rssi_bounds_nonintersecting(rssi, count = True)
        print("number of intersections: ",c)
        next(rssi)
        print("NEXT BOUNDS")
        print(rssi.ssi.de_bounds())
        print()
        while not rssi.iterateIt and not rssi.terminated:
            q = next(rssi)
            print(".")

        next(rssi)
        print("NEXT BOUNDS")
        print(rssi.ssi.de_bounds())
        print()

        # display rzoom bounds cache

        print("[1] displaying activation ranges")
        for (i,z) in enumerate(rssi.ri.rzoomBoundsCache):
            print("{} : {}".format(i,z))

        c = check_rssi_bounds_nonintersecting(rssi, count = True)
        print("[1] number of intersections: ",c)

        # case: activation is 1/2
        # check bounds for intersection
        return -1

    #
    """
    if a = 0.5 => midpoint reference will activate entire bounds
    """
    def test__relevance_zoom_func_1(self):

        # test out rz func 1
        q = sample_rssi_args_2()

        bq = q["bounds"]

        # get midpoint
        rf = bq[:,0] + (bq[:,1] - bq[:,0]) / 2.0
        d = euclidean_point_distance(bq[:,0],bq[:,1])
        act = 0.5

        rz = relevance_zoom_func_1(rf,d,act)

        print("LETSGO")
        # make an SSI and test out relevance of next()
        columnOrder = np.arange(bq.shape[0])[::-1]
        ssi = SearchSpaceIterator(bq, np.copy(bq[:,0]), columnOrder,\
            SSIHop = q["ssih"],cycleOn = True, cycleIs = 0)

        ## CASE 1
        # calculate ratio of relevant points w.r.t. midpoint(bounds)
        t,rc = 0,0
        while not ssi.reached_end():
            p2 = rz(next(ssi))
            if p2:
                rc += 1
            t += 1

        r1 = rc/float(t)
        assert r1 == 1.0, "[0] incorrect ratio"
        next(ssi)

        ## CASE 2
        # calculate ratio of relevant points w.r.t. bound[0]
        t,rc = 0,0
        rz = relevance_zoom_func_1(np.copy(bq[:,0]),d,act)
        while not ssi.reached_end():
            p2 = rz(next(ssi))
            if p2:
                rc += 1
            t += 1
        r2 = rc/float(t)
        assert r2 > 0.5, "[1] incorrect ratio"
        return -1


    def test__RSSI__iterate_one_bound(self):#  relevance_zoom_func_1
        rssi = sample_rssi_1()
        x = ResplattingSearchSpaceIterator.iterate_one_bound(rssi)
        i = 0
        for x_ in x:
            print("i ",i, "\t",x_)
            i += 1

    """
    tests iterate_one_bound on sample 2;
    """
    def test__RSSI__iterate_one_bound_2(self):#  relevance_zoom_func_1
        rssi = sample_rssi_3()
        rssi._summary()

        x = ResplattingSearchSpaceIterator.iterate_one_bound(rssi)
        i = 0
        for x_ in x:
            print("i ",i, "\t",x_)
            i += 1

        print("$$$")
        x = ResplattingSearchSpaceIterator.iterate_one_bound(rssi)
        i = 0
        for x_ in x:
            print("i ",i, "\t",x_)
            i += 1

        return -1
        """
        print("TYPE ",type(rssi.ssi))
        rssi.ssi.referencePoint = rssi.ssi.de_start()
        i = 0
        while not rssi.iterateIt:
            print("I  ",i, "\t", next(rssi))
            i += 1

        return -1
        """


    """
    tests that ResplattingInstructor uses default points with noise
    """
    def test__RSSI___resplattingmode___relevancezoomnoise(self):

        q = sample_rssi_args_3()
        rm = ("relevance zoom noise",None)
        ssih = 3

        ResplattingSearchSpaceIterator.add_noise_to_samples = MagicMock(name='noisia')
        rssi = ResplattingSearchSpaceIterator(q["bounds"], q["start"],\
            q["columnOrder"], ssih, rm, q["act"])
        ##ResplattingSearchSpaceIterator.add_noise_to_samples = MagicMock(name='noisia')
        rssi.add_noise_to_samples.assert_called()
        # TODO: use unittest.mock.Mock to trace method call for noise
        """
        real = ProductionClass()
        mock = Mock()
        real.closer(mock)
        mock.close.assert_called_with()
        """

        return -1







###########################################################################

def test__KDimProjector__point_at_degree_and_distance():

    p = np.array([0.0,0.5,5.0,1.7])
    d = 70.0
    d2 = 10.0

    p2 = KDimProjector.point_at_degree_and_distance(p,d,d2)
    print("P: ",p)
    print("P2: ",p2)
    print("D: ", euclidean_point_distance(p,p2))

    # do normalized here
    return

# TODO: refactor into i,j loop
def check_rssi_bounds_nonintersecting(rssi, count = False):
    c = 0
    for i in range(len(rssi.ri.rzoomBoundsCache) - 1):
        for j in range(i + 1, len(rssi.ri.rzoomBoundsCache)):
            ib = intersection_of_bounds(rssi.ri.rzoomBoundsCache[i],rssi.ri.rzoomBoundsCache[j])

            q = type(ib) == type(None)
            if not count:
                assert q, "rzoom bounds cache cannot intersect!"
            if not q:
                c += 1
    return c

'''
[2] bounds
[[  1.44444   1.44444]
 [ 12.77778  12.77778]
 [-14.33333 -14.33333]
 [ 15.33333  15.33333]
 [  5.55556   8.91358]]
'''

"""
strange bounds will have >= 1 0-length dim.
"""
def test__strange_bounds():

    bounds = np.array([[1.44444, 1.44444],\
        [12.77778, 12.77778],\
        [-14.33333, -14.33333],\
        [15.33333, 15.33333],\
        [5.55556, 8.91358]])

    startPoint = np.array([1.44444, 12.77778, -14.33333, 15.33333, 5.55556])
    columnOrder = [4,2,0,1,3]
    ssih = 2
    ssi = SearchSpaceIterator(bounds, startPoint, columnOrder, ssih,cycleOn = True)

    print("NEXT")
    print(next(ssi))
    print("NEXT")
    print(next(ssi))
    print(ssi.reached_end())
    print("NEXT")
    print(next(ssi))
    print("NEXT")
    print(next(ssi))

##############

"""
order of testing:

- relevance zoom

HOW?
- iterate through SSI once.
- go back through and check
"""
# TODO: check for rzoomBoundsCache


# TEST:  SkewedSearchSpaceIterator on case::(strange bounds)n


################### start: test activation ranges ####################

################### end: test relevant points ####################

### TODO: clean up, delete
def test__RSSI___initwithdifferentargs():

    # tests the following modes

    q = sample_rssi_args_3()
    rm = ("relevance zoom",None)
    ssih = 3
    rssi = ResplattingSearchSpaceIterator(q["bounds"], q["start"],\
        q["columnOrder"], ssih, rm, q["act"])


    print("RPRV")
    print(rssi.rprv)
    print()

    print("[d]eclaring new ssi w/ bounds")
    rssi.display_stat()
    ib = ResplattingSearchSpaceIterator.iterate_one_bound(rssi)
    for (i,ib_) in enumerate(ib):
        print(i, " next ", next(rssi))
        #next(rssi)
    ### default points
    # relevance zoom
    print("--------")
    rssi.display_stat()
    z = 3 ** 5
    #for i in range(z):
    """
    while not rssi.iterateIt:
        print(i, " next ", next(rssi))
        #print("end ", rssi.ssi.reached_end())
        #next(rssi)
        return -1
    """
    ib = ResplattingSearchSpaceIterator.iterate_one_bound(rssi)
    for (i,ib_) in enumerate(ib):
        print(i, " next ", next(rssi))
    return -1

    #ib = ResplattingSearchSpaceIterator.iterate_one_bound(rssi)

    for (i,ib_) in enumerate(ib):
        print(i, " next ", next(rssi))
    return -1
    # display     ##def ranges_of_interest()

    ######################################3
    # relevance zoom (w/ noise)
    rm = ("relevance zoom noise", None)
    rssi = ResplattingSearchSpaceIterator(q["bounds"], q["start"],\
        q["columnOrder"], ssih, rm, q["act"])

    print("[d]eclaring new ssi w/ bounds")
    print(rssi.ssi.de_bounds())
    ib = ResplattingSearchSpaceIterator.iterate_one_bound(rssi)

    # print the ranges of interest

    for (i,ib_) in enumerate(ib):
        print(i, " next ", next(rssi))

    ### TODO: compare the relevance of each point

    ### TODO: compare activation ranges
    ### make a range history for RSSI.
     #x = "relevance of each point"


    return -1





    qz = 3 ** 5
    print("end ", rssi.ssi.reached_end())
    print(rssi.ssi.endpoint)
    print(rssi.ssi.de_end())
    print()
    i = 0
    #for i in range(qz):
    while not rssi.iteratedOnce:
        print(i, " next ", next(rssi))
        print("end ", rssi.ssi.reached_end())
        i += 1

    print()
    print("NEXT")
    print()
    print()
    while not rssi.iterateIt:
        print(qz, " next ", next(rssi))



    return -1

    # do next

    #print("iterating one bound")


    #while not rssi.ssi.reached_end() and not rssi.terminated:



    x = ResplattingSearchSpaceIterator.iterate_one_bound(rssi)
    for x_ in x: print("{}\n".format(x_))

    print("ENDPOINT")
    print(rssi.ssi.de_end())
    print(rssi.ssi.reached_end())
    print()
    # try with noise






    ### custom points
    # png
    # png::noise

# TODO: make verbose now.

############################################### START: finish this for BallComp test data generation

def test__RSSI___resplattingmode___relevancezoom__auto___():
    return -1

# devise rf functions by RCH
'''
'''
def test__RSSI___resplattingmode___relevancezoom__manual___():

    return -1

# TODO: rf_generator
'''
given a bounds and an ssih
'''



# make function that outputs valvue v in range

'''
'''
def ballcomp_test_data_0():

    # add noise to each partition copy
    ssih = 5.0
    bounds = np.array([[-5,15.0],[-55,-21],[7.0,29.0],[1.0,9.0],[-3.0,10.0]])
    partitionDensities = [1,3,2]
    startPoint = np.copy(bounds[:,0])
    activationThreshold = (0.0,0.2)

    # 0.2,0.4,0.7 from [0]
    # by line [1] - [0]
    l = bounds[:,1] - bounds[:,0]

    p1 = bounds[:,0] + (l * 0.2)
    p2 = bounds[:,0] + (l * 0.4)
    p3 = bounds[:,0] + (l * 0.7)
    pointsOfInterest = [p1,p2,p3]

    rssi = ResplattingSearchSpaceIterator(bounds, startPoint, columnOrder = None, SSIHop = ssih, resplattingMode = ("relevance zoom",pointsOfInterest), activationThreshold = None)

    ## TODO: output_methods for each mode
    ## TODO: write "batch" to file

    # case 2
    # '''
    # by euclidean distance random
    d = euclidean_point_distance(bounds[:,0],bounds[:,1])



    return -1

############################################### END: finish this for BallComp test data generation


if __name__ == '__main__':
    #test__ResplattingSearchSpaceIterator__sample_1()
    #test__ResplattingSearchSpaceIterator__mock_next()
    print()
    #test__RSSI___initwithdifferentargs()
    unittest.main()
    #test__RSSI__iterate_one_bound_2()

    #test__RSSI___resplattingmode___relevancezoomnoise()

    #test__ResplattingSearchSpaceIterator__sample_2()
    #test__relevance_zoom_func_1()
    #test__relevance_zoom_func_1_2()
# TODO
'''
declining returns on best sol'n (convergence)
    > bounds of interest must be non-intersecting
    > ^ ^ needs to be checked!!

make a mystery polynomial function.

rewind and focus on LPS.
'''

###

# clean up code
# check relevant point values for a different case
