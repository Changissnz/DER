from rssi_test_cases import *

################################ start tests: relevance zoom, euclidean point distance on 5.0
#

"""
>
"""
def test__rssi__case1():
    rssi = sample_rssi_1(rchl = 1)
    rssi__display_n_bounds(rssi, 3)
    return

"""
<
"""
def test__rssi__case2():
    rssi = sample_rssi_1(rchl = 2)
    rssi__display_n_bounds(rssi, 2)
    return

"""
>= and <=
"""
def test__rssi__case3():
    rssi = sample_rssi_1(rchl = 3)
    rssi__display_n_bounds(rssi, 2)
    return

################################ end tests: relevance zoom, euclidean point distance on 5.0

################################ start tests: png, euclidean point distance on 5.0

def test__rssi__case4():
    rssi = sample_rssi_1(rmMode = "prg")
    rssi__display_n_bounds(rssi,1)

    # check relevant points
    '''
    print("relevant points")
    for (i,v) in enumerate(rssi.ri.centerResplat.relevantPointsCache):
        print("{} : {}".format(i,v))
    '''
    assert len(rssi.ri.centerResplat.relevantPointsCache) == 32, "incorrect number of relevant pts."

    #print("\n\n\tNEZXTING 10\n")
    for i in range(10):
        #print(next(rssi))
        q = next(rssi)
        assert point_in_bounds(rssi.ri.centerResplat.centerBounds,q), "point w/ noise must stay in bounds"

    # display index counter
    print("RPI COUNTER")
    print(rssi.ri.centerResplat.rpIndexCounter)
    assert len(rssi.ri.centerResplat.rpIndexCounter) > 0, "center-resplat index counter cannot be empty"
    return

# testing for relevant points
################################ end tests: png, euclidean point distance on 5.0

# NEXT: verbose mode
