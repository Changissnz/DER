from rssi import *

"""
a version of the first chain; used in unit testing

rchl := cf function identifier
"""
def test_rch_chain_1(rmMode = "relevance zoom", rchl = 1):
    bounds = np.array([[-7.0,12.0],\
                        [3.0,25.0],\
                        [-20.0,-3.0],\
                        [9.0,28.0],\
                        [-2.0,32.0]])

    startPoint = np.copy(bounds[:,0])

    ssih = 2

    rf = np.array([8.0,2.3,3.1,4.5,8.8])
    dm = euclidean_point_distance
    dt = 5.0

    if rchl == 1:
        cf = operator.gt
    elif rchl == 2:
        cf = operator.lt
    else:
        cf = lambda x, c: x + c >= 2.5 and x - c <= 5.0

    kwargs = ['r', rf,dm,cf,dt]
    rc = RChainHead()
    rc.add_node_at(kwargs)
    rm = (rmMode, rc)
    return ResplattingSearchSpaceIterator(bounds, startPoint, columnOrder = None, SSIHop = ssih, resplattingMode = rm)

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


################################ start tests: relevance zoom, euclidean point distance on 5.0
#

"""
>
"""
def test__x():
    rssi = test_rch_chain_1(rchl = 1)
    rssi__display_n_bounds(rssi, 3)
    return

"""
<
"""
def test__x2():
    rssi = test_rch_chain_1(rchl = 2)
    rssi__display_n_bounds(rssi, 2)
    return

"""
>= and <=
"""
def test__x3():
    rssi = test_rch_chain_1(rchl = 3)
    rssi__display_n_bounds(rssi, 2)
    return

################################ end tests: relevance zoom, euclidean point distance on 5.0

################################ start tests: png, euclidean point distance on 5.0

def test__x4():
    rssi = test_rch_chain_1(rmMode = "prg")
    rssi__display_n_bounds(rssi,1)
    print("\n\n\tNEZXTING 10\n")

    for i in range(10):
        print(next(rssi))

    return -1



################################ end tests: png, euclidean point distance on 5.0
