from .contextia_de_lo_vego_de_la_vegas import rssi
import numpy as np
import operator

"""
a version of the first chain; used in unit testing

rchl := cf function identifier
"""
def test_rch_chain_1(rchl = 1):

    rf = np.array([8.0,2.3,3.1,4.5,8.8])
    dm = rssi.euclidean_point_distance
    dt = 5.0

    if rchl == 1:
        cf = operator.gt
    elif rchl == 2:
        cf = operator.lt
    else:
        cf = lambda x, c: x + c >= 2.5 and x - c <= 5.0

    kwargs = ['r', rf,dm,cf,dt]
    rc = rssi.RChainHead()
    rc.add_node_at(kwargs)
    return rc

def sample_rssi_1(rmMode = "relevance zoom", rchl = 1):

    bounds = np.array([[-7.0,12.0],\
                        [3.0,25.0],\
                        [-20.0,-3.0],\
                        [9.0,28.0],\
                        [-2.0,32.0]])

    startPoint = np.copy(bounds[:,0])
    ssih = 2

    rc = test_rch_chain_1(rchl)
    rm = (rmMode, rc)
    return rssi.ResplattingSearchSpaceIterator(bounds, startPoint, columnOrder = None, SSIHop = ssih, resplattingMode = rm)
