from rssi import *


'''
used for samples 1,2 to test `resplattingMode`
'''
def sample_rssi_args():
    bounds = np.array([[-7.0,12.0],\
                        [3.0,25.0],\
                        [-20.0,-3.0],\
                        [9.0,28.0],\
                        [-2.0,32.0]])

    ssih = 9
    diff = bounds[:,1] - bounds[:,0]
    diff = diff / ssih

    ## for 9
    joresin = [4,4,3,3,2]
    ##joresin = [0,1,1,2,0]
    p = np.array([joresin[i] * d for (i,d) in enumerate(diff)])
    start = bounds[:,0] + p

    columnOrder = None
    act = None

    d = {}
    d["bounds"] = bounds
    d["start"] = start
    d["columnOrder"] = columnOrder
    d["ssih"] = ssih
    d["act"] = act
    return d

"""
"""
def sample_rssi_args_2():
    # delete
    """
    bounds = np.array([[-7.0,12.0],\
                        [3.0,25.0],\
                        [-20.0,-3.0],\
                        [9.0,28.0],\
                        [-2.0,32.0]])
    """
    ###
    bounds = np.array([[-2.0,1.0],\
                        [2.0,6.0],\
                        [3.0,15.0],\
                        [-5.0,2.0],\
                        [-10.0,4.0]])


    ssih = 3
    start = np.array([-2,6,3,2,-10])
    columnOrder = None
    act = 0.5

    d = {}
    d["bounds"] = bounds
    d["start"] = start
    d["columnOrder"] = columnOrder
    d["ssih"] = ssih
    d["act"] = act
    return d

def sample_rssi_args_3():

    d = sample_rssi_args()
    d["ssih"] = 2
    return d

"""
"""
def sample_rssi_1():
    argos = sample_rssi_args()
    rm = ("relevance zoom",None)
    return ResplattingSearchSpaceIterator(argos["bounds"], argos["start"],\
            argos["columnOrder"], argos["ssih"], rm, argos["act"])

def sample_rssi_2():
    argos = sample_rssi_args()

    # make the prg function
    rf2 = relevance_func_2(6.2, (0.4,0.6))
    rm = ("prg",rf2)
    return ResplattingSearchSpaceIterator(argos["bounds"], argos["start"],\
            argos["columnOrder"], argos["ssih"], rm, argos["act"])


def sample_rssi_3():
    argos = sample_rssi_args_3()

    s1 = np.array([-3.5,13.0,-10.0,21.0,8.0])
    s2 = np.array([6.0,19.0,-5.0,11.0,24.0])
    rm = ("relevance zoom",np.array([s1,s2])) 
    return ResplattingSearchSpaceIterator(argos["bounds"], argos["start"],\
            argos["columnOrder"], argos["ssih"], rm, argos["act"])
