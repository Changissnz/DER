from ball_comp_test_cases import *
from message_streamer import *

"""
"""
def sample_ballcomp_data_1():
    f = "indep/s.txt"

    # open with message streamer
    ms = MessageStreamer(f,readMode = "r")
    return ms

def t1():
    ms = sample_ballcomp_data_1()
    while ms.stream():
        print("wuncenshos")

###----------------------------
def kpointgenerator():
    return -1
###----------------------------

def test__BallComp__1():
    # test data from here
    ms = sample_ballcomp_data_1()

    maxBalls = 12
    maxRadius = 1.5
    k = 4
    bc = BallComp(maxBalls,maxRadius,k)

    ms.stream()
    q = ms.blockData[0]

    for i in range(13):
        bc.add_point(ms.blockData[i])

    return -1
