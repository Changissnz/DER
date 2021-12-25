from ball_comp import *



def sample_BallComp_1():
    return -1

def test__BallComp__sample_1_sample_data_1():

    maxBalls = 20
    maxRadius = 5.0
    td = test_data_1()

    # TODO: delete k
    bc = BallComp(maxBalls,maxRadius,5,True)

    #bc.conduct_decision(td[0])

    for t in td:
        bc.conduct_decision(t)
    return

def test__BallComp__sample_1_sample_data_2():

    maxBalls = 20
    maxRadius = 5.0
    td = test_data_2()

    # TODO: delete k
    bc = BallComp(maxBalls,maxRadius,5,True)

    for t in td:
        bc.conduct_decision(t)

    print("********************")

    print("BALLS ", len(bc.balls))
    for k,v in bc.balls.items():
        print("k ",k)
        print(v)
        print()
    return

def test__BallComp__sample_1_sample_data_3():

    maxBalls = 20
    maxRadius = 5.0
    td = test_data_3()

    # TODO: delete k
    bc = BallComp(maxBalls,maxRadius,5,True)
    for t in td:
        bc.conduct_decision(t)

    print("********************")
    print("BALLS ", len(bc.balls))
    for k,v in bc.balls.items():
        print("k ",k)
        print(v)
        print()
    return
