from .contextia_de_lo_vego_de_la_vegas import ball_comp_components
import numpy as np

def sample_ball_1():
    center = np.array([4,7,11,14])
    points = np.array([[4.1,6.9,11.1,13.9]])
    return ball_comp_components.Ball.one_ball(center,points)

'''
intersections: 1,3
'''
def sample_ball_2():
    center = np.array([10,10,10,10])
    points = np.array([[18.1,10,10,10]])
    return ball_comp_components.Ball.one_ball(center,points)

'''
intersections: 2,4
'''
def sample_ball_3():
    center = np.array([10,10,10,10])
    points = np.array([[10,14.0,10,10]])
    return ball_comp_components.Ball.one_ball(center,points)

'''
intersections: 3
'''
def sample_ball_4():
    center = np.array([10,20.0,10.0,10])
    points = np.array([[10,12.0,10,10]])
    return ball_comp_components.Ball.one_ball(center,points)

'''
intersections:
'''
def sample_ball_5():
    center = np.array([10,12,10,10])
    points = np.array([[10,16.0,10,10]])
    return ball_comp_components.Ball.one_ball(center,points)

'''
'''
def sample_ball_6():
    center = np.array([10,8,10,10])
    points = np.array([[10,12.0,10,10]])
    return ball_comp_components.Ball.one_ball(center,points)

def sample_ball_7():
    center = np.array([10,18,10,10])
    points = np.array([[8,18.0,10,10]])
    return ball_comp_components.Ball.one_ball(center,points)

'''
used for intersection
'''
def test_ball_pair_1():
    return sample_ball_1(),sample_ball_2()
