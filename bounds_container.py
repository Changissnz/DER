## TODO: check and refactor imports
from numerical_generator import *

'''
performs excision of a subbounds in bounds
'''
class BoundsCutter:

    def __init__(self,bounds):
        assert is_proper_bounds_vector(bounds),"invalid bounds"
        self.bounds = bounds
        self.targetSb = None
        return

    def set_target_subbounds(self,tsb):
        return -1

    def cut_piece(pos):
        return -1


class BoundsContainer:

    def __init__(self,bounds,points = None):
        assert is_proper_bounds_vector(bounds), "invalid bounds vector"
        self.bounds = bounds

        if type(points) is type(None):
            self.points = np.empty((0,bounds.shape[0]))
        elif is_2dmatrix(points):
            assert points.shape[1] == bounds.shape[0], "invalid dim. 1 for points"
            self.points = np.empty((0,bounds.shape[0]))
        else:
            raise ValueError("invalid type for points")

        self.xSubbounds = [] # list of sub-bounds in self.bounds


    '''
    '''
    @staticmethod
    def start_point_in_bounds(bounds,point):
        assert is_proper_bounds_vector(bounds)
        v = np.asarray(bounds[:,1] - bounds[:,0], dtype=float)
        q = np.asarray(point - bounds[:,0], dtype = float)
        return q / v

    @staticmethod
    def center_point(bounds):
        assert is_proper_bounds_vector(bounds)
        d = (bounds[:,1] - bounds[:,0] ) / 2.0
        return bounds[:,0] + d

    '''
    calculates the min bound B that would accomodate
    both self and `bc2`
    '''
    def __add__(self,bc2):
        return -1

    '''
    calculate
    d1 := distance b/t b.center and bc2.center
    d2 := max(distance(b.center,b.points))
    d3 := max(distance(bc2.center,bc2.points))

    generate line b/t b.center and bc2.center as long

    '''

    '''
    '''
    def __sub__(self,bc2):
        return -1
