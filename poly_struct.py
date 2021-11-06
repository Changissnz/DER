from poly_interpolation import *

class SPoly:

    """
    v := vector of values, length l -1 is power, index 0 is greatest power
    """
    def __init__(self,v):
        assert is_vector(v), "invalid vector"
        self.v = v

    def apply(self,x):
        s = 0.0
        l = len(self.v) - 1
        for v_ in self.v:
            s += (v_ * x ** l)
            l -= 1
        return s

def test__SPoly__apply():

    # poly case 1
    sp = SPoly(np.array([12.0,0.0,3.0,1.0,2.0]))

    #   x1
    v1 = sp.apply(3.0)
    print(v1)
    assert v1 == 999 + 3 + 2, "incorrect case 1.1"

    v2 = sp.apply(0.0)
    assert v2 == 2.0, "incorrect case 1.2"


class PartialSoln:

    def __init__(self):
        return


"""

f(x,y,z,...) = each element is a linear combination; index i corresponds to power

    each element a matrix

"""
class MPolyStruct:

    def __init__(self,x,y,z):

        return
