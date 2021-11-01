from relevance_functions import *

"""
    examples:
    epd:
        - rf := reference (vector)
        - dm := euclidean_point_distance | vector modulo
        - cf := operator
        - dt := operand

    in bounds:
        [a] bool::(range each)
        [b] float::(ratio of columns in range)
        [c] bool([b])
"""
##----------------------------------------------------------

# normalized euclidean point distance, w/ reference?
def test__rf__euclidean_point_distance():
    rf = np.array([8.0,2.3,3.1,4.5,7.1,8.8])
    dm = euclidean_point_distance
    cf = operator.lt

    # distance threshold is float
    dt = 20.0
    q = addon_singleton__bool__criteria_distance_from_reference(rf, dm, dt,cf)

    # test the points
    p1 = np.array([6.2,3.1,5.5,4.5,7.1,8.8])
    p2 = np.array([25.5,3.1,5.7,14.5,8.0,9.8])
    assert q(p1), "[0] incorrect case 1"
    assert not q(p2), "[0] incorrect case 2"

    # distance threshold is range
    cf = lambda_floatin
    dt = (5.0,25.0)
    q = addon_singleton__bool__criteria_distance_from_reference(rf, dm, dt,cf)
    assert not q(p1), "[1] incorrect case 1"
    assert q(p2), "[1] incorrect case 2"


def test__RChainHead__euclidean_point_distance__():

    rf = np.array([8.0,2.3,3.1,4.5,7.1,8.8])
    dm = euclidean_point_distance
    cf = operator.lt
    dt = 20.0
    kwargs = ['r', rf,dm,cf,dt]

    rc = RChainHead()
    rc.add_node_at(kwargs)

    # args. for euclidean_point_distance

    p1 = np.array([6.2,3.1,5.5,4.5,7.1,8.8])
    p2 = np.array([25.5,3.1,5.7,14.5,8.0,9.8])
    assert rc.apply(p1), "[0] incorrect case 1"
    assert not rc.apply(p2), "[0] incorrect case 2"

def test__RChainHead___in_bounds___():

    # case 1: bool::(range each)
    p1 = np.array([6.2,3.1,5.5,4.5,7.1,8.8])

        # subcase: dt is pair
    bounds0 = np.array([[3.1,9]])
    kwargs = ['nr', lambda_pointinbounds, bounds0]
    rc = RChainHead()
    rc.add_node_at(kwargs)
    assert rc.apply(p1) == True, "case 1.1: incorrect"

        # subcase: dt is bounds
    bounds = np.array([[5.0,7.0],\
            [3.2,4.1],\
            [5,6],\
            [4.5,4.7],\
            [6.5,7.5],\
            [8.5,9.2]])

    kwargs = ['nr', lambda_pointinbounds, bounds]
    rc = RChainHead()
    rc.add_node_at(kwargs)
    assert rc.apply(p1) == False, "case 1.2: incorrect"

    # case 2: float::(number of columns in range)

        # subcase: dt is pair
    kwargs = ['nr', lambda_countpointsinbounds, bounds0]
    rc = RChainHead()
    rc.add_node_at(kwargs)
    print("X ", rc.apply(p1))
    assert rc.apply(p1) == 6, "case 2.1: incorrect"

        # subcase: dt is pair
    bounds2 = np.array([[3.2,8.8]])
    kwargs = ['nr', lambda_countpointsinbounds, bounds2]
    rc = RChainHead()
    rc.add_node_at(kwargs)
    print("X ", rc.apply(p1))
    assert rc.apply(p1) == 5, "case 2.2: incorrect"

        # subcase: dt is bounds
    kwargs = ['nr', lambda_countpointsinbounds,bounds]
    rc = RChainHead()
    rc.add_node_at(kwargs)
    print("X ", rc.apply(p1))
    assert rc.apply(p1) == 5, "case 2.3: incorrect"

# TODO: add to
"""
outputs the func for
"""
def RCHF___in_bounds(bounds0):
    kwargs = ['nr', lambda_pointinbounds, bounds0]

    # f : v -> v::bool
    rc = RChainHead()
    rc.add_node_at(kwargs)

    # f : filter out True | False
    subvectorSelector = boolies
    ss = subvector_selector(subvectorSelector,2)
    kwargs = ['nr',ss]
    rc.add_node_at(kwargs)

    # f : get indices
    ss = column_selector([0],True)
    kwargs = ['nr',ss]
    rc.add_node_at(kwargs)

    # f : apply indices on reference
    kwargs = ['nr',(vector_index_inverse_selector,[0])]
    rc.add_node_at(kwargs)

    return rc.apply

def test__RCHF___in_bounds():
    # code case: point p in bounds
    b = np.array([[-2,4],\
                    [5,15],\
                    [12,37],\
                    [-13,99],\
                    [-1,900]])
    kwargs = ['nr', lambda_pointinbounds, b]

    # f : v -> v::bool
    rc = RChainHead()
    rc.add_node_at(kwargs)
    p = np.array([0,7,15,-29,902])
    q = rc.apply(p)
    print("RES: ",q)

    def qf(xi):
        return operator.le(xi[1],b[xi[0],1]) and operator.ge(xi[1],b[xi[0],0])

    """
    def add_on_sample_i(i_):
         return i_ in indices
    """
    q2 = subvector_selector(qf,inputType = 2,outputType = 1)

    kwargs = ['nr', q2]
    rc = RChainHead()
    rc.add_node_at(kwargs)

    q = rc.apply(p)
    print("RES: ",q)

    return -1



    ##return -1

    rchf = RCHF___in_bounds(b)
    p = np.array([0,7,15,-29,902])

    q = rchf(p)
    print("RES: ",q)
    return -1

# TODO: refactor into 2 branches
"""
demonstrates how subvector selector is used
by their input
"""
def test__RChainHead___subvectorselector___():

    ## case 1: uses function `add_on_sample`
    p1 = np.array([6.2,3.1,5.5,4.5,7.1,8.8])

    def add_on_sample(v_):
        return v_ / 1.5 + 1 > np.mean(p1)

    q = subvector_selector(add_on_sample,outputType =2)
    kwargs = ['nr', q]

    rc = RChainHead()
    rc.add_node_at(kwargs)

    q = rc.apply(p1)
    print("Q: ",q)
    assert equal_iterables(q,np.array([[5.,8.8]])), "[0] invalid result"


    ## case 2: uses function `add_on_sample_i`
    indices = [3,4,5]
    def add_on_sample_i(i_):
         return i_ in indices

    q2 = subvector_selector(add_on_sample_i,inputType = 0)
    kwargs = ['nr', q2]
    rc = RChainHead()
    rc.add_node_at(kwargs)
    x = rc.apply(p1)
    assert equal_iterables(x,[4.5,7.1,8.8]), "not equal iterables, {}".format(x)

    return -1

    ## DELETE BELOW

    # case: selector output is boolean vector according to
    #       subvector selector
    q = subvector_selector(add_on_sample,outputType =1)
    q2 = subvector_selector(add_on_sample_i,inputType = 0)
    q2 = subvector_selector(add_on_sample_i,inputType = 0)

    # case: add another node that selects only boolean values
    """
    v -> vi -> select(v,vi)
    example of a referential selector
    """
    # an index selector
    ##q = lambda x: x[-1]

    subvectorSelector = boolies
    ss = subvector_selector(subvectorSelector,2)
    kwargs = ['nr',ss]

    ##rc = RChainHead()
    rc.add_node_at(kwargs)
    x = rc.apply(p1)
    print("RES: ",x)
    # case: filter out 0 indices

    def lcast_func(l):
        return np.array(l)

    ss = column_selector([0],True)
    kwargs = ['nr',ss]

    rc.add_node_at(kwargs)

    x = rc.apply(p1)
    print("RES: ",x)
    #return -1
    return -1

    # case: apply index vector on value at 0 in vpath

        # subcase: vector_index_selector(vpath[-1])
        #           acts on reference 0

        #   ... dir is supposed to be inverse
    kwargs = ['nr',(vector_index_inverse_selector,[0])]
    rc.add_node_at(kwargs)

    x = rc.apply(p1)
    print("RES: ",x)

    # will need a method<update_function>=

    return -1
############################# FUTURE


'''
NEXT: (interval|external)
[0] multiplier of index pairs (i,i +1)
'''

###
'''
    ## TODO: make invertible
    ss = filter(lambda x: x < 5, q)
'''
###

##
"""
after code clean-up, do demo on:

- direction_to(rv) on sequence of vectors of rv + noise

run frequency tests on directionality per index set

test output can be used to devise the optimum number of
particles required to travel the rest of the bound.

next(p) := ADD_NOISE(p,multiplier)
next(poly) := MOD_POLY
"""
