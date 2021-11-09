from rssi import *

'''
vector -> ed vector -> (ed in bounds):bool
'''
def sample_rch_1_with_update(parentBounds, bounds, h, coverageRatio):

    def dm(rp,v):
        return np.array([euclidean_point_distance(v,rp_) for rp_ in rp])

    def cf(ds,dt_):
        return np.any(ds <= dt_)

    def update_dt_function(parentBounds,bounds,h,coverageRatio):
        return (euclidean_point_distance_of_bounds(parentBounds,bounds) / h)\
                    * coverageRatio

    rch = RChainHead()
    # add the node
    rf = hops_to_coverage_points_in_bounds(parentBounds,bounds,h)
    dm = dm
    cf = cf
    ed = euclidean_point_distance_of_bounds(parentBounds,bounds)
    dt = coverage_ratio_to_distance(ed,9,coverageRatio)

    kwargs = ['r',rf,dm,cf,dt]
    rch.add_node_at(kwargs)

    # add update functionality
    rch.s[0].updateFunc = {'rf': hops_to_coverage_points_in_bounds,\
            'dt': update_dt_function}
    rch.s[0].updatePath = {'rf': [0,1,2],'dt':[0,1,2,3]}
    rch.updatePath = {0: [0,1,2,3]}
    return rch

def test__sample_rch_1_with_update():
    b = np.array([[0,1.0],[0,1.0],[0,1.0]])
    pb = np.copy(b)
    h = 9
    cv = 0.4
    rch = sample_rch_1_with_update(b,pb,h,cv)

    # do one
    p = np.array([0.5,0.5,0.5])
    q = rch.apply(p)
    print("Q ", q)
    assert not q, "failed case 1"

    # update args
    pb2 = np.array([[0,0.6],[0,0.6],[0,0.6]])
    pb2 = np.array([[0,0.5],[0,0.5],[0,0.5]])

    updateArgs = [b,pb2,h,cv]
    rch.load_update_vars(updateArgs)
    rch.update_rch()

    p2 = np.array([0.3,0.3,0.3])
    p2 = np.array([0.25,0.25,0.25])

    q = rch.apply(p2)
    print("Q ", q)
    assert not q, "failed case 2"

"""
an RSSI instance with an updating RCH.

RSSI runs in mode::(relevance zoom)
"""
def sample_rssi_1_with_update():

    b = np.array([[0,1.0],[0,1.0],[0,1.0]])
    pb = np.copy(b)
    h = 9
    cv = 0.4

    rch = sample_rch_1_with_update(b,pb,h,cv)

    ##
    r = ResplattingSearchSpaceIterator(b, np.copy(b[:,0]), None, h,\
        resplattingMode = ("relevance zoom",rch), additionalUpdateArgs = (cv,))
    return r

def test__sample_rssi_1_with_update():
    r = sample_rssi_1_with_update()
    rssi__display_n_bounds(r,2)
    return -1
