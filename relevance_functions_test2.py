from rssi import *

def euclidean_point_distance_of_bounds(parentBounds,bounds):

    if is_proper_bounds_vector(bounds):
        ed = euclidean_point_distance(bounds[:,1],bounds[:,0])
    else:
        pd = point_difference_of_improper_bounds(bounds, parentBounds)
        z = np.zeros((bounds.shape[0],))
        ed = euclidean_point_distance(z,pd)
    return ed

'''
vector -> ed vector -> (ed in bounds):bool
'''
def sample_rch_1_with_update(parentBounds, bounds, h, coverageRatio):

    def dm(rp,v):
        return np.array([euclidean_point_distance(v,rp_) for rp_ in rp])

    def cf(ds,dt_):
        return np.any(ds <= dt_)

    def update_rf_function(parentBounds,bounds,h):
        hops_to_coverage_points_in_bounds(parentBounds,bounds,h)

    def update_dt_function(parentBounds,bounds,h,coverageRatio):
        return (euclidean_point_distance_of_bounds(parentBounds,bounds) / h)\
                    * coverageRatio

    rch = RChainHead()
    dt = update_dt_function(parentBounds,bounds,h,coverageRatio)
    kwargs = ['r',cp,dm,cf,dt]
    rch.add_node_at(kwargs)

    # add update mechanism
    rch.s[0].updateFunc = {'rf': update_rf_function, 'dt': update_dt_function}
    rch.s[0].updatePath = {'rf': [0,1,2],'dt':[0,1,2,3]}
    rch.updatePath = {0: [0,1,2,3]}
    return rch

"""
an RSSI instance with an updating RCH.

RSSI runs in mode::(relevance zoom)
"""
def sample_rssi_1_with_update():

    b = np.array([[0,1.0],[0,1.0],[0,1.0]])
    pb = np.copy(b)
    h = 9
    coverageRatio = 0.4

    rch = sample_rch_1_with_update(b,pb,h,cv)

    ##
    rssi = ResplattingSearchSpaceIterator(b, np.copy(b[:,0]), None, h,\
        resplattingMode = ("relevance zoom",rch), additionalUpdateArgs = (coverageRatio))
    return rssi
