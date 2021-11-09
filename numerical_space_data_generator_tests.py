from numerical_space_data_generator import *


def sample_nsdi_1():

    bounds = np.array([[0,1.2],[-0.5,1.7],[-1.0,0.8],[-0.5,1.2]])
    sp = np.copy(bounds[:,0])
    columnOrder = None
    ssih = 4
    cv = 0.6
    bInf = (bounds,sp,columnOrder,ssih,(cv,))

    # make relevance function
    rch = sample_rch_1_with_update(bounds, np.copy(bounds), ssih, cv)
    rm = ('relevance zoom', rch)

    sp = np.copy(bounds[:,1])
    filePath = "tests/s.txt"
    modeia = 'w'

    q = NSDataInstructions(bInf, rm, sp, filePath,modeia)
    return q

'''
'''
def test__sample_nsdi_1():
    q = sample_nsdi_1()
    q.make_rssi()

    q.next_batch_()
    q.next_batch_()

    q.close()

#def
