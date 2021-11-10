from numerical_space_data_generator import *

"""
"""
def sample_nsdi_1(q, filePath = "tests/s.txt"):
    if type(q) == str:
        assert q in ['relevance zoom', 'prg']
    else:
        assert type(q) == np.ndarray, "333"

    bounds = np.array([[0,1.2],[-0.5,1.7],[-1.0,0.8],[-0.5,1.2]])
    sp = np.copy(bounds[:,0])
    columnOrder = None
    ssih = 4
    cv = 0.6
    bInf = (bounds,sp,columnOrder,ssih,(cv,))

    # make relevance function
    rch = sample_rch_1_with_update(bounds, np.copy(bounds), ssih, cv)
    rm = (q, rch)

    sp = np.copy(bounds[:,1])
    modeia = 'w'

    q = NSDataInstructions(bInf, rm, sp, filePath,modeia)
    return q

'''
'''
def test__sample_nsdi_11():
    q = sample_nsdi_1("relevance zoom",filePath="tests/s.txt")
    q.make_rssi()

    c = 0
    while q.fp:
        q.next_batch_()
        c += 1
    print("# batches: ",c)
    q.batch_summary()

def test__sample_nsdi_12():
    q = sample_nsdi_1('prg',"tests/sneaht.txt")
    q.make_rssi()

    c = 0
    while q.fp and c < 100:
        q.next_batch_()
        c += 1
    print("# batches: ",c)
    q.batch_summary()

def test__sample_nsdi_13():
    #bounds = np.array([[0,1.2],[-0.5,1.7],[-1.0,0.8],[-0.5,1.2]])
    b1 = np.array([[0.9,1.2],[1.2,1.7],[-1.0,-0.5],[1.0,1.55]])
    b2 = np.array([[0.7,0.05],[1.6,0.2],[-1.0,0.5],[1.1,0.2]])
    bx = np.array([b1,b2])

    print(bx)


    ##return -1
    q = sample_nsdi_1(bx,"tests/sneaht.txt")
    q.make_rssi()

    q.next_batch_()
    print("XS")
    return -1
    c = 0
    while q.fp and c < 2:
        q.next_batch_()
        c += 1
    print("# batches: ",c)
    q.batch_summary()
