from .contextia_de_lo_vego_de_la_vegas import numerical_space_data_generator, relevance_functions
import unittest
import numpy as np

"""
"""
def sample_nsdi_1(q, filePath = "tests/s.txt", nr = None):
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
    rch = relevance_functions.sample_rch_1_with_update(bounds, np.copy(bounds), ssih, cv)
    rm = (q, rch)

    ##sp = np.copy(bounds[:,1])
    modeia = 'w'

    q = numerical_space_data_generator.NSDataInstructions(bInf,rm,filePath,modeia,nr)
    return q

def sample_nsdi_2(q, filePath = "tests/s.txt", nr = None):
    if type(q) == str:
        assert q in ['relevance zoom', 'prg']
    else:
        assert type(q) == np.ndarray, "333"

    bounds = np.array([[0,1.2],[-0.5,1.7],[-1.0,0.8],[-0.5,1.2]])
    sp = np.copy(bounds[:,0])
    columnOrder = None
    ssih = 7
    bInf = (bounds,sp,columnOrder,ssih,())

    # make relevance function
    rch = sample_rch_2_with_update(bounds, np.copy(bounds))
    rm = (q, rch)
    modeia = 'w'

    q = NSDataInstructions(bInf,rm,filePath,modeia,nr)
    return q

class TestNSDataInstructionClass(unittest.TestCase):

    '''
    '''
    def test__sample_nsdi_11(self):
        q = sample_nsdi_1("relevance zoom",filePath="tests/s11.txt")
        q.make_rssi()

        c = 0
        while q.fp:
            q.next_batch_()
            c += 1
        print("case 1")
        print("# batches: ",c)
        q.batch_summary()

    def test__sample_nsdi_12(self):
        q = sample_nsdi_1('prg',"tests/s12.txt")
        q.make_rssi()

        c = 0
        while q.fp and c < 100:
            q.next_batch_()
            c += 1
        print("case 2")
        print("# batches: ",c)

        if q.fp:
            q.fp.close()

        q.batch_summary()

    def test__sample_nsdi_13(self):
        b1 = np.array([[0.9,1.2],[1.2,1.7],[-1.0,-0.5],[0.7,1.1]])
        bx = np.array([b1])

        q = sample_nsdi_1(bx,"tests/s13.txt")
        q.make_rssi()

        while q.fp:
            q.next_batch_()
        #assert q.c == 3, "incorrect number of batches"
        print("case 3")
        q.batch_summary()

    def test__sample_nsdi_14(self):
        nr = np.array([[0.01,0.07]])
        q = sample_nsdi_1('relevance zoom',"tests/s14.txt",nr)
        q.make_rssi()

        c = 0
        while q.fp and c < 100:
            q.next_batch_()
            c += 1
        if q.fp:
            q.fp.close()
        print("# batches: ",c)
        q.batch_summary()

    def test__sample_nsdi_21(self):
        q = sample_nsdi_2("relevance zoom",filePath="tests/s21.txt")
        q.make_rssi()
        while q.fp:
            q.next_batch_()
        return

if __name__ == '__main__':
    unittest.main()
