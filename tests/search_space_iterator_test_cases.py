from .contextia_de_lo_vego_de_la_vegas import search_space_iterator
import numpy as np

def SearchSpaceIterator_case_1():

    bounds = np.array([[0,1],\
                    [0,1],\
                    [0,1]
        ])
    startPoint = np.array([0,1,0])

    # TODO: test w/ this column order
    columnOrder = [0,1,2]
    return search_space_iterator.SearchSpaceIterator(bounds, startPoint, columnOrder, 2)

def SearchSpaceIterator_case_2():
    bounds = np.array([[0,1],\
                    [0,1],\
                    [0,1]
        ])
    startPoint = np.array([0,0,0])

    # TODO: test w/ this column order
    columnOrder = [2,0,1]
    return search_space_iterator.SearchSpaceIterator(bounds, startPoint, columnOrder, 5)

def SearchSpaceIterator_case_3():
    bounds = np.array([[0,1],\
                    [0,1],\
                    [0,1]
        ])
    startPoint = np.array([0,0,0])

    # TODO: test w/ this column order
    columnOrder = [0,1,2]
    return search_space_iterator.SearchSpaceIterator(bounds, startPoint, columnOrder, 3)

def SearchSpaceIterator_case_4():
    bounds = np.array([[0,1],\
                    [0,1],\
                    [0,1],\
                    [0,1],\
                    [0,1]])

    startPoint = np.array([0.0,0.5,0.3, 1,0.75])
    columnOrder = [4,2,0,1,3]
    search_space_iterator.HopPattern.DEF_INCREMENT_RATIO = round(1/7,10)
    ##return SearchSpaceIterator(bounds, startPoint, columnOrder,\
    ##    7, "proportional")
    return search_space_iterator.SearchSpaceIterator(bounds, startPoint, columnOrder,7)

def SearchSpaceIterator_case_5():
    ssi = SearchSpaceIterator_case_4()
    hop_pattern.HopPattern.DEF_INCREMENT_RATIO = round(1/4,2)
    return ssi

def SearchSpaceIterator_case_6():

    bounds = np.array([[0,10],\
                    [0,10],\
                    [0,10.00],\
                    [0,10],\
                    [0,10.00]])

    startPoint = np.array([5.0, 4.0, 3.0, 8.0, 7.5])
    columnOrder = [4,2,0,1,3]
    ssih = 2
    return search_space_iterator.SearchSpaceIterator(bounds, startPoint, columnOrder, ssih,cycleIs = 1)

def SearchSpaceIterator_case_7():

    bounds = np.array([[0,10],\
                    [0,10],\
                    [0,10.00],\
                    [0,10],\
                    [0,10.00]])

    bounds = invert_bounds(bounds)
    startPoint = np.array([5.0, 4.0, 3.0, 8.0, 7.5])
    columnOrder = [4,2,0,1,3]
    ssih = 2
    return search_space_iterator.SearchSpaceIterator(bounds, startPoint, columnOrder, ssih)

"""
case is used to test out values given head in {0,1}
"""
def SearchSpaceIterator_case_8(head):

    bounds = np.array([[0,1],\
                    [0,1],\
                    [0,1],\
                    [0,1],\
                    [0,1]])

    startPoint = np.array([0,1.0,0.0,1.0,0.5])
    columnOrder = [4,3,2,1,0]
    ssih = 2
    return search_space_iterator.SearchSpaceIterator(bounds,startPoint,columnOrder,ssih, True,cycleIs = head)

def SkewedSearchSpaceIterator_args_1():

    twoThirds = round(2/3.0,5)
    oneThirds = round(1/3.0,5)

    bounds = np.array([[twoThirds,oneThirds],\
                    [twoThirds,oneThirds],\
                    [twoThirds,oneThirds],\
                    [twoThirds,oneThirds],\
                    [twoThirds,oneThirds]])

    parentBounds = np.array([[0,1.0],\
                    [0,1.0],\
                    [0,1.0],\
                    [0,1.0],\
                    [0,1.0]])

    return bounds,parentBounds

def SkewedSearchSpaceIterator_case_1():
    b,pb = SkewedSearchSpaceIterator_args_1()
    return search_space_iterator.SkewedSearchSpaceIterator(b,pb,None,None,SSIHop = 3, cycleOn = True, cycleIs = 1)

def SkewedSearchSpaceIterator_case_2():
    b,pb = SkewedSearchSpaceIterator_args_1()
    sp = b[:,1]
    return search_space_iterator.SkewedSearchSpaceIterator(b,pb,sp,None,SSIHop = 3, cycleOn = True, cycleIs = 1)

def SkewedSearchSpaceIterator_case_3():
    b,pb = SkewedSearchSpaceIterator_args_1()

    bi = [0,1,1,0,0]
    sp = np.array([b[i,bi_] for (i,bi_) in enumerate(bi)])
    print("START POINT: ",sp)
    return search_space_iterator.SkewedSearchSpaceIterator(b,pb,sp,None,SSIHop = 3, cycleOn = True, cycleIs = 1)



########
