from numerical_generator import *

######################## START: cycle map

def test_CycleMap__random_map():
    vr = CycleMap.random_cycle_map(5)
    ##print("VR")
    ##print(vr)

    f = CycleMap.is_valid_map(vr)
    ##print("F: ", f)
    assert f, "random cycle map is not cycle"

    q = OrderedDict()
    x = [(3,1),(1,3),(2,4),(4,2)]
    for k,v in x:
        q[k] = v

    f2 = CycleMap.is_valid_map(q)
    ##print("F2: ", f2)
    assert not f2, "non-cyclic map"
    return

def test_CycleMap__set_map():
    f = CycleMap.random_cycle_map(5)
    cm = CycleMap(5)
    cm.set_map(f)
    return

###########################

def test__generate_possible_binary_sequences():
    g = generate_possible_binary_sequences(5, [])
    g = list(g)
    assert len(g) == 2 ** 5, "incorrect generation"

    # uncomment for viewing

    for g_ in g:
        print(g_)

def test_FloatDeltaGenerator__next__():
    # case 1
    func = DELTA_ADD
    mox = round(1/7,5)
    fdg = FloatDeltaGenerator(0.0, DELTA_ADD, mox,1.0)

    c = 10
    while c > 0:
        q = next(fdg)
        print("r: ", q)
        if q == None: break
        c -= 1

    print("CELAS ", c)

    # case 2
    fdg = FloatDeltaGenerator_pattern1((1.0,0.0), 15.0)

    c = 20
    while c > 0:
        q = next(fdg)
        print("Q: ", q)
        c -= 1
        if q == None: break

    print("CELAS ", c)
