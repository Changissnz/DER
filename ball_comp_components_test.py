from ball_comp_components_test_cases import *
import unittest

######## start: class<SetMerger> tests ############

    ## cases: no merge
def sample_set_sequence_1():
    return [set((1,2)),set((2,3))]

def sample_set_sequence_2():
    return [set((1,2)),set((2,3)), set((3,4))]

def sample_set_sequence_3():
    return [set((1,2,3)),set((3,4,5)),set((2,4,5))]

def sample_set_sequence_4():
    return [set((1,2)),set((2,3))]

    ## cases: merge
def sample_set_sequence_5():
    return [set((1,2)),set((2,3)),set((1,3))]

def sample_set_sequence_6():
    return [set((1,2)),set((1,3)),set((1,4)),set((2,3)),set((2,4)),set((3,4))]

def sample_set_sequence_7():
    return [set((1,2,3)),set((1,2,4)),set((2,3,4))]

def sample_set_sequence_8():
    return [set([1]),set([2]),set([3]),set([4])]

def sample_set_sequence_9():
    return [set((1,2,3)),set((2,3,5)),\
        set((1,2,4)),set((8,9,10)),\
        set((2,3,4)),set((0,1,2))]

class TestBallCompComponents(unittest.TestCase):


    def test__SetMerger__is_closed_implication_for_merge(self):

        # test each sample set sequence
        s = sample_set_sequence_1()
        q = SetMerger.is_closed_implication_for_merge(s,1,2)
        assert not q
        q = SetMerger.is_closed_implication_for_merge(s,1,1)
        assert q

        s = sample_set_sequence_2()
        q = SetMerger.is_closed_implication_for_merge(s,1,2)
        assert not q
        q = SetMerger.is_closed_implication_for_merge(s,1,1)
        assert not q

        s = sample_set_sequence_3()
        q = SetMerger.is_closed_implication_for_merge(s,1,2)
        assert not q
        q = SetMerger.is_closed_implication_for_merge(s,1,1)
        assert not q
        q = SetMerger.is_closed_implication_for_merge(s,2,1)
        assert not q

        s = sample_set_sequence_4()
        q = SetMerger.is_closed_implication_for_merge(s,1,2)
        assert not q

        s = sample_set_sequence_5()
        q = SetMerger.is_closed_implication_for_merge(s,1,2)
        assert q

        s = sample_set_sequence_6()
        q = SetMerger.is_closed_implication_for_merge(s,1,3)
        assert not q

        s = sample_set_sequence_7()
        q = SetMerger.is_closed_implication_for_merge(s,1,2)
        assert q

        s = sample_set_sequence_8()
        q = SetMerger.is_closed_implication_for_merge(s,1,2)
        assert not q
        q = SetMerger.is_closed_implication_for_merge(s,1,1)
        assert q

    def test__SetMerger__merges_at_index(self):

        ##
        s = sample_set_sequence_6()
        sm = SetMerger(s)

        for i in range(len(s)):
            q = sm.merges_at_index(i)

            if i == 3:
                assert len(q) > 0
                continue
            assert len(q) == 0
        return






######## end: class<SetMerger> tests ############

######## start: class<Ball> tests ############

def test__Ball__intersection2():
    b3,b4 = sample_ball_3(),sample_ball_4()
    q = Ball.does_intersect(b3,b4)
    assert
    print("int? ",q)

    ai = Ball.area_intersection_estimation(b3,b4)
    print("ai? ",ai)
    return -1

def h(b1,b2):
    print("**")
    b12 = Ball.area_intersection_estimation_(b1,b2)
    b21 = Ball.area_intersection_estimation_(b2,b1)
    bae = Ball.area_intersection_estimation(b1,b2)

    print("\tarea estimation")
    print("b1->2: ",b12)
    print("b2->1: ",b21)
    print("est. area: ",bae)
    print("**")
    return

'''
tests for basic pairwisescores
'''
def test__Ball__intersection_():
    b1,b2 = test_ball_pair_1()
    b3 = sample_ball_3()
    b4 = sample_ball_4()

    h(b1,b2)
    a = Ball.area_intersection_estimation(b1,b2)
    assert (a > 0), "incorrect intersection for balls 1,2"
    h(b1,b3)
    h(b2,b3)

    a = Ball.area_intersection_estimation(b2,b3)
    assert (a - b3.area()) < 10 ** -5, "incorrect intersection for balls 2,3"

    h(b2,b2)
    a = Ball.area_intersection_estimation(b2,b2)
    assert (a - b2.area()) < 10 ** -5, "incorrect intersection for balls 2,3"

    h(b3,b4)

def test__Ball__intersection2_():
    b3 = sample_ball_3()
    b5 = sample_ball_5()
    b6 = sample_ball_6()
    b7 = sample_ball_7()

    h(b3,b5)
    h(b3,b6)
    h(b5,b6)

    threeA = Ball.threeway_area_intersection_estimation(b3,b5,b6)
    print("TA ",threeA)

    h(b3,b7)
    return

######## end: class<Ball> tests ############

if __name__ == '__main__':
    unittest.main()
