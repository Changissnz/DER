from .ball_comp_components_test_cases import *
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

def h(b1,b2):
    print("**")
    b12 = ball_comp_components.Ball.area_intersection_estimation_(b1,b2)
    b21 = ball_comp_components.Ball.area_intersection_estimation_(b2,b1)
    bae = ball_comp_components.Ball.area_intersection_estimation(b1,b2)

    print("\tarea estimation")
    print("b1->2: ",b12)
    print("b2->1: ",b21)
    print("est. area: ",bae)
    print("**")
    return

class TestBallCompComponents(unittest.TestCase):

    def test__SetMerger__is_closed_implication_for_merge(self):

        # test each sample set sequence
        s = sample_set_sequence_1()
        q = ball_comp_components.SetMerger.is_closed_implication_for_merge(s,1,2)
        assert not q
        q = ball_comp_components.SetMerger.is_closed_implication_for_merge(s,1,1)
        assert q

        s = sample_set_sequence_2()
        q = ball_comp_components.SetMerger.is_closed_implication_for_merge(s,1,2)
        assert not q
        q = ball_comp_components.SetMerger.is_closed_implication_for_merge(s,1,1)
        assert not q

        s = sample_set_sequence_3()
        q = ball_comp_components.SetMerger.is_closed_implication_for_merge(s,1,2)
        assert not q
        q = ball_comp_components.SetMerger.is_closed_implication_for_merge(s,1,1)
        assert not q
        q = ball_comp_components.SetMerger.is_closed_implication_for_merge(s,2,1)
        assert not q

        s = sample_set_sequence_4()
        q = ball_comp_components.SetMerger.is_closed_implication_for_merge(s,1,2)
        assert not q

        s = sample_set_sequence_5()
        q = ball_comp_components.SetMerger.is_closed_implication_for_merge(s,1,2)
        assert q

        s = sample_set_sequence_6()
        q = ball_comp_components.SetMerger.is_closed_implication_for_merge(s,1,3)
        assert not q

        s = sample_set_sequence_7()
        q = ball_comp_components.SetMerger.is_closed_implication_for_merge(s,1,2)
        assert q

        s = sample_set_sequence_8()
        q = ball_comp_components.SetMerger.is_closed_implication_for_merge(s,1,2)
        assert not q
        q = ball_comp_components.SetMerger.is_closed_implication_for_merge(s,1,1)
        assert q

    def test__SetMerger__merges_at_index(self):

        ##
        s = sample_set_sequence_6()
        sm = ball_comp_components.SetMerger(s)

        for i in range(len(s)):
            q = sm.merges_at_index(i)

            if i == 3:
                assert len(q) > 0
                continue
            assert len(q) == 0
        return

    '''
    tests for basic pairwisescores
    '''
    def test__Ball__intersection_(self):
        b1,b2 = test_ball_pair_1()
        b3 = sample_ball_3()
        b4 = sample_ball_4()

        h(b1,b2)
        a = ball_comp_components.Ball.area_intersection_estimation(b1,b2)
        assert (a > 0), "incorrect intersection for balls 1,2"
        h(b1,b3)
        h(b2,b3)

        a = ball_comp_components.Ball.area_intersection_estimation(b2,b3)
        assert (a - b3.area()) < 10 ** -5, "incorrect intersection for balls 2,3"

        h(b2,b2)
        a = ball_comp_components.Ball.area_intersection_estimation(b2,b2)
        assert (a - b2.area()) < 10 ** -5, "incorrect intersection for balls 2,3"
        h(b3,b4)

    def test__Ball__add_element(self):
        c = np.array([0,0,0,0,0])
        b = ball_comp_components.Ball(c)

        # add these three points
        p1 = np.array([10,5.0,0.0,1.0,3.0])
        p2 = np.array([0,15.0,1.0,2.0,8.0])
        p3 = np.array([1.0,2.0,3.0,4.0,5.0])

        # calculate euclidean distance for each
        ## 11.61895003862225 17.146428199482248 7.416198487095663
        #ed1 = round(euclidean_point_distance(c,p1),5)
        ed1 = ball_comp_components.euclidean_point_distance(c,p1)
        ed2 = ball_comp_components.euclidean_point_distance(c,p2)
        ed3 = ball_comp_components.euclidean_point_distance(c,p3)

        b.add_element(p1)
        assert ed1 == b.radius, "incorrect @ p1, {}|{}".format(ed1,b.radius)

        b.add_element(p2)
        assert max([ed1,ed2]) == b.radius, "incorrect @ p2"

        b.add_element(p3)
        assert max([ed1,ed2,ed3]) == b.radius, "incorrect @ p3"

        return

    def test__Ball__add_element__radius_delta(self):
        c = np.array([0,0,0,0,0])
        b = ball_comp_components.Ball(c)

        # add these three points
        p1 = np.array([10,5.0,0.0,1.0,3.0])
        p2 = np.array([0,15.0,1.0,2.0,8.0])
        p3 = np.array([1.0,2.0,3.0,4.0,5.0])

        # calculate euclidean distance for each
        ## 11.61895003862225 17.146428199482248 7.416198487095663
        ed1 = ball_comp_components.euclidean_point_distance(c,p1)
        ed2 = ball_comp_components.euclidean_point_distance(c,p2)
        ed3 = ball_comp_components.euclidean_point_distance(c,p3)

        b.add_element(p1)
        assert ed1 == b.radius, "incorrect @ p1, {}|{}".format(ed1,b.radius)
        ##print("rd: ", b.radiusDelta)

        b.add_element(p2)
        assert max([ed1,ed2]) == b.radius, "incorrect @ p2"
        ##print("rd: ", b.radiusDelta)

        b.add_element(p3)
        assert max([ed1,ed2,ed3]) == b.radius, "incorrect @ p3"
        ##print("rd: ", b.radiusDelta)

    def test__ball_area(self):
        r = 4.0
        k1 = 197

        q = ball_comp_components.ball_area(r,k1)
        b2 = np.ones((k1,)) * r
        b = np.array([-b2,b2]).T
        q2 = ball_comp_components.area_of_bounds(b)
        assert q < q2
        return

    def test__Ball__intersection2(self):
        b3,b4 = sample_ball_3(),sample_ball_4()
        q = ball_comp_components.Ball.does_intersect(b3,b4)
        assert q,"balls intersect!"

        ai = ball_comp_components.Ball.area_intersection_estimation(b3,b4)
        assert ai >= 0.0, "incorrect estimation for area intersection"
        return

######## end: class<SetMerger> tests ############

######## start: class<Ball> tests ############

# TODO: further testing on this test required.
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
