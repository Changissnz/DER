'''
this file tests the subset of
methods found in `matrix methods`
that are for bounds-related calculations
'''
from .contextia_de_lo_vego_de_la_vegas import matrix_methods
import unittest
import numpy as np

class TestNumericalGeneratorClass(unittest.TestCase):

    def test__point_in_improper_bounds(self):

        pb = np.array([[0.4,12],[-30,40],[70,112],[10,14]])
        b = np.array([[10,2],[9,-28], [93,72],[13.5,11.0]])
        q = matrix_methods.split_improper_bound(pb,b)
        p1 = np.array([3,-25,77,11.7]) # not
        p2 = np.array([11,39,102,10.1])

        x = matrix_methods.point_in_improper_bounds(pb,b,p1)
        assert not matrix_methods.point_in_improper_bounds(pb,b,p1), "incorrect case 1"
        assert matrix_methods.point_in_improper_bounds(pb,b,p2), "incorrect case 2"
        return

    def test__point_on_improper_bounds_by_ratio_vector(self):

        pb = np.array([[0.4,12],[-30,40],[70,112],[10,14]])
        b = np.array([[10,2],[9,-28], [93,72],[13.5,11.0]])

        # case 1:
        v = np.array([0.5,0.5,0.5,0.5])
        q = matrix_methods.point_on_improper_bounds_by_ratio_vector(pb,b,v)
        assert matrix_methods.point_in_improper_bounds(pb,b,q), "incorrect case 1.1"
        assert matrix_methods.equal_iterables(q,np.array([11.8,25.5,103.5,10.25])), "incorrect case 1.2"

        # case 2:
        v = np.array([0.9,0.9,0.8,0.1])
        q = matrix_methods.point_on_improper_bounds_by_ratio_vector(pb,b,v)
        assert matrix_methods.point_in_improper_bounds(pb,b,q), "incorrect case 2.1"
        assert matrix_methods.equal_iterables(q,np.array([1.64,38.7,109.8,13.65])), "incorrect case 2.2"
        return

    def test__vector_ratio_improper(self):

        pb = np.array([[0.4,12],[-30,40],[70,112],[10,14]])
        b = np.array([[10,2],[9,-28], [93,72],[13.5,11.0]])

        p1 = np.array([11.8,25.5,103.5,10.25])
        p2 = np.array([1.64,38.7,109.8,13.65])

        vr1 = matrix_methods.vector_ratio_improper(pb,b,p1)
        assert matrix_methods.equal_iterables(vr1,np.array([0.5,0.5,0.5,0.5])), "incorrect case 1.1"

        vr2 = matrix_methods.vector_ratio_improper(pb,b,p2)
        assert matrix_methods.equal_iterables(vr2,np.array([0.9,0.9,0.8,0.1])), "incorrect case 1.2"
