'''
file determines variance measures
'''
import numpy as np

# INPUT YOUR NUMBER HERE.
rng = np.random.default_rng()

'''
description:
- generates a random matrix with each column
  containing randomly-generated values corresponding
  to its (mean, std. dev.).

return:
- np.array, shape is (n, |MInfo|)
'''
def random_matrix_by_normaldist_values(n, MInfo):

    q = np.zeros((n, len(MInfo)))
    for i in range(n):
        q[i] = one_random_sample_by_normaldist_values(MInfo)
    return q

'''
'''
def one_random_sample_by_normaldist_values(MInfo):

    s = np.zeros(len(MInfo),)
    for (i,m) in enumerate(MInfo):
        s[i] = rng.normal(m[0], m[1])
    return s


#############################################################

'''
from numpy.random import default_rng

rng = default_rng()
numbers = rng.choice(20, size=10, replace=False)
'''



########

# geometric samples
'''
>>> z = np.random.default_rng().geometric(p=0.35, size=10000)
'''

'''
random.Generator.multinomial(n, pvals, size=None)
'''
