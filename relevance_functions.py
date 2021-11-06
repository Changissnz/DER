from collections import defaultdict
from line import *
import random
from copy import  deepcopy
import operator

lambda_floatin = lambda x,b: x >= min(b) and x <= max(b)

lambda_pointinbounds = lambda p,b: point_in_bounds_(b,p)

def lambda_countpointsinbounds(p,b):
    if b.shape[0] == 1:
        ##q = len(np.where(lambda_floatin(p,b[0]) == True)[0])
        mask = np.logical_and(p >= b[0,0],p <= b[0,1])
        print("MASK 0")
        print(mask)
    else:
        print("MASK")
        mask = np.logical_and(p >= b[:,0],p <= b[:,1])
        print(mask)
    q = len(np.where(mask == True)[0])
    print("Q: ",q)
    return q

def lambda_ratiopointsinbounds(p,b):
    x = lambda_countpointsinbounds(p,b)
    return zero_div(x,len(p),np.inf)

def random_select_k_unique_from_sequence(s, k):
    s = list(s)
    assert len(s) >= k
    random.shuffle(s)
    return s[:k]


################################### TODO: delete these functions after refactor

# TODO:
def relevance_zoom_func_1(referencePoint,boundsDistance,activationThreshold):
    # TODO: check for 0-case
    assert boundsDistance >= 0.0, "invalid bounds distance {}".format(boundsDistance)
    assert activationThreshold >= 0.0 and activationThreshold <= 1.0, ""
    return lambda p: euclidean_point_distance(p, referencePoint) <= activationThreshold * boundsDistance

# TODO: add this
"""
tests if all
"""
def relevance_zoom_func_2(referencePoint, modulo, activationThreshold):
    rp = np.array(referencePoint, dtype = "int")
    lim = referencePoint.shape[0]
    def p(x):
        q = (rp - np.array(x,dtype="int")) % 2
        return lambda p: True if len(q == 1.0) >= lim else False
    return p


## TODO: delete?
'''
a sample relevance function to help demonstrate the work of CenterResplat.

return:
- function(vector) ->
        True if all values in (point % modulo) fall within moduloPercentileRange
'''
def relevance_func_2(modulo, moduloPercentileRange):
    assert modulo >= 0.0, "invalid modulo"
    assert is_valid_point(moduloPercentileRange)
    assert min(moduloPercentileRange) >= 0.0 and max(moduloPercentileRange) <= 1.0
    assert moduloPercentileRange[0] <= moduloPercentileRange[1]

    minumum,maximum = moduloPercentileRange[0] * modulo,moduloPercentileRange[1] * modulo

    def f(p):
        p_ = p % modulo
        return np.all(p_ >= minumum) and np.all(p_ <= maximum)

    return f

# should addOn? be opt.
"""
boolean function, addon determines
"""
def vector_modulo_function_with_addon(modulo, addOn):
    def x(v):
        q = np.array(v,dtype="int")
        v_ = q % modulo
        return addOn(v_)
    return x

##### add-on functions : vector -> bool

"""
"""
def vf_vector_reference(vr, pw):
    def x(v):
        return pw(vr,v)
    return x

##### merge below 2

### list of pairwise functions

## standard functions
# np.cross
# np.dot
# np.multiply
def ndim_dot_referential():
    return -1

def ndim_dot_path():
    return -1


# do n-dim degree

"""
pairwise vector function 1
"""
def pairwise_vector_function_1(v1,v2,op):
    return -1

"""
uses referential vector and a vector pairwise function that outputs
 a singleton, rv will be the first argument.
"""
def vector_function_rv_with_addon_dec(rv, pf, addOn):

    def x(rv1):
        return addOn(pf(rv,rv1))
    return x

def vector_function_rv_with_addon_nondec(rv, pf, addOn):
    return -1

"""
outputs elements by indices

addOn := function that outputs either singleton or pair
"""
def subvector_iselector(indices):
    def a(v):
        return v[indices]
    return a

"""
m is index|value selector function for arg. to func<addOn>
"""
def m(v,addOn,iov,outputType):
    assert iov in [0,1,2]

    x = []
    for t in enumerate(v):
        qi = t[iov] if iov != 2 else t
        if addOn(qi):
            if outputType == 1: q = t[1]
            else: q = t
            x.append(q)
    return np.array(x)

"""
outputs elements by addOn

addOn := function that outputs either singleton or pair
"""
def subvector_selector(addOn, inputType =1, outputType = 1):
    assert inputType in [0,1,2], "invalid input"
    assert outputType in [1,2], "invalid output"

    def m_(v):
        return m(v,addOn,inputType,outputType)

    return m_

##### TODO: unused
def is_in(v,b):
    if type(v) in [type(47.0),type(47)]:
        return v in b
    elif is_vector(v):
        q = indices_of_vector_in_matrix(v,b)
        return len(q) > 0

lambda__in_subset = lambda x,s: is_in(x,s)

##### END TODO: unused

## $
'''
rangeReq := proper bounds vector
'''
def addon_singleton__bool__criteria_range_each(rangeReq):
    assert is_proper_bounds_vector(rangeReq), "invalid ranges"
    # rangeReq := length 1 or vr.shape[0]
    if rangeReq.shape == (1,2):
        q = rangeReq[0]
        p = lambda v: np.all(v >= q[0]) and np.all(v <= q[1])
    else:
        p = lambda v: point_in_bounds(rangeReq,v)
    return p

"""
rf := point
dm := func((v1,v2)->float), distance measure between (rf,v)
dt := distance threshold, float
cf := comparator function on (dist,dt)
"""
def addon_singleton__bool__criteria_distance_from_reference(rf, dm, dt,cf):
    return lambda v: cf(dm(rf,v),dt)


def addon_pwcomp__x():
    #dm := func((v1,v2)->v3), distance measure between (rf,v)
    return -1

# RChain is a sequence-like structure of nodes with
# modification capabilities
def lambda__vector_modulo(m):
    return lambda v: v % m

"""
class that acts as a node-like structure,

node is designed to be used w/ both

(1) referential data (from outside of chain)
and
(2) (standard operator,operand)

"""
class RCInst:

    """
    rf: reference value, as argument to dm(v,rf)
    dm: function f(v,rf)
    cf: function f(v,*dt)
        - operator.lt(a, b)
        - operator.le(a, b)
        - operator.eq(a, b)
        - operator.ne(a, b)
        - operator.ge(a, b)
        - operator.gt(a, b)
        - lambda_floatin
        - np.cross
        - np.dot
        - np.multiply
    dt: value, use in the case of decision
    """
    def __init__(self):
        #self.instName = instructionName
        self.rf = None
        self.dm = None
        self.cf = None
        self.dt = None
        ##self.ct = ()

    """
    """
    def mod_cf(self,dcf):
        self.cf = dcf(self.cf)
        return

    def set_reference_data(self,rf,dm):
        self.load_var_ref(rf)
        self.load_var_dm(dm)
        return

    ############# some functions

    def branch_at(self,n,i):
        return -1

    def load_var_ref(self,rf):
        self.rf = rf

    """
    """
    def load_var_cf(self,cf):
        self.cf = cf

    """
    threshold variable, use as cf(v,dt)
    """
    def load_var_dt(self,dt):
        self.dt = dt

    def path_type(self):
        if type(self.rf) != type(None):
            return "ref"
        return "dir"

    # TODO: class does not know.
    def output_type():
        return -1

    """
    dm := function on (rf,v)
    """
    def load_var_dm(self,dm):
        self.dm = dm

    """
    """
    def load_path(self):
        # deciding path
        if type(self.dt) != type(None):
            # for output type bool|float
            if self.path_type() == "ref":
                # calculates distance from reference
                self.f = lambda v: self.cf(self.dm(self.rf,v),self.dt)#*self.ct,self.dt)
            else:
                self.f = lambda v: self.cf(v, self.dt)#*self.ct,self.dt) # *self.ct
        # non-deciding path
        else:
            if self.path_type() == "ref":
                # calculates distance from reference
                self.f = lambda v: self.cf(self.dm(self.rf,v))#,*self.ct)
            else:
                self.f = lambda v: self.cf(v)#,*self.ct)
        return deepcopy(self.f)

    # distance threshold is float
    ##dt = 20.0

    # rangeReq := bounds
    # rf := point

"""
RChainHead is a node-like structure
"""
class RChainHead:

    def __init__(self):
        self.s = []
        self.vpath = []

    """
    es := expression string,
    """
    @staticmethod
    def make_linker_func(es):
        return -1

    def link_rch(self,rch,linkerFunc, prev = False):
        if prev:
            return linkerFunc(self,rch)
        return linkerFunc(self,rch)

    def vpath_subset(self,si):
        return [x for (i2,x) in enumerate(self.vpath) if i2 in si]

    def load_cf_(self, rci,cfq):
        if type(cfq) == type(()):
            xs = tuple(self.vpath_subset(cfq[1]))
            cf = cfq[0](*xs)

            # below method
            rci.load_var_cf(cf)
        else:
            rci.load_var_cf(cfq)

    """
    list of all possible functions:
    -

    kwargz:

    [0] := r|nr

        if r:
            [1] rf
            [2] dm | (dm,selectorIndices)
            [3] cf | (cf,selectorIndices)
            [?4] dt

        if nr:
            [1] cf | (cf,selectorIndices)
            [?2] dt

    * selectorIndices refer values in vpath
    """
    def make_node(self,kwargz):#, outputType = "bool"):
        #assert outputType in ["vector-real","vector-bool", "float","bool"]
        assert kwargz[0] in ["r","r+","nr"]

        rci = RCInst()
        if kwargz[0] == "r":
            assert len(kwargz) in [4,5], "invalid length for kwargs"
            rci.set_reference_data(kwargz[1],kwargz[2])
            self.load_cf_(rci,kwargz[3])
            try: rci.load_var_dt(kwargz[4])
            except: pass

        elif kwargz[0] == "nr":
            assert len(kwargz) in [2,3], "invalid length for kwargs"
            self.load_cf_(rci,kwargz[1])
            try: rci.load_var_dt(kwargz[2])
            except: pass

        elif kwargz[0] == "r+":
            return -1

        rci.load_path()
        return rci

    def add_node_at(self, kwargz, index = -1):
        assert index >= -1, "invalid index"
        n = self.make_node(kwargz)
        if index == -1:
            self.s.append(n)
        else:
            self.s.insert(index,n)
        return -1

    """
    applies the composite function (full function path)
    onto v
    """
    def apply(self,v, index = 0):
        i = 0
        v_ = np.copy(v)
        self.vpath = [v_]

        while i < len(self.s):
            q = self.s[i]
            v_ = q.f(v_)
            self.vpath.append(v_)
            i += 1
        return v_

    def cross_check(self):
        return -1

    def merge(self):
        return -1

    def __next__(self):
        return -1

####----------------------------------------------------------

###### START: helper functions for next section

def boolies(v_):
    return v_ == True

def column_selector(columns, flatten = False):

    def p(v):
        q = v[:,columns]
        if flatten: return q.flatten()
        return q
    return p

def vector_index_selector(indices):
    def p(v):
        return v[indices.astype('int')]
    return p

def vector_index_inverse_selector(v):
    def p(indices):
        return v[indices.astype('int')]
    return p


###### END: helper functions for next section

###### START: functions used for relevance zoom

'''
'''
def RCHF__point_in_bounds(b):
    kwargs = ['nr', lambda_pointinbounds, b]
    # f : v -> v::bool
    rc = RChainHead()
    rc.add_node_at(kwargs)
    return rc.apply

'''
'''
def RCHF__point_in_bounds_subvector_selector(b):
    def qf(xi):
        return operator.le(xi[1],b[xi[0],1]) and operator.ge(xi[1],b[xi[0],0])

    q2 = subvector_selector(qf,inputType = 2,outputType = 1)
    kwargs = ['nr', q2]
    rc = RChainHead()
    rc.add_node_at(kwargs)
    return rc.apply

def RCHF__point_distance_to_references(r,ed0):

    return -1

"""
pass string is boolean expression
"""
def RCHF__point_distance_to_references_dec(r,ed0,passString):
    return -1

    # pass vector<distances> -> <bool> --> pass-string -> bool

# TODO: incorporate expresso now.

def ffilter(v,f):

    t,l = [],[]
    for v_ in v:
        if f(v_): t.append(v_)
        else: l.append(v_)

    return t,l

# make an rch by the following:
'''
using reference rf,

odd_multiply
even

[0] multiplier of even indices
[1] multiplier of odd indices

recent memory
+ - => -
- - => -
- + => +
+ + => +

past memory
+ - => +
- - => -
- + => -
+ + => +
'''
def rpmem_func(rf,rOp):
    rfo0,rfo1 = ffilter(rf,lambda i: i % 2)
    r1 = np.product(rfo1) if rOp else np.product(rfo0)

    def p(v):
        v0,v1 = ffilter(v,lambda i: i % 2)
        r2 = np.product(v1) if rOp else np.product(v0)
        return int(r2) % 2

    # try swapping them

    return p

def is_valid_pm(pm):
    assert pm.shape[1] == 2, "invalid pm shape"
    tf1 = len(np.unique(pm[:,0])) == pm.shape[0]
    tf2 = len(np.unique(pm[:,1])) == pm.shape[0]
    return tf1 and tf2

def is_proper_pm(pm):
    s = is_valid_pm(pm)
    if not s: return s
    s1 = min(pm[:,0]) == 0 and max(pm[:,0])  == pm.shape[0]
    s2 = min(pm[:,1]) == 0 and max(pm[:,1])  == pm.shape[0]
    return s1 and s2

# a version of rp mem that uses a permutation map
# TODO:
"""
the original func<rpmem_func> operates on the binary choice

"""
def rpmem_func__pm(rfs,pm):
    assert is_proper_pm(pm), "[0] invalid permutation map"
    assert pm.shape[0] == len(rfs) + 1,"[1] invalid permutation map"
    return -1

'''
'''
def is_valid_subset_sequence(s,n):
    q = []
    for s_ in s:
        q.extend(list(s))

    tf0 = len(q)
    q = np.unique(q)
    tf1 = len(q)
    if tf0 != tf1: return False

    m0,m1 = min(q),max(q)
    return m0 == 0 and m1 == n - 1

'''
'''
def func__subset__boolvector_to_bool(bv, sseq, boolSetFunc):
    q = is_valid_subset_sequence(sseq,bv.shape[0])

    return -1

# TODO: boolSet function should use `expresso` to construct AND,OR,NOT


###### END: functions used for relevance zoom
