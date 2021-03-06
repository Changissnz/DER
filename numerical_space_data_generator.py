from rssi import *

DEFAULT_SINGLE_WRITE_SIZE = 2000

"""
version of `one_random_noise` that can do proper
and improper bounds
"""
def one_random_noise_(parentBounds,bounds, noiseRange):

    if not is_proper_bounds_vector(bounds):
        pd = point_difference_of_improper_bounds(parentBounds,bounds)
        s1 = np.zeros((parentBounds.shape[0],))
        bx = np.array([s1,pd]).T
    else:
        bx = bounds
    return one_random_noise(bx,noiseRange)

# TODO: arg<sp>
'''
Generates data into file according to instructions given by arguments.

If arg<noiseRange> != None, then add noise to each point value.
'''
class NSDataInstructions:

    '''
    bInf := (bounds, startPoint, columnOrder, SSIHop, additionalUpdateArgs)
    rm := (mode := `relevance zoom` | `prg` | sequence::(relevant points), RCH)
    writeOutMode := literal to write out every point iterated
                    relevant to write out only points deemed relevant by RCH
    '''
    def __init__(self, bInf, rm,filePath,modeia,noiseRange = None,writeOutMode = "literal"):
        self.bInf = bInf
        self.rm = rm
        self.filePath = filePath
        self.fp = None
        self.load_filepath(modeia)
        self.nr = noiseRange
        assert writeOutMode in {'literal','relevant'}
        self.wom = writeOutMode
        self.rchPrev = None # used for writeOutMode == 'relevant'

        self.c = 0
        self.terminated = False
        self.bs = []
        return

    def load_filepath(self,modeia):

        # folder
        if "/" in self.filePath:
            # check exists
            #
            s = self.filePath[::-1]
            q = s.find('/')
            s = s[q + 1:][::-1]
            print("FP ",s)

            # make dir
            if not os.path.isdir(s):
                # make directory
                os.mkdir(s)
                modeia = 'w'

        self.fp = open(self.filePath,modeia)
        return

    def make_rssi(self):
        # mock a delaani
        if type(self.rm[0]) != str:
            delaani = ("relevance zoom",self.rm[1])
        else:
            delaani = self.rm

        #bounds,star
        self.rssi = ResplattingSearchSpaceIterator(self.bInf[0], self.bInf[1],\
                self.bInf[2],self.bInf[3],delaani,additionalUpdateArgs = self.bInf[4])
        return

    def next_batch(self):
        # load next bound in self.rm[0]
        if type(self.rm[0]) != str:
            delaani = ("relevance zoom",self.rm[1])
        else:
            delaani = self.rm

        # update the instructor if not `relevance zoom` or `prg`
        if type(self.rm[0]) != str and self.c > 1:
            if len(self.rm[0]) == 0:
                self.terminated = True
                return None

            q = self.rm[0][0]
            x = self.rm[0][1:]
            self.rm = (x,self.rm[1])

            # start point is left
            DEFAULT_START_POINT = np.copy(q[:,0])
            self.rssi.update_resplatting_instructor((q,DEFAULT_START_POINT))

        if self.rssi.terminated:
            return None

        if self.wom == "relevant":
            self.rchPrev = deepcopy(self.rm[1].apply)

        # fetch the bound
        if delaani[0] == "relevance zoom":
            #q = ResplattingSearchSpaceIterator.iterate_one_bound(self.rssi)
            dsws = DEFAULT_SINGLE_WRITE_SIZE
            q = ResplattingSearchSpaceIterator.iterate_one_batch(self.rssi,dsws)
        else: # prg
            q = []
            qc = 0

            while qc < DEFAULT_SINGLE_WRITE_SIZE:
                nx = next(self.rssi)
                if type(nx) == None:
                    break
                q.append(nx)
                qc += 1

        # filter out by relevance function
        if self.wom == "relevant":
            q = self.relevance_filter(q)
            ##print("LEN ", len(q))
        return q

    def relevance_filter(self,q):
        x = []
        for q_ in q:
            if self.rchPrev(q_):
                x.append(q_)
        self.rchPrev = None
        return x

    '''
    '''
    def add_noise_to_point(self,p):

        h = one_random_noise_(self.rssi.bounds,\
                self.rssi.ssi.de_bounds(),\
                self.nr)

        if type(self.rssi.ssi) is SkewedSearchSpaceIterator:
            p_ = self.rssi.ssi.inverse_round_value(p)
            p_ = vector_hop_in_bounds(p_,h,self.rssi.ssi.iBounds)
            return self.rssi.ssi.round_value(p_)
        else:
            return vector_hop_in_bounds(p,h,self.rssi.ssi.de_bounds())

    # TODO: untested
    def add_noise_to_batch(self,b):
        if type(self.nr) == type(None):
            ##print("here")
            return b

        for p in b:
            yield self.add_noise_to_point(p)

    def next_batch_(self):
        self.c += 1
        if type(self.fp) == type(None):
            return

        q = self.next_batch()
        if type(q) != type(None):
            if type(self.rm[0]) == str and self.rm[0] == 'relevance zoom':
                b = np.copy(self.rssi.ssi.bounds)
            else:
                b = np.copy(self.rssi.bounds)

            # check for adding noise
            if type(self.nr) != type(None):
                q2 = []
                for q_ in q:
                    q3 = self.add_noise_to_point(q_)
                    q2.append(q3)
                q = q2

            ##print("BEFORE ")
            ##print(q)
            q = [vector_to_string(q_,cr) + "\n" for q_ in q]
            ##print("AFTER")
            ##print(q)

            self.fp.writelines(q)
            self.fp.flush()
            
            # summarize
            l = len(q)
            self.bs.append([b,l])
        else:
            self.close()

    """
    size of batch
    bounds
    """
    def batch_summary(self):
        for (i,bs) in enumerate(self.bs):
            print("batch #",i)
            print("- bound")
            print(bs[0])
            print("- size ",bs[1])
            print()

    def close(self):
        self.fp.close()
        self.fp = None
