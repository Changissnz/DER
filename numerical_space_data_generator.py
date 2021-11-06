from rssi import *

DEFAULT_SINGLE_WRITE_SIZE = 2000
cr = lambda x: round(float(x),5)

'''
generates data into file
'''
class NSDataInstructions:

    '''
    bInf := (bounds, startPoint, columnOrder, SSIHop)
    rm := (mode := `relevance zoom` | `prg` | sequence::(relevant points), RCH)
    sp := # of resplat bounds | # of resplat samples
    '''
    def __init__(self, bInf, rm, sp, filePath,modeia):
        self.bInf = bInf
        self.rm = rm
        self.sp = sp
        self.filePath = filePath
        self.fp = None
        self.load_filepath(modeia)
        self.c = 0
        self.terminated = False
        return

    def load_filepath(self,modeia):

        # folder
        if "/" in self.filePath:
            # check exists
            #
            s = self.filePath[::-1]
            q = s.find('/')
            s = s[q + 1:][::-1]

            if not os.path.isdir(self.filePath):
                # make directory
                s = self.filePath[::-1]
                q = s.find('/')
                s = s[q + 1:][::-1]
                os.mkdir(s)
                modeia = 'w'
                
        self.fp = open(self.filePath,modeia)
        return

    def make_rssi(self):
        # mock a delaani
        if type(self.rm[0]) != str:
            assert is_2dmatrix(self.rm[0]), "invalid matrix"
            delaani = ("relevance zoom",self.rm[1])
        else:
            delaani = self.rm

        #bounds,star
        self.rssi = ResplattingSearchSpaceIterator(self.bInf[0], self.bInf[1],\
                self.bInf[2],self.bInf[3],delaani)
        return

    # TODO: not tested
    def next_batch(self):
        # load next bound in self.rm[0]
        delaani = self.rm
        if type(self.rm[0]) != str:
            if len(self.rm[0]) == 0:
                self.terminated = True
                return None

            q = self.rm[0][0]
            x = self.rm[0][1:]
            self.rm = (x,self.rm[1])
            delaani = ("relevance zoom",self.rm[1])

            # start point is left
            DEFAULT_START_POINT = np.copy(q[:,0])
            self.rssi = ResplattingSearchSpaceIterator(q, DEFAULT_START_POINT,\
                    self.bInf[2],self.bInf[3],delaani)

        if self.rssi.terminated:
            return None

        if delaani[0] == "relevance zoom":
            q = ResplattingSearchSpaceIterator.iterate_one_bound(self.rssi)
        else: # prg
            q = []
            qc = 0

            while qc < DEFAULT_SINGLE_WRITE_SIZE:
                nx = next(self.rssi)
                if type(nx) == None:
                    break
                q.append(nx)
                qc += 1
        return q

    def next_batch_(self):
        q = self.next_batch()

        if type(q) != type(None):
            q = [vector_to_string(q_,cr) for q_ in q]
            self.fp.writelines(q)
