from rssi import *

DEFAULT_SINGLE_WRITE_SIZE = 2000

'''
generates data into file
'''
class NSDataInstructions:

    '''
    bInf := (bounds, startPoint, columnOrder, SSIHop, additionalUpdateArgs)
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
            ##assert is_2dmatrix(self.rm[0]), "invalid matrix, got {}".format(self.rm[0])
            delaani = ("relevance zoom",self.rm[1])
        else:
            delaani = self.rm

        #bounds,star
        self.rssi = ResplattingSearchSpaceIterator(self.bInf[0], self.bInf[1],\
                self.bInf[2],self.bInf[3],delaani,additionalUpdateArgs = self.bInf[4])
        return

    # TODO: not tested
    def next_batch(self):
        # load next bound in self.rm[0]
        if type(self.rm[0]) != str:
            delaani = ("relevance zoom",self.rm[1])
        else:
            delaani = self.rm

        if type(self.rm[0]) != str and self.c:
            print("YAHH")
            if len(self.rm[0]) == 0:
                self.terminated = True
                return None

            q = self.rm[0][0]
            x = self.rm[0][1:]
            self.rm = (x,self.rm[1])

            # start point is left
            DEFAULT_START_POINT = np.copy(q[:,0])
            self.rssi.update_resplatting_instructor((q,DEFAULT_START_POINT))
                ##self.rssi.ssi.set_value(DEFAULT_START_POINT)
            """
            self.rssi = ResplattingSearchSpaceIterator(q, DEFAULT_START_POINT,\
                    self.bInf[2],self.bInf[3],delaani)
            """
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
        self.c += 1
        if type(self.fp) == type(None):
            return

        q = self.next_batch()
        if type(q) != type(None):
            if type(self.rm[0]) == str and self.rm[0] == 'relevance zoom':
                b = np.copy(self.rssi.ssi.bounds)
            else:
                b = np.copy(self.rssi.bounds)

            q = [vector_to_string(q_,cr) + "\n" for q_ in q]
            self.fp.writelines(q)
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
