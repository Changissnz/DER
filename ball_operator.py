from ball_comp_components import *

"""
an operator that can:
- navigate balls by the "division"-scheme
- split balls into sub-balls


# Ball splitting procedures
- split value
* literal: float value, the radius of the subball
* dividor: float value, divides the distance between the target point of the sub-ball                        and the ball center

- sub-ball radius
* static: sub-ball will attempt to fulfill radius value requirement
* minimal: algorithm will use sub-ball's actual radius, after the
           method <add_ball_points_to_subball>
"""
# CAUTION: modifies arg<ball> and may results in misallocated variables
# TODO: use `savePointCopy`
# TODO: case: ball falls under (literal,r)
class BallOperator:

    def __init__(self, ball,savePointCopy = False):
        self.ball = ball
        self.savePointCopy = savePointCopy
        self.clear_cache()

        # variables for navigation
        self.location = None
        self.basis = None
        self.nHop = None
        self.counterLocation = None

        # float value, floor corresponds to division
        self.divMarker = 0.0

        ## used for ball-splitting operations
        self.subballs = []
        self.subballRadii = []

        # target point is running radial reference of ball's points
        # not in sub-ball
        self.targetPoint = None
        return

    def clear_cache(self):
        self.cache = np.empty((0,self.ball.center.shape[0]))
        self.cache2 = None
        return


    ####################### TODO:
    ##################################### start: navigation

    # TODO: add arg<ordered>
    @staticmethod
    def nav_basis(point):
        assert is_vector(point), "invalid point"
        x = [point]
        for i in range(1,point.shape[0]):
            x2 = point[:point.shape[0] - i]
            x1 = point[point.shape[0] - i:]
            x_ = np.append(x1,x2)
            x.append(x_)
        return np.array(x)

    """
    a counterpoint is the point of equal distance to ||`point` - `center`||
    on the line containing the points, `point` and `center`.
    """
    @staticmethod
    def calculate_counterpoint(point,center):
        d = point - center
        return center - d

    def set_navigation(self,p):
        basis = BallOperator.nav_basis(p)
        self.location = p
        self.bGen = generate_possible_binary_sequences(p.shape[0], [], elements = [1,-1])
        self.counterLocation = BallOperator.calculate_counterpoint(self.location,self.ball.center)

    """
    """
    def nav_one(self,operator):
        return -1


    """
    """
    @staticmethod
    def basis_division(point,counterpoint,multiplier):
        assert is_vector(point) and is_vector(counterpoint), "invalid args. point and center"

        x = []
        for (i,p) in enumerate(point):
            if multiplier[i] == 1: x.append(p)
            else: x.append(counterpoint[i])
        x = np.array(x)
        return BallOperator.nav_basis(x)

    @staticmethod
    def hop_in_division(b, hop, center):
        assert len(b.shape) == 2 and b.shape[0] == b.shape[1], "incorrect shape for basis"
        assert hop >= 0.0 and hop <= 1.0, "invalid hop"

        i = int(math.floor(hop / (1/b.shape[0])))
        i2 = (i + 1) % b.shape[0]
        q = np.vstack((center,b))
        r = hop - i * 1/b.shape[0]
        c = r / (1 / b.shape[0])
        return q[i] + c * (q[i2] - q[i])

    ##################################### end: navigation

    """
    Continually splits the ball into sub-balls until all points lie w/in 1+ sub-ball.

    arguments:
    - split := (float,literal|dividor)
    - subballRadiusType := static|minimal
    """
    def run_subball_split(self, split,subballRadiusType = "static"):
        assert split[1] in {"literal","dividor"}, "invalid split"
        assert subballRadiusType in {"static","minimal"}, "invalid subball radius type"

        self.subballs = []
        self.subballRadii = []

        if split[1] == "literal" and self.ball.radius <= split[0]:
            self.subballs.append(deepcopy(self.ball))
            self.ball.data.newData = np.empty((0,self.ball.data.newData.shape[1]))
            return

        while self.ball.data.newData.shape[0] > 0:
            self.update_radial_ref()
            if split[1] == "dividor":
                sr = cr(euclidean_point_distance(self.ball.center,\
                    self.targetPoint)  / split[0])
            else:
                sr = cr(split[0])

            sb = self.subball_at_radial_ref(sr,self.targetPoint)
            self.subballs.append(sb)
            q = sr if subballRadiusType == "static" else sb.radius
            self.subballRadii.append(q)

            self.cache2 = np.copy(sb.data.newData)
            self.add_cache_points_to_subball(sb,q)
            self.cache = np.vstack((self.cache,self.cache2))

        return

    def antiradial_ref(self,subballRadius, radialRef):
        i = np.argmin([abs(euclidean_point_distance(p,radialRef) - subballRadius) for p in self.ball.data.newData])
        return self.ball.data.newData[i]

    def subball_at_radial_ref(self, subballRadius, radialRef):
        # draw line segment (radialRef,e) towards ball center of length
        # `subballRadius`; e is the center

        #### old calculation for center
        # delta = self.ball.center - radialRef
        # c = zero_div(subballRadius,euclidean_point_distance(self.ball.center,radialRef),0.0)
        # newCenter = np.round(radialRef + c * delta,5)

        #### new calculation for center
        ref2 = self.antiradial_ref(subballRadius,radialRef)
        newCenter = ref2
        b = Ball(newCenter)
        subballRadius = cr(euclidean_point_distance(b.center,radialRef))

        # save remaining target ball points to subball
        self.add_ball_points_to_subball(b,subballRadius)
        return b

    ############## start: method for after post-add

    def save_subball_data_to_cache(self,sb):
        self.cache = np.vstack((self.cache,sb.data.newData))

    def add_cache_points_to_subball(self,sb,r):
        for x in self.cache:
            if euclidean_point_distance(sb.center,x) <= r:
                sb.add_element(x)

    ############## end: method for after post-add

    # QUESTION: should ball center also be updated?
    """
    """
    def update_radial_ref(self):
        index = np.argmax([euclidean_point_distance(self.ball.center,x) for x in self.ball.data.newData])
        self.targetPoint = self.ball.data.newData[index]

    def add_ball_points_to_subball(self,subball, subballRadius):
        BallOperator.ball_points_to_another(self.ball,subball,subballRadius,True)

    """
    Transfers each point p from `b1` to `b2` if p falls w/in distance of
    `b2Radius`. If `deleteFromB1` is set to True, delete p from `b1` if `b2`
    accepts p.
    """
    @staticmethod
    def ball_points_to_another(b1,b2,b2Radius,deleteFromB1):

        ### case:
        # farthest possible point in b1 with respect to b2 falls out of b2Radius
        ed = cr(euclidean_point_distance(b1.center,b2.center))
        if ed - b1.radius > b2Radius:
            return

        i = 0
        while i < b1.data.newData.shape[0]:
            x = b1.data.newData[i]
            ed = cr(euclidean_point_distance(x,b2.center))
            if ed <= b2Radius:
                b2.add_element(x)

                if deleteFromB1:
                    b1.data.newData = np.concatenate(\
                        (b1.data.newData[:i],b1.data.newData[i + 1:]))
                else:
                    i += 1
            else:
                i += 1
        return


    ########################### start: volume estimation

    ########################### end: volume estimation
