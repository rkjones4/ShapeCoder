from scipy.spatial.transform import Rotation as R
import numpy as np
import math
from copy import deepcopy
import vis as vis

# Execution for 3D Shapes

# X, Y, and Z axis 
AXES = ['AX', 'AY', 'AZ']

# Defining orientations
RIGHTF = np.array([1.0, 0., 0.0])
TOPF = np.array([0.0, 1.0, 0.0])
FRONTF = np.array([0.0, 0., 1.0])

# Account for different scipy versioning
def safe_get_matrix(M):
    try:
        return M.as_matrix()
    except:
        return M.as_dcm()

def safe_from_matrix(M):
    try:
        return R.from_matrix(M)
    except:
        return R.from_dcm(M)

# cosine sim between two vectors
def vector_cos(norm1, norm2):
    norm1 = np.asarray(norm1)
    norm2 = np.asarray(norm2)
    dot = np.dot(norm1, norm2)
    magnitude = np.linalg.norm(norm1) * np.linalg.norm(norm2)
    if magnitude == 0.:
        return 0.
    return dot / float(magnitude)

# Base primitive 
class Cuboid:
    def __init__(
        self, W, H, D, X=0, Y=0, Z=0, RotM = None
    ):
        self.W = W
        self.H = H
        self.D = D
        self.X = X
        self.Y = Y
        self.Z = Z

        if RotM is None:
            self.RotM = np.array([
                [1.0,0.0,0.0],
                [0.0,1.0,0.0],
                [0.0,0.0,1.0]
            ])        
        else:
            self.RotM = RotM.copy()

    # get corners of cuboid
    def get_corners(self):
        center = np.array([self.X, self.Y, self.Z])
        lengths = np.array([self.W, self.H, self.D])

        mat = self.RotM

        dir_1 = mat @ np.array([1.0,0.0,0.0])
        dir_2 = mat @ np.array([0.0,1.0,0.0])
        dir_3 = mat @ np.array([0.0,0.0,1.0])
                                    
        dir_1 = dir_1/np.linalg.norm(dir_1)
        dir_2 = dir_2/np.linalg.norm(dir_2)
        dir_3 = dir_3/np.linalg.norm(dir_3)
            
        cornerpoints = np.zeros([8, 3])

        d1 = 0.5*lengths[0]*dir_1
        d2 = 0.5*lengths[1]*dir_2
        d3 = 0.5*lengths[2]*dir_3
            
        cornerpoints[0][:] = center - d1 - d2 - d3
        cornerpoints[1][:] = center - d1 + d2 - d3
        cornerpoints[2][:] = center + d1 - d2 - d3
        cornerpoints[3][:] = center + d1 + d2 - d3
        cornerpoints[4][:] = center - d1 - d2 + d3
        cornerpoints[5][:] = center - d1 + d2 + d3
        cornerpoints[6][:] = center + d1 - d2 + d3
        cornerpoints[7][:] = center + d1 + d2 + d3

        return cornerpoints

    # apply a rotation
    def apply_rotation(self, T):

        mat = safe_get_matrix(T)            
        self.RotM = mat @ self.RotM

    # orient normals to match directions
    def orient_normals(self, xdir, ydir, zdir):
        rt = np.asarray([1., 0., 0.])
        up = np.asarray([0., 1., 0.])
        fwd = np.asarray([0., 0., 1.])
        
        l = [
            (xdir, 0),
            (ydir, 1),
            (zdir, 2),
            (-1 * xdir, 3),                
            (-1 * ydir, 4),        
            (-1 * zdir, 5)
        ]

        rtdir, rind = sorted(deepcopy(l), key=lambda x: vector_cos(rt, x[0]))[-1]

        if rind >= 3:
            l.pop(rind)
            l.pop((rind+3)%6)    
        else:
            l.pop((rind+3)%6)    
            l.pop(rind)
        
        for i in range(0, 4):
            p_ind = l[i][1]
            if p_ind > max(rind, (rind+3)%6):
                l[i] = (l[i][0], l[i][1] - 2)
            elif p_ind > min(rind, (rind+3)%6):
                l[i] = (l[i][0], l[i][1] - 1)
                
        updir, upind = sorted(deepcopy(l), key=lambda x: vector_cos(up, x[0]))[-1]

        if upind >= 2:    
            l.pop(upind)
            l.pop((upind+2)%4)    
        else:
            l.pop((upind+2)%4)    
            l.pop(upind)
        
        fwdir, _ = sorted(l, key=lambda x: vector_cos(fwd, x[0]))[-1]

        return rtdir, updir, fwdir

    # find normals of right, top, and front faces
    def getFaceNormals(self):
        return self.RotM @ RIGHTF, self.RotM @ TOPF, self.RotM @ FRONTF

    # get angle information
    def get_EAng_info(self, orient):

        if orient:
            
            Xnorm, Ynorm, Znorm = self.getFaceNormals()
            Rnorm, TNorm, FNorm = self.orient_normals(
                Xnorm, Ynorm, Znorm
            )
            M = np.linalg.inv(np.stack([Rnorm, TNorm, FNorm]))

        else:                    
            M = self.RotM

        V = safe_from_matrix(M).as_euler('xyz', degrees=False)        

        XA = V[0]
        YA = V[1]
        ZA = V[2]

        # If we are close to no rotation, round to 0 rotation
        pi = 3.14
        
        if abs(abs(XA) - pi) < 0.01:
            XA = 0.

        if abs(XA) < 0.01 and abs(abs(YA) - pi) < 0.01:
            YA = 0.

        if abs(XA) < 0.01 and abs(YA) < 0.01 and abs(abs(ZA) - pi) < 0.01:
            ZA = 0.
            
        return XA, YA, ZA        

    # print parameter information of primitive
    def printinfo(self):
        NI_X, NI_Y, NI_Z = self.get_EAng_info(False)
        print(
            f' Dims: ({self.W}, {self.H}, {self.D})'
            ' | Pos: ({self.X}, {self.Y}, {self.Z})'
            ' | EAngles: ({NI_X}, {NI_y}, {NI_z}) '
        )
        
    def copy(self):
        n = Cuboid(self.W, self.H, self.D, self.X, self.Y, self.Z, self.RotM)
        return n

    # return a parameterization for this primtive
    def getPrimRep(self, PREC, orient=False):
        NI_X, NI_Y, NI_Z = self.get_EAng_info(orient)
        
        return (
            round(self.W, PREC), # width
            round(self.H, PREC), # height
            round(self.D, PREC), # depth
            round(self.X, PREC), # x pos
            round(self.Y, PREC), # y pos
            round(self.Z, PREC), # z pos
            round(NI_X, PREC), # x rot
            round(NI_Y, PREC), # y rot
            round(NI_Z, PREC),  # z rot
        )

# Shape is a collection of primitives
class Shape:
    def __init__(self):
        self.parts = []

# Defining an abstraction, has a name, variable arguments,
# logic for how those arguments turn into output, and optionally types of variables
class Abstraction:
    def __init__(self, name, varbs, logic, varb_types = None):
        self.name = name
        self.varbs = [v.strip() for v in varbs.split(',')]
        self.logic = logic
        self.varb_types = varb_types
        
    def substitute(self, params):
        s = self.logic
        for a,b in zip(self.varbs, params):
            s = s.replace(str(a), str(b))
        return s

# Program class
class Program:
    def __init__(self, abstractions = []):
        self.state = None
        self.abs_map = {a.name:a for a in abstractions}

    def reset(self):
        self._expr = None
        self.state = None
        self.soft_error = False
        self.hard_error = False
        
    def add_abst(self, a):
        self.abs_map[a.name] = a

    # execution logic for the following commands
        
    def ex_Cuboid(self, w, h, d):
        if w <= 0 or h <= 0 or d <= 0:
            # invalid to have negative dimension
            self.soft_error = True
            self.hard_error = True
            
        r = Cuboid(w, h, d)
        s = Shape()
        s.parts.append(r)
        return s

    def ex_SymRef(self, S, A):
        new_parts = []
        
        for r in S.parts:
            n = r.copy()
            
            if A == 'AX':

                if n.X >= 0:
                    # invalid , should only do forward axis reflection
                    self.soft_error = True
                
                n.X = -1 * n.X
                n.RotM[0] *= -1

                Xnorm, Ynorm, Znorm = n.getFaceNormals()
                Rnorm, TNorm, FNorm = n.orient_normals(
                    Xnorm, Ynorm, Znorm
                )
                n.RotM = np.linalg.inv(np.stack([Rnorm, TNorm, FNorm]))
                                
            elif A == 'AY':

                if n.Y >= 0:
                    # invalid , should only do forward axis reflection
                    self.soft_error = True
                
                n.Y = -1 * n.Y

                n.RotM[1] *= -1
                Xnorm, Ynorm, Znorm = n.getFaceNormals()
                Rnorm, TNorm, FNorm = n.orient_normals(
                    Xnorm, Ynorm, Znorm
                )
                n.RotM = np.linalg.inv(np.stack([Rnorm, TNorm, FNorm]))

                
            elif A == 'AZ':

                if n.Z >= 0:
                    self.soft_error = True
                
                n.Z = -1 * n.Z
                
                n.RotM[2] *= -1
                Xnorm, Ynorm, Znorm = n.getFaceNormals()
                Rnorm, TNorm, FNorm = n.orient_normals(
                    Xnorm, Ynorm, Znorm
                )
                n.RotM = np.linalg.inv(np.stack([Rnorm, TNorm, FNorm]))
                
            new_parts.append(n)

        S.parts += new_parts
        return S
        
    def ex_SymTrans(self, S, A, K, D):
        if D <= 0:
            # invalid, should not have negative distance
            self.soft_error = True
        
        new_parts = []
        for r in S.parts:
            for k in range(1, K+1):
                n = r.copy()

                perc = (k * 1.)/K
                
                if A == 'AX':                
                    n.X += perc * D

                elif A == 'AY':
                    n.Y += perc * D

                elif A == 'AZ':
                    n.Z += perc * D

                new_parts.append(n)

        S.parts += new_parts
        return S
    
    def ex_Move(self, S, X, Y, Z):        
        for r in S.parts:
            r.X += X
            r.Y += Y
            r.Z += Z
        return S

    # D -> radians
    def ex_Rotate(self, S, A, D):        
        T = R.from_euler(A, D, degrees=False)
        for r in S.parts:
            r.apply_rotation(T)            
        return S

    def ex_Union(self, A, B):
        s = Shape()
        s.parts += A.parts + B.parts
        return s

    # if statement only used by grammar, not inferred programs
    def ex_If(self, T, A, B):        
        if T is True:
            return A
        elif T is False:
            return B

    # parametric operators
    def ex_Add(self, X, Y):
        return X+Y

    def ex_Sub(self, X, Y):
        return X-Y

    def ex_Mul(self, X, Y):
        return X*Y

    def ex_Div(self, X, Y):
        if Y == 0:
            # div by 0
            self.soft_error = True
            self.hard_error = True
            return 0
        return X / Y

    def ex_Inv(self, X):
        return -1 * X

    # lazily execute, only paths of the program that will lead to output geometry
    def expand(self, p):
        if p[0]:
            return self.execute(p[1])
        else:
            return p[1]

    # main execution loop branching logic, depending on function
    def _execute(self, fn, params):
        
        if fn == 'Cuboid':
            params = [self.expand(p) for p in params]
            assert len(params) == 3
            return self.ex_Cuboid(
                float(params[0]),
                float(params[1]),
                float(params[2])
            )

        elif fn == 'Move':
            params = [self.expand(p) for p in params]
            assert isinstance(params[0], Shape)
            assert len(params) == 4
            return self.ex_Move(params[0], float(params[1]), float(params[2]), float(params[3]))

        elif fn == 'Rotate':
            params = [self.expand(p) for p in params]
            assert isinstance(params[0], Shape)
            assert len(params) == 3
            assert params[1][1].lower() in ['x','y','z']
            return self.ex_Rotate(params[0], params[1][1].lower(), float(params[2]))
            
        elif fn == 'Union':
            params = [self.expand(p) for p in params]
            assert isinstance(params[0], Shape)
            assert isinstance(params[1], Shape)
            assert len(params) == 2
            return self.ex_Union(params[0], params[1])

        elif fn == 'If':
            params = [self.expand(p) for p in params]
            assert params[0] in ('True', 'False')
            assert isinstance(params[1], Shape)
            assert isinstance(params[2], Shape)
            
            assert len(params) == 3
            return self.ex_If(params[0] == 'True', params[1], params[2])

        elif fn == 'Add':
            params = [self.expand(p) for p in params]
            assert len(params) == 2
            return self.ex_Add(float(params[0]), float(params[1]))

        elif fn == 'Sub':
            params = [self.expand(p) for p in params]
            assert len(params) == 2
            return self.ex_Sub(float(params[0]), float(params[1]))

        elif fn == 'Mul':
            params = [self.expand(p) for p in params]
            assert len(params) == 2
            return self.ex_Mul(float(params[0]), float(params[1]))

        elif fn == 'Div':
            params = [self.expand(p) for p in params]
            assert len(params) == 2
            return self.ex_Div(float(params[0]), float(params[1]))

        elif fn == 'Inv':
            params = [self.expand(p) for p in params]
            assert len(params) == 1
            return self.ex_Inv(float(params[0]))
                
        elif fn == 'SymRef':
            params = [self.expand(p) for p in params]
            assert len(params) == 2
            assert isinstance(params[0], Shape)
            assert params[1] in AXES
            return self.ex_SymRef(params[0], params[1])

        elif fn == 'SymTrans':
            params = [self.expand(p) for p in params]
            assert len(params) == 4
            assert isinstance(params[0], Shape)
            assert params[1] in AXES
            assert 'INT#' in params[2]
            return self.ex_SymTrans(params[0], params[1], int(params[2].split('#')[1]), float(params[3]))

        # Switch not used by inferred programs
        elif fn == 'Switch':
            params[0] = self.expand(params[0])
            
            if 'I_' in params[0]:
                ind = int(params[0].split('_')[1]) + 1
            elif 'INT#' in params[0]:
                ind = int(params[0].split('#')[1])
            else:
                assert False, f'bad params {params}'

            if ind < 1 or ind >= len(params):
                ind = 1
                self.soft_error = True                

            return self.expand(params[ind])
        
        elif fn in self.abs_map:
            # if we have an abstraction, call out to the abstraction to figure out how it should be executed
            params = [self.expand(p) for p in params]
            sub = self.abs_map[fn].substitute(params)            
            return self.execute(sub)
            
        else:
            assert False, f'bad function {fn}'

    # parse expression into functions and parameters, recursively execute functions
    def execute(self, expr):
        params = []

        fn = None
        cur = ''
        lp = 0
        rp = 0

        for c in expr:
            if c == '(':                
                lp += 1
                if lp == 1:
                    fn = cur.strip()
                    cur = ''
                    continue

            if c == ')':
                rp += 1
                
            if (c == ',' and (rp+1) == lp) or \
               (c == ')' and rp == lp):
                # record expand flag
                if '(' in cur:
                    params.append((True, cur )) 
                else:
                    params.append((False, cur.strip()))
                cur = ''
                continue

            cur += c

        return self._execute(fn, params)

    # Used by grammar to convert abstraction expression into base function expansion
    def parse(self, expr):
        s = ''
        cur = ''
        lp = 0
        rp = 0
        fn = None
        params = []
                
        for c in expr:
            if c == '(':                
                lp += 1
                if lp == 1:
                    fn = cur.strip()
                    cur = ''
                    continue

            if c == ')':
                rp += 1
                
            if (c == ',' and (rp+1) == lp) or \
               (c == ')' and rp == lp):
                if '(' in cur:
                    params.append(self.parse(cur))
                else:
                    params.append(cur.strip())
                cur = ''
                continue

            cur += c

        if fn in self.abs_map:
            sub = self.abs_map[fn].substitute(params)
            return self.parse(sub)
        else:
            return f"{fn}({', '.join(params)})"
            
    def printinfo(self):
        for c in self.state.parts:
            c.printinfo()

    # get primitive representation of execution state
    def getPrimitives(self, PREC=4, orient=False,skip_sort=False):
        prims = []
        for c in self.state.parts:
            prims.append(
                c.getPrimRep(PREC, orient)
            )

        if skip_sort:
            return prims
        prims.sort()        
        return prims

    # render a side by side viewing of two part sets
    def side_by_side_render(self,a,b,name):
        vis.side_by_side_render(a,b,name)

    # render a set of parts
    def render(self, name):
        vis.scene_render(self.state.parts, name)        

    # run executor
    def run(self, expr, name = None):
        self.reset()
        
        self._expr = expr
        self.state = self.execute(expr)
                
        if name is not None:
            self.render(name)

