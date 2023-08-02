import matplotlib.pyplot as plt
import matplotlib.patches as patches

# An executor for 2D shapes

# Primitive class
class Rectangle:
    def __init__(self, W, H, X=0, Y=0):
        self.W = W
        self.H = H
        self.X = X
        self.Y = Y

    def printinfo(self):
        print(f' Dims: ({self.W}, {self.H}) | Pos: ({self.X}, {self.Y})')

    def copy(self):
        n = Rectangle(self.W, self.H, self.X, self.Y)
        return n

# Shape is collection of primitives
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
    
    def ex_Rectangle(self, w, h):
        if w <= 0 or h <= 0:
            # invalid 
            self.soft_error = True
            self.hard_error = True
            
        r =  Rectangle(w, h)
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

            elif A == 'AY':

                if n.Y >= 0:
                    # invalid , should only do forward axis reflection
                    self.soft_error = True
                
                n.Y = -1 * n.Y

            new_parts.append(n)

        S.parts += new_parts
        return S

    def ex_SymTrans(self, S, A, K, D):

        if D <= 0:
            # invalid, distance should be positive
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

                new_parts.append(n)

        S.parts += new_parts
        return S
    
    def ex_Move(self, S, X, Y):        
        for r in S.parts:
            r.X += X
            r.Y += Y
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
    def ex_Inv(self, X):
        return -1 * X
    
    def ex_Add(self, X, Y):
        return X+Y

    def ex_Sub(self, X, Y):
        return X-Y

    def ex_Mul(self, X, Y):
        return X*Y

    def ex_Div(self, X, Y):
        if Y == 0:
            # divide by 0
            self.soft_error = True
            self.hard_error = True
            return 0
        return X / Y

    # main execution loop branching logic, depending on function
    def _execute(self, fn, params):
        if fn == 'Rectangle':
            assert len(params) == 2
            return self.ex_Rectangle(float(params[0]), float(params[1]))

        elif fn == 'Move':
            assert isinstance(params[0], Shape)
            assert len(params) == 3
            return self.ex_Move(params[0], float(params[1]), float(params[2]))

        elif fn == 'Union':
            assert isinstance(params[0], Shape)
            assert isinstance(params[1], Shape)
            assert len(params) == 2
            return self.ex_Union(params[0], params[1])

        elif fn == 'If':
            assert params[0] in ('True', 'False')
            assert isinstance(params[1], Shape)
            assert isinstance(params[2], Shape)
            
            assert len(params) == 3
            return self.ex_If(params[0] == 'True', params[1], params[2])

        elif fn == 'Inv':
            assert len(params) == 1
            return self.ex_Inv(float(params[0]))
        
        elif fn == 'Add':
            assert len(params) == 2
            return self.ex_Add(float(params[0]), float(params[1]))

        elif fn == 'Sub':
            assert len(params) == 2
            return self.ex_Sub(float(params[0]), float(params[1]))

        elif fn == 'Mul':
            assert len(params) == 2
            return self.ex_Mul(float(params[0]), float(params[1]))

        elif fn == 'Div':
            assert len(params) == 2
            return self.ex_Div(float(params[0]), float(params[1]))

        elif fn == 'SymRef':
            assert len(params) == 2
            assert isinstance(params[0], Shape)
            assert params[1] in ('AX', 'AY')
            return self.ex_SymRef(params[0], params[1])

        elif fn == 'SymTrans':
            assert len(params) == 4
            assert isinstance(params[0], Shape)
            assert params[1] in ('AX', 'AY')
            assert 'INT#' in params[2]
            return self.ex_SymTrans(params[0], params[1], int(params[2].split('#')[1]), float(params[3]))
        
        elif fn in self.abs_map:
            # if we have an abstraction, call out to the abstraction to figure out how it should be executed
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
                if '(' in cur:
                    params.append(self.execute(cur))
                else:
                    params.append(cur.strip())
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

    # get a primitive representation of the shape
    def getPrimitives(self, PREC=4,orient=None):
        prims = []
        for c in self.state.parts:
            prims.append((
                round(c.W, PREC),
                round(c.H, PREC),
                round(c.X, PREC),
                round(c.Y, PREC), 
            ))

        prims.sort()
        return prims

    # render shape, and save to name file, or show it if name is None
    def render(self, name):
        plt.clf()
        plt.xlim([-1., 1.])
        plt.ylim([-1., 1.])
        for c in self.state.parts:
            lc = (c.X - (c.W/2.), c.Y - (c.H/2.))
            plt.gca().add_patch(patches.Rectangle(lc, c.W, c.H, facecolor = 'lightgrey', edgecolor = 'black', fill = True))
        if name is not None:
            plt.savefig(f'{name}.png')
        else:
            plt.show()

    # run executor
    def run(self, expr, name = None):
        self.reset()
        self._expr = expr
        self.state = self.execute(expr)

        if name is not None:
            self.render(name)
        
