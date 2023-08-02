import executor as ex
from executor import Abstraction as Abs
import random
import sys

# A simple grammar for 2D chair like structures

# Useful parametric expression
def HF2():
    return Abs(
        'HF2',
        'V_a, V_b',
        "Mul(Sub(Div(V_a, 2.), Div(V_b, 2.)), -1.)"        
    )

# Divide by 2
def HF1():
    return Abs(
        'HF1',
        'V_a',
        "Div(V_a, 2.)"
    )

# First kind of back
def BACKV1():
    return Abs(
        'BACKV1',
        'V_X, V_Y, V_o, V_u, V_v, V_k',
        """
        Union(
          Move(Rectangle(V_X, V_v), 0., Mul(HF2(V_Y, V_v), -1.)),
          SymTrans(
            Move(Rectangle(V_u, Sub(V_Y, V_v)), Add(HF2(V_X, V_u), Mul(V_X, V_o)), HF2(V_Y, Sub(V_Y, V_v))),
            AX,
            V_k,
            Sub(V_X, Add(V_u, Mul(Mul(V_X, V_o), 2.0)))
          )
        ) 
        """
    )

# Second kind of back
def BACKV2():
    return Abs(
        'BACKV2',        
        'V_X, V_Y, V_d, V_o, V_u, V_v, V_k',
        """
        Union(
          BASEV1(V_X, V_Y, V_u),
          SymTrans(
            Move(Rectangle(Sub(V_X, Mul(V_u, 2.)), V_v), 0., Add(HF2(V_Y,V_v), Mul(V_o, V_Y))),
            AY,
            V_k,
            Sub(V_Y, Add(V_v, Mul(2.0, Mul(V_o, V_Y))))
          )
        )
        """
    )

# Third kind of back
def BACKV3():
    return Abs(
        'BACKV3',        
        'V_X, V_Y, V_o, V_u, V_v',
        """
        Union(
          SymRef(Move(Rectangle(V_X, V_v), 0., HF2(V_Y, V_v)), AY),
          SymRef(Move(Rectangle(V_u, Sub(V_Y, Mul(V_v, 2.))), Add(HF2(V_X, V_u), Mul(V_X, V_o)), 0.), AX)
        )
        """
    )

# Back is basically a switch statement over 4 kinds of back
def BACK():
    return Abs(
        'BACK',        
        'V_b5, V_b6, V_X, V_Y, V_d, V_o, V_u, V_v, V_k',
        """
        If(V_b5,
          If(V_b6,
            Rectangle(V_X, V_Y),        
            BACKV1(V_X, V_Y, V_o, V_u, V_v, V_k)
          ),
          If(V_b6,
            BACKV2(V_X, V_Y, V_d, V_o, V_u, V_v, V_k),
            BACKV3(V_X, V_Y, V_o, V_u, V_v)
          )
        )            
        """
    )

# Base version 2
def BASEV2():
    return Abs(
        'BASEV2',
        'V_X, V_Y, V_a, V_b',
        """
        Union(
          Move(Rectangle(V_X, V_b), 0., HF2(V_Y,V_b)),
          Move(Rectangle(V_a, Sub(V_Y, V_b)), 0., Mul(HF2(V_Y, Sub(V_Y, V_b)), -1.))
        )
        """
    )

# Base version 3
def BASEV3():
    return Abs(
        'BASEV3',
        'V_X, V_Y, V_a, V_b',
        """
        Union(
          BASEV1(V_X, V_Y, V_a),
          SymRef(
            Move(Rectangle(Sub(V_X, Mul(V_a, 2.)), V_b), 0., HF2(V_Y, V_b)),
            AY
          )
        ) 
        """
    )

# Base version 4
def BASEV4():
    return Abs(
        'BASEV4',
        'V_X, V_Y, V_a, V_b, V_c',
        """
        Union(
          BASEV1(V_X, V_Y, V_a),
          Move(Rectangle(Sub(V_X, Mul(V_a, 2.)), V_b), 0., Add(HF2(V_Y, V_b), Mul(V_c, V_Y)))
        ) 
        """
    )

# Base version 1, used as sub function in some other base versions
def BASEV1():
    return Abs(
        'BASEV1',
        'V_X, V_Y, V_a',
        """
        SymRef(
          Move(  
            Rectangle(V_a, V_Y),
            HF2(V_X, V_a),
            0.
          ), AX
        )
        """
    )

# Base is another switch statement over different base types
def BASE():
    return Abs(
        'BASE',
        'V_b3, V_b4, V_X, V_Y, V_a, V_b, V_c',
        """
        If(V_b3,
          If(V_b4,
            BASEV1(V_X, V_Y, V_a),
            BASEV2(V_X, V_Y, V_a, V_b)
          ),
          If(V_b4,
            BASEV3(V_X, V_Y, V_a, V_b),
            BASEV4(V_X, V_Y, V_a, V_b, V_c)
          )
        )            
        """
    )

# First way that an entire shape can be composed: combine back, seat and base
def ROOTV1():
    return Abs(
        'ROOTV1',
        'V_b3, V_b4, V_b5, V_b6, V_X, V_a, V_b, V_c, V_d, V_o, V_u, V_v, V_k',
        """
        Union(
          Move(
            BACK(V_b5, V_b6, V_X, Mul(V_X, 0.75), V_d, V_o, V_u, V_v, V_k), 
            0., HF1(Mul(V_X, 0.75))
          ), 
          Union(
            Move(Rectangle(V_X, Mul(V_X, .15)), 0., HF1(Mul(V_X, -.15))),
            Move(BASE(V_b3, V_b4, V_X, Mul(V_X, .6), V_a, V_b, V_c), 0., HF1(Mul(V_X, -.9)))
          )
        )
        """
    )

# Second way that an entire shape can be composed, combine simple back, head, seat, with base
def ROOTV2():
    return Abs(
        'ROOTV2',
        'V_b2, V_b3, V_b4, V_X, V_a, V_b, V_c, V_h',
        """
        Union(
          If(V_b2,
          Union(
            Move(Rectangle(V_X, Mul(Sub(0.4, V_h), HF1(V_X))), 0., HF1(Mul(Sub(0.4, V_h), HF1(V_X)))),
            Move(Rectangle(Mul(V_h, HF1(V_X)), Mul(V_h, HF1(V_X))), 0., Add(Mul(Sub(0.4, V_h), HF1(V_X)), HF1(Mul(V_h, HF1(V_X)))))
          ),
          Move(Rectangle(V_X, Mul(0.4, HF1(V_X))), 0., HF1(Mul(0.4, HF1(V_X))))
         ),
         Move(BASE(V_b3, V_b4, V_X, Mul(0.6, HF1(V_X)), V_a, V_b, V_c), 0., HF1(Mul(-.6, HF1(V_X))))
        )
        """
    )

# Top level function of grammar. Returns one of the two root sub-functions, depending on V_b1
def ROOT():
    return Abs(
        'ROOT',
        'V_b1, V_b2, V_b3, V_b4, V_b5, V_b6, V_X, V_a, V_b, V_c, V_h, V_d, V_o, V_u, V_v, V_k',
        """
        If(
          V_b1,
          ROOTV1(V_b3, V_b4, V_b5, V_b6, V_X, V_a, V_b, V_c, V_d, V_o, V_u, V_v, V_k),
          ROOTV2(V_b2, V_b3, V_b4, V_X, V_a, V_b, V_c, V_h)
        )    
        """
    )

# Random parameter samplers
def randb(r):
    return random.random() > (1-r)

def randf(a, b):
    return round((random.random() * (b-a)) + a, 2)

def randint(a, b):
    return random.randint(a, b)

# Sample a shape program from the grammar    
def sample_shape_program(ret_parse=False, round_prec=None):

    # Load abstractions into executor
    
    abstractions = [HF1(), BACK(), BASE(), ROOTV1(), ROOTV2(), ROOT(), BASEV1(), HF2(), BASEV2(), BASEV3(), BASEV4(), BACKV1(), BACKV2(), BACKV3()]
    P = ex.Program(abstractions)

    # Sample parameters
    
    b1 = randb(0.5)
    b2 = randb(0.5)
    b3 = randb(0.5)
    b4 = randb(0.5)
    b5 = randb(0.5)
    b6 = randb(0.5)
    
    X = randf(0.6, 1.2)
    a = randf(0.05, .25)
    b = randf(0.05, .15)
    c = randf(0.2, 0.8)
    h = randf(.05, .2)
    d = randf(0.05, 0.15)
    o = randf(0., 0.2)
    u = randf(0.05, 0.15)
    v = randf(0.05, 0.1)
    k = randint(2, 4)

    # Round parameters, if precision is given
    if round_prec is not None:
        X = round(X * round_prec) / round_prec
        a = round(a * round_prec) / round_prec
        b = round(b * round_prec) / round_prec
        c = round(c * round_prec) / round_prec
        h = round(h * round_prec) / round_prec
        d = round(d * round_prec) / round_prec
        o = round(o * round_prec) / round_prec
        u = round(u * round_prec) / round_prec
        v = round(v * round_prec) / round_prec

    # program is just a call to root node
    prog = f"ROOT({b1},{b2},{b3},{b4},{b5},{b6},{X},{a},{b},{c},{h},{d},{o},{u},{v},INT#{k}#)"

    if ret_parse:        
        parse = P.parse(prog)
        return parse
    else:
        P.run(prog)

    # return program
        
    return P

# Converts the program into primitives
def sample_shape_prims(round_prec=None):
    P = sample_shape_program(round_prec=round_prec)
    prims = P.getPrimitives()
    def _round(i):
        return round(i * round_prec) / round_prec
    
    prims = [[_round(i) for i in _prim] for _prim in prims]
    sub_progs = [f'Move(Rectangle({a},{b}),{c},{d})' for a,b,c,d in prims]

    def comb(L):
        if len(L) == 1:
            return L[0]
        elif len(L) > 1:
            return f'Union({L[0]},{comb(L[1:])})'

    comb_prog = comb(sub_progs)
    
    return comb_prog


if __name__ == '__main__':
    random.seed(42)
    with open(sys.argv[1], 'w') as f:
        for i in range(int(sys.argv[2])):
            prog = sample_shape_prims(2)
            f.write(f'{i}:{prog}\n')
        
