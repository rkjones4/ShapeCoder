from copy import deepcopy
import utils
import random
import rewrites as rws
import math
import prog_utils as pu
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from scipy.optimize import linear_sum_assignment
import time

def safe_get_matrix(M):
    try:
        return M.as_matrix()
    except:
        return M.as_dcm()

# Helper function to check for a match between predicted and target primitives
# D is a distance matrix, B is the list of target prims, max error thresh represents
# the distance at which a match is accepted
def check_dist_for_match(D, B, max_error_thresh):

    if D.shape[0] > D.shape[1]:
        print(f"Unexpected, saw D with more predicted prims than target prims {D.shape}")
        return None, max_error_thresh, max_error_thresh * 2

    # give non matches high score
    MD = D + ((D > max_error_thresh).float() * 1000.)

    row_ind, col_ind = linear_sum_assignment(MD)

    costs = D[row_ind, col_ind]
                
    minds = [B[ci][0] for ci in col_ind.tolist()]
    
    max_err = costs.max().item()
    avg_err = costs.mean().item()
        
    match = max_err <= max_error_thresh

    # all assignments must be valid for this condition to pass
    if match:
        return minds, avg_err, max_err
    
    return None, avg_err, max_err
    

# Mathematical operations, used in Library class
# lmbda, pt_lmbda define executor behavior
# invalid are list of invalid constants
# assoc -> is associative, oneparam -> if it takes in a single param

class AddFn():
    def __init__(self):
        self.name = 'Add'
        self.lmbda = lambda x,y: x+y
        self.pt_lmbda = torch.add
        self.invld = set([0., 2.])
        self.assoc = True
        self.oneparam = False
        
    def make_libtoken(self):
        return LibToken(self.name, ['Float', 'Float'], 'Float', 'float_fn')
        
class SubFn():
    def __init__(self):
        self.name = 'Sub'
        self.lmbda = lambda x,y: x-y
        self.pt_lmbda = torch.sub
        self.invld = set([0., 2.])
        self.assoc = False
        self.oneparam = False

    def make_libtoken(self):
        return LibToken(self.name, ['Float', 'Float'], 'Float', 'float_fn')
        
class MulFn():
    def __init__(self):
        self.name = 'Mul'
        self.lmbda = lambda x,y: x*y
        self.pt_lmbda = torch.mul
        self.invld = set([0., 1.])
        self.assoc = True
        self.oneparam = False

    def make_libtoken(self):
        return LibToken(self.name, ['Float', 'Float'], 'Float', 'float_fn')

# negative of the input (inverse)
class InvFn():
    def __init__(self):
        self.name = 'Inv'
        self.lmbda = lambda x: x*-1
        self.pt_lmbda = torch.neg
        self.invld = set()
        self.assoc = None
        self.oneparam = True

    def make_libtoken(self):
        return LibToken(self.name, ['Float'], 'Float', 'float_fn')

    
class DivFn():
    def __init__(self):
        self.name = 'Div'
        self.lmbda = lambda x,y: x/y if y != 0. else 0.
        self.pt_lmbda = torch.div
        self.invld = set([0., 1.])
        self.assoc = False
        self.oneparam = False

    def make_libtoken(self):
        return LibToken(self.name, ['Float', 'Float'], 'Float', 'float_fn')

# End math operators definitions

# Language for 2D shapes

class Lang2D:

    def __init__(self):
        # Valid constants
        self.float_consts = [0., 1., 2.]
        self.int_consts = [1,2,3,4]

        # floats that can be used independently of an expression
        self.singleton_float_consts = set(['0.0'])
        self.expr_float_consts = [1., 2.]
        
        self.token_map = {}
        
        # Specify type of primitives, and the combinator operator
        self.prim_name = 'Shape'
        self.comb_op_name = 'Union'

        # Defining language
        self.base_tokens = [
            LibToken(
                'Rectangle', ['Float', 'Float'], 'Shape', 'fn',
                ['Rectangle_0_', 'Rectangle_1_'],
                self.lookup_fn,
                self.token_map
            ),
            LibToken(
                'Move', ['Shape', 'Float', 'Float'], 'Shape', 'fn',
                [None, 'Move_1_', 'Move_2_'],
                self.lookup_fn,
                self.token_map
            ),
            LibToken('Union', ['Shape', 'Shape'], 'Shape', 'fn'),
            LibToken(
                'SymTrans', ['Shape', 'Axis', 'Int', 'Float'], 'Shape', 'fn',
                [None, None, None, 'SymTrans_3_'],
                self.lookup_fn,
                self.token_map
            ),
            LibToken('SymRef', ['Shape', 'Axis'], 'Shape', 'fn'),                     
            LibToken('AX', [], 'Axis', 'cat'),
            LibToken('AY', [], 'Axis', 'cat'),
        ]

        for C in self.int_consts:
            self.base_tokens.append(
                LibToken(f'INT#{C}#', [], 'Int', 'cat'),
            )

        for C in self.float_consts:
            self.base_tokens.append(
                LibToken(str(C), [], 'Float', 'float_const'),
            )

            
        self.cat_tokens = [lt.name for lt in self.base_tokens if lt.cost_type == 'cat']

        # These functions will not be top-level sub-functions for dream/wake phases
        self.prim_skip_fns = ['Rectangle', 'Move', 'Union']

        # These functions can always be sub-functions in dreams
        self.dream_keep_fns = ['Union', 'Rectangle', 'Move']

        # These functions can be used only once in each sub-dream
        self.dream_single_use_fns = ['Move', 'Union']

        # These functions must have new dreams created each round
        self.dream_fns_to_rm = ['SymRef', 'SymTrans']

        # Floats functions
        self.float_fns = [AddFn(), SubFn(), MulFn(), DivFn(), InvFn()]
        self.float_fn_map = {fn.name: fn for fn in self.float_fns}

        for fn in self.float_fns:
            self.base_tokens.append(fn.make_libtoken())

        self.base_token_map = {t.name:t for t in self.base_tokens}

        # Add rewrites to the library
        
        self.base_rewrites = []
        rws.add_rewrites(rws.SEM_2DL_REWRITES, self.base_rewrites)


    # Preference scoring for an input abstraction (represented with sig)
    # See comments in Lang3D for explanation
    def get_abs_pref_score(self, sig, freq, args, L):

        if freq < args.ab_min_use_perc:
            return None
        
        val = 0.0
                
        ab_info = pu.parse_ab_info(sig, L)
        
        if ab_info['num_prm_expr'] > 0:
            if ab_info['num_prm_expr'] >= 2:
                val += 0.25
            else:
                val += 0.1

        elif ab_info['num_const'] > 0 or ab_info['num_dupl'] > 0:
            sm = ab_info['num_const'] + ab_info['num_dupl']

            if sm >= 3:
                val += 0.1
            elif sm >= 2:
                val += 0.0
            else:
                val -= 0.2
            
        else:
            val -= 0.5

        nu = ab_info['num_unions']
        nf = ab_info['num_fns']

        if nu >= 1:
            val += 0.25
            
        else:
            if nf == 1:
                val -= 0.15
            elif nf == 2:
                val -= 0.1
            elif nf == 3:
                val -= 0.05
        
        # Penalize too many free parameters
        nfp = ab_info['num_params']

        if nfp >= 10:
            return None
        elif nfp >= 9:
            val -= 0.75
        elif nfp >= 8:
            val -= 0.5
        elif nfp >= 7:
            val -= 0.25
        elif nfp >= 6:
            val -= 0.1
            
        return min(max(val, -1.0), 0.5)
    
    def get_valid_singleton_consts(self):
        return {
            f'{f}': torch.tensor(float(f)) for f in self.singleton_float_consts
        }

    # Describe how parameters for functions should be sampled while dreaming
    # Abstractions inherit parameter distributions from base functions in the library
    def lookup_fn(self, t, tmap):
        import prm_sample_dist as psd        
        
        if t is None:
            return [], None
        
        fn, ind = t.split('_')[:2]
        vpti = []

        ind = int(ind)
        
        sf = None
        if 'Rectangle' in fn:
            vpti += [0,1]

            if ind == 0:
                sf = psd.samp_s3d_w
            elif ind == 1:
                sf = psd.samp_s3d_h
            else:
                assert False
                
        elif 'Move' in fn:
            vpti += [2,3]

            if ind == 1:
                sf = psd.samp_s3d_x
            elif ind == 2:
                sf = psd.samp_s3d_y
            else:
                assert False
                
        elif 'SymTrans' in fn:
            vpti += [0,1,2,3]

            if ind == 3:
                sf = psd.samp_s3d_TransDist
            else:
                assert False

        elif 'Abs' in fn:
            
            fn, abs_ind, param_ind,_ = t.split('_')

            assert fn == 'Abs'

            fn = f'Abs_{abs_ind}'
            
            vpti += deepcopy(tmap[fn].valid_param_type_inds)
            sf = tmap[fn].samp_fns[int(param_ind)]
        
        return vpti, sf

    # Starting information for wake
    def get_start_stack(self):
        return [('token', 'Shape')]

    # Return a naive sub-expression to explain a single primitive
    def get_base_prim_tokens(self, ind, data):
        return [
            self.base_token_map['Move'],
            self.base_token_map['Rectangle'],
            LibToken(f'S_{ind}_0_', [], 'Float', 'float_var'),
            LibToken(f'S_{ind}_1_', [], 'Float', 'float_var'),
            LibToken(f'S_{ind}_2_', [], 'Float', 'float_var'),
            LibToken(f'S_{ind}_3_', [], 'Float', 'float_var'),
        ]

    # Distance between two primitives
    def prim_dist(self, U, V):
        dist =[abs(u-v) for u,v in zip(U,V)]
        md = max(dist)
        td = sum(dist)        
        return md, td
    
    # see if the set A is a subset of the prims in B
    def check_match(self, A, B, args):

        if len(A) > len(B):
            max_error_thresh = args.of_max_error
            return None, max_error_thresh, max_error_thresh * 2

        D = self.get_dist(A, B, args)

        return check_dist_for_match(D, B, args.of_max_error)

    # Create distance matrix from primitives in A to primitives in B
    def get_dist(self, A, B, args):

        D = torch.zeros(len(A), len(B))
        
        for i,a in enumerate(A):
            oprims = []
            min_err = 0
            
            for j,b in enumerate(B):

                md, td = self.prim_dist(a, b[1])
                
                D[i,j] = td
                    
        return D


    # get the alignment from a prim to b prim, using data d and args
    def get_align(self, a, b, d, args):

        if abs(a[0] - b[0]) >= args.of_max_error or\
           abs(a[1] - b[1]) >= args.of_max_error:
            return None, None

        Val = (
            round(b[2] - a[2], args.prec),
            round(b[3] - a[3], args.prec),
        )
        
        return Val, Val

    # change prims in A with alignment t
    def do_align(self, A, t):
        B = []

        for a in A:
            B.append((a[0], a[1], a[2] + t[0], a[3] + t[1]))
        
        return B

    # take in created prim set, gt prim set, data, and args
    # return true if the created prim set can be aligned to match gt
    def check_align_match(self, bprog, bprim, P, d, args):
        
        for p in P:
            tV, tExpr = self.get_align(bprim[0], p[1], d, args)
        
            if tV is None:
                continue
            
            match, err, max_err = self.check_match(
                self.do_align(bprim, tV),
                P,
                args
            )

            if match is not None and len(match) > 0:            
                bprog.text = f'Move({bprog.text},{tExpr[0]},{tExpr[1]})'
                return match, err, max_err
            
        return None, None, None

    # How to combine sub expressions into a full program
    def comb_progs(self, sub_progs):    
        if len(sub_progs) == 1:
            return sub_progs[0]
        else:
            return f'Union({sub_progs[0]},{self.comb_progs(sub_progs[1:])})'


    # optionally add transformation to dreamed data, for more
    # coverage in training data
    def dream_sample_unalign(self, text, prims, args):
        x = round(random.random() * args.dm_move_range, args.prec)
        y = round(random.random() * args.dm_move_range, args.prec)

        moved_text =  f'Move({text}, {x}, {y})'

        moved_prims = [
            (
                p[0],
                p[1],
                round(p[2] + x, args.prec),
                round(p[3] + y, args.prec)
            ) for p in prims
        ]

        return moved_text, moved_prims, (x,y)

    # Calculates the overlap between two primitives
    def calc_overlap(self, p1, p2):
        r1_r, r1_l, r1_b, r1_t, r1_A = p1
        r2_r, r2_l, r2_b, r2_t, r2_A = p2
    
        x_overlap = max(0, min(r1_r, r2_r) - max(r1_l, r2_l))
        y_overlap = max(0, min(r1_b, r2_b) - max(r1_t, r2_t))

        overlapArea = x_overlap * y_overlap;
        
        return overlapArea / min(r1_A, r2_A)

    # Sample a set of distractor primitives to make training data more robust
    def sample_distractors(self, D, args):
        P = [p for _, p in D.sample().prims]
        
        num = min(random.randint(1, args.dm_max_distract_prims), len(P))
        SD = random.sample(P, num)

        return SD

    # Checks if prims are valid
    def check_valid_prims(self, prims, args):

        # r, l, b, t, A
        exp = []

        for w, h, x, y in prims:
            # dims must be positive
            if w <= 0. or h <= 0.:
                return False
            
            _corners = [
                (x + w/2, y + h/2),
                (x + w/2, y - h/2),
                (x - w/2, y + h/2),
                (x - w/2, y - h/2),
            ]
            for a, b in _corners:
                # checks corners within bounds
                if max(abs(a), abs(b)) > args.dm_max_bounds:
                    return False

            A = w * h

            # checks area is valid
            if A < args.dm_min_area:
                return False
            
            exp.append((x + w/2, x - w/2, y + h/2, y - h/2, A))

        for i, p1 in enumerate(exp):
            for j, p2 in enumerate(exp):
                if i == j:
                    continue

                # Checks overlap between any pair is not too great
                
                oratio = self.calc_overlap(p1, p2)
                if oratio > args.dm_max_overlap:
                    return False

        return True


    # See if dream is valid based on text
    def check_valid_dream_text(self, text, args):
        for fn in self.dream_single_use_fns:
            if f'{fn}({fn}(' in text:
                return False
        return True

    # Top level function to see if primitives created by dream is valid
    def check_valid_dream_prims(self, prims, args):

        if len(prims) > args.dm_max_primitives:
            return False
            
        if not self.check_valid_prims(prims, args):
            return False

        return True

    # Dreaming related rejection/valid parameters, can be overwritten by command line args
    def make_dream_params(self, args):
        params = {
            'max_overlap': 0.5,
            'min_area': 0.00025,
            'max_bounds': 1.1,
            'max_primitives': args.rn_max_prims,
            'move_range': 0.5,
        }

        if args.dm_dream_params is not None:
            for expr in args.dm_dream_params.split(','):
                key, value = expr.split(':')
                params[key] = float(value)

        args.dm_max_overlap = params['max_overlap']
        args.dm_min_area = params['min_area']
        args.dm_max_bounds = params['max_bounds']
        args.dm_max_primitives = params['max_primitives']
        args.dm_move_range = params['move_range']

        utils.log_print(f"Dream Params : {params}", args)


    # Cost of the naive program that would have to create prims
    def get_naive_costs(self, prims, args):

        fn = -1
        flt = 0

        for _,_,x,y in prims:
            fn += 2
            flt += 2

            # Check if we would need to move that prim
            if x != 0. or y != 0.:
                fn += 1
                flt += 2
                        
        naive_struct_cost = (fn * args.of_weights['fn']) 
        naive_param_cost = (flt * args.of_weights['float_var']) 
                        
        return naive_struct_cost, naive_param_cost

# Language for 3D Shapes
class Lang3D:
    
    def __init__(self):
        # Valid constants
        self.float_consts = [0., 1., 2.]
        self.int_consts = [1,2,3,4,5,6]

        # floats that can be used independently of an expression
        self.singleton_float_consts = set(['0.0'])
        # floats that can be used in expression
        self.expr_float_consts = [1., 2.]    

        # Specify type of primitives, and combinator operator
        self.prim_name = 'Shape'
        self.comb_op_name = 'Union'

        self.token_map = {}

        # Defining language
        self.base_tokens = [
            LibToken(
                'Cuboid', ['Float', 'Float', 'Float'], 'Shape', 'fn',
                ['Cuboid_0_', 'Cuboid_1_', 'Cuboid_2_'],
                self.lookup_fn,
                self.token_map
            ),            

            LibToken(
                'Move', ['Shape', 'Float', 'Float', 'Float'], 'Shape', 'fn',
                [None, 'Move_1_', 'Move_2_', 'Move_3_'],
                self.lookup_fn,
                self.token_map
            ),

            LibToken(
                'Rotate', ['Shape', 'Axis', 'Float'], 'Shape', 'fn', 
                [None, None, 'Rotate_2_'],
                self.lookup_fn,
                self.token_map
            ),

            LibToken('Union', ['Shape', 'Shape'], 'Shape', 'fn'),
            
            LibToken('SymRef', ['Shape', 'Axis'], 'Shape', 'fn'),                     

            LibToken(
                'SymTrans', ['Shape', 'Axis', 'Int', 'Float'], 'Shape', 'fn',
                [None, None, None, 'SymTrans_3_'],
                self.lookup_fn,
                self.token_map
            ),
                        
            LibToken('AX', [], 'Axis', 'cat'),
            LibToken('AY', [], 'Axis', 'cat'),
            LibToken('AZ', [], 'Axis', 'cat'),

        ]
        
        for C in self.int_consts:
            self.base_tokens.append(
                LibToken(f'INT#{C}#', [], 'Int', 'cat'),
            )

        for C in self.float_consts:
            self.base_tokens.append(
                LibToken(str(C), [], 'Float', 'float_const'),
            )

        self.cat_tokens = [lt.name for lt in self.base_tokens if lt.cost_type == 'cat']

        # These functions will not be top-level sub-functions for dream/wake phases
        self.prim_skip_fns = ['Cuboid', 'Move', 'Rotate', 'Union']

        # These functions can always be sub-functions in dreams
        self.dream_keep_fns = ['Cuboid', 'Move', 'Rotate', 'Union']

        # These functions can be used only once in each sub-dream
        self.dream_single_use_fns = ['Move', 'Union']

        # These functions must have new dreams created each round
        self.dream_fns_to_rm = ['SymRef', 'SymTrans']
        
        # Floats functions
        self.float_fns = [AddFn(), SubFn(), MulFn(), DivFn(), InvFn()]
        self.float_fn_map = {fn.name: fn for fn in self.float_fns}

        for fn in self.float_fns:
            self.base_tokens.append(fn.make_libtoken())

        self.base_token_map = {t.name:t for t in self.base_tokens}

        # Add rewrites to the library
        
        self.base_rewrites = []
        rws.add_rewrites(rws.SEM_3DL_REWRITES, self.base_rewrites)        


    # Preference scoring for an input abstraction (represented with sig)
    # positive scores (max 0.5) will make it easier to add abstraction into lib
    # negative scores (min of -1) will make it harder to add abstraction into lib
    def get_abs_pref_score(self, sig, freq, args, L):

        # if frequency is below min use percentage, bad abstraction
        if freq < args.ab_min_use_perc:
            return None

        # starting val
        val = 0.0

        # parse information about abstraction
        ab_info = pu.parse_ab_info(sig, L)

        # Get the number of parameter expression
        if ab_info['num_prm_expr'] > 0:
            if ab_info['num_prm_expr'] >= 2:
                val += 0.25
            else:
                val += 0.1

        # get the number of constants used, or re-used variable
        elif ab_info['num_const'] > 0 or ab_info['num_dupl'] > 0:
            sm = ab_info['num_const'] + ab_info['num_dupl']

            if sm >= 3:
                val += 0.1
            elif sm >= 2:
                val += 0.0
            else:
                val -= 0.2
            
        else:
            val -= 0.5

        # check number combinators, and total functions
        nu = ab_info['num_unions']
        nf = ab_info['num_fns']

        # if atleast one combinator
        if nu >= 1:
            val += 0.25
            
        else:
            # penalize if abstraction is too simple
            if nf == 1:
                val -= 0.15
            elif nf == 2:
                val -= 0.1
            elif nf == 3:
                val -= 0.05
        

        # Penalize too many free parameters
        nfp = ab_info['num_params']

        if nfp >= 10:
            return None
        elif nfp >= 9:
            val -= 0.75
        elif nfp >= 8:
            val -= 0.5
        elif nfp >= 7:
            val -= 0.25
        elif nfp >= 6:
            val -= 0.1
            
        return min(max(val, -1.0), 0.5)
    
    def get_valid_singleton_consts(self):
        return {
            f'{f}': torch.tensor(float(f)) for f in self.singleton_float_consts
        }

    # Describe how parameters for functions should be sampled while dreaming
    # Abstractions inherit parameter distributions from base functions in the library
    def lookup_fn(self, t, tmap):
        import prm_sample_dist as psd        

        if t is None:
            return [], None
        
        fn, ind = t.split('_')[:2]
        vpti = []

        ind = int(ind)
        
        sf = None
        if 'Cuboid' in fn:
            vpti += [0,1,2]

            if ind == 0:
                sf = psd.samp_s3d_w
            elif ind == 1:
                sf = psd.samp_s3d_h
            elif ind == 2:
                sf = psd.samp_s3d_d
            else:
                assert False
                
        elif 'Move' in fn:
            vpti += [3,4,5]

            if ind == 1:
                sf = psd.samp_s3d_x
            elif ind == 2:
                sf = psd.samp_s3d_y
            elif ind == 3:
                sf = psd.samp_s3d_z
            else:
                assert False
                
        elif 'Rotate' in fn:
            vpti += [6,7,8]

            if ind == 2:
                sf = psd.samp_s3d_angle
            else:
                assert False
                
        elif 'SymTrans' in fn:
            vpti += [0,1,2,3,4,5]

            if ind == 3:
                sf = psd.samp_s3d_TransDist
            else:
                assert False

        elif 'Abs' in fn:
            
            fn, abs_ind, param_ind,_ = t.split('_')

            assert fn == 'Abs'

            fn = f'Abs_{abs_ind}'
            
            vpti += deepcopy(tmap[fn].valid_param_type_inds)
            sf = tmap[fn].samp_fns[int(param_ind)]
        
        return vpti, sf

        
    # Starting information for wake
    def get_start_stack(self):
        return [('token', 'Shape')]

    # Return a naive sub-expression to explain a single primitive
    def get_base_prim_tokens(self, ind, data):

        commands = [
            self.base_token_map['Move'],
            self.base_token_map['Cuboid'],
        ]

        params = [
            LibToken(f'S_{ind}_0_', [], 'Float', 'float_var'),
            LibToken(f'S_{ind}_1_', [], 'Float', 'float_var'),
            LibToken(f'S_{ind}_2_', [], 'Float', 'float_var'),
            LibToken(f'S_{ind}_3_', [], 'Float', 'float_var'),
            LibToken(f'S_{ind}_4_', [], 'Float', 'float_var'),
            LibToken(f'S_{ind}_5_', [], 'Float', 'float_var'),
        ]

        assert len(data) == 9

        # only add rotations if they are needed
        for v, AX, PIND in zip(data[6:], ['AX', 'AY', 'AZ'], [6,7,8]):

            # non-axis aligned rotation
            if abs(v) >= 0.01:
                commands = [self.base_token_map['Rotate']] + commands
                params += [self.base_token_map[AX], LibToken(f'S_{ind}_{PIND}_', [], 'Float', 'float_var')]
                
            
        return commands + params
                                                

    # get the alignment from a prim to b prim, using data d and args
    def get_align(self, a, b, d, args):

        if abs(a[0] - b[0]) >= args.of_max_error or\
           abs(a[1] - b[1]) >= args.of_max_error or\
           abs(a[2] - b[2]) >= args.of_max_error:
            return None, None

        Val = (
            round(b[3] - a[3], args.prec),
            round(b[4] - a[4], args.prec),
            round(b[5] - a[5], args.prec)
        )

        Expr = []
            
        return Val, Val
        
    # change prims in A with alignment t
    def do_align(self, A, t):
        B = []

        for a in A:
            B.append((a[0], a[1], a[2], a[3] + t[0], a[4] + t[1], a[5] + t[2], a[6], a[7], a[8]))
        
        return B                                

    # geometric distance between prim set A and B
    # comptued by Hausdorff between corner sets of each primitive pair
    def get_geom_dist(self, A, B, args):
                
        def cornerize(params):
            with torch.no_grad():
                return np.stack([_cornerize(p) for p in params])

        def _cornerize(prms):
            center = np.array([prms[3], prms[4], prms[5]])
            lengths = np.array([prms[0], prms[1], prms[2]])

            T = R.from_euler('xyz', [prms[6],prms[7],prms[8]], degrees=False)
            mat = safe_get_matrix(T)

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
                
        b_corners = torch.from_numpy(cornerize(A))
        t_corners = torch.from_numpy(cornerize([t for _,t in B]))
        
        V = b_corners.view(b_corners.shape[0], 1, 8, 1, 3) -\
            t_corners.view(1, t_corners.shape[0], 1, 8, 3)
        
        V = V.norm(dim=4)
        
        VF = V.min(dim=3).values.max(dim=-1).values
        VB = V.min(dim=2).values.max(dim=-1).values

        VS = torch.stack((VF, VB),dim=-1)

        VC = VS.max(dim=-1).values
        
        return VC

    # see if the set A is a subset of the prims in B
    def check_match(self, A, B, args):

        if len(A) > len(B):
            max_error_thresh = args.of_max_error
            return None, max_error_thresh, max_error_thresh * 2
        
        D = self.get_geom_dist(A, B, args)
                        
        return check_dist_for_match(D, B, args.of_max_error)

    # take in created prim set, gt prim set, data, and args
    # return true if the created prim set can be aligned to match gt
    def check_align_match(self, bprog, b_prims, t_prims, d, args):
        
        for tP in t_prims:
            tV, tExpr = self.get_align(b_prims[0], tP[1], d, args)

            if tV is None:
                continue

            match, err, max_err = self.check_match(
                self.do_align(b_prims, tV),
                t_prims,
                args
            )

            if match is not None and len(match) > 0:            
                bprog.text = f'Move({bprog.text},{tExpr[0]},{tExpr[1]},{tExpr[2]})'
                return match, err, max_err
            
        return None, None, None
    
    # optionally add transformation to dreamed data, for more
    # coverage in training data
    def dream_sample_unalign(self, text, prims, args):
        x = round(random.random() * args.dm_move_range, args.prec)
        y = round(random.random() * args.dm_move_range, args.prec)
        z = round(random.random() * args.dm_move_range, args.prec)

        moved_text =  f'Move({text}, {x}, {y}, {z})'

        moved_prims = [
            (
                p[0],
                p[1],
                p[2],
                round(p[3] + x, args.prec),
                round(p[4] + y, args.prec),
                round(p[5] + z, args.prec),
                p[6],
                p[7],
                p[8],
            ) for p in prims
        ]

        return moved_text, moved_prims, (x,y,z)

    # How to combine sub expressions into a full program
    def comb_progs(self, sub_progs):
        if len(sub_progs) == 1:
            return sub_progs[0]

        else:
            return f'Union({sub_progs[0]},{self.comb_progs(sub_progs[1:])})'

    # Sample a set of distractor primitives to make training data more robust
    def sample_distractors(self, D, args):
        
        P = [p for _, p in D.sample().prims]
        
        num = min(random.randint(1, args.dm_max_distract_prims), len(P))
        SD = random.sample(P, num)
        
        return SD


    # Calculates the overlap between two primitives,
    # using simplifying assumption that they are axis aligned (for speed)
    def calc_overlap(self, p1, p2):        
        r1_r, r1_l, r1_t, r1_b, r1_fr, r1_bk, r1_A = p1
        r2_r, r2_l, r2_t, r2_b, r2_fr, r2_bk, r2_A = p2
    
        x_overlap = max(0, min(r1_r, r2_r) - max(r1_l, r2_l))
        y_overlap = max(0, min(r1_t, r2_t) - max(r1_b, r2_b))
        z_overlap = max(0, min(r1_fr, r2_fr) - max(r1_bk, r2_bk))
        
        overlapArea = x_overlap * y_overlap * z_overlap;
        
        return overlapArea / min(r1_A, r2_A)


    # Checks if prims are valid
    def check_valid_prims(self, prims, args):
        
        exp = []

        for w, h, d, x, y, z, rf, tf, ff in prims:
            # dims must be positive
            if w <= 0. or h <= 0. or d <= 0.:
                return False
            
            _corners = [
                (x + w/2, y + h/2, z + d/2),
                (x + w/2, y + h/2, z - d/2),
                (x + w/2, y - h/2, z + d/2),
                (x + w/2, y - h/2, z - d/2),
                (x - w/2, y + h/2, z + d/2),
                (x - w/2, y + h/2, z - d/2),
                (x - w/2, y - h/2, z + d/2),
                (x - w/2, y - h/2, z - d/2)                
            ]
            for a, b, c in _corners:
                # checks corners within bounds
                if max(abs(a), abs(b), abs(c)) > args.dm_max_bounds:
                    return False

            A = w * h * d

            # checks area is valid
            if A < args.dm_min_area:
                return False
            
            exp.append((x + w/2, x - w/2, y + h/2, y - h/2, z + d/2, z - d/2, A))

        for i, p1 in enumerate(exp):
            for j, p2 in enumerate(exp):
                if i == j:
                    continue

                # Checks overlap between any pair is not too great
                oratio = self.calc_overlap(p1, p2)

                if oratio > args.dm_max_overlap:
                    return False

        return True

    
    # See if dream is valid based on text
    def check_valid_dream_text(self, text, args):        
        for fn in self.dream_single_use_fns:
            if f'{fn}({fn}(' in text:
                return False

        return True
    
    # Top level function to see if primitives created by dream is valid
    def check_valid_dream_prims(self, prims, args):        
        if len(prims) > args.dm_max_primitives:
            return False
            
        if not self.check_valid_prims(prims, args):
            return False
        
        return True

    # Dreaming related rejection/valid parameters, can be overwritten by command line args
    def make_dream_params(self, args):

        params = {
            'max_overlap': 0.5,
            'min_area': 0.005,
            'max_bounds': 1.1,
            'max_primitives': args.rn_max_prims,
            'move_range': 0.5
        }

        if args.dm_dream_params is not None:
            for expr in args.dm_dream_params.split(','):
                key, value = expr.split(':')
                params[key] = float(value)

        args.dm_max_overlap = params['max_overlap']
        args.dm_min_area = params['min_area']
        args.dm_max_bounds = params['max_bounds']
        args.dm_max_primitives = params['max_primitives']
        args.dm_move_range = params['move_range']
        
        utils.log_print(f"Dream Params : {params}", args)

    # Cost of the naive program that would have to create prims
    def get_naive_costs(self, prims, args):        

        fn = -1
        flt = 0
        cat = 0

        for _,_,_,x,y,z,a,b,c in prims:
            fn += 2
            flt += 3

            # requires move
            if x != 0. or y != 0. or z != 0.:
                fn += 1
                flt += 3

            # requires x axis rot
            if a != 0.: 
                fn += 1
                flt += 1
                cat += 1

            # requires y axis rot
            if b != 0.: 
                fn += 1
                flt += 1
                cat += 1

            # requires z axis rot
            if c != 0.: 
                fn += 1
                flt += 1
                cat += 1
        
                
        naive_struct_cost = (fn * args.of_weights['fn']) 
        naive_param_cost = (flt * args.of_weights['float_var']) + (cat * args.of_weights['cat'])
        
        return naive_struct_cost, naive_param_cost
        

# Helper class representing a token in the library
class LibToken:
    def __init__(
        self, name, inp_types, out_type, cost_type,
        param_info = None,
        lookup_fn = None,
        token_map = None    
    ):
            
        self.name = name
        # input types
        self.inp_types = inp_types
        # output types
        self.out_type = out_type
        # cost type of this token
        self.cost_type = cost_type

        if param_info is None:
            self.valid_param_type_inds = None
            self.samp_fns = None
            return

        assert lookup_fn is not None
        assert token_map is not None

        # How to sample parameters during dreaming
        self.valid_param_type_inds = set()
        self.samp_fns = []
        for pt in param_info:
            vpti, sf = lookup_fn(pt, token_map)
            for _vpti in vpti:
                self.valid_param_type_inds.add(_vpti)
                
            self.samp_fns.append(sf)

        self.valid_param_type_inds = list(self.valid_param_type_inds)

# Class represinting the library, starts with base functions, then abstractions are added omtp ot
class Library:
    def __init__(self, ex, lang):
        # base language (has most of logic)
        self.lang = lang

        # Map of all current abstractions
        self.abs_map = {}
        # Map of all abstractions EVER ADDED into library
        # important, as some may be removed later, but still used as sub-functions
        self.abs_register = deepcopy(self.abs_map)

        # executor program and abstraction information
        self.ex_prog_class = ex.Program
        self.ex_abst_class = ex.Abstraction

        self.num_abs_made = 0        
        self.reset()      
    
    def reset(self):
        self.init_lib()
        self.init_maps()
        self.init_executor()

    # Iniatilize executor
    def init_executor(self):
        if 'abs_register' not in self.__dict__:
            self.abs_register = deepcopy(self.abs_map)

        self.executor = self.ex_prog_class([a.ex_abs for a in self.abs_register.values()])

    # Initialize library
    def init_lib(self):

        self.tokens = []
        self.tokens += deepcopy([a.token for a in self.abs_map.values()])
        self.tokens += deepcopy([b for b in self.lang.base_tokens])        

    # Initialize type and token maps
    def init_maps(self):
        self.type_map = {}
        self.token_map = {}
        
        for t in self.tokens:
            self.token_map[t.name] = t
            
            if t.out_type not in self.type_map:
                self.type_map[t.out_type] = []
            self.type_map[t.out_type].append(t)

    # Remove abstractions from the library (don't change register)
    def remove_abs(self, absts):
        rmvd = []
        for abst in absts:
            ab = self.abs_map.pop(abst)
            rmvd.append(ab)
        self.reset()
        return rmvd

    # Add an abstraction into the library, then reset everything
    def add_expr_fn(self, abst):
        name = f"Abs_{self.num_abs_made}"
        self.num_abs_made += 1
        abst.name = name
        abst.make_token_and_ex_abs(self)        
        self.abs_map[name] = abst        
        self.abs_register[name] = abst
        self.reset()
        
        return name

    # Get rewrites associated with abstractions
    def make_abs_rewrites(self):
        return [a.get_rewrite() for a in self.abs_map.values()]

    # get all rewrites for a current library version
    def get_rewrites(self):
        return self.lang.sem_rewrites + self.make_abs_rewrites()

    # Add prim parameter tokens into the library, used during dreaming and wake,
    # should not be active during proposal or integration
    def add_prims(self, prims):
        
        prim_name_to_vals = {}
        
        for i, prim in prims:
            for j, val in enumerate(prim):
                name = f'S_{i}_{j}_'
                self.tokens.append(LibToken(name, [], 'Float', 'float_var'))
                prim_name_to_vals[name] = val

        self.prim_name_to_vals = prim_name_to_vals
        self.init_maps()
        
    def getToken(self, token):
        if token in self.token_map:
            return self.token_map[token]

        assert False

    # Return shape generating functions that will have dreams made for them
    def getShapeFns(self):        
        return [t for t in self.type_map['Shape'] if t.name not in self.lang.prim_skip_fns]

    # Return shape generating functions that will be considered during wake
    def getWakeSkipFns(self):
        return [t for t in self.type_map['Shape'] if t.name not in self.lang.dream_keep_fns]

    # Skip these functions during dreaming
    def getDreamSkipFns(self):                        
        return self.getWakeSkipFns() + [
            t for t in self.type_map['Float'] if t.cost_type != 'float_var'
        ]

    # Get the cost type of a token
    def getCostType(self, token):
        
        if 'E_' in token or token in ['(', ')', 'Err', ',']:
            return None

        if 'P_F' in token or 'V_F' in token or 'S_' in token:
            return 'float_var'

        if 'P_C' in token or 'V_C' in token:
            return 'cat'
        
        try:
            f = float(token)

            if f in self.lang.float_consts:
                return 'float_const'
            else:
                return 'float_var'
                
        except:
            pass
        try:
            return self.token_map[token].cost_type    
        except:
            if 'Abs' in token:
                print("Found an unaccounted for abstraction, something went wrong")
                return 'fn'

            assert False, f'Unrecognized token, something went very wrong'

# Make a library, for a given domain
def makeLibrary(
    executor, domain,
):

    if domain == 's2d':
        lang = Lang2D()
        
    elif domain == 's3d':
        lang = Lang3D()
                
    L = Library(ex=executor, lang=lang)
    
    return L
    

