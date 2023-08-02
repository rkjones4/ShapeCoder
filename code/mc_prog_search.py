import time
from copy import copy, deepcopy
from random import sample

import torch
import numpy as np
import prog_cost
import utils

# Class that builds up a program 
class ProgInfo:
    def __init__(self, lang):
        self.lang = lang
        self.text = ''

        self.stack = lang.get_start_stack()

        self.prims = None
        self.struct_cost = None
        self.param_cost = None

    # Make sure prog has valid execution, and cost is up to date
    def update_info(self, L, args, err):
        try:
            self.prims = run_prog(L, self.text, args.prec)
            self.update_cost(L, args, err)
        except Exception as e:
            utils.log_print("Saw a bad mc prog search result", args, fn='mc_err', do_print=False)
            self.prims = []
            self.struct_cost = 1000000
            self.param_cost = 1000000
            self.text = ''

    # calculate cost based on program
    def update_cost(self, L, args, err):
        self.struct_cost, self.param_cost = prog_cost.split_calc_obj_fn(
            L, self.text, err, args
        )

    # Score is based mostly on finding better structural matches (in terms of obj)
    # normalized by number of primitives that the program covers
    def get_score(self):

        den = len(self.prims) * 1.
        struct_score = self.struct_cost / (den + 1e-8)
        param_score = self.param_cost / (den + 1e-8)

        score = struct_score + ((1/1000.) * param_score)

        return score        

    # Add token into program
    def add_token(self, token):

        assert token.out_type == self.stack.pop(0)[1]

        self.text += token.name

        if 'fn' in token.cost_type:
            self.text += '('
                    
        front = []
        for ip in token.inp_types:
            front.append(('token', ip))
            front.append(('text', ','))

        if len(front) > 0:            
            front[-1] = ('text', ')')
            
        self.stack = front + self.stack

        while len(self.stack) > 0 and self.stack[0][0] == 'text':
            self.text += self.stack.pop(0)[1]

    # Return the next expected type, if the program is not complete
    def get_next_type(self):
        if len(self.stack) == 0:
            return None
        assert self.stack[0][0] == 'token'
        return self.stack[0][1]

# Monte carlo sample a batch of programs
def batch_mc_sample(N, P, state, fn = None):    
    
    batch = []

    for _ in range(N.bs):
        p = ProgInfo(N.L.lang)
        # if fn is not None, then add it as the starting function 
        if fn is not None:
            p.add_token(fn)
        batch.append(p)
        
    finished = False
    c = 0
    net = N.net  

    # Expand batch, by sampling tokens, until all the programs in the batch are complete
    while(not finished):
        if c > net.ms - 2:
            batch = [b for b in batch if b.get_next_type() is None]
            break
        
        finished = net.batch_expand(batch, state)
        c += 1

    # Some reindexing logic
    def prog_fix_prim_inds(txt):        
        for i, (j, _) in enumerate(P):            
            txt = txt.replace(f'S_{i}_', f'T_{j}_')
        txt = txt.replace('T_', 'S_')
        
        return txt
        
    for b in batch:
        b.text = prog_fix_prim_inds(b.text)        
        
    return batch

# Try executing the program
def run_prog(L, lines, prec, soft_error=False):
    for _n, _v in L.prim_name_to_vals.items():
        lines = lines.replace(_n, str(_v))    
    
    L.executor.run(lines)
    prims = L.executor.getPrimitives(prec)
    ord_prims = utils.order_prims(prims)

    # Should we be checking for a soft error, to reject dreams
    if soft_error:
        return ord_prims, L.executor.soft_error, L.executor.hard_error

    assert not L.executor.hard_error
    
    return ord_prims

# helper function to call update info over batch, 
def update_prog_info(L, batch_progs, args):
    
    for bp in batch_progs:
        bp.update_info(L, args, 0.)

# get default prog for primitive
def get_default_prog(L, prim, args):
    
    ind = prim[0]

    prog = ProgInfo(L.lang)
    
    for token in L.lang.get_base_prim_tokens(prim[0], prim[1]):
        prog.add_token(token)                
    
    update_prog_info(L, [prog], args)
    prog.match_prims = [ind]
    
    return prog
        

# Checks for a mtch between predicted program, bprog, and data point d
def calc_match(L, d, bprog, args):

    P = d.P
    
    assert bprog is not None

    bprim = bprog.prims

    if len(bprim) == 0:
        return False
    
    # first check direct match
    match, err, max_err = L.lang.check_match(bprim, P, args)
    
    if match is not None:

        bprog.update_cost(L, args, err)
        bprog.match_prims = match        
        
        return True

    # otherwise try to align the program
    match, err, max_err = L.lang.check_align_match(bprog, bprim, P, d, args)

    if match is not None:

        try:
            moved_prims = run_prog(L, bprog.text, args.prec)

            move_match, _, _ = L.lang.check_match(moved_prims, P, args)

            if not move_match:
                return False
            
            bprog.update_info(L, args, err)
            bprog.match_prims = match                
        
            return True
        
        except:
            pass
        
    return False
    

# Sample a batch of programs to explain shape d, with network N
def sample_progs(L, N, d, args):
    P = d.P

    # set default, naive program as best
    
    best_prog = get_default_prog(L, P[0], args)
    best_score = best_prog.get_score()

    count = 0
    uniq = set()

    # to maintain indentation
    if True:
        N.build_net_state(P)

        # maximum time to spend on wake search per program
        wake_time = args.wk_prog_search_timeout
        t = time.time()
        
        while(time.time() - t < wake_time):
            state = N.build_state(P)            

            # sample batch
            batch_progs = batch_mc_sample(N, P, state)

            # update program info
            update_prog_info(L, batch_progs, args)
            
            for bprog in batch_progs:
                count += 1
                uniq.add(bprog.text)
                
                bscore = bprog.get_score()

                # continue if we haven't beat score
                if bscore > best_score:
                    continue

                is_match = calc_match(
                    L, d, bprog, args
                )

                # continue if we don't have a match
                if not is_match:
                    continue
                
                mscore = bprog.get_score()

                # if we find better program, keep it
                if mscore < best_score:
                    best_score = mscore
                    best_prog = bprog

    # remove all prims that are covered by the best found program
    rem_prims = [p for p in P if p[0] not in best_prog.match_prims]
    return best_prog, rem_prims, (count, len(uniq))

# Run a program search using network N to find a program from L to explain
# the primitives in d
def mc_prog_search(L, N, d, args):        
    
    sub_progs = []

    d.P = deepcopy(d.prims)

    num_progs = None
    
    with torch.no_grad():
        # until all of the primitives are covered
        while len(d.P) > 0:
            # sample a program to explain a subset of the target primitives
            best_sub_prog, d.P, NP = sample_progs(L, N, d, args)
            sub_progs.append(best_sub_prog)

            if num_progs is None:
                num_progs = NP

    # return the list of expressions to cover all of the primitives of d
    return sub_progs, NP
