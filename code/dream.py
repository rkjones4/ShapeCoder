from copy import deepcopy
import time
import dill, os
import random
from random import sample
import utils
import mc_prog_search as mps
import data
from tqdm import tqdm
import prog_utils as pu
import prog_cost

# add var holes to text
def expr_to_var_holes(text):

    for O in ['(',')',',']:
        text = text.replace(O, f' {O} ')

    m = {}

    e = []
    
    for token in text.split():
        try:
            float(token)
            nm = f'VAR_{len(m)}_'
            m[nm] = token
            e.append(nm)
        except:
            e.append(token)
        
            
    return ''.join(e), m

# class for combined scenes, composed of multiple sub function scenes
class CombDream:
    def __init__(self, sub_dreams, args, L, D):

        self.sub_dreams = []
        self.prims = []
        
        for _d in sub_dreams:

            d = deepcopy(_d)

            # maybe randomly translate the sub-dream
            if random.random() < args.dm_unalign_chance:
                sub_prims, d.prim_move = self.maybe_unalign(d, args, L)
            else:
                sub_prims, d.prim_move = d.prims, None
                
            tmap = {}
            for i, prim in enumerate(sub_prims):
                tmap[i] = len(self.prims)
                self.prims.append((len(self.prims), prim))

            self.change_prim_inds(d, tmap)            
            self.sub_dreams.append(d)                            

        # check the dream primitives are valid
        self.is_valid = L.lang.check_valid_dream_prims([p for _,p in self.prims], args)

        if not self.is_valid:
            return
        
        # add random distractor primitives into the scene
        if random.random() < args.dm_distract_prim_chance:                        
            cur_prims = [p for _,p in self.prims]
            for _ in range(10):                
                new_prims = cur_prims + L.lang.sample_distractors(D, args)                               
                if L.lang.check_valid_dream_prims(new_prims, args):
                    self.prims = list(enumerate(new_prims))
                    break

        ord_prims, ord_inds = utils.order_prims([P for _, P in self.prims], True)

        self.prims = list(enumerate(ord_prims))

        TMAP = {}
        for i,j in enumerate(ord_inds):
            TMAP[j] = i

        # fix primitive indexing
        for d in self.sub_dreams:
            tmap = {i:TMAP[i] for i in d.comb_prim_inds}
            self.change_prim_inds(d, tmap)
            
        
    def maybe_unalign(self, d, args, L):

        moved_text, moved_prims, move = L.lang.dream_sample_unalign(d.text, d.prims, args)

        # THIS CHECKS WHETHER OLD program on new prims still creates old prims
        
        DL = deepcopy(L)
        DL.add_prims(list(enumerate(moved_prims)))        

        try:
            ex_dt_prims = mps.run_prog(DL, d.text, args.prec)                                
            ex_mt_prims = mps.run_prog(DL, moved_text, args.prec)
        except:
            return d.prims, None
            
        # THIS CHECKS
        # does executing moved program text == moving previous prims

        creates_old = (ex_dt_prims == d.prims)        
        creates_new = (ex_mt_prims == moved_prims)
        
        if creates_old and creates_new:
            return moved_prims, move
        else:
            return d.prims, None
        
    def change_prim_inds(self, d, tmap):
        d.comb_prim_inds = []
        for i, j in tmap.items():
            d.text = d.text.replace(f'S_{i}_', f'T_{j}_')
            d.comb_prim_inds.append(j)
            
        d.text = d.text.replace('T_', 'S_')

    def get_sub_shape_info(self):
        return [d.get_info() for d in self.sub_dreams]    

# Checks whether found program is better than the one that can be naively parsed
def better_than_naive_prog(L, prims, struct_cost, param_cost, args):
    
    naive_struct_cost, naive_param_cost = L.lang.get_naive_costs(prims, args)        
        
    if (struct_cost > naive_struct_cost) or (              
        struct_cost == naive_struct_cost and
        param_cost >= naive_param_cost):
        return False
    else:
        return True

# Sub-dream function for a single function    
class Dream:
    def __init__(self, text, args, L):

        self.is_valid = True
        self.invalid_reason = None

        # try running program
        try:
            self.prims, soft_error, hard_error = mps.run_prog(L, text, args.prec, soft_error=True)
        except:
            self.is_valid = False
            self.invalid_reason = 'exec error'
            return

        # See if the dream had any errors, if so record the error         
        if hard_error:
            self.is_valid = False
            self.invalid_reason = "hard error"
        
        if soft_error and 'Abs' not in text:            
            self.is_valid = False
            self.invalid_reason = "soft error"
            return
            
        if not L.lang.check_valid_dream_text(text, args):
            self.is_valid = False
            self.invalid_reason = "Bad Text"
            return
        
        # validate prims
        if not L.lang.check_valid_dream_prims(self.prims, args):
            self.is_valid = False
            self.invalid_reason = "Bad Prims"
            return
        
        if not self.make_valid_prog(text, L, args):
            self.is_valid = False
            self.invalid_reason = "Bad reproduce"
            return 

        # get cost of the dream
        struct_cost, param_cost = prog_cost.split_calc_obj_fn(L, self.text, 0., args)
        
        if not better_than_naive_prog(
            L,
            self.prims,
            struct_cost,
            param_cost,
            args
        ):
            # only keep dreams that would improve our objective function, if inferred
            self.is_valid = False
            self.invalid_reason = "Bad Obj Fn"
            return
                            
    def get_info(self):

        fn = self.comb_fn
        text = self.text
        inds = self.comb_prim_inds

        return (fn, text, set(inds))

    # format original text into dream format
    def make_valid_prog(self, orig_text, L, args):
        
        used_to_ex_map = {}
        
        new_text = orig_text
        
        pmap = {}
        for i, _prim_vals in enumerate(self.prims):
            for j, val in enumerate(_prim_vals):
                name = f'T_{i}_{j}_'
                pmap[name] = val


        hole_text, hole_map = expr_to_var_holes(new_text)
        smap = {}

        for hk, hv in hole_map.items():
            for k, v in pmap.items():
                if abs(float(v) - float(hv)) < 1. / 10**(args.prec):
                    smap[hk] = k
                    break
            
        for k,v in smap.items():
            hole_text = hole_text.replace(k,v)
                
        if 'VAR_' in hole_text:
            return False
            
        self.text = hole_text.replace('T_', 'S_')
            
        return True
            
        
            
                                
# Make a set of dreams for function (fn)                        
def make_fn_dreams(
    fn, OL, N, D, args
):

    num_dreams = args.num_dreams
    dreams = []
    
    t = time.time()
    pbar = tqdm(total=num_dreams)

    err_count = 0

    valid_fail = 0.
    valid_suc = 0.
    
    while len(dreams) < num_dreams and (time.time() - t) < args.dm_timeout:

        # if we fail to often, we might waste too much time on this abstraction, so just skip it
        if (valid_fail > 20000) and (valid_suc < 10):
            utils.log_print(
                f"Breaking early for dreaming for {fn}.\n"
                f"Ratio of valid Fail to Suc {valid_fail} -> {valid_suc} was too high \n",
                args, fn='derr_log'
            )
            return None
        
        pbar.update(len(dreams) - pbar.n)

        P = D.sample().prims
        
        L = deepcopy(OL)
        L.add_prims(P)        
        
        N.init_with_lib(L, skip_fns = L.getDreamSkipFns())

        # Sample from library function
        # adding function to begin state of inference of a dummy recognition network
        
        state = N.build_state(P, fn)        
        
        try:        
            samp_progs = mps.batch_mc_sample(N, P, state, fn)
        except Exception as e:
            err_count += 1
            if err_count < 5:
                utils.log_print(f"Err for  {fn} : {e}", args, fn='derr_log')
                
            if err_count > 100:
                utils.log_print(f"Broke for {fn}, too many sampling errors", args, fn='derr_log')
                return None
            continue
            
        for sp in samp_progs:
            # for everything in batch, try to record as dream
            
            d = Dream(sp.text, args, L)                        
            
            if d.is_valid:
                valid_suc += 1
                dreams.append(d)
                
            else:
                valid_fail += 1
                                
        
    pbar.close()
    return dreams

# convert the fn specific dreams into combined scenes that can train the recognition network
def make_train_data(L, fn_to_dreams, args, D):
    fn_datasets = {fn:[] for fn in fn_to_dreams.keys()}

    # remove bad abstractions
    bad_fns = []
    for k in fn_datasets.keys():
        if len(fn_to_dreams[k]) == 0:
            bad_fns.append(k)

    for k in bad_fns:
        utils.log_print(f"Bad fn {k} was skipped", args, fn='derr_log')
        fn_datasets.pop(k)

    # end remove bad abstractions
    
    pbar = tqdm(total=args.num_dreams)

    count = 0 
    
    while(True):
        # See which functions still don't have enough representation in training data
        
        rem_fns = [fn for fn, data in fn_datasets.items() if len(data) < args.num_dreams]
        left = min([len(FD) for FD in fn_datasets.values()])        
        pbar.update(left - pbar.n)
        
        if len(rem_fns) == 0:
            # if all functions are covered, break
            break

        # Sample a set of sub dreams
        num_sub_fns = random.randint(1, args.dm_max_sub_fns)        
        sub_fns = []        
        for _ in range(num_sub_fns):
            sub_fns.append(sample(rem_fns, 1)[0])

        sub_dreams = []        
        for fn in sub_fns:
            d = sample(fn_to_dreams[fn], 1)[0]
            d.comb_fn = fn
            sub_dreams.append(d)

        # Create combind scene
        comb_dream = CombDream(sub_dreams, args, L, D)                        

        if not comb_dream.is_valid:
            continue
        
        comb_prims = comb_dream.prims
        
        # List of (fn, sub_prog_text, primitive indices)
        sub_shape_info = comb_dream.get_sub_shape_info()        

        # Record for training
        for fn, sub_prog_text, prim_inds in sub_shape_info:
            fn_datasets[fn].append((sub_prog_text, comb_prims, prim_inds))

    pbar.close()
    return fn_datasets

# Entrypoint to dream phase
def make_dreams(
    L, # Library
    N_dummy,
    D, # list of primitives
    args
):    
        
    fn_to_dreams = {}

    # where dreams are saved
    SPTH = f'{args.outpath}/{args.exp_name}/shared'
    past_fn_dreams = os.listdir(SPTH)

    # dream for every function that generates a 'shape' type
    for fn in list(L.getShapeFns()):
        fn_name = fn.name

        # Don't redream for old functions
        if f'{fn_name}.pkl' in past_fn_dreams:
            utils.log_print(f"Loading old dreams for {fn_name}", args)
            fn_to_dreams[fn_name] = dill.load(open(f'{SPTH}/{fn_name}.pkl', 'rb'))
            
        else:
            utils.log_print(f"Making new dreams for {fn_name}", args)
            
            t = time.time()
            # sample new dreams for function
            new_fn_dreams = make_fn_dreams(
                fn, L, N_dummy, D, args
            )
            if new_fn_dreams is not None:
                fn_to_dreams[fn_name] = new_fn_dreams
                if fn.name not in L.lang.dream_fns_to_rm:
                    # Check if we should save dreams for this function
                    utils.log_print(f"Saving dream for {fn_name}", args)
                    dill.dump(new_fn_dreams, open(f'{SPTH}/{fn_name}.pkl', 'wb'))   
                # Log results
                num_dreams = len(new_fn_dreams)
                time_took = time.time() - t
                utils.log_print(f"Made dreams for {fn_name} | Num : {num_dreams} | Time : {time_took}", args, fn='d_log')

    # Make and return training data
    dream_data = make_train_data(L, fn_to_dreams, args, D)

    return dream_data

