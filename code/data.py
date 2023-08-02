import utils
import random
import dill
import prog_utils as pu
import prog_cost
import torch
from tqdm import tqdm

# Check if expr, defined by fn, left side (_a) and right side (_b) is valid under library definition, and previous expressions (pmap)
def is_valid_expr(fn, _a, _b, pmap):

    skip, assoc = fn.invld, fn.assoc

    const_a = False
    const_b = False

    try:
        float(_a)
        const_a = True
    except:
        pass

    try:
        float(_b)
        const_b = True
    except:
        pass
    
    if (const_a and float(_a) in skip) or (const_b and float(_b) in skip):
        return False

    if (const_a and const_b):
        return False

    if assoc and const_a:
        return False        
    
    a = _a.replace('(', '').replace(')','').split()
    b = _b.replace('(', '').replace(')','').split()
    seen = set()

    for v in a + b:
        if v in pmap:
            if v in seen:
                return False
            else:
                seen.add(v)

    return True                    

# Program data structure (loaded from primitives)
class ProgData:
    def __init__(self, lang, prims, text, args, ind):
        self.ind = ind
        self.lang = lang
        self.prims = prims
        self.text = text

        vmap = {}
        for i, prim in prims:
            for j, p in enumerate(prim):
                name = f'S_{i}_{j}_'
                vmap[name] = p

        # variable map
        self.vmap = vmap

        # inferred variable map, program, score
        self.inf_vmap = None
        self.inf_prog = None
        self.inf_score = None

        # abstracted program
        self.abs_prog = None
        self.abs_rewrites = None

        self.new_abs_prog = None        

        self.prog_info = ProgInfo(lang, prims, args, self.ind)
        self.args = args
        
    # Format a complete program from sub-programs found during wake
    def make_inf_prog(self, L):

        sub_progs, sp_error = self.prog_info.get_sub_progs(self.prog_info.best_prog)

        if sub_progs is None:
            self.inf_prog = None
            self.inf_vmap = None
            self.rev_vmap = None
            self.inf_score = None
            return 
        
        comb_prog = self.lang.comb_progs(
            [sp for sp in sub_progs]
        )

        self.inf_prog, self.inf_vmap, self.rev_vmap = \
            pu.clean_prog(
                comb_prog,
                self.lang,
                opt_prims=self.prims,
                opt_vmap=self.vmap,
                do_var_for_cats=True
            )

        self.inf_score = prog_cost.full_calc_prog_cost(
            L,
            pu.ps_tokenize(self.inf_prog),
            self.args.of_weights,
            sp_error,
        )

# Dataset class, represents an entire collection of shapes
class Dataset:
    def __init__(self, lang, args):
        self.lang = lang
        
        self.data = []

        for dind in range(args.data_size):
            # "Sample" programs -- either from a grammar or by reading from file
            try:
                prog = args.grammar.sample_shape_program()
            except Exception as e:
                print(f"Sample failed with {e}, assuming this was expected")
                break
                    
            text = prog._expr

            # take expr and parse it into primitives
            raw_prims = prog.getPrimitives(args.prec, orient=True)            
            raw_prims = utils.order_prims(raw_prims)
            prims = [(i, p) for i, p in enumerate(raw_prims)]
            
            self.data.append(ProgData(lang, prims, text, args, dind))

            
    # Add the results from the integration phase into this data structure
    def integrate_abs_results(self, L, ID):

        for ie in tqdm(ID.data):
                        
            ind = ie.ind

            # remove any removed abstractions from the current programs
            self.data[ind].prog_info.remove_bad_abs(L)
            
            prog = ie.best_prog
            prog = pu.lmbda_to_prog(prog)
            
            for k,v in ie.best_params.items():
                if k in prog:
                    prog = prog.replace(k, str(v))

            # find all sub progs in the best program from abstraction
            sub_progs = pu.split_by_union(prog)            

            try:
                for sp in sub_progs:
                    self.data[ind].prog_info.add(L, sp)
            except Exception as e:
                print(f"During Integration for {ie.ind}, failed add with {e}")
                    
            if len(self.data[ind].prog_info.best_prog_cov) != len(self.data[ind].prog_info.prims):
                # Failsafe logic, that should almost never happen
                self.data[ind].prog_info.best_prog_cov = set()
                self.data[ind].prog_info.best_prog = {}
                self.data[ind].prog_info.add(
                    L,
                    prog,
                    [i for i, _ in self.data[ind].prog_info.prims]
                )
                
                print(f"During Integration for {ie.ind} recitfied by blowing it up")
                
    def sample(self):
        return random.sample(self.data, 1)[0]

    def save(self, fn):        
        dill.dump(self, open(fn, "wb"))    

    def __iter__(self):
        for d in self.data:
            yield d

    def size(self):
        return len(self.data)

    def update_progs(self):
        for d in self.data:
            d.abs_prog = d.new_abs_prog
            d.new_abs_prog = None
    
def getPrimitiveDataset(lang, args):    
    return Dataset(lang, args)

# Class representing program information for a single shape
class ProgInfo:
    def __init__(self, lang, prims, args, ind):
        self.ind = ind
        self.lang = lang
        self.prims = prims
        self.args = args

        self.cov_info = {}
        self.prog_info = {}
        
        self.best_prog = {}
        self.best_prog_cov = set()

    # remove any abstractions no longer present from best prog
    def remove_bad_abs(self, L):
        to_remove = list(set(L.abs_register.keys()) - set(L.abs_map.keys()))

        allrm = False
        for inds, prog in self.best_prog.items():
            for tr in to_remove:
                if f'{tr}(' in prog:
                    allrm = True
                    break

            if allrm:
                break

        if allrm:
            self.best_prog = {}
            self.best_prog_cov = set()


        cut_info = []
        for cp, (score, prog) in self.cov_info.items():            
            for tr in to_remove:
                if f'{tr}(' in prog:
                    cut_info.append(cp)
                    break
 
        for ci in cut_info:
            R = self.cov_info.pop(ci)        

    # find all sub programs
    def get_sub_progs(self, sub_prog_map):
        cov_prims = []
        sub_progs = []
        error = 0.
        
        for _prims,_prog in sub_prog_map.items():
            cov_prims += list(_prims)
            sub_progs.append(_prog)
            error += self.prog_info[_prog][2]
            
        if len(set(cov_prims)) != len(self.prims):
            return None, None

        return sub_progs, error

    # converts sub programs into cleaned combind program
    def get_prog(self, L, sub_prog_map):
        sub_progs, error = self.get_sub_progs(sub_prog_map)

        if sub_progs is None:
            return None, None, None
        
        comb_prog = self.lang.comb_progs(
            [sp for sp in sub_progs]
        )

        clean_prog,_,_ = pu.clean_prog(
            comb_prog,
            self.lang,
            opt_vmap={},
            opt_prims=self.prims,
            do_var_for_cats=True
        )
        
        c_score = prog_cost.full_calc_prog_cost(
            L,
            pu.ps_tokenize(clean_prog),
            self.args.of_weights,
            error
        )

        return clean_prog, sub_progs, c_score        

    # returns current best program
    def get_best_prog(self, L):
        return self.get_prog(L, self.best_prog)
    
    # greedily assemble a candidate best program
    def greedy_find_best_sub_progs(self):
        
        prims_left = set([p_ind for p_ind, _ in self.prims])

        q = []

        for pinds, (score, prog) in self.cov_info.items():
            q.append((score, set(pinds), prog))
        
        q.sort()
        
        best_sub_progs = []

        while len(prims_left) > 0:

            if len(q) == 0:                
                return None
            
            norm_score, pinds, prog = q.pop(0)

            if len(pinds - prims_left) > 0:
                continue

            best_sub_progs.append(prog)
            prims_left = prims_left - pinds

        return best_sub_progs
            
    # get info about program: what primitives it covers, how it affects objective function
    def get_info(self, L, prog):

        lines = prog
        
        L.add_prims(self.prims)
        
        for _n, _v in L.prim_name_to_vals.items():
            lines = lines.replace(_n, str(_v))    
        
        L.executor.run(lines)

        assert not L.executor.hard_error, 'added a bad program,'\
            ' this should not happen, please raise issue'
        
        prims = L.executor.getPrimitives(self.args.prec)

        match, err, max_err = self.lang.check_match(prims, self.prims, self.args)

        unnorm_cost = prog_cost.full_calc_prog_cost(
            L,
            pu.ps_tokenize(prog),
            self.args.of_weights,
            err,
        )

        score = unnorm_cost 
        
        if match is None:

            utils.log_print(
                f"!!!\nProg Info {self.ind} resulted in bad match"
                f"\n{lines}\n{prims}\n{self.prims}\n{err}\n~~~~",
                self.args,
                fn='gerr_log'
            )
                                                
            return None, score, err

        match.sort()
        cov_prims = tuple(match)        

        if len(match) != len(cov_prims):
            utils.log_print(
                f"!!!\nProg Info {self.ind} matched on the same prim"
                f"\n{lines}\n{prims}\n{self.prims}\n{match}\n{err}\n~~~~",
                self.args,
                fn='gerr_log'
            )
            return None, score, err
        
        return cov_prims, score, err

    # add sub program (prog) into this data structure
    def add(self, L, prog, match_prims=None):
        
        cov_prims, score, err = self.get_info(L, prog)        

        # We are in wake phase, then just take the match from wake to make sure we have full cov
        if len(self.prims) != len(self.best_prog_cov):
            if match_prims is not None:
                match_prims.sort()
                cov_prims = tuple(match_prims)
            
        if cov_prims is None:
            return

        # If cov prims is not None, it means we have an existing best program, so see if it has changed with the new addition
        self.update_best_prog(L, prog, cov_prims, score, err)


    # Try to change best program by looking for an overlap replacement
    def try_overlap_replace(self, L, new_prog, no_progs):
        cand_prog = {self.prog_info[p][1]: p for p in [new_prog] + no_progs}

        cand_prog, cand_sub_progs, cand_score = self.get_prog(L, cand_prog)

        best_prog, best_sub_progs, best_score = self.get_best_prog(L)

        assert best_prog is not None

        if cand_prog is None:
            return
        
        if cand_score < best_score:            
            self.T_best_prog = {self.prog_info[p][1] : p for p in cand_sub_progs}
            self.T_best_prog_cov = set()
            
            for k in self.T_best_prog.keys():
                self.T_best_prog_cov = self.T_best_prog_cov.union(k)

            if len(self.T_best_prog_cov) == len(self.prims):
                self.best_prog = self.T_best_prog
                self.best_prog_cov = self.T_best_prog_cov
        
    # try to change best program by building a new program from scratch
    def try_greedy_replace(self, L):
        greedy_sub_progs = self.greedy_find_best_sub_progs()
        
        if greedy_sub_progs is None:
            return 
        
        cand_prog = {self.prog_info[p][1]: p for p in greedy_sub_progs}

        cand_prog, cand_sub_progs, cand_score = self.get_prog(L, cand_prog)

        best_prog, best_sub_progs, best_score = self.get_best_prog(L)

        assert best_prog is not None

        if cand_prog is None:
            return
        
        if cand_score < best_score:
            self.T_best_prog = {self.prog_info[p][1] : p for p in cand_sub_progs}
            self.T_best_prog_cov = set()
            for k in self.T_best_prog.keys():
                self.T_best_prog_cov = self.T_best_prog_cov.union(k)

            if len(self.T_best_prog_cov) == len(self.prims):
                self.best_prog = self.T_best_prog
                self.best_prog_cov = self.T_best_prog_cov

    # Update best program logic
    def update_best_prog(self, L, prog, cov_prims, score, err):

        self.prog_info[prog] = (score, cov_prims, err)
        
        if cov_prims not in self.cov_info:
            self.cov_info[cov_prims] = (score, prog)            
            
        elif score < self.cov_info[cov_prims][0]:
            self.cov_info[cov_prims] = (score, prog)
        
        overlap = False
        
        for c in cov_prims:
            if c in self.best_prog_cov:
                overlap = True
                break

        if overlap is False:
            # Not a best prog yet, so greedily fill up best_prog
            self.best_prog[cov_prims] = prog
            self.best_prog_cov = self.best_prog_cov.union(cov_prims)
            return

        if overlap is True and len(self.best_prog_cov) != len(self.prims):
            utils.log_print(f"!!!ERROR: Something has gone wrong, missing progs\n {prog}\n {cov_prims}\n {self.prims}\n {self.best_prog_cov}\n {self.best_prog}\n!!!!", self.args)
            return
            
        best_sub_progs, _ = self.get_sub_progs(self.best_prog)

        if best_sub_progs is None:
            return

        overlap_cov = set()
        non_overlap_progs = []
        has_overlap = True
        
        for bsp in best_sub_progs:
            cp = set(self.prog_info[bsp][1])

            sub_res = len(cp - set(cov_prims))
            
            if sub_res == 0:
                overlap_cov = overlap_cov.union(cp)

            elif sub_res == len(cp):
                non_overlap_progs.append(bsp)
                
            else:
                has_overlap= False
                break

        try:
            if has_overlap:
                assert len(overlap_cov) == len(cov_prims)            
                self.try_overlap_replace(L, prog, non_overlap_progs)
            else:
                self.try_greedy_replace(L)
        except Exception as e:
            pass

