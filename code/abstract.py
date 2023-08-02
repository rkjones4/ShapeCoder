import torch
import library as lib
import prog_utils as pu
from copy import deepcopy
from tqdm import tqdm
from scipy import stats
import json
import prog_cost
import time
import dill
import utils
import canon 
import extract as extr
import op_ord as oord
from tqdm import tqdm

VERBOSE = True
PPROGS = False

# Hold information about saturation / extraction time from e-graphs
TOI = {'c': [], 't': [], 'sat_c': [], 'sat_t': []}

# Helper function that converts e-graph extract output back into normalized form
def _clean_egg_expr(expr, params, lang):
    if '(' in expr:
        r = pu.split_expr(expr)
        R = []
        for a in r[1:]:
            R.append(_clean_egg_expr(a.strip(), params, lang))
        mid = ' '.join(R)
        return f'({r[0]} {mid})'
        
    else:
        if expr in lang.cat_tokens:
            r =  f'P_C_{len(params)}_'
            params[r] = expr
            return r

        try:
            v = float(expr)
            r = f'P_F_{len(params)}_'
            params[r] = v
            return r
        
        except:
            return expr
                    
# Converts e-graph extract output back into normalized form
def clean_egg_expr(expr, ie_params, lang):

    for k,v in ie_params.items():
        expr = expr.replace(k, str(v))

    cparams = {}

    cexpr = _clean_egg_expr(expr, cparams, lang)

    return cexpr, cparams

# Called on the output of the e-graph extraction output
def egg_postprocess(L, ne_egg_expr, ie_params):
    
    egg_prog, egg_params = clean_egg_expr(ne_egg_expr, ie_params, L.lang)
        
    return egg_prog, egg_params, 0.0

# dedup any re-used parameters, in terms of direct value matches
def do_dedup(_expr, _params):

    expr = deepcopy(_expr)
    params = deepcopy(_params)
    
    repl = []
    svals = {}
    
    for k,_v in list(params.items()):

        if isinstance(_v, str):
            continue
        
        v = round(_v,2)
        
        if v not in svals:
            svals[v] = k
        else:
            repl.append((k, svals[v]))
            params.pop(k)

    for k,v in repl:
        expr = expr.replace(k, v)
        
    return expr, params

# Main entrypoint into refactor operation: given input expr (ie) and library (L) returns egraph output
def egg_extract(L, ie, args, mode):

    # Whether to start from the best prog at the beginning of this abstraction phase
    # or whether we should remove all abstractions from the program (input vs bout)
    if mode == 'input':
        egg_expr, ie_params = ie.get_input_prog()

    elif mode == 'bout':
        egg_expr, ie_params = ie.get_bout_prog()
    else:
        assert False, f'bad extract mode {mode}'

    # optional speedup
    egg_expr, ie_params = do_dedup(egg_expr, ie_params)

    # get one step rewriters
    egg_os_rws = get_os_rws(ie_params)

    # get the rewrites we should saturate
    egg_sat_rws = L.lang.base_rewrites + L.make_abs_rewrites()

    # run refactor operation
    egg_out, egg_cost, had_sat, had_to = extr.safe_extract(        
        egg_expr,
        egg_os_rws,
        egg_sat_rws,
        args.egg_max_rounds,
        args.egg_max_nodes,
        args.egg_max_time,
    )

    # had a timeout
    if had_to:
        TOI['c'].append(1.0)
        TOI['t'].append(1.0)
    else:
        TOI['c'].append(0.0)
        TOI['t'].append(0.0)

    # had a saturation of the e-graph
    if had_sat:
        TOI['sat_c'].append(1.0)
        TOI['sat_t'].append(1.0)
    else:
        TOI['sat_c'].append(0.0)
        TOI['sat_t'].append(0.0)

    # clean up and return output
    egg_prog, egg_params, egg_error =  egg_postprocess(L, egg_out, ie_params)

    return egg_prog, egg_params, egg_error, egg_cost

# Different input expressions (semantically equivalent) to try
EGG_INP_MODES = [
    'bout',    
    'input',    
]

# Given a library version, and a set of input expresisons (ID), calculate the objective function value
def calc_obj_fn(L, ID, args, ret_freq_info=False):
    
    costs = []
    influences = []
    weights = args.of_weights
    avg_counts = {ttype:[] for ttype in weights}

    fn_freq = {}

    # for each program
    for ie in tqdm(ID.data):
        ie.abs_prog = None
        ie.abs_prog_cost = None
        ie.abs_params = None

        # record starting program cost
        if ie.best_prog_cost is None:
            ie.best_prog_cost = prog_cost.full_calc_prog_cost(
                L, pu.ps_tokenize(ie.best_prog), weights, 0.0
            )

        abs_set = False
        
        for egg_mode in EGG_INP_MODES:
            # run refactor with e-graphs on program
            egg_prog, egg_params, egg_error, rst_egg_cost = egg_extract(
                L, ie, args, egg_mode
            )
            # get the cost of the output program
            egg_cost = prog_cost.full_calc_prog_cost(
                L, pu.ps_tokenize(egg_prog), weights, egg_error, avg_counts
            )                                       

            # make sure the output of the e-graph extract step covers primitives in the input shape
            egg_match_prims = pu.find_match_prims(
                pu.lmbda_to_prog(egg_prog),
                egg_params,
                L,
                ie.match_prims,
                args
            )
                    
            if egg_match_prims is not None and egg_cost < ie.best_prog_cost:
                if ie.abs_prog_cost is None or egg_cost < ie.abs_prog_cost:
                    # if we improved our fit, record
                    
                    f_prog = egg_prog
                    f_cost = egg_cost
            
                    ie.abs_prog = egg_prog
                    ie.abs_prog_cost = egg_cost
                    ie.abs_params = egg_params
                    abs_set = True
                    
        if not abs_set:            
            f_prog = ie.best_prog
            f_cost = ie.best_prog_cost
                
        if PPROGS:
            print(f'{ie.ind} : {f_prog}')

        # record final cost, and how much we should weight each expression
        costs.append(f_cost)
        influences.append(ie.weight)

    # calculate average counts for terms in objective function
    avg_counts = {k:torch.tensor(v).float().mean().item() for k,v in avg_counts.items()}

    avg_counts['lib'] = 0.
    ab_cnt = 0

    # see if the frequency of any functions in the library changed
    func_changes, func_freq = ID.get_abs_func_changes(L.abs_map.keys())

    # add in library objective function terms
    for name, ab in L.abs_map.items():
        ab_cnt += 1
        sig = ab.exp_fn(ab.param_names)
        
        abs_pref_score = L.lang.get_abs_pref_score(
            sig,
            1.0, 
            args,
            L
        )
        
        if abs_pref_score is None:
            # max badness
            avg_counts['lib'] += 2.
        else:        
            avg_counts['lib'] += 1. - abs_pref_score

    avg_lib_cost = avg_counts['lib'] * weights['lib']
        
    avg_prog_cost = ((torch.tensor(costs).sum()) / \
                     torch.tensor(influences).sum()).item()

    ret_string = ""
    
    ret_string += f"Total Obj Cost : {avg_prog_cost + avg_lib_cost}\n"
    ret_string += f"  Total Lib Cost : {avg_lib_cost} (Cnt : {ab_cnt})\n"
    ret_string += f"  Total Prog Cost : {avg_prog_cost}\n"
    
    for ttype in avg_counts:
        ret_string += f"    Count {ttype} : {round(avg_counts[ttype], 3)} ({weights[ttype]})\n"

    if ret_freq_info:        
        return avg_prog_cost + avg_lib_cost, ret_string, func_changes, func_freq
    else:
        return avg_prog_cost + avg_lib_cost, ret_string

# Helper function to remove all abstractions from a program
def make_no_abs_prog(prog, params, L):
    
    prog, params = pu.remove_abs(
        prog, params, L, list(L.abs_map.keys())
    )
    
    return prog, params

# Get rewrites that record the value of input parameter e-nodes
def get_var_rec_rws(prms):
    rws = []
    for k,v in prms.items():
        if '_F_' in k:
            rws.append([f"vrrw{len(rws)}", k, k, f"var_record:{v}"])
    return rws

# Get rewrites for categorical variables
def get_abs_cat_rws(prms):
    rws = []
    for p in prms:
        if '_C_' in p:
            rws.append([f'cat_rw_{len(rws)}', p, prms[p], "none"])

    return rws

# one step rewrites
def get_os_rws(prms):
    return get_abs_cat_rws(prms) + \
        get_var_rec_rws(prms)

# IntegExpr corresponds with a given program from the wake phase
class IntegExpr:
    def __init__(
        self, L, args, sig, params, ind, weight, match_prims
    ):

        # which primitives this program should be matching
        self.match_prims = match_prims
        self.ab_rw_max_error = args.ab_rw_max_error
        self.prec = args.prec
        self.ind = ind
        self.weight = weight
        self.params = params
        
        self.prog = pu.func_to_lmbda(sig)

        self.best_prog = self.prog
        self.best_prog_cost = None
        self.best_params = params
        
        self.abs_prog = None
        self.abs_prog_cost = None
        self.abs_params = None

        if args.hold_out_infer == 'y':
            return
        
        self.no_abs_prog, self.no_abs_params = make_no_abs_prog(self.prog, self.params, L)
                
    def get_input_prog(self):
        return self.prog, self.params

    def get_bout_prog(self):
        return self.no_abs_prog, self.no_abs_params

# Holds the program data during the integration phase
class IntegData:

    def __init__(self, L, prog_data, args):

        self.data = []
                
        for sig, params, ind, weight,match_prims in tqdm(prog_data.data):
            self.data.append(IntegExpr(
                L, args, sig, params, ind, weight, match_prims
            ))        
            

    # Called whenever a new library version is accepted
    def update_abs_progs(self, lang):

        for d in self.data:
            if d.abs_prog is None:
                continue
            
            d.best_prog = d.abs_prog
            d.best_params = d.abs_params
            d.best_prog_cost = d.abs_prog_cost

            d.abs_prog = None
            d.abs_params = None
            d.abs_prog_cost = None


    # Called whenever a function is removed from the library
    def remove_abs_progs(self, L):

        to_remove = list(set(L.abs_register.keys()) - set(L.abs_map.keys()))

        for d in self.data:

            d.best_prog_cost = None
            
            d.best_prog, d.best_params = pu.remove_abs(
                d.best_prog, d.best_params, L, to_remove
            )

            d.prog, d.params = pu.remove_abs(
                d.prog, d.params, L, to_remove
            )

    # See how the frequency of abstraction functions changed during this library update
    def get_abs_func_changes(self, Abs):
        B = {a:0. for a in Abs}
        A = {a:0. for a in Abs}
        T = len(self.data) * 1.
        for ie in self.data:
            for ab in Abs:
                B[ab] += ie.best_prog.count(f'({ab} ')
                if ie.abs_prog is not None:
                    A[ab] += ie.abs_prog.count(f'({ab} ')
                else:
                    A[ab] += ie.best_prog.count(f'({ab} ')

        changed = [ab for ab in Abs if (B[ab] / T) > (A[ab] * 2. / T)]
        return changed, {ab:round(c/T,3) for ab,c in A.items()}

# Helper class for abstraction related logic, these are what get added in to the library
class Abstraction:
    def __init__(
        self, expr, L, args
    ):

        self.rw_inp = ''
        self.rw_rels = {}
        self.rw_prms = []
        self.rw_prm_map = {}

        self.parse_expr(
            expr, L
        )        
        
        self.rw_parse_expr(expr, L)

        self.rw_inp = self.rw_inp.strip()        
        
        self.rw_info = []
        
        for k,v in self.rw_prm_map.items():
            for A,B in self.rw_rels.items():
                self.rw_rels[A] = B.replace(k, v)

        for v,e in self.rw_rels.items():

            expr = e.replace('(', ' ').replace(')', ' ').split()
            res = oord.make_op_order(expr, L.lang)
            cmds = '+'.join([f'{r[0]}|{r[1]}|{r[2].split("_")[1]}' for r in res])
                        
            self.rw_info.append(f'{v}={cmds}')
            
        self.rw_info = f"abs:{args.ab_rw_max_error}:{','.join(self.rw_info)}"

        
    def make_token_and_ex_abs(self, L):

        argus = [self.rev_param_map[pn] for pn in self.param_names]

        sig = self.exp_fn(argus)
        
        param_info = pu.get_param_info(
            sig,
            self.param_names
        )

        # This token represents the libraries view of this abstraction
        token = lib.LibToken(
            self.name,
            self.param_out_types,
            self.out_type,
            'fn',
            param_info,
            L.lang.lookup_fn,
            L.token_map
        )
                
        prog = pu.lmbda_to_prog(sig)

        # ex_abs is added into the executor, so the abstraction can create geometry
        ex_abs = L.ex_abst_class(self.name, ','.join(argus), prog, None)
        
        self.token, self.ex_abs = token, ex_abs    

    # Parses the expr into a rewrite for the e-graph logic
    def rw_parse_expr(self, expr, L):
        
        assert '(' in expr

        r = pu.split_expr(expr)
        fn = r[0]

        fn_types = L.token_map[fn.strip()].inp_types

        assert len(fn_types) == len(r) - 1
        self.rw_inp += f' ({fn}'

        def get_name(cnt):
            name = '?'
            while cnt >= 0:
                name += str(chr(97+ (cnt % 26)))
                cnt -= 26
            return name

        
        for param_expr, fn_typ in zip(r[1:], fn_types):
            if fn_typ == 'Float':
                cnt = len(self.rw_prms) + len(self.rw_rels)
                name = get_name(cnt)
                self.rw_inp += f' {name}'

                if '_F_' in param_expr and \
                   '(' not in param_expr and \
                   param_expr not in self.rw_prm_map:

                    self.rw_prms.append(name)
                    self.rw_prm_map[param_expr] = name
                    
                else:
                    self.rw_rels[name] = param_expr

            elif '(' not in param_expr and '_C_' in param_expr:
                cnt = len(self.rw_prms) + len(self.rw_rels)
                name = get_name(cnt)
                self.rw_inp += f' {name}'

                if param_expr not in self.rw_prm_map:
                    self.rw_prms.append(name)
                    self.rw_prm_map[param_expr] = name
                                    
            else:
                if '(' in param_expr:
                    self.rw_parse_expr(param_expr, L)                    
                else:
                    self.rw_inp += f' {param_expr}'
                
        self.rw_inp += ')'


    # Parses the string expression into function logic of the abstraction
    def parse_expr(self, expr, L):
        
        param_names = []
        param_types = []
        param_out_types = []
        
        raw_string = ''
        
        tokens = pu.nd_tokenize(expr)
        
        self.tokens = tokens

                
        param_map = {}

        out_type = None
        
        for t in tokens:

            if t in L.token_map and out_type is None:
                out_type = L.token_map[t].out_type                
            
            if '_F_' in t:
                
                if t in param_map:
                    vname = param_map[t]
                    
                else:
                    pos = len(param_names)
                    vname = f'V_F_{pos}_'
                    param_map[t] = vname
                    param_names.append(vname)
                    param_types.append('Float')
                    param_out_types.append('Float')
                raw_string += f'{vname} '                

            elif '_C_' in t:
                
                if t in param_map:
                    vname = param_map[t]
                else:
                    pos = len(param_names)
                    ot = t.split('_')[2]
                    vname = f'V_C_{ot}_{pos}_'
                    param_map[t] = vname
                    param_names.append(vname)
                    param_types.append('Cat')
                    param_out_types.append(ot)

                raw_string += f'{vname} '
                
            else:
                    
                if t == '(':
                    raw_string += t
                elif t == ')':
                    raw_string = raw_string[:-1] + t + ' '
                else:
                    raw_string += f'{t} '

        raw_string = raw_string.strip()
        
        assert raw_string == expr, f'uh oh, something broke : {expr}'
        
        def exp_fn(params):
            assert len(params) == len(param_names), 'bad number of params'
            expr = raw_string
            for p, pn in zip(params, param_names): 
                expr = expr.replace(pn, str(p))

            return expr
        
        self.out_type = out_type
        self.param_names = param_names
        self.param_types = param_types
        self.param_out_types = param_out_types
                
        self.exp_fn = exp_fn
        self.param_map = param_map
        self.rev_param_map = {v:k for k,v in param_map.items()}

    # Returns the fields that the e-graph rewriter expects
    def get_rewrite(self):

        rw_out = f'({self.name} {" ".join(self.rw_prms)})'
        
        d = [
            f'abs_rw_{self.name}',
            self.rw_inp,
            rw_out,
            self.rw_info
        ]                
        
        return d

# Given an old library OL, adds an abstraction (defined by sig) into a new library version
def make_new_library(OL, sig, args):    

    abst = Abstraction(
        sig, OL, args
    )
    NL = deepcopy(OL)    
    NL.add_expr_fn(abst)
    NL.newest_abs = abst.name
    return NL

# makes a copy of the program data
def make_new_integ_data(ID, sig, lang):
    NID = deepcopy(ID)
    return NID

# Class that holds the candidate proposal abstraction functions
class CandProposalData:
    def __init__(self, args):
        self.args = args
        self.freq_data = {}
        self.gain_data = {}
        self.weight_data = {}

        self.struct_to_sigs = {}
        self.sig_to_struct = {}

    # update the score of candidates, if they have been covered by the added signature
    def cover_weights(self, cov_inds, added_sig):

        for ci in cov_inds:
            if ci in self.weight_data:
                # Coverage discount factor
                self.weight_data[ci] *= self.args.ab_cov_discount

        added_sig = pu.lmbda_to_prog(added_sig)
        struct_sig = self.sig_to_struct[added_sig]

        for msig in self.struct_to_sigs[self.sig_to_struct[added_sig]]:
            # Shared structural match discount factor
            self.gain_data[msig] *= self.args.ab_struct_match_discount        

    # record weights for each ind
    def add_inds_and_weights(self, inds, weights):
        for i,w in zip(inds, weights):
            if i not in self.weight_data:
                self.weight_data[i] = w

    # add a proposal function into the data structure
    def add_cand_prop(self, prop):
        
        sig, gain, inds, struct_sig = prop

        if struct_sig is not None:
            self.sig_to_struct[sig] = struct_sig
            if struct_sig not in self.struct_to_sigs:
                self.struct_to_sigs[struct_sig] = []
            self.struct_to_sigs[struct_sig].append(sig)
                    
        if sig not in self.freq_data:
            self.freq_data[sig] = set()
            self.gain_data[sig] = gain

        self.freq_data[sig].update(inds)

    # get the top scoring candidate proposal
    def pop_top_prop(self, L):
        total = sum(self.weight_data.values()) * 1.

        def calc_weight(V):
            freq = sum([self.weight_data[v] for v in V]) / total
            return freq

        best_sig = None
        best_score = 0
        best_info = None

        for sig, V in self.freq_data.items():
            gain = self.gain_data[sig]
            freq = calc_weight(V)

            abs_pref_score = L.lang.get_abs_pref_score(sig, freq, self.args, L)
            lib_weight = self.args.of_lib_weight

            if abs_pref_score is not None:
                score = gain * freq + (lib_weight * abs_pref_score)
                
                if score > best_score:
                    best_score = score
                    best_sig = sig
                    best_info = (score, gain, freq, abs_pref_score, V)
                    
        if best_sig is None:
            return None, None, None

        # Don't pick this again
        self.freq_data[best_sig] = set()
        
        return round(best_score,3), pu.func_to_lmbda(best_sig), best_info

    # rank the candidate proposals by score
    def rank_proposals(self, L):

        total = sum(self.weight_data.values()) * 1.

        def calc_weight(V):
            freq = sum([self.weight_data[v] for v in V]) / total
            return freq
        
        ranks = []
        
        for sig, V in self.freq_data.items():
            gain = self.gain_data[sig]
            freq = calc_weight(V)
            
            abs_pref_score = L.lang.get_abs_pref_score(sig, freq, self.args, L)        
            
            lib_weight = self.args.of_lib_weight

            if abs_pref_score is not None:
                score = gain * freq + (lib_weight * abs_pref_score)
                ranks.append((round(score,4), round(gain,3), round(freq,3), round(abs_pref_score,3), sig))

        ranks.sort(reverse=True)

        return ranks

    # utility print function 
    def print_top_info(self, num, L):
        proposals = self.rank_proposals(L)

        for i, (score, gain, freq, aps, sig) in enumerate(proposals[:num]):
            print(
                f'Prop {i} '
                f'[{round(score,3)}, {round(gain,3)}, {round(freq,3)}, {round(aps, 3)}]\n'
                f'    -> {sig}'
            )
    
# Helper class that consumes the programs discovered from wake
class ProgData:
    def __init__(self, args):
        self.args = args

        self.data = []

        self.prog_to_sigs = {}

        self.sig_to_params = {}

    # Prints the structural clusters identified
    def print_clusters(self):
        info = []
        for sig, (param_list, _) in self.sig_to_params.items():
            info.append((len(list(param_list.values())[0]), sig, param_list))

        info.sort(reverse=True)

        for I, (num, sig, param_list) in enumerate(info[:4]):
            print(f"~~ Sig Rank {I} ({num}) ~~")
            print(f"  Prog Sig : {sig}")
            for k, V in param_list.items():
                print(f"        {k} : {V[:5]}")
            

    # add program into the class
    def add_prog(self, prog, vmap, ind, L, prims):

        
        if prog.count(')') > prog.count('('):
            assert False

        # canonicalize the program
        sig, params = canon.canonicalize(
            L, prog, vmap, self.args, prims
        )

        # find which primitives the program should cover
        match_prims = pu.find_match_prims(
            sig, params, L, prims, self.args
        )

        assert match_prims is not None
        
        self.data.append((sig, params, ind, 1.0, match_prims))

        # find all valid sub expressions for candidate proposal logic
        sub_exprs = pu.get_all_order_sub_exprs(
            sig, params, L,
            self.args.ab_num_all_ord,
        )

        # record different sub-expressions signatures and parameters into clusters
        for se_sig, se_params in sub_exprs:

            if ind not in self.prog_to_sigs:
                self.prog_to_sigs[ind] = []                

            self.prog_to_sigs[ind].append((se_sig, se_params))
            
            if se_sig not in self.sig_to_params:
                # params, prog_ind
                self.sig_to_params[se_sig] = ({}, [])

            for k,v in se_params.items():
                if k not in self.sig_to_params[se_sig][0]:
                    self.sig_to_params[se_sig][0][k] = []
                self.sig_to_params[se_sig][0][k].append(v)
                
            self.sig_to_params[se_sig][1].append(ind)

        
# Loads the proposal module
def load_prop_mode(args, L):    
    import proposal as PM
    prop_mode = PM.PropMode(args, L)
        
    if args.ab_prop_print_per is not None:
        PM.PRINT_PER = args.ab_prop_print_per
        
    return prop_mode

# Helper function that populates ProgData instance from D
def load_prog_data(D, L, args):
                
    count = 0

    prog_data = ProgData(args)

    for i,d in tqdm(list(enumerate(D))):

        if count == args.data_size:
            break
        
        wake_prog = d.inf_prog
        wake_vars = d.inf_vmap

        count += 1
        
        prog_data.add_prog(
            wake_prog,
            wake_vars,
            i,
            L,
            d.prims
        )

    if args.hold_out_infer == 'y':
        return prog_data
        
    if VERBOSE:
        print("Cluster Info")
        prog_data.print_clusters()
        
    return prog_data

# Logging helper function
def write_toi_info(key, fn, args, line, print_info='Time out'):
    perc_to = round(torch.tensor(TOI.pop(key)).float().mean().item(), 2)
    TOI[key] = []
    utils.log_print(f' {print_info} | {line} : {perc_to}', args, fn=fn)

# Run integration phase
def integration_phase(L, CPD, prog_data, args):
    # create integration phase data structure
    ID = IntegData(L, prog_data, args)

    # calc initial obj fn value
    init_obj_fn, ret_string = calc_obj_fn(
        L,
        ID,
        args
    )

    utils.log_print(
        f"Starting obj {init_obj_fn} \n {ret_string}", args
    )
    utils.log_print(
        f"Starting obj {init_obj_fn} \n {ret_string}", args, fn='a_log',do_print=False
    )

    write_toi_info('c', 'aerr_log', args, "Init")
    write_toi_info('sat_c', 'egg_sat_log', args, "Init", " Sat Rate ")

    best_obj_fn = init_obj_fn
    
    res = []

    # number of candidate proposals to try
    for i in range(args.ab_num_abs):

        # get best proposal
        score, sig, info = CPD.pop_top_prop(L)

        if score is None:
            utils.log_print("Breaking Early", args)
            break

        # create new library version        
        NL = make_new_library(L, sig, args)        

        NID = make_new_integ_data(ID, sig, L.lang)

        utils.log_print(
            f'Cand {i} as {NL.newest_abs} > {(info[0], info[1], info[2], info[3])}'
            f' : {sig}', args, fn='a_log'
        )

        # calculate new objective score under the library version
        new_obj_fn, ret_string, func_changes, func_freq = calc_obj_fn(
            NL, NID, args, ret_freq_info=True
        )        
        
        write_toi_info('c', 'aerr_log', args, f"Cand {i}")        
        write_toi_info('sat_c', 'egg_sat_log', args, f"Cand {i}", " Sat Rate ")
        
        func_freq_log = "Ab freqs > "
        func_freq_log += ' | '.join([f'{k}:{v}' for k,v in func_freq.items()])        
        
        ret_string += func_freq_log 

        # if we should accept the new library version
        if new_obj_fn < best_obj_fn:
            utils.log_print("!!!!", args, fn='a_log',do_print=False)
            utils.log_print("!!!!", args, fn='a_log',do_print=False)
            utils.log_print(f"Adding {NL.newest_abs}({i}) with improvement {best_obj_fn - new_obj_fn} ({best_obj_fn} -> {new_obj_fn})", args, fn='a_log',do_print=False)
            utils.log_print(ret_string, args, fn='a_log',do_print=False)
            utils.log_print("!!!!", args, fn='a_log',do_print=False)
            utils.log_print("!!!!", args, fn='a_log',do_print=False)
            utils.log_print(
                f"Added Cand {i} with score {score} and sig {sig}"
                f" into fn {NL.newest_abs} with improvement {best_obj_fn - new_obj_fn} ({best_obj_fn} -> {new_obj_fn})", args
            )
            utils.log_print(ret_string, args)
            
            res.append((i, score, sig, best_obj_fn - new_obj_fn))
            
            best_obj_fn = new_obj_fn

            del L
            L = NL
            del ID
            ID = NID

            cov_inds = []
            nabsn = L.newest_abs
            
            for ie in ID.data:
                if ie.abs_prog is not None:
                    if nabsn in ie.abs_prog:
                        cov_inds.append(ie.ind)
                                
            ID.update_abs_progs(L.lang)
            
            CPD.cover_weights(cov_inds, sig)            
            
        else:
            utils.log_print(f"Not directly adding {NL.newest_abs}({i}) with change {best_obj_fn - new_obj_fn} ({best_obj_fn} -> {new_obj_fn})", args, fn='a_log')

            # if no function frequencies changed, just go to next candidate function
            if len(func_changes) == 0:
                continue

            utils.log_print(f"Checking func changes {func_changes}", args, fn='a_log')
            
            # make a new library version where we remove the changed set of abstraction functions
            _ = NL.remove_abs(func_changes)
            NID = make_new_integ_data(ID, sig, L.lang)            
            NID.remove_abs_progs(NL)

            # recalculate objective function
            new_obj_fn, ret_string, _, func_freq = calc_obj_fn(
                NL, NID, args, ret_freq_info=True
            )

            func_freq_log = "Ab freqs > "
            func_freq_log += ' | '.join([f'{k}:{v}' for k,v in func_freq.items()])        
        
            ret_string += func_freq_log 

            # Once again, if we improve objective function accept new library
            if new_obj_fn < best_obj_fn:
                utils.log_print("!!!!", args, fn='a_log', do_print=False)
                utils.log_print("!!!!", args, fn='a_log', do_print=False)
                utils.log_print(f"CNG Adding {NL.newest_abs}(i), removing {func_changes}, with improvement {best_obj_fn - new_obj_fn} ({best_obj_fn} -> {new_obj_fn})", args, fn='a_log', do_print=False)
                utils.log_print(ret_string, args, fn='a_log', do_print=False)
                utils.log_print("!!!!", args, fn='a_log', do_print=False)
                utils.log_print("!!!!", args, fn='a_log', do_print=False)

                utils.log_print(
                    f"Added (in change of {func_changes}) Cand {i} with score {score} and sig {sig}"
                    f" into fn {NL.newest_abs} with improvement {best_obj_fn - new_obj_fn} ({best_obj_fn} -> {new_obj_fn})", args
                )
                utils.log_print(ret_string, args)
                
                res.append((i, score, sig, best_obj_fn - new_obj_fn))
                best_obj_fn = new_obj_fn
                del L
                L = NL
                del ID
                ID = NID

                cov_inds = []
                nabsn = L.newest_abs
                for ie in ID.data:
                    if ie.abs_prog is not None:
                        if nabsn in ie.abs_prog:
                            cov_inds.append(ie.ind)
                                
                ID.update_abs_progs(L.lang)                
                CPD.cover_weights(cov_inds, sig)
            else:
                utils.log_print(f"Not CNG adding {NL.newest_abs}({i}) with change {best_obj_fn - new_obj_fn} ({best_obj_fn} -> {new_obj_fn})", args, fn='a_log')
                utils.log_print(ret_string, args, fn='a_log')

    # After trying a set of candidate abstraction functions, try removing all abstractions in library one at a time
    utils.log_print("Try pruning abstraction", args, fn = 'a_log')
    for ab_name in list(L.abs_map.keys()):
        utils.log_print(
            f'Checking remove for {ab_name}', args, fn='a_log'
        )
        NL = deepcopy(L)
        
        _ = NL.remove_abs([ab_name])
        
        NID = make_new_integ_data(ID, sig, NL.lang)
        
        NID.remove_abs_progs(NL)
        
        new_obj_fn, ret_string, _, func_freq = calc_obj_fn(
            NL,
            NID,
            args,
            ret_freq_info=True
        )

        func_freq_log = "Ab freqs > "
        func_freq_log += ' | '.join([f'{k}:{v}' for k,v in func_freq.items()])        

        ret_string += func_freq_log

        if new_obj_fn < best_obj_fn:
            utils.log_print("!!!!", args, fn='a_log', do_print=False)
            utils.log_print("!!!!", args, fn='a_log', do_print=False)
            utils.log_print(f"Removing {ab_name} with improvement {best_obj_fn - new_obj_fn} ({best_obj_fn} -> {new_obj_fn})", args, fn='a_log', do_print=False)
            utils.log_print(ret_string, args, fn='a_log', do_print=False)
            utils.log_print("!!!!", args, fn='a_log', do_print=False)
            utils.log_print("!!!!", args, fn='a_log', do_print=False)

            utils.log_print(
                f"Removed {ab_name} with improvement {best_obj_fn - new_obj_fn} ({best_obj_fn} -> {new_obj_fn})", args
            )

            utils.log_print(ret_string, args)
                
            best_obj_fn = new_obj_fn
            del L
            L = NL
            del ID
            ID = NID
            
            ID.update_abs_progs(L.lang)                
        
    write_toi_info('t', 'aerr_log', args, f"Total")
    write_toi_info('sat_t', 'egg_sat_log', args, f"Total", " Sat Rate ")
    
    return L, ID, best_obj_fn, init_obj_fn
        
# top level logic for proposal and integration phases
def abstract(L, D, args):

    
    prop_mode = load_prop_mode(args, L)
    
    if args.ab_load_prop_path is None:

        # load data
        prog_data = load_prog_data(D, L, args)
            
        utils.log_print("In Abstract Proposal Phase", args)

        if args.hold_out_infer != 'y':
            # run proposal phase
            CPD = CandProposalData(args)                
            prop_mode.proposal_phase(prog_data, CPD)                    
            dill.dump(CPD, open(f"{args.outpath}/{args.exp_name}/round_{args.sc_round}/abstract/prop_data.pkl", "wb"))
            dill.dump(prog_data, open(f"{args.outpath}/{args.exp_name}/round_{args.sc_round}/abstract/prog_data.pkl", "wb"))

    else:
        # otherwise load this from saved files
        prog_data = dill.load(open(f'{args.ab_load_prop_path}/prog_data.pkl', 'rb'))

        prog_data.data = prog_data.data[:args.data_size]
        
        CPD = dill.load(open(f'{args.ab_load_prop_path}/prop_data.pkl', 'rb'))

        args.ab_load_prop_path = None
    
    if args.hold_out_infer == 'y':
        # if doing only hold out inference
        
        ID = IntegData(L, prog_data, args)

        D.integrate_abs_results(
            L, ID
        )

    else:
        
        CPD.print_top_info(args.ab_num_abs, L)

        if 'abs_register' not in L.__dict__:
            L.abs_register = {}
        
        for k,v in L.abs_map.items():
            if k not in L.abs_register:
                L.abs_register[deepcopy(k)] = deepcopy(v)
    
        utils.log_print("In Abstract Integration Phase", args)

        # Run integration phase
        L, ID, best_obj_fn, start_obj_fn = integration_phase(L, CPD, prog_data, args)        

        dill.dump(ID, open(f"{args.outpath}/{args.exp_name}/round_{args.sc_round}/abstract/integ_data.pkl", "wb"))    

        # Take outputs of integration phase, and update the data structure D with them
        D.integrate_abs_results(
            L, ID
        )        
    
        utils.log_print(
            "Finished with Abstraction phase"
            f", saw improvement of {start_obj_fn - best_obj_fn} ({start_obj_fn} -> {best_obj_fn})",
            args
        )

    w_imps = []
    w_prevs = []
    w_news = []

    # Calculate improvement we have observed during this phase
    
    for i,d in enumerate(D):
        prev_prog = d.inf_prog
        prev_score = d.inf_score
        d.make_inf_prog(L)
        new_prog = d.inf_prog
        new_score = d.inf_score

        do_print = i < 10
        
        utils.log_print(
            f'Ind {i} : Prev -> New \n'
            f'    {prev_prog} ({prev_score})\n    {new_prog} ({new_score})',
            args, fn = 'a_pres', do_print=do_print
        )

        w_imps.append(prev_score - new_score)
        w_prevs.append(prev_score)
        w_news.append(new_score)
        
    w_imps = torch.tensor(w_imps)
    perc_imp = round((w_imps > 0.).float().mean().item(),2)
    avg_imp = round(w_imps.mean().item(),2)

    utils.log_print(f"Abs res | Perc Imp :  {perc_imp} | Avg Imp : {avg_imp}", args)
    utils.log_print(
        f'Abs Obj Change |'
        f' {round(torch.tensor(w_prevs).float().mean().item(),2)} '
        f'-> {round(torch.tensor(w_news).float().mean().item(),2)}',
        args
    )

    utils.log_print(
        f'Abs Rnd {args.sc_round} : {round(torch.tensor(w_news).float().mean().item(),2)}',
        args, fn='obj_log'
    )

    return L, D
