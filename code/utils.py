import torch
import numpy
import random
import sys
import matplotlib.pyplot as plt
import os
import argparse
import abstract as ab
import dill
import prog_utils as pu

# device for rec net training
device = torch.device('cuda')

# fields to log while training rec net
TRAIN_LOG_INFO = [
    ('Loss', 'loss', 'batch_count'),
    ('Token Loss', 'token_loss', 'batch_count'),
    ('Var Loss', 'var_loss', 'batch_count'),
    ('Pos Var Loss', 'pos_var_loss', 'batch_count'),
    ('Neg Var Loss', 'neg_var_loss', 'batch_count'),
    ('Token Acc', 'token_corr', 'token_total'),
    ('Var Acc', 'var_corr', 'var_total'),
    ('Fn Acc', 'fn_corr', 'fn_total'),
]

# fields to log during evaluation for rec net
EVAL_LOG_INFO = [    
    ('Prog Match Perc', 'prog_match', 'batch_count'),
    ('Shape Match Perc', 'shape_match', 'batch_count'),    
]

# Helper function to print results, based on given fields
def print_results(
    LOG_INFO,
    result,
    args        
):
    res = ""
    for info in LOG_INFO:
        if len(info) == 3:
            name, key, norm_key = info
            if key not in result:
                continue
            _res = result[key] / (result[norm_key]+1e-8)
                
        elif len(info) == 5:
            name, key1, norm_key1, key2, norm_key2 = info
            if key1 not in result or key2 not in result:
                continue
            res1 = result[key1] / (result[norm_key1]+1e-8)
            res2 = result[key2] / (result[norm_key2]+1e-8)
            _res = (res1 + res2) / 2
                
        else:
            assert False, f'bad log info {info}'
                                     
        res += f"    {name} : {round(_res, 4)}\n"

    log_print(res, args, fn='w_log')

# Helper function to initalize run
def init_run(args, set_rd, folders):

    if set_rd:
        
        random.seed(args.rd_seed)
        numpy.random.seed(args.rd_seed)
        torch.manual_seed(args.rd_seed)

        os.system(f'mkdir {args.outpath} > /dev/null 2>&1')    
        os.system(f'mkdir {args.outpath}/{args.exp_name} > /dev/null 2>&1')

        CMD_STRING = f'CMD: {" ".join(sys.argv)}\n'
        ARG_STRING = f"ARGS: {args}\n"
        
        with open(f'{args.outpath}/{args.exp_name}/config.txt', "w") as f:
            f.write(CMD_STRING)        
            f.write(ARG_STRING)

        log_print("$$$ NEW RUN $$", args, do_print=False)
        log_print(CMD_STRING, args, do_print=False)
        log_print(ARG_STRING, args, do_print=False)

    for fld in folders:        
        os.system(f'mkdir {args.outpath}/{args.exp_name}/{fld} > /dev/null 2>&1')    

# Helper function to make plots, given fields
def make_plots(
    LOG_INFO,
    results,
    plots,
    epochs,
    args,
    dir_name,
    do_eval    
):
    for info in LOG_INFO:
        
        for rname, result in results.items():
            if len(info) == 3:
                name, key, norm_key = info
                if key not in result:
                    continue
                res = result[key] / (result[norm_key]+1e-8)
                
            elif len(info) == 5:
                name, key1, norm_key1, key2, norm_key2 = info
                if key1 not in result or key2 not in result:
                    continue
                res1 = result[key1] / (result[norm_key1]+1e-8)
                res2 = result[key2] / (result[norm_key2]+1e-8)
                res = (res1 + res2) / 2
                
            else:
                assert False, f'bad log info {info}'
                        
            if name not in plots[rname]:
                plots[rname][name] = [res]
            else:
                plots[rname][name].append(res)

        if not do_eval:
            continue

        plt.clf()
        some=False
        for key in plots:
            if name not in plots[key]:
                continue
            some = True
            plt.plot(
                epochs,
                plots[key][name],
                label= key
            )

        if not some:
            continue
            
        plt.legend()
        plt.grid()
        plt.savefig(f'{args.save_net_path}/{dir_name}/{name}.png')

# helper print function, that writes to stdout and to log file, optionally for both
def log_print(s, args, fn='log',do_print=True, do_log=True):

    if do_log:
        of = f"{args.outpath}/{args.exp_name}/{fn}.txt"
        with open(of, 'a') as f:
            f.write(f"{s}\n")
        
    if do_print:
        print(s)

# helper function to get cmd line arguments from arg list
def getArgs(arg_list):       

    parser = argparse.ArgumentParser()
    
    for s,l,d,t in arg_list:        
        parser.add_argument(s, l, default=d, type = t)

    args, _ = parser.parse_known_args()
    
    return args


# give primitives an order, based on position
def order_prims(prims, return_ord=False):

    def get_val(P):
        if len(P) == 4:
            _,_,x,y = P
            z = 0
        elif len(P) == 9:
            _,_,_,x,y,z,_,_,_ = P

        nx = round(x * 20) / 20.
        ny = round(y * 20) / 20.
        nz = round(z * 20) / 20.
        
        return (nx + ny + nz) * 10000 + (nx) * 1000 + ny * 100 + nz * 10

    N = [(get_val(P), P, i) for i,P in enumerate(prims)]
        
    N.sort()

    if return_ord:
        return [P for _,P,_ in N], [i for _,_,i in N]
    
    return [P for _, P, _ in N]

# Dummy argument class
class DARGS:
    def __init__(self):
        self.ab_rw_max_error = 0.0

# parse abstraction from text file
def parse_abs(sc_load_path):
    with open(f'{sc_load_path}/end_lib.txt') as f:
        for line in f:
            name, sig = line.split(':')
            yield name.split('(')[0].strip(), sig.strip()

# load library from file
def load_lib(
    lib_load_path, L
):

    dargs = DARGS()
    
    raw_lib_load_path = '/'.join(lib_load_path.split('/')[:-2])
    added = set()
    # look for previous rounds, in case some abstractions were removed
    # add all abstractions
    for i in range(5):
        try:
            for name, sig in parse_abs(f'{raw_lib_load_path}/round_{i}/'):
                if name not in added:
                    abst = ab.Abstraction(
                        sig,
                        L,
                        dargs
                    )
                    abst.name = name
                    abst.make_token_and_ex_abs(L)
                    L.abs_register[name] = abst
                    L.abs_map[name] = abst
                    L.reset()
                    added.add(name)
        except:
            pass

    # figure out which abstractions to keep
    keep = set()
    for name, sig in parse_abs(lib_load_path):
        abst = ab.Abstraction(
            sig,
            L,
            dargs
        )
        abst.name = name
        abst.make_token_and_ex_abs(L)
        L.abs_map[name] = abst
        L.abs_register[name] = abst
        L.reset()
        keep.add(name)

    # remove all abstractions that shouldn't be kept
    for name in added:
        if name not in keep:
            L.abs_map.pop(name)

    L.reset()

class DummyVisData:
    def __init__(self, prog):
        self.inf_prog = prog
        self.inf_vmap = {}
        self.prims = []

def dummy_union(L, sub_progs):
    prog = L.lang.comb_progs(sub_progs)
    dvd = DummyVisData(prog)
    return dvd
    
def vis_load_lib_and_data(sc_load_path, L):
    load_lib(sc_load_path, L)

    D = []
    cur = []
    with open(f'{sc_load_path}/end_data.txt') as f:
        for line in f:
            if 'Data' in line:
                if len(cur) > 0:
                    D.append(dummy_union(L, cur))
                cur = []
                continue
            else:
                cur.append(line.strip())

    if len(cur) > 0:
        D.append(dummy_union(L, cur))

    return D
        
# load library and data from file 
def load_lib_and_data(sc_load_path, _L, canon_fn=None):

    L = load_lib(sc_load_path, _L)
    
    D = dill.load(open(f'{sc_load_path}/data.pkl', 'rb'))

    progs = []
    
    for d in list(D):
        text = d.inf_prog

        for k,v in d.inf_vmap.items():
            text = text.replace(str(k), str(v))

        if canon_fn is not None:
            sub_progs = pu.split_by_union(text)            
            sub_progs = canon_fn(sub_progs)
            text = L.lang.comb_progs(sub_progs)
        progs.append(text)
        
    return progs

# Save model
def save_model(d, p):
    try:
        torch.save(d,p,_use_new_zipfile_serialization=False)
    except:
        torch.save(d,p)
