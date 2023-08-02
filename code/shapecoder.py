import torch
import random
import utils
import sys
from dream import make_dreams

from rec_net import RecNet, DummyDistNetwork
import mc_prog_search as mps
import data
import numpy as np
import os
import time
from tqdm import tqdm

import abstract

from copy import deepcopy
import dill
import prog_utils as pu
import library as lib

# Load args
def load_args():
        
    arg_list = def_arg_list()
    args = utils.getArgs(arg_list)

    # for each domain:
    # get domain specifics args,
    # program loading logic (grammar)
    # executor
    
    if args.domain == 's2d':

        arg_list += s2d_arg_list()
        args = utils.getArgs(arg_list)
        
        sys.path.append('../s2d_lang')
        # 2D has toy grammar
        import simp_chair_grammar as scg
        import executor as ex
        args.grammar = scg
        args.executor = ex
        
    elif args.domain == 's3d':
        arg_list += s3d_arg_list()
        args = utils.getArgs(arg_list)
        
        sys.path.append('../s3d_lang')

        # data path has 3D shapes
        assert args.data_path is not None, 'please select data path'        
        import partnet_grammar as pg
        import executor as ex            
        pg.load_progs(
            args.data_path
        )            
        args.grammar = pg        
        args.executor = ex
        
    else:
        assert False, f'Bad domain {args.domain}'
        
    
    return args



def s3d_arg_list():
    return [

        ('-dp', '--data_path', None, str), # path to 3d shapes (in txt file)
        ('-dm_msf', '--dm_max_sub_fns', 4, int), # sub expressions to combine in dream
        ('-dm_mdp', '--dm_max_distract_prims', 4, int), # max distractor prims for dream
        
        ('-rn_pd', '--rn_prim_dim', 9, int), # primitive dim rep (pos - 3, scale - 3, rot_angs - 3)
        ('-rn_mp', '--rn_max_prims', 16, int), # max prims to give as input to rec net as once
        
        ('-of_me', '--of_max_error', 0.1, float), # maximum acceptable error
        ('-ab_rme', '--ab_rw_max_error', 0.055, float), # max error allowable during abstractio nrwrite
        ('-ab_cme', '--ab_cand_max_error', 0.05, float),  # max error allowable during candidate proposal
    ]

# see s3d_arg_list for explanation
def s2d_arg_list():
    return [
        
        ('-dm_msf', '--dm_max_sub_fns', 3, int),                
        ('-dm_mdp', '--dm_max_distract_prims', 3, int),
        
        ('-rn_pd', '--rn_prim_dim', 4, int),
        ('-rn_mp', '--rn_max_prims', 12, int),
        
        ('-of_me', '--of_max_error', 0.05, float),
        ('-ab_rme', '--ab_rw_max_error', 0.04, float),
        ('-ab_cme', '--ab_cand_max_error', 0.03, float), 
    ]


def def_arg_list():
    
    arg_list = [
        ('-en', '--exp_name', None, str), # name of experiment 

        ('-ds', '--data_size', None, int), # number of shapes to run shapecoder over
        ('-dn', '--domain', None, str), # what domain (options s2d, or s3d) 

        ('-nd', '--num_dreams', 10000, int), # number of dreams, for each function

        ('-slp', '--sc_load_path', None, str), # load path of previous run
        ('-hoi', '--hold_out_infer', 'n', str), # whether we are doing regular run ('n') or hold out inference ('y')
        
        ('-ssr', '--start_sc_round', 0, int), # what round we starting on
        ('-msr', '--max_sc_rounds', 4, int), # max number of shapecoder rounds

        ('-out', '--outpath', 'abs_output', str), # output directory
        ('-rd', '--rd_seed', 42, int), # random seed
        ('-pr', '--prec', 2, int), # precision digits for real-values

        ('-ab_cd', '--ab_cov_discount', 0.5, float), # how much to discount score of candidate abstraction, when added abstraction covers their programs
        ('-ab_smd', '--ab_struct_match_discount', 1.0, float), # how much to discount score of candidate abstraction, when added abstraction has same structural sig
                
        ('-dm_t', '--dm_timeout', 1800, int), # max time to dream for each function
        ('-dm_mc', '--dm_unalign_chance', 0.5, float), # chance to unalign each dream
        ('-dm_dpc', '--dm_distract_prim_chance', 0.5, float), # chance to add distractor prims to each prim
        ('-rn_ms', '--rn_max_seq', 32, int), # max length of rec net token prediction sequence
        
        ('-dm_ld', '--dm_load_dreams', 'n', str), # can be used to load previously made dreams        
        ('-dm_lp', '--dm_load_path', None, str), # if loading dreams, load them from this path

        ('-dm_dp', '--dm_dream_params', None, str), # can be used to overwrite dreaming related validation parameters (see library.py)
        
        ('-rn_mo', '--rn_mode', 'nocontext', str), # architecture choice of recognition network, nocontext is best performing

        ('-rn_nl', '--rn_num_layers', 2, int), # number of transformer layers
        ('-rn_nh', '--rn_num_heads', 8, int), # number of transformer heads
        ('-rn_bs', '--rn_batch_size', 64, int), # batch size
        ('-rn_dp', '--rn_dropout', 0.5, float), # dropout
        ('-rn_hd', '--rn_hidden_dim', 128, int), # hidden dimension
        ('-rn_vd', '--rn_var_dim', 16, int), # dimension of variable related spaces

        ('-rn_ho', '--rn_holdout', 0.1, float), # how much data to holdout for val set
        ('-rn_lr', '--rn_lr',0.0001, float), # learning rate
        ('-rn_ep', '--rn_epochs', 300, int), # number of epochs to train rec net for
        
        ('-rn_prp', '--rn_print_per', 5, int), # how many epochs in between printing train information
        ('-rn_evp', '--rn_eval_per', 25, int), # how many epochs between running evaluation 
        ('-rn_nte', '--rn_num_to_eval', 50, int), # number of shapes to eval each period
        ('-rn_esp', '--rn_patience', 30, int), # patience, used to determine early stopping
        ('-rn_est', '--rn_threshold', 0.025, float), # early stopping threshold, loss must decrease by atleast this amount every patience
        ('-rn_to', '--rn_timeout', 12000, int), # max amount of time to spend training rec net, each round

        ('-rn_mn', '--rn_margin', 1.0, float), # used by alternative architectures

        ('-wk_pst', '--wk_prog_search_timeout', 1, float), # max time to spend searching for each expression during wake
        ('-wk_lp', '--wk_load_path', None, str), # can load results of wake phase
        
        # Abstract

        ('-ab_nao', '--ab_num_all_ord', 2, int), # max number of structural sub-expressions to consider combining during proposal phase
        ('-abc_msp', '--abc_min_sig_perc', 0.05, float), # minimum frequency needed to propose candidate abstractions for a given cluster
        ('-ab_mcmp', '--ab_min_cluster_match_perc', 0.75, float), # to avoid adding new categorical var during proposal, must explain this percent of cluster


        ('-ab_mup', '--ab_min_use_perc', 0.01, float), # each abstraction should be used at a higher rate than this
        ('-ab_micn', '--ab_min_cluster_num', 8, int), # each cluster should be at minimum this size
        ('-ab_macn', '--ab_max_cluster_num', 16, int), # each cluster should be at maximum this size
        ('-ab_mcmn', '--ab_min_cluster_match_num', 4, int), # each candidate abstraction should explain this number of shapes in each cluster

        
        ('-ab_mcep', '--ab_max_cexpr_params', 12, int),  # each cluster can have at most this amount of free paramaters
        
        ('-ab_rmt', '--ab_rw_max_terms', 3, int), # most amount of terms that can be in any one parametric expression        
        ('-ab_na', '--ab_num_abs', 20, int), # number of abstractions to consider adding during integration phase
        
        ('-ab_lpp', '--ab_load_prop_path', None, str), # load proposal data from this path
                        
        ('-ab_npr', '--ab_num_prop_rounds', 10000, int), # number of candidate abstractions to propose
                 
        ('-ab_p3', '--ab_prop_print_per', 500, int), # how often to print information while proposing candidate abstractions

        # Egg related parameters for refactoring        
        ('-eg_mr', '--egg_max_rounds', 5, int), # max number of rounds of applying rewriters
        ('-eg_mn', '--egg_max_nodes' , 10000, int), # max nodes allowable in e-graph
        ('-eg_mt', '--egg_max_time', 5, int), # max time to run refactor for, for each call
        
        # Obj Function weights        
        ('-of_fnw', '--of_fn_weight', 1.0, float), # program cost of using default function
        ('-of_ffw', '--of_float_fn_weight', 0.1, float), # program cost of using parametric function
        ('-of_fvw', '--of_float_var_weight', 2.0, float), # program cost of using a free float
        ('-of_fw', '--of_float_weight', 2.0, float), # program cost of using a float        
        ('-of_fcw', '--of_float_const_weight', 0.5, float), # program cost of using a constant float        
        ('-of_few', '--of_float_error_weight', 10., float), # error cost for any non perfect fit (dependt on fit metric)
        ('-of_cw', '--of_cat_weight', 0.5, float), # program cost of using categorical variable
        
        ('-of_lw', '--of_lib_weight', .25, float), # Base cost of adding abstraction into library
        
        ('-of_pt', '--of_param_terms', 'float_fn,float_var,float_const,float_error,cat', str), # parametric cost terms        

        ('-of_st', '--of_struct_terms', 'fn', str), # structural cost terms
    ] 

    return arg_list


# Dream phase, trains recognition network by sampling dreams from library
def dream_phase(L, D,  args):            
    
    utils.log_print("In Dream Phase", args)
    
    pth = f'{args.outpath}/{args.exp_name}/round_{args.sc_round}/dream/'

    dream_data = None

    # If we should load premade dreams, do it
    if args.dm_load_dreams == 'y':
        dream_data = dill.load(open(pth+"dream_data.pkl", "rb"))

    args.dm_load_dreams = None
    # reset after first round
    
    if dream_data is None:
        with torch.no_grad():            

            # make a dummy network to sample dreams
            N_dummy = DummyDistNetwork(
                L,
                args
            )

            # Sample dreams
            dream_data = make_dreams(
                L, N_dummy, D, args
            )
            
        dill.dump(dream_data, open(pth+"dream_data.pkl", "wb"))

    args.save_net_path = f'{args.outpath}/{args.exp_name}/round_{args.sc_round}/wake'

    # Create recogntion netwrok
    N = RecNet(L, args)    

    # train recognition network on dreams
    data_loader = N.make_dataset(dream_data)    
    N.train(data_loader)

    # load best epoch
    N.load_best()    
    
    return N

# Load a previously trained model
def load_dream_phase(L, D, args):            

    utils.log_print("DEBUG: In Load Dream Phase", args)
    
    N = RecNet(L, args)    
    
    N.load_best(args.dm_load_path)
    
    return N

# Wake phase, uses recognition network to infer programs over shapes in dataset
def wake_phase(L, D, N, args):
    
    utils.log_print("In Wake Phase", args)

    w_imps = []
    w_prevs = []
    w_news = []
    # for shapes in data
    for i,d in tqdm(list(enumerate(D.data[:args.data_size]))):
                
        LT = deepcopy(L)    
        LT.add_prims(d.prims)

        # run search over primitives to find a set of explaining expressions
        inf_prog_infos, _ = mps.mc_prog_search(
            LT, N, d, args
        )    

        # Add inferred expressions into data
        
        if d.inf_prog is not None:
            prev_prog = d.inf_prog
            prev_score = d.inf_score
        else:
            prev_prog = 'START'
            nsc, npc = L.lang.get_naive_costs([_prims for _, _prims in d.prims], args)
            prev_score = nsc + npc
                    
        for ip in inf_prog_infos:
            d.prog_info.add(LT, ip.text, ip.match_prims)
        
        # Using inferred information (this round, and previous rounds) make a best program
        d.make_inf_prog(LT)
                
        new_prog = d.inf_prog
        new_score = d.inf_score
                        
        # print info, record improvements over previous best program
        
        do_print = i < 10
        utils.log_print(
            f'Ind {i} : Prev -> New \n'
            f'    {prev_prog} ({prev_score})\n    {new_prog} ({new_score})',
            args, fn = 'w_pres', do_print=do_print
        )

        wake_lines = new_prog
        for k,v in d.inf_vmap.items():
            wake_lines = wake_lines.replace(k, str(v))

        utils.log_print(f'Ind_{i} : {wake_lines}', args, fn = f'round_{args.sc_round}/wake_progs', do_print=False)
        
        w_prevs.append(prev_score)
        w_news.append(new_score)
        w_imps.append(prev_score - new_score)

    w_imps = torch.tensor(w_imps)
    perc_imp = round((w_imps > 0.).float().mean().item(),2)
    avg_imp = round(w_imps.mean().item(),2)

    # Print and log information from dream and wake
    
    utils.log_print(f"Wake res | Perc Imp :  {perc_imp} | Avg Imp : {avg_imp}", args)
    utils.log_print(
        f'Wake Obj Change |'
        f' {round(torch.tensor(w_prevs).float().mean().item(),2)} '
        f'-> {round(torch.tensor(w_news).float().mean().item(),2)}',
        args
    )

    utils.log_print(
        f'Dream Rnd {args.sc_round} : {round(torch.tensor(w_prevs).float().mean().item(),2)}\n'
        f'Wake Rnd {args.sc_round} : {round(torch.tensor(w_news).float().mean().item(),2)}',
        args, fn='obj_log'
    )        
    
    fn = f'{args.outpath}/{args.exp_name}/round_{args.sc_round}/wake/data.pkl'

    D.save(fn)
    
    return D

# folders to be created each round
def get_rnd_folders(RND):
    return [
        f'round_{RND}',
        f'round_{RND}/dream/',
        f'round_{RND}/wake/',
        f'round_{RND}/wake/train_plots/',
        f'round_{RND}/wake/eval_plots/',
        f'round_{RND}/abstract/',                
    ]

# init shapecoder run
def init_sc_run(L, args):
    args.of_weights = {
        'fn': args.of_fn_weight,
        
        'float_fn': args.of_float_fn_weight,
        'float_var': args.of_float_var_weight,
        'float': args.of_float_weight,
        'float_const': args.of_float_const_weight,
        'float_error': args.of_float_error_weight,

        'cat': args.of_cat_weight,
        'lib': args.of_lib_weight,
    }
    args.of_struct_terms = args.of_struct_terms.split(',')
    args.of_param_terms = args.of_param_terms.split(',')
    
    L.lang.make_dream_params(args)

# record results at end of each round
def write_rnd_res(L, D, args):
    data_file = f'round_{args.sc_round}/end_data'
    lib_file = f'round_{args.sc_round}/end_lib'

    freq_info = {ab_name:0 for ab_name in L.abs_map.keys()}

    for i, d in enumerate(D):
        
        prog = d.inf_prog

        for ab_name in L.abs_map.keys():
            if f'{ab_name}(' in prog:
                freq_info[ab_name] += 1
        
        for k,v in d.inf_vmap.items():
            prog = prog.replace(k, str(v))

        sub_progs = pu.split_by_union(prog)
        utils.log_print(
            f"Data {i}:",
            args, fn=data_file, do_print=False
        )
        for sp in sub_progs:
            utils.log_print(
                f'    {sp}',
                args, fn=data_file, do_print=False
            )        
            
    for ab in L.abs_map.values():
        utils.log_print(
            f"{ab.name}({freq_info[ab.name]}) : {ab.exp_fn(ab.param_names)}",
            args, fn=lib_file, do_print=False
        )

# ShapeCoder algorithm
def shapecoder(L, D, args):

    # initialize
    init_sc_run(L, args)    

    # for number of rounds (+1, for hoi round)
    for RND in range(args.start_sc_round, args.max_sc_rounds + 1):

        # last round we do hold out inference pass, to not add any more abstractions
        if RND == args.max_sc_rounds:
            utils.log_print(f"Entering last round {RND}, doing hold out inference pass", args)

            args.hold_out_infer = 'y'

        # logging
        for lfn in ['a_log', 'aerr_log', 'a_pres', 'w_log', 'w_pres']:                
            utils.log_print(
                f'~~~~ Round {RND}', args, fn=lfn, do_print=False
            )
        
        t = time.time()
        
        # Make sure to reset library
        L.reset()        
        
        utils.log_print(f"Shapecoder Round {RND} (/{args.max_sc_rounds})", args)
        
        utils.init_run(args, False, get_rnd_folders(RND))

        args.sc_round = RND

        # Dream phase
        if args.dm_load_path is not None:
            N = load_dream_phase(L, D, args)
        else:
            N = dream_phase(L, D, args)

        utils.log_print(f"Time Dream {RND} : {round(time.time()-t)}", args, fn='t_log')
        t = time.time()
        
        # reset load path after first round
        args.dm_load_path = None

        utils.log_print(f"New Progs Wake {RND}", args, fn='gerr_log')

        # Wake phase
        if args.wk_load_path is not None:
            D = dill.load(open(f'{args.wk_load_path}', 'rb'))
            D.data = D.data[:args.data_size]
        else:
            D = wake_phase(L, D, N, args)
                
        # reset wk_load_path after first round
        args.wk_load_path = None

        utils.log_print(f"Time Wake {RND} : {round(time.time()-t)}", args, fn='t_log')
        t = time.time()
        
        utils.log_print(f"New Progs Abstract {RND}", args, fn='gerr_log')

        # Abstraction has proposal and integration phases
        L, D = abstract.abstract(L, D, args)        

        utils.log_print(f"Time Abstract {RND} : {round(time.time()-t)}", args, fn='t_log')
        t = time.time()
        
        dill.dump(L, open(f'{args.outpath}/{args.exp_name}/round_{RND}/lib.pkl', 'wb'))
        D.save(f'{args.outpath}/{args.exp_name}/round_{RND}/data.pkl')

        # Record information
        write_rnd_res(L, D, args)

        if args.hold_out_infer == 'y':
            # break out if hoi
            utils.log_print("Hold out infer mode, so stopping", args)
            return 
        
    return 

# Main entrypoint into shapecoder        
def main():

    args = load_args()
    
    utils.init_run(args, True, ['shared'])

    # If we are starting out doing hold out inference
    if args.hold_out_infer == 'y':
        utils.log_print("Hold out Inference", args)

        assert  args.sc_load_path is not None
        assert args.dm_load_path is not None
        
        args.ab_num_abs = 0
        args.ab_num_prop_rounds = 0
        args.max_sc_rounds = 0

        # Load past library, data
        try:
            L = dill.load(open(f'{args.sc_load_path}/lib.pkl', 'rb'))
            D = data.getPrimitiveDataset(L.lang, args)    
        except:
            L = lib.makeLibrary(
                args.executor, args.domain,
            )
            _ = utils.load_lib(
                args.sc_load_path,
                L,
            )
            D = data.getPrimitiveDataset(L.lang, args)
            
    # load starting lib and data if provided
    elif args.sc_load_path is not None:
        print("Loading from previous experiment")
        L = dill.load(open(f'{args.sc_load_path}/lib.pkl', 'rb'))        
        D = dill.load(open(f'{args.sc_load_path}/data.pkl', 'rb'))

    # start previous run from a round above 0
    elif args.start_sc_round > 0:
        print("Loading from previous round")
        L = dill.load(
            open(f'{args.outpath}/{args.exp_name}/round_{args.start_sc_round - 1}/lib.pkl', 'rb')
        )
        D = dill.load(
            open(f'{args.outpath}/{args.exp_name}/round_{args.start_sc_round - 1}/data.pkl', 'rb')
        )

    # start an entirely new run
    else:
        
        L = lib.makeLibrary(
            args.executor, args.domain,
        )
        D = data.getPrimitiveDataset(L.lang, args)    
    
    shapecoder(L, D, args)
        
        
if __name__ == '__main__':
    main()
