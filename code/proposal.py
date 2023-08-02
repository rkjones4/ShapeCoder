import numpy as np
import random
import torch
import sys
import prog_utils as pu
from tqdm import tqdm
from copy import deepcopy
import time

import prm_expr_search as pes

PRINT_PER = 100
PRINT_NUM = 10

MAX_CAND_PROP_TIME = 1

# From observed programs, and their structures/parameterizations
# find all valid structures we should search over for proposed candidate abstractions
def all_ord_get_clusters(prop_data, L, args):
    
    cluster_info = []
    cluster_dist = []

    total = len(prop_data.prog_to_sigs.keys())    

    for k,v in prop_data.sig_to_params.items():

        num_in_sig = len(list(v[0].values())[0])

        sig = k

        res = pu.parse_ab_info(sig, L)

        uc = sig.count('Union')

        freq = min(num_in_sig / total, 1.0)

        # Minimum frequency we need to consider structure
        if freq < args.abc_min_sig_perc:
            continue

        # Too many free parameters
        if res['num_params'] > args.ab_max_cexpr_params:
            continue
        
        # freq * num sub programs is used to determine how often we sample this cluster
        score = freq * (uc + 1)
        min_num = min(args.ab_min_cluster_num, num_in_sig)
        max_num = min(args.ab_max_cluster_num, num_in_sig)
        
        cluster_info.append((sig, min_num, max_num))
        cluster_dist.append(score)

    print(f"Num Valid Clusters {len(cluster_info)}")
    
    cluster_dist = np.array(cluster_dist)
    cluster_dist = cluster_dist / cluster_dist.sum()

    return cluster_info, cluster_dist
    
    
# Helper function
def avg(L):
    return sum(L) / (1.0 * len(L))

# Checks if tvalues can be explained by previously instantiated prameters in sparams
# e.g. if we should add a parametric expression to the candidate abstraction 
def sc_has_fexpr_exp(S, tvalues, sparams):
        
    lang = S.L.lang
    cme = S.args.ab_cand_max_error
    mcmp = (S.args.ab_min_cluster_match_num - 0.1)  / len(tvalues) 

    pvars = {
        k:torch.tensor(sparams[k])
        for k in sparams.keys() if '_F_' in k
    }
    
    singleton_consts = lang.get_valid_singleton_consts()

    expr_consts = {
        f'{f}': torch.tensor(f) for f in lang.expr_float_consts
    }

    # different float operations to consider
    tp_ops = {k:v for k,v in lang.float_fn_map.items() if not v.oneparam}
    op_ops = {k:v for k,v in lang.float_fn_map.items() if v.oneparam}
    
    inp = {
        'pvars': pvars,
        'st_consts': singleton_consts,
        'ex_consts': expr_consts,
        'targets': torch.tensor(tvalues),
        'tp_ops': tp_ops,
        'op_ops': op_ops,
        'cme': cme,
        'mcmp': mcmp,
    }

    # L has order of parametric expressions we check
    # TODO CHECK
    
    L = []
    if len(inp['pvars']) > 0:
        L += [
            pes.sc_hfe_pvar, # use a prev var
        ]
        
    L += [        
        pes.sc_hfe_sconst, # use a singleton const
    ]
    
    if len(inp['pvars']) > 0:
        L += [
            pes.sc_hfe_oneprm_op, # one prm input operators
            pes.sc_hfe_one_op, # one regular operator
            pes.sc_hfe_two_op, # two operators
            pes.sc_hfe_three_op, # three operators
        ]

    res = {}

    covered = set()
    count = 0
    
    for fit_fn in L:
        # iterate through each batch of parametric expressions to check
        
        count += 1
        # see if any of the expressions found a match
        ffom = fit_fn(inp, ret_inds=True)

        # Record which expressions match to what target indices
        matches = []
        for e,minds in ffom:
            if pes.sc_is_valid_expr(lang, e):
                matches.append((e, minds))

        # filter matches
        fmatches = pes.filter_matches_winds(matches, sparams.keys())

        if len(fmatches) == 0:
            continue
                
        for m, inds in fmatches:
            inds.sort()
            k = tuple(inds)
            if k in res:
                continue
            res[k] = m.replace('(',' ( ').replace(')',' ) ').replace(',', ' , ').split()
            
            covered.update(inds)

        # once all target values have been covered by some expression, break out of this loop (for speed)
        if len(covered) == len(tvalues):
            break
        
    return [(k, m) for k,m in res.items()]


        
class PropMode:
    def __init__(self, args, L):
        self.args = args
        self.L = L

    # Create a distance array, from a seed member of cluster to rest of members
    # of that cluster, used to sample related program instances 
    def make_dist_array(self, mt_params, params, mt_inds):

        opt_inds = []
        
        keys = list(params.keys())
        D = torch.zeros(len(mt_inds), len(keys))
        
        def dist(k, a, b):
            if '_F_' in k:
                return abs(a-b)
            else:
                return float(a != b)

            
        for i, mti in enumerate(mt_inds):

            opt_inds.append(i)
            
            for j,k in enumerate(keys):
            
                D[i, j] = dist(k, params[k], mt_params[k][i])
                
        D = D.norm(dim=1)

        return D, opt_inds
        

    # Sample a cluster, for a given structural signature group
    def sample_based_on_sig(
        self, sig_to_params, sig, min_num, max_num
    ):

        # first sample all elements of one structural group
        mt_params, mt_inds = sig_to_params[sig]

        # pick a random index, as the seed member of the cluster
        ri = random.choice(list(range(len(mt_inds))))        
        params = {k: mt_params[k][ri] for k in mt_params}

        # create distance array based on seed member values
        D, opt_inds = self.make_dist_array(mt_params, params, mt_inds)

        # turn distance into probability distribution, where closer values
        # get higher probabilities
        D = torch.clamp(D, 0.1, 10.)        
        P = torch.softmax(1./(D+1e-8), dim =0)        

        # sample random cluster size, within allowable limits
        cluster_size = random.randint(
            min_num,
            max_num,
        )

        # Sample a cluster, based on the probability distribution
        samp_inds = np.random.choice(opt_inds, cluster_size, replace=False, p=P.numpy())                
        cl_params = {k: [V[ind] for ind in samp_inds] for k,V in mt_params.items()}
        cl_inds = [mt_inds[ind] for ind in samp_inds]
        
        return sig, cl_params, cl_inds


    # Checks whether there is an expression to explain tvalues, which is
    # a categorical type, based on previous sparam categorical values
    def has_cat_expr(self, tvalues, sparams, cinds):

        btoken = None
        bmatch = 0.0
        binds = None
        
        o_vals = set(tvalues)

        for ov in o_vals:
            match = avg([float(t == ov) for t in tvalues])
            if match > bmatch:
                bmatch = match
                btoken = ov
                binds = [ci for ci,t in zip(cinds, tvalues) if t == ov]
                
        for k, V in sparams.items():
            if '_C_' in k:

                match = avg([float(t == s) for t,s in zip(tvalues, V)])
                
                if match > bmatch:
                    bmatch = match
                    btoken = k
                    binds = [ci for ci,t,s in zip(cinds,tvalues,V) if t == s]

        if bmatch > self.args.ab_min_cluster_match_perc:
            return btoken, binds

        else:        
            return None, None

    # make float tokens, that may be added into the candidate abstraction 
    def make_float_token(self, token, sparams, cparams, cinds):

        tvalues = [cparams[token][ci] for ci in cinds]                

        # Check what expression to use
        fexpr_info = sc_has_fexpr_exp(self, tvalues, sparams)
        
        d_sp = deepcopy(sparams)
        d_nt = f'V_F_{len(d_sp)}_'
        d_sp[d_nt] = tvalues

        n_info = [
            ([d_nt], cinds, d_sp)
        ]

        if len(fexpr_info) == 0:
            # default, when no expression is found
            return n_info

        # return possible float expressions
        for minds, fexpr in fexpr_info:
            
            n_inds = [cinds[mi] for mi in minds]
            
            n_info.append((
                fexpr,
                n_inds,
                {k:[V[mi] for mi in minds] for k,V in sparams.items()}
            ))
            
        return n_info
        
    # make cat tokens, that may be added into the candidate abstraction         
    def make_cat_token(self, token, sparams, cparams, cinds):

        tvalues = [cparams[token][ci] for ci in cinds]
        
        cat_expr, n_inds = self.has_cat_expr(tvalues, sparams, cinds)

        d_sp = deepcopy(sparams)
        CT = self.L.token_map[tvalues[0]].out_type
        d_nt = f'V_C_{CT}_{len(d_sp)}_'
        d_sp[d_nt] = tvalues

        n_info = [
            ([d_nt], cinds, d_sp)
        ]

        if cat_expr is not None:
            minds = [cinds.index(ni) for ni in n_inds]
                
            n_info.append((
                [cat_expr],
                n_inds,
                {k:[V[mi] for mi in minds] for k,V in sparams.items()}
            ))
        
        return n_info

    # calc cost of target expressions
    def calc_old_cost(self, T):
        weights = {
            'fn': self.args.of_fn_weight,
            'cat': self.args.of_cat_weight,
            'float_var': self.args.of_float_var_weight,
            'float_const': self.args.of_float_const_weight,            
        }
        cost = 0.
        for t in T:
            ttype = self.L.getCostType(t)
            if ttype in weights:
                cost += weights[ttype]

        return cost

    # calc cost of new expression being built up in candidate abstraction
    def calc_new_cost(self, T):

        float_vars = set()
        cat_vars = set()

        for t in T:
            if 'V_F_' in t:
                float_vars.add(t)
                
            elif 'V_C_' in t:
                cat_vars.add(t)

        cost = self.args.of_fn_weight + \
            (len(float_vars) * self.args.of_float_var_weight) + \
            (len(cat_vars) * self.args.of_cat_weight)

        return cost

    # calc gain of new expression over old expression, whenever the new expression
    # can be applied
    def calc_exp_gain(self, N, O):
        
        old_cost = self.calc_old_cost(O)
        new_cost = self.calc_new_cost(N)
        
        return old_cost - new_cost
        
    # Make a candidate proposal abstraction for a given structure (cl_sig), based
    # on its parameterization (cl_params). Cl_inds will be used to record which
    # shapes are known to be covered by the returned abstraction

    def make_cand_prop(self, cl_sig, cl_params, cl_inds):
        
        tokens = cl_sig.replace('(', ' ( ').replace(')',' ) ').replace(',', ' , ').split()

        # res has:
        #  score (gain * freq)
        #  partial signature
        #  shared_params
        #  what inds in the cluster are covered

        res = (0.0, [], {}, cl_inds)    

        tme_track = time.time()

        self.float_token_stack = []

        # create candidate abstraction token by token
        for ti in range(len(tokens)):

            # don't get stuck when logic is too complex
            if time.time() - tme_track > MAX_CAND_PROP_TIME:
                return None, None, None
            
            t = tokens[ti]

            old_cost = self.calc_old_cost(tokens[:ti+1])
            
            _, b_sig, b_sparams, b_inds = res

            # find possible expressions to represent the next token
            if 'P_' not in t:                    
                n_info = [([t], b_inds, deepcopy(b_sparams))]
                
            elif '_F_' in t:
                n_info = self.make_float_token(t, b_sparams, cl_params, b_inds)
                                        
            elif '_C_' in t:
                n_info = self.make_cat_token(t, b_sparams, cl_params, b_inds)

            else:
                assert False, f'bad token {t}'

            next_res = None

            # for each potential expression to reprent the next token
            for (nt, ni, nsp) in n_info:
                n_sig = b_sig + nt

                n_score = (old_cost - self.calc_new_cost(n_sig)) *\
                    ((1. * len(ni)) / len(cl_inds))

                NR = (
                    n_score,
                    n_sig,
                    nsp,
                    ni
                )
                if next_res is None or n_score > next_res[0]:
                    # Take the one that best improves score
                    next_res = NR
                        
            res = next_res            

        cand_props = []

        _, b_sig, b_params, b_inds = res
        
        cand_tokens = [s.strip() for s in b_sig]
        cand_prop = ''.join(cand_tokens)

        # Try exectuting first instance in the cluster substituted into the discovered abstraction, to see whether something bad has happened
        test_expr = cand_prop
                
        for k,V in b_params.items():
            test_expr = test_expr.replace(k, str(V[0]))

        try:
            self.L.executor.run(test_expr)            
            assert not self.L.executor.hard_error, 'hard error'
        except Exception as e:
            return None, None, None

        gain = self.calc_exp_gain(cand_tokens, tokens)
        
        # return the candidate proposal, its gain, and what inds it covers
        return cand_prop, gain, b_inds
            

    # Run the ptoposal phase
    def proposal_phase(self, prop_data, CPD):

        # record how much we care about each expression from the proposal data
        # which is created in abstract.py
        CPD.add_inds_and_weights(
            [d[2] for d in prop_data.data],
            [d[3] for d in prop_data.data]
        )
        
        args = self.args 
        
        pbar = tqdm(total=args.ab_num_prop_rounds)
        PITR = 0        

        # Find all clusters of valid structures to consider making proposal
        # abstractions for, additionally get a sampling distribution
        cluster_info, cluster_dist = all_ord_get_clusters(
            prop_data, self.L, self.args
        )

        ttl = 1e-8
        flr = 1e-8

        # the number of candidate abstractions to propose
        while PITR < args.ab_num_prop_rounds:

            # Periodically print recorded information, and top scoring
            # proposal candidates
            if PRINT_PER is not None and ((PITR+1) % PRINT_PER == 0):
                print(f"Fail rate : {flr/ttl}")
                proposals = CPD.rank_proposals(self.L)
                for p in proposals[:PRINT_NUM]:
                    print(p)
            
            # Choose a structure
            smpl_sig, smpl_min_num, smpl_max_num = random.choices(
                cluster_info,                
                weights=cluster_dist,
                k=1
            )[0]

            # Choose a cluster, within that structure
            cl_sig, cl_params, cl_inds = self.sample_based_on_sig(
                prop_data.sig_to_params, smpl_sig, smpl_min_num, smpl_max_num
            )
            
            PITR += 1
            pbar.update(1)
                                    
            with torch.no_grad():

                inp_inds = list(range(len(cl_params[list(cl_params.keys())[0]])))

                # Make a candidate abstraction based on cluster
                cand_prop, cand_gain, cand_inds = self.make_cand_prop(
                    cl_sig,
                    cl_params,
                    inp_inds,
                )

                ttl += 1.
                
                if cand_prop is None:
                    flr +=1.
                    continue

            # If we have succesfully sampled an abstraction, record it
            CPD.add_cand_prop((
                cand_prop,
                cand_gain,
                [cl_inds[ci] for ci in cand_inds],
                cl_sig
            ))            
            
        


