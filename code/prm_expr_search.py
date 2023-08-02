import torch
import numpy as np
import itertools
import prog_utils as pu

# Record what found expressions should be kept, removing overly complex ones
def filter_matches_winds(matches, keys):
    sm = [(len(m), m, inds) for m, inds in matches]
    sm.sort()

    fmatches = []
    var_set = set()

    for _,m,inds in sm:
        ks = tuple(k for k in keys if k in m)
        # if we've already used this variable set, then skip
        if ks in var_set:
            continue
        fmatches.append((m, inds))
        var_set.add(ks)

    return fmatches
        
# is the found expression valid?
# These are typically helpful heuristics
def sc_is_valid_expr(lang, e):
    fnmaps = {fn:set() for fn in lang.float_fn_map.keys()}
    
    seen = set()

    if '(' not in e:
        return True
    
    q = [e]

    while len(q) > 0:
        fn, R = pu.split_func_expr(q.pop(0))

        saw_var = False
        
        for r in R:

            if '_F_' in r:
                saw_var = True
            
            if '(' in r:                
                q.append(r)
            else:                
                if '_F_' in r:
                    # are multiple of same parameter used
                    if r in seen:
                        return False
                    seen.add(r)
                else:
                    # is an invalid float used
                    if float(r) in lang.float_fn_map[fn].invld:
                        return False                

                if r in fnmaps[fn]:
                    # has this param already been used by this operator,
                    # in this expression
                    return False
                
                fnmaps[fn].add(r)
                    
        if not saw_var:
            # if there wasn't a variable, but was an operator
            return False
        
    for a,b in [('Add', 'Sub'), ('Mul', 'Div')]:
        # Don't use inverse operators, where same operator shows up twice
        if len(fnmaps[a].intersection(fnmaps[b])) > 0:
            return False
        
    return True

# Do values v, match targets, within max error cme, and cover atleast mcmp instances
def sc_match(v, targets, cme, mcmp):
    DA = (v-targets).abs() <= cme
    return DA.float().mean().item() >= mcmp, DA.nonzero().flatten().tolist()

# Do matching, but in batch, to take advantage of parallelism
def sc_batch_match(R, targets, cme, mcmp, I, K):

    # find which expressions had matches within range
    DA = (R-targets.view(1,-1)).abs() <= cme

    # Then find those that had enough matches, about mcmp instances
    ii = (
        DA.float().mean(dim=1) >= mcmp
    ).nonzero().flatten()
    
    matches = []

    # record these results
    for _ii in ii:
    
        combo_inds = I[_ii]

        match_inds = DA[_ii].nonzero().flatten().tolist()
        
        matches.append((
            [K[ci] for ci in combo_inds],
            match_inds
        ))    
    
    return matches

# Logic to check for zero op expressions
# E.g. a previous variable, or a constant is used
def sc_hfe_zero_op(inp, ret_inds=False):

    matches = []

    targets = inp['targets']
    cme = inp['cme']
    mcmp = inp['mcmp']

    for k,v in list(inp['pvars'].items()) + list(inp['st_consts'].items()):
        sm, minds = sc_match(v, targets, cme, mcmp)
        if sm:
            if ret_inds:
                matches.append((k, minds))
            else:
                matches.append(k)

    return matches

# TODO COMMENT
def sc_hfe_pvar(inp, ret_inds=False):

    matches = []

    targets = inp['targets']
    cme = inp['cme']
    mcmp = inp['mcmp']

    for k,v in list(inp['pvars'].items()):
        sm, minds = sc_match(v, targets, cme, mcmp)
        if sm:
            if ret_inds:
                matches.append((k, minds))
            else:
                matches.append(k)

    return matches

# TODO COMMENT
def sc_hfe_sconst(inp, ret_inds=False):

    matches = []

    targets = inp['targets']
    # TODO GENERALIZE
    cme = 0.01
    mcmp = inp['mcmp']

    for k,v in list(inp['st_consts'].items()):
        sm, minds = sc_match(v, targets, cme, mcmp)
        if sm:
            if ret_inds:
                matches.append((k, minds))
            else:
                matches.append(k)

    return matches

# Check for one parameter operator expressions,
# by looking over all one parameter expressions in lang + all previously declared vars
def sc_hfe_oneprm_op(inp, ret_inds=False):

    matches = []

    targets = inp['targets']
    cme = inp['cme']
    mcmp = inp['mcmp']

    for name,op in inp['op_ops'].items():        
        for k,v in list(inp['pvars'].items()):
            
            sm, minds = sc_match(op.pt_lmbda(v), targets, cme, mcmp)
            if sm:
                if ret_inds:
                    matches.append((f'{name}({k})', minds))
                else:
                    matches.append(f'{name}({k})')

    return matches

# Check for one parameter expressions, in batch so its optomized
def sc_hfe_one_op(inp, ret_inds=False):

    matches = []

    targets = inp['targets']
    cme = inp['cme']
    mcmp = inp['mcmp']

    # consider previous vars, and constants
    K = list(inp['pvars'].keys()) + list(inp['ex_consts'].keys())
    
    V = torch.stack((list(inp['pvars'].values())), dim=0)
    os = V.shape[0]
    V = torch.cat((
        V, torch.zeros(len(inp['ex_consts']), V.shape[1])
    ),dim=0)

    for i, v in enumerate(inp['ex_consts'].values()):
        V[os+i] = v

    # find allowable combinations
    I = torch.cartesian_prod(
        *(
            [torch.arange(os)] +  \
            [torch.arange(V.shape[0]) for _ in range(1)]
        )
    )    

    C = V[I]

    # try for all regular operators (two inptus)
    for name,op in inp['tp_ops'].items():
        res = op.pt_lmbda(C[:,0,:], C[:,1,:])

        # check matches in batch
        _m = sc_batch_match(
            res,
            targets,
            cme,
            mcmp,
            I,
            K
        )
        
        for (a,b), minds in _m:
            if ret_inds:
                matches.append((f'{name}({a},{b})', minds))
            else:
                matches.append(f'{name}({a},{b})')
            
    return matches

# Similiar to one operator logic, but with two operators
# Also done in batch
def sc_hfe_two_op(inp, ret_inds=False):

    matches = []

    targets = inp['targets']
    cme = inp['cme']
    mcmp = inp['mcmp']

    K = list(inp['pvars'].keys()) + list(inp['ex_consts'].keys())
    
    V = torch.stack((list(inp['pvars'].values())), dim=0)
    os = V.shape[0]
    V = torch.cat((
        V, torch.zeros(len(inp['ex_consts']), V.shape[1])
    ),dim=0)

    for i, v in enumerate(inp['ex_consts'].values()):
        V[os+i] = v

    I = torch.cartesian_prod(
        *(
            [torch.arange(os)] +  \
            [torch.arange(V.shape[0]) for _ in range(2)]
        )
    )    

    C = V[I]

    op_combos = list(
        itertools.product(*[list(inp['tp_ops'].items()) for _ in range(2)])
    )
    # find operator combinations to search over
    for op_set in op_combos:
    
        ((a_n,a_op), (b_n, b_op)) = op_set
                
        x_res = a_op.pt_lmbda(C[:,0,:],b_op.pt_lmbda(C[:,1,:], C[:,2,:]))

        x_m = sc_batch_match(
            x_res,
            targets,
            cme,
            mcmp,
            I,
            K
        )
        
        for (a,b,c), minds in x_m:
            if ret_inds:
                matches.append((f'{a_n}({a},{b_n}({b},{c}))', minds))
            else:
                matches.append(f'{a_n}({a},{b_n}({b},{c}))')

        y_res = a_op.pt_lmbda(b_op.pt_lmbda(C[:,0,:],C[:,1,:]), C[:,2,:])

        y_m = sc_batch_match(
            y_res,
            targets,
            cme,
            mcmp,
            I,
            K
        )
        
        for (a,b,c), minds in y_m:
            if ret_inds:
                matches.append((f'{a_n}({b_n}({a},{b}),{c})', minds))
            else:
                matches.append(f'{a_n}({b_n}({a},{b}),{c})')

    return matches

# Similiar to one operator logic, but with two operators
# Also done in batch
def sc_hfe_three_op(inp, ret_inds=False):

    matches = []

    targets = inp['targets']
    cme = inp['cme']
    mcmp = inp['mcmp']

    K = list(inp['pvars'].keys()) + list(inp['ex_consts'].keys())
    
    V = torch.stack((list(inp['pvars'].values())), dim=0)
    os = V.shape[0]
    V = torch.cat((
        V, torch.zeros(len(inp['ex_consts']), V.shape[1])
    ),dim=0)

    for i, v in enumerate(inp['ex_consts'].values()):
        V[os+i] = v

    I = torch.cartesian_prod(
        *(
            [torch.arange(os)] +  \
            [torch.arange(V.shape[0]) for _ in range(3)]
        )
    )    

    C = V[I]

    # find operator combinations
    op_combos = list(
        itertools.product(*[list(inp['tp_ops'].items()) for _ in range(3)])
    )
    
    for op_set in op_combos:
    
        ((a_n,a_op), (b_n, b_op), (c_n, c_op)) = op_set
                
        x_res = a_op.pt_lmbda(C[:,0,:],b_op.pt_lmbda(C[:,1,:], c_op.pt_lmbda(C[:,2,:], C[:,3,:])))
        y_res = a_op.pt_lmbda(b_op.pt_lmbda(C[:,0,:],C[:,1,:]), c_op.pt_lmbda(C[:,2,:], C[:,3,:]))
        z_res = a_op.pt_lmbda(b_op.pt_lmbda(c_op.pt_lmbda(C[:,0,:],C[:,1,:]), C[:,2,:]), C[:,3,:])
        
        x_m = sc_batch_match(
            x_res,
            targets,
            cme,
            mcmp,
            I,
            K
        )
        
        for (a,b,c,d), minds in x_m:
            if ret_inds:
                matches.append((f'{a_n}({a},{b_n}({b},{c_n}({c},{d})))', minds))
            else:
                matches.append(f'{a_n}({a},{b_n}({b},{c_n}({c},{d})))')

        y_m = sc_batch_match(
            y_res,
            targets,
            cme,
            mcmp,
            I,
            K
        )
        
        for (a,b,c,d), minds in y_m:
            if ret_inds:
                matches.append((f'{a_n}({b_n}({a},{b}),{c_n}({c},{d}))', minds))
            else:
                matches.append(f'{a_n}({b_n}({a},{b}),{c_n}({c},{d}))')

        z_m = sc_batch_match(
            z_res,
            targets,
            cme,
            mcmp,
            I,
            K
        )
        
        for (a,b,c,d), minds in z_m:
            if ret_inds:
                matches.append((f'{a_n}({b_n}({c_n}({a},{b}),{c}),{d})', minds))
            else:
                matches.append(f'{a_n}({b_n}({c_n}({a},{b}),{c}),{d})')
            
    return matches
