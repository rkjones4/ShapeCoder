import prog_utils as pu

# Calculate cost of structural parts of a program (tokens)
def struct_calc_prog_cost(L, tokens, weights, terms):

    counts = {ttype:0. for ttype in terms}
        
    for t in tokens:
            
        ttype = L.getCostType(t)
            
        if ttype in counts:
            counts[ttype] += 1.

    cost = 0.
    for ttype in counts:
        cost += weights[ttype] * counts[ttype]

    return cost

# Calculate cost of parametric parts of a program
def param_calc_prog_cost(L, tokens, weights, terms, flt_err):

    seen = {ttype: set() for ttype in terms if 'var' in ttype}
    counts = {ttype:0. for ttype in terms}
    counts['float_error'] = flt_err    
        
    for t in tokens:
            
        ttype = L.getCostType(t)

        if ttype in seen:
            seen[ttype].add(t)
        elif ttype in counts:
            counts[ttype] += 1.
        
    for k, v in seen.items():
        counts[k] = len(v) * 1.
            
    cost = 0.
    for ttype in counts:
        cost += weights[ttype] * counts[ttype]

    return cost

# Calculate cost of all program components
def full_calc_prog_cost(L, tokens, weights, flt_err, avg_counts = None):

    counts = {ttype:0. for ttype in weights}
    counts['float_error'] = flt_err    
        
    for t in tokens:
            
        ttype = L.getCostType(t)
            
        if ttype is not None:
            counts[ttype] += 1.

    prog_costs = 0.
    for ttype in weights:
        prog_costs += weights[ttype] * counts[ttype]

        if avg_counts is not None:
            avg_counts[ttype].append(counts[ttype])

    return prog_costs

# Return struct and param parts of cost seperately
def split_calc_obj_fn(L, text, err, args):
    
    tokens = pu.ps_tokenize(text)

    weights = args.of_weights
    
    struct_cost = struct_calc_prog_cost(
        L,
        tokens,
        weights,
        args.of_struct_terms,
    )
    
    param_cost = param_calc_prog_cost(
        L,
        tokens,
        weights,
        args.of_param_terms,
        err,
    )

    return struct_cost, param_cost      
