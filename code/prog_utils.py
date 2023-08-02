import itertools
from copy import deepcopy 
from tqdm import tqdm

# Tokenize an expression
def ps_tokenize(expr):
    expr = expr.replace('(', ' ')\
               .replace(')', ' ')\
               .replace(',', ' ')
    
    return [
        e for e in
        [e.strip() for e in expr.split()]
        if e != ''
    ]

# Helper function to convert a parametric expression to lambda formatting
def _prw_to_lmbda(q, lang):
    p = q.pop(0)
    c = ''
    if p in lang.float_fn_map:                
        c += f'({p} '
        c += _prw_to_lmbda(q, lang)
        if not lang.float_fn_map[p].oneparam:
            c += ' '
            c += _prw_to_lmbda(q, lang)
        c += ')'
    else:
        c += str(p)

    return c

# Function to convert a parametric expression to lambda formatting
def prw_to_lmbda(prw, lang):
    q = list(prw)                
    return _prw_to_lmbda(q, lang) 

# Split a lambda formatted expression into a function and parameters
def split_expr(expr):
    c = 0
    cur = ''
    bind = expr.index(' ')
    r = [expr[1:bind]]
    
    for s in expr[bind+1:-1]:
        cur += s
        
        if s == '(':
            c += 1
            
        if s == ')':
            c -= 1
            if c == 0:
                r.append(cur.strip())
                cur = ''

        if s == ' ' and c == 0:
            r.append(cur.strip())
            cur = ''

    r.append(cur)
    r = [R for R in r if len(R) > 0]
    return r


# Function, and helper function, to evaluate a parametric expression into a real-value
def _expr_eval(expr, lang):
    
    if '(' in expr:
        R = split_expr(expr)
        if len(R) == 3:
            op, a, b = R
            return lang.float_fn_map[op].lmbda(_expr_eval(a, lang), _expr_eval(b, lang))

        else:
            assert len(R) == 2
            op, a = R
            return lang.float_fn_map[op].lmbda(_expr_eval(a, lang))
        
    else:
        return float(expr)
        
def expr_eval(expr, m, lang):    
    for k, v in m.items():
        expr = expr.replace(str(k), str(v))

    val = _expr_eval(expr, lang)
    return val


# Parse lambda expression into parameters
def find_lmbda_params(expr):
    params = []
    n = 0
    e = []
    
    for t in expr.split()[1:]:

        if '(' in t:
            n += t.count('(')

        if ')' in t:
            n -= t.count(')')
            
        e.append(t)

        if n <= 0:
            params.append(' '.join(e))
            e = []
            
    return params

# Convert a lambda formatted expression into a regularly formatted expression
def lmbda_to_prog(expr):
    if expr[0] != '(':
        e = expr.replace(')', '')
        return e
    
    fn = expr.split()[0][1:]
    raw_params = find_lmbda_params(expr)
    params = [lmbda_to_prog(p) for p in raw_params]
    
    return f'{fn}({",".join(params)})'


# Convert a program into a cleaned, canonical form
def clean_prog(
    comb_prog,
    lang,
    opt_prims = None,
    opt_vmap=None,
    do_var_for_cats = False
):
    # maintain indent
    if True:
        VAR_MAP = {}
        IVAR_MAP = {}
        
        inf_prog = ''
        cur = ''

        float_fn_count  = 0
        float_expr = ''

        def _clean_val(val):
            if str(val) == str(-0.0):
                nval = abs(val)
            else:
                nval = val

            return nval
        
        
        for c in comb_prog:

            if float_fn_count > 0:
                if c == ')':
                    float_fn_count -= 1
                elif c == '(':
                    float_fn_count += 1

                float_expr += c

                if float_fn_count == 0:
                    
                    new = float_expr.replace(',', ' ')

                    for name in lang.float_fn_map:
                        new = new.replace(f'{name}(', f'({name} ')

                    name = f'P_F_{len(VAR_MAP)}_'

                    if opt_vmap is not None and len(opt_vmap) > 0:
                        val = expr_eval(new, opt_vmap, lang)                    
                        VAR_MAP[name] = _clean_val(val)
                        IVAR_MAP[name] = float_expr
                    elif 'S_' not in new:
                        assert '_F_' not in new                        
                        
                        val = expr_eval(new, {}, lang)
                        VAR_MAP[name] = _clean_val(val)
                        IVAR_MAP[name] = float_expr

                        
                    inf_prog += name                    
                    float_expr = ''
                    cur = ''
                
                                
            elif c in ['(',')',',']:
                
                if c == '(' and cur in lang.float_fn_map:
                    float_fn_count += 1
                    float_expr += cur + c
                    cur = ''
                    continue                                
                    
                elif 'S_' == cur[:2]:
                    _,i,j,_ = cur.split('_')
                    name = f'P_F_{len(VAR_MAP)}_'
                    VAR_MAP[name] = _clean_val(opt_prims[int(i)][1][int(j)])
                    IVAR_MAP[name] = cur.strip()
                    inf_prog += name

                elif cur.strip() in lang.cat_tokens:
                    if do_var_for_cats:
                        name = f'P_C_{len(VAR_MAP)}_'
                        VAR_MAP[name] = cur.strip()
                        IVAR_MAP[name] = cur.strip()
                        inf_prog += name
                    else:
                        inf_prog += cur

                elif 'P_' == cur[:2] and opt_vmap is not None:
                    name = f'{cur[:3]}_{len(VAR_MAP)}_'
                    VAR_MAP[name] = opt_vmap[cur.strip()]
                    IVAR_MAP[name] = opt_vmap[cur.strip()]
                    inf_prog += name
                                        
                else:
                    try:
                        v = float(cur)
                        name = f'P_F_{len(VAR_MAP)}_'
                        VAR_MAP[name] = _clean_val(v)
                        IVAR_MAP[name] = str(v)
                        inf_prog += name
                    except Exception:
                        inf_prog += cur
                
                inf_prog += c
                cur = ''
            else:
                cur += c

        assert len(cur) == 0 and float_fn_count == 0 and float_expr == ''

        # return prog, map of vars to values, and inverse of that map
        return inf_prog, VAR_MAP, IVAR_MAP

# Convert a regularly formatted expression into a lambda formatted expression
def func_to_lmbda(raw):
    s = ''
    e = ''
    
    cur = ''
    cnt = 0
    for c in raw:
        if c == '(':
            if cnt == 0:
                s = '(' + cur 
                e = ')'
                cur = ''
            else:
                cur += c
                
            cnt += 1
    
            
        elif c == ')' or c == ',' :
            if cnt == 1:
                
                cur = func_to_lmbda(cur)
                s += f' {cur}'
                cur = ''
                
            else:
                cur += c

            if c == ')':
                cnt -= 1

        else:
            cur += c

    s += f'{cur}'

    r = s + e
        
    return r

# Simple tokenizer
def nd_tokenize(expr):
    expr = expr.replace('(', ' ( ').replace(')', ' ) ')
    return [e.strip() for e in expr.split()]

# Split a regularly formatted expression into a function and parameters
def split_func_expr(expr):

    if '(' not in expr:
        return expr, []

    func = expr.split('(')[0]
    
    params = []
    
    cur = ''
    cnt = 0
    
    for c in expr[len(func):]:
        if c == '(':
            if cnt > 0:
                cur += c
                
            cnt += 1
    
            
        elif c == ')' or c == ',' :
            if cnt == 1:
                params.append(cur)
                cur = ''                                
            else:
                cur += c

            if c == ')':
                cnt -= 1

        else:
            cur += c

    params.append(cur)

    r = [R for R in params if len(R) > 0]
            
    return func, r

# convert regular formatted expression into lambda formatted expression
def format_program(raw):
    s = ''
    e = ''
    
    cur = ''
    cnt = 0
    for c in raw:
        if c == '(':
            if cnt == 0:
                s = '(' + cur 
                e = ')'
                cur = ''
            else:
                cur += c
                
            cnt += 1
    
            
        elif c == ')' or c == ',' :
            if cnt == 1:
                
                cur = format_program(cur)
                s += f' {cur}'
                cur = ''
                
            else:
                cur += c

            if c == ')':
                cnt -= 1

        else:
            cur += c

    s += f'{cur}'

    r = s + e
        
    return r

# Simple canonicalization
def canon_sub_progs(sub_progs):
    sub_progs.sort()
    return sub_progs

# Find all valid orderings of sub expressions
def get_all_order_sub_exprs(o_prog, o_vmap, L, max_k):
    for k,v in o_vmap.items():
        o_prog = o_prog.replace(k, str(v))
        
    sub_progs = split_by_union(o_prog)

    c_sub_progs = canon_sub_progs(sub_progs)

    sub_exprs = []
    
    for k in range(1,max_k+1):
        for _comb in itertools.combinations(c_sub_progs, k):
            comb = L.lang.comb_progs(_comb)

            cl_sig, cl_params, _ = clean_prog(comb, L.lang, do_var_for_cats=True)
            
            sub_exprs.append((cl_sig, cl_params))

    return sub_exprs


# Parsing parameter info from an abstraction (sig)
def get_param_info(sig, param_names):
    param_info = {pn: None for pn in param_names}

    q = [sig]

    while len(q) > 0:
        expr = q.pop(0)
        R = split_expr(expr)
        fn = R[0]

        nq = []
        
        for ind, r in enumerate(R[1:]):
            if '(' in r:
                nq.append(r)
            else:
                if r in param_names:
                    if '_F_' in r and param_info[r] is None:
                        param_info[r] = (fn, ind)

        q = nq + q
                        
    param_info = [
        f'{param_info[pn][0]}_{param_info[pn][1]}_' if param_info[pn] is not None else None
        for pn in param_names
    ]
    return param_info
                        
# Split a program by union (this can be generalized to other combinators)
def split_by_union(prog):
    sub_progs = []

    q = [prog]

    while len(q) > 0:
        expr = q.pop(0)
        
        fn, params = split_func_expr(expr)

        if fn == 'Union':
            assert len(params) == 2

            for prm in params:
                if 'Union' == prm[:5]:
                    q.append(prm)
                else:
                    sub_progs.append(prm)
        else:
            sub_progs.append(expr)
            
    return sub_progs


# Checks if there is a match between prog, and the primitives in gt_prims
# Should atleast cover a subset
def find_match_prims(prog, vmap, L, gt_prims, args):
    lines = prog
    
    for _n, _v in vmap.items():
        lines = lines.replace(_n, str(_v))

    L.executor.run(lines)
    e_prims = L.executor.getPrimitives(args.prec)

    if L.executor.hard_error:
        return None
    
    match, err, max_err = L.lang.check_match(e_prims, gt_prims, args)
        
    if match is None:
        return None
        
    res = [(gti, gtprim) for gti, gtprim in gt_prims if gti in match]

    return res

# Parse abstraction info, used to score preference for abstraction
def parse_ab_info(sig, L):

    tokens = sig.replace(',',' , ').replace('(', ' ( ').replace(')', ' ) ').split()
    tokens = [t.strip() for t in tokens if len(t.strip()) > 0]

    num_abs_fns = 0
    abs_fns = set()
    num_fns = 0
    free_params = set()
    duplicate= False

    param_exprs = set()
    _cur = []
    _cnt = 0

    duplicates = []
    consts = []
    
    for t in tokens:

        if '_F_' in t:
            if t in free_params:
                duplicate = True
            free_params.add(t)
            
        elif t in L.token_map and L.token_map[t].cost_type == 'fn':
            if 'Abs' in t:
                num_abs_fns += 1
                abs_fns.add(t)
                
            num_fns += 1

        # expr logic
        if _cnt == 0 and '.' in t:
            try:
                if float(t) in L.lang.float_consts:
                    consts.append(t)
                    continue
            except:
                pass

        if _cnt == 0 and t in param_exprs:
            duplicates.append(t)
            continue
        
        if len(_cur) > 0:
            _cur.append(t)
            if t in L.lang.float_fn_map:
                if L.lang.float_fn_map[t].oneparam:
                    _cnt += 1
                else:
                    _cnt += 2

            if t not in [',','(',')']:
                _cnt -= 1

            if _cnt == 0:
                param_exprs.add(''.join(_cur))
                _cur = []
                
        else:
            if '_F_' in t:
                param_exprs.add(t)
            elif t in L.lang.float_fn_map:
                _cur = [t]
                if L.lang.float_fn_map[t].oneparam:
                    _cnt += 1
                else:
                    _cnt += 2

    param_exprs = [p for p in param_exprs if '(' in p]
                
    num_params = len(free_params) + sig.count('_C_')

    r = {
        'num_fns': num_fns,
        'num_params': num_params,
        'num_dupl': len(duplicates),
        'num_const': len(consts),
        'num_prm_expr': len(param_exprs),
        'num_abs_fns': num_abs_fns,
        'num_uniq_abs_fns': len(abs_fns),
        'num_unions': sig.count('Union'),
    }

    return r

# Remove an abstraction from a program
def remove_abs(prog, vmap, L, rmv_abs):
    for k, v in vmap.items():
        prog = prog.replace(k, str(v))

    raw = lmbda_to_prog(_keep_cur_abs(prog, L, rmv_abs))

    prog_bo_prog, bo_vmap, _ = clean_prog(
        raw,
        L.lang,
        do_var_for_cats =True
    )

    bo_prog = func_to_lmbda(prog_bo_prog)
        
    return bo_prog, bo_vmap

# Keep only abstraction in the prog that should not be removed
# Whenever a bad abstraction is encountered, expand it into sub functions
def _keep_cur_abs(prog, L, rmv_abs):

    if 'Abs' not in prog:
        return prog

    R = split_expr(prog)

    if 'Abs' in R[0]:
        if R[0] in rmv_abs:
            ab = L.abs_register[R[0]]
            assert len(R[1:]) == len(ab.param_names)
            fill_in = ab.exp_fn(R[1:])            
            return _keep_cur_abs(fill_in, L, rmv_abs)
    
    out = [R[0]]
    
    for r in R[1:]:
        if 'Abs' in r:
            out.append(_keep_cur_abs(r, L, rmv_abs))
        else:
            out.append(r)

    return f'({" ".join(out)})'
    
    
