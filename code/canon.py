from copy import deepcopy
import prog_utils as pu
import prog_cost
import extract as extr
import abstract as abst

# get all parameter related rewrites from info in vmap
def get_param_rws(vmap):
    rws = []

    for k,v in vmap.items():
        if '_F_' in k:
            rws.append([f"vrrw{len(rws)}", k, k, f"var_record:{v}"])
        elif '_C_' in k:
            rws.append([f'cat_rw_{len(rws)}', k, v, "none"])

    return rws

# canonicalize program by running it through e-graph
def canonicalize(L, prog, vmap, args, match_prims):
    # reformat program    
    program = pu.format_program(prog)

    # get obj function cost
    orig_struct_cost, _ = prog_cost.split_calc_obj_fn(
        L,
        prog,
        0.0,
        args
    )

    # get os rws
    egg_os_rws = get_param_rws(vmap)

    egg_sat_rws = L.lang.base_rewrites + L.make_abs_rewrites()

    # run e-graph logic
    egg_out, _, _, _ = extr.safe_extract(
        program,
        egg_os_rws,
        egg_sat_rws,
        args.egg_max_rounds,
        args.egg_max_nodes,
        args.egg_max_time,
    )
    
    egg_prog, egg_params, egg_error = abst.egg_postprocess(L, egg_out, vmap)

    egg_match_prims = pu.find_match_prims(
        pu.lmbda_to_prog(egg_prog),
        egg_params,
        L,
        match_prims,
        args
    )

    # find out cost of e-graph extraction step
    struct_cost, _ = prog_cost.split_calc_obj_fn(
        L,
        egg_prog,
        0.0,
        args
    )    
    
    if egg_match_prims is not None and \
       struct_cost < orig_struct_cost:

        # if we cover the right primitives
        # and we improve structural cost, then switch to new program
        
        prog = pu.lmbda_to_prog(egg_prog)

        vmap = egg_params
        
        for k,v in vmap.items():
            prog = prog.replace(k, str(v))


    # find canonical order by sorting all sub programs
    sub_progs = pu.split_by_union(prog)
    canon_sub_progs = pu.canon_sub_progs(
        sub_progs,
    )
    comb_prog= L.lang.comb_progs(
        canon_sub_progs
    )
            
    new_prog, new_vmap, _ = pu.clean_prog(
        comb_prog,
        L.lang,
        opt_vmap = vmap,
        do_var_for_cats=True
    )
    
    return new_prog, new_vmap
