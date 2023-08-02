
import multiprocessing
try:
    import sc_egraph as sce
    print("Succesfully imported egg refactor modules")
except:
    print("!!!!!!!! Failed to import egg refactor module, please investigate unless this was expected")
    
manager = multiprocessing.Manager()


# Grace Factor
# How long to wait, before killing the job
GF = 2

def _extract(expr, os_rws, sat_rws, mr, mn, mt, R):
    egg_out = sce.sc_extract(
        expr,
        os_rws,
        sat_rws,
        mr,
        mn,
        mt
    )

    egg_out, cost, stop_reason = egg_out.split('+')
    
    R['res'] = egg_out
    R['cost'] = cost
    R['stop_reason'] = stop_reason

# Extract (refactor) takes in expression, one time rewrites, saturating rewrites
# max rounds, max nodes, max time, and creates an egraph, returning the minimum cost expression under the egraph
def safe_extract(expr, os_rws, sat_rws, mr, mn, mt):
    
    R = manager.dict()
    
    p = multiprocessing.Process(
        target = _extract,
        args=(expr, os_rws, sat_rws, mr, mn, mt, R),
        kwargs={}
    )

    p.start()
    p.join(mt * GF)

    # Call sometimes hangs, so with an extended grace factor of time, sometimes kill the process
    if p.is_alive():
        p.terminate()
        print("Failed on time")
        return expr, None, None, True

    try:
        # Check if the reason extraction was stopped is because we saturated the e-graph
        sat = 'Saturated' in R['stop_reason']
        if 'cost' in R:    
            return R['res'], R['cost'], sat, False

        return R['res'], None, sat, False
    except Exception as e:
        print(
            f"##################"
            f"##################"
            f"Failed extract with {e}\n"
            f"    expr :  {expr}\n"
            f"    os_rws :  {os_rws}\n"
            f"    sat_rws :  {sat_rws}\n"
            f"##################"
            f"##################"
        )
        return expr, None, None, True
    
