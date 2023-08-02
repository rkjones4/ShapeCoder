# Directory structure

**abs_output** location where results are saved

**abstract.py** logic for integration phase, some scaffolding for proposal phase

**bb_rec_net.py** back-bone of the recognition network

**canon.py** convert an inferred program into a standardized form

**data.py** represent primitive-based shapes, and how those shapes are explained by programs

**dream.py** logic for creating dreams used to train the recognition network

**egg/Cargo.toml** defines building logic for refactor opreation

**egg/src/lib.rs** defines egg refactor operation logic

**env.yml** conda environment to run code

**extract.py** run the refactor operation

**library.py** has library and language related logic

**mask_info.py** type-checking logic for recongition network during inference

**mc_prog_search.py** has logic for sampling from recognition network to explain input shapes

**op_ord.py** convert parametric expressions in abstractions into conditional rewrite logic

**prm_expr_search.py** search for parametric expressions to add to candidate abstraction functions

**prm_sample_dist.py** defines sampling distributions for parameters, used during the dreaming phase

**prog_cost.py** defines the objective function

**prog_utils.py** util functions related to programs

**proposal.py** main logic of proposal phase

**rec_net.py** training logic for recognition network

**rewrites.py** defines semantic rewrites for each domain

**shapecoder.py** entry-point for shapecoder method, connects the rest of the files

**utils.py** any other utility functions
