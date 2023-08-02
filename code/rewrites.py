
# Defining semantic rewrites, inherent to the 3D and 2D shape modeling domains

# Each rewrite is (name, left hand side, right hand side, if arrow is both ways, and any conditional checks


SEM_3DL_REWRITES = [

    # record expression for parametric operator nodes, created by other rewrites
    ["er_inv", "(Inv ?a)", "(Inv ?a)", False, "expr_record:new|?a|0+Mul|-1.0|0"],
    ["er_add", "(Add ?a ?b)", "(Add ?a ?b)", False, "expr_record:new|?a|0+Add|?b|0"],

    # flip symmetry over axis
    ['s3dl_sym_x_flip', '(SymRef (Move ?S ?X ?Y ?Z) AX)', '(SymRef (Move ?S (Inv ?X) ?Y ?Z) AX)', True, "none"],
    ['s3dl_sym_y_flip', '(SymRef (Move ?S ?X ?Y ?Z) AY)', '(SymRef (Move ?S ?X (Inv ?Y) ?Z) AY)', True, "none"],
    ['s3dl_sym_z_flip', '(SymRef (Move ?S ?X ?Y ?Z) AZ)', '(SymRef (Move ?S ?X ?Y (Inv ?Z)) AZ)', True, "none"],

    # defining what it means to be a symmetry
    [
        's3dl_sym_x_def',
        '(Union (Move (Cuboid ?W1 ?H1 ?D1) ?X1 ?Y1 ?Z1) (Move (Cuboid ?W2 ?H2 ?D2) ?X2 ?Y2 ?Z2))',
        '(SymRef (Move (Cuboid ?W1 ?H1 ?D1) ?X1 ?Y1 ?Z1) AX)',
        False,
        "abs:0.01:?X1=new|?X2|0+Inv|reg_0|0,?Y1=new|?Y2|0,?Z1=new|?Z2|0,?W1=new|?W2|0,?H1=new|?H2|0,?D1=new|?D2|0"
    ],
    [
        's3dl_sym_y_def',
        '(Union (Move (Cuboid ?W1 ?H1 ?D1) ?X1 ?Y1 ?Z1) (Move (Cuboid ?W2 ?H2 ?D2) ?X2 ?Y2 ?Z2))',
        '(SymRef (Move (Cuboid ?W1 ?H1 ?D1) ?X1 ?Y1 ?Z1) AY)',
        False,
        "abs:0.01:?Y1=new|?Y2|0+Inv|reg_0|0,?X1=new|?X2|0,?Z1=new|?Z2|0,?W1=new|?W2|0,?H1=new|?H2|0,?D1=new|?D2|0"
    ],
    [
        's3dl_sym_z_def',
        '(Union (Move (Cuboid ?W1 ?H1 ?D1) ?X1 ?Y1 ?Z1) (Move (Cuboid ?W2 ?H2 ?D2) ?X2 ?Y2 ?Z2))',
        '(SymRef (Move (Cuboid ?W1 ?H1 ?D1) ?X1 ?Y1 ?Z1) AZ)',
        False,
        "abs:0.01:?Z1=new|?Z2|0+Inv|reg_0|0,?Y1=new|?Y2|0,?X1=new|?X2|0,?W1=new|?W2|0,?H1=new|?H2|0,?D1=new|?D2|0"
    ],

    # undoing a symmetry operation into moves
    ['s3dl_sym_x_def_rev', '(SymRef (Move ?S ?X ?Y ?Z) AX)', '(Union (Move ?S ?X ?Y ?Z) (Move ?S (Inv ?X) ?Y ?Z))', False, "none"],
    ['s3dl_sym_y_def_rev', '(SymRef (Move ?S ?X ?Y ?Z) AY)', '(Union (Move ?S ?X ?Y ?Z) (Move ?S ?X (Inv ?Y) ?Z))', False, "none"],
    ['s3dl_sym_z_def_rev', '(SymRef (Move ?S ?X ?Y ?Z) AZ)', '(Union (Move ?S ?X ?Y ?Z) (Move ?S ?X ?Y (Inv ?Z)))', False, "none"],

    # integrating move into a translational symmetry group
    ['s3dl_symt_move', '(Move (SymTrans (Move ?S ?X1 ?Y1 ?Z1) ?A ?I ?D) ?X2 ?Y2 ?Z2)', '(SymTrans (Move ?S (Add ?X1 ?X2) (Add ?Y1 ?Y2) (Add ?Z1 ?Z2)) ?A ?I ?D)', True, "none"],

    # bring move outside union, if move is the same
    [
        "s3dl_move_over_union_fwd",
        "(Union (Move ?x ?a1 ?b1 ?c1) (Move ?y ?a2 ?b2 ?c2))",
        "(Move (Union ?x ?y) ?a1 ?b1 ?c1)",
        False,
        "abs:0.01:?a1=new|?a2|0,?b1=new|?b2|0,?c1=new|?c2|0"
    ],
    # convert a move over union into union over moves
    [
        "s3dl_move_over_union_bwd",
        "(Move (Union ?x ?y) ?a ?b ?c)",
        "(Union (Move ?x ?a ?b ?c) (Move ?y ?a ?b ?c))",        
        False,
        "none"
    ],

    # combine moves with addition
    ["s3dl_move_comb",  "(Move (Move ?s ?a ?b ?c) ?x ?y ?z)", "(Move ?s (Add ?a ?x) (Add ?b ?y) (Add ?c ?z))", True, "none"],    

    # remove a no-op move
    ["s3dl_zero_mov",  "(Move ?x ?a ?b ?c)" , "?x", False, "abs:0.01:?a=new|0.0|0,?b=new|0.0|0,?c=new|0.0|0"],

    # union re-ordering rewrites, together can create any new order of sub expressions
    ["s3dl_union_comm",  "(Union ?a ?b)" , "(Union ?b ?a)", False, "none"],
    ["s3dl_union_over", "(Union ?a (Union ?b ?c))", "(Union (Union ?a ?b) ?c)", True, "none"],

    # no-op double inverse
    ["s3dl_inv_cancel",  "(Inv (Inv ?a))" , "?a", False, "none"],

    # combine rotations about same axis
    ["s3dl_rot_comb", "(Rotate (Rotate ?X ?A ?v1) ?A ?v2)", "(Rotate ?X ?A (Add ?v1 ?v2))", True, "none"],

    # remove no-op rotation
    ["s3dl_zero_rot", "(Rotate ?X ?A ?v1)", "?X", False, "abs:0.01:?v1=new|0.0|0"],

    # bring rotate over union when same rotation is applied
    [
        "s3dl_rot_over_union_fwd",
        "(Union (Rotate ?X ?A ?v1) (Rotate ?Y ?A ?v2))",
        "(Rotate (Union ?X ?Y) ?A ?v1)",
        False,
        "abs:0.01:?v1=new|?v2|0"
    ],
    # convert a rotate over union into union over rotates
    [
        "s3dl_rot_over_union_bwd",
        "(Rotate (Union ?X ?Y) ?A ?v)",
        "(Union (Rotate ?X ?A ?v) (Rotate ?Y ?A ?v))",
        False,
        "none"
    ],

    # flip rotate and move order
    ["s3dl_rot_over_move", "(Move (Rotate ?X ?A ?v) ?a ?b ?c)", "(Rotate (Move ?X ?a ?b ?c) ?A ?v)", True, "none"],

    # symref can be brought over, or be combined under, union
    ["s3dl_union_over_sym", "(SymRef (Union ?X ?Y) ?A)", "(Union (SymRef ?X ?A) (SymRef ?Y ?A))", True, "none"],

    # add no op when one parameter is equal to 0
    ["zero_add_p1", "(Add ?a ?b)" , "?b", False, "abs:0.01:?a=new|0.0|0"],
    ["zero_add_p2", "(Add ?a ?b)" , "?a", False, "abs:0.01:?b=new|0.0|0"],
    
    
]

# 2D rewrites are a subset, and simplification of the 3D ones, so look above for explanations
SEM_2DL_REWRITES = [

    ["er_inv", "(Inv ?a)", "(Inv ?a)", False, "expr_record:new|?a|0+Mul|-1.0|0"],
    ["er_add", "(Add ?a ?b)", "(Add ?a ?b)", False, "expr_record:new|?a|0+Add|?b|0"],
    
    ['s2dl_sym_x_flip', '(SymRef (Move ?S ?X ?Y) AX)', '(SymRef (Move ?S (Inv ?X) ?Y) AX)', True, "none"],
    ['s2dl_sym_y_flip', '(SymRef (Move ?S ?X ?Y) AY)', '(SymRef (Move ?S ?X (Inv ?Y)) AY)', True, "none"],

    [
        's2dl_sym_x_def',
        '(Union (Move (Rect ?W1 ?H1) ?X1 ?Y1) (Move (Rect ?W2 ?H2) ?X2 ?Y2))',
        '(SymRef (Move (Rect ?W1 ?H1) ?X1 ?Y1) AX)',
        False,
        "abs:0.01:?X1=new|?X2|0+Inv|reg_0|0,?Y1=new|?Y2|0,?W1=new|?W2|0,?H1=new|?H2|0"
    ],
    [
        's2dl_sym_y_def',
        '(Union (Move (Rect ?W1 ?H1) ?X1 ?Y1) (Move (Rect ?W2 ?H2) ?X2 ?Y2))',
        '(SymRef (Move (Rect ?W1 ?H1) ?X1 ?Y1) AY)',
        False,
        "abs:0.01:?Y1=new|?Y2|0+Inv|reg_0|0,?X1=new|?X2|0,?W1=new|?W2|0,?H1=new|?H2|0"
    ],    
    

    ['s2dl_sym_x_def_rev', '(SymRef (Move ?S ?X ?Y) AX)', '(Union (Move ?S ?X ?Y) (Move ?S (Inv ?X) ?Y))', False, "none"],
    ['s2dl_sym_y_def_rev', '(SymRef (Move ?S ?X ?Y) AY)', '(Union (Move ?S ?X ?Y) (Move ?S ?X (Inv ?Y)))', False, "none"],
    
    ['s2dl_symt_move', '(Move (SymTrans (Move ?S ?X1 ?Y1) ?A ?I ?D) ?X2 ?Y2)', '(SymTrans (Move ?S (Add ?X1 ?X2) (Add ?Y1 ?Y2)) ?A ?I ?D)', True, "none"],

    [
        "s2dl_move_over_union_fwd",
        "(Union (Move ?x ?a1 ?b1) (Move ?y ?a2 ?b2))",
        "(Move (Union ?x ?y) ?a1 ?b1)",
        False,
        "abs:0.01:?a1=new|?a2|0,?b1=new|?b2|0"
    ],
    [
        "s2dl_move_over_union_bwd",
        "(Move (Union ?x ?y) ?a ?b)",
        "(Union (Move ?x ?a ?b) (Move ?y ?a ?b))",        
        False,
        "none"
    ],
    
    ["s2dl_move_comb",  "(Move (Move ?s ?a ?b) ?x ?y)", "(Move ?s (Add ?a ?x) (Add ?b ?y))", True, "none"],    

    ["s2dl_zero_mov",  "(Move ?x ?a ?b)" , "?x", False, "abs:0.01:?a=new|0.0|0,?b=new|0.0|0"],
        
    ["s2dl_union_comm",  "(Union ?b ?a)" , "(Union ?a ?b)", False, "none"],
    ["s2dl_union_over", "(Union ?a (Union ?b ?c))", "(Union (Union ?a ?b) ?c)", True, "none"],
    ["s2dl_inv_cancel",  "(Inv (Inv ?a))" , "?a", False, "none"],
    
    ["s2dl_union_over_sym", "(SymRef (Union ?X ?Y) ?A)", "(Union (SymRef ?X ?A) (SymRef ?Y ?A))", True, "none"],

    ["zero_add_p1", "(Add ?a ?b)" , "?b", False, "abs:0.01:?a=new|0.0|0"],
    ["zero_add_p2", "(Add ?a ?b)" , "?a", False, "abs:0.01:?b=new|0.0|0"],
        
]

# Add rewrites in list L into repository R
def add_rewrites(L, R):
    for defn in L:
        name, lhs, rhs, rev, info = defn
        R.append([name, lhs, rhs, info])
        # if rewrite should be reversed (e.g. its a two way rewrite)
        if rev:
            assert info == "none"
            R.append([name+'_rev', rhs, lhs, info])

