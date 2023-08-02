
# for a given expression (prw), and language (lang), return instructions for egraph of how to calculate that expression
# See rewrite.py for example outputs of various parametric expressions
def make_op_order(prw, lang):
    # To maintain indent
    if True:
        oo = []        
        regs = []

        for t in prw:
            
            if t in lang.float_fn_map:
                full = 2 - int(lang.float_fn_map[t].oneparam)
                
                if len(regs) == 0:
                    regs.append([t, f'reg_0', 0, full])

                else:                                    
                    if regs[-1][2] == 0:
                        regs[-1][2] += 1
                        regs.append([t, regs[-1][1], 0, full])

                    else:
                        regs.append([t, f'reg_{int(regs[-1][1].split("_")[1]) + 1}', 0, full])  
                    
            else:
                if len(regs) == 0:
                    oo.append(('new', t, 'reg_0'))
                else:
                    if regs[-1][2] == 0:
                        regs[-1][2] += 1
                        oo.append(('new', t, regs[-1][1]))
                        
                    else:
                        top_reg = regs.pop(-1)
                        oo.append((top_reg[0], t, top_reg[1]))
            
            while len(regs) > 0 and regs[-1][2] == regs[-1][3]:
                top_reg = regs.pop(-1)
                oo.append((top_reg[0], oo[-1][2], top_reg[1]))
                
                
        while len(regs) > 0:
            top_reg = regs.pop(-1)
            oo.append((top_reg[0], oo[-1][2], top_reg[1]))
            
        return oo

