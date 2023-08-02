import torch
import data
import random

# Class that controls masking strategy of different tokens for network
class MaskInfo:
    def __init__(self, L, token_map, args):
        self.L = L
        self.token_map = token_map
        self.args = args

        self.token_mask = {}
        float_tokens = ['float_var']        
        for typ, tokens in L.type_map.items():
            if typ == 'Float':

                for t in tokens:
                    if t.name not in float_tokens:
                        float_tokens.append(t.name)
                
                continue
            
            if typ not in self.token_mask:
                self.token_mask[typ] = torch.zeros(len(token_map)).float()

            for t in tokens:
                token_ind = token_map[t.name]
                self.token_mask[typ][token_ind] = 1.

        float_token_left = [
            len(L.token_map[n].inp_types) if n != 'float_var' else 0 for n in float_tokens
        ]        

        self.float_token_left_map = {
            n:len(L.token_map[n].inp_types) for n in float_tokens\
            if n != 'float_var'
        }
        
        q = [([], 1)]

        # construct all valid expressions
        valid_exprs = set()
        
        while len(q) > 0:
            bexpr, left = q.pop(0)            
            left -= 1

            assert left >= 0
            
            for tn,tl in zip(float_tokens, float_token_left):
                nexpr = bexpr + [tn]
                nleft = left + tl
                
                if self.check_invalid_expr(nexpr, nleft):
                    continue
                
                if nleft == 0:
                    valid_exprs.add(tuple(nexpr))
                else:
                    q.append((nexpr, nleft))

        # calculate float prediction masks
        self.float_mask = {}

        for vexpr in valid_exprs:
            
            cur = []

            for n in vexpr:

                sig = tuple(cur)

                if sig not in self.float_mask:
                    self.float_mask[sig] = torch.zeros(len(token_map)).float()
                
                self.float_mask[sig][token_map[n]] = 1.0

                cur.append(n)        

        self.prev_left = {}
        self.prev_used = {}
        self.prev_map = {}

    # Check if a float expression is invalid
    def check_invalid_expr(self, expr, left):

        if len(expr) + left > (self.args.ab_rw_max_terms * 2) - 1:
            return True

        if left == 0:
            # if complete expr
            if len(expr) == 1:
                # if 1 thing needs to be singleton const or expr
                t = expr[0]
                if t == 'float_var' or t in self.L.lang.singleton_float_consts:
                    return False
                else:
                    return True

            else:
                # if more than 1 thing, need float_var
                if 'float_var' not in expr:
                    return True
                
        cur = []
        ind = -1
        
        for e in expr:

            if e in self.L.lang.float_fn_map:                
                ind += 1
                cur.append([e])                

            else:
                cur[ind].append(e)
                
            while len(cur[ind]) == 3:
                fn, a, b = cur[ind]
                valid = data.is_valid_expr(self.L.lang.float_fn_map[fn], a, b, {})

                if not valid:
                    return True
                
                comb = ' '.join(cur[ind])
                ind -= 1
                cur[ind].append(comb)
                                        
                
        return False

    # Get mask info for the next token
    def get_mask_info(self, i, next_type):
        if next_type in self.token_mask:
            return self.token_mask[next_type]

        assert next_type == 'Float'

        # Starting a new expr group for float
        if i not in self.prev_map:
            self.prev_map[i] = []
            self.prev_used[i] = []
            self.prev_left[i] = 1            
        
        prev = tuple(self.prev_map[i])
        float_mask = self.float_mask[prev]
        
        return float_mask

    # Add token selected into mask info
    def update_mask_info(self, i, name):
                        
        if 'S_' not in name and name not in self.float_token_left_map:
            return        
        
        self.prev_left[i] -= 1
        if name in self.float_token_left_map:                
            self.prev_left[i] += self.float_token_left_map[name]

        if self.prev_left[i] == 0:
            self.prev_map.pop(i)
            self.prev_used.pop(i)
            self.prev_left.pop(i)
            return
        
        # Group is still going
        
        if 'S_' in name:
            _,I,J,_ = name.split('_')
            p = (int(I) * self.args.rn_prim_dim) + int(J)            
            self.prev_used[i].append(p)
            self.prev_map[i].append('float_var')
        else:
            self.prev_map[i].append(name)
            
    def reset(self):
        self.prev_map = {}
        self.prev_used = {}

        
# Mask info for dreaming "network", which just samples from distributions
        
class DistDummyMaskInfo:
    def __init__(self, net, skip_fns):

        self.L = net.L
        self.type_map = {}
        self.args = net.args        

        self.float_samp_dist = {}
        self.rev_token_map = net.rev_token_map

        for typ, tokens in self.L.type_map.items():
            # Don't sample float operations during dreaming
            if typ == 'Float':
                continue
            if typ not in self.type_map:
                self.type_map[typ] = []

            for t in tokens:
                token_ind = net.token_map[t.name]
                self.type_map[typ].append(token_ind)

        self.skip_fns = set()
        for token in skip_fns:
            token_ind = net.token_map[token.name]
            self.skip_fns.add(token_ind)

        self.type_map = {
            k: list(set(v) - self.skip_fns) for k,v in self.type_map.items()
        }
                
    def sample_token(self, i, next_type):

        if next_type in self.type_map:
            # if next type is not float, sample a matching function
            opts = self.type_map[next_type]

            choice = random.choice(opts)
            token_name = self.rev_token_map[choice]

            self.init_mask_info(i, token_name)

        else:
            # if next type is float, sample from prescribed distribution
            assert next_type == 'Float'
            
            float_samp_dist = self.float_samp_dist[i].pop(0)

            float_val = float_samp_dist(self.args.prec)

            token_name = str(float_val)
            
        return token_name        

    # Initialize mask info based on immediate function call
    def init_mask_info(self, i, name):

        if i not in self.float_samp_dist:
            self.float_samp_dist[i] = []

        if name in self.L.token_map and self.L.token_map[name].samp_fns is not None:
            dfns = [fn for fn in self.L.token_map[name].samp_fns if fn is not None]
            self.float_samp_dist[i] = dfns + self.float_samp_dist[i]
            
