import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import device
import library
import time

# fourier basis
class Fourier(nn.Module):
    def __init__(self, dim, nmb=16, scale=1.):
        super(Fourier, self).__init__()
        assert nmb % 2 == 0
        self.b = torch.randn(dim, nmb//2)*scale
        self.pi = 3.14159265359
        
    def forward(self, v):
        x_proj = torch.matmul(2*self.pi*v, self.b.to(v.device))
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], -1)        

# MLP with dropout
class DMLP(nn.Module):
    def __init__(self, ind, hdim1, hdim2, odim, DP):
        super(DMLP, self).__init__()
        
        self.l1 = nn.Linear(ind, hdim1)
        self.l2 = nn.Linear(hdim1, hdim2)
        self.l3 = nn.Linear(hdim2, odim)
        self.d1 = nn.Dropout(p=DP)
        self.d2 = nn.Dropout(p=DP)
                
    def forward(self, x):
        x = self.d1(F.leaky_relu(self.l1(x), 0.2))
        x = self.d2(F.leaky_relu(self.l2(x), 0.2))
        return self.l3(x)

# Small MLP with dropout
class SDMLP(nn.Module):
    def __init__(self, ind, odim, DP):
        super(SDMLP, self).__init__()
        
        self.l1 = nn.Linear(ind, odim)
        self.l2 = nn.Linear(odim, odim)
        self.d1 = nn.Dropout(p=DP)
                
    def forward(self, x):
        x = self.d1(F.leaky_relu(self.l1(x), 0.2))
        return self.l2(x)

# Small MLP
class SMLP(nn.Module):
    def __init__(self, ind, odim):
        super(SMLP, self).__init__()
        
        self.l1 = nn.Linear(ind, odim)
        self.l2 = nn.Linear(odim, odim)
                
    def forward(self, x):
        x = F.leaky_relu(self.l1(x), 0.2)
        return self.l2(x)

# Attention layer
class AttnLayer(nn.Module):
    def __init__(self, nh, hd, dropout):
        super(AttnLayer, self).__init__()

        self.nh = nh
        self.hd = hd

        self.self_attn = torch.nn.MultiheadAttention(self.hd, self.nh)

        self.l1 = nn.Linear(hd, hd)
        self.l2 = nn.Linear(hd, hd)

        self.d1 = nn.Dropout(dropout)
        self.d2 = nn.Dropout(dropout)
        self.d3 = nn.Dropout(dropout)        

        self.n1 = nn.LayerNorm(hd)
        self.n2 = nn.LayerNorm(hd)
                
    def forward(self, _src, attn_mask, key_padding_mask):
        
        src = _src.transpose(0, 1)
            
        src2 = self.self_attn(
            src,
            src,
            src,
            attn_mask=attn_mask,
            key_padding_mask = key_padding_mask
        )[0]

        src = src + self.d1(src2)
        src = self.n1(src)
        src2 = self.l2(self.d2(F.leaky_relu(self.l1(src), .2)))
        src = src + self.d2(src2)
        src = self.n2(src)

        return src.transpose(0, 1)

# Hyper net encoder
class EncHyperNet(nn.Module):
    def __init__(self, mp, pd, vd, A):
        super(EncHyperNet, self).__init__()

        self.mp = mp
        self.pd = pd
        self.vd = vd
        
        self.context_layer = nn.Linear(mp * pd, A)
        self.ind_layer = nn.Embedding(mp * pd, A)
        self.hn_layer = SMLP(A, vd)
        self.ar = torch.arange(mp * pd)
        
    def forward(self, data, context):

        cl = self.context_layer(context)
        el = self.ind_layer(self.ar.to(context.device))

        comb_l = cl.unsqueeze(1) + el.unsqueeze(0)

        hn_w = self.hn_layer(comb_l)
        
        flat_inds = ((torch.arange(data.shape[0], device=data.device) * self.mp * self.pd).view(-1, 1) + data).flatten()
        flat_hn_w = hn_w.view(-1, self.vd)        

        flat_encs = flat_hn_w[flat_inds]

        return flat_encs

# Pointer net encoder
class EncPointerNet(nn.Module):
    def __init__(self, mp, pd, vd, nh, dropout):
        super(EncPointerNet, self).__init__()

        self.mp = mp
        self.pd = pd
        self.vd = vd
        self.nh = nh
        self.dropout = dropout
        
        self.lift_layer = SMLP(vd, vd)
        self.ff_emb = Fourier(dim=1, nmb=self.vd, scale=1.)
        self.attn = AttnLayer(self.nh, self.vd, self.dropout)
        self.attn_mask = torch.zeros(mp * pd, mp * pd, device= device).float()

    def generate_key_mask(self, num):
        mask = torch.zeros(num.shape[0], self.mp * self.pd, device=device).bool()
        for i,np in enumerate(num):
            mask[i,np:] = True

        return mask
        
    def forward(self, data, context, num):

        ff = self.ff_emb(context.unsqueeze(-1))

        o1 = self.lift_layer(ff)

        key_mask = self.generate_key_mask(num)

        hn_w = self.attn(o1, self.attn_mask, key_mask).contiguous()

        flat_inds = ((torch.arange(data.shape[0], device=data.device) * self.mp * self.pd).view(-1, 1) + data).flatten()

        flat_hn_w = hn_w.view(-1, self.vd)        

        flat_encs = flat_hn_w[flat_inds]
        
        return flat_encs
        
# No context encoder, only receives indices of values
class EncNoContextNet(nn.Module):
    def __init__(self, mp, pd, vd):
        super(EncNoContextNet, self).__init__()

        self.mp = mp
        self.pd = pd
        self.vd = vd
        
        self.ind_layer = nn.Embedding(mp * pd, vd)
        
    def forward(self, data):                
        el = self.ind_layer(data)
        return el

# Hyper network for prediction head
class HeadHyperNet(nn.Module):
    def __init__(self, mp, pd, hd, vd, DP):
        super(HeadHyperNet, self).__init__()        
    
        self.mp = mp
        self.pd = pd
        self.hd = hd
        self.vd = vd

        self.context_layer = nn.Linear(mp * pd, hd)
        self.ind_layer = nn.Embedding(mp * pd, hd)
        self.ar = torch.arange(mp * pd)
        self.hn_l2_layer = SMLP(hd, vd)
        self.hn_l1_layer = SMLP(hd, hd * vd)

        self.d1 = nn.Dropout(p=DP)
        
    def forward(self, seq_out, context):
        
        cl = self.context_layer(context)
        el = self.ind_layer(self.ar.to(context.device))

        l1_w = self.hn_l1_layer(cl).view(-1, self.hd, self.vd)
                
        l2_comb = cl.unsqueeze(1) + el.unsqueeze(0)
        l2_w = self.hn_l2_layer(l2_comb).transpose(1,2)    
        
        l1_o = self.d1(F.leaky_relu(seq_out @ l1_w, 0.2))
        
        l2_o = l1_o @ l2_w
        
        return l2_o

    def eval_init(self, context):        
        
        cl = self.context_layer(context)
        el = self.ind_layer(self.ar.to(context.device))

        l1_w = self.hn_l1_layer(cl).view(-1, self.hd, self.vd)
                
        l2_comb = cl.unsqueeze(1) + el.unsqueeze(0)
        l2_w = self.hn_l2_layer(l2_comb).transpose(1,2)

        self.EVAL_l1_w = l1_w
        self.EVAL_l2_w = l2_w
    
    def eval_forward(self, seq_out):
            
        l1_o = self.d1(F.leaky_relu(seq_out @ self.EVAL_l1_w, 0.2))
        
        l2_o = l1_o @ self.EVAL_l2_w
        
        return l2_o

# Calculate info for example prims
def pre_hn_info(args, prims, num_prims):

    L = args.rn_max_prims * args.rn_prim_dim
    
    hn_data = torch.zeros(L).long()
    hn_context = torch.zeros(L).float()
            
    uvals = prims[:num_prims].unique().sort().values
    hn_context[:uvals.shape[0]] = uvals
    hn_num = uvals.shape[0]
    
    m = {uv.item():j for j,uv in enumerate(uvals)}            
    
    for j, uv in enumerate(prims[:num_prims].flatten()):
        hn_data[j] = m[uv.item()]
        
    return hn_data, hn_context, hn_num

# Back bone of the rec net
class BBRecNet(nn.Module):    
    def __init__(
            self,
            args,
            num_tokens,            
    ):
        super(BBRecNet, self).__init__()

        self.mode = args.rn_mode    
        self.device = device
        
        self.pd = args.rn_prim_dim
        self.mp = args.rn_max_prims
        self.ms = args.rn_max_seq

        self.nl = args.rn_num_layers
        self.nh = args.rn_num_heads
                
        self.bs = args.rn_batch_size
        self.dropout = args.rn_dropout
        
        self.nt = num_tokens # lib function + 2 (for float var and start)        
        self.hd = args.rn_hidden_dim # emb hidden dim
        self.vd = args.rn_var_dim # variable emb dim
        
        self.token_enc_net = nn.Embedding(self.nt, self.hd)        
        self.token_head = SDMLP(self.hd + self.vd, self.nt, self.dropout)

        self.token_pos_enc = nn.Embedding(self.ms, self.hd + self.vd)
        
        self.token_pos_arange = torch.arange(self.ms).unsqueeze(0).repeat(self.bs, 1)

        # Different encoding and decoding (prediction head) modes
        # nocontext is only officially supported mode        
        
        if 'enc:ff' in self.mode:
            self.var_enc_net = nn.Sequential(                
                Fourier(dim=1, nmb=self.vd, scale=1.),
                SMLP(self.vd, self.vd),
            )
            
        elif 'enc:mlp' in self.mode:
            self.var_enc_net = SMLP(1, self.vd)

        elif 'hyper' in self.mode or 'hypheadabl' in self.mode:
            self.var_enc_net = EncHyperNet(self.mp, self.pd, self.vd, self.hd)

        elif 'enc:habl' in self.mode:
            self.var_enc_net = nn.Embedding(self.mp * self.pd, self.vd)
            
            self.cnt_enc_net = SMLP((self.mp * self.pd) + self.vd, self.vd)

            self.cnt_learn_const = SMLP(self.mp * self.pd, self.hd)

        elif 'nocontext' in self.mode:
            # Creates primitive encoder
            self.var_enc_net = EncNoContextNet(self.mp, self.pd, self.vd)

        elif 'pointer' in self.mode:
            self.var_enc_net = EncPointerNet(self.mp, self.pd, self.vd, self.nh, self.dropout)
            
        else:
            assert False, f'Bad mode {self.mode}'
            
        if 'head:mlp' in self.mode:
            self.var_head = SMLP(self.hd + self.vd, self.vd)

        elif 'hyper' in self.mode:
            self.var_head = HeadHyperNet(self.mp, self.pd, self.hd + self.vd, self.vd, self.dropout)

        elif 'head:ind' in self.mode or 'nocontext' in self.mode:
            # Creates var head
            self.var_head = SMLP(self.hd + self.vd, self.mp * self.pd)

        elif 'hypheadabl' in self.mode or 'pointer' in self.mode:
            self.var_head = SMLP(self.hd + self.vd, self.vd)
            
        else:
            assert False, f'Bad mode {self.mode}'
            
        self.var_pos_enc = nn.Embedding(self.pd, self.vd)
        self.var_pos_arange = torch.arange(self.pd).view(1,1,self.pd).repeat(self.bs, self.mp, 1).flatten()

        self.learn_enc_ind = torch.zeros(self.bs * self.mp * self.pd).long()
            
        self.learn_enc_const = nn.Embedding(1, self.hd)
            
        self.attn_mask = self.generate_attn_mask()

        self.attn_layers = nn.ModuleList([AttnLayer(self.nh, self.hd + self.vd, self.dropout) for _ in range(self.nl)])

        self.celoss = torch.nn.CrossEntropyLoss(reduction='none')

    def build_state(self, tprims, nprims):
        # During inference, build up state for prediction
        
        prims = tprims.unsqueeze(0).repeat(self.bs, 1, 1).to(self.device)
        num_prims = [nprims for _ in range(self.bs)]
        
        if 'hyper' in self.mode:
            with torch.no_grad():
                hn_data, hn_context, hn_num = self.get_hn_info(prims, num_prims)
                
            self.EVAL_hn_data = hn_data
            self.EVAL_hn_context = hn_context
            self.EVAL_hn_num = hn_num

            self.EVAL_sing_hn_num = hn_num[0].item()

            m = {}
            
            _hn_data = hn_data[0]

            for i in range(hn_num[0].item()):
                var = (_hn_data == i).nonzero().flatten()[0].item()
                I = int(var / self.pd)
                J = int(var % self.pd)
                m[i] = (I, J)

            self.EVAL_hn_map = m
                            
            flat_var_encs = self.var_enc_net(hn_data, hn_context)

            var_encs = None

            self.var_head.eval_init(hn_context)
            
        elif 'habl' in self.mode:
            with torch.no_grad():
                hn_data, hn_context, hn_num = self.get_hn_info(prims, num_prims)
        
            self.EVAL_hn_data = hn_data
            self.EVAL_hn_context = hn_context
            self.EVAL_hn_num = hn_num

            self.EVAL_sing_hn_num = hn_num[0].item()

            m = {}
            
            _hn_data = hn_data[0]            
            
            for i in range(hn_num[0].item()):
                var = (_hn_data == i).nonzero().flatten()[0].item()
                I = int(var / self.pd)
                J = int(var % self.pd)
                m[i] = (I, J)

            self.EVAL_hn_map = m

            enc_out = self.var_enc_net(hn_data)

            cnt_enc_inp = torch.cat((
                enc_out,
                hn_context.unsqueeze(1).repeat(1,self.mp*self.pd,1)
            ), dim =2)                 

            var_encs = self.cnt_enc_net(cnt_enc_inp)            

            flat_var_encs = var_encs.view(-1, self.vd)                        

            # UNIQUE STUFF
            
            uinds, _ = self.get_unique_info(prims[0,:nprims].flatten())
            
            u_encs = var_encs[0,uinds]
                                                                
            u_encs /= u_encs.norm(dim=1).unsqueeze(-1) + 1e-8
            
            self.EVAL_u_encs = u_encs.cpu()
            self.EVAL_uinds = uinds.cpu()            

        elif 'hypheadabl' in self.mode or 'nocontext' in self.mode or 'pointer' in self.mode: 
            with torch.no_grad():
                hn_data, hn_context, hn_num = self.get_hn_info(prims, num_prims)
                
            self.EVAL_hn_data = hn_data
            self.EVAL_hn_context = hn_context
            self.EVAL_hn_num = hn_num

            self.EVAL_sing_hn_num = hn_num[0].item()
            
            m = {}
            
            _hn_data = hn_data[0]

            for i in range(hn_num[0].item()):
                var = (_hn_data == i).nonzero().flatten()[0].item()
                I = int(var / self.pd)
                J = int(var % self.pd)
                m[i] = (I, J)

            self.EVAL_hn_map = m

            if 'hypheadabl' in self.mode:            
                flat_var_encs = self.var_enc_net(hn_data, hn_context)

            elif 'nocontext' in self.mode:
                var_encs = self.var_enc_net(hn_data)
                flat_var_encs = var_encs.view(-1, self.vd)

            elif 'pointer' in self.mode:
                flat_var_encs = self.var_enc_net(hn_data, hn_context, hn_num)
                
            var_encs = flat_var_encs.view(self.bs, self.mp * self.pd, self.vd)[0]

            uinds, _ = self.get_unique_info(prims[0,:nprims].flatten())
            
            u_encs = var_encs[uinds]        

            # Find unique values of the input primitives, record this information
            
            self.EVAL_u_encs = u_encs.cpu()
            self.EVAL_uinds = uinds.cpu()
                        
        else:
            flat_vars = prims.flatten().unsqueeze(-1)
            flat_var_encs = self.var_enc_net(flat_vars)
            var_encs = flat_var_encs.view(self.bs, self.mp * self.pd, self.vd)[0]         
            uinds, _ = self.get_unique_info(prims[0,:nprims].flatten())
            
            u_encs = var_encs[uinds]
                                                                
            u_encs /= u_encs.norm(dim=1).unsqueeze(-1) + 1e-8
            
            self.EVAL_u_encs = u_encs.cpu()
            self.EVAL_uinds = uinds.cpu()            
            
            
        pe_flat_var_encs = flat_var_encs + self.var_pos_enc(
            self.var_pos_arange[:flat_var_encs.shape[0]].to(self.device)
        )                

        if 'enc:habl' in self.mode:
            lc_inp = hn_context.unsqueeze(1).repeat(1,self.mp * self.pd, 1)            
            const_enc = self.cnt_learn_const(lc_inp).view(-1, self.hd)            

        else:        
            const_enc = self.learn_enc_const(
                self.learn_enc_ind[:flat_var_encs.shape[0]].to(device)
            )
                
        flat_prim_encs = torch.cat((            
            const_enc,
            pe_flat_var_encs
        ), dim = 1)                
            
        self.EVAL_flat_var_encs = flat_var_encs
        
        self.EVAL_var_encs = var_encs

        self.EVAL_prim_encs = flat_prim_encs.view(self.bs, self.mp * self.pd, self.vd + self.hd)
            
        self.EVAL_token_pos_emb = self.token_pos_enc(self.token_pos_arange.to(self.device))

        self.EVAL_key_mask = self.generate_key_mask(num_prims).to(self.device)
        
        self.EVAL_attn_mask = self.attn_mask.to(self.device)
            

    # get context info from primitives
    def get_hn_info(self, prims, num_prims):

        hn_data = torch.zeros(prims.shape[0], self.mp * self.pd, device = prims.device).long()
        hn_context = torch.zeros(prims.shape[0], self.mp * self.pd, device = prims.device).float()
        hn_num = torch.zeros(prims.shape[0]).long()
        
        for i, (_prims, _num_prims) in enumerate(zip(prims, num_prims)):
            uvals = _prims[:_num_prims].unique().sort().values
            hn_context[i,:uvals.shape[0]] = uvals
            hn_num[i] = uvals.shape[0]
            
            m = {uv.item():j for j,uv in enumerate(uvals)}            
            
            for j, uv in enumerate(_prims[:_num_prims].flatten()):
                hn_data[i, j] = m[uv.item()]
                        
        #hn_data -> B X (MP * PD) [where each values is between 0 and #U]
        #hn_context -> B X (MP * PD) [where the first #U values are sorted unique]
        
        return hn_data, hn_context, hn_num
    

    # attn mask
    def generate_attn_mask(self):
        sz = (self.mp * self.pd) + self.ms
        mask = (torch.triu(torch.ones(sz, sz)) == 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).T
        mask[:(self.mp * self.pd), :(self.mp * self.pd)] = 0.
        return mask

    # key mask
    def generate_key_mask(self, num_prims):
        sz = (self.mp * self.pd) + self.ms
        mask = torch.zeros(len(num_prims), sz).bool()
        for i,np in enumerate(num_prims):
            mask[i,(np * self.pd):(self.mp * self.pd)] = True

        return mask

    # flatten seq_to_prim sequences into two lists of indices
    def get_flat_inds(self, seq_to_prims):
        sp = []
        vp = []
        
        for b, s2p in enumerate(seq_to_prims):
            for s,p,v in s2p:
                sp.append((b * self.ms) + s)
                vp.append((b * self.mp * self.pd) + (p * self.pd) + v) 
                
        sp = torch.tensor(sp, device=self.device).long()
        vp = torch.tensor(vp, device=self.device).long()

        return sp, vp
        
                
    # prims: B X MP X PD , floats
    # seq: B X MS , long
    # num_prims: B -> number of primitives for each shape , list of ints
    # seq_to_prims: B size list of  [MS -> index into associated MP X PD map if its a float variable]

    def forward(self, prims, seq, num_prims, seq_to_prims, hn_data, hn_context, hn_num):
        
        bs = prims.shape[0]
        
        if 'hyper' in self.mode:

            flat_var_encs = self.var_enc_net(hn_data, hn_context)
            var_encs = (hn_data, hn_context, hn_num)

        elif 'enc:habl' in self.mode:
                
            enc_out = self.var_enc_net(hn_data)

            cnt_enc_inp = torch.cat((
                enc_out,
                hn_context.unsqueeze(1).repeat(1,self.mp*self.pd,1)
            ), dim =2)                 

            var_encs = self.cnt_enc_net(cnt_enc_inp)            

            flat_var_encs = var_encs.view(-1, self.vd)

            if 'head:ind' in self.mode:
                var_encs = (hn_data, hn_context, hn_num)

        elif 'hypheadabl' in self.mode:
            flat_var_encs = self.var_enc_net(hn_data, hn_context)
            var_encs = flat_var_encs.view(bs, self.mp * self.pd, self.vd)

        elif 'nocontext' in self.mode:
            # encode the scene information
            var_encs = self.var_enc_net(hn_data)
            flat_var_encs = var_encs.view(-1, self.vd)
            var_encs = (hn_data, hn_context, hn_num)
            
        elif 'pointer' in self.mode:
            flat_var_encs = self.var_enc_net(hn_data, hn_context, hn_num)
            var_encs = flat_var_encs.view(bs, self.mp * self.pd, self.vd)
            
        else:
            flat_vars = prims.flatten().unsqueeze(-1)
            flat_var_encs = self.var_enc_net(flat_vars)
            var_encs = flat_var_encs.view(bs, self.mp * self.pd, self.vd)

        # add positional encoding
        pe_flat_var_encs = flat_var_encs + self.var_pos_enc(
            self.var_pos_arange[:flat_var_encs.shape[0]].to(self.device)
        )
        

        if 'enc:habl' in self.mode:
            lc_inp = hn_context.unsqueeze(1).repeat(1,self.mp * self.pd, 1)            
            const_enc = self.cnt_learn_const(lc_inp).view(-1, self.hd)            

        else:        
            const_enc = self.learn_enc_const(
                self.learn_enc_ind[:flat_var_encs.shape[0]].to(device)
            )
                
        flat_prim_encs = torch.cat((            
            const_enc,
            pe_flat_var_encs
        ), dim = 1)
        
        prim_encs = flat_prim_encs.view(bs, self.mp * self.pd, self.vd + self.hd)
                
        token_encs = self.token_enc_net(seq)        
        flat_token_encs = token_encs.view(-1, self.hd)

        flat_seq_pos_inds, flat_var_pos_inds = self.get_flat_inds(seq_to_prims)
        
        flat_tvar_encs = torch.zeros(bs * self.ms, self.vd, device = device).float()                
        flat_tvar_encs[flat_seq_pos_inds] += flat_var_encs[flat_var_pos_inds]
        
        token_encs = torch.cat((flat_token_encs, flat_tvar_encs), dim = 1).view(
            bs, self.ms, self.hd+self.vd
        )
                                        
        token_encs = token_encs + self.token_pos_enc(self.token_pos_arange[:bs].to(self.device))

        # input sequence
        out = torch.cat((prim_encs, token_encs), dim = 1)

        key_mask = self.generate_key_mask(num_prims).to(self.device)
        attn_mask = self.attn_mask.to(self.device)        

        # run attention layers
        for attn_layer in self.attn_layers:            
            out = attn_layer(out, attn_mask, key_mask)

        # get final layer output
            
        seq_out = out[:,(self.mp * self.pd):,:]

        # get prediction from transformer output
        token_out = self.token_head(seq_out)
        
        if 'hyper' in self.mode:
            var_out = self.var_head(seq_out, hn_context)
        else:                    
            var_out = self.var_head(seq_out)
    
        return token_out, var_out, var_encs


    def eval_forward(self, prims, seq, num_prims, seq_to_prims):

        # same as train logic, but using eval saved values
        
        flat_var_encs = self.EVAL_flat_var_encs                                        
        prim_encs = self.EVAL_prim_encs
                
        token_encs = self.token_enc_net(seq)        
        flat_token_encs = token_encs.view(-1, self.hd)

        flat_seq_pos_inds, flat_var_pos_inds = self.get_flat_inds(seq_to_prims)
        
        flat_tvar_encs = torch.zeros(self.bs * self.ms, self.vd, device = device).float()                
        flat_tvar_encs[flat_seq_pos_inds] += flat_var_encs[flat_var_pos_inds]
        
        token_encs = torch.cat((flat_token_encs, flat_tvar_encs), dim = 1).view(
            self.bs, self.ms, self.hd+self.vd
        )
                                        
        token_encs = token_encs + self.EVAL_token_pos_emb
        
        out = torch.cat((prim_encs, token_encs), dim = 1)
            
        for attn_layer in self.attn_layers:        
            out = attn_layer(out, self.EVAL_attn_mask, self.EVAL_key_mask)
        
        seq_out = out[:,(self.mp * self.pd):,:]
                
        token_out = self.token_head(seq_out)
                
        if 'hyper' in self.mode:
            var_out = self.var_head.eval_forward(seq_out)
        else:                    
            var_out = self.var_head(seq_out)
            
        return token_out.cpu(), var_out.cpu()

    def get_unique_info(self, vals):
        m = {}

        uinds = []
        
        for i,v in enumerate(vals):
            if v.item() not in m:
                m[v.item()] = len(m)
                uinds.append(i)

        return torch.tensor(uinds,device=vals.device).long(), m
    
    def var_loss_info_contrast(self, s2p, vals):
        
        uinds, m = self.get_unique_info(vals)

        bvar_mask = torch.zeros(self.ms, len(uinds), device=vals.device)
        margin_mask = torch.zeros(self.ms, len(uinds), device=vals.device)

        for s,p,v in s2p:
            bvar_mask[s-1] = 1.0
            val = vals[(self.pd * p) + v]            
            margin_mask[s-1, m[val.item()]] = 1.0

        return uinds, bvar_mask, margin_mask

    def var_loss_info_ce(self, s2p, vals):

        uinds, m = self.get_unique_info(vals)

        mask = torch.zeros(self.ms, device=vals.device)
        targets = torch.zeros(self.ms, device=vals.device).long()

        for s,p,v in s2p:
            mask[s-1] = 1.0
            val = vals[(self.pd * p) + v]            
            targets[s-1] = m[val.item()]

        return uinds, mask, targets

    def var_loss_info_hyper(self, s2p, hnd):
        
        mask = torch.zeros(self.ms, device=hnd.device)
        targets = torch.zeros(self.ms, device=hnd.device).long()

        for s,p,v in s2p:
            mask[s-1] = 1.0
            targets[s-1] = hnd[(self.pd * p) + v]            

        return mask, targets
            

    def var_loss_ce(self, var_pred, var_encs, num_prims, seq_to_prims, prims):
        var_loss = 0.
        var_loss_norm = 0.
        res = {'var_corr':0., 'var_total':0.}
        
        for bvar_pred, bvar_enc, s2p, np, _bprims in zip(var_pred, var_encs, seq_to_prims, num_prims, prims):

            # get uinds 
            # mask : 32 -> 1.0 if pred matters else 0.0
            # targets: 32 -> IND
            with torch.no_grad():
                uinds, mask, targets = self.var_loss_info_ce(s2p, _bprims[:np].flatten())
                
            # U x 16
            u_encs = bvar_enc[uinds]

            # bvar_pred 32 x 16
            
            var_q = bvar_pred
            var_k = u_encs
            
            nvar_q = var_q / (var_q.norm(dim=1).unsqueeze(-1) + 1e-8)
            nvar_k = var_k / (var_k.norm(dim=1).unsqueeze(-1) + 1e-8)
            
            dot = (nvar_q @ nvar_k.T) * 4.
            
            with torch.no_grad():
                res['var_corr'] += ((dot.argmax(dim=1) == targets) * mask).sum().item()
                res['var_total'] += mask.sum().item()
                    
            var_loss += (self.celoss(dot, targets) * mask).sum() / mask.sum()
            var_loss_norm += 1.0
            
        total_loss = var_loss / var_loss_norm
            
        res['var_loss'] = total_loss.item()
        
        return total_loss, res


    def var_loss_pointer(self, var_pred, var_encs, num_prims, seq_to_prims, prims):
        var_loss = 0.
        var_loss_norm = 0.
        res = {'var_corr':0., 'var_total':0.}
        
        for bvar_pred, bvar_enc, s2p, np, _bprims in zip(var_pred, var_encs, seq_to_prims, num_prims, prims):

            # get uinds 
            # mask : 32 -> 1.0 if pred matters else 0.0
            # targets: 32 -> IND
            with torch.no_grad():
                uinds, mask, targets = self.var_loss_info_ce(s2p, _bprims[:np].flatten())
                
            # U x 16
            u_encs = bvar_enc[uinds]
            
            # bvar_pred 32 x 16
            
            var_q = bvar_pred
            var_k = u_encs
                        
            dot = (var_q @ var_k.T)
            
            with torch.no_grad():
                res['var_corr'] += ((dot.argmax(dim=1) == targets) * mask).sum().item()
                res['var_total'] += mask.sum().item()
                    
            var_loss += (self.celoss(dot, targets) * mask).sum() / mask.sum()
            var_loss_norm += 1.0
            
        total_loss = var_loss / var_loss_norm
            
        res['var_loss'] = total_loss.item()
        
        return total_loss, res
            
    def var_loss_contrast(self, var_pred, var_encs, num_prims, seq_to_prims, margin, prims):
        var_loss = 0.
        var_loss_norm = 0.
        res = {'var_corr':0., 'var_total':0.}
        
        for bvar_pred, bvar_enc, s2p, np, _bprims in zip(var_pred, var_encs, seq_to_prims, num_prims, prims):

            # PREDS -> 32 x 16
            # ENCS -> 48 x 16

            # get U = # unique values + uinds so that
            # bvar_mask : 32 X U -> 1.0 if pred matters else 0.0
            # margin_mask: 32 X U -> 1.0 if supposd to be close
            with torch.no_grad():
                #bprims = _bprims[:np].flatten()            
                #qinds, kinds, targets = self.get_uniq_var_inds(s2p, bprims)

                uinds, bvar_mask, margin_mask = self.var_loss_info_contrast(s2p, _bprims[:np].flatten())
                
            # U x 16
            u_encs = bvar_enc[uinds]
            
            # 32 X U 
            D = ((bvar_pred.unsqueeze(1) - u_encs.unsqueeze(0)).abs() + 1e-8).norm(dim=2)
            
            raw_loss = (margin_mask * D) + ((1. - margin_mask) * torch.relu(margin - D))            
            
            var_loss += (raw_loss * bvar_mask).sum() / bvar_mask.sum()

            with torch.no_grad():
                _preds = D < (margin / 2)                
                res['var_corr'] += ((_preds == margin_mask).float() * bvar_mask).sum().item()
                res['var_total'] += bvar_mask.sum().item()
                
            var_loss_norm += 1.0

        total_loss = var_loss / var_loss_norm
            
        res['var_loss'] = total_loss.item()
        
        return total_loss, res


    def var_loss_hyper(self, var_pred, var_encs, seq_to_prims):
        var_loss = 0.
        var_loss_norm = 0.
        res = {'var_corr':0., 'var_total':0.}

        hn_data, hn_context, hn_num = var_encs
        
        for bvar_pred, bhn_data, bhn_context, bhn_num, s2p in zip(
                var_pred, hn_data, hn_context, hn_num, seq_to_prims
        ):

            with torch.no_grad():
                mask, targets = self.var_loss_info_hyper(s2p, bhn_data)
                
            pred = bvar_pred[:, :bhn_num.item()]
            
            with torch.no_grad():
                res['var_corr'] += ((pred.argmax(dim=1) == targets) * mask).sum().item()
                res['var_total'] += mask.sum().item()

            # do cross entropy from preds to targets, where mask indicates if the slot is real
            var_loss += (self.celoss(pred, targets) * mask).sum() / mask.sum()
            var_loss_norm += 1.0
            
        total_loss = var_loss / var_loss_norm
            
        res['var_loss'] = total_loss.item()
        
        return total_loss, res
    
    def loss(self, token_pred, var_pred, var_encs, seqs, seq_lens, num_prims, seq_to_prims, margin, prims):
        
        if 'loss:contrast' in self.mode:
            var_loss, res = self.var_loss_contrast(var_pred, var_encs, num_prims, seq_to_prims, margin, prims)
            
        elif 'loss:ce' in self.mode:
            var_loss, res = self.var_loss_ce(var_pred, var_encs, num_prims, seq_to_prims, prims)

        elif 'hyper' in self.mode or 'head:ind' in self.mode or 'nocontext' in self.mode:
            var_loss, res = self.var_loss_hyper(var_pred, var_encs, seq_to_prims)

        elif 'hypheadabl' in self.mode or 'pointer' in self.mode:
            var_loss, res = self.var_loss_pointer(var_pred, var_encs, num_prims, seq_to_prims, prims)        
            
        else:
            assert False, f'bad mode {self.mode}'
            
        # TOKEN BASED LOSS_LOGIC
        
        flat_token_preds = token_pred[:,:-1,:].reshape(-1, token_pred.shape[2])
        flat_targets = seqs[:,1:].flatten()
        token_mask = (flat_targets != 0.).float()

        assert int(token_mask.sum().item()) == sum(seq_lens).item()
                
        token_loss = (self.celoss(flat_token_preds, flat_targets) * token_mask).sum() / token_mask.sum()

        token_corr = ((flat_token_preds.argmax(dim=1) == flat_targets).float() * token_mask).sum().item()

        res['token_corr'] = token_corr
        res['token_total'] = token_mask.sum().item()
        res['token_loss'] = token_loss.item()        
        
        return token_loss, var_loss, res

    def batch_expand(self, batch_progs, state):
               
        finished = True

        # make a batch wise prediction, adding a single token to each sequence in batch
        batch_token_preds, batch_var_preds = self.eval_forward(
            state['prims'],
            state['seq'],
            state['num_prims'],
            state['seq_to_prims']
        )        

        # figure out how to handle each predicted sequence
        for i, bp in enumerate(batch_progs):
            next_type = bp.get_next_type()

            if next_type is None:
                # we've finished the sequence, so stop
                continue

            # otherwise need to keep predicting for the whole batch (as long as one sequence is unfinished)
            finished = False
            
            token_pred = batch_token_preds[i, state['seq_ind']]
            var_pred = batch_var_preds[i, state['seq_ind']]            
            
            mtoken_dist = torch.softmax(token_pred, dim=0)
            
            token_mask = state['mask_info'].get_mask_info(i, next_type)
            
            if state['seq_ind'] > 0 and 'token_skip_info' in state:
                token_dist = mtoken_dist * token_mask * state['token_skip_info']
            else:                
                token_dist = mtoken_dist * token_mask        

            if token_dist.sum() == 0.:
                bp.stack = []
                continue

            # Sample a token based on output token distribution
            token_ind = torch.distributions.Categorical(token_dist).sample().item()       
            token_name = state['rev_token_map'][token_ind]

            state['seq'][i, state['seq_ind'] + 1] = token_ind            
            
            if token_name == 'float_var':                
                                
                if 'loss:ce' in self.mode:

                    var_pred /= var_pred.norm() + 1e-8
                    
                    u_encs = self.EVAL_u_encs
                    
                    u_dot = (var_pred @ u_encs.T).cpu() * 4

                    u_dist = torch.softmax(u_dot, dim=0)

                    u_var = torch.distributions.Categorical(u_dist).sample().item()
                    var = self.EVAL_uinds[u_var].item()
                    
                    I = int(var / self.pd)
                    J = int(var % self.pd)

                    token_name = f'S_{I}_{J}_'
                    token = library.LibToken(token_name, [], 'Float', 'float_var')
                    state['seq_to_prims'][i].append([state['seq_ind'] + 1, I, J])

                elif 'hypheadabl' in self.mode or 'pointer' in self.mode:
                    
                    u_encs = self.EVAL_u_encs
                    
                    u_dot = (var_pred @ u_encs.T).cpu()                    
                    
                    u_dist = torch.softmax(u_dot, dim=0)

                    u_var = torch.distributions.Categorical(u_dist).sample().item()
                    var = self.EVAL_uinds[u_var].item()
                    
                    I = int(var / self.pd)
                    J = int(var % self.pd)

                    token_name = f'S_{I}_{J}_'
                    token = library.LibToken(token_name, [], 'Float', 'float_var')
                    state['seq_to_prims'][i].append([state['seq_ind'] + 1, I, J])

                elif 'hyper' in self.mode:
                    
                    
                    u_dist = torch.softmax(var_pred[:self.EVAL_sing_hn_num], dim =0 )
                    u_var = torch.distributions.Categorical(u_dist).sample().item()

                    I, J = self.EVAL_hn_map[u_var]
                
                    token_name = f'S_{I}_{J}_'
                    token = library.LibToken(token_name, [], 'Float', 'float_var')
                    state['seq_to_prims'][i].append([state['seq_ind'] + 1, I, J])                    

                elif 'head:ind' in self.mode or 'nocontext' in self.mode:

                    # When eval predicting a float variable, take predicted index and lookup into the unique values
                    
                    u_dist = torch.softmax(var_pred[:self.EVAL_sing_hn_num], dim =0 )
                    u_var = torch.distributions.Categorical(u_dist).sample().item()

                    I, J = self.EVAL_hn_map[u_var]
                
                    token_name = f'S_{I}_{J}_'
                    token = library.LibToken(token_name, [], 'Float', 'float_var')
                    state['seq_to_prims'][i].append([state['seq_ind'] + 1, I, J]) 
                    
                else:
                    assert False, f'bad mode {self.mode}'
                    
            else:
                token = state['L'].getToken(token_name)

            bp.add_token(token)

            state['mask_info'].update_mask_info(i, token_name)
                
        state['seq_ind'] += 1
                
        return finished
