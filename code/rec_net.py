import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os

from tqdm import tqdm
from random import shuffle, random, uniform, choices, choice
import utils
import mc_prog_search as mps
import library
import numpy as np
from utils import device
from mask_info import MaskInfo, DistDummyMaskInfo

from bb_rec_net import BBRecNet, pre_hn_info

def robust_norm(var, dim=2):
    return ((var ** 2).sum(dim=dim) + 1e-8).sqrt()

# A dummy class, to check we've matched primitives
class DummyProg:
    def __init__(self, tar_prims):
        self.P = tar_prims
        self.is_dummy_prog = True
    

# This is a dummy network, used to sample functions during dream phase
class DummyDistNetwork:
    def __init__(self, L, args):

        self.args = args
        self.device = torch.device('cpu')
        self.L = L
        self.ms = args.rn_max_seq
        self.bs = args.rn_batch_size
        
        self.init_with_lib(L)

        self.pd = args.rn_prim_dim
        
        self.a = -1.
        self.b = 1.

    # Init network with parameters for a specific shape
    def init_with_lib(self, L, skip_fns = []):
    
        self.net = self            
        
        self.token_map = {
            'start': 0,
        }
                
        for name, token in L.token_map.items():
            if token.cost_type != 'float_var':
                self.token_map[name] = len(self.token_map)
                
        self.rev_token_map = {v:k for k,v in self.token_map.items()}

        # form masking information
        self.mask_info = DistDummyMaskInfo(self, skip_fns)

    # build a state, to get ready for sampling, based on a function
    def build_state(self, P, fn):            

        for i in range(self.bs):
            self.mask_info.init_mask_info(i, fn.name)
            
        return {
            'mask_info': self.mask_info,
        }

    # expand each member of batch by token
    def batch_expand(self, batch_progs, state):
        with torch.no_grad():
            return self._batch_expand(batch_progs, state)

    # sample the next token for each member of batch
    def _batch_expand(self, batch_progs, state):        
               
        finished = True
        for i, bp in enumerate(batch_progs):

            # get type
            next_type = bp.get_next_type()
            
            if next_type is None:
                continue

            # if one program is not done, we are not done yet
            finished = False

            # sample according to the masking information
            token_name = state['mask_info'].sample_token(i, next_type)                  
            
            if '.' in token_name:                
                bp.add_token(library.LibToken(token_name, [], 'Float', 'float_const'))
            else:                
                bp.add_token(self.L.getToken(token_name))                        
            
            
        return finished

# Helper function to convert primitives into a tensor
def parse_prims(prims, N):

    d = torch.zeros(N.args.rn_max_prims, N.args.rn_prim_dim).float()
    
    for i, (_,prim) in enumerate(prims):
        # if we have too many primitives, versus what the network can consume,
        # just break early (we'll cover them in subsequent step)
        if i >= N.args.rn_max_prims:
            break
        
        for j, e in enumerate(prim):
            d[i][j] = float(e)

    return d            


# Data loader class
class Dataset:
    def __init__(
        self, dreams, N, batch_size, num_to_eval, net_mode, device
    ):

        self.device = device
        self.mode = 'train'
        self.net_mode = net_mode
        
        self.prims = []
        self.seqs = []
        self.num_prims = []
        self.seq_to_prims = []
        self.seq_lens = []
        self.hn_data = []
        self.hn_context = []
        self.hn_num = []
        
        self.batch_size = batch_size

        valid_dreams = []

        # convert dreams into training data
        for dream in dreams:

            text, prims, pinds = dream

            seq, seq_to_prims, seq_len = self.parse_prog(text, N)
                        
            if seq is None:
                continue
            
            self.seq_to_prims.append(seq_to_prims)
                
            pprims = parse_prims(prims, N)

            self.num_prims.append(len(prims))            
            self.prims.append(pprims)                                    
            self.seqs.append(seq)            
            self.seq_lens.append(seq_len)

            valid_dreams.append(dream)

            _hn_data, _hn_context, _hn_num = pre_hn_info(N.args, pprims, len(prims))
            self.hn_data.append(_hn_data)
            self.hn_context.append(_hn_context)
            self.hn_num.append(_hn_num)
            
        self.prims = torch.stack(self.prims, dim =0)
        self.seqs = torch.stack(self.seqs, dim =0)
        self.num_prims = torch.tensor(self.num_prims)
        self.seq_lens = torch.tensor(self.seq_lens)
        self.dreams = valid_dreams

        self.hn_data = torch.stack(self.hn_data, dim=0).long()
        self.hn_context = torch.stack(self.hn_context, dim=0).float()
        self.hn_num = torch.tensor(self.hn_num).long()
        
        self.num_to_eval = min(num_to_eval, len(self.dreams)) \
                           if num_to_eval > 0 else len(self.dreams)

    # parse text into tokens
    def parse_tokens(self, text):
        tokens = []
        c = ''
        for s in text:
            if s == '(' or s == ')' or s == ',':
                if c.strip() != '':
                    tokens.append(c.strip())
                c = ''                
            else:
                c += s

        return tokens

    # parse a program into a sequence of tokens, how the float vars map to the input
    # primitives, and how long the sequence is
    def parse_prog(self, text, N):
        
        d = torch.zeros(N.args.rn_max_seq).long()
        seq_to_prims = []

        d[0] = N.token_map['start']
        
        tokens = self.parse_tokens(text)

        if len(tokens) > (N.args.rn_max_seq - 1):
            return None, None, None
        
        for i, t in enumerate(tokens):
            if t in N.token_map:
                d[i+1] = N.token_map[t]
            else:
                r = t.split('_')
                assert len(r) == 4

                d[i+1] = N.token_map['float_var']
                seq_to_prims.append([i+1, int(r[1]), int(r[2])])
                        
        return d, seq_to_prims, i+1        

    # iterator 
    def __iter__(self):
        if self.mode == 'train':
            yield from self.train_iter()
        elif self.mode == 'eval':
            yield from self.eval_iter()
        


    # create a training batch
    def train_iter(self):
        inds = torch.randperm(self.prims.shape[0])
        
        for i in range(0, self.prims.shape[0], self.batch_size):        
            with torch.no_grad():
                binds = inds[i:i+self.batch_size] # sample a set of inds

                bprims = self.prims[binds].to(self.device) # input primitives
                bseqs = self.seqs[binds].to(self.device) # target sequences
                bnum_prims = self.num_prims[binds].tolist() # number of primitives
                bseq_lens = self.seq_lens[binds].to(self.device) # sequence lengths
                bseq_to_prims = [self.seq_to_prims[j] for j in binds] # mapping seq floats to prim values

                bhn_data = self.hn_data[binds].to(self.device) # parameter data (prediction)
                bhn_context = self.hn_context[binds].to(self.device) # paramter data (input)
                bhn_num = self.hn_num[binds].to(self.device) # number of unique parameters
                
            yield bprims, bseqs, bnum_prims, bseq_to_prims, bseq_lens, bhn_data, bhn_context, bhn_num

    # during evaluation, what we return for iterator
    def eval_iter(self):
        inds = list(range(self.num_to_eval))
        for i in inds:
            yield self.dreams[i], self.prims[i], self.num_prims[i]

# Recognition network class
class RecNet:
    def __init__(self, L, args):

        self.args = args
        # account for extra fn token
        
        self.L = L
        self.device = device

        # create mapping from tokens to indices
        
        self.token_map = {
            'start': 0,
            'float_var': 1,            
        }                
        for name, token in L.token_map.items():
            if token.cost_type != 'float_var':
                self.token_map[name] = len(self.token_map)

        self.rev_token_map = {v:k for k,v in self.token_map.items()}                

        # batch size
        self.bs = args.rn_batch_size                

        # backbone
        self.net = BBRecNet(
            args,
            len(self.token_map.keys())
        )

        # mask information, used during evaluation to make sure typing is correct
        self.mask_info = MaskInfo(self.L, self.token_map, self.args)

    # begin to prep state of network, by parsing primitives
    def build_net_state(self, P):
        nprims = len(P)                
        net = self.net
        tprims = parse_prims(P, self)

        net.to(self.device)        
        net.build_state(tprims, nprims)

    # initialize network for evaluation based on primitives
    def build_state(self, P):

        nprims = len(P)
                
        net = self.net

        tprims = parse_prims(P, self)
        
        prims = tprims.unsqueeze(0).repeat(net.bs, 1, 1).to(net.device)
        num_prims = [nprims for _ in range(net.bs)]
        seq = torch.zeros(net.bs, net.ms).long().to(net.device)

        seq[:,0] = self.token_map['start']        
        
        seq_to_prims = [[] for _ in range(net.bs)]

        self.mask_info.reset()

        # for first prediction, predict a shape creating function
        skip_info = torch.ones(len(self.token_map))
        for t in self.L.getWakeSkipFns():
            token_ind = self.token_map[t.name]
            skip_info[token_ind] = 0.

        state = {
            'prims': prims,
            'seq': seq,
            'seq_to_prims': seq_to_prims,
            'num_prims': num_prims,
            'seq_ind': 0,
            'rev_token_map': self.rev_token_map,

            'mask_info': self.mask_info, 
            'L': self.L,
            'token_skip_info': skip_info
        }                
        
        return state

    # Make data loaders based on dreams
    def make_dataset(self, dreams):

        all_train_dreams = []
        all_val_dreams = []

        # create artifical split beween train / val for each fn
        for fn, fn_dreams in dreams.items():
            
            shuffle(fn_dreams)
            num_train = int((1 - self.args.rn_holdout) * len(fn_dreams))
                        
            train_dreams = fn_dreams[:num_train]
            val_dreams = fn_dreams[num_train:]

            all_train_dreams += train_dreams
            all_val_dreams += val_dreams

        shuffle(all_train_dreams)
        shuffle(all_val_dreams)

        # train and validation loaders
        train_loader = Dataset(
            all_train_dreams, self, self.args.rn_batch_size, self.args.rn_num_to_eval,
            self.args.rn_mode, self.device
        )
        val_loader = Dataset(
            all_val_dreams, self, self.args.rn_batch_size, self.args.rn_num_to_eval,
            self.args.rn_mode, self.device
        )

        utils.log_print(f"Info: Num Train {len(train_loader.dreams)} Num Val {len(val_loader.dreams)}", self.args, fn='w_log')
            
        return {'train': train_loader, 'val': val_loader}

    # load the best network version, based on validation, from the training phase
    def load_best(self, load_path=None):    
        if load_path is None:
            load_path = self.args.save_net_path
        
        weights = torch.load(
            f'{load_path}/rec_net.pt'
        )        
        self.net.load_state_dict(weights)
        self.net.to(self.device)

    # entrypoint into training the network
    def train(self, loader):                    
        utils.log_print(f"Training Rec Net", self.args)
        train_loader = loader['train']
        val_loader = loader['val']

        self.train_net(self.net, train_loader, val_loader, self.train_fn_batch)

    # training logic for a given batch
    def train_fn_batch(self, net, batch, opt):

        # unpack batch
        prims, seqs, num_prims, seq_to_prims, seq_lens, hn_data, hn_context, hn_num = batch

        # make network predictions
        token_pred, var_pred, var_encs = net(
            prims, seqs, num_prims, seq_to_prims, hn_data, hn_context, hn_num
        )

        # calculate loss
        token_loss, var_loss, loss_res = net.loss(
            token_pred, var_pred, var_encs, seqs,
            seq_lens, num_prims, seq_to_prims, self.args.rn_margin, prims
        )
        
        loss = token_loss + var_loss

        # make update
        
        if opt is not None:
            opt.zero_grad()
            loss.backward()
            opt.step()
            
        loss_res['loss'] = loss.item()
    
        return loss_res

    # complete a training epoch
    def train_epoch(self, net, loader, opt, batch_fn):
        if opt is None:
            net.eval()
        else:
            net.train()
        
        ep_result = {}
        bc = 0.
        
        for batch in loader:
            bc += 1.

            batch_result = batch_fn(net, batch, opt)
        
            for key in batch_result:                        
                if key not in ep_result:                    
                    ep_result[key] = batch_result[key]
                else:
                    ep_result[key] += batch_result[key]

                
        ep_result['batch_count'] = bc

        return ep_result

    # run an evaluation step for a given shape
    def eval_batch(self, net, batch):

        # targets
        (text, P, pinds), tprims, nprims = batch

        # create an identical batch, for a single shape
        prims = tprims.unsqueeze(0).repeat(net.bs, 1, 1).to(net.device)
        num_prims = [nprims.item() for _ in range(net.bs)]
        seq = torch.zeros(net.bs, net.ms).long().to(net.device)

        seq[:,0] = self.token_map['start']
        
        seq_to_prims = [[] for _ in range(net.bs)]

        self.build_net_state(P)
        state = self.build_state(P)        

        # sample a set of programs conditioned on the target primitives of the shape
        samp_progs = mps.batch_mc_sample(
            self,
            P,
            state
        )
        self.L.init_lib()
        self.L.add_prims(P)

        mps.update_prog_info(self.L, samp_progs, self.args)
                
        def fill_text(t):
            for i, prim in P:
                for j, v in enumerate(prim):
                    t = t.replace(f'S_{i}_{j}_', str(v))
            return t

        tar_text = fill_text(text)

        prog_match = False        
        shape_match = False
        moved_shape_match = False

        # for each sampled program
        for s in samp_progs:
            p_text = fill_text(s.text)

            # check if we recreated text exactly
            if p_text == tar_text:
                prog_match = True
                shape_match = True
                moved_shape_match = True
                break

        tar_prims = [P[i] for i in pinds]

        # otherwise check if we found a shape match
        for s in samp_progs:            
            
            d = DummyProg(tar_prims)
            vmap = {}
            b_text = fill_text(s.text)

            # See if match required an alignment
            if mps.calc_match(self.L, d, s, self.args):
                a_text = fill_text(s.text)                
                moved_shape_match = True                    

                if b_text == a_text:
                    shape_match = True
                    break
                
        self.L.init_lib()
            
        res = {}
        res['prog_match'] = float(prog_match)
        res['shape_match'] = float(shape_match)
        res['moved_shape_match'] = float(moved_shape_match)
        
        return res

    # Epcoh of evaluation logic
    def eval_epoch(self, net, loader):
        ep_result = {}

        bc = 0.
        for batch in tqdm(loader, total=loader.num_to_eval):
            bc += 1.

            batch_result = self.eval_batch(net, batch)

            for key in batch_result:                        
                if key not in ep_result:                    
                    ep_result[key] = batch_result[key]
                else:
                    ep_result[key] += batch_result[key]

                
        ep_result['batch_count'] = bc
        return ep_result

    # print evaluation results
    def print_res(self, train_res, val_res, e, t, LOG_INFO):                
        utils.log_print(f"Train Epoch {e} Results: ", self.args, fn='w_log')
        utils.print_results(LOG_INFO, train_res, self.args)
        utils.log_print(f"Val Epoch {e} Results: ", self.args, fn='w_log')
        utils.print_results(LOG_INFO, val_res, self.args)

    # training logic, outer level
    def train_net(self, net, train_loader, val_loader, batch_fn, has_eval_mode = True):
        
        net.to(net.device)

        # optomizer
        opt = torch.optim.Adam(
            net.parameters(),
            lr = self.args.rn_lr,
            eps = 1e-6
        )

        # results
        res = {
            'train_epochs': [],
            'eval_epochs': [],
            'train_plots': {
                'train': {},
                'val': {}
            },
            'eval_plots': {
                'train': {},
                'val': {}
            }
        }

        best_epoch = 0
        best_loss_val = 1e8

        ST = time.time()

        # for each epoch
        for e in range(self.args.rn_epochs):

            # early stop
            if (e - best_epoch > self.args.rn_patience) or\
               (time.time() - ST) > self.args.rn_timeout :
                
                utils.log_print(f"Stopping at Epoch {e}", self.args, fn='w_log')                
                break
                
            t = time.time()

            train_loader.mode = 'train'
            val_loader.mode = 'train'

            # do a train epoch
            et_train_res = self.train_epoch(net, train_loader, opt, batch_fn)

            with torch.no_grad():
                # do a train epoch on validation
                et_val_res = self.train_epoch(net, val_loader, None, batch_fn)

                # use val loss for early stopping and best network logic
                if et_val_res['loss'] < best_loss_val - self.args.rn_threshold:
                    best_epoch = e
                    best_loss_val = et_val_res['loss']
                    torch.save(
                        net.state_dict(),                        
                        self.args.save_net_path + f'/rec_net.pt'
                    )
                    
            
            ep_result = {
                'train': et_train_res,
                'val': et_val_res
            }

            do_print = (e+1) % self.args.rn_print_per == 0

            # print training results
            if do_print:
                utils.log_print(f"Epoch {e} took {time.time() - t}", self.args, fn='w_log')
                self.print_res(et_train_res, et_val_res, e, t, utils.TRAIN_LOG_INFO)                       
            res['train_epochs'].append(e)

            # update training plots
            utils.make_plots(
                utils.TRAIN_LOG_INFO,
                ep_result,
                res['train_plots'],
                res['train_epochs'],
                self.args,
                'train_plots',
                do_print
            )
            
            do_eval = (e+1) % self.args.rn_eval_per == 0

            # whether to run evaluation
            if do_eval and has_eval_mode:
                                
                with torch.no_grad():
                    t = time.time()
                    
                    res['eval_epochs'].append(e)

                    train_loader.mode = 'eval'
                    val_loader.mode = 'eval'

                    net.eval() 

                    # run evaluation for both train and val
                    ev_train_res = self.eval_epoch(net, train_loader)
                    ev_val_res = self.eval_epoch(net, val_loader) 

                    utils.log_print(f"Eval {e} took {time.time() - t}", self.args, fn='w_log')
                    self.print_res(ev_train_res, ev_val_res, e, t, utils.EVAL_LOG_INFO)             
                    eval_ep_result = {
                        'train': ev_train_res,
                        'val': ev_val_res
                    }

                    # make validation plots
                    utils.make_plots(
                        utils.EVAL_LOG_INFO,
                        eval_ep_result,
                        res['eval_plots'],
                        res['eval_epochs'],
                        self.args,
                        'eval_plots',
                        True
                    )
