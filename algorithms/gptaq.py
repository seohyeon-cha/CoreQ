"""GPTQ with a closed-form cross-layer correction term.

In addition to the calibration Hessian ``H = X X^T`` (collected from the
*quantized* forward pass), this method also stores the cross statistic
    ``dXXT = (X_fp - X_q) X^T``,
where ``X_fp`` are the activations from the full-precision path
(``self.fp_inp``, populated in ``llama_step.py``). The corrected target is
    ``W' = W + alpha * dXXT @ H^{-1}``,
which is then rounded with a GPTQ-style sequential sweep. ``alpha`` is a
scalar that interpolates between no correction (``alpha=0``) and full
correction (``alpha=1``).
"""
import math
import time

import torch
import torch.nn as nn
import transformers
import quant
from texttable import Texttable
from utils import torch_snr_error
import utils.quip_utils as quip_utils

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class Observer:

    def __init__(self, topk=32):
        self.loss_list = []
        self.topk = topk

    def submit(self, name: str, layerid: int, gptq, error: float):

        item = (name, layerid, {'gptq': gptq, 'error': error})

        if len(self.loss_list) < self.topk:
            self.loss_list.append(item)
            return

        min_error = error
        min_idx = -1
        for idx, data in enumerate(self.loss_list):
            if min_error > data[2]['error']:
                min_idx = idx
                min_error = data[2]['error']

        if min_idx >= 0:
            self.loss_list[min_idx] = item

    def print(self):
        self.loss_list = sorted(self.loss_list, key=lambda s: s[2]['error'], reverse=True)

        table = Texttable()

        table.header(['name', 'error'])
        table.set_cols_dtype(['t', 'f'])

        for item in self.loss_list:
            table.add_row([f"{item[0]}.{item[1]}", item[2]['error']])
        print(table.draw())
        print('\n')

    def items(self):
        return self.loss_list


class GPTAQ:

    def __init__(self, layer, observe=False, store_delta_x=False):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.dXXT = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.quantizer = quant.Quantizer()
        self.observe = observe
        self.inps = []
        self.inp1 = None
        self.out1 = None
        self.store_delta_x = store_delta_x
        self.delta_x_values = [] if store_delta_x else None  # Store |deltaX| values for plotting

    def add_batch(self, inp, out):
        if self.observe:
            self.inp1 = inp
            self.out1 = out
        else:
            self.inp1 = None
            self.out1 = None

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))

        inp = inp.t()

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.dXXT *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

        dX = self.fp_inp[0].float() * math.sqrt(2 / self.nsamples) - inp
        self.dXXT += dX.matmul(inp.t())


        # Store |deltaX| for plotting only if enabled: shape is [channels, samples]
        if self.store_delta_x:
            abs_dX = torch.abs(dX)  # |deltaX| per channel
            self.delta_x_values.append(abs_dX.cpu().clone())

        del self.fp_inp[0]


    def print_loss(self, name, q_weight, weight_error, timecost):
        table = Texttable()
        name += ' ' * (16 - len(name))

        table.header(['name', 'weight_error', 'fp_inp_SNR', 'q_inp_SNR', 'time'])

        # assign weight
        self.layer.weight.data = q_weight.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        if self.inp1 is not None:
            # quantize input to int8
            quantizer = quant.Quantizer()
            quantizer.configure(8, perchannel=False, sym=True, mse=False)
            quantizer.find_params(self.inp1)
            q_in = quantizer.quantize(self.inp1).type(torch.float16)
            q_out = self.layer(q_in)

            # get kinds of SNR
            q_SNR = torch_snr_error(q_out, self.out1).item()
            fp_SNR = torch_snr_error(self.layer(self.inp1), self.out1).item()
        else:
            q_SNR = '-'
            fp_SNR = '-'

        table.add_row([name, weight_error, fp_SNR, q_SNR, timecost])
        print(table.draw().split('\n')[-2])

    def fasterquant(self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, name='', alpha=0.25, args=None):
        self.layer.to(self.dev)


        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        H = self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        self.dXXT[:, dead] = 0
        D = self.dXXT.clone()

        if args.incoh_process:
            Hr, Dr, Wr, SU, SV, scaleWH = incoherence_preprocess(W, H, D, args)
        else:
            Hr = H
            Dr = D
            Wr = W
            SU = None
            SV = None
            scaleWH = None
        
        del W, H
        torch.cuda.empty_cache()
        
        if actorder:
            perm = torch.argsort(torch.diag(Hr), descending=True)
            Wr = Wr[:, perm]
            Hr = Hr[perm][:, perm]
            Dr= Dr[perm][:, perm]

        if not self.quantizer.ready():
            self.quantizer.find_params(Wr, weight=True)


        # import pdb; pdb.set_trace()
        damp = percdamp * torch.mean(torch.diag(Hr))
        diag = torch.arange(self.columns, device=self.dev)
        Hr[diag, diag] += damp
        Hr = torch.linalg.cholesky(Hr)
        Hr = torch.cholesky_inverse(Hr)
        Hr = torch.linalg.cholesky(Hr, upper=True)
        Hinv = Hr
        del Hr 
        
        Losses = torch.zeros_like(Wr)
        Q = torch.zeros_like(Wr)

        g_idx = []
        scale = []
        zero = []
        now_idx = 1

        P = alpha * ((Dr @ Hinv.T).triu(diagonal=1)) @ Hinv
        del self.dXXT, Dr

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        torch.cuda.synchronize()
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt   = torch.cuda.Event(enable_timing=True)
        start_evt.record()


        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = Wr[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            P1 = P[i1:i2, i1:i2]

            for i in range(count):

                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if (i1 + i) % groupsize == 0:
                        self.quantizer.find_params(Wr[:, (i1 + i):(i1 + i + groupsize)], weight=True)

                    if ((i1 + i) // groupsize) - now_idx == -1:
                        scale.append(self.quantizer.scale)
                        zero.append(self.quantizer.zero)
                        now_idx += 1

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q)**2 / d**2

                # import pdb; pdb.set_trace()

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0)) - w.unsqueeze(1).matmul(P1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            Wr[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:]) - W1.matmul(P[i1:i2, i2:])


        end_evt.record()
        torch.cuda.synchronize()
        rounding_ms = start_evt.elapsed_time(end_evt)  # milliseconds

        peak_mem_bytes = torch.cuda.max_memory_allocated()
        peak_mem_gb = peak_mem_bytes / (1024**3)

        # Store metrics as instance attributes for logging
        self.rounding_ms = rounding_ms
        self.peak_mem_gb = peak_mem_gb

        error = torch.sum(Losses).item()

        groupsize = groupsize if groupsize != -1 else self.columns
        g_idx = [i // groupsize for i in range(self.columns)]
        g_idx = torch.tensor(g_idx, dtype=torch.int32, device=Q.device)
        
        if actorder:
            invperm = torch.argsort(perm)
            Q = Q[:, invperm]
            g_idx = g_idx[invperm]

        if args.incoh_process:
            Q = incoherence_process(Q, SU, SV, scaleWH, args)
        torch.cuda.synchronize()
        
        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()

        # todo
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )

        self.print_loss(name=name, q_weight=Q, weight_error=error, timecost=(time.time() - tick))

        if scale == []:
            scale.append(self.quantizer.scale)
            zero.append(self.quantizer.zero)
        scale = torch.cat(scale, dim=1)
        zero = torch.cat(zero, dim=1)
        return scale, zero, g_idx, error

    def free(self):
        self.inp1 = None
        self.out1 = None
        self.H = None
        self.dXXT = None 
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()



def RHT_H(H, SU):
    return quip_utils.matmul_hadUt(quip_utils.matmul_hadUt(H * SU).T * SU)


def RHT_W(W, SU, SV):
    return quip_utils.matmul_hadUt(quip_utils.matmul_hadUt(W.T * SV).T * SU)


def incoherence_preprocess(W, H, D, args):
    dtype_ = torch.float32
    device = H.device
    (m, n) = W.shape

    # diagonally rescale W,H to minimize proxy loss
    scaleWH = None
    Wr = W
    Hr = H
    Dr = D 
    if args.rescale_WH:
        Hr = H / H.abs().max()
        diagH = torch.diag(Hr)
        diagW2 = torch.diag(W.T @ W)
        diagH = torch.clamp(diagH, min=1e-8)
        diagW2 = torch.clamp(diagW2, min=1e-8)
        scaleWH = (diagH / diagW2).sqrt().sqrt().to(torch.float32)
        scaleWH = scaleWH.clamp(min=1e-8)
        Wr = Wr * scaleWH[None, :]
        Hr = Hr / scaleWH[None, :]
        Hr = Hr / scaleWH[:, None]
        if D is not None:
            Dr = Dr / scaleWH[None, :]
            Dr = Dr / scaleWH[:, None]
        scaleWH = scaleWH.cpu()

    # randomized hadamard transformation on H, W
    if args.incoh_mode == "had":
        SU = (torch.randn(n, device=device).sign() + 1e-5).sign().to(dtype_)
        SV = (torch.randn(m, device=device).sign() + 1e-5).sign().to(dtype_)
        Hr = RHT_H(Hr, SU)
        if D is not None:
            Dr = RHT_H(Dr, SU).T # transpose since D is not symmetric 
        Wr = RHT_W(Wr, SU, SV)
    
    # randomized kronecker product on H, W
    elif args.incoh_mode == "kron":
        SU = quip_utils.rand_ortho_butterfly_noblock(n).to(dtype_).to(device)
        SV = quip_utils.rand_ortho_butterfly_noblock(m).to(dtype_).to(device)
        Hr = SU @ Hr @ SU.T
        if D is not None:
            Dr = SU @ Dr @ SU.T
        Wr = SV @ Wr @ SU.T
    else:
        raise NotImplementedError
    SV = SV.cpu()
    SU = SU.cpu()

    # Handle dead columns after transformation
    dead = torch.diag(Hr) == 0
    Hr[dead, dead] = 1
    Wr[:, dead] = 0

    Wr = Wr.to(device)

    return Hr, Dr, Wr, SU, SV, scaleWH


def incoherence_process(hatWr, SU, SV, scaleWH, args):
    device = hatWr.device
    # reverse hadamard transformation
    if args.incoh_mode == 'had':
        hatWr = (quip_utils.matmul_hadU(
            (quip_utils.matmul_hadU(hatWr) * SU.to(device)).T) * SV.to(device)).T
    # reverse kronecker product
    elif args.incoh_mode == 'kron':
        hatWr = SV.T.to(device) @ hatWr @ SU.to(device)
    else:
        raise NotImplementedError

    # reverse rescale W,H
    if args.rescale_WH:
        hatWr /= scaleWH[None, :].to(device)

    assert torch.isfinite(hatWr).all()
    return hatWr

