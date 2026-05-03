"""GuidedQuant: row-grouped weighted-Hessian rounding (baseline).

Per-channel rows are partitioned into ``guided_num_groups`` equal blocks.
For each group, a separate Hessian
    ``H_g = sum_i s_{i,g} * x_i x_i^T``
is accumulated, where ``s_{i,g}`` is a precomputed *saliency* weight for
sample ``i`` in group ``g``. Each row block is then quantized with a
GPTQ-style sequential rounding using its own Hessian.

The saliency tensors are produced by an external precomputation pipeline
and loaded from disk via ``--saliency-path`` (one ``l{layer_idx}.pt`` file
per transformer block, mapping linear-module name to a saliency tensor of
shape ``[nsamples, seqlen, guided_num_groups]``).
"""
import math
import time
import pprint
import logging
import torch
import torch.nn as nn
import transformers
import quant
from texttable import Texttable
from utils import torch_snr_error

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class Observer:

    def __init__(self, topk=32):
        self.loss_list = []
        self.topk = topk

    def submit(self, name: str, layerid: int, guidedquant, error: float):

        item = (name, layerid, {'guidedquant': guidedquant, 'error': error})

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


class GuidedQuant:

    def __init__(self, layer, saliency, guided_num_groups, observe=False):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]

        # Instead of passing H in, we allocate a 3D Hessian buffer
        # that will hold sub-channel Hessians along the last dim.
        self.num_groups = guided_num_groups
        assert self.num_groups == saliency.shape[2], "Number of groups for GuidedQuant must match saliency shape!"
        self.saliencies = saliency.float()

        self.H = torch.zeros(
            (self.columns, self.columns, self.num_groups),
            device=self.dev
        )
        self.nsamples = 0
        self.inp1 = None
        self.out1 = None
        self.quantizer = quant.Quantizer()
        self.observe = observe
        self.index = 0
        # we do the same partition as before:
        assert self.rows % self.num_groups == 0, (
            f"Number of rows ({self.rows}) must be divisible "
            f"by num_groups ({self.num_groups})"
        )

    def add_batch(self, inp, out):
        # Hessian H = 2 X XT + λ I
        if self.observe:
            self.inp1 = inp
            self.out1 = out
        else:
            self.inp1 = None
            self.out1 = None

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)

        tmp = inp.shape[0]
        sal_batch = self.saliencies[self.index: self.index + tmp].to(self.dev)
        self.index += tmp
        sal_batch = sal_batch.float()

        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1])) # [bsz * seq_len, dim]
            sal_batch = sal_batch.reshape((-1, sal_batch.shape[-1])) # [bsz * seq_len, num_groups]
        
        # Convert inp to float to match sal_batch dtype for einsum operations
        inp = inp.float()
            
        sal_weighted_inp = torch.einsum("nj, ng->njg", inp, sal_batch)
        block = torch.einsum("ni,njg->ijg", inp, sal_weighted_inp)
        self.H.add_(block)


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


    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        groupsize=-1,
        actorder=False,
        name='',
        static_groups=False,
        export_to_et=False,
        args=None,
    ):
        W = self.layer.weight.data.clone()
        W = W.float()
        Scale = self.layer.weight.data.clone()
        Scale = Scale.float()
        W_int = self.layer.weight.data.clone()
        W_int = W_int.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        # Initialize scale, zero, and g_idx collections
        scale = []
        zero = []
        now_idx = 1

        # We will partition the rows into num_groups slices
        rows_per_sub = self.rows // self.num_groups

        # Prepare final Q
        Q_final = torch.zeros_like(W)

        # Loop over each row partition, using H[..., sub_idx]
        for sub_idx in range(self.num_groups):
            row_start = sub_idx * rows_per_sub
            row_end = (sub_idx + 1) * rows_per_sub

            # Sub-slice of W
            W_sub = W[row_start:row_end, :]

            # Hessian sub-part
            H_sub = self.H[:, :, sub_idx].clone()

            # Apply the same "dead columns" logic
            dead = torch.diag(H_sub) == 0
            H_sub[dead, dead] = 1
            W_sub[:, dead] = 0

            # Possibly reorder columns by diag(H_sub)
            if actorder:
                perm = torch.argsort(torch.diag(H_sub), descending=True)
                W_sub = W_sub[:, perm]
                H_sub = H_sub[perm][:, perm]
                invperm = torch.argsort(perm)

            # Create local buffers
            Losses = torch.zeros_like(W_sub)
            Q = torch.zeros_like(W_sub)

            damp = percdamp * torch.mean(torch.diag(H_sub))
            diag = torch.arange(self.columns, device=self.dev)
            H_sub[diag, diag] += damp
            

            H_sub = torch.linalg.cholesky(H_sub)
            H_sub = torch.cholesky_inverse(H_sub)
            H_sub = torch.linalg.cholesky(H_sub, upper=True)
            Hinv = H_sub

            
            for i1 in range(0, self.columns, blocksize):
                i2 = min(i1 + blocksize, self.columns)
                count = i2 - i1

                W1 = W_sub[:, i1:i2].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Losses1 = torch.zeros_like(W1)
                Hinv1 = Hinv[i1:i2, i1:i2]

                for i in range(count):
                    w = W1[:, i]
                    d = Hinv1[i, i]

                    if groupsize != -1:
                        if not static_groups:
                            if (i1 + i) % groupsize == 0:
                                self.quantizer.find_params(
                                    W[:, (i1 + i) : (i1 + i + groupsize)], weight=True
                                )
                    q = self.quantizer.quantize(
                        w.unsqueeze(1), st_idx=row_start, end_idx=row_end)
                    Q1[:, i] = q.flatten()
                    q = q.flatten()
                    Losses1[:, i] = (w - q) ** 2 / d**2

                    err1 = (w - q) / d
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    Err1[:, i] = err1

                Q[:, i1:i2] = Q1
                Losses[:, i1:i2] = Losses1 / 2

                # Propagate error across the rest
                W_sub[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            # If we permuted columns, we un-permute the result
            if actorder:
                Q = Q[:, invperm]

            # Write the subchannel results back
            Q_final[row_start:row_end, :] = Q

        error = torch.sum(Losses).item()
        torch.cuda.synchronize()
        self.print_loss(name=name, q_weight=Q_final, weight_error=error, timecost=(time.time() - tick))
        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.warning("NaN in weights")

            pprint.pprint(
                self.quantizer.bits, self.quantizer.scale, self.quantizer.zero_point
            )
            raise ValueError("NaN in weights")
        if scale == []:
            scale.append(self.quantizer.scale)
            zero.append(self.quantizer.zero)
        scale = torch.cat(scale, dim=1)
        zero = torch.cat(zero, dim=1)
        return scale, zero, None, error

    def free(self):
        self.H = None
        torch.cuda.empty_cache()
