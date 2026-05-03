"""CoreQ rounding (no beam search).

Given the calibration Hessian ``H = X X^T`` from the quantized forward pass
and the cross statistic ``dXXT = (X_fp - X_q) X^T`` from the
full-precision pass, CoreQ picks a single per-layer coefficient α (one
scalar per linear sub-module, computed by summing the correlation
statistics over all rows and columns of that module). ``alpha_method``
selects how α is set:

  * ``corr``  — α from a local correlation estimate, clamped to ``[0, 1]``.
  * ``fixed`` — use the user-supplied ``--alpha``.

The corrected continuous target ``W' = W + α * dXXT @ H^{-1}`` is then
rounded sequentially with the standard backward-residual step. See
:mod:`algorithms.coreq_beam` for the successive (beam) rounding variant.
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


class CoreQ:

    def __init__(self, layer, observe=False, store_delta_x=False, alpha_method="corr"):
        assert alpha_method in ("fixed", "corr"), \
            f"alpha_method must be 'fixed' or 'corr', got {alpha_method!r}"
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]

        # === Second-order statistics ===
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.dXXT = torch.zeros((self.columns, self.columns), device=self.dev)
        # ΔX·ΔXᵀ — only needed for the data-driven α (alpha_method == "corr").
        self.dXdXT = torch.zeros((self.columns, self.columns), device=self.dev) if alpha_method == "corr" else None

        self.inp1 = None
        self.out1 = None
        self.nsamples = 0
        self.quantizer = quant.Quantizer()
        self.observe = observe
        self.inps = []
        self.store_delta_x = store_delta_x
        self.delta_x_values = [] if store_delta_x else None

        self.alpha_method = alpha_method
        self.optimize_ms = 0


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
        if self.dXdXT is not None:
            self.dXdXT *= self.nsamples / (self.nsamples + tmp)

        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())
        dX = self.fp_inp[0].float() * math.sqrt(2 / self.nsamples) - inp

        # Accumulate dXdXT before any alpha scaling (used for the corr α).
        if self.dXdXT is not None:
            self.dXdXT += dX.matmul(dX.t())

        self.dXXT += dX.matmul(inp.t())

        # Store |deltaX| for plotting only if enabled: shape is [channels, samples]
        if self.store_delta_x:
            abs_dX = torch.abs(dX)  # |deltaX| per channel
            self.delta_x_values.append(abs_dX.cpu().clone())
        
        dX = None 
        inp = None 
        del self.fp_inp[0]


    def print_loss(self, name, q_weight, alpha, timecost):
        table = Texttable()
        name += ' ' * (16 - len(name))

        table.header(['name', 'alpha', 'fp_inp_SNR', 'q_inp_SNR', 'time'])

        # assign weight
        self.layer.weight.data = q_weight.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        if self.inp1 is not None:
            # quantize input to int8
            quantizer = quant.Quantizer()
            quantizer.configure(8, perchannel=False, sym=True, mse=False)
            quantizer.find_params(self.inp1, weight=True)
            q_in = quantizer.quantize(self.inp1).type(torch.float16)
            q_out = self.layer(q_in)

            # get kinds of SNR
            q_SNR = torch_snr_error(q_out, self.out1).item()
            fp_SNR = torch_snr_error(self.layer(self.inp1), self.out1).item()
        else:
            q_SNR = '-'
            fp_SNR = '-'

        table.add_row([name, alpha, fp_SNR, q_SNR, timecost])
        print(table.draw().split('\n')[-2])


    def fasterquant(self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, name='', alpha=0.25, args=None):

        self.layer.to(self.dev)
        self.quantizer.to(self.dev)

        W = self.layer.weight.data.clone()
        W = W.float()

        tick = time.time()

        H = self.H
        self.H = None

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        self.dXXT[:, dead] = 0
        if self.dXdXT is not None:
            self.dXdXT[:, dead] = 0
            self.dXdXT[dead, :] = 0

        D = self.dXXT.clone()
        self.dXXT = None

        # === Corr-α Phase 1: total = ‖W ΔX‖²_F (original space) ===
        _corr_total = None
        _corr_phase1_ms = 0.0
        if args.alpha_method == "corr" and self.dXdXT is not None:
            torch.cuda.synchronize()
            _p1_start = torch.cuda.Event(enable_timing=True)
            _p1_end   = torch.cuda.Event(enable_timing=True)
            _p1_start.record()
            _corr_total = (W * (W @ self.dXdXT)).sum(dtype=torch.float32)
            self.dXdXT = None
            _p1_end.record()
            torch.cuda.synchronize()
            _corr_phase1_ms = _p1_start.elapsed_time(_p1_end)
        else:
            if self.dXdXT is not None:
                self.dXdXT = None

        if args.incoh_process:
            Hr, Dr, Wr, SU, SV, scaleWH = incoherence_preprocess(W, H, D, args)
        else:
            Hr = H
            Dr = D
            Wr = W
            SU = None
            SV = None
            scaleWH = None

        del H, D, W 
        

        # === Damping & permutation ===
        damp = args.percdamp * torch.mean(torch.diag(Hr))
        diag = torch.arange(Hr.shape[0], device=Hr.device)
        Hr[diag, diag] += damp

        p = torch.argsort(torch.diag(Hr), descending=False)
        inv_p = torch.argsort(p)
        Hp = Hr[p][:, p]
        Dp = Dr[p][:, p]
        del Hr

        # === Cholesky factor and feedback matrix ===
        cd_passes = int(getattr(args, "cd_passes", 0))
        L = torch.linalg.cholesky(Hp)
        Hp_inv = torch.cholesky_inverse(L)
        if cd_passes <= 0:
            del Hp

        L_diag = L.diagonal().clone()
        L.div_(L_diag.unsqueeze(0))
        L.diagonal().zero_()

        # === Mismatch-correction direction ===
        tmp = Dp @ Hp_inv
        W_base = Wr[:, p]
        W_dir = W_base @ tmp

        # === Corr-α Phase 2: signal energy and α* ===
        if _corr_total is not None:
            torch.cuda.synchronize()
            _p2_start = torch.cuda.Event(enable_timing=True)
            _p2_end   = torch.cuda.Event(enable_timing=True)
            _p2_start.record()

            _corr_signal = (W_dir * (W_base @ Dp)).sum(dtype=torch.float32)

            _p2_end.record()
            torch.cuda.synchronize()
            _corr_phase2_ms = _p2_start.elapsed_time(_p2_end)

            t = float(_corr_total.item())
            s = float(_corr_signal.item())
            if t == 0.0 or not math.isfinite(t) or not math.isfinite(s):
                alpha_corr = 0.0
            else:
                alpha_corr = torch.clamp(
                    _corr_signal / _corr_total, 0.0, 1.0
                ).item()
                if not math.isfinite(alpha_corr):
                    alpha_corr = 0.0
            args.alpha = alpha_corr
            self.optimize_ms = _corr_phase1_ms + _corr_phase2_ms
            del _corr_total, _corr_signal

        del tmp, Dp

        # === Corrected continuous target ===
        W_ref = W_base + args.alpha * W_dir

        if not self.quantizer.ready():
            self.quantizer.find_params(W_ref, weight=True)

        Q = torch.zeros_like(W_ref)

        g_idx = []
        scale = []
        zero = []
        seen_groups = set()

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        torch.cuda.synchronize()
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt   = torch.cuda.Event(enable_timing=True)
        start_evt.record()

        # === Backward residual rounding ===
        for i2 in range(self.columns, 0, -blocksize):
            i1 = max(i2 - blocksize, 0)
            count = i2 - i1

            W1 = W_ref[:, i1:i2].clone()
            What1 = Q[:, i1:i2].clone()
            Wdiff = W_ref[:, i2:] - Q[:, i2:]
            L1 = L[:, i1:i2]
            tail_corr = Wdiff @ L1[i2:, :]

            for i in reversed(range(count)):
                if groupsize != -1:
                    gstart = (i1 + i) // groupsize * groupsize
                    gend   = min(gstart + groupsize, self.columns)
                    group_id = (i1 + i) // groupsize

                    if group_id not in seen_groups:
                        self.quantizer.find_params(W_ref[:, gstart:gend], weight=True)
                        scale.append(self.quantizer.scale)
                        zero.append(self.quantizer.zero)
                        seen_groups.add(group_id)

                What = W1[:,i] + (W1 - What1) @ L1[i1:i2,i] + tail_corr[:, i]
                What1[:, i] = self.quantizer.quantize(What.unsqueeze(1)).flatten()
            Q[:, i1:i2] = What1
        
        # === Optional coordinate-descent refinement ===
        if cd_passes > 0:
            with torch.no_grad():
                H_cd = Hp
                H_cd = H_cd / H_cd.diag().max().clamp(min=1e-8)

                cols = self.columns
                gs = cols if groupsize == -1 else groupsize

                s = Q - W_ref

                for _ in range(cd_passes):
                    any_change = False
                    curr_gid = None

                    for i2 in range(cols, 0, -blocksize):
                        i1 = max(i2 - blocksize, 0)
                        count = i2 - i1

                        W1 = Q[:, i1:i2].clone()
                        S0 = s[:, :i1]
                        S1 = s[:, i1:i2].clone()
                        S2 = s[:, i2:]

                        H0 = H_cd[:i1,  i1:i2]
                        H1 = H_cd[i1:i2, i1:i2]
                        H2 = H_cd[i2:,  i1:i2]

                        Hs_pre = torch.zeros(
                            (Q.shape[0], count), device=Q.device, dtype=Q.dtype
                        )
                        if i1 > 0:
                            Hs_pre += S0 @ H0
                        if i2 < cols:
                            Hs_pre += S2 @ H2

                        S1H1 = S1 @ H1

                        for ii in reversed(range(count)):
                            col_abs = i1 + ii

                            if groupsize != -1:
                                gid = col_abs // gs
                                if gid != curr_gid:
                                    curr_gid = gid
                                    gstart = gid * gs
                                    gend = min(gstart + gs, cols)
                                    self.quantizer.find_params(W_ref[:, gstart:gend], weight=True)

                            denom = H1[ii, ii].clamp(min=1e-8)

                            Hs = Hs_pre[:, ii] + S1H1[:, ii]

                            proposal = W1[:, ii] - (Hs / denom)
                            q_new = self.quantizer.quantize(proposal.unsqueeze(1)).flatten()

                            eps = W1[:, ii] - q_new
                            if torch.any(eps != 0):
                                any_change = True

                            W1[:, ii] = q_new
                            S1[:, ii] -= eps

                            S1H1 += (-eps).unsqueeze(1) * H1[ii, :].unsqueeze(0)

                        Q[:, i1:i2] = W1
                        s[:, i1:i2] = S1

                    if not any_change:
                        break

        if cd_passes > 0:
            del Hp

        end_evt.record()
        torch.cuda.synchronize()
        rounding_ms = start_evt.elapsed_time(end_evt)  # milliseconds

        peak_mem_bytes = torch.cuda.max_memory_allocated()
        peak_mem_gb = peak_mem_bytes / (1024**3)

        self.rounding_ms = rounding_ms
        self.peak_mem_gb = peak_mem_gb

        Q = Q[:, inv_p].to(Q.device)

        # === Per-module α bookkeeping (corr method) ===
        if args.alpha_method == "corr":
            alpha = args.alpha
            if hasattr(args, 'alpha_per_module') and name in args.alpha_per_module:
                args.alpha_per_module[name].append(alpha)
            if hasattr(args, 'alpha_track'):
                args.alpha_track.append(alpha)
            print("Time for corr α (ms):", self.optimize_ms)

        if args.incoh_process:
            Q = incoherence_process(Q, SU, SV, scaleWH, args)

        torch.cuda.synchronize()

        groupsize = groupsize if groupsize != -1 else self.columns
        g_idx = [i // groupsize for i in range(self.columns)]
        g_idx = torch.tensor(g_idx, dtype=torch.int32, device=Q.device)
        g_idx = g_idx[inv_p] 

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )

        self.print_loss(name=name, q_weight=Q, alpha=alpha, timecost=(time.time() - tick))
        
        if scale == []:
            scale.append(self.quantizer.scale)
            zero.append(self.quantizer.zero)
        
        # Reverse scale and zero to match original column order since we processed in reverse
        scale = torch.cat(scale[::-1], dim=1)
        zero = torch.cat(zero[::-1], dim=1)

        return scale, zero, g_idx, None

    def frob_inner_chunked(self, A, B, col_bs=2048):
        # returns sum_{i,j} A_ij * B_ij in float32 without big intermediates
        assert A.shape == B.shape
        out = torch.zeros((), device=A.device, dtype=torch.float32)
        for c0 in range(0, A.shape[1], col_bs):
            c1 = min(c0 + col_bs, A.shape[1])
            out += (A[:, c0:c1] * B[:, c0:c1]).sum(dtype=torch.float32)
        return out

    def free(self):
        self.inp1 = None
        self.out1 = None
        self.H = None
        self.dXXT = None 
        self.dXdXT = None
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
    if Dr is not None:
        Dr[:, dead] = 0
        Dr[dead, :] = 0

    Hr = Hr.to(device)
    if Dr is not None:
        Dr = Dr.to(device)
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