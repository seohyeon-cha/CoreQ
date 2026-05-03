"""CoreQ rounding with successive (beam) search.

Builds on :mod:`algorithms.coreq` (same data-driven α and mismatch
correction term). On top of the backward-residual rounding step it
performs a row-wise beam search over the discrete codewords:

  * ``--beam-size``  K — number of candidates carried per row.
  * ``--cd_passes``    — optional coordinate-descent refinement passes.

For each block of columns the routine expands K candidates per row,
scores them with the proxy loss
    sum_i (W_q[i, :] - W[i, :])^T H (W_q[i, :] - W[i, :]),
and keeps the best K beams.
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


class CoreQBeam:

    def __init__(
        self,
        layer,
        observe=False,
        store_delta_x=False,
        alpha_method="corr",
    ):
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
        self.dXdXT = (
            torch.zeros((self.columns, self.columns), device=self.dev)
            if alpha_method == "corr"
            else None
        )

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

        if self.dXdXT is not None:
            self.dXdXT += dX.matmul(dX.t())

        self.dXXT += dX.matmul(inp.t())

        if self.store_delta_x:
            abs_dX = torch.abs(dX)
            self.delta_x_values.append(abs_dX.cpu().clone())

        dX = None
        inp = None
        del self.fp_inp[0]

    def print_loss(self, name, q_weight, alpha, timecost):
        table = Texttable()
        name += ' ' * (16 - len(name))
        table.header(['name', 'alpha', 'fp_inp_SNR', 'q_inp_SNR', 'time'])
        self.layer.weight.data = q_weight.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if self.inp1 is not None:
            quantizer = quant.Quantizer()
            quantizer.configure(8, perchannel=False, sym=True, mse=False)
            quantizer.find_params(self.inp1, weight=True)
            q_in = quantizer.quantize(self.inp1).type(torch.float16)
            q_out = self.layer(q_in)
            q_SNR = torch_snr_error(q_out, self.out1).item()
            fp_SNR = torch_snr_error(self.layer(self.inp1), self.out1).item()
        else:
            q_SNR = '-'
            fp_SNR = '-'
        table.add_row([name, alpha, fp_SNR, q_SNR, timecost])
        print(table.draw().split('\n')[-2])

    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        groupsize=-1,
        actorder=False,
        name="",
        alpha=0.25,
        args=None,
    ):
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
        if args.alpha_method == "corr" and self.dXdXT is not None:
            _corr_total = self.frob_inner_chunked(W, W @ self.dXdXT)
            self.dXdXT = None
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
        L = torch.linalg.cholesky(Hp)
        Hp_inv = torch.cholesky_inverse(L)
        L_diag = torch.diag(L)
        L = L / L_diag.unsqueeze(0)
        L = L - torch.eye(L.shape[0], device=L.device)

        # === Mismatch-correction direction ===
        tmp = Dp @ Hp_inv
        W_base = Wr[:, p]
        W_dir = W_base @ tmp

        # === Corr-α Phase 2: signal energy and α* ===
        if _corr_total is not None:
            _corr_signal = self.frob_inner_chunked(W_dir, W_base @ Dp)
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
            del _corr_total, _corr_signal

        del tmp, Dp

        # === Corrected continuous target ===
        W_ref = W_base + args.alpha * W_dir

        if not self.quantizer.ready():
            self.quantizer.find_params(W_ref, weight=True)

        # === Block-wise beam rounding ===
        rows, cols = W_ref.shape
        device = W_ref.device

        beam_width = int(getattr(args, "beam_size", 1))
        beam_width = max(1, beam_width)
        B = beam_width

        beam_cands = int(getattr(args, "beam_cands", 0))
        beam_cands = max(0, beam_cands)

        maxq = int(self.quantizer.maxq.item())
        A = maxq + 1
        codes = torch.arange(A, device=device)

        W_ref_f = W_ref.float()
        L_f = L.float()
        L_diag_f = L_diag.float()

        beam_scores = torch.full((rows, B), float("inf"), device=device)
        beam_scores[:, 0] = 0.0

        tail_dtype = torch.float32
        Q_tail = torch.empty((rows, B, 0), device=device, dtype=tail_dtype)

        scale = []
        zero = []
        seen_groups = set()

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        torch.cuda.synchronize()
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()

        for i2 in range(cols, 0, -blocksize):
            i1 = max(i2 - blocksize, 0)
            count = i2 - i1
            tail_len = Q_tail.shape[2]

            W1 = W_ref_f[:, i1:i2]
            Lblk = L_f[i1:i2, i1:i2]

            if tail_len > 0:
                W2 = W_ref_f[:, None, i2:]
                E2 = W2 - Q_tail
                Ltail = L_f[i2:, i1:i2]
                tail_corr = E2 @ Ltail
            else:
                tail_corr = torch.zeros((rows, B, count), device=device)

            beam_states = torch.zeros((rows, B, count), device=device, dtype=torch.float32)
            tail_src = torch.arange(B, device=device, dtype=torch.long).view(1, B).expand(rows, B)

            for ii in reversed(range(count)):
                col_abs = i1 + ii
                Ljj2 = L_diag_f[col_abs] ** 2

                if groupsize != -1:
                    gstart = (i1 + ii) // groupsize * groupsize
                    gend = min(gstart + groupsize, self.columns)
                    group_id = (i1 + ii) // groupsize

                    if group_id not in seen_groups:
                        self.quantizer.find_params(W_ref[:, gstart:gend], weight=True)
                        scale.append(self.quantizer.scale)
                        zero.append(self.quantizer.zero)
                        seen_groups.add(group_id)

                sc = self.quantizer.scale.reshape(rows, 1).float().to(device)
                ze = self.quantizer.zero.reshape(rows, 1).float().to(device)

                v = Lblk[:, ii]
                What = (
                    W1[:, ii].unsqueeze(1)
                    + (W1.unsqueeze(1) - beam_states) @ v
                    + tail_corr[:, :, ii]
                )

                levels = (codes.view(1, A) - ze) * sc
                u = What / sc + ze
                du = u.unsqueeze(-1) - codes.view(1, 1, A)
                inc = (du * sc.unsqueeze(1)).pow(2) * Ljj2

                new_scores = beam_scores.unsqueeze(-1) + inc

                flat = new_scores.reshape(rows, -1)
                keep = B
                topv, topi = torch.topk(flat, k=keep, dim=1, largest=False, sorted=True)

                parent = topi // A
                choice = topi % A

                gather_idx = parent.unsqueeze(-1).expand(-1, -1, count)
                beam_states = beam_states.gather(1, gather_idx).clone()
                tail_corr = tail_corr.gather(1, parent.unsqueeze(-1).expand(-1, -1, count)).clone()
                tail_src = tail_src.gather(1, parent)
                beam_scores = topv

                q_sel = levels.gather(1, choice)
                beam_states[:, :, ii] = q_sel

            if tail_len > 0:
                idx = tail_src.unsqueeze(-1).expand(-1, -1, tail_len)
                old_tail = Q_tail.gather(1, idx)
                Q_tail = torch.cat([beam_states.to(tail_dtype), old_tail], dim=2)
            else:
                Q_tail = beam_states.to(tail_dtype)

        best = beam_scores.argmin(dim=1)
        Qp = Q_tail[torch.arange(rows, device=device), best, :].to(W_ref.dtype)
        obj_total = float(beam_scores.min(dim=1).values.sum().item())

        cd_passes = int(getattr(args, "cd_passes", 0))
        if cd_passes > 0:
            with torch.no_grad():
                H_cd = Hp
                H_cd = H_cd / H_cd.diag().max().clamp(min=1e-8)

                cols_cd = self.columns
                gs = cols_cd if groupsize == -1 else groupsize

                s = Qp - W_ref

                for igp in range(cd_passes):
                    any_change = False
                    curr_gid = None

                    for i2b in range(cols_cd, 0, -blocksize):
                        i1b = max(i2b - blocksize, 0)
                        count = i2b - i1b

                        W1 = Qp[:, i1b:i2b].clone()
                        S0 = s[:, :i1b]
                        S1 = s[:, i1b:i2b].clone()
                        S2 = s[:, i2b:]

                        H0 = H_cd[:i1b, i1b:i2b]
                        H1 = H_cd[i1b:i2b, i1b:i2b]
                        H2 = H_cd[i2b:, i1b:i2b]

                        Hs_pre = torch.zeros(
                            (Qp.shape[0], count), device=Qp.device, dtype=Qp.dtype
                        )
                        if i1b > 0:
                            Hs_pre += S0 @ H0
                        if i2b < cols_cd:
                            Hs_pre += S2 @ H2

                        S1H1 = S1 @ H1

                        for ii in reversed(range(count)):
                            col_abs = i1b + ii

                            if groupsize != -1:
                                gid = col_abs // gs
                                if gid != curr_gid:
                                    curr_gid = gid
                                    gstart = gid * gs
                                    gend = min(gstart + gs, cols_cd)
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

                        Qp[:, i1b:i2b] = W1
                        s[:, i1b:i2b] = S1

                    if not any_change:
                        break

        end_evt.record()
        torch.cuda.synchronize()
        rounding_ms = start_evt.elapsed_time(end_evt)

        peak_mem_bytes = torch.cuda.max_memory_allocated()
        peak_mem_gb = peak_mem_bytes / (1024**3)

        self.rounding_ms = rounding_ms
        self.peak_mem_gb = peak_mem_gb

        Q = Qp[:, inv_p]

        alpha = float(getattr(args, "alpha", 0.25))
        # === Per-module α bookkeeping (corr method) ===
        if args.alpha_method == "corr":
            alpha = args.alpha
            if hasattr(args, "alpha_per_module") and name in args.alpha_per_module:
                args.alpha_per_module[name].append(alpha)
            if hasattr(args, "alpha_track"):
                args.alpha_track.append(alpha)

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

        scale = torch.cat(scale[::-1], dim=1)
        zero = torch.cat(zero[::-1], dim=1)

        return scale, zero, g_idx, obj_total

    def frob_inner_chunked(self, A, B, col_bs=2048):
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

    if args.incoh_mode == "had":
        SU = (torch.randn(n, device=device).sign() + 1e-5).sign().to(dtype_)
        SV = (torch.randn(m, device=device).sign() + 1e-5).sign().to(dtype_)
        Hr = RHT_H(Hr, SU)
        if D is not None:
            Dr = RHT_H(Dr, SU).T
        Wr = RHT_W(Wr, SU, SV)

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
    if args.incoh_mode == "had":
        hatWr = (
            quip_utils.matmul_hadU(
                (quip_utils.matmul_hadU(hatWr) * SU.to(device)).T
            )
            * SV.to(device)
        ).T
    elif args.incoh_mode == "kron":
        hatWr = SV.T.to(device) @ hatWr @ SU.to(device)
    else:
        raise NotImplementedError

    if args.rescale_WH:
        hatWr /= scaleWH[None, :].to(device)

    assert torch.isfinite(hatWr).all()
    return hatWr
