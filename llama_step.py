"""Main entry point for layer-wise post-training quantization of LLaMA models.

Pipeline (function ``llama_sequential``):
  1. Catch the input activations of each transformer block via a forward hook.
  2. For every linear sub-module (q/k/v/o, gate/up/down), instantiate the
     selected per-method quantizer (GPTQ / LDLQ / GPTAQ / CoreQ / CoreQBeam).
  3. Accumulate the second-order statistics (Hessian H = X X^T, etc.).
  4. Call ``fasterquant`` to produce the quantized weights and pack scales.
  5. Re-run the (now quantized) block to produce inputs for the next block.

Choose the algorithm with ``--method {gptq, ldlq, gptaq, coreq, coreq_beam}``.
"""
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import quant
import pickle
import os
import glob
import re

from transformers import LlamaConfig, LlamaForCausalLM, modeling_utils
from algorithms.gptq import GPTQ
from algorithms.gptaq import GPTAQ
from algorithms.coreq import CoreQ
from algorithms.coreq_beam import CoreQBeam
from algorithms.ldlq import LDLQ
from algorithms.guidedquant import GuidedQuant
from algorithms.gptq import Observer  # Observer is shared across all algorithms
from utils import find_layers, DEV, get_loaders, export_quant_table, gen_conditions
from texttable import Texttable
import copy
import transformers
import utils
from utils.plot_delta_x import plot_delta_x_2d_3d, compute_and_save_layer_norms, plot_unified_mae, load_mae_from_pickle

def get_llama(model):

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    return model


@torch.no_grad()
def llama_sequential(model, dataloader, dev, fp_path):

    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    # model = model.cuda()
    layers[0] = Catcher(layers[0].cuda())
    for batch in dataloader:
        try:
            model(batch[0].to(dev).cuda())
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)


    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    model.model.rotary_emb = model.model.rotary_emb.to(dev)
    bsz, seqlen = position_ids.shape
    cache_position = torch.arange(seqlen, device=dev)

    # rotary_emb only needs x for dtype/device; values don't matter
    dummy = torch.empty((1, seqlen, model.config.hidden_size), device=dev, dtype=inps.dtype)

    # (cos, sin) tuple
    position_embeddings = model.model.rotary_emb(dummy, position_ids)

    print('Ready.')

    quantizers = {}
    observer = Observer()
    begin_time = time.time()

    # Store transformer block output dX for plotting
    transformer_block_dx = [] if getattr(args, 'plot_delta_x', False) else None  # per-layer per-channel MAE
    layer_mae = []   # scalar MAE per layer
    layer_fro = []   # scalar Frobenius norm per layer

    sequential = [
                ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
                ['self_attn.o_proj'],
                ['mlp.up_proj', 'mlp.gate_proj'],
                ['mlp.down_proj']
            ]

    if args.method in ["gptaq", "coreq", "coreq_beam"]:
        fp_inputs_cache = utils.modelutils.FPInputsCache(sequential)
        fp_inps = inps.clone()

    args.alpha_track = []
    args.alpha_per_module = {}

    # Track per-module metrics across layers for wandb logging
    module_rounding_ms = {}  # module_name -> list of rounding times (ms) per layer
    module_peak_mem_gb = {}  # module_name -> list of peak memory (GB) per layer

    for i in range(len(layers)):

        print(f'Quantizing layer {i+1}/{len(layers)}..')
        print('+------------------+--------------+------------+-----------+-------+')
        print('|       name       | weight_error | fp_inp_SNR | q_inp_SNR | time  |')
        print('+==================+==============+============+===========+=======+')

        layer = layers[i].to(dev)
        full = find_layers(layer)

        # GuidedQuant: load precomputed saliency tensors for this transformer block.
        saliency_dict = None
        if args.method == "guidedq":
            saliency_dict = torch.load(os.path.join(args.saliency_path, f"l{i}.pt"))

        if args.method in ["gptaq", "coreq", "coreq_beam"]:
            fp_inputs_cache.add_hook(full)

            for j in range(args.nsamples):
                fp_inps[j] = layer(fp_inps[j].unsqueeze(0), 
                                   attention_mask=attention_mask, 
                                   position_ids=position_ids,
                                    position_embeddings=position_embeddings,
                                    cache_position=cache_position
                                    )[0]
            fp_inputs_cache.clear_hook()


        for names in sequential:
            subset = {n: full[n] for n in names}
            gptq = {}

            for name in subset:
                if args.method == "gptaq":
                    gptq[name] = GPTAQ(subset[name], observe=args.observe)
                elif args.method == "coreq":
                    gptq[name] = CoreQ(
                        subset[name],
                        observe=args.observe,
                        alpha_method=getattr(args, "alpha_method", "corr"),
                    )
                elif args.method == "coreq_beam":
                    gptq[name] = CoreQBeam(
                        subset[name],
                        observe=args.observe,
                        alpha_method=getattr(args, "alpha_method", "corr"),
                    )
                elif args.method == "gptq":
                    gptq[name] = GPTQ(subset[name], observe=args.observe)
                elif args.method == "ldlq":
                    gptq[name] = LDLQ(subset[name], observe=args.observe)
                elif args.method == "guidedq":
                    gptq[name] = GuidedQuant(
                        subset[name],
                        saliency=saliency_dict[name],
                        guided_num_groups=args.guided_num_groups,
                        observe=args.observe,
                    )
                else:
                    raise ValueError(f"Method {args.method} not supported.")
                gptq[name].quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=args.w_clip)
                # Methods that consume the cached full-precision inputs.
                if args.method in ["gptaq", "coreq", "coreq_beam"]:
                    gptq[name].fp_inp = fp_inputs_cache.fp_cache[name]

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp

            first_module_name = list(subset.keys())[0]
            handle = subset[first_module_name].register_forward_hook(add_batch(first_module_name))

            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), 
                                attention_mask=attention_mask, 
                                position_ids=position_ids,
                                position_embeddings=position_embeddings,
                                cache_position=cache_position,
                                )[0]
            
            handle.remove()

            # GuidedQuant uses a per-group 3D Hessian populated through its own
            # add_batch hook on each module, so we re-register hooks per name and
            # skip the cross-module Hessian sharing below.
            if args.method == "guidedq":
                handle.remove()
                for name in subset:
                    h = subset[name].register_forward_hook(add_batch(name))
                    for j in range(args.nsamples):
                        outs[j] = layer(inps[j].unsqueeze(0),
                                        attention_mask=attention_mask,
                                        position_ids=position_ids,
                                        position_embeddings=position_embeddings,
                                        cache_position=cache_position,
                                        )[0]
                    h.remove()
            else:
                for name in subset:
                    if name != first_module_name:
                        # Hessian H is shared across modules in the same sequential group.
                        gptq[name].H = gptq[first_module_name].H
                        if args.method in ["gptaq", "coreq", "coreq_beam"]:
                            gptq[name].dXXT = gptq[first_module_name].dXXT
                            if hasattr(gptq[first_module_name], 'dXdXT'):
                                gptq[name].dXdXT = gptq[first_module_name].dXdXT

            for name in subset:
                if args.method == "guidedq":
                    scale, zero, g_idx, error = gptq[name].fasterquant(
                        percdamp=args.percdamp, groupsize=args.groupsize,
                        actorder=args.act_order, name=name, args=args,
                    )
                else:
                    scale, zero, g_idx, error = gptq[name].fasterquant(
                        percdamp=args.percdamp, groupsize=args.groupsize,
                        actorder=args.act_order, name=name,
                        alpha=args.alpha, args=args,
                    )
                quantizers['model.layers.%d.%s' % (i, name)] = (gptq[name].quantizer.cpu(), scale.cpu(), zero.cpu(), g_idx.cpu(), args.wbits, args.groupsize)

                if hasattr(gptq[name], 'rounding_ms') and hasattr(gptq[name], 'peak_mem_gb'):
                    if name not in module_rounding_ms:
                        module_rounding_ms[name] = []
                    if name not in module_peak_mem_gb:
                        module_peak_mem_gb[name] = []
                    module_rounding_ms[name].append(gptq[name].rounding_ms)
                    module_peak_mem_gb[name].append(gptq[name].peak_mem_gb)

                if args.observe:
                    observer.submit(name=name, layerid=i, gptq=gptq[name], error=error)
                else:
                    gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids,
                            position_embeddings=position_embeddings,
                            cache_position=cache_position
                            )[0]

        if getattr(args, 'plot_delta_x', False) and args.method in ["gptaq", "coreq", "coreq_beam"]:
            # X_f = fp_inps, X_q = outs
            dx_block = fp_inps - outs  # [nsamples, seqlen, hidden_size]

            # Flatten over batch & time, keep hidden dimension as "channels"
            # shape: [hidden_size, nsamples * seqlen]
            dx_block_reshaped = dx_block.permute(2, 0, 1).reshape(dx_block.shape[2], -1)

            # ---- scalar diagnostics per layer ----
            # MAE over all positions & channels
            mae_layer = dx_block_reshaped.abs().mean().item()
            layer_mae.append(mae_layer)

            # Frobenius norm over all positions & channels
            # (linalg.norm with default ord=2 on 2D is Frobenius)
            fro_layer = torch.linalg.norm(dx_block_reshaped).item()
            layer_fro.append(fro_layer)

            # ---- per-channel MAE for 3D plot ----
            # average |deltaX| for each hidden channel
            mae_per_channel = dx_block_reshaped.abs().mean(dim=1).cpu()  # [hidden_size]
            transformer_block_dx.append(mae_per_channel)

        if args.method in ["gptaq", "coreq", "coreq_beam"]:
            fp_inputs_cache.clear_cache()
        
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

        print('+------------------+--------------+------------+-----------+-------+')
        print('\n')

    end_time = time.time()

    print(f'time cost: {end_time-begin_time}s')

    # Compute and log per-module averages for rounding time and peak memory
    if args.wandb and (module_rounding_ms or module_peak_mem_gb):
        wandb_metrics = {}
        per_module_avg_rounding_ms = {}
        per_module_avg_peak_mem_gb = {}
        
        for module_name in module_rounding_ms:
            if module_rounding_ms[module_name]:
                avg_rounding_ms = sum(module_rounding_ms[module_name]) / len(module_rounding_ms[module_name])
                per_module_avg_rounding_ms[module_name] = avg_rounding_ms
                wandb_metrics[f'avg_rounding_ms/{module_name}'] = avg_rounding_ms
                print(f'Average rounding time for {module_name}: {avg_rounding_ms:.2f}ms')
        
        for module_name in module_peak_mem_gb:
            if module_peak_mem_gb[module_name]:
                avg_peak_mem_gb = sum(module_peak_mem_gb[module_name]) / len(module_peak_mem_gb[module_name])
                per_module_avg_peak_mem_gb[module_name] = avg_peak_mem_gb
                wandb_metrics[f'avg_peak_mem_gb/{module_name}'] = avg_peak_mem_gb
                print(f'Average peak memory for {module_name}: {avg_peak_mem_gb:.2f}GB')
        
        # Compute overall averages across all modules
        if per_module_avg_rounding_ms:
            overall_avg_rounding_ms = sum(per_module_avg_rounding_ms.values()) / len(per_module_avg_rounding_ms)
            wandb_metrics['avg_rounding_ms/overall'] = overall_avg_rounding_ms
            print(f'Overall average rounding time across all modules: {overall_avg_rounding_ms:.2f}ms')
        
        if per_module_avg_peak_mem_gb:
            overall_avg_peak_mem_gb = sum(per_module_avg_peak_mem_gb.values()) / len(per_module_avg_peak_mem_gb)
            wandb_metrics['avg_peak_mem_gb/overall'] = overall_avg_peak_mem_gb
            print(f'Overall average peak memory across all modules: {overall_avg_peak_mem_gb:.2f}GB')
        
        if wandb_metrics:
            wandb.log(wandb_metrics)
    

    # Plot transformer block output dX if enabled
    if getattr(args, 'plot_delta_x', False) and len(transformer_block_dx) > 0:
        plot_path_2d = f"{args.plot_delta_x_path}/transformer_block_dx_2d_{args.method}.png"
        plot_path_3d = f"{args.plot_delta_x_path}/transformer_block_dx_3d_{args.method}.png"

        # transformer_block_dx: list of [hidden_size] per layer (per-channel MAE)
        # layer_mae: list of scalar MAE per layer
        # layer_fro: list of scalar Frobenius norms per layer
        plot_delta_x_2d_3d(
            transformer_block_dx,   # per-channel MAE per layer
            layer_mae,              # scalar MAE per layer
            layer_fro,              # Fro norm per layer
            method_name=args.method,
            output_path_2d=plot_path_2d,
            output_path_3d=plot_path_3d
        )
        
        # === Save per-layer MAE / Frobenius norms ===
        alpha_val = getattr(args, 'alpha', None)
        alpha_method = getattr(args, 'alpha_method', 'corr')

        if alpha_method == 'fixed' and alpha_val is not None:
            norm_save_path = f"{args.plot_delta_x_path}/layer_norms_alpha{alpha_val}.pkl"
            compute_and_save_layer_norms(layer_fro, norm_save_path, layer_mae_list=layer_mae)

            calibration_mae_path = f"{args.plot_delta_x_path}/calibration_mae_alpha{alpha_val}.pkl"
        else:
            calibration_mae_path = f"{args.plot_delta_x_path}/calibration_mae_{args.method}.pkl"

        os.makedirs(os.path.dirname(calibration_mae_path), exist_ok=True)
        with open(calibration_mae_path, 'wb') as f:
            pickle.dump({
                'layer_mae': layer_mae,
                'alpha': alpha_val,
                'alpha_method': alpha_method,
                'method': args.method,
                'dataset': 'calibration',
            }, f)
        print(f'Saved calibration MAE to {calibration_mae_path}')
    
    if args.observe:
        observer.print()
        conditions = gen_conditions(args.wbits, args.groupsize)
        for item in observer.items():
            name = item[0]
            layerid = item[1]
            gptq = item[2]['gptq']
            error = item[2]['error']
            target = error / 2

            table = Texttable()
            table.header(['wbits', 'groupsize', 'error'])
            table.set_cols_dtype(['i', 'i', 'f'])
            table.add_row([args.wbits, args.groupsize, error])

            print('Optimizing {} {} ..'.format(name, layerid))
            for wbits, groupsize in conditions:

                if error < target:
                    # if error dropped 50%, skip
                    break

                gptq.quantizer.configure(wbits, perchannel=True, sym=args.sym, mse=False)

                # CoreQ variants take the args object; baselines do not.
                if args.method in ["coreq", "coreq_beam"]:
                    scale, zero, g_idx, error = gptq.fasterquant(percdamp=args.percdamp, groupsize=groupsize, actorder=args.act_order, name=name, alpha=args.alpha, args=args)
                else:
                    scale, zero, g_idx, error = gptq.fasterquant(percdamp=args.percdamp, groupsize=groupsize, actorder=args.act_order, name=name, alpha=args.alpha)

                table.add_row([wbits, groupsize, error])
                quantizers['model.layers.%d.%s' % (layerid, name)] = (gptq.quantizer.cpu(), scale.cpu(), zero.cpu(), g_idx.cpu(), wbits, groupsize)

            print(table.draw())
            print('\n')
            gptq.layer.to('cpu')
            gptq.free()

    model.config.use_cache = use_cache

    return quantizers


@torch.no_grad()
def llama_eval(model, testenc, dev):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    # model = model.to(dev)
    layers[0] = Catcher(layers[0].to(dev))
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch.to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    model.model.rotary_emb = model.model.rotary_emb.to(dev)
    bsz, seqlen = position_ids.shape
    cache_position = torch.arange(seqlen, device=dev)

    # rotary_emb only needs x for dtype/device; values don't matter
    dummy = torch.empty((1, seqlen, model.config.hidden_size), device=dev, dtype=inps.dtype)

    # (cos, sin) tuple
    position_embeddings = model.model.rotary_emb(dummy, position_ids)


    for i in range(len(layers)):
        layer = layers[i].to(dev)

        if args.nearest:
            subset = find_layers(layer)
            for name in subset:
                quantizer = quant.Quantizer()
                quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False)
                W = subset[name].weight.data
                quantizer.find_params(W, weight=True)
                subset[name].weight.data = quantizer.quantize(W).to(next(iter(layer.parameters())).dtype)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), 
                            attention_mask=attention_mask, 
                            position_ids=position_ids,
                            position_embeddings=position_embeddings,
                            cache_position=cache_position,
                            )[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    ppl_out = ppl.item()
    print(ppl_out)

    del nlls, ppl, shift_logits, shift_labels, lm_logits, hidden_states, loss
    del inps, outs
    testenc = testenc.cpu()

    model.config.use_cache = use_cache
    if model.model.norm is not None:
        model.model.norm = model.model.norm.cpu()
    model.lm_head = model.lm_head.cpu()
    model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    return ppl_out


@torch.no_grad()
def llama_eval_with_mae(quantized_model, fp_model, testenc, dev, method_name: str = ""):
    """
    Evaluate quantized model on validation set and collect MAE between FP and quantized activations.
    
    Args:
        quantized_model: The quantized model
        fp_model: The full precision model
        testenc: Test encoder/dataloader
        dev: Device
        method_name: Name of quantization method (for saving)
        
    Returns:
        layer_mae: List of MAE per layer
    """
    print('Evaluating with MAE collection on validation set...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // quantized_model.seqlen
    # Limit to reasonable number of samples for efficiency
    nsamples = min(nsamples, 32)  # Use up to 32 samples

    use_cache_q = quantized_model.config.use_cache
    use_cache_fp = fp_model.config.use_cache
    quantized_model.config.use_cache = False
    fp_model.config.use_cache = False
    
    layers_q = quantized_model.model.layers
    layers_fp = fp_model.model.layers

    # Setup for quantized model
    quantized_model.model.embed_tokens = quantized_model.model.embed_tokens.to(dev)
    layers_q[0] = layers_q[0].to(dev)
    
    # Setup for FP model
    fp_model.model.embed_tokens = fp_model.model.embed_tokens.to(dev)
    layers_fp[0] = layers_fp[0].to(dev)

    dtype = next(iter(quantized_model.parameters())).dtype
    inps_q = torch.zeros((nsamples, quantized_model.seqlen, quantized_model.config.hidden_size), dtype=dtype, device=dev)
    inps_fp = torch.zeros((nsamples, fp_model.seqlen, fp_model.config.hidden_size), dtype=dtype, device=dev)
    
    cache_q = {'i': 0, 'attention_mask': None, 'position_ids': None}
    cache_fp = {'i': 0, 'attention_mask': None, 'position_ids': None}

    class Catcher(nn.Module):
        def __init__(self, module, cache_dict):
            super().__init__()
            self.module = module
            self.cache_dict = cache_dict

        def forward(self, inp, **kwargs):
            inps = self.cache_dict['inps']
            inps[self.cache_dict['i']] = inp
            self.cache_dict['i'] += 1
            self.cache_dict['attention_mask'] = kwargs['attention_mask']
            self.cache_dict['position_ids'] = kwargs['position_ids']
            raise ValueError

    # Capture inputs for quantized model
    cache_q['inps'] = inps_q
    layers_q[0] = Catcher(layers_q[0].to(dev), cache_q)
    for i in range(nsamples):
        batch = testenc[:, (i * quantized_model.seqlen):((i + 1) * quantized_model.seqlen)].to(dev)
        try:
            quantized_model(batch.to(dev))
        except ValueError:
            pass
    layers_q[0] = layers_q[0].module
    
    # Capture inputs for FP model
    cache_fp['inps'] = inps_fp
    layers_fp[0] = Catcher(layers_fp[0].to(dev), cache_fp)
    for i in range(nsamples):
        batch = testenc[:, (i * fp_model.seqlen):((i + 1) * fp_model.seqlen)].to(dev)
        try:
            fp_model(batch.to(dev))
        except ValueError:
            pass
    layers_fp[0] = layers_fp[0].module

    layers_q[0] = layers_q[0].cpu()
    layers_fp[0] = layers_fp[0].cpu()
    quantized_model.model.embed_tokens = quantized_model.model.embed_tokens.cpu()
    fp_model.model.embed_tokens = fp_model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs_q = torch.zeros_like(inps_q)
    outs_fp = torch.zeros_like(inps_fp)
    attention_mask_q = cache_q['attention_mask']
    position_ids_q = cache_q['position_ids']
    attention_mask_fp = cache_fp['attention_mask']
    position_ids_fp = cache_fp['position_ids']

    layer_mae = []
    
    for i in range(len(layers_q)):
        layer_q = layers_q[i].to(dev)
        layer_fp = layers_fp[i].to(dev)

        # Run quantized layer
        for j in range(nsamples):
            outs_q[j] = layer_q(inps_q[j].unsqueeze(0), attention_mask=attention_mask_q, position_ids=position_ids_q)[0]
        
        # Run FP layer
        for j in range(nsamples):
            outs_fp[j] = layer_fp(inps_fp[j].unsqueeze(0), attention_mask=attention_mask_fp, position_ids=position_ids_fp)[0]
        
        # Compute MAE between FP and quantized outputs
        dx = outs_fp - outs_q  # [nsamples, seqlen, hidden_size]
        dx_reshaped = dx.permute(2, 0, 1).reshape(dx.shape[2], -1)  # [hidden_size, nsamples * seqlen]
        mae_layer = dx_reshaped.abs().mean().item()
        layer_mae.append(mae_layer)
        
        layers_q[i] = layer_q.cpu()
        layers_fp[i] = layer_fp.cpu()
        del layer_q, layer_fp
        torch.cuda.empty_cache()
        inps_q, outs_q = outs_q, inps_q
        inps_fp, outs_fp = outs_fp, inps_fp

    quantized_model.config.use_cache = use_cache_q
    fp_model.config.use_cache = use_cache_fp
    
    print(f'Collected MAE for {len(layer_mae)} layers')
    return layer_mae


# TODO: perform packing on GPU
def llama_pack(model, quantizers, wbits, groupsize):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    quant.make_quant_linear(model, quantizers, wbits, groupsize)
    qlayers = find_layers(model, [quant.QuantLinear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name], scale, zero, g_idx, _, _ = quantizers[name]
        qlayers[name].pack(layers[name], scale, zero, g_idx)
    print('Done.')
    return model


def load_quant(model, checkpoint, wbits, groupsize=-1, fused_mlp=True, eval=True, warmup_autotune=True):
    from transformers import LlamaConfig, LlamaForCausalLM
    config = LlamaConfig.from_pretrained(model)

    def noop(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = LlamaForCausalLM(config)
    torch.set_default_dtype(torch.float)
    if eval:
        model = model.eval()
    layers = find_layers(model)
    for name in ['lm_head']:
        if name in layers:
            del layers[name]
    quant.make_quant_linear(model, layers, wbits, groupsize)

    del layers

    print('Loading model ...')
    if checkpoint.endswith('.safetensors'):
        from safetensors.torch import load_file as safe_load
        model.load_state_dict(safe_load(checkpoint), strict=False)
    else:
        model.load_state_dict(torch.load(checkpoint), strict=False)

    if eval:
        quant.make_quant_attn(model)
        quant.make_quant_norm(model)
        if fused_mlp:
            quant.make_fused_mlp(model)
    if warmup_autotune:
        quant.autotune_warmup_linear(model, transpose=not (eval))
        if eval and fused_mlp:
            quant.autotune_warmup_fused(model)
    model.seqlen = 2048
    print('Done.')

    return model

def llama_multigpu(model, gpus, gpu_dist):
    model.model.embed_tokens = model.model.embed_tokens.to(gpus[0])
    if hasattr(model.model, 'norm') and model.model.norm:
        model.model.norm = model.model.norm.to(gpus[0])
    model.lm_head = copy.deepcopy(model.lm_head).to(gpus[0])

    cache = {'mask': None, 'position_ids': None}

    class MoveModule(nn.Module):

        def __init__(self, module, invalidate_cache):
            super().__init__()
            self.module = module
            self.dev = next(iter(self.module.parameters())).device
            self.invalidate_cache=invalidate_cache

        def forward(self, *inp, **kwargs):
            inp = list(inp)
            if inp[0].device != self.dev:
                inp[0] = inp[0].to(self.dev)

            if cache['mask'] is None or cache['mask'].device != self.dev or self.invalidate_cache:
                cache['mask'] = kwargs['attention_mask'].to(self.dev)
            kwargs['attention_mask'] = cache['mask']

            if cache['position_ids'] is None or cache['position_ids'].device != self.dev or self.invalidate_cache:
                cache['position_ids'] = kwargs['position_ids'].to(self.dev)
            kwargs['position_ids'] = cache['position_ids']
            
            tmp = self.module(*inp, **kwargs)
            return tmp

    layers = model.model.layers
    from math import ceil
    if not gpu_dist:
        pergpu = ceil(len(layers) / len(gpus))
        for i in range(len(layers)):
            layers[i] = MoveModule(layers[i].to(0 if i == 0 or i == len(layers) -1 else gpus[(i-1) // pergpu]), i==0)
    else:
        assert gpu_dist[0] >= 2, "At least two layers must be on GPU 0."
        assigned_gpus = [0] * (gpu_dist[0]-1)
        for i in range(1, len(gpu_dist)):
            assigned_gpus = assigned_gpus + [i] * gpu_dist[i]

        remaining_assignments = len(layers)-len(assigned_gpus) - 1
        if remaining_assignments > 0:
            assigned_gpus = assigned_gpus + [-1] * remaining_assignments

        assigned_gpus = assigned_gpus + [0]

        for i in range(len(layers)):
            layers[i] = MoveModule(layers[i].to(gpus[assigned_gpus[i]]), i==0)

    model.gpus = gpus


def benchmark(model, input_ids, check=False):
    input_ids = input_ids.to(model.gpus[0] if hasattr(model, 'gpus') else DEV)
    torch.cuda.synchronize()

    cache = {'past': None}

    def clear_past(i):

        def tmp(layer, inp, out):
            if cache['past']:
                cache['past'][i] = None

        return tmp

    for i, layer in enumerate(model.model.layers):
        layer.register_forward_hook(clear_past(i))

    print('Benchmarking ...')

    if check:
        loss = nn.CrossEntropyLoss()
        tot = 0.

    def sync():
        if hasattr(model, 'gpus'):
            for gpu in model.gpus:
                torch.cuda.synchronize(gpu)
        else:
            torch.cuda.synchronize()

    max_memory = 0
    with torch.no_grad():
        attention_mask = torch.ones((1, input_ids.numel()), device=DEV)
        times = []
        for i in range(input_ids.numel()):
            tick = time.time()
            out = model(input_ids[:, i:i + 1], past_key_values=cache['past'], attention_mask=attention_mask[:, :(i + 1)].reshape((1, -1)))
            sync()
            times.append(time.time() - tick)
            print(i, times[-1])
            if hasattr(model, 'gpus'):
                mem_allocated = sum(torch.cuda.memory_allocated(gpu) for gpu in model.gpus) / 1024 / 1024
            else:
                mem_allocated = torch.cuda.memory_allocated() / 1024 / 1024
            max_memory = max(max_memory, mem_allocated)
            if check and i != input_ids.numel() - 1:
                tot += loss(out.logits[0].to(DEV), input_ids[:, (i + 1)].to(DEV)).float()
            cache['past'] = list(out.past_key_values)
            del out
        sync()
        print('Median:', np.median(times))
        if check:
            print('PPL:', torch.exp(tot / (input_ids.numel() - 1)).item())
            print('max memory(MiB):', max_memory)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('model', type=str, help='llama model to load')
    parser.add_argument('dataset', type=str, choices=['wikitext2', 'ptb', 'c4'], help='Where to extract calibration data from.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration data samples.')
    parser.add_argument('--percdamp', type=float, default=.01, help='Percent of the average Hessian diagonal to use for dampening.')
    parser.add_argument('--nearest', action='store_true', help='Whether to run the RTN baseline.')
    parser.add_argument('--wbits', type=int, default=16, choices=[2, 3, 4, 5, 6, 7, 8, 16], help='#bits to use for quantization; use 16 for evaluating base model.')
    parser.add_argument('--trits', action='store_true', help='Whether to use trits for quantization.')
    parser.add_argument('--w_clip', action='store_true', help='Whether to run the RTN baseline.')
    parser.add_argument('--groupsize', type=int, default=-1, help='Groupsize to use for quantization; default uses full row.')
    parser.add_argument('--eval', action='store_true', help='evaluate quantized model.')
    parser.add_argument('--test-generation', action='store_true', help='test generation.')
    parser.add_argument('--lm-eval', action='store_true', help='evaluate quantized model using lm_eval.')
    parser.add_argument('--lm-eval-batch-size', type=int, default=32, help='Batch size for lm_eval.')
    parser.add_argument(
        '--lm-eval-cpu',
        action='store_true',
        help='Run lm_eval on CPU (slow, but avoids GPU OOM). Needed for full fp16/bf16 70B on one ~80–96GB GPU.',
    )
    parser.add_argument(
        '--lm-eval-no-auto-cpu',
        action='store_true',
        help='If set, do not fall back to CPU when model.to(cuda) OOMs for lm_eval (default: auto-fallback).',
    )
    parser.add_argument(
        '--tasks',
        nargs='+',
        default=["piqa", "arc_easy", "arc_challenge", "hellaswag", "winogrande", "boolq"],
        help='Tasks for lm_eval. Use format "task_name:num_fewshot" for few-shot tasks (e.g., "mmlu:5" for 5-shot MMLU).'   
    )
    parser.add_argument('--num-fewshot', type=int, default=0,
                        help='Global few-shot for lm_eval tasks.')
    parser.add_argument('--save', type=str, default='', help='Save quantized checkpoint under this name.')
    parser.add_argument('--save_safetensors', type=str, default='', help='Save quantized `.safetensors` checkpoint under this name.')
    parser.add_argument('--load', type=str, default='', help='Load quantized model.')
    parser.add_argument('--benchmark', type=int, default=0, help='Number of tokens to use for benchmarking.')
    parser.add_argument('--check', action='store_true', help='Whether to compute perplexity during benchmarking for verification.')
    parser.add_argument('--sym', action='store_true', help='Whether to perform symmetric quantization.')
    parser.add_argument('--act-order', action='store_true', help='Whether to apply the activation order GPTQ heuristic')
    parser.add_argument('--true-sequential', action='store_true', help='Whether to run in true sequential model.')
    parser.add_argument('--new-eval', action='store_true', help='Whether to use the new PTB and C4 eval')
    parser.add_argument('--layers-dist', type=str, default='', help='Distribution of layers across GPUs. e.g. 2:1:1 for 2 layers on GPU 0, 1 layer on GPU 1, and 1 layer on GPU 2. Any remaining layers will be assigned to your last GPU.')
    parser.add_argument('--observe',
                        action='store_true',
                        help='Auto upgrade layer precision to higher precision, for example int2 to int4, groupsize 128 to 64. \
            When this feature enabled, `--save` or `--save_safetensors` would be disable.')
    parser.add_argument('--quant-directory', type=str, default=None, help='Specify the directory for export quantization parameters to toml format. `None` means no export by default.')
    parser.add_argument('--step', action='store_true', help='')
    parser.add_argument('--step_bits', type=int, default=8)
    parser.add_argument('--method', type=str, default='', help='Method to use for quantization.')
    parser.add_argument('--sort-asym', action='store_true', help='Whether to sort asymmetric quantization levels.')
    parser.add_argument(
        '--alpha-method',
        type=str,
        default='corr',
        choices=['fixed', 'corr'],
        help='α selection: "corr" (default) is the data-driven CoreQ α; "fixed" uses --alpha as-is.',
    )
    parser.add_argument('--alpha', type=float, default=0.25, help='Coefficient for the weight-correction term (used when --alpha-method fixed).')
    parser.add_argument('--cd_passes', type=int, default=0, help='Number of coordinate-descent passes for CoreQ.')
    parser.add_argument('--saliency-path', type=str, default='cache/saliency', help='Directory of precomputed per-block saliency tensors (used by --method guidedq).')
    parser.add_argument('--guided-num-groups', type=int, default=4, help='Number of saliency groups for --method guidedq.')
    # for beam search
    parser.add_argument('--beam-size', type=int, default=1, help='Coefficient for weight correction term')
    parser.add_argument('--beam-cands', type=int, default=3, help='Coefficient for weight correction term')
    parser.add_argument('--beam-sigma', type=float, default=0.25, help='Coefficient for weight correction term')
    parser.add_argument('--beam-k', type=int, default=16, help='Coefficient for weight correction term')
    parser.add_argument('--nn_beam', action='store_true', help='Whether to plot delta X values and generate 3D plots of |X_q - X_f|')

    parser.add_argument('--incoh-process', action='store_true', help='Whether to perform incoherence process.')
    parser.add_argument('--incoh-mode', type=str, default='kron', choices=['had', 'kron'], help='Incoherence mode (Hadamard or Kronecker).')
    parser.add_argument('--rescale-WH', action='store_true', help='Whether to rescale W and H to minimize proxy loss.')
    parser.add_argument('--rescale-D', action='store_true', help='Whether to rescale W and H to minimize proxy loss.')
    parser.add_argument('--plot-delta-x', action='store_true', help='Whether to plot delta X values and generate 3D plots of |X_q - X_f|')
    parser.add_argument('--plot-delta-x-path', type=str, default='plots/delta_x', help='Directory path to save delta X plots')
    parser.add_argument('--eval-mae-validation', action='store_true', help='Collect MAE on C4 validation set and save for unified plotting')
    parser.add_argument('--plot-unified-mae', type=str, default='', help='Path to directory containing pickle files with MAE data to plot. Files should be named like "validation_mae_alpha{alpha}.pkl"')
    parser.add_argument('--ours', action='store_true', help='Use our method')
    parser.add_argument('--ours_v2', action='store_true', help='Use our method')
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--wandb-project', type=str, default='llm-quantization', help='Wandb project name')
    parser.add_argument('--wandb-name', type=str, default='', help='Wandb run name (default: auto-generated)')

    args = parser.parse_args()


    if args.wandb:
        import wandb
        run_name = args.wandb_name if args.wandb_name else f"{args.method}_{args.wbits}bit_seed{args.seed}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                'model': args.model,
                'method': args.method,
                'wbits': args.wbits,
                'groupsize': args.groupsize,
                'seed': args.seed,
                'alpha': args.alpha,
                'alpha_method': args.alpha_method,
                'act_order': args.act_order,
                'true_sequential': args.true_sequential,
                'incoh_process': args.incoh_process,
                'incoh_mode': args.incoh_mode,
                'rescale_D': args.rescale_D,
                'beam_size': args.beam_size,
                'beam_cands': args.beam_cands, 
                'nn_beam': args.nn_beam,
                'cd_passes': args.cd_passes,
                'nsamples': args.nsamples, 

            }
        )

    if args.layers_dist:
        gpu_dist = [int(x) for x in args.layers_dist.split(':')]
    else:
        gpu_dist = []

    if type(args.load) is not str:
        args.load = args.load.as_posix()

    if args.load and not args.step:
        model = load_quant(args.model, args.load, args.wbits, args.groupsize)
    else:
        model = get_llama(args.model)
        model.eval()
        if args.step:
            print('load step!')
            model_high = load_quant(args.model, args.load, args.step_bits, args.groupsize)
            model_high.eval()
            for name, param in model.named_parameters():
                print(name)
                param.data = model_high.get_buffer('.'.join(name.split('.')[:-1]+['qweight']))

    dataloader = get_loaders(args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen)

    quantizers = {}  # Initialize quantizers dict
    if (not args.load and args.wbits < 16 and not args.nearest) or args.step:
        # Default to gptq if method not specified

        torch.cuda.synchronize()
        tick = time.time()
        quantizers = llama_sequential(model, dataloader, DEV, args.model)
        torch.cuda.synchronize()
        quant_time = time.time() - tick
        print(f"Quantization time: {quant_time}s")
        if args.wandb:
            wandb.log({'quantization_time': quant_time})

    if args.benchmark:
        gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
        if len(gpus) > 1:
            llama_multigpu(model, gpus, gpu_dist)
        else:
            model = model.to(DEV)
        if args.benchmark:
            input_ids = next(iter(dataloader))[0][:, :args.benchmark]
            benchmark(model, input_ids, check=args.check)

    if args.eval:
        # datasets = ['wikitext2', 'ptb', 'c4']
        datasets = ['wikitext2', 'c4']
        if args.new_eval:
            datasets = ['wikitext2', 'c4-new']
        eval_results = {}
        for dataset in datasets:
            testloader = get_loaders(dataset, seed=args.seed, model=args.model, seqlen=model.seqlen, eval_mode=True)
            print(dataset)
            ppl = llama_eval(model, testloader, DEV)
            eval_results[f'{dataset}_perplexity'] = ppl
        
        if args.wandb:
            # Log eval results
            wandb.log(eval_results)
    
    # Collect MAE on C4 validation set if requested
    if args.eval_mae_validation:
        print('Collecting MAE on C4 validation set...')
        # Load full precision model for comparison
        fp_model = get_llama(args.model)
        fp_model.eval()
        
        # Get C4 validation set
        c4_testloader = get_loaders('c4', seed=args.seed, model=args.model, seqlen=model.seqlen, eval_mode=True)
        
        # Collect MAE
        validation_layer_mae = llama_eval_with_mae(model, fp_model, c4_testloader, DEV, method_name=args.method)
        
        # === Save validation MAE to pickle ===
        alpha_val = getattr(args, 'alpha', None)
        alpha_method = getattr(args, 'alpha_method', 'corr')

        if alpha_method == 'fixed' and alpha_val is not None:
            mae_save_path = f"{args.plot_delta_x_path}/validation_mae_alpha{alpha_val}.pkl"
        else:
            mae_save_path = f"{args.plot_delta_x_path}/validation_mae_{args.method}.pkl"
        save_data = {
            'layer_mae': validation_layer_mae,
            'alpha': alpha_val,
            'alpha_method': alpha_method,
            'method': args.method,
        }

        os.makedirs(os.path.dirname(mae_save_path), exist_ok=True)
        with open(mae_save_path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f'Saved validation MAE to {mae_save_path}')
        
        del fp_model
        torch.cuda.empty_cache()
    
    # === Optional unified MAE plot across α values ===
    if args.plot_unified_mae:
        validation_files = glob.glob(os.path.join(args.plot_unified_mae, 'validation_mae_alpha*.pkl'))
        calibration_files = glob.glob(os.path.join(args.plot_unified_mae, 'calibration_mae_alpha*.pkl'))
        
        if not validation_files and not calibration_files:
            print(f'No MAE pickle files found in {args.plot_unified_mae}')
        else:
            validation_mae_dict = {}  # Will store label -> mae_list mapping
            calibration_mae_dict = {}  # Will store label -> mae_list mapping
            
            # Load validation MAE data
            for pickle_file in validation_files:
                try:
                    data = load_mae_from_pickle(pickle_file)
                    if 'layer_mae' not in data:
                        print(f'Warning: {pickle_file} does not contain layer_mae data')
                        continue
                    
                    if 'alpha' in data and data['alpha'] is not None:
                        label = f"alpha={data['alpha']}"
                    else:
                        basename = os.path.basename(pickle_file)
                        label = basename.replace('.pkl', '').replace('validation_mae_', '')
                    validation_mae_dict[label] = data['layer_mae']
                except (ValueError, KeyError) as e:
                    print(f'Warning: Could not process {pickle_file}: {e}')

            for pickle_file in calibration_files:
                try:
                    data = load_mae_from_pickle(pickle_file)
                    if 'layer_mae' not in data:
                        print(f'Warning: {pickle_file} does not contain layer_mae data')
                        continue

                    if 'alpha' in data and data['alpha'] is not None:
                        label = f"alpha={data['alpha']}"
                    else:
                        basename = os.path.basename(pickle_file)
                        label = basename.replace('.pkl', '').replace('calibration_mae_', '')
                    calibration_mae_dict[label] = data['layer_mae']
                except (ValueError, KeyError) as e:
                    print(f'Warning: Could not process {pickle_file}: {e}')
            
            if validation_mae_dict or calibration_mae_dict:
                # Create separate plots for validation and calibration
                if validation_mae_dict:
                    validation_output_path = os.path.join(args.plot_unified_mae, 'unified_validation_mae.png')
                    plot_unified_mae(
                        validation_mae_dict,
                        validation_output_path,
                        title='MAE on C4 Validation Set',
                        calibration_data_dict=None
                    )
                    print(f'Saved validation MAE plot (with zoomed inset) to {validation_output_path}')
                
                if calibration_mae_dict:
                    calibration_output_path = os.path.join(args.plot_unified_mae, 'unified_calibration_mae.png')
                    plot_unified_mae(
                        calibration_mae_dict,
                        calibration_output_path,
                        title='MAE on C4 Calibration Set',
                        calibration_data_dict=None
                    )
                    print(f'Saved calibration MAE plot (with zoomed inset) to {calibration_output_path}')
            else:
                print('No valid MAE data found in pickle files')

    if args.test_generation:
        gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
        if len(gpus) > 1:
            llama_multigpu(model, gpus, gpu_dist)
        else:
            model = model.to(DEV)

        from transformers import LlamaTokenizer, TextStreamer
        tokenizer = LlamaTokenizer.from_pretrained(args.model, use_fast=False)
        input_ids = tokenizer(["The capital of New Mexico is"], return_tensors="pt").input_ids.to(gpus[0])
        streamer = TextStreamer(tokenizer)
        with torch.no_grad():
            generated_ids = model.generate(input_ids, streamer=streamer)
        

    if args.quant_directory is not None:
        export_quant_table(quantizers, args.quant_directory)

    if not args.observe and args.save:
        # import pdb; pdb.set_trace()
        model.save_pretrained(f'ckpts/{args.save}')
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False, use_auth_token=getattr(args, 'hf_token', None))
        tokenizer.save_pretrained(f'ckpts/{args.save}')
        # llama_6(model, quantizers, args.wbits, args.groupsize)
        # torch.save(model.state_dict(), args.save)

    if args.lm_eval:
        import gc
        import lm_eval
        from lm_eval.models.huggingface import HFLM

        if args.wbits >= 16 and not args.lm_eval_cpu:
            print(
                "[lm_eval] Note: --wbits 16 means no quantization; a 70B fp16/bf16 checkpoint is ~140GB and "
                "usually does not fit on one 80–96GB GPU. Use quantized --wbits (e.g. 4), or "
                "--lm-eval-cpu, or run lm_eval in a separate job after --save."
            )

        # Free VRAM from layer-wise --eval and quantization leftovers.
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model, use_fast=False, use_auth_token=getattr(args, 'hf_token', None)
        )

        use_cpu = bool(args.lm_eval_cpu)
        model = model.cpu()
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        if not use_cpu:
            try:
                model.to(DEV)
            except torch.cuda.OutOfMemoryError as oom:
                if args.lm_eval_no_auto_cpu:
                    raise
                print(
                    "[lm_eval] CUDA OOM while moving model to GPU (typical for full 70B on one GPU).\n"
                    "        Falling back to CPU lm_eval (slow). To force this up front: --lm-eval-cpu.\n"
                    "        To fit on GPU: quantize with --wbits < 16, or use a larger / multi-GPU setup."
                )
                model = model.cpu()
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                use_cpu = True

        if use_cpu:
            try:
                hflm = HFLM(
                    pretrained=model,
                    tokenizer=tokenizer,
                    batch_size=args.lm_eval_batch_size,
                    device="cpu",
                )
            except TypeError:
                hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.lm_eval_batch_size)
        else:
            try:
                hflm = HFLM(
                    pretrained=model,
                    tokenizer=tokenizer,
                    batch_size=args.lm_eval_batch_size,
                    device=DEV,
                )
            except TypeError:
                hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.lm_eval_batch_size)

        task_names = args.tasks
        results = lm_eval.simple_evaluate(
            hflm, tasks=task_names, num_fewshot=args.num_fewshot, batch_size=args.lm_eval_batch_size
        )['results']

        metric_vals = {task: round(result.get('acc_norm,none', result['acc,none']), 4) for task, result in results.items()}
        metric_vals['acc_avg'] = round(sum(metric_vals.values()) / len(metric_vals.values()), 4)
        print(metric_vals)
        
        if args.wandb:
            wandb.log(metric_vals)
            
            # Create a table with run information and metrics
            # Each row represents one run
            table_data = []
            
            # Prepare row data: metadata first, then task metrics
            row = [
                args.method if args.method else 'fp16',
                args.wbits,
                args.seed,
                args.groupsize if hasattr(args, 'groupsize') else None,
            ]
            
            # Add task metrics in order
            for task in sorted(results.keys()):
                acc = round(results[task].get('acc_norm,none', results[task].get('acc,none', 0)), 4)
                row.append(acc)
            
            # Add average accuracy
            row.append(metric_vals['acc_avg'])
            
            table_data.append(row)
            
            # Define column names
            columns = ['method', 'wbits', 'seed', 'groupsize']
            columns.extend([f'{task}_acc' for task in sorted(results.keys())])
            columns.append('acc_avg')
            
            # Create and log the table
            table = wandb.Table(data=table_data, columns=columns)
            wandb.log({"lm_eval_results_table": table})


    if args.wandb:
        wandb.finish()

    if not args.observe and args.save_safetensors:
        llama_pack(model, quantizers, args.wbits, args.groupsize)
        from safetensors.torch import save_file as safe_save
        state_dict = model.state_dict()
        state_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}
        safe_save(state_dict, args.save_safetensors)
        