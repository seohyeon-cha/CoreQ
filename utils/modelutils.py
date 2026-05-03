import torch
import torch.nn as nn
import functools

DEV = torch.device('cuda:0')


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + '.' + name1 if name != '' else name1))
    return res


def gen_conditions(_wbits, _groupsize):
    wbits = _wbits
    groupsize = _groupsize
    conditions = []
    while True:
        if wbits >= 8:
            if groupsize == -1 or groupsize == 32:
                break

        if groupsize > 32:
            groupsize /= 2
        else:
            wbits *= 2
            groupsize = _groupsize

        conditions.append((int(wbits), int(groupsize)))
    return conditions


# copy from https://github.com/openppl-public/ppq/blob/master/ppq/quantization/measure/norm.py
def torch_snr_error(y_pred: torch.Tensor, y_real: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """
    Compute SNR between y_pred(tensor) and y_real(tensor)
    
    SNR can be calcualted as following equation:
    
        SNR(pred, real) = (pred - real) ^ 2 / (real) ^ 2
    
    if x and y are matrixs, SNR error over matrix should be the mean value of SNR error over all elements.
    
        SNR(pred, real) = mean((pred - real) ^ 2 / (real) ^ 2)
    Args:
        y_pred (torch.Tensor): _description_
        y_real (torch.Tensor): _description_
        reduction (str, optional): _description_. Defaults to 'mean'.
    Raises:
        ValueError: _description_
        ValueError: _description_
    Returns:
        torch.Tensor: _description_
    """
    y_pred = y_pred.type(torch.float32)
    y_real = y_real.type(torch.float32)

    if y_pred.shape != y_real.shape:
        raise ValueError(f'Can not compute snr loss for tensors with different shape. '
                         f'({y_pred.shape} and {y_real.shape})')
    reduction = str(reduction).lower()

    if y_pred.ndim == 1:
        y_pred = y_pred.unsqueeze(0)
        y_real = y_real.unsqueeze(0)

    y_pred = y_pred.flatten(start_dim=1)
    y_real = y_real.flatten(start_dim=1)

    noise_power = torch.pow(y_pred - y_real, 2).sum(dim=-1)
    signal_power = torch.pow(y_real, 2).sum(dim=-1)
    snr = (noise_power) / (signal_power + 1e-7)

    if reduction == 'mean':
        return torch.mean(snr)
    elif reduction == 'sum':
        return torch.sum(snr)
    elif reduction == 'none':
        return snr
    else:
        raise ValueError(f'Unsupported reduction method.')


class FPInputsCache:
    """
    class for saving the full-precision output in each layer.
    """
    def __init__(self, sequential):
        self.fp_cache = {}
        self.names = sequential[0]+sequential[1]+sequential[2]+sequential[3]
        for name in self.names:
            self.fp_cache[name] = []
        self.handles = []

    def cache_fp_input(self, m, inp, out, name):
        inp = inp[0].detach()
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        self.fp_cache[name] += [inp.t()]

    def add_hook(self, full):
        for name in self.names:
            self.handles.append(
                full[name].register_forward_hook(
                    functools.partial(self.cache_fp_input, name=name)
                )
            )

    def clear_hook(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        torch.cuda.empty_cache()

    def clear_cache(self):
        for name in self.names:
            self.fp_cache[name] = []

