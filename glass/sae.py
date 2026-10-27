"""Inference-only EleutherAI top-k SAE for global style extraction.

The encoder follows the ReLU/linear/topk implementation in EleutherAI's
``sparsify`` package.
"""

import json
from pathlib import Path

import torch
from torch.nn import functional as F


class OfficialSAE:
    """Load a single official ``layers.N`` directory without recasting weights.

    Encoder parameters remain FP32. Dense sparse activations intentionally use
    BF16 on CPU; the decoder intentionally uses BF16 matrix multiplication.
    """

    def __init__(self, model_path, device='cuda'):
        from safetensors.torch import load_file

        directory = Path(model_path)
        with (directory / 'cfg.json').open(encoding='utf-8') as stream:
            config = json.load(stream)
        if config.get('activation', 'topk') != 'topk':
            raise ValueError('GLASS requires the official topk SAE')
        if any(config.get(key, False) for key in ('signed', 'transcode', 'skip_connection')):
            raise ValueError('Signed, transcoder, and skip-connection SAEs are unsupported')
        self.device = torch.device(device)
        self.k = int(config['k'])
        state = load_file(str(directory / 'sae.safetensors'), device=str(self.device))
        self.encoder_weight = state['encoder.weight']
        self.encoder_bias = state['encoder.bias']
        self.W_dec = state['W_dec']
        self.b_dec = state['b_dec']
        self.d_hidden, self.d_in = self.encoder_weight.shape
        if not 0 < self.k <= self.d_hidden:
            raise ValueError(f'Invalid SAE top-k: {self.k}')
        expected = ((self.d_hidden,), (self.d_hidden, self.d_in), (self.d_in,))
        actual = (tuple(self.encoder_bias.shape), tuple(self.W_dec.shape), tuple(self.b_dec.shape))
        if actual != expected or int(config['d_in']) != self.d_in:
            raise ValueError(f'Incompatible official SAE parameter shapes: {actual}')
        if any(weight.dtype != torch.float32 for weight in state.values()):
            raise ValueError('The verified official SAE stores FP32 parameters')

    @torch.no_grad()
    def encode(self, hidden):
        """Encode every answer token together, returning dense CPU BF16 rows."""
        if hidden.ndim != 2 or hidden.shape[1] != self.d_in:
            raise ValueError(f'Expected [tokens, {self.d_in}], got {tuple(hidden.shape)}')
        hidden = hidden.to(self.device).to(torch.bfloat16)
        preacts = F.relu(F.linear(hidden - self.b_dec, self.encoder_weight, self.encoder_bias))
        values, indices = torch.topk(preacts, self.k, dim=-1, sorted=False)
        dense = torch.zeros(hidden.shape[0], self.d_hidden, dtype=torch.bfloat16)
        dense.scatter_(1, indices.cpu().long(), values.cpu().to(torch.bfloat16))
        return dense

    @torch.no_grad()
    def decode(self, latent):
        latent = latent.to(self.device).to(torch.bfloat16)
        return latent @ self.W_dec.to(torch.bfloat16) + self.b_dec.to(torch.bfloat16)
