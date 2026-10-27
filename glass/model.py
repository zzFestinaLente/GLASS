"""Llama residual-stream steering and activation extraction."""

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def add_vector(matrix, vector, position_ids):
    """Apply a steering direction to token activations."""
    from_id = position_ids.min().item() - 1
    if position_ids.ndim == 3 and position_ids.shape[1] == 1:
        position_ids = position_ids[:, 0, :]
    elif position_ids.ndim == 2 and position_ids.shape[0] == 1:
        position_ids = position_ids[0]
    mask = (position_ids >= from_id).unsqueeze(-1)
    if vector.ndim == 1:
        vector = vector.view(1, 1, -1)
    elif vector.ndim == 2:
        if matrix.ndim == 3 and vector.shape[0] == matrix.shape[0]:
            vector = vector.view(vector.shape[0], 1, vector.shape[1])
        else:
            vector = vector.view(1, 1, vector.shape[1])
    elif vector.ndim != 3:
        raise ValueError(f'Unsupported steering vector shape: {tuple(vector.shape)}')
    if matrix.ndim == 2:
        matrix = matrix.unsqueeze(0)
    return matrix + mask.to(matrix.dtype) * vector.to(matrix.dtype)


class SteeringBlock(nn.Module):
    """Record decoder activations and apply a steering direction."""

    def __init__(self, block):
        super().__init__()
        self.block = block
        self.vector = None
        self.hidden = None
        self.capture_hidden = True

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        hidden = output[0] if isinstance(output, tuple) else output
        self.hidden = hidden if self.capture_hidden else None
        if self.vector is None:
            return output
        augmented = add_vector(hidden, self.vector, kwargs['position_ids'])
        return (augmented,) + output[1:] if isinstance(output, tuple) else augmented


class SteeringModel:
    """A BF16 Base Llama with left padding and greedy generation.

    Layer indices are zero based. Forward passes expose decoder activations
    for teacher-forced response extraction.
    """

    def __init__(self, model_name, device='cuda'):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16,
        ).to(self.device).eval()
        # RoPE buffers retain the FP32 precision assigned during initialization.
        for index, block in enumerate(self.model.model.layers):
            self.model.model.layers[index] = SteeringBlock(block)
        self.model.eval()

    @property
    def layers(self):
        return self.model.model.layers

    def reset(self):
        for block in self.layers:
            block.vector = None
            block.hidden = None

    def set_vector(self, layer, vector):
        if not 0 <= layer < len(self.layers):
            raise ValueError(f'Invalid decoder layer: {layer}')
        self.layers[layer].vector = vector.to(self.device)

    def activations(self, layer):
        hidden = self.layers[layer].hidden
        if hidden is None:
            raise RuntimeError(f'No activations recorded for layer {layer}; run a forward pass first.')
        return hidden.unsqueeze(0) if hidden.ndim == 2 else hidden

    def generate(self, prompts: list[str], max_new_tokens: int) -> list[str]:
        """Decode a batch of prompts with greedy generation."""
        if isinstance(prompts, str):
            raise TypeError('prompts must be a list of strings')
        if not prompts:
            return []
        capture_flags = [block.capture_hidden for block in self.layers]
        try:
            for block in self.layers:
                block.hidden = None
                block.capture_hidden = False
            encoded = self.tokenizer(prompts, return_tensors='pt', padding=True).to(self.device)
            input_length = encoded.input_ids.shape[1]
            with torch.no_grad():
                output = self.model.generate(
                    encoded.input_ids, attention_mask=encoded.attention_mask,
                    max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        finally:
            for block, capture_hidden in zip(self.layers, capture_flags):
                block.capture_hidden = capture_hidden
        return self.tokenizer.batch_decode(output[:, input_length:], skip_special_tokens=True)
