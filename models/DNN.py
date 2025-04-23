import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math

class DNN(nn.Module):
    """
    A deep neural network for the reverse diffusion process.
    Supports both standard and conditional diffusion with similarity vectors.
    """
    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5, conditional=False, n_items=None):
        super(DNN, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm
        self.conditional = conditional
        self.n_items = n_items

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        if self.time_type == "cat":
            if conditional and n_items is not None:
                # For conditional model, add space for similarity vector
                in_dims_temp = [self.in_dims[0] + self.time_emb_dim + n_items] + self.in_dims[1:]
                print(f"in_dims_temp ({in_dims_temp}) = [self.in_dims[0] ({self.in_dims[0]}) + self.time_emb_dim ({self.time_emb_dim}) + n_items ({n_items})] + self.in_dims[1:] ({self.in_dims[1:]}) ")
            else:
                # Standard non-conditional model
                in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
        out_dims_temp = self.out_dims
        
        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])
        
        self.drop = nn.Dropout(dropout)
        self.init_weights()
    
    def init_weights(self):
        for layer in self.in_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.out_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)
    
    def forward(self, x, timesteps, cond_vec=None):
        """
        Forward pass with optional conditional input
        x: input tensor
        timesteps: diffusion timesteps
        cond_vec: optional similarity vector for conditional diffusion
        """
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x)
        x = self.drop(x)
        
        # Handle conditional vs non-conditional forward pass
        if self.conditional and cond_vec is not None:
            # Concatenate input, time embedding, and condition vector
            #print(f"Conditional vector shape: {cond_vec.shape}")
            h = torch.cat([x, emb, cond_vec], dim=-1)
            #print(f"Concatenated tensor shape: {h.shape}")
            #print(f"First layer weight shape: {self.in_layers[0].weight.shape}")
        else:
            # Standard non-conditional forward pass
            h = torch.cat([x, emb], dim=-1)
        
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)
        
        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)
        
        return h


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding