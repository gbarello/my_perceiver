import torch 
from torch import nn
from mytorch import mytorch

class ffw(nn.Module):
    '''
    Description: A simple feed forward network class. Two layers with a gated latent layer and optional normalization.

    __init__:
        Parameters:
            in_dim (int) - dimention of the input data
            scale (int) - number of times by which to expand the data dimention in the inner layer (Default: 2)
            norm (bool) - If True then apply LayerNorm to input
            
        Returns: 
            None

    forward:
        Parameters:
            self
            x (Tensor) - Tensor of shape [... , in_dim]
            
        Returns:
            (Tensor) - also of shape [... , in_dim]

    '''
    def __init__(self,in_dim, scale = 2, norm = True):
        super().__init__()
        
        self.inp = nn.Linear(in_dim, in_dim * scale * 2)
        self.out = nn.Linear(in_dim * scale , in_dim)
        self.nonlin = nn.GELU()
        self.norm = nn.LayerNorm(in_dim) if norm else nn.Identity
        
    def forward(self,x):
        inner_units = self.inp(self.norm(x))
        
        latent, gate = torch.split(inner_units,inner_units.shape[-1]//2,-1)
        latent = latent * self.nonlin(gate)
        out = self.out(latent)
        
        return x + out

class AttentionModule(nn.Module):
    '''
    Description: Applies attention, either cross-attention or self-attention, to its inputs.
    
    __init__:
        Parameters: 
            self
            data_channels (int) - number of channels (final dimention) of the required input (the input which generates K-V pairs in cross-attention)
            head_d (int) - dimention of the 
    
    '''
    def __init__(self, data_channels, head_d, n_heads, output_dim, latent_channels = None, dropout = .2, **kwargs):
        super().__init__()
        self.n_heads = n_heads
        self.head_d = head_d
        
        inner_dim = head_d * n_heads
        latent_channels = latent_channels if latent_channels else data_channels
            
        self.attn_scaling = head_d ** -0.5
        
        self.Q_net = nn.Linear(latent_channels, inner_dim, bias = False)
        self.K_net = nn.Linear(data_channels, inner_dim, bias = False)
        self.V_net = nn.Linear(data_channels, inner_dim, bias = False)
                
        self.output = nn.Sequential(nn.Linear(inner_dim, output_dim),nn.Dropout(dropout))
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, data, latent = None, **kwargs):
        if latent is None:
            latent = data
            
        b, ds, dc = data.shape
        b, ls, lc = latent.shape
            
        Q = self.Q_net(latent).view([b,ls,self.n_heads,self.head_d])
        K = self.K_net(data).view([b,ds,self.n_heads,self.head_d])
        V = self.V_net(data).view([b,ds,self.n_heads,self.head_d])
        
        attn_logits = torch.sum(
            torch.unsqueeze(Q,1) * torch.unsqueeze(K,2)  * self.attn_scaling,
            -1)
        
        #softmax over the data dimention
        attn = attn_logits.softmax(1)
                
        #align/dot product and sum over data dimention
        output = torch.sum(torch.unsqueeze(V,2)*torch.unsqueeze(attn,-1),1).view([b,ls,-1])

        return self.output(output)

class Perceiver(nn.Module):
    def __init__(
        self, 
        data_channels, 
        latent_dimention, 
        latent_channels, 
        n_attends,
        n_transform_per_attend,
        share_params = False, 
        share_first_params = False,
        n_heads = 8,
        head_dim = 16,
        dropout = .5,
        output_dimention = None,
        initial_latents = None,
        ffw_depth = 1,
    ):
        '''
        Parameters:
            data_channels (int) - number of channels in the input data
            latent_dimention (int) - number of latents
            latent_channels (int) - number of dimentions in each latent
            n_attends (int) - number of attends to the data
            n_transform_per_attend (int) - number of transformer layers between attention instances
            share_params (bool) - if True share parameters after first attention block
            share_first_params (bool) - if True, share parameters between first block and later blocks (all block)
            n_heads (int) - number of attention heads
            head_dim (int) - dimention of each attention head
            dropout (float) - dropout parameter to apply
            output_dimention (int or None) - if None, return latent tensor, else average latents and return a final dense layer of dimention output_dimention
            initial_latents (Tensor or None) - if given, initialize the initial latent array to this value
            ffw_depth (int) - depth of the feed-forward networks at each block.
        
        '''
        
        super().__init__()
        
        if share_first_params == True:
            share_params = True
        self.share_first_params = share_first_params
        self.share_params = share_params
        self.n_attends = n_attends
        self.n_transform_per_attend = n_transform_per_attend
        
        self.initial_latents = nn.Parameter(torch.randn(latent_dimention, latent_channels) if initial_latents is None else initial_latents)
        
        self.initial_cross_attention = AttentionModule(data_channels, head_dim, n_heads, latent_channels, latent_channels, dropout)
        self.initial_ca_ff = nn.Sequential(*[ffw(latent_channels,latent_channels) for _ in range(ffw_depth)])
        self.initial_transformer = nn.Sequential(*[AttentionModule(latent_channels, head_dim, n_heads, latent_channels, dropout = dropout) for _ in range(n_transform_per_attend)])
        self.initial_tr_ff = nn.Sequential(*[ffw(latent_channels,latent_channels) for _ in range(ffw_depth)])
        
        if self.share_first_params == False:
            if self.share_params == False:
                self.cross_attention = nn.ModuleList([
                    AttentionModule(data_channels, head_dim, n_heads, latent_channels, latent_channels, dropout)
                    for _ in range(n_attends)
                ])
                self.ca_ff = nn.ModuleList([
                    nn.Sequential(*[ffw(latent_channels,latent_channels) for _ in range(ffw_depth)])
                    for _ in range(n_attends)
                ])
                self.transformer = nn.ModuleList([
                    nn.Sequential(*[AttentionModule(latent_channels, head_dim, n_heads, latent_channels, dropout = dropout) for _ in range(n_transform_per_attend)]) 
                    for _ in range(n_attends)
                ])
                self.tr_ff = nn.ModuleList([
                    nn.Sequential(*[ffw(latent_channels,latent_channels) for _ in range(ffw_depth)])
                    for _ in range(n_attends)
                ])
            else:
                self.cross_attention = AttentionModule(data_channels, head_dim, n_heads, latent_channels, latent_channels, dropout)
                self.ca_ff = nn.Sequential(*[ffw(latent_channels,latent_channels) for _ in range(ffw_depth)])
                self.transformer = nn.Sequential(*[AttentionModule(latent_channels, head_dim, n_heads, latent_channels, dropout = dropout) for _ in range(n_transform_per_attend)])
                self.tr_ff = nn.Sequential(*[ffw(latent_channels,latent_channels) for _ in range(ffw_depth)])
                
        self.output_layer = None if output_dimention is None else nn.Sequential(nn.LayerNorm(latent_channels),nn.Linear(latent_channels, output_dimention))
        
    def forward(self, data, n_attends = None, n_transform_per_attend = None):
        if n_attends is None or self.share_params is False:
            n_attends = self.n_attends
                
        latents = torch.unsqueeze(self.initial_latents, 0).repeat([data.shape[0]] + [1 for s in self.initial_latents.shape])
        latents = self.initial_cross_attention(data, latents)
        latents = self.initial_transformer(latents)
        
        for i in range(n_attends):
            if self.share_params == False:
                CA_mod = self.cross_attention[i]
                ca_mod = self.ca_ff[i]
                T_mod = self.transformer[i]
                tr_mod = self.tr_ff[i]
                
            elif self.share_first_params == False:
                CA_mod = self.cross_attention
                ca_mod = self.ca_ff
                T_mod = self.transformer
                tr_mod = self.tr_ff
            else:
                CA_mod = self.initial_cross_attention
                ca_mod = self.initial_ca_ff
                T_mod = self.initial_transformer
                tr_mod = self.initial_tr_ff
                
            latents = (CA_mod(data,latents) + latents)/2
            latents = ca_mod(latents)
            latents = (T_mod(latents) + latents)/2
            latents = tr_mod(latents)
            
        if self.output_layer is not None:
            return self.output_layer(torch.mean(latents,-2))
        return latents