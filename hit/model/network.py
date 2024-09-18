""" The code is based on https://github.com/lioryariv/idr with adaption. """

import numpy as np
import torch
import torch.nn as nn
import torchvision
from hit.model.helpers import expand_cond, grid_sample_feat, mask_dict

class MySoftplus(nn.Softplus):
    
    def __init__(self, beta=100, threshold=20, **kwargs):
        super(MySoftplus, self).__init__(beta=beta, threshold=threshold)

        
    

class ImplicitNetworkNew(nn.Module):
    def __init__(
            self,
            d_in,
            d_out,
            width,
            depth,
            geometric_init=True,
            bias=1.0,
            skip_in=-1,
            mask=False,
            weight_norm=True,
            multires=0,
            pose_cond_layer=-1,
            pose_cond_dim=-1,
            pose_embed_dim=-1,
            shape_cond_layer=-1,
            shape_cond_dim=-1,
            shape_embed_dim=-1,
            latent_cond_layer=-1,
            latent_cond_dim=-1,
            latent_embed_dim=-1,
            feat_cond_dim=0,
            feat_cond_layer=[],
            dropout = 0,
            **kwargs):
        """
        Initializes the Occupancy Network with MLP from torchvision.

        Args:
            d_in (int): Input dimensionality.
            d_hidden (int): Hidden layer dimensionality.
            d_out (int): Output dimensionality.
            n_layers (int): Number of hidden layers.
            use_batchnorm (bool): Whether to use batch normalization.
        """
        super().__init__()
        self.multires = multires
        self.cond_labels = [] 
        use_batchnorm = weight_norm
        
        # import ipdb; ipdb.set_trace()
        d_input = d_in
        if multires > 0:
            # Change the input dimensionality to account for positional encoding
            d_input = d_in + d_in * 2 * multires
            
        # Add conditional inputs size to the input dimensionality
        if len(pose_cond_layer) > 0:
            d_input += pose_cond_dim
            self.cond_labels.append('thetas')
        if len(shape_cond_layer) > 0:
            d_input += shape_cond_dim
            self.cond_labels.append('betas')
        if len(latent_cond_layer) > 0:
            d_input += latent_cond_dim
            self.cond_labels.append('latent')

        # Creating the MLP layers
        self.mlp = torchvision.ops.MLP(
            in_channels=d_input,
            hidden_channels=[width] * depth,
            activation_layer=MySoftplus, # Should be nn.Softplus(beta=100) but not working
            norm_layer=nn.BatchNorm1d if use_batchnorm else None,
            dropout=dropout,
        )
        
        # Output layer
        self.output_layer = nn.Linear(width, d_out)
        # self.output_activation = nn.Softplus(beta=100)  # Assuming the output is a probability
        
        # Apply geometric initialization if specified
        if geometric_init:
            self.apply(self._geometric_init)

    def _geometric_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight,  mean=0, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        
    def forward(self, x, cond, **kwargs):
        #TODO clean the arguments
        """
        Forward pass through the network.

        Args:
            x (tensor): Input tensor.

        Returns:
            tensor: Output of the network.
        """
        # import ipdb; ipdb.set_trace()
        
        cond = {key:cond[key] for key in cond if key in self.cond_labels}
        
        input_dim = x.ndim
        if input_dim == 3:
            # Bring the input to 2D and propagate the conditional input to have the same size
            try:
                n_batch, n_point, n_dim = x.shape     
                mask = torch.ones( (n_batch, n_point), device=x.device, dtype=torch.bool)                     
                cond = { key:expand_cond(cond[key], x) for key in cond if key in self.cond_labels if key is not None}
                cond = mask_dict(cond, mask)
                x = x[mask]
                
            except Exception as e:
                print(f'Error in forward pass of ImplicitNetwork: {e}')
                import ipdb; ipdb.set_trace()
        
        # if 'betas' in self.cond_labels:
        #     import ipdb; ipdb.set_trace()
        if self.multires > 0:
            x = self.pos_encoding(x, self.multires)
            
        #append conditional information
        for key, val in cond.items():
            if key in self.cond_labels:
                    x = torch.cat([x, val], -1)
        
        x = self.mlp(x)
        x = self.output_layer(x)
        # x = self.output_activation(x)
              
        if input_dim == 3:
            # Bring the output back to 3D
            x_full = torch.ones( (n_batch, n_point, x.shape[-1]), device=x.device)
            x_full[mask] = x
            x = x_full
            
        return x

    def pos_encoding(self, p, L):
        out = [p]

        for i in range(L):
            out.append(torch.sin(2**i*p))
            out.append(torch.cos(2**i*p))

        return torch.cat(out, 1)

class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            d_in,
            d_out,
            width,
            depth,
            geometric_init=True,
            bias=1.0,
            skip_in=-1,
            weight_norm=True,
            multires=0,
            pose_cond_layer=-1,
            pose_cond_dim=-1,
            pose_embed_dim=-1,
            shape_cond_layer=-1,
            shape_cond_dim=-1,
            shape_embed_dim=-1,
            latent_cond_layer=-1,
            latent_cond_dim=-1,
            latent_embed_dim=-1,
            feat_cond_dim=0,
            feat_cond_layer=[],
            dropout = 0,
            **kwargs
    ):
        super().__init__()
        dims = [d_in] + [width]*depth + [d_out]
        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] = input_ch
            
        self.dropout = dropout

        self.cond_names = []

        self.pose_cond_layer = pose_cond_layer
        self.pose_cond_dim = pose_cond_dim
        self.pose_embed_dim = pose_embed_dim
        if len(pose_cond_layer)>0:
            self.cond_names.append('thetas')
        if pose_embed_dim > 0:
            self.lin_p0 = nn.Linear(pose_cond_dim, pose_embed_dim)
            self.pose_cond_dim = pose_embed_dim

        self.shape_cond_layer = shape_cond_layer
        self.shape_cond_dim = shape_cond_dim
        self.shape_embed_dim = shape_embed_dim
        if len(shape_cond_layer)>0:
            self.cond_names.append('betas')
        if shape_embed_dim > 0:
            self.lin_p1 = nn.Linear(shape_cond_dim, shape_embed_dim)
            self.shape_cond_dim = shape_embed_dim

        self.latent_cond_layer = latent_cond_layer
        self.latent_cond_dim = latent_cond_dim
        self.latent_embed_dim = latent_embed_dim
        if len(latent_cond_layer)>0:
            self.cond_names.append('latent')
        if latent_embed_dim > 0:
            self.lin_p2 = nn.Linear(latent_cond_dim, latent_embed_dim)
            self.latent_cond_dim = latent_embed_dim

        self.feat_cond_layer = feat_cond_layer
        self.feat_cond_dim = feat_cond_dim
        
        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 == self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            input_dim = dims[l]
            if l in self.pose_cond_layer:
                input_dim += self.pose_cond_dim
            if l in self.shape_cond_layer:
                input_dim += self.shape_cond_dim
            if l in self.latent_cond_layer:
                input_dim += self.latent_cond_dim
            if l in self.feat_cond_layer:
                input_dim += self.feat_cond_dim

            lin = nn.Linear(input_dim, out_dim)
            
            dp =  nn.Dropout(p=self.dropout)
            setattr(self, "dp" + str(l), dp)
            
            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(input_dim), std=0.0001)
                    torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight, 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    if l in self.latent_cond_layer:
                        torch.nn.init.normal_(lin.weight[:, -latent_cond_dim:], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l == self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(input_ch - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)
        self.activation = nn.Softplus(beta=100)
        # self.activation = nn.Softplus(beta=1)
        # self.activation = nn.LeakyReLU()

    def forward(self, input, cond, input_feat=None,  mask=None, return_feat=False, spatial_feat=False, val_pad=0, normalize=False):
        
        input_dim = input.ndim

        if normalize:
            input = input.clone()
            input[..., 1] += 0.28 
            input[...,-1] *= 4

        if input_dim == 3:
            n_batch, n_point, n_dim = input.shape
            if mask is None:
                mask = torch.ones( (n_batch, n_point), device=input.device, dtype=torch.bool)

            if spatial_feat:
                cond = { key:grid_sample_feat(cond[key], input) for key in cond if key in self.cond_names}
            else:
                cond = { key:expand_cond(cond[key], input) for key in cond if key in self.cond_names}

            cond = mask_dict(cond, mask)

            input = input[mask]

            if len(self.feat_cond_layer) > 0:
                input_feat = input_feat[mask]

        if len(self.pose_cond_layer) > 0:
            input_pose_cond = cond['thetas']
            if self.pose_embed_dim>0:
                input_pose_cond = self.lin_p0(input_pose_cond)

        if len(self.shape_cond_layer) > 0:
            input_shape_cond = cond['betas']
            if self.shape_embed_dim>0:
                input_shape_cond = self.lin_p1(input_shape_cond)      

        if len(self.latent_cond_layer) > 0:
            input_latent_cond = cond['latent']
            if self.latent_embed_dim>0:
                input_shape_cond = self.lin_p2(input_latent_cond)     
        
        input_embed = input if self.embed_fn is None else self.embed_fn(input)

        x = input_embed

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.pose_cond_layer:
                x = torch.cat([x, input_pose_cond], dim=-1)

            if l in self.shape_cond_layer:
                x = torch.cat([x, input_shape_cond], dim=-1)

            if l in self.latent_cond_layer:
                x = torch.cat([x, input_latent_cond], dim=-1)

            if l in self.feat_cond_layer:
                x = torch.cat([x, input_feat], dim=-1)

            if l == self.skip_in:
                x = torch.cat([x, input_embed], dim=-1) / np.sqrt(2)

            x = lin(x)
            
            if self.dropout!=0:
                dp = getattr(self, "dp" + str(l))
                x = dp(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
                if return_feat:
                    feat = x.clone()

        if input_dim == 3:
            x_full = torch.ones( (n_batch, n_point, x.shape[-1]), device=x.device) * val_pad
            x_full[mask] = x
            x = x_full

            if return_feat:
                feat_full = torch.ones( (n_batch, n_point, feat.shape[-1]), device=x.device) * val_pad
                feat_full[mask] = feat
                feat = feat_full
                
        if x.isnan().any():
            print('NAN in forward pass')
            import ipdb; ipdb.set_trace()
                
        if return_feat:
            return x, feat
        else:
            return x
        
""" Positional encoding embedding. Code was taken from https://github.com/bmild/nerf. """
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires):
    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim