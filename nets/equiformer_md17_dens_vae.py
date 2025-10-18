from locale import normalize
import torch
from torch_cluster import radius_graph
from torch_scatter import scatter

import e3nn
from e3nn import o3
from e3nn.util.jit import compile_mode
from e3nn.nn.models.v2106.gate_points_message_passing import tp_path_exists

import torch_geometric
import math

from .registry import register_model
from .instance_norm import EquivariantInstanceNorm
from .graph_norm import EquivariantGraphNorm
from .layer_norm import EquivariantLayerNormV2
from .radial_func import RadialProfile
from .tensor_product_rescale import (TensorProductRescale, LinearRS,
    FullyConnectedTensorProductRescale, irreps2gate, sort_irreps_even_first)
from .fast_activation import Activation, Gate
from .drop import EquivariantDropout, EquivariantScalarsDropout, GraphDropPath
from .gaussian_rbf import GaussianRadialBasisLayer
import torch.nn as nn
import torch.nn.functional as F

# for bessel radial basis
from ocpmodels.models.gemnet.layers.radial_basis import RadialBasis

from .graph_attention_transformer import (get_norm_layer, 
    FullyConnectedTensorProductRescaleNorm, 
    FullyConnectedTensorProductRescaleNormSwishGate, 
    FullyConnectedTensorProductRescaleSwishGate,
    DepthwiseTensorProduct, SeparableFCTP,
    Vec2AttnHeads, AttnHeads2Vec,
    GraphAttention, FeedForwardNetwork, 
    TransBlock, 
    NodeEmbeddingNetwork, EdgeDegreeEmbeddingNetwork, ScaledScatter
)
from .graph_attention_transformer_md17 import (
    CosineCutoff, 
    ExpNormalSmearing
)
from.equiformer_md17_dens import (
    Equiformer_MD17_DeNS,
)

_RESCALE = True
_USE_BIAS = True

_MAX_ATOM_TYPE = 64 # Set to some large value

# Statistics of QM9 with cutoff radius = 5
# For simplicity, use the same statistics for MD17
_AVG_NUM_NODES = 18.03065905448718
_AVG_DEGREE = 15.57930850982666
      

class Equiformer_MD17_DeNS_VAE(torch.nn.Module):
    def __init__(self,
        irreps_in='64x0e',
        irreps_equivariant_inputs='1x0e+1x1e+1x2e',     # for encoding forces during denoising positions
        irreps_node_embedding='128x0e+64x1e+32x2e', 
        num_layers=6,
        irreps_node_attr='1x0e', 
        irreps_sh='1x0e+1x1e+1x2e',
        max_radius=5.0,
        number_of_basis=32, 
        basis_type='exp', 
        fc_neurons=[64, 64], 
        irreps_feature='512x0e+256x1e+128x2e',          # increase numbers of channels by 4 times
        irreps_head='32x0e+16x1o+8x2e', 
        num_heads=4, 
        irreps_pre_attn='128x0e+64x1e+32x2e',
        rescale_degree=False, 
        nonlinear_message=True,
        irreps_mlp_mid='128x0e+64x1e+32x2e',
        norm_layer='layer',
        alpha_drop=0.0, 
        proj_drop=0.0, 
        out_drop=0.0,
        drop_path_rate=0.0,
        mean=None, 
        std=None, 
        scale=None, 
        atomref=None,
        use_force_encoding=True,                        # for ablation study
        pretraining=False,
        onlyEnergy=False,
        test_type="AniDS",
        fix_encoder_parameters=True,
        kappa_target = 0.30,                  # 
        gamma_reg_w = 1.0,
    ):
        self.kappa_target = kappa_target
        self.gamma_reg_w = gamma_reg_w
        if fix_encoder_parameters:
            temp_irreps_in='64x0e'
            temp_irreps_equivariant_inputs='1x0e+1x1e+1x2e'  # for encoding forces during denoising positions
            temp_irreps_node_embedding='64x0e+32x1e+16x2e'
            temp_num_layers=2
            temp_irreps_node_attr='1x0e'
            temp_irreps_sh='1x0e+1x1e+1x2e'
            temp_max_radius=max_radius
            temp_number_of_basis=number_of_basis
            temp_basis_type=basis_type
            temp_fc_neurons=[32, 32]
            temp_irreps_feature='256x0e+128x1e+64x2e'      # increase numbers of channels by 4 times
            temp_irreps_head='32x0e+16x1e+8x2e' 
            temp_num_heads=num_heads
            temp_irreps_pre_attn='64x0e+32x1e+16x2e'
            temp_rescale_degree=rescale_degree
            temp_nonlinear_message=nonlinear_message
            temp_irreps_mlp_mid='192x0e+96x1e+48x2e'
        else:
            temp_irreps_in=irreps_in
            temp_irreps_equivariant_inputs=irreps_equivariant_inputs # for encoding forces during denoising positions
            temp_irreps_node_embedding=irreps_node_embedding
            temp_num_layers=2
            temp_irreps_node_attr=irreps_node_attr
            temp_irreps_sh=irreps_sh
            temp_max_radius=max_radius
            temp_number_of_basis=number_of_basis
            temp_basis_type=basis_type
            temp_fc_neurons=fc_neurons
            temp_irreps_feature=irreps_feature     # increase numbers of channels by 4 times
            temp_irreps_head=irreps_head
            temp_num_heads=num_heads
            temp_irreps_pre_attn=irreps_pre_attn
            temp_rescale_degree=rescale_degree
            temp_nonlinear_message=nonlinear_message
            temp_irreps_mlp_mid=irreps_mlp_mid


        super().__init__()
        self.test_type = test_type
        self.pretraining = pretraining
        self.onlyEnergy = onlyEnergy

        self.max_radius = temp_max_radius
        self.number_of_basis = temp_number_of_basis
        self.alpha_drop = alpha_drop
        self.proj_drop = proj_drop
        self.out_drop = out_drop
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.task_mean = mean
        self.task_std = std
        self.scale = scale
        self.register_buffer('atomref', atomref)
        self.use_force_encoding = use_force_encoding
        
        self.irreps_node_attr   = o3.Irreps(temp_irreps_node_attr)
        self.irreps_node_input  = o3.Irreps(temp_irreps_in)
        self.irreps_node_equivariant_inputs = o3.Irreps(temp_irreps_equivariant_inputs)  # for encoding forces during denoising positions
        self.irreps_node_embedding = o3.Irreps(temp_irreps_node_embedding)
        self.lmax = self.irreps_node_embedding.lmax
        self.irreps_feature = o3.Irreps(temp_irreps_feature)
        self.num_layers     = num_layers
        self.irreps_edge_attr = o3.Irreps(temp_irreps_sh) if temp_irreps_sh is not None \
            else o3.Irreps.spherical_harmonics(self.lmax)
        self.fc_neurons = [self.number_of_basis] + temp_fc_neurons
        self.irreps_head    = o3.Irreps(temp_irreps_head)
        self.num_heads      = temp_num_heads
        self.irreps_pre_attn    = temp_irreps_pre_attn
        self.rescale_degree     = temp_rescale_degree
        self.nonlinear_message  = temp_nonlinear_message
        self.irreps_mlp_mid     = o3.Irreps(temp_irreps_mlp_mid)

        self.basis_type = temp_basis_type
        if self.basis_type == 'gaussian':
            self.rbf = GaussianRadialBasisLayer(self.number_of_basis, cutoff=self.max_radius)
        elif self.basis_type == 'bessel':
            self.rbf = RadialBasis(self.number_of_basis, cutoff=self.max_radius, 
                rbf={'name': 'spherical_bessel'})
        elif self.basis_type == 'exp':
            self.rbf = ExpNormalSmearing(cutoff_lower=0.0, cutoff_upper=self.max_radius, 
                num_rbf=self.number_of_basis, trainable=False)
        else:
            raise ValueError
        
        irreps_feature_scalars = []
        for mul, ir in self.irreps_feature:
            if (ir.l == 0) and (ir.p == 1):
                irreps_feature_scalars.append((mul, ir))
        irreps_feature_scalars = o3.Irreps(irreps_feature_scalars)
        
        self.log_ai = torch.nn.Sequential(
            LinearRS(self.irreps_feature, irreps_feature_scalars, rescale=_RESCALE), 
            Activation(irreps_feature_scalars, acts=[torch.nn.SiLU()]),
            LinearRS(irreps_feature_scalars, o3.Irreps('1x0e'), rescale=_RESCALE)
        )

        self.linear_scalar=LinearRS(self.irreps_feature, irreps_feature_scalars, rescale=_RESCALE)
        self.linear_edge=LinearRS(o3.Irreps([(self.number_of_basis,"0e")]),irreps_feature_scalars, rescale=_RESCALE)
        
        self.bij_net = torch.nn.Sequential(
            LinearRS(o3.Irreps([(3*irreps_feature_scalars.num_irreps,"0e")]), irreps_feature_scalars, rescale=_RESCALE),
            Activation(irreps_feature_scalars, acts=[torch.nn.SiLU()]),
            LinearRS(irreps_feature_scalars, o3.Irreps('1x0e'), rescale=_RESCALE)
        )
        
        self.log_ci = torch.nn.Sequential(
            LinearRS(self.irreps_feature, irreps_feature_scalars, rescale=_RESCALE),
            Activation(irreps_feature_scalars, acts=[torch.nn.SiLU()]),
            LinearRS(irreps_feature_scalars, o3.Irreps('1x0e'), rescale=_RESCALE)
        )

        self.encoder=Equiformer_MD17_DeNS(
            irreps_in=temp_irreps_in,
            irreps_equivariant_inputs=temp_irreps_equivariant_inputs,   # for encoding forces during denoising positions
            irreps_node_embedding=temp_irreps_node_embedding,
            num_layers=temp_num_layers,
            irreps_node_attr=temp_irreps_node_attr,
            irreps_sh=temp_irreps_sh,
            max_radius=temp_max_radius,
            number_of_basis=temp_number_of_basis,
            basis_type=temp_basis_type,
            fc_neurons=temp_fc_neurons,
            irreps_feature=temp_irreps_feature,        # increase numbers of channels by 4 times
            irreps_head=temp_irreps_head,
            num_heads=temp_num_heads,
            irreps_pre_attn=temp_irreps_pre_attn,
            rescale_degree=temp_rescale_degree,
            nonlinear_message=temp_nonlinear_message,
            irreps_mlp_mid=temp_irreps_mlp_mid,
            norm_layer=norm_layer,
            alpha_drop=alpha_drop,
            proj_drop=proj_drop,
            out_drop=out_drop,
            drop_path_rate=drop_path_rate,
            mean=mean,
            std=std,
            scale=scale,
            atomref=atomref,
            use_force_encoding=False,
            backprop=False
        )
        self.decoder=Equiformer_MD17_DeNS(
            irreps_in,
            irreps_equivariant_inputs,     # for encoding forces during denoising positions
            irreps_node_embedding, 
            num_layers,
            irreps_node_attr, 
            irreps_sh,
            max_radius,
            number_of_basis, 
            basis_type, 
            fc_neurons, 
            irreps_feature,          # increase numbers of channels by 4 times
            irreps_head, 
            num_heads, 
            irreps_pre_attn,
            rescale_degree, 
            nonlinear_message,
            irreps_mlp_mid,
            norm_layer,
            alpha_drop, 
            proj_drop, 
            out_drop,
            drop_path_rate,
            mean, 
            std, 
            scale, 
            atomref,
            use_force_encoding,
            backprop=not self.pretraining,
            onlyEnergy=onlyEnergy,
        )
        
    def safe_softmax(self, x):
        max_x = torch.max(x, dim=-1, keepdim=True)[0]
        exp_x = torch.exp(x - max_x)
        sum_exp_x = torch.sum(exp_x, dim=-1, keepdim=True)
        safe_softmax = exp_x / (sum_exp_x + 1e-8)
        return safe_softmax

    def compute_covariance_matrix(self, node_feature, edge_index, pos):
        log_ai = self.log_ai(node_feature)  # [N, 1]
        ai = torch.exp(log_ai)              # [N, 1]
        
        src, dst = edge_index
        
        src_feature = node_feature[src]  # [E, C]
        dst_feature = node_feature[dst]  # [E, C]
        
        rel_pos = pos[dst] - pos[src]  # [E, 3]
        rel_pos_norm = torch.norm(rel_pos, dim=1, keepdim=True)  # [E, 1]
        rel_pos_normalized = rel_pos / (rel_pos_norm + 1e-8)  # [E, 3]
        
        edge_length = rel_pos_norm.squeeze(-1)  # [E]
        edge_length_embedding = self.rbf(edge_length)
        
        src_feature = self.linear_scalar(src_feature)  # [E, C]
        dst_feature = self.linear_scalar(dst_feature)
        edge_length_embedding = self.linear_edge(edge_length_embedding)
        edge_feature = torch.cat([src_feature, dst_feature, edge_length_embedding], dim=-1)
        bij = self.bij_net(edge_feature)  # [E, 16]
        
        log_ci = self.log_ci(node_feature)  # [N, 1]
        ci = torch.exp(log_ci)  # [N, 1]
        
        bij_batch_max=scatter(bij, dst, dim=0, dim_size=pos.size(0), reduce='max')  # [N, 1]
        bij_batch_max = bij_batch_max[dst]  # [E, 1]
        bij = bij - bij_batch_max  # [E, 1]
        bij_exp = torch.exp(bij)  # [E, 1]
        bij_sum = scatter(bij_exp, dst, dim=0, dim_size=pos.size(0), reduce='sum')  # [N, 1]
        bij_sum = bij_sum + ci  # [N, 1]

        bij_sum_expanded = bij_sum[dst]  # [E, 1]
        
        # b'_ij = (e^b_ij / (sum_{j,j∈N(i)} e^b_ij + c_i)) * a_i
        gamma_edge = bij_exp / (bij_sum_expanded + 1e-7)
        bij_prime = gamma_edge * ai[dst]  # [E, 1]
        
        # b'_ij * (r_ij / ||r_ij||) ⊗ (r_ij / ||r_ij||)
        outer_product = torch.bmm(
            rel_pos_normalized.unsqueeze(2),  # [E, 3, 1]
            rel_pos_normalized.unsqueeze(1)   # [E, 1, 3]
        )  # [E, 3, 3]
        
        scaled_outer_product = bij_prime.unsqueeze(-1) * outer_product  # [E, 3, 3]
        sigma_contribution = scatter(scaled_outer_product, dst, dim=0, dim_size=pos.size(0), reduce='sum')  # [N, 3, 3]
        
        # Σ_i = a_i * I - sum_{j,j∈N(i)} b'_ij * (r_ij/||r_ij||) ⊗ (r_ij/||r_ij||)
        identity = torch.eye(3, device=pos.device).unsqueeze(0).expand(pos.size(0), -1, -1)  # [N, 3, 3]
        sigma = ai.unsqueeze(-1) * identity - sigma_contribution  # [N, 3, 3]
        
        jitter = 1e-6 * torch.eye(3, device=pos.device).unsqueeze(0)
        sigma = sigma + jitter
        # for regularization
        sum_gamma = scatter(gamma_edge, dst, dim=0, dim_size=pos.size(0), reduce='sum')

        return sigma, ai, sum_gamma

    def generate_anisotropic_noise(self, sigma, pos):
        L = torch.linalg.cholesky(sigma)  # [N, 3, 3]
        standard_noise = torch.randn_like(pos)  # [N, 3]
        anisotropic_noise = torch.bmm(L, standard_noise.unsqueeze(-1)).squeeze(-1)  # [N, 3]
        
        return anisotropic_noise, L,standard_noise

    def cal_kl_loss(self, sigma, std,noise_mask):
        mask = noise_mask.bool()  # noise_mask shape = [N]
        sigma_noise = sigma[mask]        # shape [M, 3, 3]
        if sigma_noise.size(0) == 0:
            return torch.tensor(0.0, device=sigma.device)
        
        target_variance = std ** 2
        k = sigma_noise.size(-1) 
        
        trace_sigma = sigma_noise.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1) #  trace: tr(sigma_i)
        
        sign, logdet_sigma = torch.linalg.slogdet(sigma_noise) 
        logdet_target = k * math.log(target_variance) # det(Σ₀) = (std^2)^k
        
        kl_per_atom = 0.5 * (trace_sigma / target_variance - k + logdet_target - logdet_sigma)
        kl_loss = kl_per_atom.mean()
        return kl_loss
    
    def cal_kl_loss_vae(self, a_i, std,noise_mask):
        std = std * torch.ones_like(a_i)
        kl_loss = -0.5 * (1+a_i - a_i.exp()/std**2 - 2*torch.log(std))
        kl_loss = kl_loss[noise_mask]
        kl_loss = kl_loss.mean()
        return kl_loss


    def forward(self, data):
        if ((self.training or self.pretraining) and "std" in data):            
            std = data["std"]
            prob = data["prob"]
            corrupt_ratio = data["corrupt_ratio"]
            data.pos.requires_grad = False

            if self.test_type == "AniDS":
                node_feature,_,__ = self.encoder(data)
                # sample which examples use denoising pos
                batch_size = data.batch.max() + 1
                denoising_pos_mask = torch.rand(batch_size, dtype=data.pos.dtype, device=data.pos.device) 
                denoising_pos_mask = (denoising_pos_mask < prob)
                denoising_pos_mask = denoising_pos_mask[data.batch]
                data.denoising_pos_mask = denoising_pos_mask
                data.noise_mask = data.denoising_pos_mask

                # for corrupting a subset of atoms
                if corrupt_ratio is not None:
                    corrupt_mask = torch.rand((data.pos.shape[0]), dtype=data.pos.dtype, device=data.pos.device)
                    corrupt_mask = (corrupt_mask < corrupt_ratio)
                    data.corrupt_mask = corrupt_mask
                    data.noise_mask = data.noise_mask * data.corrupt_mask
                
                if self.pretraining:
                    data.force = torch.zeros_like(data.pos)
                else:
                    data.force = data.dy.clone()
                data.force[(~data.noise_mask)] *= 0

                edge_index = radius_graph(data.pos, r=self.max_radius, batch=data.batch,max_num_neighbors = 1000)

                sigma, ai, sum_gamma = self.compute_covariance_matrix(node_feature, edge_index, data.pos)
                noise_vec, L,standard_noise = self.generate_anisotropic_noise(sigma, data.pos)# noise_vec = L standard_noise
                
                data.pos[data.noise_mask] = data.pos[data.noise_mask] + noise_vec[data.noise_mask]

                L = L.transpose(-1, -2)
                L = torch.linalg.inv(L)
                if ai is not None:
                    L = L * ai[data.noise_mask].mean().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # make training more stable
                data.noise_vec = torch.bmm(L, standard_noise.unsqueeze(-1)).squeeze(-1) # supervised noise: L^{-T} * standard_noise
                
                kl_loss = self.cal_kl_loss(sigma, std,data.noise_mask)
                sum_gamma_flat = sum_gamma.squeeze(-1)
                mask = data.noise_mask.bool()
                deficit = (self.kappa_target - sum_gamma_flat[mask]).clamp(min=0.0)
                loss_gamma = (deficit ** 2).mean() if mask.any() else torch.zeros((), device=deficit.device)
                kl_loss = kl_loss + self.gamma_reg_w * loss_gamma
            else:
                kl_loss=torch.tensor(0.0, dtype=data.pos.dtype, device=data.pos.device)
        else:
            kl_loss = 0.0

        energy_outputs, vector_outputs,_ = self.decoder(data)

        if self.pretraining:
            return energy_outputs, vector_outputs,kl_loss,data
        else:
            return energy_outputs, vector_outputs,kl_loss

@register_model
def equiformer_md17_dens_vae(**kwargs):
    return Equiformer_MD17_DeNS_VAE(**kwargs)