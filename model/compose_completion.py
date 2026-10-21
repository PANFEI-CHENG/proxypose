"""ComPose-style progressive completion after CleanPose's FD encoder, decoded
jointly with camera-frame axis-structure proxies.

Reuses the vendored AdaPoinTr geometry-aware attention blocks. This is a
post-keypoint adaptation of ComPose equations 3--10, not its unpublished code.
All coordinates are in the same centered/augmented metric frame as CleanPose.
"""
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from easydict import EasyDict

from model.adapointr.AdaPoinTr import PointTransformerEncoderEntry
from model.adapointr.Transformer_utils import Attention, CrossAttention, DynamicGraphAttention, knn_point
from model.adapointr.compat import pointnet2_utils


def gather_points(points, indices):
    return points.gather(1, indices.long().unsqueeze(-1).expand(-1, -1, points.size(-1)))


def sample_indices(points, count):
    return pointnet2_utils.furthest_point_sample(points.contiguous(), min(count, points.size(1))).long()


def chamfer_squared(pred, target):
    """ComPose eq. 8: sum of two mean squared nearest-neighbor distances."""
    distance = torch.cdist(pred.float(), target.float(), compute_mode='donot_use_mm_for_euclid_dist').square()
    return distance.amin(2).mean() + distance.amin(1).mean()


def canonical_to_camera(model, rotation, translation, size):
    scale = size.norm(dim=-1).clamp_min(1e-8)[:, None, None]
    return (model * scale) @ rotation.transpose(1, 2) + translation[:, None]


class CompletionDecoderLayer(nn.Module):
    """Global interaction for all queries; KNN graph attention only for surface queries.

    Axis proxies lie on the object axes rather than on its surface, so they are
    never KNN queries or neighbors.
    """
    def __init__(self, cfg, local):
        super().__init__()
        dim = cfg.embed_dim
        self.norm_self, self.norm_q, self.norm_v, self.norm_ffn = (nn.LayerNorm(dim) for _ in range(4))
        self.self_attn = Attention(dim, cfg.num_heads)
        self.cross_attn = CrossAttention(dim, dim, cfg.num_heads)
        self.ffn = nn.Sequential(nn.Linear(dim, int(dim * cfg.mlp_ratio)), nn.GELU(),
                                 nn.Linear(int(dim * cfg.mlp_ratio), dim))
        self.local = local
        if local:
            self.local_self = DynamicGraphAttention(dim, k=cfg.k)
            self.local_cross = DynamicGraphAttention(dim, k=cfg.k)
            self.merge_self, self.merge_cross = nn.Linear(2 * dim, dim), nn.Linear(2 * dim, dim)

    def forward(self, q, memory, surface_xyz, memory_xyz, self_idx, cross_idx):
        count = surface_xyz.size(1)
        norm_q = self.norm_self(q)
        update = self.self_attn(norm_q)
        if self.local:
            local = self.local_self(norm_q[:, :count], surface_xyz, idx=self_idx)
            surface = self.merge_self(torch.cat((update[:, :count], local), dim=-1))
            update = torch.cat((surface, update[:, count:]), dim=1)
        q = q + update
        norm_q, norm_v = self.norm_q(q), self.norm_v(memory)
        update = self.cross_attn(norm_q, norm_v)
        if self.local:
            local = self.local_cross(norm_q[:, :count], surface_xyz, v=norm_v,
                                     v_pos=memory_xyz, idx=cross_idx)
            surface = self.merge_cross(torch.cat((update[:, :count], local), dim=-1))
            update = torch.cat((surface, update[:, count:]), dim=1)
        q = q + update
        return q + self.ffn(self.norm_ffn(q))


class CompletionDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.k = cfg.k
        self.blocks = nn.ModuleList(CompletionDecoderLayer(cfg, local=i == 0) for i in range(cfg.depth))

    def forward(self, q, memory, surface_xyz, memory_xyz):
        self_idx = knn_point(self.k, surface_xyz, surface_xyz)
        cross_idx = knn_point(self.k, memory_xyz, surface_xyz)
        layers = []
        for block in self.blocks:
            q = block(q, memory, surface_xyz, memory_xyz, self_idx, cross_idx)
            layers.append(q)
        return layers


class ComposeCompletion(nn.Module):
    def __init__(self, cfg, feat_dim):
        super().__init__()
        cfg = EasyDict(deepcopy(dict(cfg)))
        self.num_query = cfg.get('num_query', 64)
        self.num_missing = cfg.get('num_missing', 64)
        self.num_visible = cfg.get('num_visible', 32)
        self.factor = cfg.get('points_per_keypoint', 16)
        self.axis_points = cfg.axis_points_per_axis
        assert 3 * self.axis_points <= self.num_query
        ticks = torch.linspace(-cfg.axis_half_length, cfg.axis_half_length, self.axis_points)
        axes = (torch.eye(3)[:, None, :] * ticks[None, :, None]).flatten(0, 1)
        self.register_buffer('axis_template', axes)
        assert cfg.encoder_config.embed_dim == cfg.decoder_config.embed_dim == feat_dim
        self.keypoint_proj = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, feat_dim))
        self.position = nn.Sequential(nn.Linear(3, feat_dim), nn.GELU(), nn.Linear(feat_dim, feat_dim))
        # self.encoder = PointTransformerEncoderEntry(cfg.encoder_config)
        self.decoder = CompletionDecoder(cfg.decoder_config)
        self.global_proj = nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.GELU(), nn.LayerNorm(feat_dim))
        self.missing_head = nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.GELU(),
                                          nn.Linear(feat_dim, 3 * self.num_missing))
        # Each proxy query is identified by its object-frame coordinate; a shared
        # head regresses its camera-frame coordinate after every decoder layer.
        self.axis_embed = nn.Sequential(nn.Linear(3, feat_dim), nn.GELU(), nn.Linear(feat_dim, feat_dim))
        self.axis_head = nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.GELU(), nn.Linear(feat_dim, 3))
        # Global conditioning lets the same coordinate receive different scores
        # for different observed shapes; the target remains ComPose eq. 10.
        self.score_head = nn.Sequential(nn.Linear(feat_dim + 3, feat_dim), nn.GELU(),
                                        nn.Linear(feat_dim, 1), nn.Sigmoid())
        self.output_norm = nn.LayerNorm(feat_dim)
        self.refine_head = nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.GELU(), nn.Linear(feat_dim, 3))
        self.fold_head = nn.Sequential(nn.Linear(feat_dim + 3, feat_dim), nn.GELU(),
                                       nn.Linear(feat_dim, self.factor * 3))
        # Small, nonzero initial offsets keep the metric coordinates sensible
        # without blocking the first backward pass to the preceding layers.
        for head in (self.missing_head, self.refine_head, self.fold_head, self.axis_head):
            nn.init.normal_(head[-1].weight, std=0.001)
            nn.init.zeros_(head[-1].bias)

    def forward(self, keypoints, features, pts, point_features):
        # Geometry guides feature learning without moving the detector to ease attention.
        # Explicit positions let global attention regress coordinates in this frame.
        xyz = keypoints
        memory = self.keypoint_proj(features) + self.position(xyz)
        # memory = self.encoder(memory, xyz)
        global_feature = self.global_proj(memory.amax(dim=1))
        missing = self.missing_head(global_feature).view(-1, self.num_missing, 3)
        # Candidates remain based on the detected keypoints, rather than moving
        # completion ahead of keypoint detection as in the original ComPose.
        visible = gather_points(keypoints, sample_indices(keypoints, self.num_visible))
        candidates = torch.cat((missing, visible), dim=1)
        scores = self.score_head(torch.cat((candidates.detach(), global_feature[:, None].expand(
            -1, candidates.size(1), -1)), dim=-1)).squeeze(-1)
        selected = scores.topk(self.num_query, dim=1).indices
        coarse = gather_points(candidates, selected)
        surface_query = global_feature[:, None] + self.position(coarse.detach())
        axis_query = global_feature[:, None] + self.axis_embed(self.axis_template)
        layers = [self.output_norm(q) for q in self.decoder(
            torch.cat((surface_query, axis_query), dim=1), memory, coarse.detach(), xyz)]
        decoded, axis_features = layers[-1][:, :self.num_query], layers[-1][:, self.num_query:]
        # Proxies are regressed directly in the camera frame at every layer, for deep supervision.
        axis_layers = torch.stack([self.axis_head(q[:, self.num_query:]) for q in layers], dim=1)
        refined = coarse + self.refine_head(decoded)
        offsets = self.fold_head(torch.cat((decoded, refined), dim=-1)).view(-1, self.num_query, self.factor, 3)
        dense = (refined[:, :, None] + offsets).flatten(1, 2)
        return {
            'completion_missing': missing,
            'completion_candidates': candidates,
            'completion_scores': scores,
            'completion_coarse': coarse,
            'completion_keypoints': refined,
            'completion_dense': dense,
            'recon_delta': offsets.flatten(1, 2),
            'axis_layers': axis_layers,
            'axis_points': axis_layers[:, -1],
            'axis_features': axis_features,
        }, decoded


class ComposeCompletionLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, endpoints, target):
        losses = {name: chamfer_squared(endpoints[key], target) for name, key in (
            ('loss_com_missing', 'completion_missing'),
            ('loss_com_keypoints', 'pred_kpt_3d'),
            ('loss_com_dense', 'recon_model'),
        )}
        with torch.no_grad():
            distances = torch.cdist(endpoints['completion_candidates'].float(), target.float(),
                                    compute_mode='donot_use_mm_for_euclid_dist').amin(dim=-1)
            score_target = torch.exp(-distances / self.temperature)
        losses['loss_score'] = F.mse_loss(endpoints['completion_scores'], score_target)
        return losses


def axis_target(pred, template, rotation, translation, size, symmetric):
    """Camera-frame GT proxies. For y-symmetric objects the GT frame is first turned
    about its y-axis by the yaw that best fits `pred`; this maximizes
    sum_k <R^T (p_k - t), R_y(yaw) c_k> and has the closed form below.
    """
    local = (pred - translation[:, None]) @ rotation
    x, z = template[..., 0], template[..., 2]
    yaw = torch.atan2((local[..., 0] * z - local[..., 2] * x).sum(1),
                      (local[..., 0] * x + local[..., 2] * z).sum(1)) * symmetric
    cos, sin, zero, one = yaw.cos(), yaw.sin(), torch.zeros_like(yaw), torch.ones_like(yaw)
    turn = torch.stack((cos, zero, sin, zero, one, zero, -sin, zero, cos), dim=1).view(-1, 3, 3)
    return canonical_to_camera(template, rotation @ turn, translation, size)


def axis_proxy_loss(layers, template, rotation, translation, size, symmetric):
    """Proxy distance in units of the object scale, summed over decoder layers."""
    scale = size.norm(dim=-1).clamp_min(1e-8)
    loss = 0.
    for pred in layers.unbind(1):
        with torch.no_grad():
            target = axis_target(pred, template, rotation, translation, size, symmetric)
        loss = loss + ((pred - target).norm(dim=-1).mean(1) / scale).mean()
    return loss
