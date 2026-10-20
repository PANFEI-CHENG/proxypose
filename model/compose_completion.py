"""ComPose-style progressive completion after CleanPose's FD encoder.

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


def geometric_relation_loss(keypoints, nocs, size):
    # A detached geometric target prevents shrinking the predicted geometry to
    # reduce correspondence losses. Completion and pose still train coordinates.
    reference = keypoints.detach() / size.norm(dim=-1).clamp_min(1e-8)[:, None, None]
    distances = lambda x: torch.cdist(x.float(), x.float(), compute_mode='donot_use_mm_for_euclid_dist')
    return F.mse_loss(distances(nocs), distances(reference))


class CompletionDecoderLayer(nn.Module):
    """Global interaction for all queries; surface geometry only for surface queries."""
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
        for block in self.blocks:
            q = block(q, memory, surface_xyz, memory_xyz, self_idx, cross_idx)
        return q


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
        self.encoder = PointTransformerEncoderEntry(cfg.encoder_config)
        self.decoder = CompletionDecoder(cfg.decoder_config)
        self.global_proj = nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.GELU(), nn.LayerNorm(feat_dim))
        self.missing_head = nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.GELU(),
                                          nn.Linear(feat_dim, 3 * self.num_missing))
        self.axis_head = nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.GELU(),
                                       nn.Linear(feat_dim, 3 * axes.size(0)))
        self.axis_embed = nn.Sequential(nn.Linear(3, feat_dim), nn.GELU(), nn.Linear(feat_dim, feat_dim))
        self.axis_refine_head = nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.GELU(), nn.Linear(feat_dim, 3))
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
        for head in (self.missing_head, self.refine_head, self.fold_head,
                     self.axis_head, self.axis_refine_head):
            nn.init.normal_(head[-1].weight, std=0.001)
            nn.init.zeros_(head[-1].bias)

    def forward(self, keypoints, features):
        # Geometry guides feature learning without moving the detector to ease attention.
        xyz, memory = keypoints.detach(), self.keypoint_proj(features)
        memory = self.encoder(memory + self.position(xyz), xyz)
        global_feature = self.global_proj(memory.amax(dim=1))
        missing = self.missing_head(global_feature).view(-1, self.num_missing, 3)
        origin = xyz.mean(1, keepdim=True)
        missing = missing + origin
        extent = 2.5 * (xyz - origin).square().sum(-1).mean(1).sqrt().clamp_min(0.02)[:, None, None]
        # Independent point displacements, not an R/t/scale transformation of the template.
        axis_offsets = self.axis_head(global_feature).view(-1, 3 * self.axis_points, 3)
        axis_coarse = origin + extent * (self.axis_template + axis_offsets)
        # Candidates remain based on the detected keypoints, rather than moving
        # completion ahead of keypoint detection as in the original ComPose.
        visible = gather_points(keypoints, sample_indices(keypoints, self.num_visible))
        candidates = torch.cat((missing, visible), dim=1)
        scores = self.score_head(torch.cat((candidates.detach(), global_feature[:, None].expand(
            -1, candidates.size(1), -1)), dim=-1)).squeeze(-1)
        selected = scores.topk(self.num_query, dim=1).indices
        coarse = gather_points(candidates, selected)
        query_xyz = torch.cat((coarse, axis_coarse), dim=1)
        query = global_feature[:, None] + self.position(query_xyz.detach())
        identity = torch.cat((torch.zeros_like(query[:, :self.num_query]),
                              self.axis_embed(self.axis_template)[None].expand(query.size(0), -1, -1)), dim=1)
        decoded = self.output_norm(self.decoder(query + identity, memory, coarse.detach(), xyz))
        decoded, axis_features = decoded[:, :self.num_query], decoded[:, self.num_query:]
        axis_refined = axis_coarse + extent * self.axis_refine_head(axis_features)
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
            'axis_coarse': axis_coarse,
            'axis_points': axis_refined,
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


def axis_proxy_loss(pred, target, scale, axis_weights):
    error = F.smooth_l1_loss(pred / scale, target / scale, beta=0.05, reduction='none')
    per_axis = error.reshape(pred.size(0), 3, -1, 3).mean(dim=(2, 3))
    return ((per_axis * axis_weights).sum(1) / axis_weights.sum(1)).mean()


def axis_scale_loss(points, scale, half_length):
    lines = points.reshape(points.size(0), 3, -1, 3)
    lengths = (lines[:, :, -1] - lines[:, :, 0]).norm(dim=-1)
    ratio = lengths / (2 * half_length) / scale.flatten()[:, None]
    return F.smooth_l1_loss(ratio, torch.ones_like(ratio), beta=0.1)


def axis_structure_loss(points, scale):
    """Soft line/spacing/center/orthogonality priors; never project generated points."""
    lines = points.reshape(points.size(0), 3, -1, 3)
    lines = (lines - lines.mean((1, 2), keepdim=True)) / scale[:, None]
    ticks = torch.linspace(-1, 1, lines.size(2), device=points.device, dtype=points.dtype)[None, None, :, None]
    centers = lines.mean(2, keepdim=True)
    vectors = (lines * ticks).sum(2) / ticks.square().sum()
    fitted = centers + ticks * vectors[:, :, None]
    straight = F.smooth_l1_loss(lines, fitted, beta=0.05)
    centered = F.smooth_l1_loss(centers, torch.zeros_like(centers), beta=0.05)
    unit = F.normalize(vectors, dim=-1, eps=1e-5)
    gram = unit @ unit.transpose(1, 2)
    orthogonal = (gram - torch.eye(3, device=points.device)).square().mean()
    handed = F.smooth_l1_loss(torch.linalg.cross(unit[:, 0], unit[:, 1]), unit[:, 2], beta=0.1)
    return straight + centered + orthogonal + handed
