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

from model.adapointr.AdaPoinTr import (
    PointTransformerEncoderEntry, PointTransformerDecoderEntry,
)
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


class ComposeCompletion(nn.Module):
    def __init__(self, cfg, feat_dim):
        super().__init__()
        cfg = EasyDict(deepcopy(dict(cfg)))
        self.num_query = cfg.get('num_query', 64)
        self.num_missing = cfg.get('num_missing', 64)
        self.num_visible = cfg.get('num_visible', 32)
        self.memory_points = cfg.get('memory_points', 256)
        self.factor = cfg.get('points_per_keypoint', 16)
        self.axis_points = cfg.axis_points_per_axis
        assert 3 * self.axis_points <= self.num_query
        ticks = torch.linspace(-cfg.axis_half_length, cfg.axis_half_length, self.axis_points)
        axes = (torch.eye(3)[:, None, :] * ticks[None, :, None]).flatten(0, 1)
        self.register_buffer('axis_template', axes)
        assert cfg.encoder_config.embed_dim == cfg.decoder_config.embed_dim == feat_dim
        self.keypoint_proj = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, feat_dim))
        self.detail_proj = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, feat_dim))
        self.position = nn.Sequential(nn.Linear(3, feat_dim), nn.GELU(), nn.Linear(feat_dim, feat_dim))
        self.encoder = PointTransformerEncoderEntry(cfg.encoder_config)
        self.decoder = PointTransformerDecoderEntry(cfg.decoder_config)
        self.global_proj = nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.GELU(), nn.LayerNorm(feat_dim))
        self.missing_head = nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.GELU(),
                                          nn.Linear(feat_dim, 3 * self.num_missing))
        self.axis_head = nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.GELU(), nn.Linear(feat_dim, 10))
        self.axis_embed = nn.Sequential(nn.Linear(3, feat_dim), nn.GELU(), nn.Linear(feat_dim, feat_dim))
        self.axis_refine_head = nn.Sequential(nn.Linear(3 * feat_dim, feat_dim), nn.GELU(), nn.Linear(feat_dim, 10))
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
        with torch.no_grad():
            self.axis_head[-1].bias[3:9].copy_(torch.tensor([1., 0., 0., 0., 1., 0.]))
            self.axis_head[-1].bias[9] = 0.54132485  # inverse softplus(1)

    def axis_coordinates(self, frame, origin, extent):
        # 3 center + 6 rotation + 1 scale: a straight, orthogonal right-handed frame.
        x = F.normalize(frame[:, 3:6], dim=-1, eps=1e-6)
        y = frame[:, 6:9] - (frame[:, 6:9] * x).sum(-1, keepdim=True) * x
        y = F.normalize(y, dim=-1, eps=1e-6)
        basis = torch.stack((x, y, torch.linalg.cross(x, y)), dim=1)
        center = origin + frame[:, None, :3] * extent[:, None, None]
        scale = F.softplus(frame[:, 9:10])[:, None] * extent[:, None, None]
        return (self.axis_template @ basis) * scale + center

    def forward(self, keypoints, features, points=None, point_features=None):
        xyz, memory = keypoints, self.keypoint_proj(features)
        if self.memory_points:
            indices = sample_indices(points, self.memory_points)
            xyz = torch.cat((xyz, gather_points(points, indices)), dim=1)
            memory = torch.cat((memory, self.detail_proj(gather_points(point_features, indices))), dim=1)
        memory = self.encoder(memory + self.position(xyz), xyz)
        global_feature = self.global_proj(memory.amax(dim=1))
        missing = self.missing_head(global_feature).view(-1, self.num_missing, 3)
        missing = missing + keypoints.mean(dim=1, keepdim=True)
        # Normalize axis prediction by observed extent; no GT scale is needed.
        extent = 2.5 * (xyz - xyz.mean(1, keepdim=True)).square().sum(-1).mean(1).sqrt().clamp_min(1e-3).detach()
        axis_frame = self.axis_head(global_feature)
        origin = keypoints.mean(1, keepdim=True)
        axis_coarse = self.axis_coordinates(axis_frame, origin, extent)
        # Candidates remain based on the detected keypoints, rather than moving
        # completion ahead of keypoint detection as in the original ComPose.
        visible = gather_points(keypoints, sample_indices(keypoints, self.num_visible))
        candidates = torch.cat((missing, visible), dim=1)
        scores = self.score_head(torch.cat((candidates, global_feature[:, None].expand(
            -1, candidates.size(1), -1)), dim=-1)).squeeze(-1)
        selected = scores.topk(self.num_query, dim=1).indices
        coarse = gather_points(candidates, selected)
        query_xyz = torch.cat((coarse, axis_coarse), dim=1)
        query = global_feature[:, None] + self.position(query_xyz)
        identity = torch.cat((torch.zeros_like(query[:, :self.num_query]),
                              self.axis_embed(self.axis_template)[None].expand(query.size(0), -1, -1)), dim=1)
        decoded = self.output_norm(self.decoder(q=query + identity, v=memory, q_pos=query_xyz, v_pos=xyz))
        decoded, axis_features = decoded[:, :self.num_query], decoded[:, self.num_query:]
        axis_global = axis_features.reshape(-1, 3, self.axis_points, decoded.size(-1)).mean(2).flatten(1)
        axis_frame = axis_frame + self.axis_refine_head(axis_global)
        axis_refined = self.axis_coordinates(axis_frame, origin, extent)
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
