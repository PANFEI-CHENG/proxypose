"""Dependency adapters for the vendored PoinTr implementation (see LICENSE)."""
import logging

import torch
from torch import nn


def print_log(message, logger=None):
    logging.getLogger(logger).info(message)


def _native():
    from model.pointnet2 import pointnet2_utils
    return pointnet2_utils


class _PointNet2:
    @staticmethod
    @torch.no_grad()
    def furthest_point_sample(xyz, count):
        if xyz.is_cuda:
            return _native().furthest_point_sample(xyz.contiguous(), count)
        batch, points, _ = xyz.shape
        indices = torch.empty(batch, count, dtype=torch.long, device=xyz.device)
        distance = xyz.new_full((batch, points), float('inf'))
        farthest = torch.zeros(batch, dtype=torch.long, device=xyz.device)
        batch_idx = torch.arange(batch, device=xyz.device)
        for i in range(count):
            indices[:, i] = farthest
            center = xyz[batch_idx, farthest].unsqueeze(1)
            distance = torch.minimum(distance, (xyz - center).square().sum(-1))
            farthest = distance.argmax(-1)
        return indices

    @staticmethod
    def gather_operation(features, indices):
        if features.is_cuda:
            return _native().gather_operation(features.contiguous(), indices.int())
        return features.gather(2, indices.long().unsqueeze(1).expand(-1, features.size(1), -1))

    @staticmethod
    @torch.no_grad()
    def three_nn(unknown, known):
        if unknown.is_cuda:
            return _native().three_nn(unknown.contiguous(), known.contiguous())
        return torch.cdist(unknown, known).topk(3, dim=-1, largest=False)

    @staticmethod
    def three_interpolate(features, indices, weight):
        if features.is_cuda:
            return _native().three_interpolate(features.contiguous(), indices.int(), weight.contiguous())
        batch, channels, _ = features.shape
        neighbors = features.gather(2, indices.long().flatten(1).unsqueeze(1).expand(-1, channels, -1))
        return (neighbors.reshape(batch, channels, -1, 3) * weight.detach().unsqueeze(1)).sum(-1)


pointnet2_utils = _PointNet2()


# These two helpers are copied from upstream utils/misc.py.
def fps(data, number):
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
    return fps_data


def jitter_points(pc, std=0.01, clip=0.05):
    bsize = pc.size()[0]
    for i in range(bsize):
        jittered_data = pc.new(pc.size(1), 3).normal_(mean=0.0, std=std).clamp_(-clip, clip)
        pc[i, :, 0:3] += jittered_data
    return pc


class ChamferDistanceL1(nn.Module):
    """Same symmetric mean Euclidean distance as upstream's CUDA ChamferL1."""
    def forward(self, xyz1, xyz2):
        distance = torch.cdist(xyz1, xyz2, compute_mode='donot_use_mm_for_euclid_dist')
        return (distance.min(2).values.mean() + distance.min(1).values.mean()) / 2
