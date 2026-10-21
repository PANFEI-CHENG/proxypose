"""Checks for axis-proxy supervision, the pose-loss yaw alignment, the proxy pose readout and the LR schedule.

Run: .pixi/envs/default/bin/python tests/test_axis_proxy.py
"""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'model'), str(ROOT / 'utils')]

import gorilla
import torch
from easydict import EasyDict
from torch import nn

from model.compose_completion import align_symmetric_rotation, axis_target, canonical_to_camera
from utils.solver import build_optimizer, build_lr_scheduler

CFG = gorilla.Config.fromfile(str(ROOT / 'config/REAL/camera_real.yaml'))


def random_pose(b):
    rotation = torch.linalg.qr(torch.randn(b, 3, 3)).Q
    rotation = rotation * torch.linalg.det(rotation)[:, None, None]
    return rotation, 0.1 * torch.randn(b, 3), 0.05 + 0.2 * torch.rand(b, 3)


def yaw(angle):
    c, s, o, i = angle.cos(), angle.sin(), torch.zeros_like(angle), torch.ones_like(angle)
    return torch.stack((c, o, s, o, i, o, -s, o, c), 1).view(-1, 3, 3)


def template(b):
    half, count = CFG.pose_net.Completion.axis_half_length, CFG.pose_net.Completion.axis_points_per_axis
    ticks = torch.linspace(-half, half, count)
    return (torch.eye(3)[:, None, :] * ticks[None, :, None]).flatten(0, 1)[None].expand(b, -1, -1)


def test_symmetric_target_matches_any_yaw():
    rotation, translation, size = random_pose(8)
    pred = canonical_to_camera(template(8), rotation @ yaw(torch.rand(8) * 2 * math.pi), translation, size)
    target = axis_target(pred, template(8), rotation, translation, size, torch.ones(8, dtype=torch.bool))
    torch.testing.assert_close(target, pred, atol=1e-5, rtol=0)


def test_asymmetric_target_is_gt():
    rotation, translation, size = random_pose(8)
    pred = torch.randn(8, template(8).size(1), 3)
    target = axis_target(pred, template(8), rotation, translation, size, torch.zeros(8, dtype=torch.bool))
    torch.testing.assert_close(target, canonical_to_camera(template(8), rotation, translation, size))


def test_aligned_target_is_never_farther_than_gt():
    rotation, translation, size = random_pose(64)
    gt = canonical_to_camera(template(64), rotation, translation, size)
    pred = gt + 0.03 * torch.randn_like(gt)
    target = axis_target(pred, template(64), rotation, translation, size, torch.ones(64, dtype=torch.bool))
    assert ((target - pred).square().sum((1, 2)) <= (gt - pred).square().sum((1, 2)) + 1e-6).all()


def test_warmup_cosine_schedule():
    model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
    optimizer = build_optimizer(model, CFG.optimizer)
    assert [g['weight_decay'] for g in optimizer.param_groups] == [CFG.optimizer.weight_decay, 0.]
    assert len(optimizer.param_groups[0]['params']) == 1  # only the Linear weight is decayed
    warmup, total = 10, 100
    scheduler = build_lr_scheduler(optimizer, EasyDict(dict(CFG.lr_scheduler), warmup_iters=warmup), total)
    lrs = []
    for _ in range(total + 5):
        lrs.append(optimizer.param_groups[0]['lr'])
        optimizer.step()
        scheduler.step()
    peak = CFG.optimizer.lr
    assert math.isclose(lrs[0], peak * CFG.lr_scheduler.warmup_factor)
    assert math.isclose(lrs[warmup], peak) and max(lrs) == lrs[warmup]
    assert all(a >= b for a, b in zip(lrs[warmup:], lrs[warmup + 1:]))
    assert math.isclose(lrs[total], CFG.lr_scheduler.min_lr) and math.isclose(lrs[-1], CFG.lr_scheduler.min_lr)



def test_pose_loss_rotation_alignment():
    rotation = random_pose(64)[0]
    symmetric = torch.arange(64) % 2 == 0
    # A prediction that differs from GT only by yaw is matched exactly for symmetric objects.
    pred = rotation @ yaw((torch.rand(64) * 2 - 1) * math.pi)
    aligned = align_symmetric_rotation(pred, rotation, symmetric)
    torch.testing.assert_close(aligned[symmetric], pred[symmetric], atol=1e-5, rtol=0)
    assert torch.equal(aligned[~symmetric], rotation[~symmetric])
    # For any prediction the y-axis is unchanged and the target is never farther than GT.
    pred = random_pose(64)[0]
    aligned = align_symmetric_rotation(pred, rotation, symmetric)
    torch.testing.assert_close(aligned[:, :, 1], rotation[:, :, 1], atol=1e-5, rtol=0)
    assert ((pred - aligned).norm(dim=(1, 2)) <= (pred - rotation).norm(dim=(1, 2)) + 1e-5).all()

def test_readout_recovers_pose_from_ideal_proxies():
    if not torch.cuda.is_available():  # rotation_utils.normalize_vector requires CUDA
        return
    from model.Net_modules import PoseEstimator
    rotation, translation, size = random_pose(8)
    proxies = canonical_to_camera(template(8), rotation, translation, size)
    estimator = PoseEstimator(CFG.pose_net.Completion.axis_points_per_axis).cuda().eval()
    for head in (estimator.rotation_estimator, estimator.translation_estimator):
        nn.init.zeros_(head[-1].weight)
    r, t = estimator(proxies.cuda(), template(8).cuda(), torch.randn(8, proxies.size(1), 256).cuda())
    torch.testing.assert_close(r.cpu(), rotation, atol=1e-5, rtol=0)
    torch.testing.assert_close(t.cpu(), translation, atol=1e-5, rtol=0)


if __name__ == '__main__':
    torch.manual_seed(0)
    for name, test in list(globals().items()):
        if name.startswith('test_'):
            test()
            print('passed', name)
