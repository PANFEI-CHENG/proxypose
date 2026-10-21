import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from model.losses import PoseDis
from model.modules import ModifiedResnet, PointNet2MSG
from model.Net_modules import (InstanceAdaptiveKeypointDetector, GeometricAwareFeatureAggregator,
                               FrontDoorEncoder, PoseEstimator, SizeEstimator)
from model.compose_completion import (ComposeCompletion, ComposeCompletionLoss, canonical_to_camera,
                                     axis_proxy_loss, align_symmetric_rotation)
from utils.data_utils import generate_augmentation


class Net(nn.Module):
    def __init__(self, cfg, ULIP_model, tensor_list):
        super().__init__()
        self.cfg, self.cat_num = cfg, cfg.cat_num
        if cfg.rgb_backbone == 'dino':
            self.rgb_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            self.rgb_extractor.requires_grad_(False)
            self.feature_mlp = nn.Conv1d(384, 128, 1)
        else:
            self.rgb_extractor = ModifiedResnet()
        self.pts_extractor = PointNet2MSG(radii_list=[[0.01, 0.02], [0.02, 0.04], [0.04, 0.08], [0.08, 0.16]])
        self.IAKD = InstanceAdaptiveKeypointDetector(cfg.IAKD)
        self.GAFA = GeometricAwareFeatureAggregator(cfg.GAFA)
        self.FD_Encoder = FrontDoorEncoder(cfg.FDEncoder)
        self.completion = ComposeCompletion(cfg.Completion, cfg.FDEncoder.feat_dim)
        self.estimator = PoseEstimator(cfg.Completion.axis_points_per_axis)
        self.size_estimator = SizeEstimator(cfg.FDEncoder.feat_dim, cfg.cat_num)
        self.ULIP_model, self.tensor_list = ULIP_model, tensor_list
        self.projector_pts = nn.Sequential(nn.Linear(128, 128, bias=False), nn.GELU(), nn.Linear(128, 128, bias=False))
        nn.init.kaiming_normal_(self.projector_pts[0].weight)
        nn.init.zeros_(self.projector_pts[2].weight)
        self.pointnet2ulip = nn.Parameter(torch.empty(128, cfg.distillation_feat_dim))
        nn.init.normal_(self.pointnet2ulip, std=cfg.distillation_feat_dim ** -0.5)

    def train(self, mode=True):
        super().train(mode)
        if self.cfg.rgb_backbone == 'dino':
            self.rgb_extractor.eval()
        if self.ULIP_model is not None:
            self.ULIP_model.eval()
        return self

    def forward(self, inputs):
        rgb, pts, choose = inputs['rgb'], inputs['pts'], inputs['choose']
        cls = inputs['category_label'].reshape(-1)
        b = pts.size(0)
        with torch.no_grad():
            selected = [torch.stack([bank[i].to(pts.device) for i in np.random.choice(
                len(bank), self.cfg.select_num, replace=False)]) for bank in self.tensor_list]
            select_global_feats = torch.stack(selected).squeeze(2)[cls]

        center = pts.mean(1, keepdim=True)
        pts = pts - center
        if self.cfg.rgb_backbone == 'dino':
            rgb_local = self.rgb_extractor.forward_features(rgb)['x_prenorm'][:, 1:]
            side, channels = int(rgb_local.size(1) ** 0.5), rgb_local.size(2)
            rgb_local = rgb_local.reshape(b, side, side, channels).permute(0, 3, 1, 2)
            rgb_local = F.interpolate(rgb_local, size=(side * 14, side * 14), mode='bilinear', align_corners=False)
            rgb_local = self.feature_mlp(rgb_local.flatten(2))
        else:
            rgb_local = self.rgb_extractor(rgb).flatten(2)
        rgb_local = rgb_local.gather(2, choose[:, None].expand(-1, rgb_local.size(1), -1)).contiguous()

        delta_r = torch.eye(3, device=pts.device)[None].expand(b, -1, -1)
        delta_t, delta_s = pts.new_zeros(b, 1, 3), pts.new_ones(b, 1)
        if self.training:
            delta_r, delta_t, delta_s = generate_augmentation(b)
            pts = (pts - delta_t) / delta_s.unsqueeze(2) @ delta_r
        pts_local = self.pts_extractor(pts)
        _, heat_map = self.IAKD(rgb_local, pts_local)
        observed_kpt = heat_map @ pts
        point_features = torch.cat((pts_local, rgb_local), dim=1).transpose(1, 2)
        kpt_feature = self.GAFA(heat_map @ point_features, observed_kpt.detach(), point_features, pts)
        kpt_feature = self.FD_Encoder(kpt_feature, select_global_feats)
        completion, decoded = self.completion(observed_kpt.detach(), kpt_feature, pts, point_features)
        axis_nocs = self.completion.axis_template[None].expand(b, -1, -1)
        # R and t are solved from the proxies and their object-frame coordinates alone. The
        # proxies are trained only by their own supervision; the solver reads them as given.
        r, t = self.estimator(completion['axis_points'].detach(), axis_nocs, completion['axis_features'])
        s = self.size_estimator(completion['completion_keypoints'].detach(), decoded, cls)

        def restore(points):
            return (points @ delta_r.transpose(1, 2)) * delta_s.unsqueeze(2) + delta_t + center

        endpoints = {key: restore(completion[key]) for key in (
            'completion_missing', 'completion_candidates', 'completion_coarse', 'axis_points')}
        axis_layers = completion['axis_layers']
        endpoints.update(
            axis_layers=restore(axis_layers.flatten(1, 2)).view_as(axis_layers),
            observed_kpt_3d=restore(observed_kpt), pred_kpt_3d=restore(completion['completion_keypoints']),
            recon_model=restore(completion['completion_dense']),
            recon_delta=(completion['recon_delta'] @ delta_r.transpose(1, 2)) * delta_s.unsqueeze(2),
            completion_scores=completion['completion_scores'], pred_heat_map=heat_map, axis_nocs=axis_nocs,
            pred_translation=delta_t.squeeze(1) + delta_s * (delta_r @ t.unsqueeze(2)).squeeze(2) + center.squeeze(1),
            pred_rotation=delta_r @ r, pred_size=s * delta_s,
        )
        if self.training:
            with torch.no_grad():
                bank_features = kpt_feature.detach().mean(1, keepdim=True)
                for i, category in enumerate(cls.tolist()):
                    self.tensor_list[category].pop(0)
                    self.tensor_list[category].append(bank_features[i])
                _, teacher = self.ULIP_model.encode_pc(pts)
            student = F.normalize(pts_local.mean(2), dim=-1)
            student = (student + 0.1 * self.projector_pts(student)) @ self.pointnet2ulip
            endpoints.update(distillation_pointnet=student, distillation_ulip=teacher, tensor_list=self.tensor_list)
        return endpoints


class Loss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.completion_loss = ComposeCompletionLoss(cfg.score_temperature)
        self.register_buffer('symmetric_categories', torch.tensor(cfg.symmetric_categories))

    def forward(self, endpoints):
        e, cfg = endpoints, self.cfg
        rotation, translation, size = e['rotation_label'], e['translation_label'], e['size_label']
        full_model = canonical_to_camera(e['model'], e['completion_rotation_label'], translation, size)
        completion = self.completion_loss(e, full_model)
        score = completion.pop('loss_score')
        symmetric = (e['category_label'].reshape(-1, 1) == self.symmetric_categories).any(1)
        axis_loss = axis_proxy_loss(e['axis_layers'], e['axis_nocs'], rotation, translation, size, symmetric)
        pose_rotation = align_symmetric_rotation(e['pred_rotation'], rotation, symmetric)
        losses = {
            'loss_pose': cfg.pose * PoseDis(e['pred_rotation'], e['pred_translation'], e['pred_size'], pose_rotation, translation, size),
            'loss_cd': cfg.cd * torch.cdist(e['pts'], e['observed_kpt_3d']).amin(1).mean(),
            'loss_diversity': cfg.diversity * self.diversity_loss(e['observed_kpt_3d']),
            'loss_recon': cfg.recon * sum(completion.values()),
            'loss_delta': cfg.delta * e['recon_delta'].norm(dim=-1).mean(),
            'loss_distillation': cfg.distillation * (e['distillation_pointnet'] - e['distillation_ulip']).norm(dim=-1).mean(),
            'loss_score': cfg.score * score,
            'loss_axis': cfg.axis * axis_loss,
        }
        losses['loss_all'] = sum(losses.values())
        losses.update({key: cfg.recon * value for key, value in completion.items()})
        return losses

    def diversity_loss(self, points):
        count = points.size(1)
        distance = torch.linalg.vector_norm(points[:, :, None] - points[:, None], dim=-1)
        distance = distance + torch.eye(count, device=points.device)[None]
        penalty = 1 - distance.clamp(max=self.cfg.th) / self.cfg.th
        return (penalty.sum(dim=(1, 2)) / (count * (count - 1))).mean()
