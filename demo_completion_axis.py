#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualize 64 predicted object keypoints, 96 GT-transformed CAD points,
and predicted/GT axis proxy points in a plain Open3D window.

Run from the proxypose / CleanPose project root.
"""

import argparse
import os
import pickle
import random
import sys
from pathlib import Path

import gorilla
import numpy as np
import open3d as o3d
import torch

ROOT = Path(__file__).resolve().parent
for subdir in ('provider', 'model', 'model/pointnet2', 'utils'):
    sys.path.insert(0, str(ROOT / subdir))

from model.Net import Net
from provider.nocs_dataset import TestDataset

CATEGORIES = {'bottle': 0, 'bowl': 1, 'camera': 2, 'can': 3, 'laptop': 4, 'mug': 5}
PRED_COLORS = ((1.0, 0.1, 0.1), (0.1, 1.0, 0.1), (0.1, 0.35, 1.0))
GT_COLORS = ((1.0, 0.6, 0.6), (0.6, 1.0, 0.6), (0.6, 0.75, 1.0))
OBJECT_COLOR = (1.0, 0.85, 0.1)  # predicted 64 keypoints: yellow
CAD_COLOR = (0.65, 0.1, 0.75)    # GT CAD sampled 96 points: purple


def parse_args():
    parser = argparse.ArgumentParser(description='Display sparse keypoints, CAD-96 and axes.')
    parser.add_argument('--category', required=True, choices=CATEGORIES)
    parser.add_argument('--config', default='config/REAL/camera_real.yaml')
    parser.add_argument('--checkpoint', default='log/camera_real_completion_axis1020/ckpt/epoch_30.pt')
    parser.add_argument('--tensor-list', default=None)
    parser.add_argument('--model-bank', default=None,
                        help='Default: <test_dataset.dataset_dir>/obj_models/real_test.pkl')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--point-size', type=float, default=7.0)
    return parser.parse_args()


def bbox_iou(box, boxes):
    top_left = np.maximum(box[:2], boxes[:, :2])
    bottom_right = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.prod(np.maximum(bottom_right - top_left, 0), axis=1)
    box_area = np.prod(np.maximum(box[2:] - box[:2], 0))
    areas = np.prod(np.maximum(boxes[:, 2:] - boxes[:, :2], 0), axis=1)
    return intersection / np.maximum(box_area + areas - intersection, 1e-8)


def choose_sample(dataset, category, rng):
    class_id = CATEGORIES[category] + 1
    candidate_indices = []
    for index, path in enumerate(dataset.result_pkl_list):
        with open(path, 'rb') as f:
            if class_id in pickle.load(f)['pred_class_ids']:
                candidate_indices.append(index)

    rng.shuffle(candidate_indices)
    for index in candidate_indices:
        sample = dataset[index]
        matches = torch.where(sample['category_label'] == class_id - 1)[0].tolist()
        if matches:
            return sample, rng.choice(matches)
    raise RuntimeError(f'No valid {category} instance found.')


def match_gt(sample, instance):
    """Match a predicted object with a same-category GT using 2D bbox IoU."""
    class_id = int(sample['category_label'][instance]) + 1
    candidates = np.where(sample['gt_class_ids'].numpy() == class_id)[0]
    if not len(candidates):
        raise RuntimeError('No same-category ground-truth instance in this image.')
    overlaps = bbox_iou(
        sample['pred_bboxes'][instance].numpy(),
        sample['gt_bboxes'][candidates].numpy(),
    )
    gt_index = int(candidates[np.argmax(overlaps)])
    return gt_index, float(np.max(overlaps))


def gt_axis_points(rt, template, category_id):
    """Apply NOCS GT Sim(3) to axis template, with yaw canonicalization."""
    scale = float(np.cbrt(np.linalg.det(rt[:3, :3])))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f'Invalid GT similarity scale: {scale}')
    rotation = rt[:3, :3] / scale

    # Symmetry canonicalization only for bottle/bowl/can axes.
    if category_id in (0, 1, 3):
        theta_x = rotation[0, 0] + rotation[2, 2]
        theta_y = rotation[0, 2] - rotation[2, 0]
        radius = np.hypot(theta_x, theta_y)
        if radius > 1e-8:
            symmetry = np.array([
                [theta_x / radius, 0.0, -theta_y / radius],
                [0.0, 1.0, 0.0],
                [theta_y / radius, 0.0, theta_x / radius],
            ])
            rotation = rotation @ symmetry

    return (template * scale) @ rotation.T + rt[:3, 3]


def model_id_for_gt(source_pkl, dataset_dir, gt_index, expected_class_id):
    """Find the CAD model ID using the real-test image's _meta.txt."""
    with open(source_pkl, 'rb') as f:
        image_path = str(pickle.load(f)['image_path'])

    # Match TestDataset's image path convention (strip initial 'data/').
    relative_path = image_path[5:] if image_path.startswith('data/') else image_path
    base = Path(dataset_dir) / relative_path
    meta_path = Path(str(base) + '_meta.txt')
    if not meta_path.is_file():
        alternative = Path(image_path + '_meta.txt')
        if alternative.is_file():
            meta_path = alternative
        else:
            raise FileNotFoundError(f'Cannot locate GT metadata: {meta_path}')

    records = [line.split() for line in meta_path.read_text().splitlines() if line.strip()]
    if gt_index >= len(records):
        raise IndexError(f'GT index {gt_index} not found in {meta_path} ({len(records)} rows).')
    fields = records[gt_index]
    if len(fields) < 3:
        raise ValueError(f'Unexpected meta format in {meta_path}: {fields}')
    if int(fields[1]) != expected_class_id:
        raise ValueError(
            f'GT/meta index mismatch: GT class {expected_class_id}, meta class '
            f'{fields[1]} at row {gt_index} of {meta_path}. Check GT ordering.'
        )
    return fields[2], meta_path


def sample_cad_96(model_points):
    """Sample 96 uniformly spread CAD surface points using Open3D FPS."""
    xyz = np.asarray(model_points, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f'CAD needs shape (N, 3), got {xyz.shape}')
    if xyz.shape[0] < 96:
        raise ValueError(f'CAD has only {xyz.shape[0]} points, cannot sample 96 distinct points.')
    if not np.isfinite(xyz).all():
        raise ValueError('CAD contains NaN or infinity.')
    cad_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    return np.asarray(cad_cloud.farthest_point_down_sample(96).points)


def make_keypoints(points, color):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    cloud.paint_uniform_color(color)
    return cloud


def show_keypoints(object_keypoints, cad96_gt, pred_axes, gt_axes,
                   points_per_axis, category, point_size):
    geometries = [
        make_keypoints(object_keypoints, OBJECT_COLOR),
        make_keypoints(cad96_gt, CAD_COLOR),
    ]
    for axis in range(3):
        start, end = axis * points_per_axis, (axis + 1) * points_per_axis
        geometries.append(make_keypoints(pred_axes[start:end], PRED_COLORS[axis]))
        geometries.append(make_keypoints(gt_axes[start:end], GT_COLORS[axis]))

    vis = o3d.visualization.Visualizer()
    if not vis.create_window(window_name=f'{category} | Keypoints + CAD96 + axes',
                             width=1280, height=800):
        raise RuntimeError('Could not open Open3D window. Check DISPLAY / graphics environment.')
    try:
        for geometry in geometries:
            vis.add_geometry(geometry)
        options = vis.get_render_option()
        options.point_size = point_size
        options.background_color = np.array([1.0, 1.0, 1.0])
        vis.run()
    finally:
        vis.destroy_window()


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is not available.')

    rng = random.Random(args.seed)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    cfg = gorilla.Config.fromfile(args.config)
    if cfg.test_dataset.dataset_name != 'real':
        raise ValueError('This script uses real_test.pkl and REAL275 metadata; select real test dataset.')

    checkpoint = Path(args.checkpoint)
    tensor_path = Path(args.tensor_list) if args.tensor_list else checkpoint.parent / 'tensor_list.pkl'
    model_bank_path = Path(args.model_bank) if args.model_bank else (
        Path(cfg.test_dataset.dataset_dir) / 'obj_models' / 'real_test.pkl'
    )
    with open(tensor_path, 'rb') as f:
        tensor_list = pickle.load(f)

    dataset = TestDataset(
        cfg.test_dataset.img_size,
        cfg.test_dataset.sample_num,
        cfg.test_dataset.dataset_dir,
        cfg.setting,
        cfg.test_dataset.dataset_name,
    )
    sample, instance = choose_sample(dataset, args.category, rng)
    # Dataset may occasionally resample an image if depth is invalid:
    source_index = int(sample['index'].item())
    source_path = dataset.result_pkl_list[source_index]

    inputs = {key: sample[key].cuda() for key in ('rgb', 'pts', 'choose', 'category_label')}
    model = Net(cfg.pose_net, ULIP_model=None, tensor_list=tensor_list).cuda().eval()
    gorilla.solver.load_checkpoint(model=model, filename=str(checkpoint), strict=True)
    with torch.inference_mode():
        prediction = model(inputs)

    object_keypoints = prediction['pred_kpt_3d'][instance].detach().cpu().numpy()
    pred_axes = prediction['axis_points'][instance].detach().cpu().numpy()
    template = model.completion.axis_template.detach().cpu().numpy()
    points_per_axis = int(cfg.pose_net.Completion.axis_points_per_axis)
    num_queries = int(cfg.pose_net.Completion.num_query)
    if len(object_keypoints) != num_queries:
        raise RuntimeError(f'Expected {num_queries} predicted keypoints, got {len(object_keypoints)}.')
    if len(pred_axes) != 3 * points_per_axis or len(template) != 3 * points_per_axis:
        raise RuntimeError('Predicted axis/template count does not match configured axis points.')

    gt_index, match_iou = match_gt(sample, instance)
    rt = sample['gt_RTs'][gt_index].numpy()
    category_id = CATEGORIES[args.category]
    gt_axes = gt_axis_points(rt, template, category_id)

    model_id, meta_path = model_id_for_gt(
        source_path, cfg.test_dataset.dataset_dir, gt_index, category_id + 1
    )
    with open(model_bank_path, 'rb') as f:
        model_bank = pickle.load(f)
    if model_id not in model_bank:
        raise KeyError(f'CAD model {model_id!r} missing from {model_bank_path}')

    # Sample in canonical CAD coordinates, then transform with GT Sim(3).
    cad96 = sample_cad_96(model_bank[model_id])
    cad96_gt = cad96 @ rt[:3, :3].T + rt[:3, 3]

    print(f'Category: {args.category}; image index: {source_index}; detection: {instance}')
    print(f'GT instance index: {gt_index}; 2D bbox matching IoU: {match_iou:.3f}')
    print(f'CAD model: {model_id}; metadata: {meta_path}')
    print(f'Predicted keypoints: {len(object_keypoints)} (yellow)')
    print(f'GT-pose CAD points: {len(cad96_gt)} (purple)')
    print(f'Axis proxies: {len(pred_axes)} predicted + {len(gt_axes)} GT')
    if match_iou < 0.3:
        print('WARNING: Low 2D detection-to-GT matching IoU; GT CAD may be mismatched.')
    print('Prediction axes: RGB; GT axes: light RGB')

    show_keypoints(
        object_keypoints, cad96_gt, pred_axes, gt_axes,
        points_per_axis, args.category, args.point_size,
    )


if __name__ == '__main__':
    main()
