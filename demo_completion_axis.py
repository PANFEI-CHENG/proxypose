import argparse
import json
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
sys.path.extend(str(ROOT / path) for path in ('provider', 'model', 'model/pointnet2', 'utils'))

from model.Net import Net
from provider.nocs_dataset import TestDataset


CATEGORIES = {'bottle': 0, 'bowl': 1, 'camera': 2, 'can': 3, 'laptop': 4, 'mug': 5}
AXIS_COLORS = ((1., 0.1, 0.1), (0.1, 1., 0.1), (0.1, 0.3, 1.))


def parse_args():
    parser = argparse.ArgumentParser(description='Randomly visualize completion and axis proxy points.')
    parser.add_argument('--category', required=True, choices=CATEGORIES)
    parser.add_argument('--config', default='config/REAL/camera_real.yaml')
    parser.add_argument('--checkpoint', default='log/camera_real_completion_axis1020/ckpt/epoch_30.pt')
    parser.add_argument('--tensor-list', default=None)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--output-dir', default='log/completion_axis_demo')
    parser.add_argument('--no-window', action='store_true')
    return parser.parse_args()


def point_cloud(points, color):
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud.paint_uniform_color(color)
    return cloud


def axis_geometry(points, points_per_axis):
    geometries = []
    for axis, color in enumerate(AXIS_COLORS):
        xyz = points[axis * points_per_axis:(axis + 1) * points_per_axis]
        cloud = point_cloud(xyz, color)
        lines = np.column_stack((np.arange(points_per_axis - 1), np.arange(1, points_per_axis)))
        line_set = o3d.geometry.LineSet(
            o3d.utility.Vector3dVector(xyz), o3d.utility.Vector2iVector(lines))
        line_set.colors = o3d.utility.Vector3dVector(np.tile(color, (len(lines), 1)))
        geometries.extend((line_set, cloud))
    return geometries


def choose_sample(dataset, category, rng):
    class_id = CATEGORIES[category] + 1
    candidates = []
    for index, path in enumerate(dataset.result_pkl_list):
        with open(path, 'rb') as handle:
            if class_id in pickle.load(handle)['pred_class_ids']:
                candidates.append(index)
    rng.shuffle(candidates)
    for index in candidates:
        sample = dataset[index]
        matches = torch.where(sample['category_label'] == class_id - 1)[0].tolist()
        if matches:
            return sample, rng.choice(matches)
    raise RuntimeError(f'No valid {category} instance found in the test set.')


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    cfg = gorilla.Config.fromfile(args.config)
    checkpoint = Path(args.checkpoint)
    tensor_list_path = Path(args.tensor_list) if args.tensor_list else checkpoint.parent / 'tensor_list.pkl'
    with open(tensor_list_path, 'rb') as handle:
        tensor_list = pickle.load(handle)

    dataset = TestDataset(cfg.test_dataset.img_size, cfg.test_dataset.sample_num,
                          cfg.test_dataset.dataset_dir, cfg.setting, cfg.test_dataset.dataset_name)
    sample, instance = choose_sample(dataset, args.category, rng)
    inputs = {key: sample[key].cuda() for key in ('rgb', 'pts', 'choose', 'category_label')}

    model = Net(cfg.pose_net, ULIP_model=None, tensor_list=tensor_list).cuda().eval()
    gorilla.solver.load_checkpoint(model=model, filename=str(checkpoint), strict=True)
    with torch.inference_mode():
        prediction = model(inputs)

    visible = sample['pts'][instance].numpy()
    completion = prediction['recon_model'][instance].cpu().numpy()
    axes = prediction['axis_points'][instance].cpu().numpy()
    points_per_axis = cfg.pose_net.Completion.axis_points_per_axis
    source_index = int(sample['index'].item())
    source = dataset.result_pkl_list[source_index]

    output = Path(args.output_dir) / f'{args.category}_{Path(source).stem}'
    output.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output / 'visible.ply'), point_cloud(visible, (0.55, 0.55, 0.55)))
    o3d.io.write_point_cloud(str(output / 'completion.ply'), point_cloud(completion, (0.05, 0.8, 0.9)))
    for axis, color in enumerate(AXIS_COLORS):
        xyz = axes[axis * points_per_axis:(axis + 1) * points_per_axis]
        o3d.io.write_point_cloud(str(output / f'axis_{"xyz"[axis]}.ply'), point_cloud(xyz, color))
    metadata = {'category': args.category, 'source': source, 'instance': instance,
                'checkpoint': str(checkpoint), 'completion_points': len(completion),
                'axis_points': len(axes)}
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    print(json.dumps(metadata, indent=2))
    print(f'Saved to {output}')
    if not args.no_window:
        geometries = [point_cloud(completion, (0.05, 0.8, 0.9)), *axis_geometry(axes, points_per_axis)]
        o3d.visualization.draw_geometries(
            geometries, window_name=f'CleanPose completion + axes | {args.category}',
            width=1280, height=800, point_show_normal=False)


if __name__ == '__main__':
    main()


# .pixi/envs/default/bin/python demo_completion_axis.py \
#   --category bottle \
#   --checkpoint log/camera_real_completion_axis1020/ckpt/epoch_30.pt