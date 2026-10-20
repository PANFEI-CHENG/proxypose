"""Real CAMERA/REAL smoke training using the production Solver and optimizer."""
import argparse
import copy
import json
import logging
import os
import pickle
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'provider'), str(ROOT / 'model'),
               str(ROOT / 'model/pointnet2'), str(ROOT / 'utils')]
os.chdir(ROOT)

import gorilla
import numpy as np
import torch
from load_ulip_model import load_ULIP_Pointbert
from model.Net import Net, Loss
from provider.create_dataloaders import create_dataloaders
from utils.solver import Solver


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def loaders_for(cfg, batch, steps):
    data_cfg = copy.deepcopy(cfg.train_dataset)
    data_cfg.syn_bs, data_cfg.real_bs = batch - max(1, batch // 4), max(1, batch // 4)
    data_cfg.syn_num_workers = data_cfg.real_num_workers = 2
    data_cfg.num_mini_batch_per_epoch = steps
    loaders = create_dataloaders(data_cfg)
    for loader in loaders.values():
        loader.dataset.reset()
    return loaders


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/REAL/camera_real.yaml')
    parser.add_argument('--output', default='log/axis_generation_smoke')
    parser.add_argument('--fixed-steps', type=int, default=120)
    parser.add_argument('--stream-steps', type=int, default=120)
    parser.add_argument('--peak-steps', type=int, default=20)
    parser.add_argument('--full-steps', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(6)
    seed(42)
    cfg = gorilla.Config.fromfile(args.config)
    config_snapshot = copy.deepcopy(dict(cfg))
    cfg.log_dir = args.output
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    logger = logging.getLogger('smoke')
    logger.setLevel(logging.INFO)
    with open(cfg.train_dataset.init_tensor_list_dir, 'rb') as handle:
        bank = pickle.load(handle)
    model = Net(cfg.pose_net, load_ULIP_Pointbert(), bank).cuda().train()
    loss_fn = Loss(cfg.loss).cuda()
    loaders = loaders_for(cfg, args.batch_size, max(args.stream_steps, 1))
    fixed = tuple(next(iter(loaders[key])) for key in ('syn', 'real'))
    solver = Solver(model, loss_fn, loaders, logger, cfg)
    optimizer_settings = config_snapshot['optimizer']
    scheduler_settings = config_snapshot['lr_scheduler']
    parameters = [p for p in model.parameters() if p.requires_grad]
    history, gradient_checks = [], {}

    def evaluate():
        model.eval()
        seed(123)
        data = {k: torch.cat([part[k] for part in fixed]).cuda() for k in fixed[0]}
        with torch.no_grad():
            prediction = model(data)
            scale = data['size_label'].norm(dim=1)[:, None, None]
            from model.compose_completion import canonical_to_camera, chamfer_squared
            target = canonical_to_camera(data['model'], data['completion_rotation_label'],
                                         data['translation_label'], data['size_label'])
            nocs = (prediction['pred_kpt_3d'] - data['translation_label'][:, None]) / scale @ data['rotation_label']
            axis_target = canonical_to_camera(prediction['axis_nocs'], data['rotation_label'],
                                             data['translation_label'], data['size_label'])
            weights = torch.ones(data['pts'].size(0), 3, device='cuda')
            symmetric = (data['category_label'].reshape(-1, 1) == loss_fn.symmetric_categories).any(1)
            weights[symmetric] = weights.new_tensor([0., 1., 0.])
            axis_error = (prediction['axis_points'] - axis_target).norm(dim=-1).reshape(-1, 3, model.completion.axis_points).mean(2)
            result = {
                'dense_cd_squared': chamfer_squared(prediction['recon_model'], target).item(),
                'nocs_l1': (prediction['kpt_nocs'] - nocs).abs().mean().item(),
                'translation_l2': (prediction['pred_translation'] - data['translation_label']).norm(dim=1).mean().item(),
                'size_l2': (prediction['pred_size'] - data['size_label']).norm(dim=1).mean().item(),
                'rotation_frobenius': (prediction['pred_rotation'] - data['rotation_label']).norm(dim=(1, 2)).mean().item(),
                'axis_l2': ((axis_error * weights).sum(1) / weights.sum(1)).mean().item(),
                'recon_delta_norm': prediction['recon_delta'].norm(dim=-1).mean().item(),
            }
            torch.testing.assert_close(prediction['recon_model'], prediction['pred_kpt_3d'].repeat_interleave(
                model.completion.factor, 1) + prediction['recon_delta'], atol=1e-6, rtol=1e-5)
            assert all(torch.isfinite(t).all() for t in prediction.values())
        model.train()
        return result

    before = evaluate()
    started = time.time()
    with (output / 'steps.jsonl').open('w') as log:
        def step(batch, phase, index):
            optimizer = solver.optimizer
            optimizer.zero_grad(set_to_none=True)
            # Solver adds predictions to its input dictionaries; use fresh ones.
            loss, info, _ = solver.step_camera_real(tuple(dict(x) for x in batch), 'train')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, float('inf'), error_if_nonfinite=True)
            if not history:
                for name in ('encoder', 'decoder', 'missing_head', 'score_head', 'refine_head',
                             'fold_head', 'keypoint_proj', 'axis_head', 'axis_embed', 'axis_refine_head'):
                    module = getattr(model.completion, name)
                    gradient_checks[name] = sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None)
                assert all(value > 0 for value in gradient_checks.values()), gradient_checks
            assert torch.isfinite(loss)
            optimizer.step()
            solver.lr_scheduler.step()
            row = dict(phase=phase, step=index, grad_norm=norm.item(), **info)
            history.append(row)
            log.write(json.dumps(row) + '\n')
            log.flush()
            if index % 20 == 0:
                print(f"{phase} {index}: loss={info['loss_all']:.4f} recon={info['loss_recon']:.4f} "
                      f"pose={info['loss_pose']:.4f} axis={info['loss_axis']:.4f} grad={norm:.3f}", flush=True)

        for index in range(args.fixed_steps):
            step(fixed, 'fixed', index)
        after_fixed = evaluate()
        for index, batch in enumerate(zip(loaders['syn'], loaders['real'])):
            if index >= args.stream_steps:
                break
            step(batch, 'stream', index)
        # Probe the existing schedule's maximum LR without changing its settings.
        if args.peak_steps:
            solver.lr_scheduler.step(scheduler_settings['step_size_up'])
        for index in range(args.peak_steps):
            step(fixed, 'peak_lr', index)
        if args.full_steps:
            full_batch = cfg.train_dataset.syn_bs + cfg.train_dataset.real_bs
            full_loaders = loaders_for(cfg, full_batch, args.full_steps)
            for index, batch in enumerate(zip(full_loaders['syn'], full_loaders['real'])):
                step(batch, 'configured_batch', index)

    after = evaluate()
    report = dict(config=str(args.config), args=vars(args), optimizer=optimizer_settings,
                  architecture=dict(keypoints=model.completion.num_query,
                                    axis_points_per_axis=model.completion.axis_points,
                                    pose_points=model.completion.num_query + 3 * model.completion.axis_points,
                                    dense_points=model.completion.num_query * model.completion.factor),
                  scheduler=scheduler_settings, steps=len(history), before=before,
                  after_fixed=after_fixed, after=after, gradients=gradient_checks,
                  max_grad_norm=max(row['grad_norm'] for row in history),
                  peak_memory_gib=torch.cuda.max_memory_allocated() / 2**30,
                  seconds=time.time() - started,
                  learning_check=all(after_fixed[key] < before[key] and after[key] < before[key] for key in (
                      'dense_cd_squared', 'axis_l2', 'translation_l2', 'rotation_frobenius', 'size_l2')))
    (output / 'report.json').write_text(json.dumps(report, indent=2))
    state = {k: v.cpu() for k, v in model.state_dict().items() if not k.startswith('ULIP_model.')}
    torch.save(state, output / 'smoke.pt')
    with (output / 'tensor_list.pkl').open('wb') as handle:
        pickle.dump([[t.cpu() for t in group] for group in model.tensor_list], handle)
    print(json.dumps(report, indent=2), flush=True)
    solver.tb_writer.writer.close()
    assert report['learning_check'], 'Fixed-batch completion/axes/pose did not improve; inspect report.json'


if __name__ == '__main__':
    main()
