"""Diagnose real-data convergence with fixed and changing batches, and disjoint image probes."""
import argparse
import copy
import json
import logging
import pickle
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.smoke_completion import (seed, Net, Loss, Solver, create_dataloaders,
                                    load_ULIP_Pointbert, gorilla, np, torch)
from model.compose_completion import canonical_to_camera, chamfer_squared


def plot_training(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output = Path(output)
    report = json.loads((output / 'report.json').read_text())
    rows = [json.loads(line) for line in (output / 'steps.jsonl').read_text().splitlines()]
    steps = np.array([r['step'] for r in rows])
    probes = report['evaluations']
    probe_steps = [p['step'] for p in probes]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, names, title in zip(axes[0], (
            ('loss_all', 'loss_diversity', 'loss_distillation'),
            ('loss_pose', 'loss_axis', 'loss_axis_structure'),
            ('loss_recon', 'loss_nocs')),
            ('Total and auxiliary losses', 'Pose and axis losses', 'Completion and NOCS losses')):
        for name in names:
            values = np.array([r[name] for r in rows])
            window = min(20, len(values))
            ax.plot(steps[window-1:], np.convolve(values, np.ones(window)/window, mode='valid'),
                    label=name.removeprefix('loss_'))
        ax.set_title(title + ' (20-step mean)')
        ax.set_ylabel('Weighted loss')
        ax.set_yscale('log')
    for split in ('fixed', 'heldout'):
        axes[1, 0].plot(probe_steps, [p[split]['rotation_deg'] for p in probes], 'o-', label=split)
        axes[1, 2].plot(probe_steps, [p[split]['nocs_relative_error'] for p in probes], 'o-', label=split)
    axes[1, 0].set_title('Rotation error (symmetry-aware)')
    axes[1, 0].set_ylabel('Degrees')
    for name in ('translation_cm', 'size_cm', 'axis_cm'):
        axes[1, 1].plot(probe_steps, [p['heldout'][name] for p in probes], 'o-', label=name.removesuffix('_cm'))
    axes[1, 1].set_title('Disjoint-image development probe')
    axes[1, 1].set_ylabel('cm')
    axes[1, 2].axhline(1, color='gray', ls=':', label='Zero-coordinate baseline')
    axes[1, 2].set_title('NOCS error / zero-coordinate error')
    for ax in axes.flat:
        ax.axvline(report['args']['fixed_steps'], color='gray', ls='--', alpha=.6)
        ax.set_xlabel('Optimizer step')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(output / 'curves.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/REAL/camera_real.yaml')
    parser.add_argument('--output', default='log/training_diagnosis_direct_size')
    parser.add_argument('--fixed-steps', type=int, default=400)
    parser.add_argument('--stream-steps', type=int, default=800)
    args = parser.parse_args()
    seed(42)
    torch.set_num_threads(6)
    cfg = gorilla.Config.fromfile(args.config)
    optimizer_settings, scheduler_settings = copy.deepcopy(dict(cfg.optimizer)), copy.deepcopy(dict(cfg.lr_scheduler))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cfg.log_dir = str(output)
    data_cfg = copy.deepcopy(cfg.train_dataset)
    data_cfg.num_mini_batch_per_epoch = args.stream_steps
    loaders = create_dataloaders(data_cfg)
    heldout = []
    for loader in loaders.values():
        dataset = loader.dataset
        selected = np.random.choice(len(dataset.img_list), loader.batch_size, replace=False)
        samples = [dataset[int(i)] for i in selected]
        heldout.append({key: torch.stack([s[key] for s in samples]) for key in samples[0]})
        reserved = {dataset.img_list[i] for i in selected}
        dataset.img_list = [p for p in dataset.img_list if p not in reserved]
        dataset.reset()
        assert not reserved.intersection(dataset.img_list)
    fixed = tuple(next(iter(loader)) for loader in loaders.values())
    with open(cfg.train_dataset.init_tensor_list_dir, 'rb') as f:
        bank = pickle.load(f)
    model = Net(cfg.pose_net, load_ULIP_Pointbert(), bank).cuda().train()
    loss_fn = Loss(cfg.loss).cuda()
    logger = logging.getLogger('debug_training')
    logger.setLevel(logging.WARNING)
    solver = Solver(model, loss_fn, loaders, logger, cfg)
    writer = solver.tb_writer.writer
    parameters = [p for p in model.parameters() if p.requires_grad]
    modules = dict(IAKD=model.IAKD, FD=model.FD_Encoder, NOCS=model.nocs_predictor, pose=model.estimator)
    modules.update({name: getattr(model.completion, name) for name in
                    ('encoder', 'decoder', 'missing_head', 'score_head', 'refine_head', 'fold_head',
                     'axis_head', 'axis_refine_head', 'axis_embed')})
    rows, evaluations, gradients = [], [], []

    @torch.no_grad()
    def evaluate(batch):
        data = {k: torch.cat([part[k] for part in batch]).cuda() for k in batch[0]}
        pred = model(data)
        scale = data['size_label'].norm(dim=1)[:, None, None]
        target = canonical_to_camera(data['model'], data['completion_rotation_label'],
                                     data['translation_label'], data['size_label'])
        nocs_target = (pred['pred_kpt_3d'] - data['translation_label'][:, None]) / scale @ data['rotation_label']
        nocs_error = (pred['pred_kpt_nocs'] - nocs_target).abs().mean()
        nocs_zero = nocs_target.abs().mean()
        axes_target = canonical_to_camera(pred['axis_nocs'], data['rotation_label'],
                                          data['translation_label'], data['size_label'])
        symmetric = (data['category_label'].reshape(-1, 1) == loss_fn.symmetric_categories).any(1)
        weights = torch.ones(data['pts'].size(0), 3, device='cuda')
        weights[symmetric] = weights.new_tensor([0., 1., 0.])
        axis_error = (pred['axis_points'] - axes_target).norm(dim=-1).reshape(-1, 3, model.completion.axis_points).mean(2)
        rotation = pred['pred_rotation'] @ data['rotation_label'].transpose(1, 2)
        angle = ((rotation.diagonal(dim1=1, dim2=2).sum(1) - 1) / 2).clamp(-1, 1).acos()
        y_angle = (pred['pred_rotation'][:, :, 1] * data['rotation_label'][:, :, 1]).sum(1).clamp(-1, 1).acos()
        angle = torch.where(symmetric, y_angle, angle) * (180 / torch.pi)
        return dict(dense_cd_squared=chamfer_squared(pred['recon_model'], target).item(),
                    translation_cm=100 * (pred['pred_translation']-data['translation_label']).norm(dim=1).mean().item(),
                    size_cm=100 * (pred['pred_size']-data['size_label']).norm(dim=1).mean().item(),
                    rotation_deg=angle.mean().item(),
                    axis_cm=100 * ((axis_error * weights).sum(1) / weights.sum(1)).mean().item(),
                    nocs_l1=nocs_error.item(), nocs_zero_baseline=nocs_zero.item(),
                    nocs_relative_error=(nocs_error / nocs_zero.clamp_min(1e-8)).item(),
                    nocs_unrepresentable_fraction=(nocs_target.abs() > (0.5 - 0.5/64)).float().mean().item())

    def probe(phase, step):
        # Probes do not alter training RNG or the FD queue; cached images are fixed.
        rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
        model.eval()
        seed(123)
        result = dict(phase=phase, step=step, fixed=evaluate(fixed), heldout=evaluate(heldout))
        evaluations.append(result)
        for split in ('fixed', 'heldout'):
            for k,v in result[split].items(): writer.add_scalar(f'probe/{split}/{k}',v,step)
        (output/'evaluations.json').write_text(json.dumps(evaluations,indent=2))
        print('probe', json.dumps(result), flush=True)
        random.setstate(rng[0]); np.random.set_state(rng[1])
        torch.set_rng_state(rng[2]); torch.cuda.set_rng_state_all(rng[3])
        model.train()

    def train_step(batch, phase, data_seconds):
        started = time.perf_counter()
        solver.optimizer.zero_grad(set_to_none=True)
        loss, info, _ = solver.step_camera_real(tuple(dict(x) for x in batch), 'train')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, float('inf'), error_if_nonfinite=True)
        step = len(rows) + 1
        if step == 1 or step % 200 == 0:
            record = dict(step=step, norms={name: sum(p.grad.square().sum().item() for p in module.parameters()
                          if p.grad is not None)**.5 for name,module in modules.items()})
            gradients.append(record)
        assert torch.isfinite(loss)
        solver.optimizer.step()
        solver.lr_scheduler.step()
        torch.cuda.synchronize()
        row = dict(step=step, phase=phase, grad_norm=norm.item(), seconds=time.perf_counter()-started,
                   data_seconds=data_seconds, **info)
        rows.append(row)
        log.write(json.dumps(row)+'\n'); log.flush()
        if step == 1 or step % 20 == 0:
            for k,v in row.items():
                if k != 'phase': writer.add_scalar(f'train/{k}',v,step)
            print(f"{phase} {step}: loss={info['loss_all']:.4f} pose={info['loss_pose']:.4f} "
                  f"axis={info['loss_axis']:.4f} nocs={info['loss_nocs']:.4f} "
                  f"recon={info['loss_recon']:.4f} grad={norm.item():.2f}",flush=True)

    probe('initial',0)
    started = time.perf_counter()
    with (output/'steps.jsonl').open('w') as log:
        for i in range(args.fixed_steps):
            train_step(fixed,'fixed',0.)
            if (i+1)%200 == 0 or i+1 == args.fixed_steps: probe('fixed',len(rows))
        iterator=iter(zip(loaders['syn'],loaders['real']))
        for i in range(args.stream_steps):
            fetched=time.perf_counter()
            batch=next(iterator)
            train_step(batch,'stream',time.perf_counter()-fetched)
            if (i+1)%200 == 0 or i+1 == args.stream_steps: probe('stream',len(rows))
    summary={}
    for phase in ('fixed','stream'):
        items=[r for r in rows if r['phase']==phase]
        keys=[k for k in items[0] if k.startswith('loss_')]
        summary[phase]={window:{k:float(np.mean([r[k] for r in window_rows])) for k in keys}
                        for window,window_rows in [('first50',items[:50]),('last50',items[-50:])]}
    report=dict(args=vars(args), batch_size=cfg.train_dataset.syn_bs+cfg.train_dataset.real_bs,
                steps=len(rows), seconds=time.perf_counter()-started,
                step_seconds_median=float(np.median([r['seconds'] for r in rows[20:]])),
                step_seconds_p95=float(np.quantile([r['seconds'] for r in rows[20:]],.95)),
                mean_data_seconds=float(np.mean([r['data_seconds'] for r in rows if r['phase']=='stream'])),
                max_grad_norm=max(r['grad_norm'] for r in rows),
                peak_memory_gib=torch.cuda.max_memory_allocated()/2**30,
                optimizer=optimizer_settings,scheduler=scheduler_settings,
                summary=summary,evaluations=evaluations,gradients=gradients)
    (output/'report.json').write_text(json.dumps(report,indent=2))
    torch.save({k:v.cpu() for k,v in model.state_dict().items() if not k.startswith('ULIP_model.')},output/'model.pt')
    with (output/'tensor_list.pkl').open('wb') as f:
        pickle.dump([[t.cpu() for t in group] for group in model.tensor_list],f)
    writer.close()
    plot_training(output)
    print(json.dumps({k:v for k,v in report.items() if k not in ('evaluations','gradients')},indent=2),flush=True)


if __name__=='__main__':
    main()
