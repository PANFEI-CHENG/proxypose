# [ICCV2025] CleanPose: Category-Level Object Pose Estimation via Causal Learning and Knowledge Distillation
### Overview
Official implementation of our CleanPose, the first solution to mitigate the confoundering effect in category-level pose estimation via causal learning and knowledge distillation.

![](/main_v3.png)

[![Conference](https://img.shields.io/badge/ICCV-2025-slateblue)](https://iccv.thecvf.com/Conferences/2025) [![Paper](https://img.shields.io/badge/arXiv-2502.01312-orange)](https://arxiv.org/pdf/2502.01312)

## Environment Settings
The code has been tested with

- python 3.9
- torch 1.12
- cuda 12.4

Some dependencies:
```
pip install gorilla-core==0.2.5.3
pip install opencv-python
pip install einops

cd model/pointnet2
python setup.py install
```

## Data Preparation
### NOCS dataset
- Download and preprocess the dataset following [DPDN](https://github.com/JiehongLin/Self-DPDN)
- Download and unzip the segmentation results [here](http://home.ustc.edu.cn/~llinxiao/segmentation_results.zip)

Put them under ```PROJ_DIR/data```and the final file structure is as follows:
```
data
├── camera
│   ├── train
│   ├── val
│   ├── train_list_all.txt
│   ├── train_list.txt
│   ├── val_list_all.txt
├── real
│   ├── train
│   ├── test
│   ├── train_list.txt
│   ├── train_list_all.txt
│   └── test_list_all.txt
├── segmentation_results
│   ├── CAMERA25
│   └── REAL275
├── camera_full_depths
├── gts
└── obj_models
```
### HouseCat6D
Download and unzip the dataset from [HouseCat6D](https://sites.google.com/view/housecat6d) and the final file structure is as follows:
```
housecat6d
├── scene**
├── val_scene*
├── test_scene*
└── obj_models_small_size_final
```

## Other Preparation
### Confounder queue generation
You can generate the queue list with following command or utilize the pre-extracted pkl file from [this link](https://drive.google.com/drive/folders/15D9kkISuEP1z6yBZhBItdp4N26wDYcJB?usp=drive_link). Subsequently, you should specify the 'init_tensor_list_dir' in the config file.
```
python queue_extraction.py
```
### ULIP model weights
Download the pretrained weights of PointBERT, Pointnet2_ssg from [ULIP Hugging Face](https://huggingface.co/datasets/SFXX/ulip) or [this link](https://drive.google.com/drive/folders/1yQhaP7AWtgu5NOW1GTVO23ytcGLt_wPW?usp=drive_link). If you want to use ResNet as the visual encoder, you can download the weights of ResNet18 (optional). The pretrained weights file structure is as follows:
```
model
├── pointbert
│   ├── pretrained_model
│       └── pretrained_models_ckpt_zero-sho_classification_pointbert_ULIP-2.pt
├── pointnet2
├── pointnet2_ulip
│   ├── pretrained_model
│       └── pretrained_models_ckpt_zero-sho_classification_checkpoint_pointnet2_ssg.pt
└── resnet18-5c106cde.pth (optional)
```

## Train
### Training on NOCS
```
python train.py --config config/REAL/camera_real.yaml
```
The NOCS config enables ComPose-style completion immediately after `FD_Encoder`,
implemented in [`model/compose_completion.py`](model/compose_completion.py) using
the vendored AdaPoinTr attention blocks. Its memory contains only the 96 detected
keypoints, with their FD features and coordinate encodings. It selects and refines
64 completion keypoints, then expands them into 1,024 points for shape supervision.
In the same decoder, 48 **axis-structure proxy** queries (16 per object axis), each
embedded from its object-frame template coordinate, are decoded jointly with the
completion queries. After every decoder layer, their camera-frame coordinates are
regressed directly, without an observed origin/extent anchor or template offsets.
Proxies use global attention only; local KNN attention only processes surface
keypoints. Proxy supervision is a scale-normalized point distance on every decoder
layer. For bottle/bowl/can, the GT yaw is aligned to the prediction in closed form,
so all three axes are supervised. R and t come from `self.estimator` and use only
the proxies: the pose is solved from the **detached** proxies and their object-frame
template coordinates, then refined with small corrections from the proxy features.
There is no keypoint NOCS branch. Size is estimated SecondPose-style: each of
the 64 completion keypoints votes a category-specific size from its decoder feature,
and the votes are averaged. Training uses AdamW (peak LR 5e-4, weight decay 0.05,
none on biases/norm weights), one epoch of linear warmup, cosine decay to 1e-6 and
gradient-norm clipping at 20. Weights and matching FD queues are saved every two
epochs and at the final epoch.

There is one completion/axis path; the old mode switches, native adapter and
Reconstructor fallback have been removed. `recon_delta` contains the actual
local reconstruction offsets, with a weak regularization weight of 0.05.
Axis point count must not exceed the completion keypoint count. Current training
requires NOCS full-CAD supervision; the legacy HouseCat6D loader/config does not
provide that supervision. Previous architecture checkpoints require retraining.

See [implementation and training notes](docs/completion.md) for supervision,
adaptations, and smoke-training results. To reproduce the real-data smoke run:

```bash
.pixi/envs/default/bin/python tools/smoke_completion.py \
  --fixed-steps 240 --stream-steps 240 --peak-steps 60 --full-steps 5 \
  --batch-size 40 --output log/axis_generation_smoke_b40
```

For a longer convergence diagnosis using the configured batch sizes and learning
rate schedule, run:

```bash
.pixi/envs/default/bin/python tools/debug_training.py
```

This uses 400 fixed-batch steps followed by 800 changing-batch steps and reserves
disjoint images for a development probe. Per-loss curves, gradient norms, timings
and TensorBoard events are written to `log/training_diagnosis_direct_size/`.

## Evaluate 
- Evaluate on NOCS:
```
python test.py --config config/REAL/camera_real.yaml --test_epoch 30
```
## Results
The tables below report the original CleanPose results, not this completion/axis modification.
You can download our training logs, detailed metrics for each category and checkpoints [here](https://drive.google.com/file/d/1JayhnaGBPBsp-wmjGcoPF5_Nka04ETng/view?usp=drive_link).
### REAL275 testset:

| IoU25 | IoU50 | IoU75 | 5 degree 2 cm | 5 degree 5 cm | 10 degree 2 cm | 10 degree 5 cm |
|---|---|---|---|---|---|---|
| 83.3 | 81.2 | 62.7 | 61.7| 67.6 | 78.3 | 86.3 |

### CAMERA25 testset:
| IoU25 | IoU50 | IoU75 | 5 degree 2 cm | 5 degree 5 cm | 10 degree 2 cm | 10 degree 5 cm |
|---|---|---|---|---|---|---|
| 94.8 | 94.3 | 92.5 | 80.3 | 84.2 | 87.7 | 92.7 |

### HouseCat6D testset:
| IoU25 | IoU50 | IoU75 | 5 degree 2 cm | 5 degree 5 cm | 10 degree 2 cm | 10 degree 5 cm |
|---|---|---|---|---|---|---|
| 89.2 | 79.8 | 53.9 | 22.4 | 24.1 | 51.6 | 56.5 |

## Visualization
For visualization, please run
```
python visualize.py --config config/REAL/camera_real.yaml --test_epoch 30
```

To randomly compare the predicted completion/axis proxy points with the ground-truth
CAD points/axes of a category in Open3D:

```bash
.pixi/envs/default/bin/python demo_completion_axis.py --category camera
```

Valid categories are `bottle`, `bowl`, `camera`, `can`, `laptop`, and `mug`.
Use `--seed 42` for a repeatable sample or `--no-window` to only export PLY files
under `log/completion_axis_demo/`. Prediction uses cyan and bright RGB; ground truth
uses yellow and light RGB. Every geometry is a named layer that can be toggled.

## Acknowledgements
Our implementation leverages the code from [DPDN](https://github.com/JiehongLin/Self-DPDN), [AG-Pose](https://github.com/Leeiieeo/AG-Pose) and [GOAT](https://github.com/CrystalSixone/VLN-GOAT). Thank them for their excellent works!
## Citation
If our work is useful to you, please consider citing our paper using the following BibTeX entry.
```
@inproceedings{lin2025cleanpose,
  title={Cleanpose: Category-level object pose estimation via causal learning and knowledge distillation},
  author={Lin, Xiao and Peng, Yun and Wang, Liuyi and Zhong, Xianyou and Zhu, Minghao and Yang, Jingwei and Feng, Yi and Liu, Chengju and Chen, Qijun},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
```
## License
Our code is released under MIT License. 
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)


