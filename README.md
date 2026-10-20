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
the vendored AdaPoinTr attention blocks. It combines 96 detected keypoints with
256 sampled point features, selects and refines 64 completion keypoints, then
expands them into 1,024 points for shape supervision. A parallel axis head
predicts three sampled coordinate axes (16 points per axis, 48 total). The
pose/size head uses **64 completion keypoints + 48 axis points**, with their
features and separate pooling for the surface and each axis. The dense 1,024
points are not pose-head inputs. Optimizer and LR scheduler settings are unchanged.

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
  --batch-size 40 --output log/axis_frame_smoke
```

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


