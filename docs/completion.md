# 关键点补全 → 轴结构点生成 → 位姿修正

当前路径为 `IAKD → GAFA → FD_Encoder → 补全/轴点生成 → 位姿尺寸`。
只有这一条路径，没有 completion_mode 或旧 Reconstructor 回退。

## 点数与注意力

[ComPose §3.2、§4.1](https://arxiv.org/html/2605.25553v1) 的 64+32 是候选数，评分后
保留 64 个关键点。本实现把补全放在 FD 之后，沿用该候选选择和完整 CAD 监督。

| 阶段 | 当前设置 |
|---|---|
| encoder memory | 96 个 IAKD 关键点及其 FD 特征，256 维 |
| 补全候选 | 64 个生成候选 + 32 个可见关键点候选 |
| decoder 查询 | 64 个补全关键点 + 48 个轴点 |
| 轴模板 | 每轴 16 点，标准坐标范围 [-0.4, 0.4] |
| 位姿头输入 | 64 个补全关键点 + 48 个轴点及对应特征 |
| 稠密补全 | 每关键点展开 16 点，共 1024 点，仅用于形状监督 |

已删除细粒度点的采样、投影与 memory 拼接，补全模块只接收 `(keypoints, features)`。
GAFA 本身仍按原结构使用可见点特征；它与本次移除的补全 memory 细粒度输入不同。
encoder/decoder 各两层、256 维、4 个注意力头。

两类查询共同参与全局 self-attention 和全局 cross-attention。decoder 第一层
额外对 **64 个物体关键点** 做局部 self/cross graph attention，K=8；第二层
仅使用全局注意力。局部 self-attention 的查询和邻居都是物体关键点，局部
cross-attention 的邻居是 96 个观测关键点。轴点完全不参与 KNN 选邻居，也不会
成为物体点的局部邻居。实现复用 AdaPoinTr 的注意力算子，见
[来源与许可](../model/adapointr/UPSTREAM.md)。

## 直接生成轴点

`axis_head` 与 `missing_head` 并行，从观测全局特征直接预测 48×3 个独立坐标偏移：

```text
coarse_axes = observed_center + observed_extent * (axis_template + point_offsets)
final_axes  = coarse_axes + observed_extent * pointwise_refinement(decoded_axis_features)
```

`axis_head` 的输出是 144 维；细化头逐点输出 3 维。没有轴分支的旋转头、平移头、
尺度头，也不把输出投影到刚性标架。模板只提供初始化形状、轴身份与采样顺序，
允许网络对每个轴点独立生成和修正，因此生成结果可以暂时不直、不正交。
观测中心和尺度来自 detached 关键点；尺度为 2.5 倍 RMS 半径，半径下限 0.02 m，
避免检测关键点初始聚集时轴模板退化到零。它不是预测的物体中心或物体尺寸。

标准轴在相机空间的监督目标仍用 GT 姿态构造：

```text
target_axes = axis_template * ||size_gt|| @ rotation_gt.T + translation_gt
```

这只是生成点的监督目标，不在推理时提供 GT，也不以网络预测的位姿生成轴点。
瓶/碗/罐（类别 0/1/3）仅监督可辨识的 Y 轴；其他类别监督三轴。

## 从轴点得到位姿，再作小幅修正

位姿头根据有序轴点与标准采样位置的线性相关，拟合三条轴的方向和长度，
用 X/Y 方向构造正交旋转，用三轴点的均值估计中心，用三轴拟合长度估计统一尺度。
这一步的方向是 **生成轴点 → 位姿**。

关键点与轴点特征分别池化，再融合用于修正：旋转采用 `identity6D + 0.1*tanh(residual)`
构造局部旋转并乘到轴点读取的旋转上；平移修正限制为每分量 `0.1*axis_scale`。
主要旋转和中心必须由轴结构承担，特征分支负责小幅修正。尺寸沿用 CleanPose 的
三维直接回归 `s = size_estimator(pose_feat)`，不乘轴尺度，也不施加 softplus。
轴尺度仅用于平移修正的幅度，不约束尺寸输出；轴点自身的尺度监督继续保留。

## 梯度边界

- encoder 的坐标输入、decoder 的坐标位置编码和 KNN 坐标 detach。
- 候选评分的坐标输入 detach，评分训练置信特征，不通过移动候选来提高评分。
- 位姿头的逐点坐标/NOCS 特征编码输入 detach；NOCS 几何监督目标仍 detach。
- **轴点坐标进入方向、中心、尺度读取时不 detach**，最终位姿损失可反传到粗轴点头、
  细化轴点头及共享特征。粗轴点到细轴点的残差路径也保留梯度。
- 所有关键点/轴点特征保持可学习；物体补全坐标由完整 CAD 损失训练，IAKD 仍有
  观测表面和多样性监督，以及特征路径的梯度。

## 损失与保存

保留三阶段完整 CAD 平方 Chamfer（权重 15）、候选评分（1）、NOCS（2）、
几何关系（1）、位姿（0.3）、原观测点/蒸馏监督。`recon_delta` 是真实局部展开
偏移，权重 0.05，在相机空间满足：

```text
recon_model = pred_kpt_3d.repeat_interleave(16, dim=1) + recon_delta
```

轴点监督为尺度归一化 Smooth L1：细点 + 0.5×粗点，权重 1。轴长度监督权重 0.1。
新增权重 0.05 的软结构先验：直线/均匀采样、共中心、方向正交与右手方向。
这些只约束生成点，不在前向中强制变换模板生成刚性轴。完整 CAD 使用原始
`completion_rotation_label`，轴/NOCS 目标使用现有对称规范化 `rotation_label`。

当前用户配置 batch size 为 40（30 CAMERA + 10 REAL），保留用户的最新修改。
Adam 和 CyclicLR 保持不变，学习率范围 `2e-5 → 5e-4`。每 **2 个 epoch** 保存 `epoch_N.pt`，并保存
匹配的 `tensor_list_epoch_N.pkl` 和更新兼容路径 `tensor_list.pkl`。最后一个 epoch
也会保存，即使轮数为奇数。测试旧 epoch 时应使用与其匹配的 FD 队列。
保存的是模型权重和 FD 队列，不是含优化器状态的断点续训包。

本次改变了轴头、decoder 和位姿读出方式，旧结构 checkpoint 不能 strict 加载。
新实验应重新训练，并使用新的实验目录：

```bash
.pixi/envs/default/bin/python train.py \
  --config config/REAL/camera_real.yaml --gpus 0 --extra _axis_generation
```

## 本次验证

下面 545 步记录对应此前的尺寸耦合版本；直接回归尺寸后的调试记录单独保存在
`log/training_diagnosis_direct_size/`，不能把旧记录当作新版本的测量结果。

```bash
.pixi/envs/default/bin/python tools/smoke_completion.py \
  --fixed-steps 240 --stream-steps 240 --peak-steps 60 --full-steps 5 \
  --batch-size 40 --output log/axis_generation_smoke_b40
```

真实 CAMERA/REAL、生产 Solver、完整 DINO/PointNet++/ULIP，batch size 40，共 545 步。
固定批次前/后 20 步平均总损失 **9.964 → 1.867**。所有损失和梯度有限，所有生成
模块均收到梯度，没有梯度裁剪。最大梯度范数 88.28；最高学习率阶段前/后 20 步
平均总损失 **1.328 → 0.855**。峰值已分配显存 15.92 GiB。

下表是在同一训练探针批次上的 eval 指标，不是独立测试集成绩。

| 指标 | 初始 | 固定批次 240 步后 | 全部 545 步后 |
|---|---:|---:|---:|
| 平方 Chamfer | 0.005954 | 0.000565 | 0.000526 |
| 轴点平均距离（m，按对称性屏蔽） | 0.07567 | 0.03854 | 0.02751 |
| 平移 L2（m） | 0.03690 | 0.02289 | 0.01783 |
| 尺寸 L2（m） | 0.21357 | 0.07663 | 0.05264 |
| 旋转矩阵 Frobenius 误差 | 2.68394 | 0.75729 | 0.41267 |
| NOCS L1 | 0.06133 | 0.09432 | 0.08793 |

NOCS 误差尚未低于初始值；当前结果证明短程可学习与数值可运行，不能证明最终
泛化精度优于旧结构。需要完整训练和 REAL275 评估。

针对设计的检查确认：KNN 的查询数仅为 64、轴头输出 144 维独立点坐标、单独位姿
损失可以训练轴生成头、被 detach 的坐标路径没有梯度、理想轴点可正确读回已知
位姿，以及第 2 轮和奇数最后一轮能保存权重与配套队列。
当时的单样本推理检查覆盖 strict 权重加载、实际 memory/位姿输入点数、合法旋转、
正尺寸与 `recon_delta` 坐标还原恒等式（当前直接尺寸回归不再强制正值）。本地记录位于
`log/axis_generation_smoke_b40/report.json`、`steps.jsonl`、`design_checks.json` 和
`inference_checks.json`；日志目录被 Git 忽略，不包含在仓库中。
此前 batch size 4 的 545 步检查记录单独保留在 `log/axis_generation_smoke/`。
