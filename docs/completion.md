# 补全关键点 + 坐标轴代理

当前只有一条网络路径：`IAKD → GAFA → FD_Encoder → 补全/轴点 → NOCS/位姿尺寸`。
已移除 `completion_mode`、旧原生适配器和 Reconstructor 回退。训练入口仍为：

```bash
.pixi/envs/default/bin/python train.py --config config/REAL/camera_real.yaml
```

## 数量与网络结构

[ComPose §3.2、§4.1](https://arxiv.org/html/2605.25553v1) 的 `64 + 32` 是候选数：
64 个生成候选 + 32 个可见候选，评分选出 64 个关键点，再每点展开 16 个局部点。
当前配置保留这个关系，同时保留 CleanPose 原有的 96 个 IAKD 观测关键点。

| 阶段 | 数量 / 维度 |
|---|---|
| IAKD / GAFA / FD 观测关键点 | 96 × 256 |
| 细粒度可见点记忆 | FPS 256 × 256 |
| 补全候选 | 64 missing + 32 visible = 96 |
| 筛选、细化后的补全关键点 | 64 × 256 |
| 坐标轴代理 | 每轴 16 点，共 48 点 × 256 |
| 共享 decoder 查询 | 64 + 48 = 112，2 层 |
| 位姿/尺寸头输入 | **64 个补全关键点 + 48 个轴点及对应特征** |
| 稠密补全输出 | 64 × 16 = 1024 点，仅用于完整形状监督 |

代理点总数不超过关键点数，默认 48 < 64。网络不会把 1024 个稠密点送进位姿头。
编码器/解码器采用本地 AdaPoinTr 几何感知注意力，深度各 2 层、通道 256、4 头。
ComPose 官方仓库尚无实现，本模块是依据论文的适配。来源和许可见
[UPSTREAM.md](../model/adapointr/UPSTREAM.md)。

## 轴代理的定义与特征

标准 NOCS 系中的三轴均在 `[-0.4, 0.4]` 均匀采样 16 点，每条线长度 0.8。
使用偶数个点避免原点在三条轴上重复。采样位置、轴身份和正负方向固定。
这是虚拟的坐标系代理，不是要求 CAD 表面上必须存在的真实点。

`axis_head` 与 `missing_head` 并行，从同一全局特征预测共享原点（3）、6D 旋转（6）
和正尺度（1），共 10 个数。6D 旋转经正交化、尺度经 softplus 后变换固定模板，
得到粗轴点。因此三轴严格共原点、等长、正交并构成右手系。
预测使用可见点尺度归一化，不依赖 GT。轴点位置编码加标准模板身份编码，
与补全查询共同经过 decoder，得到每个轴点的独立 256 维特征。随后按三轴汇聚
这些特征，经 `axis_refine_head` 修正同一组 10 维参数，输出细化轴点。

这三条标准轴按 `||size_gt||` 统一缩放，目标为：

```text
axis_camera = axis_template * ||size_gt|| @ rotation_gt.T + translation_gt
```

代理轴表示旋转、原点和统一尺度。物体三边长仍由位姿尺寸头结合补全关键点
几何和特征学习，不把三条等长代理轴误当作物体长宽高。

位姿头接收补全关键点的预测 NOCS、轴点已知的标准模板坐标，以及两类点的特征。
它先提取逐点几何/特征，使用表面与轴等权的全局上下文，再分别池化表面、X/Y/Z
四组特征后融合。这样不会因采样数变化而让一类点淹没另一类点。

## 监督

完整 CAD 使用 `completion_rotation_label` 的原始旋转还原到相机空间；轴点与
NOCS 使用现有对称规范化后的 `rotation_label`。所有预测先还原随机几何增强
和点云中心化，再计算损失。轴坐标误差除以 GT 尺度，避免大物体主导训练。

瓶、碗、罐（0-index 类别 0/1/3）绕 Y 轴的旋转不可辨识，因此仅对 Y 轴作
有序点监督，不给 X/Z 强加任意方位；三轴的等长、正交和右手结构由参数化保证。
其他类别监督全部三轴。这个约束思路与 [FS-Net 的对称轴处理](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_FS-Net_Fast_Shape-Based_Network_for_Category-Level_6D_Object_Pose_Estimation_CVPR_2021_paper.html)
一致，但这里使用采样轴点与共享特征。损失只使用监督标签，推理无需 CAD 或 GT。

| 损失 | 内容 | 权重 |
|---|---|---:|
| recon | missing、细化关键点、稠密点与完整 CAD 的三阶段平方 Chamfer | 15 |
| score | 候选评分对 `exp(-到 CAD 最近距离 / 0.05)` 的 MSE | 1 |
| axis | 尺度归一化 Smooth L1：细轴点 + 0.5 × 粗轴点，按对称性屏蔽不可辨识轴 | 1 |
| axis_scale | 代理轴预测尺度 / GT 尺度与 1 的 Smooth L1 | 0.1 |
| delta | 真实局部展开偏移的平均 L2 范数 | 0.05 |
| geo | 补全关键点与 NOCS 的距离关系一致性 | 1 |
| NOCS / pose / distillation / observed CD / diversity | 保留现有监督 | 保留原值 |

`recon_delta` 现在是实际输出，满足相机坐标系下：

```text
recon_model = pred_kpt_3d.repeat_interleave(16, dim=1) + recon_delta
```

它在训练和推理时都返回，也参与正则。只约束稠密点相对所属关键点的局部偏移，
不限制 missing 点对缺失部分的全局补全。由于补全采用平方 Chamfer，把 delta
权重从 1 降为 0.05，避免局部点过度收缩到关键点。NOCS/geo 几何目标 detach；
原可见表面与多样性损失只作用于 IAKD 观测关键点。

Adam 和 CyclicLR 参数不变，实际学习率范围为 `2e-5 → 5e-4`。冻结 DINO/ULIP
使用 eval 模式，FD 队列只存 FD 特征。本路径要求完整 CAD 监督，当前旧
HouseCat6D loader/config 不满足要求；没有保留另一种训练路径。
旧架构 checkpoint 不能 strict 加载到新架构，需要重新训练。

## 验证方法

```bash
.pixi/envs/default/bin/python tools/smoke_completion.py
.pixi/envs/default/bin/python tools/smoke_completion.py \
  --fixed-steps 240 --stream-steps 240 --peak-steps 60 --full-steps 5 \
  --batch-size 40 --output log/axis_frame_smoke
```

使用真实 CAMERA/REAL、完整 DINO/PointNet++/ULIP 和生产 Solver。先固定批次
检查可学习性，再切换样本和跳到现有调度器最高学习率检查稳定性；没有梯度裁剪
或降低优化器学习率。日志、指标、权重及 FD 队列保存在对应输出目录。
冒烟中的固定批次指标只证明短程可学习性；最终精度收益需要完整训练和独立
REAL275 评估，不能从训练损失下降推断。

## 本次实测

最终 10 维正交标架版本完成 545 步、batch size 40：固定批次 240 步、混合样本
240 步、最高学习率 60 步、配置批次新样本 5 步。所有损失和梯度有限，各补全
及轴点模块均收到非零梯度。固定阶段前/后 20 步平均总损失为 **10.302 → 1.859**，
轴监督为 **0.263 → 0.103**。峰值已分配显存 16.81 GiB。

下表为同一训练探针批次在 eval 模式下的指标，不是独立测试集成绩；末列在全部
545 步后测量，包含最高学习率压力阶段。旋转误差使用矩阵 Frobenius 范数。

| 指标 | 初始 | 固定阶段后 | 全部步骤后 |
|---|---:|---:|---:|
| 稠密平方 Chamfer | 0.005359 | 0.000407 | 0.000419 |
| 轴点平均距离（m，按对称性屏蔽） | 0.09145 | 0.03182 | 0.02224 |
| 平移 L2（m） | 0.04687 | 0.02842 | 0.02788 |
| 尺寸 L2（m） | 0.20440 | 0.08952 | 0.05291 |
| 旋转矩阵误差 | 2.8123 | 0.4466 | 0.7074 |

最终权重还通过单样本推理、实际 112 点位姿输入、真实 `recon_delta` 恒等式和
对称轴监督检查。仅反传位姿损失时，轴粗预测/细化/身份编码、共享 decoder、
关键点细化和 FD 均有梯度；稠密展开头无梯度，确认稠密点没有送入位姿头。
预测三轴正交误差小于 `6e-7`、公共原点误差小于 `1.2e-7`、基底行列式为 1。

记录：[训练指标](../log/axis_frame_smoke/report.json)、
[逐步损失](../log/axis_frame_smoke/steps.jsonl)、
[几何与梯度检查](../log/axis_frame_smoke/geometry_checks.json)。
冒烟权重和配套 FD 队列位于 `log/axis_frame_smoke/smoke.pt` 与 `tensor_list.pkl`；
该权重仅用于检查，不替代正式训练的模型。
