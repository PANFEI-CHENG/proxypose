# 关键点补全 + 轴结构代理点 → 位姿与尺寸

当前路径为 `IAKD → GAFA → FD_Encoder → 补全与轴结构代理点联合解码 → 位姿 (R, t) / 尺寸 s`。
只有这一条路径，没有 completion_mode 或旧 Reconstructor 回退。

## 点数与注意力

[ComPose §3.2、§4.1](https://arxiv.org/html/2605.25553v1) 的 64+32 是候选数，评分后
保留 64 个关键点。本实现把补全放在 FD 之后，沿用该候选选择和完整 CAD 监督。

| 阶段 | 当前设置 |
|---|---|
| decoder memory | 96 个 IAKD 关键点：FD 特征 + 坐标位置编码，256 维 |
| 补全候选 | 64 个生成候选 + 32 个可见关键点候选 |
| decoder 查询 | 64 个补全关键点 + 48 个轴结构代理点 |
| 代理点模板 | 每轴 16 点，物体坐标 [-0.4, 0.4]（以 ‖size‖ 为单位） |
| 位姿头输入 | 只有 48 个代理点（坐标 detach）、它们的模板物体坐标和特征 |
| 尺寸头输入 | 64 个补全关键点（坐标 detach）及其 decoder 特征 |
| 稠密补全 | 每关键点展开 16 点，共 1024 点，仅用于形状监督 |

encoder 目前在代码中注释掉；decoder 两层、256 维、4 个注意力头。memory 加上关键点
坐标的位置编码（与补全查询共用 `position` MLP）：去掉 encoder 后 memory 不再含显式
坐标，而代理点需要在相机坐标中直接回归。

两类查询共同参与全局 self-attention 和全局 cross-attention。decoder 第一层额外对
**64 个补全关键点** 做局部 self/cross graph attention，K=8；第二层仅使用全局注意力。
代理点位于物体坐标轴上而不是物体表面，KNN 邻域没有几何意义，所以代理点既不是
KNN 查询，也不会成为任何点的邻居。实现复用 AdaPoinTr 的注意力算子，见
[来源与许可](../model/adapointr/UPSTREAM.md)。

## 轴结构代理点

48 个代理点有固定的物体坐标 `c_k = τ_m e_a`（a ∈ {x, y, z}，τ_m 为 16 个等距刻度）。
每个代理点查询由其物体坐标嵌入与实例全局特征相加得到，与补全查询一起解码；每层
decoder 之后由共享的 `axis_head` 直接回归该代理点在相机坐标系（网络的中心化/增强
坐标系，输出时还原）中的坐标：

```text
query_k     = global_feature + axis_embed(c_k)
p_k^(l)     = axis_head(LayerNorm(decoder_layer_l(query))_k),   l = 1, 2
axis_points = p^(2)
```

没有观测中心 (origin)、观测尺度 (extent)、"模板 + 偏移"，也不预测 R/t/scale 去变换模板；
48 个点各自独立回归，结果可以暂时不直、不正交。

监督目标由 GT 位姿构造，只在训练时使用：

```text
target_k = ‖size_gt‖ · R_gt · R_y(ψ) · c_k + t_gt
```

非对称类别 ψ = 0。瓶/碗/罐（类别 0/1/3）绕自身 Y 轴的 yaw 不可观测，ψ 取与当前预测
最吻合的 yaw，有闭式解。记 `q_k = R_gtᵀ (p_k − t_gt)`：

```text
ψ = atan2( Σ_k (q_x c_z − q_z c_x),  Σ_k (q_x c_x + q_z c_z) )
```

所以对称类别的 X/Z 轴也受到监督（只差一个不可观测的 yaw），不再只监督 Y 轴。

## 代理点监督

```text
L_axis = Σ_l mean_k ‖p_k^(l) − target_k‖₂ / ‖size_gt‖,   权重 2.0
```

相对旧版（细点 + 0.5×粗点的尺度归一化 Smooth L1，β = 0.05，权重 1）的加强：

- 欧氏距离代替平方型 Smooth L1，误差变小时梯度不衰减；
- 每个 decoder 层都监督（deep supervision）；
- 对称类别也监督全部 48 个点；
- 位姿损失不再回传到代理点（见下），代理点只由 `L_axis` 训练。

已删除 `axis_scale_loss` 和 `axis_structure_loss`（长度、直线、共中心、正交、右手系软先验）：
所有代理点都有 GT 监督后，这些先验是冗余的。

## 位姿：代理点与物体坐标的对应关系

R 和 t 只由 48 个代理点及其模板物体坐标（代理点的 NOCS）求解，不再使用补全关键点
和它们的 NOCS，因此也去掉了 NOCS 预测器及其 `loss_nocs`、`loss_geo`。

`self.estimator`（`PoseEstimator`，即原 `PoseSizeEstimator` 去掉尺寸分支和表面关键点
输入）：用每根轴的代理点和刻度 τ 做最小二乘直线拟合得到方向 `v_a`，用 X/Y 方向
（Y 优先的 Gram-Schmidt）构造旋转，用代理点均值作中心；再把代理点特征按轴分别池化
（3 × 512），输出小幅修正：旋转 `identity6D + 0.1·tanh(residual)`，平移每分量不超过
`0.1·axis_scale`。

代理点在进入 estimator 前 **detach**：位姿损失只训练修正分支和代理点特征，不移动代理点。

## 尺寸：SecondPose 的 t/s 分支结构

尺寸不再由 estimator 回归，而由 `SizeEstimator` 从补全关键点及其特征估计。结构对应
SecondPose/VI-Net `PN2.Net` 中 `s_mlp` 的末端：

```text
每个关键点: [MLP(xyz) (64), decoder 特征 (256)]
  → Conv1d-BN-ReLU (256) → Conv1d-BN-ReLU (128) → Dropout(0.2) → Conv1d (3 × 6 类)
  → 对 64 个关键点取平均 → 取所属类别的 3 维尺寸
```

关键点坐标 detach，特征保持可学习。尺寸仍由 `loss_pose` 中的尺寸项监督。

## 梯度边界

- 补全模块输入的 IAKD 关键点、decoder 的坐标位置编码和 KNN 坐标 detach。
- 候选评分的坐标输入 detach，评分训练置信特征，不通过移动候选来提高评分。
- 位姿头：代理点坐标 detach（几何读出和逐点编码都不回传）；代理点的 **特征** 保持可学习。
- 尺寸头：补全关键点坐标 detach，特征可学习。
- 检查结果：`loss_pose` 对 `axis_head` 的梯度恰好为 0；`loss_axis` 对 estimator 和
  尺寸头的梯度恰好为 0。

## 优化器

AdamW（betas 0.9 / 0.999，weight decay 0.05；bias 和归一化层权重等 1 维参数不衰减），
按 iteration 调度：线性 warmup 4000 iter（1 个 epoch，从 0.001×lr 起）到峰值 5e-4，
再余弦衰减到 1e-6，总长 `max_epoch × num_mini_batch_per_epoch = 30 × 4000`。
全局梯度范数裁剪 `grad_clip: 20`：warmup 之后梯度范数中位数约 10，只有离群 batch
（下文的 CAMERA 补全 Chamfer 尖峰）会被裁剪。
gorilla 的构建器不能设置参数组，所以 `utils/solver.py` 中的 Solver 自己构建 AdamW 和
调度器；其他 optimizer / scheduler 名字（例如旧的 Adam + CyclicLR）仍交给 gorilla 构建。

## 损失与保存

保留三阶段完整 CAD 平方 Chamfer（权重 15）、候选评分（1）、位姿（0.3）、
原观测点/蒸馏监督；NOCS（2）和几何关系（1）随 NOCS 分支一起删除。`recon_delta`
是真实局部展开偏移，权重 0.05，在相机空间满足：

```text
recon_model = pred_kpt_3d.repeat_interleave(16, dim=1) + recon_delta
```

完整 CAD 使用原始 `completion_rotation_label`，代理点目标使用现有对称规范化的
`rotation_label`。

学习率按 batch size 40（30 CAMERA + 10 REAL）选择。每 **2 个 epoch** 保存 `epoch_N.pt`，
并保存匹配的 `tensor_list_epoch_N.pkl` 和更新兼容路径 `tensor_list.pkl`。最后一个 epoch
也会保存，即使轮数为奇数。测试旧 epoch 时应使用与其匹配的 FD 队列。
保存的是模型权重和 FD 队列，不是含优化器状态的断点续训包。

本次改变了代理点头、尺寸头和损失，旧结构 checkpoint 不能 strict 加载。
新实验应重新训练，并使用新的实验目录：

```bash
.pixi/envs/default/bin/python train.py \
  --config config/REAL/camera_real.yaml --gpus 0 --extra _axis_proxy
```

## 本次验证

`tests/test_axis_proxy.py` 检查：对称 yaw 对齐对任意 yaw 都能精确还原预测、非对称目标
等于 GT、对齐后的目标不会比 GT 更远、理想代理点经 estimator 读回 GT 位姿（误差 < 1e-5）、
学习率曲线（warmup 起点、峰值、单调衰减、终点 min_lr）和不衰减参数组：

```bash
.pixi/envs/default/bin/python tests/test_axis_proxy.py
```

另外确认：`loss_pose` 对 `axis_head` 的梯度为 0，`loss_axis` 对 estimator / 尺寸头的
梯度为 0；生产 `Solver.solve()` 跑通两个小 epoch（裁剪、学习率日志、保存
`epoch_N.pt` 与 FD 队列），并能按 `test.py` 的方式 strict 加载。

### 学习率扫描

真实 CAMERA/REAL、batch 30 + 10、AdamW（wd 0.05），压缩日程：2000 iter
（200 iter warmup + 余弦）。探针为从训练列表中移除的 120 张 CAMERA + 40 张 REAL
训练图像（与训练图像不重叠，但来自训练场景），**不是 REAL275 测试集**，数字不能与
benchmark 结果比较；压缩日程也不等于 120k iter 的完整训练。下表为第 2000 iter 的
REAL 探针结果，括号内为第 500 iter 的代理点误差：

| 峰值 lr | 裁剪 | 末 200 iter 损失 | 旋转 ° | 平移 cm | 尺寸 cm | 代理点 cm | 5°2cm | 10°5cm |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2.5e-4 | 无 | 0.968 | 10.6 | 0.98 | 1.95 | 1.36 (3.65) | 17.5% | 57.5% |
| 5e-4 | 无 | 0.870 | 6.6 | 0.80 | 2.02 | 1.14 (3.22) | 52.5% | 82.5% |
| 1e-3 | 无 | 0.856 | 7.0 | 0.89 | 1.56 | 1.19 (6.02) | 47.5% | 77.5% |
| **5e-4** | **20** | 0.875 | 5.7 | 0.82 | 1.73 | 1.12 (3.02) | 57.5% | 92.5% |

CAMERA 探针（120 张）趋势相同：5°2cm 依次为 5.8%、50.0%、48.3%、50.8%。
2.5e-4 收敛明显更慢；1e-3 在峰值附近不稳定（第 500 iter REAL 代理点误差 6.0 cm、平移
4.7 cm，约为 5e-4 的两倍），衰减后才追上；5e-4 全程最好。四个运行都没有非有限值；
每个运行约 8 次单步损失尖峰，全部来自 CAMERA 部分的补全 Chamfer（缺失 / 关键点 /
稠密三项同时跳变，下一步立即恢复），是离群样本而不是优化发散。裁剪 20 只作用于
4.4% 的 iter（warmup 后 0.34%），最终损失不变、探针相当或略好（40 张 REAL 图像，
差 2.5% 即一张图），因此保留。

以上扫描在去掉 NOCS 分支之前完成。去掉后用同一设置（5e-4、裁剪 20、同一种子与数据）
重跑：第 2000 iter CAMERA 探针旋转 6.83°（原 6.89°）、代理点 1.14 cm（原 1.19 cm）、
5°2cm 55.0%（原 50.8%）；REAL 探针旋转 5.87°（原 5.65°）、代理点 1.07 cm（原 1.12 cm）、
5°2cm 52.5%（原 57.5%）、10°5cm 80.0%（原 92.5%）。REAL 两次带 NOCS 的运行之间也差了
10 个百分点，所以这些差别都在 40 张图的波动范围内；前期收敛更快（第 500 iter 代理点
误差 2.5–2.7 cm，原 3.0–3.4 cm）。是否有影响需要完整训练和 REAL275 测试确认。
