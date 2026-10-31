---
title: sea_raft:一种改进的raft光流方法
date: 2026-10-22 12:00:00
tags:
  - 学习笔记 立体匹配 光流
categories:
  - 技术
---


SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow 
本文提出了一种改进的raft方法，提高了精度和效率：相比于原始raft，本文有四个改进点：\
(1)混合拉普拉斯损失函数：这里借鉴了稀疏匹配的方法，主要为了处理遮挡而引起的歧义区域 \
(2)根据上下文直接预测初始的光流，使用类似flownet的方法，减少迭代次数 \
(3)简化算法结构：特征提取直接使用imagenet的预训练resnet作为backbone，其次不使用ConvGRU 而是使用ConvNeXt \
(4)在刚体光流TartanAir数据集进行预训练提高了泛化性
# 1 主要方法
1.1回顾raft算法结构，剩余节为sea raft改进点。
## 1.1 raft算法架构回顾
输入左右图像$I_1,I_2$，首先获取$I_1,I_2$相似性特征图$F(I_1),F(I_2)$和$I_1$上下文特征图$C(I_1)$。然后计算左右图的金字塔特征相似性：

$$
V_k = F(I_1) \circ \operatorname{AvgPool}\left(F(I_2), 2^k\right)^{\top} \in \mathbb{R}^{h \times w \times \frac{h}{2^k} \times \frac{w}{2^k}},
$$

利用查询点附近的特征相似性lookup阶段构建motionfeature：

$$
M = \operatorname{MotionEncoder}\left( \operatorname{LookUp}\left({V_k}, \mu, r\right) \right) \in \mathbb{R}^{h \times w \times D_M},
$$

最后将上一次的隐状态$h$,motion feature $M$，上下文特征$C$输入ConvGRU得到最新的隐状态$h'$，再进行flowhead卷积得到光流变化量:

$$
\begin{aligned}
h' &= \operatorname{RNN}\left(h, M, C(I_1)\right), \\
\Delta \mu &= \operatorname{FlowHead}\left(h'\right).
\end{aligned}
$$

## 1.2 SEA-RAFT 的 Mixture-of-Laplace Loss
普通 RAFT 通常直接预测光流，并使用 L1 或 endpoint error 监督：
$$
L_{L1}=\lVert \mu_{gt}-\mu\rVert_1
$$
其中：
$\mu$ 是模型预测的光流；
$\mu_{gt}$ 是光流真值。
这种损失隐含地认为每个像素的光流真值都是确定、可预测的。但是在下列区域中，光流可能具有天然歧义：
遮挡区域；
图像边界外新出现或消失的区域；
重复纹理；
弱纹理区域；
反光和透明物体；
数据标注错误。
这些像素可能产生很大的误差，并在普通 L1 loss 中占据过高权重。SEA-RAFT 因此不仅预测光流，还预测每个像素的不确定性。
首先回顾一下 单个 Laplace 分布：
对一个标量 $x$，Laplace 分布的概率密度是：
$$
p(x\mid\mu,b)
\frac{1}{2b} \exp\left(-\frac{|x-\mu|}{b}\right), \qquad b>0
$$
其中：
$x$：真值；
$\mu$：网络预测的均值，在光流任务中就是预测光流；
$b$：Laplace 分布的尺度参数；
$b$ 越小，分布越窄，代表模型越有信心；
$b$ 越大，分布越宽，代表模型认为该像素越不确定。
这里的 $b$ 不是神经网络层的 bias，也不能简单理解为方差。Laplace 分布的方差为：
$$
\operatorname{Var}(x)=2b^2
$$
所以 $b$ 更准确的名称是 scale，即尺度参数。
再从概率密度得到负对数似然：
训练时希望真值在预测分布下具有尽可能高的概率，即最大化：
$$
p(x\mid\mu,b)
$$
等价地，可以最小化负对数似然：
$$
L_{Lap}=-\log p(x\mid\mu,b)
$$
代入 Laplace 密度：
$$
\begin{aligned}
L_{Lap} &=-\log\left[ \frac{1}{2b} \exp\left(-\frac{|x-\mu|}{b}\right) \right] \\
&=\log(2b)+\frac{|x-\mu|}{b}
\end{aligned}
$$
它包含两个相互制约的部分：
$$
\underbrace{\log(2b)}_{\text{惩罚过大的不确定性}} + \underbrace{\frac{|x-\mu|}{b}}_{\text{经过不确定性缩放的预测误差}}
$$
如果误差很大，模型可以增大 $b$，降低第二项；但第一项会随 $b$ 增大，因此模型不能无限增大不确定性来逃避误差。
例如令 $|x-\mu|=10$：
$b$
loss 近似值
1
$\log 2+10=10.69$
10
$\log 20+1=4.00$
100
$\log 200+0.1=5.40$

对于误差为 10 的模糊像素，$b=10$ 比 $b=1$ 更合理；但 $b=100$ 又会因为 $\log b$ 的惩罚变差。
对于原始拉普拉斯直接预测 b 的会存在数值问题：
Laplace 分布要求：
$ b>0 $ 在实现过程中通常 clamp $b$ to $[\epsilon, \infty)$, where $\epsilon$ is a small positive number.
但是这种方式存在几个问题。
(1)b 接近 0 时不稳定
当 $b\to0^+$：
$$
\log b\to-\infty
$$
同时：
$$
\frac{1}{b}\to+\infty
$$
损失和梯度可能出现极大的数值。
(2) clamp 会产生不光滑的边界
小于 $\epsilon$ 的预测全部变成同一个值：
raw_b = -10、-1、0  ->  b = epsilon

在被截断区域中，网络很难通过梯度知道应该如何修正原始预测。
(3) 正数约束增加优化难度
网络本身输出实数，但 $b$ 只能位于正数域，直接优化会受到参数边界的影响。
为了解决这一问题： SEA-RAFT 不直接预测 $b$，而是预测在 log-space 中回归 b：
$$
\boxed{\beta=\log b}
$$
因此：
$$
\boxed{b=e^\beta}
$$
$\beta$ 可以是任意实数，而 $e^\beta$ 天然满足：
$$
e^\beta>0
$$
将 $b=e^\beta$ 代入单 Laplace ：
即普通 Laplace 中的 $b_i$ 替换为 $e^{\beta_i}$，就恰好得到论文中的形式：
$$
Lap(x;\mu,\beta_i)
\frac{ \exp\left(-\frac{|x-\mu|}{e^{\beta_i}}\right) }{2e^{\beta_i}}
$$

对概率密度取负对数：

$$
\begin{aligned}
L_{Lap} &=\log(2e^\beta)+\frac{|x-\mu|}{e^\beta} \\
&=\log2+\beta+|x-\mu|e^{-\beta}
\end{aligned}
$$
因此我们得到了log空间新的的拉普拉斯概率密度函数。
因此论文中的 “regress b(u,v) directly in log-space” 更准确的含义是：
网络直接预测 log b，也就是 $\beta$，需要使用尺度时再通过 $b=e^\beta$ 得到正数尺度。它不是指把整个网络或所有变量都变换到对数空间。 有了log空间的拉普拉斯损失，我们考虑光流匹配的确定性与歧义性得到两个拉普拉斯混合的损失函数 Mixture-of-Laplace Loss 。单 Laplace 只有一组 $\mu,b$。如果模型为困难像素预测较大的 $b$，虽然能够表达不确定性，但会使训练目标偏离光流常用的 L1/EPE 评价目标。
SEA-RAFT 使用两个共享均值 $\mu$、但尺度不同的 Laplace 分量：
$$
p_{mix}(x)
\alpha p_1(x)+(1-\alpha)p_2(x)
$$
其中：
$p_1$ 负责普通、容易预测的像素；
$p_2$ 负责遮挡等歧义像素；
$\alpha\in[0,1]$ 是混合系数；
两个分量共享光流均值 $\mu$，但使用不同尺度。
论文固定：$\beta_1=0$。
在 Mixture-of-Laplace 公式中改用 $\beta_1,\beta_2$：
$$
MixLap(x;\alpha,\beta_1,\beta_2,\mu)
\alpha \frac{ e^{-\frac{|x-\mu|}{e^{\beta_1}}} }{2e^{\beta_1}} + (1-\alpha) \frac{ e^{-\frac{|x-\mu|}{e^{\beta_2}}} }{2e^{\beta_2}}
$$

进一步得到要优化的负log mix Laplace：
$$
\mathcal L_{MoL}
-\frac{1}{2HW} \sum_u\sum_v\sum_{d\in{x,y}} \log\left[ MixLap(\mu_{gt}(u,v)_d;\alpha,0,\beta_2,\mu(u,v)_d) \right]
$$
与原始 RAFT 的 sequence loss 类似，SEA-RAFT 不只监督最后一次预测，而是对每一轮光流预测的 MoL loss 加权求和：
$$
\mathcal L_{all}
\sum_{i=1}^{N} \gamma^{N-i}\mathcal L_{MoL}^{i}
$$
其中：
$N$ 表示参与监督的预测总数；
$\mathcal L_{MoL}^{i}$ 表示第 $i$ 次预测的 Mixture-of-Laplace 负对数似然；
$\gamma<1$ 是衰减因子；
越早的预测指数 $N-i$ 越大、权重越小；
最后一次预测的权重为 $\gamma^0=1$。
例如 $N=4,\gamma=0.85$ 时，各轮权重大约为：
$$
[\gamma^3,\gamma^2,\gamma,1] =[0.614,0.723,0.85,1]
$$
这种设计让中间预测也能得到梯度，同时更重视后期 refinement 的结果。
## 1.3 直接回归初始光流
在原始raft中 初始光流设置为0，这样显示不是很合理。在 SEA-RAFT中,借鉴flownet的思路，使用raft获得左右目上下文特征，直接预测光流值，作为sea raft的初值。这样很明显的效果是减少推理的迭代次数，加快速度。
## 1.4 大规模TartanAir 刚性流预训练
使用TartanAir 数据集进行预训练。
## 1.5算法架构的改进
使用imagenet预训练权重的resnet作为特征提取网络主干，并使用两个ConvNeXt替代convGRU。
# 2 实验
## 2.1 S、M、L 网络差异
SEA-RAFT 提供 S、M、L 三种配置，主要区别是 backbone 和推理迭代次数：

| 模型 | Backbone | 说明 |
| --- | --- | --- |
| SEA-RAFT(S) | 截断的 ImageNet 预训练 ResNet-18（前 6 层） | 速度最快、模型最小 |
| SEA-RAFT(M) | 截断的 ImageNet 预训练 ResNet-34（前 13 层） | 精度和速度折中 |
| SEA-RAFT(L) | 与 M 相同 | 与 M 共享权重，只在推理时增加迭代次数 |

S/M 使用相同的循环更新结构（两个 ConvNeXt block），L 并不是重新训练的更大网络，而是用 M 的权重进行更多轮 refinement。

## 2.2 训练阶段与细节
训练采用逐阶段加载上一阶段 checkpoint 的方式：

| 阶段 | 数据集 | Steps | Batch | Crop | 最大学习率 |
| --- | --- | ---: | ---: | ---: | ---: |
| 预训练 | TartanAir | 300k | 32 | 480×640 | 4e-4 |
| C | FlyingChairs | 100k | 16 | 368×496 | 2.5e-4 |
| T | FlyingThings3D | 120k | 32 | 432×960 | 4e-4 |
| S+T+K+H | Sintel、Things clean、KITTI、HD1K 混合 | 300k | 32 | 432×960 | 4e-4 |
| KITTI 微调 | KITTI | 10k | 16 | 432×960 | 1e-4 |
| Spring 微调 | Spring | 120k | 32 | 540×960 | 4e-4 |

其中，C/T/S/K/H 分别代表 Chairs、Things、Sintel、KITTI、HD1K。

优化器使用 AdamW 和 OneCycleLR，训练时只进行 4 轮迭代，并对初始光流及各轮 refinement 都施加 MoL sequence loss，后期预测权重更大。TartanAir 的刚体光流预训练主要用于提高场景真实性和跨数据集泛化能力。

## 2.3 实验结果
**Spring：** 微调后的 SEA-RAFT(M) 在测试集取得 1px=3.686、EPE=0.363、Fl=1.347、WAUC=94.534，其中 EPE 和 1px 相比此前最佳公开结果分别降低 22.9% 和 17.8%。

**Sintel/KITTI：** SEA-RAFT(L) 通过把推理迭代从 4 增加到 12，进一步改善 Sintel 和 KITTI 精度；SEA-RAFT 在 KITTI 上也表现出较强的跨数据集泛化能力。

**效率：** SEA-RAFT 只需 4 轮训练迭代，M/L 在 540×960 输入下约为 70.96/108.0 ms。Spring 测试采用先将 1080p 图像下采样 2 倍、推理后再上采样光流的方式，单张 RTX3090 可达到 20 fps 以上。

**消融实验：** MoL loss、直接预测初始光流和 TartanAir 预训练都能改善精度；固定一个 Laplace 分量的 $\beta_1=0$ 很关键。朴素单 Laplace、朴素混合 Laplace和 Gaussian mixture 均不如论文提出的 MoL 设计。

**RNN 结构消融：** 默认的两个 ConvNeXt block 取得 EPE=0.187、284.7G MACs；换回原始 RAFT 的 ConvGRU 后为 EPE=0.189、297.9G MACs；增加到四个 ConvNeXt block 后仍为 EPE=0.189，计算量增至 314.7G MACs。这说明两个 ConvNeXt block 已能取得更好的精度/计算量平衡，继续加深更新模块没有带来收益。

总体上，SEA-RAFT 的主要优势不是单纯扩大网络，而是通过更合理的初始光流、概率损失和预训练，使较少的迭代次数仍能取得较好的精度与速度平衡。
# 附录
A. mix Laplace不能把两个分量分别取 log 后再加权
正确形式是：
$$
-\log\left[\alpha p_1+(1-\alpha)p_2\right]
$$
下面这种写法不等价：
$$
-\alpha\log p_1-(1-\alpha)\log p_2
$$
前者表示真值由整个混合分布产生，是标准 mixture model 的负对数似然。
后者是两个分量各自 NLL 的加权平均，会同时要求两个分量解释样本，从而改变训练目标。
因此 SEA-RAFT 使用的是：
Mixture-of-Laplace 的负对数似然，而不是两个 log-Laplace loss 的简单加权和。
