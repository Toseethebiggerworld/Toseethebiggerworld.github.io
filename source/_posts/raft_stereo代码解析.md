---
title: raft stereo代码解析
date: 2026-10-31 12:00:00
tags:
  - 学习笔记 立体匹配
categories:
  - 技术
---
# 1前置条件
raft-stereo是利用轻量级特征加上迭代方法替代3DCNN，因此网络结构更加轻量化同时又节约内存可以在更高分辨率图像直接输出视差图。\
raft-stereo全文都是围绕GRU迭代进行的，因此有必要搞清楚GRU是什么（参考理论篇）。\

GRU主要的目的之一就是如何有效的利用过去的信息，这一点恰好符合迭代的逻辑（类似卡尔曼滤波），用过去和现在的数据预测未来，可以说GRU完美符合利用迭代求解视差/光流的思想！！
既然raft使用迭代的GRU求解光流/视差，因此我们紧盯raft stereo的输入输出就能够把握整体的代码逻辑了。



# 2raft-stereo
想了解整体数据流，必然要从`raft_stereo.py`forward函数窥见整个流程，
## 2.1 raft-stereo forward
(1)首先是输入两张照片以及Feature Encoder和Context Encoder（这里对应原文的两个网络）
```py
    # image1: 左图，shape 通常为 [N, 3, H, W]
    # image2: 右图，shape 通常为 [N, 3, H, W]
    # 将输入从 [0, 255] 归一化到 [-1, 1]，这是编码器训练时使用的输入范围。
    image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
    image2 = (2 * (image2 / 255.0) - 1.0).contiguous()
    #  Context Encoder
    # cnet_list:仅由左图 image1 提取出的多尺度上下文结果，
    # 其中每个元素都包含 [用于初始化 GRU 的特征, 作为 GRU 输入的上下文特征]
    # cnet_list的结构如下：
    # len(cnet_list)=3,表示三层金字塔
    # cnet_list[i]=[hidden_init, context_feat]，其中 hidden_init 用于初始化第 i 层 GRU 的隐状态，context_feat 用作第 i 层 GRU 的输入。
    # len(cnet_list[i])=2
    # cnet_list[i][0]: 用于初始化第 i 层 GRU 的特征，形状为 [N, hidden_dims[i], H_i, W_i]，其中 H_i 和 W_i 是第 i 层特征图的空间尺寸。
    # cnet_list[i][1]: 作为第 i 层 GRU 输入的上下文特征，形状为 [N, context_dims[i], H_i, W_i
    cnet_list = self.cnet(image1, num_layers=self.args.n_gru_layers)
    # self.fnet：Feature Encoder
    # fmap1/fmap2: 分别是左图和右图的匹配特征图，
    fmap1, fmap2 = self.fnet([image1, image2])
    # cnet_list 中每个元素都形如 [hidden_init, context_feat]
    # net_list: 各尺度 GRU 的初始隐状态，使用 tanh 限制到 [-1, 1]，后续迭代中会被更新
    # inp_list: 各尺度上下文输入特征，使用 ReLU 保留非负激活，保持不变
    net_list = [torch.tanh(x[0]) for x in cnet_list] 
    inp_list = [torch.relu(x[1]) for x in cnet_list]

    # 每个尺度的上下文特征在这里预先切成 z/r/q 三份，这里切出来的是 cz / cr / cq 三个“上下文偏置项”，

    # 这句列表推导式的等价写法是：
    # new_inp_list = []
    # for i, conv in zip(inp_list, self.context_zqr_convs):
    #     zqr = conv(i)  
    #     cz, cr, cq = zqr.split(split_size=conv.out_channels // 3, dim=1)
    #     new_inp_list.append([cz, cr, cq])
    # inp_list = new_inp_list
    #
   
    # 此时 inp_list[i] = [cz, cr, cq]
    inp_list = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) for i,conv in zip(inp_list, self.context_zqr_convs)]

```
 (2)然后到了寻找近邻的特征相似度：
 ```py
         corr_block = CorrBlock1D
        # corr_fn 是一个“可调用相关体对象”：
        # 输入当前坐标 coords1，输出该坐标附近的多尺度一维相关特征。
        # corr_fn(coords1) 不是只取当前视差点的点积特征，
        # 而是围绕当前估计视差，在每个金字塔层上沿水平方向取 radius 范围内的一维邻域相关特征，
        # 然后把这些邻域特征拼起来，作为后续 GRU 更新的输入
        corr_fn = corr_block(fmap1, fmap2, radius=self.args.corr_radius, num_levels=self.args.corr_levels)

 ```
(3)循环迭代求解，这里就要关注GRU（update_block）输入输出了
```py
 for itr in range(iters):
        corr = corr_fn(coords1) # corr: 当前坐标附近采样得到的相关特征/匹配代价体
        flow = coords1 - coords0  # flow: 当前迭代下的位移估计（双目中主要对应视差）
    # 最细尺度 GRU 融合相关特征与当前 flow，输出本轮增量 delta_flow
                # up_mask: 学习式上采样权重
                # delta_flow: 本轮预测的位移残差增量
        net_list, up_mask, delta_flow = self.update_block(net_list, inp_list, corr, flow, iter32=self.args.n_gru_layers==3, iter16=self.args.n_gru_layers>=2)
        # in stereo mode, project flow onto epipolar
        # 标准校正后的双目图像只需预测水平位移，因此强制将垂直分量置零。
        delta_flow[:,1] = 0.0

        # F(t+1) = F(t) + \Delta(t)
        # RAFT 的核心就是反复预测残差增量，而不是一次性回归最终结果。
        coords1 = coords1 + delta_flow
            
        # flow_up: 上采样到原图尺度后的预测结果
        flow_up = self.upsample_flow(coords1 - coords0, up_mask)
        # 双目最终只保留 x 方向位移，shape 变为 [N, 1, H, W]
        flow_up = flow_up[:,:1]
        flow_predictions.append(flow_up)

```
看完上面感觉核心forward也就不超过20行，是不是很简单。可以发现raft-stereo forward的核心就是迭代中的self.update_block。这个函数的**输入**是：
```
net_list, inp_list, corr, flow
```
net_list, inp_list：原始图像经过context network得到cnetlist。net_list=cnetlist[i][0],inp_list=cnetlist[i][1],inp_list[i] = [cz, cr, cq]。
corr：为通过feature encoder得到fmap1, fmap2，然后通过corr_block计算相似度，然后corr_block寻找特征金字塔得到特征集合
flow：初始或者估计的光流
**输出**为：
```
net_list, up_mask, delta_flow
```
net_list：输出的某些特征，因此netlist是一直更新项，inplist不更新
up_mask,delta_flow：分别是上采样权重以及光流/视差变化量。
继续探究self.update_block为何物，我们发现其在一个class BasicMultiUpdateBlock(nn.Module)，打开其forward可以发现：
```py
    def forward(self, net, inp, corr=None, flow=None, iter08=True, iter16=True, iter32=True, update=True):
        # net: 各尺度 GRU 的隐藏状态列表 [net32, net16, net08]
        # inp: 各尺度上下文特征，已经在 raft_stereo.py 里拆成 [cz, cr, cq]
        # corr: 当前坐标处采样得到的相关体特征
        # flow: 当前迭代的位移估计
        # iter32/iter16/iter08: 控制是否更新对应尺度的 GRU


        if iter32:
            # gru32 对应论文里更粗尺度的隐藏状态更新
            net[2] = self.gru32(net[2], *(inp[2]), pool2x(net[1]))
        if iter16:
            if self.args.n_gru_layers > 2:
                # gru16 融合：本层上下文 + 更粗一层的下采样状态 + 更细一层的上采样状态
                net[1] = self.gru16(net[1], *(inp[1]), pool2x(net[0]), interp(net[2], net[1]))
        if iter08:
            # motion_features = [相关体特征, flow 特征]
            motion_features = self.encoder(flow, corr)
            if self.args.n_gru_layers > 1:
                # gru08 是最终输出层，结合 motion_features 和更粗尺度插值后的隐藏状态
                net[0] = self.gru08(net[0], *(inp[0]), motion_features, interp(net[1], net[0]))

        # delta_flow: 由最细层 hidden state 回归的本轮位移增量，对应论文中的残差更新
        delta_flow = self.flow_head(net[0])

        # scale mask to balence gradients
        # mask: 学习式上采样权重，用于把低分辨率 flow 还原到高分辨率
        mask = .25 * self.mask(net[0])
        return net, mask, delta_flow
```
我们直接关注最核心的三个函数gru32,gru16,gru08：
```py
        self.gru08 = ConvGRU(hidden_dims[2], encoder_output_dim + hidden_dims[1] * (args.n_gru_layers > 1))
        self.gru16 = ConvGRU(hidden_dims[1], hidden_dims[0] * (args.n_gru_layers == 3) + hidden_dims[2])
        self.gru32 = ConvGRU(hidden_dims[0], hidden_dims[1])

class ConvGRU(nn.Module):                    
    def forward(self, h, cz, cr, cq, *x_list):
        # h: 论文中的 h_{t-1}，上一轮隐藏状态
        # x_list: 当前时刻的输入特征，通常由 context / motion / correlation 等特征组成
        # cz, cr, cq: 由上下文分支提前算好的偏置/上下文项，
        #             分别对应 z 门、r 门、候选状态 h~_t 的上下文贡献；
        #             它们不是 z_t / r_t / h~_t 本身，而是参与这些量的计算
        x = torch.cat(x_list, dim=1)
        # hx 对应论文中的 [h_{t-1}, x_t]
        hx = torch.cat([h, x], dim=1)

        #1更新门
        # z_t = sigma(Conv3x3([h_{t-1}, x_t], Wz))
        # 代码里把上下文项 cz 预先算好并加到卷积结果上，
        # 所以 z_t = sigmoid(convz([h_{t-1}, x_t]) + cz)
        z = torch.sigmoid(self.convz(hx) + cz)
        # r_t = sigma(Conv3x3([h_{t-1}, x_t], Wr))
        #2重置门
        r = torch.sigmoid(self.convr(hx) + cr)
        # 3候选隐藏状态
        # \tilde{h}_t = tanh(Conv3x3([r_t ⊙ h_{t-1}, x_t], Wh))
        # 代码里的 q 就是候选隐藏状态 h~_t；
        # 先用 r_t 过滤旧状态，再与输入 x_t 拼接后卷积，最后加上 cq
  
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)) + cq)
        # 4隐藏状态
        # h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ \tilde{h}_t
        h = (1-z) * h + z * q
        return h
```
这里我们终于找到核心的代码了！raft_stereo forward中对应的update_block底层就是GRU，在这里让我们重点关注gru08：
```py
if iter08:
    # motion_features = [相关体特征, flow 特征]
    motion_features = self.encoder(flow, corr)
    if self.args.n_gru_layers > 1:
        # gru08 是最终输出层，结合 motion_features 和更粗尺度插值后的隐藏状态
        net[0] = self.gru08(net[0], *(inp[0]), motion_features, interp(net[1], net[0]))
```
对比：
```
GRU:
forward(self, h, cz, cr, cq, *x_list)
```
GRU实参与形参的对应关系可以得到：
```
h=net[0]: h_{t-1}，也就是上一时刻的隐藏状态
cz, cr, cq=*(inp[0])为gru公式的偏执项
*x_list=[motion_features, interp(net[1], net[0])]为gru的X，这里插值的目的是获取上层金字塔的隐藏状态
```
让我们回顾raft_stereo forward->update_block->gru08->GRU forward函数：

从`raft_stereo.forward`到`GRU.forward`的变量对应关系可以整理为：

| `raft_stereo.forward` 中的变量 | `update_block` 中的变量 | `gru08` 调用中的位置 | `GRU.forward` 中的变量 | 含义 |
|---|---|---|---|---|
| `net_list` | `net` | `net[0]` | `h` | 最细层 GRU 上一时刻的隐藏状态 `h_{t-1}` |
| `inp_list` | `inp` | `inp[0]` | `cz, cr, cq` | GRU 的上下文偏置项 |
| `corr` | `corr` | `motion_features=self.encoder(flow, corr)` | `x_list[0]` 的一部分来源 | 当前像素坐标对应的相关特征 |
| `flow = coords1 - coords0` | `flow` | `motion_features=self.encoder(flow, corr)` | `x_list[0]` 的一部分来源 | 当前迭代的位移/视差估计 |
| `/` | `interp(net[1], net[0])` | `self.gru08(..., interp(net[1], net[0]))` | `x_list[1]` | 上一层金字塔隐藏状态插值到当前层后的特征 |

其中，`x_list`GRU公式里的`x_t`。

# 3各模块代码解析


## 3.1相关性特征提取模块

来看这行代码：
```
                   self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', downsample=args.n_downsample)


```
在```extractor.py``` 找到 ```BasicEncoder```:
```
    def forward(self, x, dual_inp=False):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = x.split(split_size=batch_dim, dim=0)

        return x
```
这里就是一个简单的卷积网络，```self.layer```是自定义的残差层。
## 3.2上下文特征提取
来看这行代码：
```
        self.cnet = MultiBasicEncoder(output_dim=[args.hidden_dims, context_dims], norm_fn=args.context_norm, downsample=args.n_downsample)
查看一下forward：
```
 def forward(self, x, dual_inp=False):

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = x.split(split_size=batch_dim, dim=0)

        return x
```
就是一个很简单的卷积网络，```self.layer```由自定义的残差层组成。
```
在```extractor.py``` 找到 ```MultiBasicEncoder```:
init定义不同层次的多尺度特征提取器：
```py
        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, dim[2], 3, padding=1))
            output_list.append(conv_out)

        self.outputs08 = nn.ModuleList(output_list)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, dim[1], 3, padding=1))
            output_list.append(conv_out)

        self.outputs16 = nn.ModuleList(output_list)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Conv2d(128, dim[0], 3, padding=1)
            output_list.append(conv_out)

        self.outputs32 = nn.ModuleList(output_list)
```
forward如下，根据 num_layers 决定是否继续提取 1/16、1/32 特征：
```py
    def forward(self, x, dual_inp=False, num_layers=3):

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if dual_inp:
            v = x
            x = x[:(x.shape[0]//2)]

        outputs08 = [f(x) for f in self.outputs08]
        if num_layers == 1:
            return (outputs08, v) if dual_inp else (outputs08,)

        y = self.layer4(x)
        outputs16 = [f(y) for f in self.outputs16]

        if num_layers == 2:
            return (outputs08, outputs16, v) if dual_inp else (outputs08, outputs16)
  
        z = self.layer5(y)
        outputs32 = [f(z) for f in self.outputs32]

        return (outputs08, outputs16, outputs32, v) if dual_inp else (outputs08, outputs16, outputs32)
```

## 3.3特征查找模块
我们以标准实现为例，在```raft_stereo.py```找到：
```py
    corr_block = CorrBlock1D
    # corr_fn 是一个“可调用相关体对象”：
    # 输入当前坐标 coords1，输出该坐标附近的多尺度一维相关特征。
    # corr_fn(coords1) 不是只取当前视差点的点积特征，
    # 而是围绕当前估计视差，在每个金字塔层上沿水平方向取 radius 范围内的一维邻域相关特征，
    # 然后把这些邻域特征拼起来，作为后续 GRU 更新的输入

    corr_fn = corr_block(fmap1, fmap2, radius=self.args.corr_radius, num_levels=self.args.corr_levels)
    # corr: 当前坐标附近采样得到的相关特征/匹配代价体
    corr = corr_fn(coords1) 
```
在```corr.py```找到:
```py
def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        """初始化相关性金字塔。

        参数：
            fmap1: 左图特征图，shape 为 [B, D, H, W1]。
            fmap2: 右图特征图，shape 为 [B, D, H, W2]。
            num_levels: 查询时使用的金字塔层数。
            radius: 每层局部相关性窗口的水平搜索半径。

        处理流程：
            1. 计算 fmap1 与 fmap2 在同一行上的全配对水平相关性。
            2. 将每个像素位置的相关性向量展平成独立样本。
            3. 沿宽度方向平均池化，构建多尺度相关性金字塔。
        """
        # num_levels 控制查询多少个金字塔尺度，radius 控制每个尺度的搜索半径。
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # 计算 fmap1 每个位置与 fmap2 同一行所有水平位置之间的全局相关性。
        # corr shape: [B, H, W1, 1, W2]
        corr = CorrBlock1D.corr(fmap1, fmap2)

        batch, h1, w1, _, w2 = corr.shape
        # 将每个 batch/行/列位置展平成独立样本，方便后续用 bilinear_sampler 查询。
        corr = corr.reshape(batch*h1*w1, 1, 1, w2)

        # 这里通过池化建立特征金字塔
        self.corr_pyramid.append(corr)
        for i in range(self.num_levels):
            corr = F.avg_pool2d(corr, [1,2], stride=[1,2])
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        """根据当前坐标从相关性金字塔中采样局部相关性特征。

        参数：
            coords: 当前匹配坐标，shape 通常为 [B, 2, H, W]，其中第 0 维通道是 x 坐标。

        返回：
            多尺度局部相关性特征，shape 为 [B, num_levels * (2 * radius + 1), H, W]。

        处理流程：
            1. 取水平 x 坐标作为中心点。
            2. 在每个金字塔尺度上围绕中心点生成 [-radius, radius] 采样窗口。
            3. 对对应尺度的相关性体做双线性采样，并拼接所有尺度结果。
        """
        # coords 通常包含 (x, y)，立体匹配只需要沿水平 x 方向采样相关性。
        r = self.radius
        coords = coords[:, :1].permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            # 在当前预测坐标附近生成 [-r, r] 的水平偏移窗口。
            dx = torch.linspace(-r, r, 2*r+1)
            dx = dx.view(2*r+1, 1).to(coords.device)
            # 第 i 层相关性宽度缩小了 2**i，因此坐标也要缩放到该尺度。
            x0 = dx + coords.reshape(batch*h1*w1, 1, 1, 1) / 2**i
            y0 = torch.zeros_like(x0)

            # 在相关性体上双线性采样局部窗口，并恢复为 [B, H, W, C]。
            coords_lvl = torch.cat([x0,y0], dim=-1)
            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        # 拼接所有尺度的局部相关性特征，输出为网络常用的 [B, C, H, W]。
        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()
```
## 3.4GRU循环迭代模块
```raft_stereo.py```找到代码：
```py
 with autocast(enabled=self.args.mixed_precision):
                if self.args.n_gru_layers == 3 and self.args.slow_fast_gru: # Update low-res GRU
                    # 先更新最粗尺度（1/32）GRU，让大范围上下文先传播
                    net_list = self.update_block(net_list, inp_list, iter32=True, iter16=False, iter08=False, update=False)
                if self.args.n_gru_layers >= 2 and self.args.slow_fast_gru:# Update low-res GRU and mid-res GRU
                    # 再更新中间尺度（1/16）GRU，最后再在 1/8 尺度输出真实增量
                    net_list = self.update_block(net_list, inp_list, iter32=self.args.n_gru_layers==3, iter16=True, iter08=False, update=False)
                # 最细尺度 GRU 融合相关特征与当前 flow，输出本轮增量 delta_flow
                # up_mask: 学习式上采样权重
                # delta_flow: 本轮预测的位移残差增量
                net_list, up_mask, delta_flow = self.update_block(net_list, inp_list, corr, flow, iter32=self.args.n_gru_layers==3, iter16=self.args.n_gru_layers>=2)
```
继续找到```self.update_block = BasicMultiUpdateBlock```。注意这里的不同尺度的GRU更新次数是不同的，这也是作者论文说的多尺度的GRU。
在```update.py```找到对应的代码：
```py
class BasicMultiUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dims=[]):
        super().__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        encoder_output_dim = 128

        # 三层 GRU 对应三个尺度的 hidden state：
        # gru32 -> 最粗尺度
        # gru16 -> 中间尺度
        # gru08 -> 最细尺度，负责最终 delta_flow 预测
        self.gru08 = ConvGRU(hidden_dims[2], encoder_output_dim + hidden_dims[1] * (args.n_gru_layers > 1))
        self.gru16 = ConvGRU(hidden_dims[1], hidden_dims[0] * (args.n_gru_layers == 3) + hidden_dims[2])
        self.gru32 = ConvGRU(hidden_dims[0], hidden_dims[1])
        self.flow_head = FlowHead(hidden_dims[2], hidden_dim=256, output_dim=2)
        factor = 2**self.args.n_downsample

        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dims[2], 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, (factor**2)*9, 1, padding=0))

    def forward(self, net, inp, corr=None, flow=None, iter08=True, iter16=True, iter32=True, update=True):
        # net: 各尺度 GRU 的隐藏状态列表 [net32, net16, net08]
        # inp: 各尺度上下文特征，已经在 raft_stereo.py 里拆成 [cz, cr, cq]
        # corr: 当前坐标处采样得到的相关体特征
        # flow: 当前迭代的位移估计
        # iter32/iter16/iter08: 控制是否更新对应尺度的 GRU
        # update=False 时只更新隐藏状态，不输出 delta_flow / mask

        if iter32:
            # gru32 对应论文里更粗尺度的隐藏状态更新
            net[2] = self.gru32(net[2], *(inp[2]), pool2x(net[1]))
        if iter16:
            if self.args.n_gru_layers > 2:
                # gru16 融合：本层上下文 + 更粗一层的下采样状态 + 更细一层的上采样状态
                net[1] = self.gru16(net[1], *(inp[1]), pool2x(net[0]), interp(net[2], net[1]))
            else:
                net[1] = self.gru16(net[1], *(inp[1]), pool2x(net[0]))
        if iter08:
            # motion_features = [相关体特征, flow 特征]
            motion_features = self.encoder(flow, corr)
            if self.args.n_gru_layers > 1:
                # gru08 是最终输出层，结合 motion_features 和更粗尺度插值后的隐藏状态
                net[0] = self.gru08(net[0], *(inp[0]), motion_features, interp(net[1], net[0]))
            else:
                net[0] = self.gru08(net[0], *(inp[0]), motion_features)

        if not update:
            return net

        # delta_flow: 由最细层 hidden state 回归的本轮位移增量，对应论文中的残差更新
        delta_flow = self.flow_head(net[0])

        # scale mask to balence gradients
        # mask: 学习式上采样权重，用于把低分辨率 flow 还原到高分辨率
        mask = .25 * self.mask(net[0])
        return net, mask, delta_flow
```
这里需要关注三个地方：GRU08是怎么更新的，通过GRU输出的结果是怎么得到视差的，最后就是怎么通过上采样得到高分辨率的视差图的。

### 3.4.1 GRU08 是怎么更新的

先看最细尺度 GRU 的调用位置：

```py
motion_features = self.encoder(flow, corr)
if self.args.n_gru_layers > 1:
    net[0] = self.gru08(net[0], *(inp[0]), motion_features, interp(net[1], net[0]))
else:
    net[0] = self.gru08(net[0], *(inp[0]), motion_features)
```

这里的 `net[0]` 就是 1/8 分辨率上的隐藏状态，也就是最细尺度的 hidden state。每一轮迭代时，代码先用当前估计的 `coords1` 去相关性金字塔里采样，得到 `corr`；再用当前位移 `flow = coords1 - coords0` 和 `corr` 送入 `BasicMotionEncoder`，得到 `motion_features`。

`motion_features` 可以理解为“当前匹配状态”的编码结果，它同时包含两类信息：

1. 当前坐标附近的相关性特征，也就是这个位置往左/往右移动时匹配分数如何变化；
2. 当前已经估计出来的位移 `flow`，也就是上一轮迭代得到的视差状态。

如果网络启用了多层 GRU，那么 `gru08` 还会额外接收 `interp(net[1], net[0])`。这个操作把 1/16 尺度的隐藏状态插值到 1/8 尺度，使最细层更新时不仅看局部匹配信息，也能利用更粗尺度传下来的上下文信息。也就是说，`gru08` 的输入大致可以理解为：

```text
上一轮 1/8 hidden state--对应的是GRU第一个输入：上一时刻的隐状态$h_{t-1}$
+ 1/8 context 特征拆出来的 cz/cr/cq---固定偏置
+ motion_features(flow, corr)
+ 可选的 1/16 hidden state 上采样结果--这两个构成了GRU的第二个输入$x_t$
```

进入 `ConvGRU` 后，更新过程就是标准的卷积 GRU 形式：

```text
z = sigmoid(convz([h, x]) + cz)
r = sigmoid(convr([h, x]) + cr)
q = tanh(convq([r * h, x]) + cq)
h_new = (1 - z) * h + z * q
```

其中 `h` 是旧的 `net[0]`，`x` 是拼接后的输入特征。`z` 控制本轮要写入多少新信息，`r` 控制旧状态参与候选状态计算的比例，`q` 是候选隐藏状态。最后得到的新 `h_new` 会覆盖 `net[0]`，成为这一轮更新后的最细尺度 GRU 状态。

所以，`GRU08` 本身并不直接输出视差图，它的作用是把“当前相关性 + 当前位移 + 上下文信息 + 上一轮隐藏状态”融合成新的隐藏状态 `net[0]`。真正的位移增量是在后面的 `flow_head(net[0])` 中回归出来的。

### 3.4.2 GRU 输出后是怎么得到视差的

`BasicMultiUpdateBlock` 在更新完 `net[0]` 后，会执行：

```py
delta_flow = self.flow_head(net[0])
mask = .25 * self.mask(net[0])
return net, mask, delta_flow
```

这里的关键是 `flow_head`。它接收更新后的 1/8 尺度隐藏状态 `net[0]`，输出 `delta_flow`。`delta_flow` 的通道数是 2，对应 x/y 两个方向的位移残差。注意它不是最终视差，而是“本轮要加到当前坐标上的增量”。

回到 `raft_stereo.py` 的迭代循环中：

```py
flow = coords1 - coords0
net_list, up_mask, delta_flow = self.update_block(...)
delta_flow[:,1] = 0.0
coords1 = coords1 + delta_flow
```

这里的 `coords0` 是固定参考坐标，表示左图特征图上每个像素原始位置；`coords1` 是不断被更新的匹配坐标，表示当前认为右图中对应点应该在哪里。因此：

```text
当前低分辨率位移 / 视差 = coords1 - coords0
```

每一轮网络不是直接预测完整视差，而是预测一个残差 `delta_flow`，再累加到 `coords1` 上：

```text
coords1(t+1) = coords1(t) + delta_flow(t)
flow(t+1) = coords1(t+1) - coords0
```

由于这是双目立体匹配，校正后的左右图满足极线约束，匹配主要只发生在水平方向。因此代码强制：

```py
delta_flow[:,1] = 0.0
```

也就是把 y 方向增量清零，只保留 x 方向位移。后面上采样完成后又执行：

```py
flow_up = flow_up[:,:1]
```

因此最终输出的单通道结果就是 x 方向位移，也就是代码里使用的视差预测。总结一下，从 GRU 到视差的链路是：

```text
net[0] --flow_head--> delta_flow --累加更新 coords1--> coords1 - coords0 --取 x 通道--> disparity
```

### 3.4.3 怎么通过上采样得到高分辨率视差图

GRU 和相关性体都工作在低分辨率特征图上，通常是原图的 1/8 尺度。因此 `coords1 - coords0` 得到的也是低分辨率位移。为了得到原图尺度的视差，需要上采样：

```py
if up_mask is None:
    flow_up = upflow8(coords1 - coords0)
else:
    flow_up = self.upsample_flow(coords1 - coords0, up_mask)
flow_up = flow_up[:,:1]
```

如果没有 `up_mask`，代码会退化成普通的 8 倍上采样；但正常情况下会使用 `self.upsample_flow` 做学习式上采样。`up_mask` 由 `self.mask(net[0])` 预测得到，说明上采样权重也是根据最终的 1/8 隐藏状态自适应生成的。

`upsample_flow` 的核心代码如下：

```py
factor = 2 ** self.args.n_downsample
mask = mask.view(N, 1, 9, factor, factor, H, W)
mask = torch.softmax(mask, dim=2)

up_flow = F.unfold(factor * flow, [3,3], padding=1)
up_flow = up_flow.view(N, D, 9, 1, 1, H, W)

up_flow = torch.sum(mask * up_flow, dim=2)
up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
return up_flow.reshape(N, D, factor*H, factor*W)
```

这里的思路不是简单插值，而是“用低分辨率每个点周围 3×3 邻域的 flow，加权组合出高分辨率的子像素结果”。具体过程是：

1. `mask.view(N, 1, 9, factor, factor, H, W)`：把 mask 整理成上采样权重，其中 `9` 表示 3×3 邻域的 9 个低分辨率 flow 候选，`factor × factor` 表示每个低分辨率点要展开成多少个高分辨率子像素；
2. `softmax(mask, dim=2)`：让 3×3 邻域上的 9 个权重归一化，形成凸组合；
3. `F.unfold(factor * flow, [3,3], padding=1)`：取出每个低分辨率位置周围 3×3 的 flow。这里先乘以 `factor`，是因为低分辨率 flow 的一个像素位移对应原图上的 `factor` 个像素位移；
4. `torch.sum(mask * up_flow, dim=2)`：用预测出来的 mask 对 9 个邻域 flow 做加权求和，得到每个高分辨率子像素的 flow；
5. `permute + reshape`：把 `[factor, factor, H, W]` 的子像素布局重新排列成 `[factor*H, factor*W]` 的高分辨率图。

所以高分辨率视差不是直接由低分辨率视差双线性插值得到的，而是由网络预测的 `up_mask` 控制每个位置如何融合周围 3×3 的低分辨率位移。最后只保留 x 方向通道，就得到原图尺度的单通道视差图。

## 3.5训练模块
训练模块主要有三个值得关注的地方：loss 计算，混合精度训练，优化器与学习率调度器的选择。

### 3.5.1 loss 是怎么计算的

训练入口在 `train_stereo.py` 中，前向传播会得到一个预测序列：

```py
flow_predictions = model(image1, image2, iters=args.train_iters)
loss, metrics = sequence_loss(flow_predictions, flow, valid)
```

这里的 `flow_predictions` 不是单个结果，而是 RAFT-Stereo 每一轮 GRU 迭代后上采样得到的视差预测列表。也就是说，如果 `train_iters=16`，模型会输出 16 个中间预测结果。训练时不会只监督最后一次预测，而是对整个预测序列都计算 loss，这也是 RAFT 系列常用的 sequence loss。

对应代码如下：

```py
def sequence_loss(flow_preds, flow_gt, valid, loss_gamma=0.9, max_flow=700):
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = ((valid >= 0.5) & (mag < max_flow)).unsqueeze(1)

    for i in range(n_predictions):
        adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
        i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * i_loss[valid.bool()].mean()
```

这里主要分成三步理解：

1. 先用 `valid` 和 `max_flow` 过滤无效像素。`valid >= 0.5` 表示数据集标注中认为有效的像素；`mag < max_flow` 会过滤掉过大的位移，避免异常视差对训练造成影响；
2. 对每一轮预测都计算 L1 loss，也就是 `(flow_preds[i] - flow_gt).abs()`；
3. 越靠后的预测权重越大，越早的预测权重越小。因为后面的迭代结果更接近最终输出，应该承担更强监督。

权重计算是：

```py
adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
```

`loss_gamma` 默认是 `0.9`。当 `i` 越靠前时，`n_predictions - i - 1` 越大，权重越小；当 `i` 是最后一次预测时，指数为 0，权重就是 1。因此最后一次预测权重最大。代码里还对 `loss_gamma` 做了一次 `15/(n_predictions - 1)` 的调整，目的是让不同迭代次数下的衰减强度尽量保持一致。

loss 之外，函数还会基于最后一次预测计算评估指标：

```py
epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
metrics = {
    'epe': epe.mean().item(),
    '1px': (epe < 1).float().mean().item(),
    '3px': (epe < 3).float().mean().item(),
    '5px': (epe < 5).float().mean().item(),
}
```

`epe` 是 End-Point Error，也就是预测视差和真实视差之间的欧氏误差。因为立体匹配最终只保留一个 x 方向通道，所以这里本质上就是视差绝对误差。`1px / 3px / 5px` 分别表示误差小于 1、3、5 个像素的比例，用来观察预测精度。

### 3.5.2 混合精度训练是怎么做的

训练脚本中先导入 `GradScaler`：

```py
try:
    from torch.cuda.amp import GradScaler
except:
    class GradScaler:
        ...
```

如果 PyTorch 版本支持 AMP，就使用官方的 `torch.cuda.amp.GradScaler`；如果不支持，就定义一个空的兼容类，让代码仍然可以跑，只是不会真正进行混合精度训练。

训练时创建 scaler：

```py
scaler = GradScaler(enabled=args.mixed_precision)
```

是否启用混合精度由命令行参数控制：

```py
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
```

需要注意的是，`train_stereo.py` 里主要负责梯度缩放，而真正的半精度前向计算是在 `core/raft_stereo.py` 中通过 `autocast` 包起来的：

```py
with autocast(enabled=self.args.mixed_precision):
    ...
```

也就是说，混合精度训练由两部分组成：

1. `autocast`：让部分前向计算自动使用 FP16，从而减少显存占用、提高计算速度；
2. `GradScaler`：对 loss 做缩放，避免 FP16 反向传播时梯度太小而下溢。

反向传播和参数更新部分如下：

```py
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

scaler.step(optimizer)
scheduler.step()
scaler.update()
```

这段顺序很重要：

1. `scaler.scale(loss).backward()`：先把 loss 放大后再反向传播，降低 FP16 梯度下溢风险；
2. `scaler.unscale_(optimizer)`：在梯度裁剪之前，把梯度恢复到真实尺度；
3. `clip_grad_norm_(..., 1.0)`：做梯度裁剪，防止 GRU 迭代训练中梯度过大；
4. `scaler.step(optimizer)`：如果梯度没有溢出，就执行优化器更新；如果检测到溢出，会跳过这次更新；
5. `scaler.update()`：根据本轮是否溢出，自动调整下一轮的缩放系数。

因此，开启 `--mixed_precision` 后，模型前向会尽量使用半精度，反向传播通过 `GradScaler` 保证数值稳定；不开启时，这些接口基本等价于普通 FP32 训练流程。

### 3.5.3 优化器与学习率调度器的选择

优化器和学习率调度器由 `fetch_optimizer` 创建：

```py
def fetch_optimizer(args, model):
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
            pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler
```

优化器使用的是 `AdamW`，而不是普通 SGD。原因是 RAFT-Stereo 这种网络包含特征提取器、相关性查询、ConvGRU 迭代更新和上采样 mask 等多个模块，参数类型比较复杂，AdamW 的自适应学习率通常更容易稳定训练。`AdamW` 还把 weight decay 和梯度更新解耦，相比传统 Adam 里直接把 L2 正则混进梯度，权重衰减行为更清晰。

主要参数来自命令行：

```py
parser.add_argument('--lr', type=float, default=0.0002, help="max learning rate.")
parser.add_argument('--num_steps', type=int, default=100000, help="length of training schedule.")
parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")
```

默认配置下：

```text
最大学习率 lr = 0.0002
训练步数 num_steps = 100000
权重衰减 wdecay = 0.00001
```

学习率调度器使用 `OneCycleLR`：

```py
OneCycleLR(
    optimizer,
    args.lr,
    args.num_steps + 100,
    pct_start=0.01,
    cycle_momentum=False,
    anneal_strategy='linear'
)
```

它的含义是：整个训练过程中学习率按照 One-Cycle 策略变化，先在很短的 warmup 阶段升到最大值 `args.lr`，然后再线性下降。这里 `pct_start=0.01` 表示前 1% 的训练步数用于升学习率，`anneal_strategy='linear'` 表示后续采用线性退火。`cycle_momentum=False` 表示不对动量做反向周期调整，因为这里使用的是 AdamW，不是典型 SGD momentum 训练。

训练循环中每完成一次参数更新，就执行一次调度器更新：

```py
scaler.step(optimizer)
scheduler.step()
scaler.update()
```

所以学习率是按 iteration 更新的，而不是按 epoch 更新的。`args.num_steps + 100` 比实际训练步数略多一点，可以避免最后阶段因为 step 数边界导致调度器提前结束。整体上，这套配置可以总结为：

```text
AdamW 负责稳定优化复杂网络参数
OneCycleLR 负责前期快速升学习率、后期线性衰减
GradScaler + gradient clipping 负责在混合精度和迭代更新网络中保持数值稳定
```

