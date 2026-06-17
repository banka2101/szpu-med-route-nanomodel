# ============================================================
# 文件 1: cls_model.py
# 作用：定义分类模型的结构（这是最核心的文件）
#
# 这个文件不能直接运行，它只是"定义"模型长什么样。
# 真正的训练在 cls_train.py 里，那里会 import 这个文件。
#
# 整体思路（从下往上搭积木）：
#   单个字 → 嵌入向量 → [注意力 → MLP] × N层 → 取一个向量 → 分类
#                         └──── 一个 Block ────┘
# ============================================================

import math
from dataclasses import dataclass   # 用来方便地存放"配置参数"

import torch
import torch.nn as nn
from torch.nn import functional as F


# ------------------------------------------------------------
# 配置类：把模型的所有"超参数"集中放在一起
# 用 @dataclass 可以让我们像填表格一样设置参数
# ------------------------------------------------------------
@dataclass
class GPTConfig:
    block_size: int = 32      # 一句话最多多少个字（超过会被截断）
    vocab_size: int = 3000    # 字表大小（总共有多少个不同的字），训练时会被真实值覆盖
    num_classes: int = 3      # 要分几类（骨科/内科/中医 就是 3）
    n_layer: int = 6          # 堆叠多少个 Transformer Block（层数）
    n_head: int = 6           # 注意力的"头"数（多头注意力，下面会解释）
    n_embd: int = 192         # 每个字用多少个数字来表示（向量维度），要能被 n_head 整除
    dropout: float = 0.1      # 随机丢弃比例，防止模型死记硬背（过拟合）


# ------------------------------------------------------------
# 模块 1：自注意力（Self-Attention）
# 这是 Transformer 最核心、最神奇的部分。
#
# 一句话理解它在干嘛：
#   让句子里的每个字，去"看"其它所有字，决定该关注谁。
#   比如"我的膝盖很疼"，"疼"这个字会重点关注"膝盖"，
#   这样模型就知道是"膝盖疼"，从而判断为骨科。
#
# 【重要】我们去掉了 nanoGPT 原版的"因果掩码"。
#   - 原版做"语言模型"（猜下一个字），所以规定每个字只能看前面的字，
#     不能偷看后面的（否则就作弊了）。这叫"因果掩码"。
#   - 我们做"分类"，要理解整句话的意思，所以每个字应该能看到
#     全句所有字（包括后面的）。因此不需要掩码。
# ------------------------------------------------------------
class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # n_embd 必须能被 n_head 整除，因为要把向量平均分给每个"头"
        assert config.n_embd % config.n_head == 0

        # 一个线性层，一次性算出 Q、K、V 三样东西（所以输出是 3 倍宽）
        #   Q (Query 查询)：我想找什么
        #   K (Key   钥匙)：我能提供什么
        #   V (Value 内容)：我实际的内容是什么
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # 注意力算完后，再过一个线性层做"输出整理"
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # 两个 dropout，用来防止过拟合
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        # x 的形状: (B, T, C)
        #   B = batch size，一次处理多少句话
        #   T = 句子长度，每句话多少个字
        #   C = n_embd，每个字用多少个数字表示
        B, T, C = x.size()

        # 第一步：算出 Q、K、V
        # c_attn 输出 (B, T, 3C)，用 split 切成三份，每份 (B, T, C)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # 第二步：把每个向量拆成多个"头"（多头注意力）
        # 为什么要多头？让模型从多个不同角度去理解关系。
        # 比如一个头看"症状-部位"关系，另一个头看"时间-程度"关系。
        #
        # 形状变化：(B, T, C) → (B, T, n_head, 每个头的维度) → (B, n_head, T, 每个头的维度)
        # transpose(1,2) 是把"头"这个维度挪到前面，方便后面每个头独立计算
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # 第三步：计算"注意力分数" —— 这是核心公式
        # q @ k 的转置：每个字的 Q 去和所有字的 K 做点积，
        #   点积越大，说明这两个字越"相关"、越该互相关注。
        # 除以 sqrt(维度)：是为了让数值不要太大，训练更稳定。
        # 结果 att 形状: (B, n_head, T, T)
        #   可以理解成一张"关注度表格"：第 i 个字对第 j 个字的关注程度。
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # 【这里原版有一行掩码，我们删掉了】
        # 原版: att = att.masked_fill(mask == 0, float('-inf'))
        # 删掉后，每个字可以关注句子里的所有字（包括后面的），适合分类。

        # 第四步：softmax 把"分数"变成"占比"（所有关注度加起来=1）
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # 第五步：用关注度去加权求和 V
        # 关注度高的字，它的内容 V 就占比大。
        # 这样每个字的新向量 = 它该关注的所有字的内容的加权混合。
        # y 形状: (B, n_head, T, 每个头的维度)
        y = att @ v

        # 第六步：把多个头的结果拼回去，变回 (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 第七步：过输出线性层 + dropout
        y = self.resid_dropout(self.c_proj(y))
        return y


# ------------------------------------------------------------
# 模块 2：MLP（多层感知机，也叫前馈网络 FeedForward）
# 作用：注意力负责"字与字之间交流信息"，
#       MLP 负责"对每个字单独做一次深加工/思考"。
# 结构很简单：放大 → 激活 → 缩回。
# ------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 先把维度放大到 4 倍（经验值，给模型更大的"思考空间"）
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # GELU 激活函数：引入非线性，让模型能学复杂的东西
        self.gelu = nn.GELU()
        # 再缩回原来的维度
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)      # 放大: (B,T,C) → (B,T,4C)
        x = self.gelu(x)      # 激活
        x = self.c_proj(x)    # 缩回: (B,T,4C) → (B,T,C)
        x = self.dropout(x)
        return x


# ------------------------------------------------------------
# 模块 3：Block（一个完整的 Transformer 层）
# 把"注意力"和"MLP"组装成一个标准积木块。
#
# 两个关键技巧：
#   1. 残差连接（x = x + ...）：让信息有"高速公路"直接通过，
#      防止层数太深导致信息丢失、梯度消失。
#   2. LayerNorm（归一化）：把数据调整到稳定范围，训练更顺。
#      这里用的是"Pre-Norm"（先归一化再进子模块），nanoGPT 也是这么做的。
# ------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)   # 注意力前的归一化
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)   # MLP 前的归一化
        self.mlp = MLP(config)

    def forward(self, x):
        # 先归一化 → 注意力 → 加回原值（残差）
        x = x + self.attn(self.ln_1(x))
        # 先归一化 → MLP → 加回原值（残差）
        x = x + self.mlp(self.ln_2(x))
        return x


# ------------------------------------------------------------
# 模块 4：GPTClassifier（最终的完整模型）
# 把上面所有积木拼成一个能分类的模型。
# ------------------------------------------------------------
class GPTClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # ===== 输入部分：把"字的编号"变成"向量" =====
        # wte = word token embedding：每个字 → 一个 n_embd 维向量
        #   就像一张查询表，第 5 号字 → 去表里取第 5 行那串数字
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # wpe = word position embedding：每个"位置"也给一个向量
        #   因为注意力本身不知道字的先后顺序，要额外告诉它"这是第几个字"
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        # ===== 主体部分：堆叠 N 个 Block =====
        # nn.ModuleList 就是一个装着多个 Block 的列表
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)   # 最后再归一化一次

        # ===== 输出部分：分类头 =====
        # 【这是和 nanoGPT 最大的区别】
        #   原版是 lm_head: n_embd → vocab_size（预测下一个字，几千个选项）
        #   我们是 classifier: n_embd → num_classes（只预测属于哪一类，3 个选项）
        self.classifier = nn.Linear(config.n_embd, config.num_classes)

        # 初始化所有参数（给模型一个合理的起点）
        self.apply(self._init_weights)

        # 打印模型有多少参数，让你心里有数
        n_params = sum(p.numel() for p in self.parameters())
        print("模型参数量: %.2f M" % (n_params / 1e6,))

    def _init_weights(self, module):
        # 用正态分布初始化线性层和嵌入层的权重，标准做法
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx 形状: (B, T)，里面装的是字的编号（整数）
        # targets 形状: (B,)，每句话的正确类别编号（训练时才传，预测时不传）
        device = idx.device
        B, T = idx.size()

        # 第一步：查表，把字编号变成向量
        # 位置编号: 0, 1, 2, ..., T-1
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        tok_emb = self.wte(idx)   # 字向量:   (B, T, C)
        pos_emb = self.wpe(pos)   # 位置向量: (T, C)，会自动广播加到每一句上
        x = self.drop(tok_emb + pos_emb)   # 字向量 + 位置向量 = 模型的输入

        # 第二步：穿过所有 Block（核心计算）
        for block in self.blocks:
            x = block(x)          # 形状始终保持 (B, T, C)
        x = self.ln_f(x)

        # 第三步：从整句话里取一个向量来代表"整句的含义"
        # 我们取最后一个位置 [:, -1, :] 的向量。
        # 为什么是最后一个？因为我们在 prepare 时做了"左侧补齐"，
        #   让真实文字靠在右边，所以最后一个位置一定是真实字符，
        #   而且它经过注意力已经"看遍"了全句，包含了整句信息。
        # 形状: (B, T, C) → (B, C)
        x = x[:, -1, :]

        # 第四步：分类头，输出每个类别的"得分"（logits）
        # 形状: (B, C) → (B, num_classes)
        logits = self.classifier(x)

        # 第五步：如果在训练（传了正确答案），就计算损失
        if targets is not None:
            # 交叉熵损失：衡量"预测的类别分布"和"正确答案"差多少
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None

        return logits, loss
