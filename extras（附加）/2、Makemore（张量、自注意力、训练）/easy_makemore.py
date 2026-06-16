# ================================================================================
#  makemore 简化版（拼音注释版）—— 第 1/3 部分
# ================================================================================
#
# 【这个脚本是干什么的】
# 给它一个文本文件（每行一个词，比如几万个英文人名），
# 它会学习这些词的"拼写规律"，然后生成长得很像、但从没见过的新词。
#
# 【核心思想：下一个字符预测】
# 比如训练数据里有 "emma"，模型学的是：
#   看到 <开始>     → 下一个字符大概率是 e
#   看到 <开始>e    → 下一个字符大概率是 m
#   看到 <开始>em   → 下一个字符大概率是 m
#   看到 <开始>emm  → 下一个字符大概率是 a
#   看到 <开始>emma → 下一个字符大概率是 <结束>
# 学会了这个，就能一个字符一个字符地"接龙"生成新词。
# ChatGPT 的原理和这个完全一样，只是它接龙的不是字符而是词块（token）。
#
# 【文件整体结构】
#   第1部分（本段）：配置 + Transformer（GPT-2 同款结构，最重要）
#   第2部分：其他5种模型（Bigram/MLP/RNN/GRU/BoW，从简单到复杂的进化史）
#   第3部分：数据集处理 + 训练循环 + 兼容段
#
# 【两个不能改名的约定】
#   forward / __init__ 是 PyTorch 规定的方法名，必须叫这个
#   wte / c_attn 等子模块属性名保持英文，保证权重文件能和原版互换
# ================================================================================

import os
import sys
import time
import math
import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

# ================================================================================
# 模型配置：把所有"尺寸旋钮"集中放在一个地方
# ================================================================================
# @dataclass 是 Python 的语法糖，自动帮你生成 __init__，
# 写 MoXingPeiZhi(kuai_chang_du=16) 就能创建配置对象。
# ================================================================================
@dataclass
class MoXingPeiZhi:
    kuai_chang_du: int = None    # 块长度：模型一次最多能看多少个字符（原版 block_size）
    ci_biao_da_xiao: int = None  # 词表大小：一共有多少种不同的字符（原版 vocab_size）
    ceng_shu: int = 4            # 层数：Transformer 堆叠几层（原版 n_layer）
    qian_ru_wei_du: int = 64     # 嵌入维度：每个字符用多少个数字表示（原版 n_embd）
    qian_ru_wei_du2: int = 64    # 第二嵌入维度：部分模型内部用（原版 n_embd2）
    tou_shu: int = 4             # 注意力头数：注意力分成几组并行算（原版 n_head）

# ================================================================================
# GELU 激活函数（GPT-2 用的激活函数）
# ================================================================================
# 【和 ReLU 的区别】
#   ReLU：负数一刀切成 0，转折点很"生硬"
#   GELU：负数附近平滑过渡，像一个圆角版的 ReLU
#
#       ReLU             GELU
#        │   ╱            │   ╱
#        │  ╱             │  ╱
#        │ ╱              │ ╱
#   ─────┼╱────      ─────┼╱────
#        │              ⌒ │        ← 这里有个平滑的小凹陷
#
# 实践证明 GELU 在 Transformer 里训练效果更好。
# 公式看起来吓人，但你只需要知道"它是个平滑版 ReLU"就够了。（为节约算力，此算式为工程近似算式）
# ================================================================================
class GELUJiHuo(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


        # ================================================================================
        # 因果自注意力（Transformer 的灵魂！）
        # ================================================================================
        # 【一句话解释注意力】
        # 每个字符在预测下一个字符前，先"回头看看前面的字符们"，
# 并自己决定"我该重点参考谁"。
#
# 【生活比喻：开会发言】
# 你是第5个发言的人，发言前你会回顾前4个人说了什么：
#   - 第1个人说的和我相关度高 → 多参考（注意力权重大）
#   - 第3个人说的无关         → 少参考（注意力权重小）
# 每个人的"相关度"不是固定的，是模型自己学出来的。
#
# 【Q、K、V 三件套】
# 每个字符会生成三个向量：
#   Q（查询 query）：我想找什么样的信息？
#   K（键   key）  ：我能提供什么样的信息？
#   V（值   value）：我实际携带的信息内容
# 计算流程：拿我的 Q 和每个人的 K 做匹配 → 匹配分数做成权重 → 按权重把大家的 V 加权平均
#
# 【"因果"是什么意思】
# 预测下一个字符时，只许看前面，不许偷看后面（后面就是答案！）。
# 实现方式：用一个下三角矩阵当"挡板"，把未来位置的注意力分数设为负无穷，
# softmax 之后负无穷就变成 0，等于完全看不见。
#
#   挡板长这样（1=能看，0=不能看）：
#     位置1: [1 0 0 0]   ← 第1个字符只能看自己
#     位置2: [1 1 0 0]   ← 第2个能看第1个和自己
#     位置3: [1 1 1 0]
#     位置4: [1 1 1 1]   ← 最后一个能看所有人
# ================================================================================
class YinGuoZhuYiLi(nn.Module):

    def __init__(self, pei_zhi):
        super().__init__()
        # 嵌入维度必须能被头数整除，因为要平均分给每个头
        assert pei_zhi.qian_ru_wei_du % pei_zhi.tou_shu == 0

        # 一个大线性层同时生成 Q、K、V 三组向量（输出是 3 倍宽度，之后切成三份）
        # 属性名 c_attn 保持和原版一致（为了权重文件兼容）
        self.c_attn = nn.Linear(pei_zhi.qian_ru_wei_du, 3 * pei_zhi.qian_ru_wei_du)

        # 输出投影：把各个头的结果"汇总融合"一下
        self.c_proj = nn.Linear(pei_zhi.qian_ru_wei_du, pei_zhi.qian_ru_wei_du)

        # 因果挡板：下三角矩阵，register_buffer 表示"它是常量，不参与训练"
        self.register_buffer("bias", torch.tril(torch.ones(pei_zhi.kuai_chang_du, pei_zhi.kuai_chang_du))
                                     .view(1, 1, pei_zhi.kuai_chang_du, pei_zhi.kuai_chang_du))

        self.tou_shu = pei_zhi.tou_shu
        self.qian_ru_wei_du = pei_zhi.qian_ru_wei_du

    def forward(self, x):
        # x 的形状：(批量大小, 序列长度, 嵌入维度)
        pi_liang, xu_lie_chang, tong_dao = x.size()

        # 第一步：一次算出所有字符的 Q、K、V，然后切成三份
        q, k, v = self.c_attn(x).split(self.qian_ru_wei_du, dim=2)

        # 第二步：把每份再切成多个"头"
        # 多头 = 把嵌入维度分成几组，每组独立做注意力，各看各的角度
        # （好比同一段话，一个人关注语法，一个人关注情感，最后汇总）
        # transpose 是为了把"头"挪到批量维度旁边，方便批量矩阵乘法
        mei_tou_wei_du = tong_dao // self.tou_shu
        k = k.view(pi_liang, xu_lie_chang, self.tou_shu, mei_tou_wei_du).transpose(1, 2)
        q = q.view(pi_liang, xu_lie_chang, self.tou_shu, mei_tou_wei_du).transpose(1, 2)
        v = v.view(pi_liang, xu_lie_chang, self.tou_shu, mei_tou_wei_du).transpose(1, 2)

        # 第三步：算注意力分数 = Q 和 K 的匹配程度
        # 除以 √维度 是为了防止数值太大，softmax 后变成"一家独大"（梯度会消失）
        zhu_yi_li = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # 第四步：放上因果挡板，把"未来"的位置设为负无穷
        zhu_yi_li = zhu_yi_li.masked_fill(self.bias[:, :, :xu_lie_chang, :xu_lie_chang] == 0, float('-inf'))

        # 第五步：softmax 把分数变成"权重"（每行加起来等于1，负无穷变成0）
        zhu_yi_li = F.softmax(zhu_yi_li, dim=-1)

        # 第六步：按权重把各位置的 V 加权平均，得到每个位置"参考完前文后的新表示"
        y = zhu_yi_li @ v

        # 第七步：把多个头的结果拼回原来的形状，过一遍输出投影
        y = y.transpose(1, 2).contiguous().view(pi_liang, xu_lie_chang, tong_dao)
        y = self.c_proj(y)
        return y


# ================================================================================
# Transformer 块：注意力 + MLP，外加两个稳定训练的技巧
# ================================================================================
# 一个块 = 两个步骤：
#   步骤1：注意力（字符之间互相交流信息）
#   步骤2：MLP（每个字符自己消化刚收到的信息）
#
# 【技巧1：残差连接（x = x + ...）】
# 不是"x 变成新结果"，而是"x 加上新结果"。
# 好处：原始信息永远有一条"高速公路"直通到底，梯度也能沿这条路畅通回传，
# 深层网络才训练得动。（没有它，超过几层的网络就很难训练了）
#
# 【技巧2：LayerNorm（层归一化）】
# 在进注意力/MLP 之前，把数值"拉回标准范围"（均值0、方差1附近）。
# 好比每道工序前先把零件尺寸校准，防止误差逐层累积放大。
# ================================================================================
class TransformerKuai(nn.Module):

    def __init__(self, pei_zhi):
        super().__init__()
        self.ln_1 = nn.LayerNorm(pei_zhi.qian_ru_wei_du)      # 注意力前的归一化
        self.attn = YinGuoZhuYiLi(pei_zhi)                     # 注意力层
        self.ln_2 = nn.LayerNorm(pei_zhi.qian_ru_wei_du)      # MLP 前的归一化
        # MLP：先放大4倍（给思考留空间），激活，再缩回原尺寸
        self.mlp = nn.ModuleDict(dict(
            c_fc   = nn.Linear(pei_zhi.qian_ru_wei_du, 4 * pei_zhi.qian_ru_wei_du),
            c_proj = nn.Linear(4 * pei_zhi.qian_ru_wei_du, pei_zhi.qian_ru_wei_du),
            act    = GELUJiHuo(),
        ))

    def qian_kui(self, x):
        """MLP 前向：放大 → 激活 → 缩回"""
        return self.mlp.c_proj(self.mlp.act(self.mlp.c_fc(x)))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))    # 残差 + 注意力（交流）
        x = x + self.qian_kui(self.ln_2(x))  # 残差 + MLP（消化）
        return x


# ================================================================================
# Transformer 完整模型（和 GPT-2 结构完全相同，只是小很多）
# ================================================================================
# 完整流程：
#   字符编号 → 字符嵌入 + 位置嵌入 → N 个 Transformer 块 → 归一化 → 输出层
#                                                                  ↓
#                                                  每个位置输出"下一个字符是谁"的打分表
#
# 【为什么需要位置嵌入】
# 注意力本身不知道"顺序"——对它来说 "abc" 和 "cba" 没区别。
# 所以要给每个位置发一个"座位号向量"，加到字符向量上，模型才知道谁先谁后。
# ================================================================================
class Transformer(nn.Module):

    def __init__(self, pei_zhi):
        super().__init__()
        self.kuai_chang_du = pei_zhi.kuai_chang_du

        # 子模块属性名全部保持原版（transformer/wte/wpe/h/ln_f/lm_head），权重可互换
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(pei_zhi.ci_biao_da_xiao, pei_zhi.qian_ru_wei_du),  # 字符嵌入表
            wpe  = nn.Embedding(pei_zhi.kuai_chang_du, pei_zhi.qian_ru_wei_du),    # 位置嵌入表
            h    = nn.ModuleList([TransformerKuai(pei_zhi) for _ in range(pei_zhi.ceng_shu)]),  # N 个块
            ln_f = nn.LayerNorm(pei_zhi.qian_ru_wei_du),                            # 最终归一化
        ))
        # 输出层：把嵌入向量翻译回"词表里每个字符的得分"
        self.lm_head = nn.Linear(pei_zhi.qian_ru_wei_du, pei_zhi.ci_biao_da_xiao, bias=False)

        # 报告参数量
        can_shu_liang = sum(p.numel() for p in self.transformer.parameters())
        print("参数数量: %.2fM" % (can_shu_liang / 1e6,))

    def qu_kuai_chang_du(self):
        return self.kuai_chang_du

    def forward(self, zi_fu_bian_hao, mu_biao=None):
        """
        前向传播。
        zi_fu_bian_hao : 输入的字符编号，形状 (批量, 序列长度)，原版叫 idx
        mu_biao        : 期望的正确答案（下一个字符的编号），原版叫 targets
                         训练时传入，用来算损失；生成时不传，只要预测结果
        """
        she_bei = zi_fu_bian_hao.device
        pi_liang, xu_lie_chang = zi_fu_bian_hao.size()
        assert xu_lie_chang <= self.kuai_chang_du, \
            f"输入长度 {xu_lie_chang} 超过了块长度 {self.kuai_chang_du}"

        # 生成位置编号 [0, 1, 2, ..., 序列长度-1]
        wei_zhi = torch.arange(0, xu_lie_chang, dtype=torch.long, device=she_bei).unsqueeze(0)

        # 第一步：查表。字符编号 → 字符向量，位置编号 → 位置向量，两者相加
        zi_fu_qian_ru = self.transformer.wte(zi_fu_bian_hao)   # (批量, 长度, 嵌入维度)
        wei_zhi_qian_ru = self.transformer.wpe(wei_zhi)        # (1, 长度, 嵌入维度)
        x = zi_fu_qian_ru + wei_zhi_qian_ru

        # 第二步：依次通过每个 Transformer 块（和 easy_nn 的"层层传递"一个道理）
        for kuai in self.transformer.h:
            x = kuai(x)

        # 第三步：最终归一化 + 输出层，得到每个位置对"下一个字符"的打分表
        x = self.transformer.ln_f(x)
        da_fen = self.lm_head(x)   # (批量, 长度, 词表大小)，原版叫 logits

        # 第四步：如果给了正确答案，就算损失
        # cross_entropy（交叉熵）= 分类任务版的"均方误差"，
        # 含义：正确答案的得分越高，损失越小
        # ignore_index=-1：答案是 -1 的位置（填充位）不计入损失
        sun_shi = None
        if mu_biao is not None:
            sun_shi = F.cross_entropy(da_fen.view(-1, da_fen.size(-1)),
                                      mu_biao.view(-1), ignore_index=-1)

        return da_fen, sun_shi

# ================================================================================
#  第 2/3 部分：其他 5 种语言模型，按"进化顺序"排列
# ================================================================================
#   Bigram   → 只看前1个字符，查表（最原始）
#   MLP      → 看前几个字符，拼接后过全连接网络（2003年 Bengio 经典论文）
#   RNN      → 用"记忆"逐个读字符（不限长度，但记性差）
#   GRU      → 带"门控"的 RNN（记性好一些）
#   BoW      → 把前文"平均"一下来用（注意力的雏形）
#   ↓ 然后才是上面的 Transformer（注意力：选择性地看前文）
# ================================================================================


# ================================================================================
# Bigram：最原始的语言模型，本质就是一张表
# ================================================================================
# 思路简单粗暴："只根据前1个字符，猜下1个字符"
# 整个模型就是一张 词表大小×词表大小 的表格：
#   表格[a][b] = 看到字符 a 后，下一个是字符 b 的得分
# 例如学完人名数据后，表格["q"]["u"] 会很大（q 后面几乎总是 u）。
# 缺点：完全不看更早的历史，生成的词只有"局部像"，整体很乱。
# ================================================================================
class Bigram(nn.Module):

    def __init__(self, pei_zhi):
        super().__init__()
        n = pei_zhi.ci_biao_da_xiao
        # 整个模型唯一的参数：一张 n×n 的得分表
        self.logits = nn.Parameter(torch.zeros((n, n)))

    def qu_kuai_chang_du(self):
        return 1   # 只需要看前1个字符

    def forward(self, zi_fu_bian_hao, mu_biao=None):
        # "前向传播"就是查表，没了
        da_fen = self.logits[zi_fu_bian_hao]

        sun_shi = None
        if mu_biao is not None:
            sun_shi = F.cross_entropy(da_fen.view(-1, da_fen.size(-1)),
                                      mu_biao.view(-1), ignore_index=-1)
        return da_fen, sun_shi


# ================================================================================
# MLP 语言模型（Bengio 2003 经典论文的做法）
# ================================================================================
# 思路：看前 N 个字符（而不是1个），把它们的向量拼接起来，喂给一个 MLP。
# 这就是你在 easy_nn.py 里写的多层感知机，只是输入变成了"字符向量拼接"。
# 比 Bigram 强：能利用更多上文。
# 缺点：能看的字符数固定死了（N 个），改不了。
# ================================================================================
class MLP(nn.Module):

    def __init__(self, pei_zhi):
        super().__init__()
        self.kuai_chang_du = pei_zhi.kuai_chang_du
        self.ci_biao_da_xiao = pei_zhi.ci_biao_da_xiao
        # +1 是给"空白符"留的位置：当前文不足 N 个字符时用它来填补
        self.wte = nn.Embedding(pei_zhi.ci_biao_da_xiao + 1, pei_zhi.qian_ru_wei_du)
        # MLP 本体：拼接后的大向量 → 隐藏层 → 输出打分表
        self.mlp = nn.Sequential(
            nn.Linear(self.kuai_chang_du * pei_zhi.qian_ru_wei_du, pei_zhi.qian_ru_wei_du2),
            nn.Tanh(),
            nn.Linear(pei_zhi.qian_ru_wei_du2, self.ci_biao_da_xiao)
        )

    def qu_kuai_chang_du(self):
        return self.kuai_chang_du

    def forward(self, zi_fu_bian_hao, mu_biao=None):
        # 收集每个位置"往前数 N 个字符"的嵌入向量
        # 技巧：用 torch.roll 把序列整体右移一格，重复 N 次，
        # 每移一次就把"再往前一个字符"的嵌入收进来
        qian_ru_lie_biao = []
        for k in range(self.kuai_chang_du):
            zi_fu_qian_ru = self.wte(zi_fu_bian_hao)          # 当前偏移下的嵌入
            zi_fu_bian_hao = torch.roll(zi_fu_bian_hao, 1, 1)  # 整体右移一格
            zi_fu_bian_hao[:, 0] = self.ci_biao_da_xiao        # 移出来的空位填"空白符"
            qian_ru_lie_biao.append(zi_fu_qian_ru)

        # 拼接所有嵌入，过 MLP
        x = torch.cat(qian_ru_lie_biao, -1)
        da_fen = self.mlp(x)

        sun_shi = None
        if mu_biao is not None:
            sun_shi = F.cross_entropy(da_fen.view(-1, da_fen.size(-1)),
                                      mu_biao.view(-1), ignore_index=-1)
        return da_fen, sun_shi


# ================================================================================
# RNN 单元：循环神经网络的"一拍"
# ================================================================================
# RNN 的思路：像人读书一样，一个字符一个字符地读，边读边更新"脑中的记忆"。
#   记忆_新 = tanh(线性变换(当前字符 拼接 记忆_旧))
# 优点：理论上能记住任意长的历史（不像 MLP 固定只看 N 个）。
# 缺点：记忆是"挤"在一个固定大小的向量里的，读得越长忘得越多。
# ================================================================================
class RNNDanYuan(nn.Module):

    def __init__(self, pei_zhi):
        super().__init__()
        # 输入是"当前字符向量 + 上一拍记忆"拼起来，输出是新记忆
        self.xh_to_h = nn.Linear(pei_zhi.qian_ru_wei_du + pei_zhi.qian_ru_wei_du2,
                                 pei_zhi.qian_ru_wei_du2)

    def forward(self, dang_qian_shu_ru, ji_yi_jiu):
        # 拼接 → 线性变换 → tanh 压缩到 (-1, 1) → 新记忆
        pin_jie = torch.cat([dang_qian_shu_ru, ji_yi_jiu], dim=1)
        ji_yi_xin = F.tanh(self.xh_to_h(pin_jie))
        return ji_yi_xin


# ================================================================================
# GRU 单元：带"门控"的升级版 RNN
# ================================================================================
# 普通 RNN 的问题：每读一个字符，整个记忆都被强制重写，旧记忆很容易被冲掉。
# GRU 加了两扇"门"来精细控制：
#   重置门 r：决定"写新记忆时，旧记忆里哪些部分先清空"
#   更新门 z：决定"每个记忆通道是保留旧值，还是换成新值"
# 最终：新记忆 = (1-z)×旧记忆 + z×候选新记忆   ← 新旧按比例混合，而不是全部重写
# ================================================================================
class GRUDanYuan(nn.Module):

    def __init__(self, pei_zhi):
        super().__init__()
        ru = pei_zhi.qian_ru_wei_du + pei_zhi.qian_ru_wei_du2
        chu = pei_zhi.qian_ru_wei_du2
        self.xh_to_z = nn.Linear(ru, chu)      # 更新门
        self.xh_to_r = nn.Linear(ru, chu)      # 重置门
        self.xh_to_hbar = nn.Linear(ru, chu)   # 候选新记忆

    def forward(self, dang_qian_shu_ru, ji_yi_jiu):
        pin_jie = torch.cat([dang_qian_shu_ru, ji_yi_jiu], dim=1)

        # 重置门：sigmoid 输出 0~1，乘到旧记忆上 = 部分擦除
        r = F.sigmoid(self.xh_to_r(pin_jie))
        ji_yi_ca_chu = r * ji_yi_jiu

        # 用"擦除后的旧记忆"算候选新记忆
        pin_jie_ca_chu = torch.cat([dang_qian_shu_ru, ji_yi_ca_chu], dim=1)
        ji_yi_hou_xuan = F.tanh(self.xh_to_hbar(pin_jie_ca_chu))

        # 更新门：决定新旧记忆的混合比例
        z = F.sigmoid(self.xh_to_z(pin_jie))
        ji_yi_xin = (1 - z) * ji_yi_jiu + z * ji_yi_hou_xuan
        return ji_yi_xin


# ================================================================================
# RNN 完整模型（外壳，内部可以装 RNN 单元或 GRU 单元）
# ================================================================================
class RNN(nn.Module):

    def __init__(self, pei_zhi, dan_yuan_lei_xing):
        super().__init__()
        self.kuai_chang_du = pei_zhi.kuai_chang_du
        self.ci_biao_da_xiao = pei_zhi.ci_biao_da_xiao
        # 初始记忆：读第一个字符之前"脑子里的状态"，也是可学习的参数
        self.start = nn.Parameter(torch.zeros(1, pei_zhi.qian_ru_wei_du2))
        self.wte = nn.Embedding(pei_zhi.ci_biao_da_xiao, pei_zhi.qian_ru_wei_du)  # 字符嵌入表
        # 根据类型装入不同的"单元"
        if dan_yuan_lei_xing == 'rnn':
            self.cell = RNNDanYuan(pei_zhi)
        elif dan_yuan_lei_xing == 'gru':
            self.cell = GRUDanYuan(pei_zhi)
        self.lm_head = nn.Linear(pei_zhi.qian_ru_wei_du2, self.ci_biao_da_xiao)

    def qu_kuai_chang_du(self):
        return self.kuai_chang_du

    def forward(self, zi_fu_bian_hao, mu_biao=None):
        pi_liang, xu_lie_chang = zi_fu_bian_hao.size()

        # 先一次性查好所有字符的嵌入（比循环里逐个查更快）
        qian_ru = self.wte(zi_fu_bian_hao)   # (批量, 长度, 嵌入维度)

        # 逐个字符往后读，每读一个就更新一次记忆
        ji_yi = self.start.expand((pi_liang, -1))   # 把初始记忆复制成一批
        ji_yi_lie_biao = []
        for i in range(xu_lie_chang):
            dang_qian = qian_ru[:, i, :]            # 取出第 i 个字符的嵌入
            ji_yi = self.cell(dang_qian, ji_yi)     # 喂给单元，得到新记忆
            ji_yi_lie_biao.append(ji_yi)            # 每一拍的记忆都留着（每个位置都要预测）

        # 把每一拍的记忆堆起来，统一过输出层
        suo_you_ji_yi = torch.stack(ji_yi_lie_biao, 1)   # (批量, 长度, 记忆维度)
        da_fen = self.lm_head(suo_you_ji_yi)

        sun_shi = None
        if mu_biao is not None:
            sun_shi = F.cross_entropy(da_fen.view(-1, da_fen.size(-1)),
                                      mu_biao.view(-1), ignore_index=-1)
        return da_fen, sun_shi


# ================================================================================
# BoW（词袋模型）：注意力的"雏形"
# ================================================================================
# 思路：预测下一个字符时，把前面所有字符的向量"简单平均"一下来参考。
#
# 和注意力的对比：
#   BoW    ：前面每个字符的权重一样（一律平均）  ← 大锅饭
#   注意力 ：每个字符的权重由模型自己学         ← 按需分配
#
# Karpathy 故意把 BoW 写得和注意力代码长得几乎一样，
# 就是为了让你看出来：注意力 = 会学权重的 BoW。
# ================================================================================
class YinGuoCiDai(nn.Module):
    """因果词袋：对前面所有字符做平均（不能看未来，和注意力同款挡板）"""

    def __init__(self, pei_zhi):
        super().__init__()
        self.kuai_chang_du = pei_zhi.kuai_chang_du
        self.register_buffer("bias", torch.tril(torch.ones(pei_zhi.kuai_chang_du, pei_zhi.kuai_chang_du))
                             .view(1, pei_zhi.kuai_chang_du, pei_zhi.kuai_chang_du))

    def forward(self, x):
        pi_liang, xu_lie_chang, tong_dao = x.size()

        # 注意力分数全部填 0（即"人人平等"），盖上因果挡板，softmax 后
        # 每行就变成均匀权重，例如第3行 = [1/3, 1/3, 1/3, 0, ...]
        quan_zhong = torch.zeros((pi_liang, xu_lie_chang, xu_lie_chang), device=x.device)
        quan_zhong = quan_zhong.masked_fill(self.bias[:, :xu_lie_chang, :xu_lie_chang] == 0, float('-inf'))
        quan_zhong = F.softmax(quan_zhong, dim=-1)
        y = quan_zhong @ x   # 加权平均（权重全相等 = 普通平均）
        return y


class CiDaiKuai(nn.Module):
    """词袋块：词袋汇总 + MLP 消化，结构模仿 Transformer 块"""

    def __init__(self, pei_zhi):
        super().__init__()
        self.cbow = YinGuoCiDai(pei_zhi)
        self.mlp = nn.ModuleDict(dict(
            c_fc   = nn.Linear(pei_zhi.qian_ru_wei_du, pei_zhi.qian_ru_wei_du2),
            c_proj = nn.Linear(pei_zhi.qian_ru_wei_du2, pei_zhi.qian_ru_wei_du),
        ))

    def qian_kui(self, x):
        return self.mlp.c_proj(F.tanh(self.mlp.c_fc(x)))

    def forward(self, x):
        x = x + self.cbow(x)      # 残差 + 词袋平均
        x = x + self.qian_kui(x)  # 残差 + MLP
        return x


class BoW(nn.Module):
    """词袋语言模型完整版：嵌入 → 词袋块 → 输出层"""

    def __init__(self, pei_zhi):
        super().__init__()
        self.kuai_chang_du = pei_zhi.kuai_chang_du
        self.ci_biao_da_xiao = pei_zhi.ci_biao_da_xiao
        self.wte = nn.Embedding(pei_zhi.ci_biao_da_xiao, pei_zhi.qian_ru_wei_du)  # 字符嵌入
        self.wpe = nn.Embedding(pei_zhi.kuai_chang_du, pei_zhi.qian_ru_wei_du)    # 位置嵌入
        self.context_block = CiDaiKuai(pei_zhi)
        self.lm_head = nn.Linear(pei_zhi.qian_ru_wei_du, self.ci_biao_da_xiao)

    def qu_kuai_chang_du(self):
        return self.kuai_chang_du

    def forward(self, zi_fu_bian_hao, mu_biao=None):
        she_bei = zi_fu_bian_hao.device
        pi_liang, xu_lie_chang = zi_fu_bian_hao.size()
        wei_zhi = torch.arange(0, xu_lie_chang, dtype=torch.long, device=she_bei).unsqueeze(0)

        x = self.wte(zi_fu_bian_hao) + self.wpe(wei_zhi)   # 字符嵌入 + 位置嵌入
        x = self.context_block(x)                          # 词袋汇总前文
        da_fen = self.lm_head(x)

        sun_shi = None
        if mu_biao is not None:
            sun_shi = F.cross_entropy(da_fen.view(-1, da_fen.size(-1)),
                                      mu_biao.view(-1), ignore_index=-1)
        return da_fen, sun_shi
# ================================================================================
#  第 3/3 部分：生成函数 + 数据集 + 训练循环 + 兼容段
# ================================================================================

# ================================================================================
# 生成函数：让训练好的模型"字符接龙"
# ================================================================================
# @torch.no_grad() 表示"这里不需要记账求导"（只是用模型，不训练它），省内存提速
# ================================================================================
@torch.no_grad()
def sheng_cheng(mo_xing, zi_fu_bian_hao, xin_zi_fu_shu, wen_du=1.0, shi_fou_cai_yang=False, qian_k=None):
    """
    字符接龙：从给定的开头出发，一次生成一个字符，循环往复。

    参数：
      mo_xing         : 训练好的模型
      zi_fu_bian_hao  : 开头序列（至少要有一个 <开始> 符）
      xin_zi_fu_shu   : 要接龙多少个新字符
      wen_du          : 温度。>1 更随机大胆，<1 更保守稳妥
      shi_fou_cai_yang: True=按概率抽签（有变化），False=永远选最高分（死板）
      qian_k          : 只在得分最高的 k 个字符里抽签，防止抽到太离谱的
    """
    kuai_chang_du = mo_xing.qu_kuai_chang_du()
    for _ in range(xin_zi_fu_shu):
        # 序列太长就只保留最后 kuai_chang_du 个（模型最多只能看这么多）
        shu_ru = zi_fu_bian_hao if zi_fu_bian_hao.size(1) <= kuai_chang_du else zi_fu_bian_hao[:, -kuai_chang_du:]
        # 前向传播，拿到打分表
        da_fen, _ = mo_xing(shu_ru)
        # 只取最后一个位置的打分（我们只关心"下一个字符是谁"），除以温度
        da_fen = da_fen[:, -1, :] / wen_du
        # top-k 截断：低于第 k 名的全部打成负无穷（抽不中）
        if qian_k is not None:
            v, _ = torch.topk(da_fen, qian_k)
            da_fen[da_fen < v[:, [-1]]] = -float('Inf')
        # 打分 → 概率
        gai_lv = F.softmax(da_fen, dim=-1)
        # 抽签 或 直接拿最高分
        if shi_fou_cai_yang:
            xia_yi_ge = torch.multinomial(gai_lv, num_samples=1)
        else:
            _, xia_yi_ge = torch.topk(gai_lv, k=1, dim=-1)
        # 接到序列末尾，继续下一轮
        zi_fu_bian_hao = torch.cat((zi_fu_bian_hao, xia_yi_ge), dim=1)

    return zi_fu_bian_hao


def da_yin_yang_ben(shu_liang=10):
    """从模型采样一批词，并区分"抄训练集的 / 抄测试集的 / 全新创作的"打印出来"""
    X_chu_shi = torch.zeros(shu_liang, 1, dtype=torch.long).to(can_shu_biao.device)
    qian_k = can_shu_biao.top_k if can_shu_biao.top_k != -1 else None
    bu_shu = xun_lian_ji.qu_shu_chu_chang_du() - 1   # -1 因为开头已有 <开始> 符
    X_cai_yang = sheng_cheng(mo_xing, X_chu_shi, bu_shu, qian_k=qian_k, shi_fou_cai_yang=True).to('cpu')

    zai_xun_lian_ji, zai_ce_shi_ji, quan_xin = [], [], []
    for i in range(X_cai_yang.size(0)):
        # 取出第 i 行的编号序列（去掉开头的 <开始> 符）
        hang = X_cai_yang[i, 1:].tolist()
        # 编号 0 是 <结束> 符，遇到它就把后面的截掉
        jie_duan_wei_zhi = hang.index(0) if 0 in hang else len(hang)
        hang = hang[:jie_duan_wei_zhi]
        ci = xun_lian_ji.jie_ma(hang)
        # 分类：是抄训练集的？抄测试集的？还是全新创作？
        if xun_lian_ji.bao_han(ci):
            zai_xun_lian_ji.append(ci)
        elif ce_shi_ji.bao_han(ci):
            zai_ce_shi_ji.append(ci)
        else:
            quan_xin.append(ci)

    print('-' * 80)
    for lie_biao, miao_shu in [(zai_xun_lian_ji, '在训练集中（背下来的）'),
                                (zai_ce_shi_ji, '在测试集中（巧合撞上）'),
                                (quan_xin, '全新创作')]:
        print(f"{len(lie_biao)} 个样本{miao_shu}:")
        for ci in lie_biao:
            print(ci)
    print('-' * 80)


@torch.inference_mode()
def ping_gu(mo_xing, shu_ju_ji, pi_da_xiao=50, zui_da_pi_shu=None):
    """在数据集上算平均损失，衡量模型水平（不训练，只考试）"""
    mo_xing.eval()    # 切换到"考试模式"
    jia_zai_qi = DataLoader(shu_ju_ji, shuffle=True, batch_size=pi_da_xiao, num_workers=0)
    sun_shi_lie_biao = []
    for i, pi in enumerate(jia_zai_qi):
        pi = [t.to(can_shu_biao.device) for t in pi]
        X, Y = pi
        _, sun_shi = mo_xing(X, Y)
        sun_shi_lie_biao.append(sun_shi.item())
        if zui_da_pi_shu is not None and i >= zui_da_pi_shu:
            break
    ping_jun_sun_shi = torch.tensor(sun_shi_lie_biao).mean().item()
    mo_xing.train()   # 切回"学习模式"
    return ping_jun_sun_shi


# ================================================================================
# 字符数据集：把"一行一个词"的文本文件变成模型能吃的数字
# ================================================================================
# 核心工作就两件：
#   编码：'emma' → [5, 13, 13, 1]   （字符 → 编号）
#   解码：[5, 13, 13, 1] → 'emma'   （编号 → 字符）
# 编号 0 是特殊符号，同时充当 <开始> 和 <结束>。
# ================================================================================
class ZiFuShuJuJi(Dataset):

    def __init__(self, ci_lie_biao, zi_fu_biao, zui_da_ci_chang):
        self.ci_lie_biao = ci_lie_biao
        self.zi_fu_biao = zi_fu_biao
        self.zui_da_ci_chang = zui_da_ci_chang
        # 字符 → 编号 的字典（i+1 是因为编号 0 留给特殊符号）
        self.zi_dao_hao = {zi: i + 1 for i, zi in enumerate(zi_fu_biao)}
        # 编号 → 字符 的反向字典
        self.hao_dao_zi = {hao: zi for zi, hao in self.zi_dao_hao.items()}

    def __len__(self):
        return len(self.ci_lie_biao)

    def bao_han(self, ci):
        return ci in self.ci_lie_biao

    def qu_ci_biao_da_xiao(self):
        return len(self.zi_fu_biao) + 1    # +1 是特殊符号 0

    def qu_shu_chu_chang_du(self):
        return self.zui_da_ci_chang + 1    # +1 是开头的 <开始> 符

    def bian_ma(self, ci):
        """'emma' → tensor([5, 13, 13, 1])"""
        return torch.tensor([self.zi_dao_hao[zi] for zi in ci], dtype=torch.long)

    def jie_ma(self, hao_lie_biao):
        """[5, 13, 13, 1] → 'emma'"""
        return ''.join(self.hao_dao_zi[hao] for hao in hao_lie_biao)

    def __getitem__(self, suo_yin):
        """
        给 DataLoader 用：取出第 suo_yin 个词，做成 (输入x, 答案y) 一对。

        以 'emma'（最大词长设为6）为例：
          x = [0, 5, 13, 13, 1,  0,  0]   ← 开头是<开始>符，后面补0
          y = [5, 13, 13, 1, 0, -1, -1]   ← y 是 x 左移一格（每个位置的"正确下一个字符"）
                              ↑   ↑↑
                        <结束>符  -1表示"这些位置不算损失"
        """
        ci = self.ci_lie_biao[suo_yin]
        hao = self.bian_ma(ci)
        x = torch.zeros(self.zui_da_ci_chang + 1, dtype=torch.long)
        y = torch.zeros(self.zui_da_ci_chang + 1, dtype=torch.long)
        x[1:1 + len(hao)] = hao
        y[:len(hao)] = hao
        y[len(hao) + 1:] = -1   # -1 的位置不计入损失
        return x, y


def chuang_jian_shu_ju_ji(wen_jian_lu_jing):
    """读取文本文件，切分成训练集和测试集"""
    with open(wen_jian_lu_jing, 'r') as f:
        shu_ju = f.read()
    ci_lie_biao = shu_ju.splitlines()
    ci_lie_biao = [c.strip() for c in ci_lie_biao]    # 去掉首尾空格
    ci_lie_biao = [c for c in ci_lie_biao if c]       # 去掉空行
    zi_fu_biao = sorted(list(set(''.join(ci_lie_biao))))   # 所有出现过的字符
    zui_da_ci_chang = max(len(c) for c in ci_lie_biao)
    print(f"数据集词数: {len(ci_lie_biao)}")
    print(f"最长的词: {zui_da_ci_chang} 个字符")
    print(f"词表大小: {len(zi_fu_biao)} 种字符")
    print(f"词表内容: {''.join(zi_fu_biao)}")

    # 切 10%（最多1000个）做测试集，剩下做训练集
    ce_shi_ji_da_xiao = min(1000, int(len(ci_lie_biao) * 0.1))
    sui_ji_shun_xu = torch.randperm(len(ci_lie_biao)).tolist()
    xun_lian_ci = [ci_lie_biao[i] for i in sui_ji_shun_xu[:-ce_shi_ji_da_xiao]]
    ce_shi_ci = [ci_lie_biao[i] for i in sui_ji_shun_xu[-ce_shi_ji_da_xiao:]]
    print(f"切分: {len(xun_lian_ci)} 个训练 + {len(ce_shi_ci)} 个测试")

    xun_lian_ji = ZiFuShuJuJi(xun_lian_ci, zi_fu_biao, zui_da_ci_chang)
    ce_shi_ji = ZiFuShuJuJi(ce_shi_ci, zi_fu_biao, zui_da_ci_chang)
    return xun_lian_ji, ce_shi_ji


class WuXianJiaZaiQi:
    """无限数据加载器：取完一轮自动从头再来，训练循环就不用关心"数据用完了"的问题"""

    def __init__(self, shu_ju_ji, **qi_ta):
        cai_yang_qi = torch.utils.data.RandomSampler(shu_ju_ji, replacement=True, num_samples=int(1e10))
        self.jia_zai_qi = DataLoader(shu_ju_ji, sampler=cai_yang_qi, **qi_ta)
        self.die_dai_qi = iter(self.jia_zai_qi)

    def qu_xia_yi_pi(self):
        try:
            pi = next(self.die_dai_qi)
        except StopIteration:
            self.die_dai_qi = iter(self.jia_zai_qi)
            pi = next(self.die_dai_qi)
        return pi


# ================================================================================
# 【兼容段】让本文件能和原版 makemore 的测试脚本一起用
# ================================================================================
# 和 easy_nn.py 一样，只做别名和薄包装，不影响上面的学习内容。
# 注意：兼容段必须放在 if __name__ == '__main__' 之前才能被 import 使用，
# 所以请把下面这段移动到主程序（if __name__ == '__main__'）的上方！
# ================================================================================
# ---- 配置类的兼容包装（不能用简单别名，因为字段名不同）----
def ModelConfig(block_size=None, vocab_size=None, n_layer=4, n_embd=64, n_embd2=64, n_head=4):
    """接收原版英文参数名，翻译成拼音参数名，返回配置对象"""
    return MoXingPeiZhi(kuai_chang_du=block_size, ci_biao_da_xiao=vocab_size,
                        ceng_shu=n_layer, qian_ru_wei_du=n_embd,
                        qian_ru_wei_du2=n_embd2, tou_shu=n_head)
    
# ---- 其余类别名（字段/参数一致，直接赋值即可）----
NewGELU = GELUJiHuo
CausalSelfAttention = YinGuoZhuYiLi
Block = TransformerKuai
CausalBoW = YinGuoCiDai
BoWBlock = CiDaiKuai
RNNCell = RNNDanYuan
GRUCell = GRUDanYuan
CharDataset = ZiFuShuJuJi
InfiniteDataLoader = WuXianJiaZaiQi
# Transformer / Bigram / MLP / RNN / BoW 类名本来就和原版相同，无需别名

# ---- 函数别名 ----
generate = sheng_cheng
evaluate = ping_gu
create_datasets = chuang_jian_shu_ju_ji

# ---- 方法别名：补上原版的英文方法名 ----
Transformer.get_block_size = Transformer.qu_kuai_chang_du
Bigram.get_block_size = Bigram.qu_kuai_chang_du
MLP.get_block_size = MLP.qu_kuai_chang_du
RNN.get_block_size = RNN.qu_kuai_chang_du
BoW.get_block_size = BoW.qu_kuai_chang_du
ZiFuShuJuJi.contains = ZiFuShuJuJi.bao_han
ZiFuShuJuJi.get_vocab_size = ZiFuShuJuJi.qu_ci_biao_da_xiao
ZiFuShuJuJi.get_output_length = ZiFuShuJuJi.qu_shu_chu_chang_du
ZiFuShuJuJi.encode = ZiFuShuJuJi.bian_ma
ZiFuShuJuJi.decode = ZiFuShuJuJi.jie_ma
WuXianJiaZaiQi.next = WuXianJiaZaiQi.qu_xia_yi_pi


# ================================================================================
# 主程序：解析命令行参数 → 建数据集 → 建模型 → 训练循环
# ================================================================================
if __name__ == '__main__':

    # ---- 命令行参数（注意：参数名保持和原版一致，方便复用原版的命令行用法）----
    jie_xi_qi = argparse.ArgumentParser(description="Make More 拼音版")
    jie_xi_qi.add_argument('--input-file', '-i', type=str, default='names.txt', help="输入文件，一行一个词")
    jie_xi_qi.add_argument('--work-dir', '-o', type=str, default='out', help="输出目录")
    jie_xi_qi.add_argument('--resume', action='store_true', help="从已有模型继续训练")
    jie_xi_qi.add_argument('--sample-only', action='store_true', help="只采样不训练")
    jie_xi_qi.add_argument('--num-workers', '-n', type=int, default=4, help="数据加载进程数")
    jie_xi_qi.add_argument('--max-steps', type=int, default=-1, help="最大训练步数，-1为无限")
    jie_xi_qi.add_argument('--device', type=str, default='cpu', help="cpu|cuda|mps")
    jie_xi_qi.add_argument('--seed', type=int, default=3407, help="随机种子")
    jie_xi_qi.add_argument('--top-k', type=int, default=-1, help="采样top-k，-1为不限制")
    jie_xi_qi.add_argument('--type', type=str, default='transformer', help="bigram|mlp|rnn|gru|bow|transformer")
    jie_xi_qi.add_argument('--n-layer', type=int, default=4, help="层数")
    jie_xi_qi.add_argument('--n-head', type=int, default=4, help="注意力头数")
    jie_xi_qi.add_argument('--n-embd', type=int, default=64, help="嵌入维度")
    jie_xi_qi.add_argument('--n-embd2', type=int, default=64, help="第二嵌入维度")
    jie_xi_qi.add_argument('--batch-size', '-b', type=int, default=32, help="批量大小")
    jie_xi_qi.add_argument('--learning-rate', '-l', type=float, default=5e-4, help="学习率")
    jie_xi_qi.add_argument('--weight-decay', '-w', type=float, default=0.01, help="权重衰减")
    can_shu_biao = jie_xi_qi.parse_args()
    print(vars(can_shu_biao))

    # ---- 系统初始化 ----
    torch.manual_seed(can_shu_biao.seed)        # 固定随机种子，让结果可复现
    torch.cuda.manual_seed_all(can_shu_biao.seed)
    os.makedirs(can_shu_biao.work_dir, exist_ok=True)
    ri_zhi_qi = SummaryWriter(log_dir=can_shu_biao.work_dir)   # TensorBoard 日志

    # ---- 创建数据集 ----
    xun_lian_ji, ce_shi_ji = chuang_jian_shu_ju_ji(can_shu_biao.input_file)
    ci_biao_da_xiao = xun_lian_ji.qu_ci_biao_da_xiao()
    kuai_chang_du = xun_lian_ji.qu_shu_chu_chang_du()
    print(f"数据集确定: 词表大小={ci_biao_da_xiao}, 块长度={kuai_chang_du}")

    # ---- 创建模型 ----
    pei_zhi = MoXingPeiZhi(ci_biao_da_xiao=ci_biao_da_xiao, kuai_chang_du=kuai_chang_du,
                           ceng_shu=can_shu_biao.n_layer, tou_shu=can_shu_biao.n_head,
                           qian_ru_wei_du=can_shu_biao.n_embd, qian_ru_wei_du2=can_shu_biao.n_embd2)
    if can_shu_biao.type == 'transformer':
        mo_xing = Transformer(pei_zhi)
    elif can_shu_biao.type == 'bigram':
        mo_xing = Bigram(pei_zhi)
    elif can_shu_biao.type == 'mlp':
        mo_xing = MLP(pei_zhi)
    elif can_shu_biao.type == 'rnn':
        mo_xing = RNN(pei_zhi, dan_yuan_lei_xing='rnn')
    elif can_shu_biao.type == 'gru':
        mo_xing = RNN(pei_zhi, dan_yuan_lei_xing='gru')
    elif can_shu_biao.type == 'bow':
        mo_xing = BoW(pei_zhi)
    else:
        raise ValueError(f'不认识的模型类型 {can_shu_biao.type}')
    mo_xing.to(can_shu_biao.device)
    print(f"模型参数总数: {sum(p.numel() for p in mo_xing.parameters())}")

    # 断点续训 / 只采样模式
    if can_shu_biao.resume or can_shu_biao.sample_only:
        print("从工作目录加载已有模型")
        mo_xing.load_state_dict(torch.load(os.path.join(can_shu_biao.work_dir, 'model.pt')))
    if can_shu_biao.sample_only:
        da_yin_yang_ben(shu_liang=50)
        sys.exit()

    # ---- 优化器：AdamW（梯度下降的智能升级版）----
    # 你在 easy_nn.py 里手写的"参数 -= 学习率×梯度"是最朴素的版本，
    # AdamW 在此基础上加了"惯性"和"自适应步长"，收敛更快更稳。
    you_hua_qi = torch.optim.AdamW(mo_xing.parameters(), lr=can_shu_biao.learning_rate,
                                   weight_decay=can_shu_biao.weight_decay,
                                   betas=(0.9, 0.99), eps=1e-8)

    # ---- 数据加载器 ----
    pi_jia_zai_qi = WuXianJiaZaiQi(xun_lian_ji, batch_size=can_shu_biao.batch_size,
                                   pin_memory=True, num_workers=can_shu_biao.num_workers)

    # ---- 训练循环（和 easy_nn.py 的五步完全对应！）----
    zui_jia_sun_shi = None
    bu_shu = 0
    while True:

        t0 = time.time()

        # 取下一批数据，搬到计算设备上
        pi = pi_jia_zai_qi.qu_xia_yi_pi()
        pi = [t.to(can_shu_biao.device) for t in pi]
        X, Y = pi

        # ① 前向传播 + ② 计算损失（模型内部一起做了）
        da_fen, sun_shi = mo_xing(X, Y)

        # ③ 清空梯度（等价于 easy_nn 的 qing_kong_ti_du）
        mo_xing.zero_grad(set_to_none=True)
        # ④ 反向传播（等价于 easy_engine 的 fan_xiang_chuan_bo）
        sun_shi.backward()
        # ⑤ 更新参数（等价于 easy_nn 的"参数 -= 学习率×梯度"，但更智能）
        you_hua_qi.step()

        # GPU 计时需要先等所有计算完成
        if can_shu_biao.device.startswith('cuda'):
            torch.cuda.synchronize()
        t1 = time.time()

        # 每 10 步打印一次进度
        if bu_shu % 10 == 0:
            print(f"第 {bu_shu} 步 | 损失 {sun_shi.item():.4f} | 用时 {(t1-t0)*1000:.2f}ms")

        # 每 500 步考一次试，成绩创新高就存档
        if bu_shu > 0 and bu_shu % 500 == 0:
            xun_lian_sun_shi = ping_gu(mo_xing, xun_lian_ji, pi_da_xiao=100, zui_da_pi_shu=10)
            ce_shi_sun_shi = ping_gu(mo_xing, ce_shi_ji, pi_da_xiao=100, zui_da_pi_shu=10)
            ri_zhi_qi.add_scalar("Loss/train", xun_lian_sun_shi, bu_shu)
            ri_zhi_qi.add_scalar("Loss/test", ce_shi_sun_shi, bu_shu)
            ri_zhi_qi.flush()
            print(f"第 {bu_shu} 步 训练损失: {xun_lian_sun_shi} 测试损失: {ce_shi_sun_shi}")
            # 注意：看的是"测试损失"——没见过的数据上的成绩才是真本事
            if zui_jia_sun_shi is None or ce_shi_sun_shi < zui_jia_sun_shi:
                cun_dang_lu_jing = os.path.join(can_shu_biao.work_dir, "model.pt")
                print(f"测试损失 {ce_shi_sun_shi} 创新低，保存模型到 {cun_dang_lu_jing}")
                torch.save(mo_xing.state_dict(), cun_dang_lu_jing)
                zui_jia_sun_shi = ce_shi_sun_shi

        # 每 200 步生成一批样本看看效果
        if bu_shu > 0 and bu_shu % 200 == 0:
            da_yin_yang_ben(shu_liang=10)

        bu_shu += 1
        # 到达指定步数就停止
        if can_shu_biao.max_steps >= 0 and bu_shu >= can_shu_biao.max_steps:
            break