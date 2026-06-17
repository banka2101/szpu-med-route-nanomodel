# easy_makemore 阅读引导（零基础友好版）

> 本文档是 `easy_makemore.py` 的配套阅读指南，由 Claude Fable 5 撰写。
> 项目改写自 Karpathy 的 [makemore](https://github.com/karpathy/makemore)，
> 是 [easy_engine.py + easy_nn.py](即 micrograd 简化版) 之后的第二站。

---

## 目录

1. [这个项目是干什么的](#一这个项目是干什么的)
2. [你已经会什么，将要学什么](#二你已经会什么将要学什么)
3. [核心概念：语言模型就是字符接龙](#三核心概念语言模型就是字符接龙)
4. [六个模型的进化史（本项目的灵魂）](#四六个模型的进化史)
5. [推荐阅读顺序](#五推荐阅读顺序)
6. [怎么运行](#六怎么运行)
7. [实验建议](#七实验建议)
8. [术语对照表](#八术语对照表)

---

## 一、这个项目是干什么的

给程序一个文本文件（默认 `names.txt`，包含 32,033 个美国人名，一行一个），
它会学习这些名字的"拼写规律"，然后**编造出像模像样、但从没存在过的新名字**：

```
训练数据里有：  emma, olivia, ava, isabella ...
模型编出来的：  kamri, jaylee, dellia, marisol ...   ← 不在数据里，但很像名字
```

听起来像玩具？**ChatGPT 的原理和它完全一样**——只是 ChatGPT 接龙的不是
字母而是词块（token），数据不是 3 万个人名而是几乎整个互联网。

---

## 二、你已经会什么，将要学什么

### 2.1 前情提要：micrograd 阶段你已经掌握的

| 已掌握 | 对应代码 |
|---|---|
| 自动求导原理（计算图、链式法则、拓扑排序） | `easy_engine.py` |
| 神经元、层、MLP 的搭建 | `easy_nn.py` |
| 训练五步循环：前向→损失→清梯度→反向→更新 | `easy_nn.py` 训练部分 |

### 2.2 本章的四个新台阶

**台阶 1：从标量到张量（最重要的思维转变）**

```
micrograd:  一次算一个数字     w1*x1 + w2*x2 + ...（for 循环逐个算）
makemore:   一次算一整批矩阵   X @ W              （一行代码，GPU 并行）
```

你手写的 `fan_xiang_chuan_bo()` 在这里变成了 PyTorch 的 `sun_shi.backward()`——
**干的事完全一样**，只是计算图的节点从"数字"变成了"矩阵"。
因为你手写过它，所以对你来说它不再是黑箱。

**台阶 2：第一个语言模型**

之前的 MLP 做的是"输入 3 个数，输出 1 个数"的回归任务。
这次是**分类任务**：输入前文字符，输出"27 个字符里谁最可能是下一个"。
随之而来的新概念：嵌入（Embedding）、交叉熵损失、softmax、采样生成。

**台阶 3：六种架构的进化史**（详见第四节）

**台阶 4：正经的训练工程**

| 新概念 | 解决什么问题 |
|---|---|
| 训练集/测试集切分 | 模型是真学会了规律，还是死记硬背？ |
| 测试损失存档 | 只保存"在没见过的数据上表现最好"的模型 |
| AdamW 优化器 | 你手写的 `参数 -= 学习率×梯度` 的智能升级版 |
| 批量训练（batch） | 一次喂 32 个样本，更新方向更稳、速度更快 |
| 温度/top-k 采样 | 控制生成结果"保守"还是"大胆" |

---

## 三、核心概念：语言模型就是字符接龙

以训练数据中的 "emma" 为例，模型学的是 5 条规则：

```
看到 <开始>          → 下一个大概率是 e
看到 <开始>e         → 下一个大概率是 m
看到 <开始>em        → 下一个大概率是 m
看到 <开始>emm       → 下一个大概率是 a
看到 <开始>emma      → 下一个大概率是 <结束>
```

训练完成后，生成新名字 = 不断掷骰子接龙：

```
从 <开始> 出发 → 按概率抽中 k → 看到 "k" 抽中 a → 看到 "ka" 抽中 m
→ ... → 抽中 <结束>，停止 → 得到新名字 "kamri"
```

**所有 6 个模型的差别，只在一件事上：预测下一个字符时，"怎么利用前文"。**

---

## 四、六个模型的进化史

这是本项目的灵魂。6 个模型按"能力递增"排列，每一个都在修复前一个的缺陷：

```
Bigram → MLP → RNN → GRU → BoW → Transformer
```

| 模型 | 怎么用前文 | 缺陷 | 修复者 |
|---|---|---|---|
| **Bigram** | 只看前 1 个字符，查一张表 | 完全不知道更早的历史 | MLP |
| **MLP** | 看前 N 个字符，向量拼接后过全连接层 | N 固定死，改不了 | RNN |
| **RNN** | 逐个读字符，维护一个"记忆"向量 | 记忆容量固定，读得越长忘得越多 | GRU |
| **GRU** | RNN + 门控（精细控制记什么忘什么） | 治标不治本，且必须串行读，慢 | BoW/注意力 |
| **BoW** | 把前文所有字符**平均**一下 | 平均 = 大锅饭，分不清重点 | Transformer |
| **Transformer** | 注意力：每个字符**自己学**该重点看前文的谁 | （当前最优答案） | —— |

### 看懂 BoW → Transformer 这一步，你就看懂了注意力

Karpathy 故意把 BoW 的代码写得和注意力**几乎一模一样**，唯一区别：

```python
# BoW：注意力分数全是 0 → softmax 后人人权重相等（强制平均）
quan_zhong = torch.zeros((B, T, T))

# Transformer：注意力分数由 Q·K 算出来 → 权重是学出来的（按需分配）
zhu_yi_li = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
```

一句话：**注意力 = 会自己学权重的加权平均**。没有更多魔法了。

---

## 五、推荐阅读顺序

不要从头读到尾！按这个顺序：

```
第 1 步：ZiFuShuJuJi（数据集类）
         先搞懂"文字怎么变成数字"，这是地基。
         重点看 __getitem__ 里 x 和 y 错开一位的设计。

第 2 步：Bigram（约 20 行）
         最简单的模型，先把"训练→生成"的全流程跑通。

第 3 步：MLP
         和你写过的 easy_nn.py 对照着看，找相同点。

第 4 步：sheng_cheng（生成函数）+ 训练循环
         确认训练五步和 easy_nn.py 一一对应。

第 5 步：RNN → GRU
         体会"记忆"的概念和门控的动机。

第 6 步：BoW → YinGuoZhuYiLi（注意力）
         逐行对比两者代码，看懂注意力的本质。

第 7 步：TransformerKuai → Transformer
         残差连接 + LayerNorm + 多头，组装成完整 GPT-2 结构。
```

---

## 六、怎么运行

```bash
# 1. 准备环境
pip install torch tensorboard

# 2. 下载数据集（32033 个美国人名，来自 ssa.gov 公开数据）
#    从 https://github.com/karpathy/makemore 下载 names.txt 放到本目录

# 3. 训练（默认 Transformer，CPU 也能跑）
python easy_makemore.py -i names.txt -o out --max-steps 10000

# 4. 换不同的模型，亲自对比效果
python easy_makemore.py --type bigram      -o out_bigram --max-steps 5000
python easy_makemore.py --type mlp         -o out_mlp    --max-steps 5000
python easy_makemore.py --type rnn         -o out_rnn    --max-steps 5000
python easy_makemore.py --type transformer -o out_tf     --max-steps 5000

# 5. 用训练好的模型生成名字（不再训练）
python easy_makemore.py -o out --sample-only

# 6. 用 TensorBoard 看损失曲线
tensorboard --logdir out
```

有 NVIDIA 显卡加 `--device cuda`，Mac 加 `--device mps`。

---

## 七、实验建议

**实验 1：六模型大比武**
依次训练 6 种模型各 5000 步，记录各自最终的测试损失，排个名。
你会亲眼看到测试损失沿"进化顺序"递减——这就是架构进步的意义。

**实验 2：观察过拟合**
把 `--n-embd` 调大到 256、层数调到 8，训练足够久。
观察训练损失还在降，但测试损失开始**回升**——这就是过拟合，
模型开始死记硬背训练集了。这是 micrograd 阶段看不到的现象。

**实验 3：温度实验**
在 `sheng_cheng` 调用处修改 `wen_du` 参数：
- `wen_du=0.5`：生成的名字保守、常见
- `wen_du=1.5`：生成的名字狂野、经常不像名字
体会 ChatGPT 的 temperature 参数到底在控制什么。

**实验 4：换数据集**
`names.txt` 换成任何"一行一个词"的文件：宝可梦名字、中文姓名拼音、
公司名……模型会学会任何你喂给它的拼写规律。这是最好玩的实验。

---

## 八、术语对照表

### 8.1 配置与通用名称

| 拼音名 | 原版英文名 | 含义 |
|---|---|---|
| `MoXingPeiZhi` | `ModelConfig` | 模型配置 |
| `kuai_chang_du` | `block_size` | 模型一次最多看多少个字符 |
| `ci_biao_da_xiao` | `vocab_size` | 一共有多少种字符 |
| `qian_ru_wei_du` | `n_embd` | 每个字符用多少个数字表示 |
| `qian_ru_wei_du2` | `n_embd2` | 第二嵌入维度（RNN/MLP 内部用） |
| `ceng_shu` | `n_layer` | Transformer 堆叠几层 |
| `tou_shu` | `n_head` | 注意力分成几个头 |

### 8.2 模型类名

| 拼音名 | 原版英文名 | 含义 |
|---|---|---|
| `GELUJiHuo` | `NewGELU` | 平滑版 ReLU 激活函数 |
| `YinGuoZhuYiLi` | `CausalSelfAttention` | 因果自注意力 |
| `TransformerKuai` | `Block` | Transformer 块 |
| `YinGuoCiDai` | `CausalBoW` | 因果词袋（强制平均版注意力） |
| `CiDaiKuai` | `BoWBlock` | 词袋块 |
| `RNNDanYuan` | `RNNCell` | RNN 单元 |
| `GRUDanYuan` | `GRUCell` | GRU 单元 |
| `Transformer` / `Bigram` / `MLP` / `RNN` / `BoW` | （相同） | 模型类名和原版一致 |

### 8.3 函数与数据处理

| 拼音名 | 原版英文名 | 含义 |
|---|---|---|
| `sheng_cheng` | `generate` | 字符接龙生成 |
| `ping_gu` | `evaluate` | 在数据集上算平均损失（考试） |
| `da_yin_yang_ben` | `print_samples` | 采样并分类打印 |
| `ZiFuShuJuJi` | `CharDataset` | 字符数据集 |
| `bian_ma` / `jie_ma` | `encode` / `decode` | 字符↔编号互转 |
| `bao_han` | `contains` | 判断词是否在数据集中 |
| `chuang_jian_shu_ju_ji` | `create_datasets` | 读文件、切分训练/测试集 |
| `WuXianJiaZaiQi` | `InfiniteDataLoader` | 无限数据加载器 |
| `qu_xia_yi_pi` | `next` | 取下一批数据 |

### 8.4 核心新概念速查

| 术语 | 一句话解释 |
|---|---|
| 嵌入（Embedding） | 一张查询表：字符编号 → 一串数字（向量），向量是学出来的 |
| logits（打分表） | 模型对每个候选字符的"原始得分"，还没变成概率 |
| softmax | 把一排得分变成一排概率（全部为正、加起来等于 1） |
| 交叉熵损失 | 分类版的损失函数：正确答案的概率越高，损失越小 |
| 因果掩码 | 下三角挡板，保证预测时只能看前文、不能偷看答案 |
| Q / K / V | 注意力三件套：我想找什么 / 我能提供什么 / 我实际带的内容 |
| 多头注意力 | 把注意力分成几组并行，各组关注不同角度的规律 |
| 残差连接 | `x = x + 新结果`，给信息和梯度留一条直通的高速公路 |
| LayerNorm | 每道工序前把数值拉回标准范围，防止逐层放大失控 |
| 温度（temperature） | 采样随机度旋钮：低=保守，高=狂野 |
| top-k 采样 | 只在得分前 k 名里抽签，防止抽到太离谱的字符 |
| 过拟合 | 训练损失还在降、测试损失却回升=开始死记硬背了 |
| AdamW | 智能版梯度下降：带惯性 + 每个参数自适应步长 |

---

## 九、学完之后去哪里

```
① micrograd（easy_engine + easy_nn）   ✅ 已完成
       搞懂了：自动求导 + 训练循环的本质
       ↓
② makemore（本项目）                   ← 你在这里
       搞懂了：张量化、语言模型、架构进化、注意力
       ↓
③ GPT from scratch（Karpathy 视频）
       用更大的数据（莎士比亚全集）从零写一个 GPT
       内容和本项目的 Transformer 部分高度重合，会很轻松
       ↓
④ NanoGPT
       工程化完整版：混合精度、分布式训练、断点续训
       此时你已经能看懂它的每一行代码
```

一个检验标准：当你能向别人讲清楚
**"为什么 BoW 加上可学习的权重就变成了注意力"**，
你就真正可以进入下一站了。

---

*文档版本：v1.0*
*配套代码：`easy_makemore.py`*
*数据集：`names.txt`（32,033 个美国人名，下载自 [karpathy/makemore](https://github.com/karpathy/makemore)）*
*参考项目：[karpathy/makemore](https://github.com/karpathy/makemore)*