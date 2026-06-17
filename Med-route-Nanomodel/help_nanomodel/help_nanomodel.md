
### 📄 基于 NanoGPT 的文本分类系统：四个代码文件详解

#### 项目背景
本项目基于 Karpathy 的 NanoGPT 改造，将一个“猜下一个字”的语言模型，转变为一个“判断句子属于哪一类”的文本分类模型。文档由 Claude Opus 4.8 撰写。
*   **任务示例**：将病人的描述（如“我膝盖疼”）自动分类到对应的科室（如骨科/内科/中医）。
*   **适合人群**：深度学习初学者。

#### 先建立整体认知
整个项目由四个文件构成一条完整的流水线，缺一不可。

| 顺序 | 文件 | 作用 | 类比 |
| :--- | :--- | :--- | :--- |
| ① | `cls_prepare.py` | 把文字数据变成数字，存成 `.pkl` 文件 | 把食材洗好切好 |
| ② | `cls_model.py` | 定义模型的结构（此文件不直接运行） | 设计图纸 |
| ③ | `cls_train.py` | 用数据训练模型，并保存效果最好的模型 | 真正下锅炒菜 |
| ④ | `cls_infer.py` | 使用训练好的模型来预测新的句子 | 上桌品尝/使用 |

**运行顺序**：先 `prepare` → 再 `train`（`train` 会导入 `model`）→ 最后 `infer`。

**核心思想**：单个字 → 嵌入向量 → [注意力 → MLP] × N层 → 取一个向量 → 分类

---

### 🧱 `cls_model.py`：模型结构（最核心）
此文件只负责“定义”模型，不能直接运行，会被 `cls_train.py` 和 `cls_infer.py` 导入使用。

#### 1.1 配置类 `GPTConfig`
使用 `@dataclass` 将所有超参数集中管理，像填表格一样。
```python
@dataclass
class GPTConfig:
    block_size: int = 32      # 一句话最多包含多少个字
    vocab_size: int = 3000    # 字表大小（会被真实数据覆盖）
    num_classes: int = 3      # 分几类（如：骨科/内科/中医）
    n_layer: int = 6          # 堆叠几个 Block
    n_head: int = 6           # 多头注意力的头数
    n_embd: int = 192         # 每个字用多少个数字表示（必须能被 n_head 整除）
    dropout: float = 0.1      # 随机丢弃比例，防止过拟合
```
**初学者注意**：`n_embd % n_head == 0` 是必须满足的约束。例如 `192 ÷ 6 = 32`，每个头分到 32 维。

#### 1.2 模块一：`SelfAttention`（自注意力）
*   **作用**：让句子里的每个字去“看”其他所有字，决定该关注谁。例如在“我的膝盖很疼”中，“疼”会重点关注“膝盖”，模型就能判断是骨科问题。
*   **Q、K、V 三兄弟**：
    *   **Q (Query/查询)**：我想找什么。
    *   **K (Key/钥匙)**：我能提供什么。
    *   **V (Value/内容)**：我实际的内容是什么。
*   **代码实现**：
    ```python
    self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) # 一次算出 Q,K,V，所以是 3 倍宽
    self.c_proj = nn.Linear(config.n_embd, config.n_embd)     # 输出整理
    ```
*   **`forward` 七步走**：
    1.  **算出 Q、K、V**：`q, k, v = self.c_attn(x).split(self.n_embd, dim=2)`。`c_attn(x)` 输出 `(B,T,3C)`，再切成三份，每份 `(B,T,C)`。
    2.  **拆成多个头**：将 `(B,T,C)` 变为 `(B,头数,T,每头维度)`。多头能让模型从多个角度理解关系（如一个头看“症状-部位”，另一个看“时间-程度”）。
    3.  **算注意力分数（核心公式）**：`att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))`。Q 和 K 做点积，值越大越相关。除以 `sqrt(维度)` 是为了防止数值过大，使训练更稳定。
    4.  **关键改动**：**删掉了 NanoGPT 原版的“因果掩码”**。原版为预测下一个字，规定每个字只能看前面。本项目做分类，需要理解整句话，所以每个字能看到全句。
    5.  **Softmax**：`att = F.softmax(att, dim=-1)`。将分数变为概率占比，每一行加起来等于 1。
    6.  **加权求和 V**：`y = att @ v`。关注度高的字，其内容 V 在结果中占比就大。
    7.  **拼回并输出**：将多个头的结果拼接回去，经过 `c_proj` 整理和 `dropout` 后返回。

#### 1.3 模块二：`MLP`（前馈网络）
*   **作用**：注意力负责“字与字交流”，MLP 负责“对每个字单独深加工”。
*   **结构**：放大 → 激活 → 缩回。
    ```python
    self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd) # 放大4倍
    self.gelu = nn.GELU()                                   # 激活(非线性)
    self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd) # 缩回
    ```
    放大 4 倍是经验值，为模型提供更大的“思考空间”。

#### 1.4 模块三：`Block`（一个完整 Transformer 层）
*   **作用**：将注意力和 MLP 组装成一块标准积木。
    ```python
    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) # 归一化 → 注意力 → 残差连接
        x = x + self.mlp(self.ln_2(x))  # 归一化 → MLP → 残差连接
        return x
    ```
*   **关键技巧**：
    *   **残差连接 (`x = x + ...`)**：为信息提供一条“高速公路”，防止层数太深导致梯度消失。
    *   **LayerNorm 归一化**：将数据调整到稳定范围。此处采用 Pre-Norm（先归一化再进子模块）。

#### 1.5 模块四：`GPTClassifier`（最终模型）
将所有积木拼起来，分为输入、主体、输出三部分。
*   **输入部分**：字编号 → 向量
    *   `self.wte` (Word Token Embedding)：字向量表，将字编号转为向量。
    *   `self.wpe` (Word Position Embedding)：位置向量表，告诉模型“这是第几个字”。
*   **主体部分**：堆叠 N 个 Block。
    *   `self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])`
*   **输出部分**：分类头（与 NanoGPT 最大区别）
    *   `self.classifier = nn.Linear(config.n_embd, config.num_classes)`

| 对比项 | NanoGPT 原版 | 本项目 |
| :--- | :--- | :--- |
| **输出层** | `lm_head`: `n_embd` → `vocab_size` | `classifier`: `n_embd` → `num_classes` |
| **含义** | 预测下一个字（几千个选项） | 预测属于哪一类（如 3 个选项） |

*   **`forward` 流程**：
    1.  字向量和位置向量相加作为输入。
    2.  穿过所有 Block。
    3.  最后进行一次归一化 `self.ln_f(x)`。
    4.  **取最后一个位置的向量** `x[:, -1, :]` 代表整句。
    5.  通过分类头 `self.classifier(x)` 得到分类得分。
*   **为什么取最后一个位置**：因为在数据预处理时做了“左侧补齐”，真实文字靠右，所以最后一个位置一定是真实字符，且它经过注意力机制已经“看遍”全句，包含了整句信息。

---

### 🥣 `cls_prepare.py`：数据预处理
将 JSONL 文字数据转换为数字，并保存为 `cls_data.pkl`。此步骤只需运行一次。
`python3 cls_prepare.py --input 你的数据.jsonl`

输入数据每行格式：`{"text": "我腰疼", "label": "骨科"}`

**七个步骤**：
1.  **读 JSONL**：读取每一行，将 `(句子, 标签)` 存入列表。
2.  **给标签编号**：对所有标签排序并编号，如 `{'中医': 0, '内科': 1, '骨科': 2}`，并创建反向映射表。
3.  **建字表（字符级 tokenizer）**：收集所有出现过的字，创建 `字→数字` (`stoi`) 和 `数字→字` (`itos`) 的映射表。`<pad>` 补齐符的编号为 0。
4.  **编码函数（含关键的左侧补齐）**：
    ```python
    def encode(text):
        ids = [stoi.get(c, 0) for c in text]        # 查不到的字用0
        ids = ids[:args.block_size]                 # 太长则截断
        pad_len = args.block_size - len(ids)
        ids = [0] * pad_len + ids                   # ⚠️ 左侧补0，保证真实文字靠右
        return ids
    ```
5.  **编码所有样本**：将所有句子和标签转换为数字序列。
6.  **打乱并切分训练/验证集**：默认将 10% 的数据作为验证集。
7.  **打包保存**：将所有数据（训练集、验证集、字表、标签表等）打包成字典，用 `pickle` 保存。

---

### 🔥 `cls_train.py`：训练模型
`python3 cls_train.py --data cls_data.pkl --device cuda --max_iters 3000 ...`

*   **工具函数**：
    *   `get_batch(data, ...)`：从数据中随机取一个批次。
    *   `evaluate(model, ...)`：在验证集上评估模型。需使用 `@torch.no_grad()` 和 `model.eval()` 来节省内存和关闭 dropout。
*   **训练主流程**：
    1.  **搭建模型**：从数据文件中读取 `block_size`, `vocab_size`, `num_classes` 等参数来初始化模型。
    2.  **核心：一步训练的四步（深度学习的灵魂）**：
        ```python
        x, y = get_batch(train_data, ...) # 1. 取数据
        logits, loss = model(x, y)        # 2. 前向：算预测和损失
        optimizer.zero_grad(set_to_none=True) # 3a. 清空旧梯度
        loss.backward()                   # 3b. 反向传播：算梯度
        optimizer.step()                  # 4. 更新参数
        ```
    3.  **保存最佳模型**：只在验证集准确率创新高时保存模型，避免保存过拟合的版本。保存内容包括模型参数、配置、字表和标签表。

---

### 🔮 `cls_infer.py`：预测新句子
`python3 cls_infer.py --text "我膝盖摔肿了"`

*   **加载模型**：加载保存的 checkpoint，还原模型结构和训练好的参数，并切换到 `eval` 模式。
*   **编码**：必须使用和 `prepare` 阶段完全一致的编码和左侧补齐逻辑。
*   **核心：路由函数 `route`**：
    1.  将输入文本编码为数字序列。
    2.  用 `torch.no_grad()` 进行前向传播，得到 `logits`。
    3.  用 `F.softmax` 将 `logits` 转为概率。
    4.  取概率最大的类别 ID，再通过 `id2label` 映射回中文标签。
    5.  返回预测的标签和置信度。

**输出示例**：
`输入: 我膝盖摔肿了`
`路由到 → 【骨科】 (置信度 95.3%)`

---

### 📚 关键概念速查表

| 概念 | 一句话理解 |
| :--- | :--- |
| **Embedding (嵌入)** | 把字编号变成一串有意义的数字（向量）。 |
| **位置编码** | 告诉模型“这是第几个字”。 |
| **Q/K/V** | 查询/钥匙/内容，注意力的三要素。 |
| **Self-Attention** | 让每个字去看全句，决定关注谁。 |
| **多头** | 从多个角度同时理解关系。 |
| **残差连接** | `x = x + f(x)`，信息高速公路，防梯度消失。 |
| **LayerNorm** | 把数据调到稳定范围。 |
| **Dropout** | 随机丢弃，防止死记硬背（过拟合）。 |
| **Softmax** | 把分数变成加起来=1的概率。 |
| **交叉熵损失** | 衡量预测和正确答案差多少。 |
| **反向传播** | 算出每个参数该往哪个方向调。 |
| **优化器(AdamW)** | 根据梯度自动更新参数。 |

---

### 🔄 本项目与原版 NanoGPT 的三大区别

| 区别点 | NanoGPT（语言模型） | 本项目（分类器） |
| :--- | :--- | :--- |
| **因果掩码** | 有，每个字只能看前面 | 删掉，每个字能看全句 |
| **输出头** | `lm_head` → `vocab_size` (预测下一个字) | `classifier` → `num_classes` (预测类别) |
| **取哪个输出** | 每个位置都要预测下一个字 | 只取最后一个位置代表整句 |

---
