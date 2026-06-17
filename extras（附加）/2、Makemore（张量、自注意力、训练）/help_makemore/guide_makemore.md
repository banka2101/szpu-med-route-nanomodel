
### 📄 Make More (拼音注释版)：字符级语言模型训练框架

这是一个最小化的字符级语言模型训练框架，移植自 Karpathy 的 makemore。它接收一个文本文件（每行一个词，如几万个英文人名），学习这些词的拼写规律，然后生成长得很像、却从没见过的新词。

#### 模型概览
包含 6 种模型，按“进化顺序”排列，可直观对比语言模型的发展史：

| 模型 | 思路 | 特点 |
| :--- | :--- | :--- |
| **bigram** | 只看前 1 个字符，查表 | 最原始 |
| **mlp** | 看前 N 个字符拼接过全连接 | Bengio 2003 论文 |
| **rnn** | 用记忆逐个读字符 | 不限长度，记性差 |
| **gru** | 带门控的 RNN | 记性更好 |
| **bow** | 把前文简单平均 | 注意力的雏形 |
| **transformer** | 注意力：选择性地看前文 | GPT-2 同款结构 |

---

### ⚙️ 环境要求与搭建

#### 环境要求
本项目在以下环境完成验证：
*   **云服务器**：阿里云 ECS
*   **GPU**：NVIDIA A10 (24GB) × 1
*   **配置**：8 vCPU / 30 GiB 内存
*   **操作系统**：Alibaba Cloud Linux 3.2104 LTS 64 位（预装 NVIDIA 驱动及 CUDA）

也可在 CPU 上运行（`--device cpu`），小模型完全够用，只是速度慢。

#### 环境搭建
1.  **验证 GPU 驱动与 CUDA**
    *   验证 GPU 驱动是否正常识别：
        ```bash
        nvidia-smi
        ```
        **预期**：能看到 A10 显卡信息，右上角显示 CUDA Version（如 12.x）。
    *   验证预装 CUDA 是否可用：
        ```bash
        nvcc --version
        ```
        **预期**：输出具体版本号。若提示找不到命令，说明选了纯净镜像，需重新换镜像。

2.  **创建并激活 Conda 环境**
    ```bash
    conda create -n makemore python=3.10 -y
    conda activate makemore
    ```

3.  **安装 PyTorch**
    ```bash
    pip install torch torchvision torchaudio \
        --index-url https://mirrors.cloud.aliyuncs.com/pypi/simple/ \
        --trusted-host mirrors.cloud.aliyuncs.com
    ```
    **注意**：`mirrors.cloud.aliyuncs.com` 是阿里云 ECS 内网专属开源镜像站，专供阿里云云服务器内网访问，免公网流量、速度更快。

4.  **安装其他依赖**
    还需要 tensorboard（训练日志）：
    ```bash
    pip install tensorboard
    ```

5.  **验证 PyTorch 能否调用 GPU**
    ```bash
    python -c "import torch; print(f'PyTorch: {torch.version}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"
    ```
    **预期输出**：`CUDA Available: True`，`Device: NVIDIA A10`。如果为 False，立即停止！检查 CUDA 版本与 pip 安装命令是否匹配。

---

### 📂 准备文件

1.  创建工作目录并进入：
    ```bash
    mkdir -p /mnt/data/makemore
    cd /mnt/data/makemore
    conda activate makemore
    ```
2.  将以下两个文件上传到该目录：
    *   `easy_makemore.py`：主程序
    *   `names.txt`：训练数据（每行一个词，例如英文人名）
    **注意**：默认输入文件名为 `names.txt`，可用 `-i` 参数指定其他文件。

---

### 🏃 训练模型

#### 训练一个 Transformer（推荐）
```bash
python easy_makemore.py \
    --type transformer \
    --work-dir out_transformer \
    --device cuda \
    --max-steps 5000 \
    --n-layer 4 \
    --n-head 4 \
    --n-embd 64 \
    --batch-size 32 \
    --learning-rate 5e-4
```

#### 训练一个 RNN
```bash
python easy_makemore.py \
    --type rnn \
    --work-dir out_rnn \
    --device cuda \
    --max-steps 5000 \
    --n-embd 64 \
    --n-embd2 64 \
    --batch-size 32 \
    --learning-rate 5e-4
```
把 `--type` 换成 `bigram` / `mlp` / `gru` / `bow` 即可训练其他模型。

#### 训练过程说明
训练过程中会：
*   每 10 步打印损失。
*   每 500 步在训练集/测试集上评估，测试损失创新低时自动保存 `model.pt`。
*   每 200 步生成一批样本，区分“背下来的 / 巧合撞上的 / 全新创作的”。

---

### 🎲 采样生成

训练完成后，用 `--sample-only` 让模型生成新词（参数需与训练时一致，才能正确加载权重）。

#### Transformer 采样
```bash
python easy_makemore.py \
    --sample-only \
    --type transformer \
    --work-dir out_transformer \
    --device cuda \
    --n-layer 4 \
    --n-head 4 \
    --n-embd 64 \
    --top-k 10
```

#### RNN 采样
```bash
python easy_makemore.py \
    --sample-only \
    --type rnn \
    --work-dir out_rnn \
    --device cuda \
    --n-embd 64 \
    --n-embd2 64 \
    --top-k 10
```

---

### 📈 查看训练曲线（可选）

```bash
tensorboard --logdir out_transformer
```
浏览器打开 `http://localhost:6006` 查看 `Loss/train` 和 `Loss/test` 曲线。

---

### ⚙️ 命令行参数说明

| 参数 | 简写 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `--input-file` | `-i` | `names.txt` | 输入文件，一行一个词 |
| `--work-dir` | `-o` | `out` | 输出目录（存模型和日志） |
| `--type` | | `transformer` | 模型类型：`bigram`/`mlp`/`rnn`/`gru`/`bow`/`transformer` |
| `--device` | | `cpu` | 计算设备：`cpu`/`cuda`/`mps` |
| `--max-steps` | | `-1` | 最大训练步数，-1 为无限训练 |
| `--resume` | | | 从已有模型继续训练 |
| `--sample-only` | | | 只采样不训练 |
| `--top-k` | | `-1` | 采样 top-k 截断，-1 为不限制 |
| `--n-layer` | | `4` | Transformer 层数 |
| `--n-head` | | `4` | 注意力头数 |
| `--n-embd` | | `64` | 嵌入维度 |
| `--n-embd2` | | `64` | 第二嵌入维度（MLP/RNN 用） |
| `--batch-size` | `-b` | `32` | 批量大小 |
| `--learning-rate` | `-l` | `5e-4` | 学习率 |
| `--weight-decay` | `-w` | `0.01` | 权重衰减 |
| `--num-workers` | `-n` | `4` | 数据加载进程数 |
| `--seed` | | `3407` | 随机种子（保证可复现） |

---

### ❓ 常见问题

*   **Q：`CUDA Available: False`？**
    检查 CUDA 版本与 PyTorch 安装命令是否匹配，必要时重装对应 CUDA 版本的 PyTorch。

*   **Q：采样时报权重加载错误（size mismatch）？**
    采样的 `--n-layer` / `--n-head` / `--n-embd` / `--n-embd2` / `--type` 必须和训练时完全一致。

*   **Q：`--num-workers` 报错或卡住？**
    在某些环境下把它设为 0：`--num-workers 0`。

*   **Q：默认 `--max-steps -1` 会一直训练？**
    是的，会无限训练，按 `Ctrl+C` 手动停止，或设置一个具体步数（如 `--max-steps 5000`）。