# ============================================================
# 文件 2: cls_prepare.py
# 作用：把你的 JSONL 数据（文字）变成模型能吃的数字，存成 .pkl
#
# 这一步只需要做一次。做完会生成 cls_data.pkl，训练时直接读它。
#
# 运行方式：
#   python3 cls_prepare.py --input 你的数据.jsonl
# ============================================================

import json       # 读 JSONL
import pickle     # 保存处理结果
import random     # 打乱数据
import argparse   # 解析命令行参数


def main():
    # ===== 解析命令行参数 =====
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='你的 JSONL 数据文件路径')
    parser.add_argument('--out', type=str, default='cls_data.pkl', help='输出文件名')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='留多少比例做验证集')
    parser.add_argument('--block_size', type=int, default=32, help='句子最大长度')
    parser.add_argument('--seed', type=int, default=42, help='随机种子，保证每次划分一样')
    args = parser.parse_args()

    random.seed(args.seed)   # 固定随机种子，让结果可复现

    # ===== 第一步：读取 JSONL 数据 =====
    # 每行长这样: {"text": "我腰疼", "label": "骨科"}
    samples = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:           # 跳过空行
                continue
            obj = json.loads(line)            # JSON 文字 → Python 字典
            samples.append((obj['text'], obj['label']))   # 存成 (句子, 标签)
    print(f"读取样本数: {len(samples)}")

    # ===== 第二步：给每个"标签"编号 =====
    # 比如 中医→0, 内科→1, 骨科→2（按字典序排列，保证稳定）
    labels = sorted(set(lab for _, lab in samples))   # 取出所有不重复标签
    label2id = {lab: i for i, lab in enumerate(labels)}   # 标签 → 数字
    id2label = {i: lab for lab, i in label2id.items()}    # 数字 → 标签（反向表）
    print(f"类别 ({len(labels)} 个): {label2id}")

    # ===== 第三步：建立"字 → 编号"的字表（字符级 tokenizer）=====
    # 我们以"单个汉字"为单位，每个字一个编号。
    chars = set()
    for text, _ in samples:
        chars.update(text)        # 把每句话的每个字都收集进来（集合自动去重）
    chars = sorted(list(chars))   # 排序，保证稳定

    # 编号从 1 开始，把 0 号留给 <pad>（补齐用的占位符）
    stoi = {ch: i + 1 for i, ch in enumerate(chars)}   # string to int: 字 → 数字
    stoi['<pad>'] = 0                                   # 0 号专门补齐
    itos = {i: ch for ch, i in stoi.items()}           # int to string: 数字 → 字
    vocab_size = len(stoi)
    print(f"字表大小 vocab_size: {vocab_size}")

    # ===== 第四步：定义"把一句话变成数字序列"的函数 =====
    def encode(text):
        # 每个字查表换成数字，查不到的字（极少见）用 0 代替
        ids = [stoi.get(c, 0) for c in text]
        ids = ids[:args.block_size]      # 太长就截断

        # 【关键】左侧补齐：在左边补 0，让真实文字靠右
        # 为什么？因为模型在 forward 里取"最后一个位置"做分类，
        #   靠右补齐能保证最后一个位置一定是真实字符，而不是占位的 0。
        pad_len = args.block_size - len(ids)
        ids = [0] * pad_len + ids
        return ids

    # ===== 第五步：把所有样本都编码成数字 =====
    data = []
    for text, lab in samples:
        data.append((encode(text), label2id[lab]))   # (数字序列, 类别编号)

    # ===== 第六步：打乱并切分成 训练集 / 验证集 =====
    random.shuffle(data)
    n_val = int(len(data) * args.val_ratio)
    val_data = data[:n_val]       # 前一部分当验证集（用来检验模型，不参与训练）
    train_data = data[n_val:]     # 剩下的当训练集
    print(f"训练集: {len(train_data)} 条, 验证集: {len(val_data)} 条")

    # ===== 第七步：把所有东西打包保存 =====
    # 训练时需要：数据本身 + 字表 + 标签表 + 各种尺寸信息
    meta = {
        'train': train_data,
        'val': val_data,
        'stoi': stoi,                 # 字→数字（推理时也要用）
        'itos': itos,
        'vocab_size': vocab_size,     # 字表大小（建模型要用）
        'label2id': label2id,
        'id2label': id2label,         # 数字→标签（推理时把结果翻译回中文）
        'num_classes': len(labels),   # 分几类（建模型要用）
        'block_size': args.block_size,
    }
    with open(args.out, 'wb') as f:
        pickle.dump(meta, f)
    print(f"已保存到 {args.out}，接下来就可以训练了")


if __name__ == '__main__':
    main()