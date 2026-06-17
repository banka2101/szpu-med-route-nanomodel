# ============================================================
# 文件 3: cls_train.py
# 作用：读取处理好的数据，训练模型，保存效果最好的那个。
#
# 运行方式（你给的命令）：
#   python3 cls_train.py --data cls_data.pkl --device cuda \
#       --max_iters 3000 --batch_size 64 --n_layer 6 --n_head 6 --n_embd 192
# ============================================================

import time
import pickle
import argparse

import torch

# 从文件1 导入我们定义的模型和配置
from cls_model import GPTClassifier, GPTConfig


# ------------------------------------------------------------
# 工具函数：随机抓一批数据出来
# ------------------------------------------------------------
def get_batch(data, batch_size, device):
    # 从数据里随机挑 batch_size 个样本的下标
    idx = torch.randint(0, len(data), (batch_size,))
    # x: 一批句子（数字序列），形状 (batch_size, block_size)
    x = torch.tensor([data[i][0] for i in idx], dtype=torch.long, device=device)
    # y: 这批句子对应的正确类别，形状 (batch_size,)
    y = torch.tensor([data[i][1] for i in idx], dtype=torch.long, device=device)
    return x, y


# ------------------------------------------------------------
# 工具函数：评估模型当前的损失和准确率
# @torch.no_grad() 表示这里面不计算梯度（省内存、加速），因为只是看效果不训练
# ------------------------------------------------------------
@torch.no_grad()
def evaluate(model, data, batch_size, device, eval_iters=20):
    model.eval()   # 切到"评估模式"（会关闭 dropout）
    losses = []
    correct = 0
    total = 0
    for _ in range(eval_iters):           # 多抽几批求平均，结果更稳
        x, y = get_batch(data, batch_size, device)
        logits, loss = model(x, y)
        losses.append(loss.item())
        pred = torch.argmax(logits, dim=-1)   # 取得分最高的类别作为预测
        correct += (pred == y).sum().item()   # 数对了几个
        total += y.size(0)
    model.train()  # 切回"训练模式"
    avg_loss = sum(losses) / len(losses)
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    # ===== 解析命令行参数 =====
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='cls_data.pkl', help='prepare 生成的数据')
    parser.add_argument('--out', type=str, default='cls_ckpt.pt', help='模型保存路径')
    parser.add_argument('--device', type=str, default='cpu', help='cpu 或 cuda(用GPU)')
    parser.add_argument('--max_iters', type=int, default=3000, help='总共训练多少步')
    parser.add_argument('--eval_interval', type=int, default=200, help='每隔多少步看一次效果')
    parser.add_argument('--batch_size', type=int, default=64, help='一次喂多少句话')
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率（步子迈多大）')
    parser.add_argument('--n_layer', type=int, default=6, help='层数')
    parser.add_argument('--n_head', type=int, default=6, help='注意力头数')
    parser.add_argument('--n_embd', type=int, default=192, help='向量维度')
    parser.add_argument('--dropout', type=float, default=0.1)
    args = parser.parse_args()

    device = args.device

    # ===== 第一步：加载数据 =====
    with open(args.data, 'rb') as f:
        meta = pickle.load(f)
    train_data = meta['train']
    val_data = meta['val']
    print(f"训练集 {len(train_data)} 条, 验证集 {len(val_data)} 条")
    print(f"类别: {meta['label2id']}")

    # ===== 第二步：根据数据信息搭建模型 =====
    # 注意 vocab_size / num_classes / block_size 都来自数据，不是瞎填的
    config = GPTConfig(
        block_size=meta['block_size'],
        vocab_size=meta['vocab_size'],
        num_classes=meta['num_classes'],
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPTClassifier(config).to(device)   # 把模型搬到 cpu 或 gpu 上

    # ===== 第三步：创建优化器 =====
    # 优化器负责根据损失，自动调整模型里的每个参数。AdamW 是最常用的。
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # ===== 第四步：训练主循环 =====
    best_val_acc = 0.0   # 记录见过的最好验证准确率
    t0 = time.time()

    for it in range(args.max_iters + 1):

        # ---- 每隔一段，评估一次并保存最佳模型 ----
        if it % args.eval_interval == 0:
            tr_loss, tr_acc = evaluate(model, train_data, args.batch_size, device)
            va_loss, va_acc = evaluate(model, val_data, args.batch_size, device)
            dt = time.time() - t0
            print(f"步 {it:5d} | 训练 loss {tr_loss:.3f} acc {tr_acc:.3f} "
                  f"| 验证 loss {va_loss:.3f} acc {va_acc:.3f} | 用时 {dt:.0f}s")

            # 只有当验证准确率创新高时，才保存模型（避免保存到过拟合的版本）
            if va_acc > best_val_acc:
                best_val_acc = va_acc
                torch.save({
                    'model': model.state_dict(),   # 模型参数
                    'config': config,              # 模型结构配置（推理时要用）
                    'stoi': meta['stoi'],          # 字→数字（推理时要用）
                    'id2label': meta['id2label'],  # 数字→标签（推理时要用）
                    'block_size': meta['block_size'],
                }, args.out)

        # 到达最后一步就结束
        if it == args.max_iters:
            break

        # ---- 真正的一步训练（核心四步）----
        x, y = get_batch(train_data, args.batch_size, device)  # 1. 取一批数据
        logits, loss = model(x, y)                # 2. 前向：算出预测和损失
        optimizer.zero_grad(set_to_none=True)     # 3a. 清空上一步的梯度
        loss.backward()                           # 3b. 反向传播：算出每个参数该怎么改
        optimizer.step()                          # 4. 更新参数

    print(f"\n训练完成！最佳验证准确率: {best_val_acc:.3f}")
    print(f"模型已保存到 {args.out}")


if __name__ == '__main__':
    main()
