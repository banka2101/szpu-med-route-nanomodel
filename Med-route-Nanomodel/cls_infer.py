# ============================================================
# 文件 4: cls_infer.py
# 作用：加载训练好的模型，对新句子做预测（路由到哪个科室）。
#       这就是你最终接到 RAG 系统前的那一步。
#
# 运行方式：
#   # 测单句
#   python3 cls_infer.py --text "我膝盖摔肿了"
#   # 交互模式（不带 --text，循环输入）
#   python3 cls_infer.py
# ============================================================

import argparse
import torch
import torch.nn.functional as F

from cls_model import GPTClassifier   # 导入模型结构


# ------------------------------------------------------------
# 加载训练好的模型
# ------------------------------------------------------------
def load_model(ckpt_path, device='cpu'):
    # weights_only=False 因为我们存了 config 等非张量对象
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt['config']                # 取出当初训练时的模型配置
    model = GPTClassifier(config).to(device)
    model.load_state_dict(ckpt['model'])   # 把训练好的参数装回模型
    model.eval()                           # 评估模式（关闭 dropout）
    # 还要返回字表、标签表、句子长度，预测时要用
    return model, ckpt['stoi'], ckpt['id2label'], ckpt['block_size']


# ------------------------------------------------------------
# 把一句话变成数字（必须和 prepare 时的处理完全一致！）
# ------------------------------------------------------------
def encode(text, stoi, block_size):
    ids = [stoi.get(c, 0) for c in text][:block_size]
    pad_len = block_size - len(ids)
    ids = [0] * pad_len + ids   # 同样是左侧补齐
    return ids


# ------------------------------------------------------------
# 核心：路由函数。输入一句话，返回（类别, 有多少把握）
# ------------------------------------------------------------
def route(text, model, stoi, id2label, block_size, device='cpu'):
    ids = encode(text, stoi, block_size)
    # 包成一个 batch（即使只有一句话，也要变成 (1, block_size) 的形状）
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, _ = model(x)                       # 前向，得到得分
        probs = F.softmax(logits, dim=-1)[0]       # 得分→概率（取出这一句）
        pred_id = int(torch.argmax(probs).item())  # 概率最大的类别编号
    label = id2label[pred_id]                      # 编号→中文标签
    confidence = float(probs[pred_id])             # 把握程度（0~1）
    return label, confidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='cls_ckpt.pt', help='训练好的模型')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--text', type=str, default=None, help='要预测的句子；不填则进入交互模式')
    args = parser.parse_args()

    model, stoi, id2label, block_size = load_model(args.ckpt, args.device)
    print(f"可分类别: {list(id2label.values())}")

    if args.text:
        # ---- 单句模式 ----
        label, conf = route(args.text, model, stoi, id2label, block_size, args.device)
        print(f"输入: {args.text}")
        print(f"路由到 → 【{label}】 (置信度 {conf:.1%})")

        # ===== 在这里接你的 RAG 系统 =====
        # if label == "骨科":
        #     答案 = 去骨科知识库检索(args.text)
        # elif label == "内科":
        #     答案 = 去内科知识库检索(args.text)
        # ...
    else:
        # ---- 交互模式 ----
        print("输入问句测试（输入 q 退出）：")
        while True:
            text = input("\n> ").strip()
            if text in ('q', 'quit', 'exit'):
                break
            if not text:
                continue
            label, conf = route(text, model, stoi, id2label, block_size, args.device)
            print(f"  → 【{label}】 (置信度 {conf:.1%})")


if __name__ == '__main__':
    main()