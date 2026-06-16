# ================================================================================
#        简单神经网络 - 零基础友好版
# ================================================================================
#
# 【一句话概括】
# 这个文件负责"搭积木"：把 easy_engine 里的 EasyValue 组装成神经网络。
# easy_engine 解决"怎么求导"，这个文件解决"怎么搭网络"。
#
# 【生活比喻：投票决策系统】
# 判断一张图片是不是猫，找3个专家投票：
#   专家1（看耳朵）投 0.9 分，专家2（看颜色）投 0.3 分，专家3（看体型）投 0.6 分
# 但每个专家"话语权"不同（这就是权重 w）：
#   话语权1 × 0.9 + 话语权2 × 0.3 + 话语权3 × 0.6 + 基础倾向（偏置 b）= 最终得分
# 这个"加权求和 + 基础倾向"的过程，就是一个神经元干的事。
#
# 【四块积木，从小到大】
#   JiChu（基础模块）    ：所有积木的父类，提供"清空梯度"和"列出参数"
#   ShenJingYuan（神经元）：一组权重 + 一个偏置 + 一个激活函数
#   CengJi（层）         ：一排神经元，接收相同的输入
#   DuoCengWang（多层网络）：多个层串起来，上一层的输出是下一层的输入
# ================================================================================

import random
from easy_engine import EasyValue   # 导入"会自动求导的数字"


# ================================================================================
# 第一块：JiChu（基础模块）
# ================================================================================
# 这是一个"父类"，下面三个积木都继承它。
# 继承就像："动物"是父类，"狗"是子类，狗自动拥有动物的所有功能。
# ================================================================================
class JiChu:

    def qing_kong_ti_du(self):
        """
        把所有参数的梯度清零。

        【为什么要清零？】
        反向传播时梯度是"累加"的（ti_du += ...）。
        不清零的话，上一轮的梯度会残留下来，和这一轮混在一起，
        网络就会越练越乱。所以每轮训练前必须先清零，
        就像做数学题前先把草稿纸擦干净。
        """
        for can_shu in self.lie_chu_can_shu():
            can_shu.ti_du = 0

    def lie_chu_can_shu(self):
        """
        列出这个模块里所有"可以被训练的参数"。

        父类默认返回空列表，子类会覆盖它返回真正的参数。
        有了这个方法，不管网络多深，一句 wang_luo.lie_chu_can_shu()
        就能拿到所有参数，统一更新。
        """
        return []


# ================================================================================
# 第二块：ShenJingYuan（单个神经元）
# ================================================================================
# 一个神经元做的事，用数学式子表示就是：
#   输出 = ReLU(w1×x1 + w2×x2 + w3×x3 + b)
#
# 图示（3个输入的神经元）：
#   x1 ──×w1──┐
#   x2 ──×w2──┼──→ 求和 ──→ + 偏置b ──→ ReLU ──→ 输出
#   x3 ──×w3──┘
# ================================================================================
class ShenJingYuan(JiChu):

    def __init__(self, ru_shu, fei_xian_xing=True):
        """
        创建一个神经元。

        参数：
          ru_shu        : 输入数，这个神经元接收几个输入
          fei_xian_xing : True  = 输出前过一遍 ReLU（隐藏层用）
                          False = 直接输出（最后一层用，因为结果可能是负数）
        """
        # 权重：每个输入配一个权重，在 -1 到 1 之间随机取值。
        # 为什么随机？如果所有权重都一样，每个神经元学到的东西就一模一样，
        # 多个神经元就没意义了。
        self.quan_zhong = []
        for _ in range(ru_shu):
            self.quan_zhong.append(EasyValue(random.uniform(-1, 1)))

        # 偏置：一个神经元只有一个，初始为 0。
        # 作用是给神经元一个"基础倾向"，
        # 就像"这个专家天生乐观，打分时会多加一点"。
        self.pian_zhi = EasyValue(0)

        self.fei_xian_xing = fei_xian_xing

    def __call__(self, x):
        """
        让神经元算一次：输入一组数字 x，输出一个数字。

        Python 里 __call__ 让对象能像函数一样用：
          yuan = ShenJingYuan(3)
          jie_guo = yuan([0.5, 0.3, 0.8])   ← 这里就触发了 __call__
        """
        # 第一步：加权求和。
        # 从偏置出发，把每个"权重×输入"一项一项加上去。
        # 算完就是：pian_zhi + w1*x1 + w2*x2 + w3*x3
        zong_he = self.pian_zhi
        for wi, xi in zip(self.quan_zhong, x):   # zip 把权重和输入一对一配好
            zong_he = zong_he + wi * xi

        # 第二步：激活函数。
        # ReLU 的规则很简单：负数变 0，正数不变。
        if self.fei_xian_xing:
            return zong_he.relu()
        else:
            return zong_he

    def lie_chu_can_shu(self):
        """返回这个神经元的所有参数：全部权重 + 偏置。"""
        return self.quan_zhong + [self.pian_zhi]

    def __repr__(self):
        """打印神经元时显示的内容，方便调试。"""
        lei_xing = 'ReLU激活' if self.fei_xian_xing else '线性'
        return f"{lei_xing}神经元(输入数量={len(self.quan_zhong)})"


# ================================================================================
# 第三块：CengJi（一层神经元）
# ================================================================================
# 一层 = 一排神经元，每个神经元都接收相同的输入，各自输出一个数。
#
# 图示（3个输入，4个神经元的一层）：
#                ┌→ 神经元1 → 输出1
# 输入[x1,x2,x3] ─┼→ 神经元2 → 输出2
#                ├→ 神经元3 → 输出3
#                └→ 神经元4 → 输出4
#
# 为什么一层要有多个神经元？因为每个神经元学的是不同的"特征"：
# 神经元1 可能学会看"耳朵"，神经元2 可能学会看"颜色"……
# ================================================================================
class CengJi(JiChu):

    def __init__(self, ru_shu, chu_shu, fei_xian_xing=True):
        """
        创建一层神经元。

        参数：
          ru_shu        : 输入数，每个神经元接收几个输入
          chu_shu       : 输出数，这层有几个神经元（也就是输出几个数字）
          fei_xian_xing : 直接转交给每个神经元
        """
        # 创建 chu_shu 个一模一样规格的神经元
        self.shen_jing_yuan_lie_biao = []
        for _ in range(chu_shu):
            self.shen_jing_yuan_lie_biao.append(
                ShenJingYuan(ru_shu, fei_xian_xing)
            )

    def __call__(self, x):
        """把输入 x 喂给每个神经元，收集所有输出。"""
        suo_you_shu_chu = []
        for yuan in self.shen_jing_yuan_lie_biao:
            suo_you_shu_chu.append(yuan(x))

        # 只有1个神经元时直接返回那个数字（不用列表包着），
        # 这样最后一层（通常只输出1个数）用起来更方便。
        if len(suo_you_shu_chu) == 1:
            return suo_you_shu_chu[0]
        return suo_you_shu_chu

    def lie_chu_can_shu(self):
        """收集这层所有神经元的所有参数，拼成一个大列表。"""
        jie_guo = []
        for yuan in self.shen_jing_yuan_lie_biao:          # 先遍历每个神经元
            for can_shu in yuan.lie_chu_can_shu():          # 再遍历它的每个参数
                jie_guo.append(can_shu)
        return jie_guo

    def __repr__(self):
        nei_rong = ', '.join(str(yuan) for yuan in self.shen_jing_yuan_lie_biao)
        return f"网络层 [{nei_rong}]"


# ================================================================================
# 第四块：DuoCengWang（多层网络，也叫 MLP）
# ================================================================================
# MLP = Multi-Layer Perceptron = 多层感知机。
# 就是把多个 CengJi 串起来，上一层的输出直接作为下一层的输入。
#
# 图示（输入3个数，两个隐藏层各4个神经元，输出1个数）：
#   [x1]      [○○○○]     [○○○○]      [○]
#   [x2]  →   [○○○○]  →  [○○○○]  →   [○]  →  最终输出
#   [x3]      4个神经元    4个神经元    1个神经元
#
# 创建方式：DuoCengWang(3, [4, 4, 1])
#   3       : 输入3个数
#   [4,4,1] : 第一层4个神经元，第二层4个，最后一层1个
# ================================================================================
class DuoCengWang(JiChu):

    def __init__(self, ru_shu, ge_ceng_chu_shu):
        """
        创建整个多层网络。

        参数：
          ru_shu          : 网络接收几个输入
          ge_ceng_chu_shu : 列表，每个元素是对应层的神经元数量
        """
        # 把输入数量和各层输出数量拼成完整的"尺寸列表"。
        # 比如 ru_shu=3, ge_ceng_chu_shu=[4,4,1] → chi_cun=[3,4,4,1]
        # 意思是：3→4→4→1，每个箭头就是一层。
        chi_cun = [ru_shu] + ge_ceng_chu_shu

        # 逐层创建。规则：最后一层不加 ReLU（因为最终输出可能要是负数），
        # 其他层都加 ReLU。
        self.ceng_ji_lie_biao = []
        for i in range(len(ge_ceng_chu_shu)):
            shi_zui_hou_yi_ceng = (i == len(ge_ceng_chu_shu) - 1)
            xin_ceng = CengJi(
                chi_cun[i],                              # 这层的输入数量
                chi_cun[i + 1],                          # 这层的神经元个数
                fei_xian_xing=not shi_zui_hou_yi_ceng    # 最后一层不加ReLU
            )
            self.ceng_ji_lie_biao.append(xin_ceng)

    def __call__(self, x):
        """
        前向传播：让输入 x 依次经过每一层。
        上一层的输出自动变成下一层的输入。
        """
        for ceng in self.ceng_ji_lie_biao:
            x = ceng(x)        # x 不断被更新为这一层的输出
        return x               # 最后的 x 就是整个网络的输出

    def lie_chu_can_shu(self):
        """收集整个网络所有层的所有参数，拼成一个大列表。"""
        jie_guo = []
        for ceng in self.ceng_ji_lie_biao:
            for can_shu in ceng.lie_chu_can_shu():
                jie_guo.append(can_shu)
        return jie_guo

    def __repr__(self):
        nei_rong = ', '.join(str(ceng) for ceng in self.ceng_ji_lie_biao)
        return f"多层网络 [{nei_rong}]"


# ================================================================================
# 测试一下，看看能不能跑起来
# ================================================================================
if __name__ == '__main__':

    print("--- 测试1：创建一个神经元 ---")
    yuan = ShenJingYuan(3)              # 接收3个输入的神经元
    shu_ru = [EasyValue(1.0), EasyValue(2.0), EasyValue(3.0)]
    jie_guo = yuan(shu_ru)
    print(f"  神经元：{yuan}")
    print(f"  输出：{jie_guo}")

    print("\n--- 测试2：创建一层（3个输入，4个神经元）---")
    ceng = CengJi(3, 4)
    shu_ru = [1.0, 2.0, 3.0]           # 普通数字也行，运算时会自动处理
    shu_chu = ceng(shu_ru)
    print(f"  层：{ceng}")
    print(f"  输出（4个数）：{shu_chu}")

    print("\n--- 测试3：创建完整网络并训练 ---")
    # 网络结构：接收3个输入，经过[4,4]两个隐藏层，输出1个数
    wang_luo = DuoCengWang(3, [4, 4, 1])
    print(f"  参数总数：{len(wang_luo.lie_chu_can_shu())} 个")

    # 准备一批训练数据（4组输入，4个期望输出）
    X = [
        [2.0,  3.0, -1.0],
        [3.0, -1.0,  0.5],
        [0.5,  1.0,  1.0],
        [1.0,  1.0, -1.0],
    ]
    Y_qi_wang = [1.0, -1.0, -1.0, 1.0]   # 期望输出

    # 训练20步
    for bu_shu in range(20):

        # 第一步：前向传播，算出预测值
        Y_yu_ce = [wang_luo(x) for x in X]

        # 第二步：计算损失（预测值和真实值的差距，越小越好）
        # 用"均方误差"：把每个(预测-真实)²加起来
        sun_shi = sum((yu_ce - qi_wang) ** 2
                      for yu_ce, qi_wang in zip(Y_yu_ce, Y_qi_wang))

        # 第三步：清空上一轮的梯度（不清空会累加，方向就错了）
        wang_luo.qing_kong_ti_du()

        # 第四步：反向传播，自动算出所有参数的梯度
        sun_shi.backward()

        # 第五步：梯度下降，更新所有参数
        # 梯度指向"让损失变大"的方向，所以要反着走（减去梯度）
        # 学习率 0.05 控制每步迈多大
        xue_xi_lv = 0.05
        for can_shu in wang_luo.lie_chu_can_shu():
            can_shu.shu_ju -= xue_xi_lv * can_shu.ti_du

        print(f"  第{bu_shu + 1:2d}步，损失 = {sun_shi.shu_ju:.4f}")


# ================================================================================
# 【兼容段】让本文件能和原版 micrograd 的测试脚本一起用
# ================================================================================
# 下面的代码不影响上面的学习内容，只是给类和方法起"英文别名"，
# 让原版风格的代码（Neuron / Layer / MLP / parameters / zero_grad）也能跑。
# ================================================================================

# --- 方法别名：补上原版的方法名 ---
JiChu.parameters = JiChu.lie_chu_can_shu
JiChu.zero_grad = JiChu.qing_kong_ti_du

# --- 类别名 + 参数名转换 ---
class Neuron(ShenJingYuan):
    def __init__(self, nin, nonlin=True):
        super().__init__(nin, fei_xian_xing=nonlin)

class Layer(CengJi):
    def __init__(self, nin, nout, nonlin=True, **kwargs):
        # 原版用 nonlin 关键字，这里转成 fei_xian_xing
        super().__init__(nin, nout, fei_xian_xing=nonlin)

class MLP(DuoCengWang):
    def __init__(self, nin, nouts):
        super().__init__(nin, nouts)

# --- 子类也补上 parameters / zero_grad（继承自 JiChu 的别名已自动生效）---
Module = JiChu   # 原版的基类名叫 Module
