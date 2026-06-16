# ================================================================================
#        简单自动求导引擎 - 零基础友好版
# ================================================================================
#
# 【一句话概括】
# 这个工具让数字"学会记日记"：每算一步都记下来，最后能自动倒推出
# "每个数字对最终结果的影响有多大"（也就是梯度/导数）。
#
# 【生活比喻】
# 想象你做菜：放了盐、放了糖、炒了一下、又加了醋……
# 最后菜太咸了，你想知道"是盐放多了还是糖放多了"。
# 这个工具就是帮你倒推"每种调料对咸度的贡献"的。
#
# 【基本用法】
#   a = EasyValue(2.0)    # 创建一个"会记账的数字"
#   b = EasyValue(3.0)
#   c = a * b + 1         # 正常写数学表达式就行
#   c.backward()           # 一键倒推所有梯度
#   print(a.ti_du)         # 查看 a 对 c 的影响程度
# ================================================================================


class EasyValue:
    """
    EasyValue = 一个"会记账的数字"

    它比普通数字多记三样东西：
      1. 我是从哪些数字算出来的？（zi_jie_dian，子节点）
      2. 用的什么运算？加法？乘法？（yun_suan，运算）
      3. 别人对我求的导数是多少？（ti_du，梯度）
    """

    # ============================================================
    # 创建一个 EasyValue 对象
    # ============================================================
    # 参数说明：
    #   shu_ju      : 这个节点存的数字，比如 3.14
    #   zi_jie_dian : 这个数字是从哪些数字算出来的（比如 c = a+b，那 c 的子节点就是 a 和 b）
    #   yun_suan    : 用什么运算算出来的，比如 'add'、'mul'，方便后面倒推时查看
    # ============================================================
    def __init__(self, shu_ju, zi_jie_dian=(), yun_suan=''):
        self.shu_ju = shu_ju              # 存的数字
        self.ti_du = 0.0                  # 梯度（导数），一开始是 0，等 backward 之后才有值
        self.zi_jie_dian = tuple(zi_jie_dian)  # 用 tuple 保存，不用 set！
                                               # 因为 set 会去重，比如 x*x 会把两个 x 合并成一个
                                               # 那后面拆的时候就崩了
        self.yun_suan = yun_suan          # 记录运算类型


    # ============================================================
    # 加法：self + qi_ta
    # ============================================================
    # 数学原理：
    #   如果 c = a + b，那么：
    #     a 变一点点，c 也变一点点（影响程度 = 1）
    #     b 变一点点，c 也变一点点（影响程度 = 1）
    #   所以加法的导数就是 1，很简单。
    # ============================================================
    def jia(self, qi_ta):
        # 如果对方是普通数字（比如 a + 5），先包装成 EasyValue
        # 不然后面访问 qi_ta.shu_ju 会报错
        if not isinstance(qi_ta, EasyValue):
            qi_ta = EasyValue(qi_ta)
        jie_guo = EasyValue(self.shu_ju + qi_ta.shu_ju, (self, qi_ta), 'add')
        return jie_guo


    # ============================================================
    # 乘法：self × qi_ta
    # ============================================================
    # 数学原理：
    #   如果 c = a × b，那么：
    #     a 对 c 的影响程度 = b（a 增加 1，c 增加 b）
    #     b 对 c 的影响程度 = a（b 增加 1，c 增加 a）
    #   口诀：求谁的导数，就乘以"对方的值"。
    #
    #   举例：c = 3 × 5 = 15
    #     3 增加到 4 → c = 4 × 5 = 20，增加了 5 → 导数是 5（对方的值）
    #     5 增加到 6 → c = 3 × 6 = 18，增加了 3 → 导数是 3（对方的值）
    # ============================================================
    def cheng(self, qi_ta):
        if not isinstance(qi_ta, EasyValue):
            qi_ta = EasyValue(qi_ta)
        jie_guo = EasyValue(self.shu_ju * qi_ta.shu_ju, (self, qi_ta), 'mul')
        return jie_guo


    # ============================================================
    # 幂运算：self 的 ci_shu 次方
    # ============================================================
    # 数学原理：
    #   如果 y = x 的 c 次方（c 是固定的数字），那么：
    #     导数 = c × x 的 (c-1) 次方
    #
    #   举例：y = x²
    #     导数 = 2x（x 在 3 的时候，导数 = 6，意思是 x 动一点，y 动 6 倍）
    #
    #   举例：y = x³
    #     导数 = 3x²（x 在 2 的时候，导数 = 12）
    #
    # 注意：ci_shu 只能是普通数字（整数或小数），不能是另一个 EasyValue
    # ============================================================
    def mi(self, ci_shu):
        assert isinstance(ci_shu, (int, float)), "现在只支持整数、小数次方"
        # 运算类型记成 'pow_2' 或 'pow_-1' 这种格式，后面倒推时能取出次数
        jie_guo = EasyValue(self.shu_ju ** ci_shu, (self,), f'pow_{ci_shu}')
        return jie_guo


    # ============================================================
    # ReLU 激活函数
    # ============================================================
    # 【ReLU 是什么？】
    #   公式超简单：正数原样输出，负数变成 0
    #     ReLU(3)  = 3
    #     ReLU(-5) = 0
    #     ReLU(0)  = 0
    #
    #   图像长这样（左边躺平，右边斜上）：
    #       y
    #       │      ╱
    #       │     ╱
    #       │    ╱
    #       │___╱________ x
    #          0
    #
    # 【为什么神经网络需要它？】
    #   纯加法和乘法叠再多层也只是直线（线性），学不了复杂东西。
    #   ReLU 加了个"折角"，让网络能学曲线、学复杂规律。
    #
    # 【导数怎么算？】
    #   输入 > 0 → 导数 = 1（梯度正常通过）
    #   输入 ≤ 0 → 导数 = 0（梯度被堵死，传不下去了）
    # ============================================================
    def relu(self):
        jie_guo = EasyValue(0 if self.shu_ju < 0 else self.shu_ju, (self,), 'relu')
        return jie_guo


    # ============================================================
    # 下面三个是用已有运算"拼"出来的，不需要单独写导数
    # ============================================================

    def fu(self):
        """取负数：-a 就是 a × (-1)"""
        return self.cheng(-1)

    def jian(self, qi_ta):
        """减法：a - b 就是 a + (-b)"""
        # 如果对方是普通数字，先包装成 EasyValue（和 jia、cheng 里的处理一样）
        if not isinstance(qi_ta, EasyValue):
            qi_ta = EasyValue(qi_ta)
        return self.jia(qi_ta.fu())

    def chu(self, qi_ta):
        """除法：a ÷ b 就是 a × (b 的 -1 次方)，也就是 a × (1/b)"""
        # 同样要先包装，不然 5 这种普通数字没有 .mi() 方法
        if not isinstance(qi_ta, EasyValue):
            qi_ta = EasyValue(qi_ta)
        return self.cheng(qi_ta.mi(-1))


    # ============================================================
    # 反向传播 - 整个引擎最关键的方法！！
    # ============================================================
    # 调用方式：在最终结果上调用 .fan_xiang_chuan_bo() 或 .backward()
    # 调用之后：每个参与运算的 EasyValue 的 .ti_du 都会被填上正确的梯度
    #
    # 【它干了三件事】
    #   第一步：把所有节点排个队（拓扑排序）
    #           → 确保"先算出来的数"排前面，"后算出来的数"排后面
    #   第二步：把最终结果的梯度设为 1（自己对自己的导数当然是 1）
    #   第三步：从后往前走，每经过一个节点，按运算类型把梯度传给它的子节点
    #
    # 【为什么要排队？】
    #   比如算 (a+b)*c，必须先知道 (a+b) 这个中间结果的梯度，
    #   才能继续往下传给 a 和 b。排队就是保证"先来后到"的顺序。
    #
    # 【什么是拓扑排序？】
    #   就是"先处理没有依赖的，再处理有依赖的"。
    #   比如穿衣服：先穿内衣，再穿外套（外套依赖内衣）。
    #   这里用的方法叫"深度优先搜索 + 后序"：
    #     先一头扎到最底层，回来的路上依次记录，最后反过来就是正确顺序。
    # ============================================================
    def fan_xiang_chuan_bo(self):

        # ---- 第一步：拓扑排序，把节点排好队 ----
        tuo_pu_lie_biao = []          # 排好队的结果放这里
        yi_fang_wen = set()           # 记录"已经来过的节点"，避免重复

        def tuo_pu(v):
            """递归地把节点 v 和它的所有祖先按顺序放进列表"""
            if v not in yi_fang_wen:
                yi_fang_wen.add(v)                   # 标记：这个节点来过了
                for fu_qin in v.zi_jie_dian:         # 先处理它的子节点（往上爬）
                    tuo_pu(fu_qin)
                tuo_pu_lie_biao.append(v)             # 子节点都处理完了，才轮到自己

        tuo_pu(self)  # 从最终结果开始，把整棵树都排好队

        # ---- 第二步：起点的梯度设为 1 ----
        # 自己对自己求导 = 1，这是反向传播的"火种"
        # 没有这一步，所有梯度都是 0，传什么都是 0
        self.ti_du = 1.0

        # ---- 第三步：从后往前，逐个节点传递梯度 ----
        # reversed = 反过来遍历（从最终结果走回最初的输入）
        for jie_dian in reversed(tuo_pu_lie_biao):
            yun_suan = jie_dian.yun_suan

            # ---- 加法的梯度传递 ----
            # c = a + b → a 和 b 各自分到 c 的全部梯度（导数是 1）
            if yun_suan == 'add':
                a, b = jie_dian.zi_jie_dian
                a.ti_du += jie_dian.ti_du             # a 的梯度 += c 的梯度 × 1
                b.ti_du += jie_dian.ti_du             # b 的梯度 += c 的梯度 × 1

            # ---- 乘法的梯度传递 ----
            # c = a × b → a 分到 b的值 × c的梯度，b 分到 a的值 × c的梯度
            #（口诀：求谁的导，乘对方的值）
            elif yun_suan == 'mul':
                a, b = jie_dian.zi_jie_dian
                a.ti_du += b.shu_ju * jie_dian.ti_du  # a 的梯度 += b的值 × c的梯度
                b.ti_du += a.shu_ju * jie_dian.ti_du  # b 的梯度 += a的值 × c的梯度

            # ---- 幂运算的梯度传递 ----
            # y = x^c → 导数 = c × x^(c-1)
            # 运算名格式是 'pow_2' 或 'pow_-1'，所以用 split('_') 取出次数
            elif yun_suan.startswith('pow_'):
                ci_shu = float(yun_suan.split('_')[1])          # 从 'pow_2' 中取出 2
                a = jie_dian.zi_jie_dian[0]                     # 幂运算只有一个子节点
                a.ti_du += (ci_shu * a.shu_ju ** (ci_shu - 1)) * jie_dian.ti_du

            # ---- ReLU 的梯度传递 ----
            # 输出 > 0：梯度原样通过（× 1）
            # 输出 ≤ 0：梯度堵死（× 0）
            elif yun_suan == 'relu':
                a = jie_dian.zi_jie_dian[0]
                a.ti_du += (jie_dian.shu_ju > 0) * jie_dian.ti_du

            # 如果是最初的输入节点（没有运算），什么也不用做
            # 因为它没有子节点需要传递梯度


    def __repr__(self):
        return f"EasyValue(shu_ju={self.shu_ju}, ti_du={self.ti_du})"


    # ============================================================
    # 兼容段：让 + - * / ** 这些符号直接能用（兼容Karpathy原版）
    # ============================================================
    # Python 看到 a + b 时，会自动找 a.__add__(b)
    # 看到 5 + a 时（左边不是 EasyValue），会反过来找 a.__radd__(5)
    # 下面就是把这些"魔法方法"都接上我们自己写的中文方法
    # ============================================================

    def __add__(self, other):      return self.jia(other)       # a + b
    def __radd__(self, other):     return self.jia(other)       # 5 + a（反向加法）
    def __mul__(self, other):      return self.cheng(other)     # a * b
    def __rmul__(self, other):     return self.cheng(other)     # 5 * a（反向乘法）
    def __pow__(self, other):      return self.mi(other)        # a ** 2
    def __neg__(self):             return self.fu()             # -a
    def __sub__(self, other):      return self.jian(other)      # a - b
    def __rsub__(self, other):     return EasyValue(other).jia(self.fu())   # 5 - a
    def __truediv__(self, other):  return self.chu(other)       # a / b
    def __rtruediv__(self, other): return EasyValue(other).chu(self)        # 5 / a
    def backward(self):            self.fan_xiang_chuan_bo()    # 英文别名，方便习惯英文的人