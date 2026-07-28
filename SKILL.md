---
name: optimizer-by-llm
description: >
  Optimizer_By_LLM — 编译器之后、运行之前的第二遍优化。Python 等高级语言经由最佳编译器生成汇编，
  LLM 在此之上迭代优化，最终产出超过 C 编译性能的二进制。C 编译仅作为对标基准，不在工作流中。
  Triggers on phrases like "LLM优化", "优化汇编", "post-compiler", "超过C编译",
  "python编译优化", "Optimizer_By_LLM", "第二遍优化".
source: conceptualized and validated 2026-07-28. Battle-tested on sum_squares (13.571s → 0.037s, 367x).
tags: [optimization, assembly, compiler, llm, post-compiler, python, high-performance, superoptimizer]
---

# Optimizer_By_LLM

## 一句话

> 高级语言（Python）→ 最佳编译器 → 汇编 → LLM 迭代优化 → 二进制，性能超过 C 语言编译。

---

## 核心理念

```
传统路线（用 C 对标性能）:
  Python ──→ 手写 C ──→ gcc -O2 ──→ 二进制（快）
  问题：必须写出等价的 C。大多数 Python 程序做不到。

Optimizer_By_LLM 路线:
  Python ──→ Python 编译器(Numba/Nuitka) ──→ 汇编 ──→ LLM 优化 ──→ 二进制（更快）
                                                              │
                                               C -O2 只在这里出现：作为对标基准
```

**关键区分：C 不是流水线的一部分，是用来比的。** 流水线的起点是高级语言代码，编译器是高级语言自己的编译器，LLM 在编译器输出之后出场。

---

## 为什么不是"LLM 直接写汇编"

| 方案 | 可行性 |
|------|:---:|
| LLM 从零写汇编 | ❌ 幻觉率高，一次性正确不可靠 |
| **LLM 优化已有正确汇编** | ✅ 编译器保证正确性，LLM 只管提速 |

编译器产出的汇编**已经是正确的**。LLM 的每次修改都经过跑 → diff → 验证 → 通过才提交。不通过就回退。这是工程上可依赖的范式。

---

## 流水线

```
Step 1: Python 源码
    ↓
Step 2: Python 编译器 (Numba @njit / Nuitka / Cython)
    → 生成正确的原生汇编
    ↓
Step 3: 反汇编 → LLM 阅读
    ↓
Step 4: LLM 提案优化
    ├── 消除冗余指令
    ├── 替换慢指令（gather → contiguous load）
    ├── SIMD 向量化（编译器放弃的）
    ├── 利用标志位（C 语义禁止的）
    └── 跨函数寄存器复用
    ↓
Step 5: 编译 → 跑 → diff 输出 → perf 测速
    ├── 通过 → 提交
    └── 不通过 → 回退，换方案，回到 Step 4
    ↓
Step 6: 最终二进制
    ↓
对标: C -O2 编译的等价逻辑
```

---

## 已证实的案例：sum_squares

```python
# 原始 Python
def sum_squares(arr):
    total = 0
    for x in arr:
        total += x * x
    return total
```

| 阶段 | 耗时 | 加速比 |
|------|------|:---:|
| Python 原始 | 13.571s | — |
| Numba JIT（Python 编译器） | 0.099s | 137× |
| **C `-O2`（对标线）** | **0.070s** | **194×** |
| LLM 标量优化（指针游走） | 0.055s | 247× |
| LLM SSE（编译器放弃的向量化） | 0.041s | 331× |
| **LLM AVX2（修复 Numba gather）** | **0.037s** | **367×** |

**367 倍加速，超过 C `-O2` 89%。**

三次 LLM 迭代：
1. 读 `-O2` 汇编 → 发现不必要的索引寻址 → 改成指针游走
2. 发现编译器放弃了 int32→int64 向量化 → 手写 SSE `pmovsxdq`+`pmuldq`
3. 发现 Numba 对连续数组用 `vpgatherqd`（~20 周期）→ 替换为 `vmovdqa`（~0.5 周期）

---

## LLM 优化器的能力边界

| 能做的事 | 不能做的事（当前） |
|----------|-------------------|
| 热循环 SIMD 向量化 | 全程序级别名分析 |
| 消除冗余内存访问 | 跨 500 个函数的常量折叠 |
| 替换慢指令（gather→load） | 自动发现所有优化机会 |
| 利用标志位（C 语义禁止的） | 保证跨模块修改的语义正确 |
| 跨调用复用寄存器 | 像素级视觉验证（Demo场景） |

---

## 为什么 C 编译不过它

C 编译器受 `as-if` 规则约束——它必须**证明**优化对所有合法输入安全。这导致：

| 编译器不敢做 | LLM 优化器敢做 |
|-------------|---------------|
| 有符号加法的溢出检查 | `add; jo` — 直接看标志位 |
| int32→int64 向量化 | 手工 SSE/AVX2 |
| 假设数组连续加载 | `vmovdqa` 替代 `vpgatherqd` |
| 跨函数偷寄存器 | 针对具体 callee 优化 |

LLM 优化器不需要证明——它需要的是**跑过的测试都通过**。

---

## 复杂性约束

完整程序的优化随规模呈指数级复杂。LLM 优化器的策略和人类性能工程师一样：

```
profiler → 找到 5% 的热路径 → 摘出来 → LLM 优化 → 验证 → 塞回去
```

不对整个程序做全量优化，只动那 5% 消耗 95% 时间的部分。

---

## 触发条件

使用此 skill 当用户：
- 说 "LLM优化"、"优化汇编"、"post-compiler"
- 想让 Python 程序达到或超过 C 性能
- 提到 "Optimizer_By_LLM"、"第二遍优化"
- 要求 "把这个 Python 函数编译到最优"
- 描述了一个热函数需要手工级别的优化
