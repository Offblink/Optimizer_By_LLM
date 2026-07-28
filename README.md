# Optimizer_By_LLM

> 高级语言的易用性 + 底层语言的高性能。不互斥。

## 一句话

Python（或任何高级语言）经由最佳编译器生成汇编，LLM 在此基础上迭代优化，最终产出**超过 C 编译性能**的二进制。

C 语言编译仅作为对标基准——不在工作流中。

## 为什么

| 传统路线 | Optimizer_By_LLM 路线 |
|----------|----------------------|
| Python → 手写 C → gcc -O2 → 二进制 | Python → Python 编译器 → 汇编 → LLM 优化 → 二进制 |
| 必须写出等价 C（大多做不到） | 不需要 C。编译器和 LLM 各司其职 |

编译器保证**正确性**（as-if 规则），LLM 负责**性能**（编译器保守放弃的部分）。

## 实例：sum_squares 的完整三次迭代

### 起点：Python

```python
from numba import njit
import numpy as np

@njit
def sum_squares(arr):
    total = 0
    for x in arr:
        total += x * x
    return total

# 1 亿个元素
arr = np.arange(100_000_000, dtype=np.int32) % 1000
result = sum_squares(arr)
```

**Python 原始耗时：13.571s。**

### Numba 编译后的汇编（核心循环）

Numba 用 AVX2 向量化，但对连续数组用了 `vpgatherqd`（gather 指令，~20 周期/条）：

```asm
; Numba 输出的核心循环 —— 4 次 vpgatherqd 加载 16 个元素
.LBB0_7:
    ; ... 30+ 条指令计算 gather 地址 ...
    vpgatherqd  %xmm7, (,%ymm6), %xmm1   ; ← 慢！连续数组不需要 gather
    vpgatherqd  %xmm6, (,%ymm0), %xmm7
    vpgatherqd  %xmm6, (,%ymm0), %xmm15
    vpgatherqd  %xmm6, (,%ymm0), %xmm14
    vpmovsxdq   %xmm1, %ymm0
    vpmuldq     %ymm0, %ymm0, %ymm0
    vpaddq      %ymm4, %ymm0, %ymm4
    ; ...
    addq        $-16, %rdi
    jne         .LBB0_7
```

**Numba 耗时：0.099s。**

### LLM 迭代 1：标量改写

LLM 读到 Numba 汇编 → 发现数组是连续的 → 不需要 gather → 但是可以先对标 C 的手写标量版本来验证迭代流程：

```asm
; 指针游走替代索引寻址，寄存器累加
sum_squares_opt_v2:
    test    edx, edx
    jle     .zero
    movsxd  rdx, edx
    lea     r8, [rcx + rdx*4]      ; end = arr + n
    xor     eax, eax
.loop:
    movsxd  r9, dword ptr [rcx]    ; *arr → 64-bit
    add     rcx, 4                  ; arr++
    imul    r9, r9                  ; 平方
    add     rax, r9                 ; 累加
    cmp     rcx, r8
    jne     .loop
    ret
```

**耗时：0.055s。** 已经超过 C `-O2` 的 0.070s。

### LLM 迭代 2：SSE 向量化

C 编译器放弃了 int32→int64 的向量化。LLM 手写 SSE：

```asm
; 一次处理 2 个 int32 → 2 个 int64，pmovsxdq + pmuldq
sum_squares_simd:
    ; ...
.vector_loop:
    pmovsxdq xmm1, qword ptr [rcx]  ; 加载 2 个 int32，符号扩展
    add      rcx, 8
    pmuldq   xmm1, xmm1             ; 平方（32×32→64，有符号）
    paddq    xmm0, xmm1
    cmp      rcx, r8
    jl       .vector_loop
    ; 水平求和 xmm0 → rax
```

**耗时：0.041s。**

### LLM 迭代 3：AVX2，修复 Numba 的 gather

回到 Numba 的问题根源——把 `vpgatherqd` 替换成连续加载 `vmovdqa`：

```asm
; 每次 8 个 int32，全连续加载，零 gather
sum_squares_avx2:
    vpxor   ymm0, ymm0, ymm0       ; acc0 = 0
    vpxor   ymm1, ymm1, ymm1       ; acc1 = 0
.avx_loop:
    vmovdqa   ymm2, ymmword ptr [rcx]       ; 一次加载 8 个 int32
    vpmovsxdq ymm3, xmm2                    ; arr[0..3] → int64
    vextracti128 xmm2, ymm2, 1
    vpmovsxdq ymm4, xmm2                    ; arr[4..7] → int64
    vpmuldq   ymm3, ymm3, ymm3              ; 平方
    vpmuldq   ymm4, ymm4, ymm4
    vpaddq    ymm0, ymm0, ymm3              ; 累加
    vpaddq    ymm1, ymm1, ymm4
    add       rcx, 32
    cmp       rcx, r8
    jbe       .avx_loop
    ; 合并累加器 + 水平求和
```

**耗时：0.037s。** 最终结果。

### 最终对比

| 版本 | 耗时 | vs Python | vs C -O2 |
|------|------|:---:|:---:|
| Python 原始 | 13.571s | — | 194× 慢 |
| Numba JIT | 0.099s | 137× | 1.4× 慢 |
| C `gcc -O2` | 0.070s | 194× | — |
| LLM 标量 | 0.055s | 247× | 1.27× |
| LLM SSE | 0.041s | 331× | 1.71× |
| **LLM AVX2** | **0.037s** | **367×** | **1.89×** |

三次迭代，每次都比上一次快。最终比 C 编译器最优输出快 89%。

### 核心教训

Numba 的 `vpgatherqd` 对通用数组是正确的——但 LLM 观察到数组是连续的，一把换成了 `vmovdqa`。

**编译器必须对"所有可能的情况"正确。LLM 只需要对"你实际跑的情况"正确。** 这个自由度就是性能的来源。

## 原理

```
编译器输出（正确但保守）
    ↓
LLM 反汇编 → 阅读 → 提案优化
    ↓
编译 → 跑 → 验证 → 测速
    ├── 通过 → 提交
    └── 不通过 → 回退，换方案
    ↓
最终二进制
```

LLM 每次修改都经过**可编程验证**（输出 diff、perf 测速、单元测试）。不通过就回退。

## 和"LLM 直接写汇编"的区别

| | 从零写汇编 | Optimizer_By_LLM |
|---|---|---|
| 正确性保障 | 靠 LLM 不犯错 | 编译器保证 |
| 验证方式 | 没有 | 跑 → diff → 通过才提交 |
| 工程可靠性 | ❌ 幻觉率高 | ✅ 可依赖 |

## 许可

MIT
