---
layout: post
title: "Post-Training 面试速记: 从分词到对齐"
date: 2026-05-16 23:00:00-0700
description: "从 MinHash 去重到 BPE/WordPiece/Unigram、Softmax 手推、FlashAttention、RM/PPO/GRPO/DPO，每节给出 Key Insight、实现层次图、代码骨架、推导和高频追问。"
tags: post-training rlhf tokenizer ppo grpo dpo flashattention
categories: notes
giscus_comments: true
related_posts: false
toc:
  sidebar: left
---


> 整理自 Day3 复习笔记。面向"白纸能写出来 + 能讲明白"的面试目标,每节固定四块:**Key Insight → 实现层次图 → 代码 → 高频追问**。

---

## 目录

1. [MinHash — 海量文档去重](#1-minhash--海量文档去重)
2. [Tokenizer 三件套: BPE / WordPiece / Unigram](#2-tokenizer-三件套-bpe--wordpiece--unigram)
3. [反向传播 — Softmax + Attention 手推](#3-反向传播--softmax--attention-手推)
4. [Memory-Efficient Attention — Online Softmax → FlashAttention](#4-memory-efficient-attention--online-softmax--flashattention)
5. [Reward Model — Bradley-Terry pairwise](#5-reward-model--bradley-terry-pairwise)
6. [RL Fine-tuning — 策略梯度 + 组内归一化](#6-rl-fine-tuning--策略梯度--组内归一化)
7. [GAE — 广义优势估计](#7-gae--广义优势估计)
8. [PPO — Clipped Surrogate + KL](#8-ppo--clipped-surrogate--kl)
9. [GRPO — PPO 砍掉 Value 网络](#9-grpo--ppo-砍掉-value-网络)
10. [DPO — 把 RM + RL 合成一个分类损失](#10-dpo--把-rm--rl-合成一个分类损失)
11. [一图全景: 四种对齐算法对比](#11-一图全景-四种对齐算法对比)

---

## 1. MinHash — 海量文档去重

> **Key Insight**: 对一个随机 hash `h`,**`P(min h(A) == min h(B)) = J(A, B)`**。
> 用 `num_perm` 个独立 hash 取最小,签名向量逐位相等的比例就是 Jaccard 的无偏估计。

### 为什么需要

预训练前必须去重: 重复样本会让模型**记忆**而不是**泛化**,还会让评测集泄露到训练集。
两两算 Jaccard 是 O(N²),百万文档跑不动。MinHash 把"集合比较"压缩到"向量比较"。

### 实现层次

```
                                ┌──────────────────────────┐
原始文档(string)               │  create_shingles(text, k) │   预处理
   │                            │  滑动 k 字符窗口          │
   └──────────────────────────►│  → set of strings        │
                                └─────────────┬────────────┘
                                              ▼
                                       shingles set
                                              │
                                ┌─────────────▼────────────────────┐
                                │  compute_signature(shingles)     │   签名生成
                                │  for shingle in document:        │
                                │    for i, (a,b) in hash_funcs:   │
                                │      h = (a*x + b) mod prime     │
                                │      sig[i] = min(sig[i], h)     │
                                └─────────────┬────────────────────┘
                                              ▼
                                  signature [num_perm]
                                              │
                                ┌─────────────▼─────────────┐
                                │  similarity(sig1, sig2)    │   相似度
                                │  Σ(sig1[i]==sig2[i])/N    │
                                │  ≈ Jaccard(A, B)          │
                                └────────────────────────────┘

工程外壳 (在 MinHash 类外):
   LSH 分桶 → 桶内两两比签名 → Union-Find 聚簇 → 每簇留一篇
```

### 代码骨架

```python
class MinHash:
    def __init__(self, num_perm=128):
        self.num_perm = num_perm
        self.prime = 2147483647
        self.hash_funcs = []
        for i in range(num_perm):
            seed = hashlib.sha256(f"minhash_{i}".encode()).digest()
            a = int.from_bytes(seed[:4], "big") % self.prime or 1
            b = int.from_bytes(seed[4:8], "big") % self.prime
            self.hash_funcs.append((a, b))           # h(x) = (a*x + b) mod p

    def compute_signature(self, document: set) -> list:
        sig = [float("inf")] * self.num_perm
        for shingle in document:
            for i, (a, b) in enumerate(self.hash_funcs):
                sig[i] = min(sig[i], self._compute_hash(shingle, a, b))
        return sig

    def similarity(self, sig1, sig2) -> float:
        return sum(x == y for x, y in zip(sig1, sig2)) / len(sig1)
```

### 关键定理的证明梗概

**命题**: 对一个**随机排列** π,`P(min π(A) == min π(B)) = J(A, B) = |A∩B| / |A∪B|`。

**证明**:
- 令 `U = A ∪ B`,在 U 上看 π 的最小值。π 在 U 上是均匀随机的全排列。
- "min π(A) == min π(B)" ⟺ U 中最小值同时属于 A 和 B ⟺ U 中最小值属于 `A ∩ B`。
- 由对称性,U 中任一元素被选为最小的概率均为 `1/|U|`。
- 所以概率 = `|A∩B| / |U| = |A∩B| / |A∪B| = J(A, B)`。

**工程化**: 真实排列 π 太贵,用 **universal hash family** `h(x) = (a·x + b) mod p`(`p` 是大素数,`a, b` 随机)
近似一个随机排列 —— 工程上够好。

### 估计精度 (num_perm 怎么选)

`num_perm = K` 个独立 hash 各自取最小,签名等值率 `Ĵ = X / K`,其中 `X ~ Binomial(K, J)`。

```
E[Ĵ] = J              ← 无偏
Var[Ĵ] = J(1-J) / K   ← 方差随 K 线性下降
```

| K | 标准差 (J=0.5) | 95% CI 宽度 |
|---|---|---|
| 64  | ±6.3% | ~12.5% |
| 128 | ±4.4% | ~8.7%  |
| 256 | ±3.1% | ~6.2%  |
| 1024 | ±1.6% | ~3.1% |

工业经验: **128-256 起步**,严格去重(LSH 阈值 0.7+)可以加到 512。

### LSH 分桶: 把 O(N²) 压成近线性

直接两两比签名仍是 `O(N²·K)`。**LSH (Locality-Sensitive Hashing) banding** 解法:

把 K 个 hash 切成 `b` 个 band,每 band 包含 `r` 行 (`b·r = K`)。
两文档在**至少一个 band 内 r 行全等** → 进入同桶 → 候选对。

**S 曲线**: 两文档 Jaccard 为 `J`,被选为候选对的概率:

```
P(候选) = 1 - (1 - J^r)^b
```

| (b, r) | J=0.5 | J=0.7 | J=0.8 | J=0.9 |
|---|---|---|---|---|
| (20, 5) | 0.47 | 0.97 | 1.00 | 1.00 |
| (50, 4) | 0.95 | 1.00 | 1.00 | 1.00 |
| (25, 10) | 0.02 | 0.36 | 0.82 | 1.00 |

`(b, r)` 控制 S 曲线的**陡峭程度和阈值位置**: 想滤掉 J<0.7,选大 `r` 小 `b`;想宽松些,反之。

### 完整去重 pipeline

```python
def dedupe_corpus(documents):
    mh = MinHash(num_perm=256)
    signatures = [mh.compute_signature(shingles(d)) for d in documents]

    # LSH banding 分桶
    buckets = defaultdict(list)
    b, r = 50, 4   # 50 bands × 4 rows = 200 (用前 200 维)
    for idx, sig in enumerate(signatures):
        for band in range(b):
            key = (band, tuple(sig[band*r : (band+1)*r]))
            buckets[key].append(idx)

    # Union-Find 聚簇
    uf = UnionFind(len(documents))
    for bucket in buckets.values():
        if len(bucket) > 1:
            for j in bucket[1:]:
                if mh.similarity(signatures[bucket[0]], signatures[j]) > 0.8:
                    uf.union(bucket[0], j)

    # 每簇留一篇
    return [documents[i] for i in uf.representatives()]
```

时间复杂度: 签名 `O(N·K·|shingles|)` + 分桶 `O(N·b)` + 桶内验证 `O(候选数 · K)` ≈ **近线性**。

### 数值例子

```
d1 = "The quick brown fox jumps over the lazy dog."
d2 = "The quick brown fox jumps over the lazy dog!"   # 只差一个标点
d3 = "Completely unrelated sentence about cats."

sim(d1, d2) ≈ 0.97   ← 近重复 (会被聚到同一簇)
sim(d1, d3) ≈ 0.00   ← 不相关
```

### MinHash vs SimHash vs 嵌入

| 方法 | 估计什么 | 输入要求 | 典型用途 |
|---|---|---|---|
| **MinHash** | Jaccard (集合) | shingle 集 (字符 / 词 n-gram) | 文档级去重 |
| **SimHash** | Cosine (向量) | 特征向量 (TF-IDF 等) | 网页指纹、相似搜索 |
| **Embedding (BGE, OpenAI)** | Cosine in 语义空间 | 文本 → 向量 | 语义去重、检索 |
| **Bloom Filter** | 完全相等 | 任意 | 精确查重 |

LLM 预训练去重: **先 MinHash + LSH 粗筛**,再 embedding 模型在候选对上**精筛**(可选)。
RedPajama, FineWeb, SlimPajama 全都用这套。

### 高频追问

| Q | A |
|---|---|
| 为什么 min-hash 碰撞概率 = Jaccard? | 见上面证明:U=A∪B 中的最小元素均匀分布,同时属于交集的概率 = \|A∩B\|/\|A∪B\| |
| `num_perm` 怎么选? | 标准差 ≈ √(J(1-J)/K);128 ±4.4%,256 ±3.1%。代价是签名内存和比较时间 |
| MinHash 用什么 hash? | 工程上用 `(a·x + b) mod p` (universal hash);近似随机排列。`p` 通常 `2^31 - 1` |
| LSH banding 的 (b, r) 怎么调? | 想 J>τ 的对都被召回 → 让 S 曲线在 τ 附近过 0.5: `τ ≈ (1/b)^(1/r)` |
| 为什么 shingle 用 5-gram 字符? | 平衡: 太短(字符)→ 无信息;太长(句子)→ 微小差异就 miss。5-7 字符是经验最优 |
| MinHash vs SimHash? | MinHash 估 **Jaccard**(集合相似),适合 shingle 集;SimHash 估 **cosine**(向量相似),适合特征向量 |
| 为什么去重对 LLM 重要? | (1) 重复 → 记忆而非泛化 (2) 评测数据泄露 (3) 浪费算力。GPT-3, LLaMA, Chinchilla 论文都强调 |
| 工业级 dedup 流程? | 文档级 MinHash+LSH → 句子级 SimHash/Bloom → 评测集污染检查 (n-gram 重叠);三层都过 |

---

## 2. Tokenizer 三件套: BPE / WordPiece / Unigram

> **Key Insight**: 都是 subword,**唯一区别在"用什么准则合并/删除 token"**。
> BPE = 频率;WordPiece = 互信息;Unigram = 删除后 likelihood 下降最小。

### 为什么需要 subword

| 方案 | 问题 |
|---|---|
| Word-level | 词表巨大(50万+),OOV 严重 |
| Char-level | 序列太长,模型学不到语义 |
| **Subword(BPE/WP/Uni)** | 词表 30k~100k,常用词整 token,稀有词降到子词 |

### 实现层次 (以 BPE 为例)

```
══════════ 训练阶段 (Training) ══════════

原始 corpus (list[str])
   │
   ▼  get_word_freqs(corpus)            按空格切词 + 计数
{"low": 5, "lower": 2, ...}
   │
   ▼  get_splits(word)                  拆字符 + </w>
{"low": [l,o,w,</w>], "lower": [l,o,w,e,r,</w>], ...}
   │
   ▼  ╔═════════════════════════════════════════════╗
      ║  loop merges_needed 次:                      ║
      ║    pairs = get_stats(splits)                ║
      ║    best  = argmax(pairs)        # 频率最高  ║
      ║    splits = merge_pair(splits, best)        ║
      ║    merges.append(best)                      ║
      ╚═════════════════════════════════════════════╝
   │
   ▼
final_vocab + merges (有序列表) + merges_lookup (pair→优先级)


══════════ 编码阶段 (Encoding) ══════════

word
   │
   ▼  get_splits(word)
[c1, c2, ..., </w>]
   │
   ▼  ╔═════════════════════════════════════════════╗
      ║  apply_bpe:                                  ║
      ║    while True:                              ║
      ║      在相邻 pair 里找 merges_lookup          ║
      ║      优先级最高 (索引最小) 的那个            ║
      ║      没有 → break;有 → 合并所有出现位置     ║
      ╚═════════════════════════════════════════════╝
   │
   ▼  vocab 查询
[id1, id2, ...]
```

### 2.1 BPE — 频率最高的 pair 合并

**完整跑一遍** `["low", "lower", "lowest"]`:

```
init splits:
  low:    [l, o, w, </w>]
  lower:  [l, o, w, e, r, </w>]
  lowest: [l, o, w, e, s, t, </w>]

第1轮 pair 频率: (l,o)=3, (o,w)=3, (w,e)=2, ...
  → 合并 (l, o)
  → splits 里 l,o 相邻处全变 lo

第2轮: (lo,w)=3 ← 最高 → 合并

第3轮: (low,e)=2 ← 最高 → 合并

... 直到攒够 vocab_size 个 token
```

**编码代码**:

```python
def apply_bpe(word, merges_lookup):
    splits = list(word) + ["</w>"]
    while True:
        # 找当前相邻 pair 里"优先级最高"(最早学到)的
        best_pair, best_pri = None, float("inf")
        for i in range(len(splits) - 1):
            pri = merges_lookup.get((splits[i], splits[i+1]))
            if pri is not None and pri < best_pri:
                best_pair, best_pri = (splits[i], splits[i+1]), pri
        if best_pair is None:
            break
        splits = merge_in_place(splits, best_pair)
    return splits
```

**关键**: 编码必须**严格按训练时的合并顺序**重放,否则同一个词可能被切成不同 token,模型会崩。

### 2.2 WordPiece (BERT) — 互信息最大的 pair 合并

**唯一的区别**: 合并准则换成互信息

```
BPE:        argmax  count(A, B)
WordPiece:  argmax  count(A, B) / (count(A) × count(B))
```

直觉: WordPiece 选「A 和 B 真的经常一起出现」而非「A 自己就常见」。

**词边界标记**: 用 `##` 表示"接续片段":
```
"playing" → ["play", "##ing"]
"unhappiness" → ["un", "##happy", "##ness"]
```

**OOV 处理**: 拼不出的输出 `[UNK]`(BPE 会用字符兜底,WP 不会 → 更脆)。

### 2.3 Unigram LM (T5, SentencePiece 默认) — 反向删词表

**自上而下删**:
```
1. 启发式构造一个超大候选词表(100万)
2. 训练 Unigram 语言模型: p(token)
3. 对每个词,Viterbi 找概率最高的切分
4. 计算每个 token 的"删除损失"(整个语料 likelihood 下降)
5. 删损失最小的 10%
6. 重复 2-5,直到词表 = vocab_size
```

**关键**: 一个词有**多种切分方式**,选概率最高的(或随机采样做数据增强)。

### 2.4 Byte-level BPE (GPT-2/3/4, LLaMA-3)

**不在字符层,在 UTF-8 字节层做 BPE**:

```
"你好" → UTF-8 字节 [228, 189, 160, 229, 165, 189] → 当作 6 个初始 token
```

- 初始词表固定 **256**(所有字节值)
- 任何 Unicode 都能编码 → **永远不会 OOV**
- GPT-2 用 Ġ 等可见字符替代不可见字节(仅显示,不影响算法)

### 三者对比

| 维度 | BPE | WordPiece | Unigram |
|---|---|---|---|
| 训练方向 | 自下而上加 | 自下而上加 | 自上而下删 |
| 合并/删除准则 | 频率最高 | 互信息最大 | 删除后 likelihood 下降最小 |
| 词边界标记 | `</w>` 后缀 | `##` 续接前缀 | `▁` 空格前缀(SP) |
| 编码 | 按 merges 贪心 | 最长前缀贪心 | Viterbi 最优切分 |
| OOV 处理 | 字符级兜底 | `[UNK]` | 字符级兜底 |
| 代表模型 | GPT, LLaMA, RoBERTa | BERT, DistilBERT | T5, ALBERT, XLNet |

**坑题**: **SentencePiece 不是算法,是 Google 的库**,默认 Unigram,也支持 BPE。LLaMA 用 SentencePiece-BPE 模式。

### 高频追问

| Q | A |
|---|---|
| 为什么 BPE 需要 `</w>`? | 区分"词尾 e"(`est</w>`)和"词中 e",避免错误合并跨词边界 |
| 编码是确定性的吗? | BPE/WP 是;Unigram 默认是(选概率最高),训练时可采样做正则 |
| 同一个词不同上下文切分会变吗? | 不会(BPE/WP);Unigram 在 subword regularization 下会 |
| 为什么 GPT 用 byte-level? | 多语言通用 + 代码符号 + 永远无 OOV |
| WordPiece 的 `[UNK]` 怎么来的? | 输入有训练时没见过的字符,且无法拼回 vocab 子串 |

---

## 3. 反向传播 — Softmax + Attention 手推

> **Key Insight**:
> Softmax 的 Jacobian = `diag(s) - s·sᵀ`,向量化后 = `s ⊙ (g − g·s)`。
> Attention 反向 = 把前向的 4 个矩阵乘**逆序求转置**,中间夹一个 softmax_backward。

### 实现层次

```
══════════ Forward ══════════
                                  ┌─ Q [N, D]
              S = Q·Kᵀ/√D         │
   ────────────────────────────── │  ↓ matmul + scale
              A = softmax(S)      ├─ K [M, D]
              out = A·V           │  ↓ softmax
   ────────────────────────────── │  ↓ matmul
                                  │
                                  └─ V [M, D]


══════════ Backward (反向链) ══════════

grad  ← upstream gradient on out
  │
  ├─►  dV = Aᵀ · grad                      [M, D]
  │           ▲
  │           └─ out = A·V → ∂out/∂V = Aᵀ
  │
  ├─►  dA = grad · Vᵀ                      [N, M]
  │           ▲
  │           └─ out = A·V → ∂out/∂A 走 V
  │
  ├─►  dS = softmax_backward(A, dA)        [N, M]
  │           │
  │           │  对每一行 s_i = A[i]:
  │           │  J = diag(s_i) - s_i·s_iᵀ
  │           │  向量化: s_i ⊙ (dA[i] - (dA[i]·s_i))
  │           ▼
  ├─►  dQ = dS · K / √D                    [N, D]
  │
  └─►  dK = dSᵀ · Q / √D                   [M, D]
              ▲
              └─ S 的列 ↔ K 的行,所以要转置
```

### Softmax Jacobian 的完整推导(必背)

对一行 `s = softmax(z)`,即 `s_i = exp(z_i) / Σ_k exp(z_k)`。

**Case 1**: `i == j`

```
∂s_i/∂z_i = [exp(z_i) · Σ_k exp(z_k) - exp(z_i) · exp(z_i)] / (Σ_k exp(z_k))²
          = s_i - s_i² = s_i · (1 - s_i)
```

**Case 2**: `i ≠ j`

```
∂s_i/∂z_j = [0 · Σ_k exp(z_k) - exp(z_i) · exp(z_j)] / (Σ_k exp(z_k))²
          = -s_i · s_j
```

**合并** 用 Kronecker δ:

```
∂s_i/∂z_j = s_i · (δ_ij − s_j)
J = diag(s) − s · sᵀ           # 对称矩阵
```

### Jacobian-Vector Product 化简(为什么不用显式构造 J)

显式构造 `J` 是 `O(M²)`,但 backward 只要 `g · J`(VJP):

```
(g · J)_j = Σ_i g_i · s_i · (δ_ij − s_j)
         = g_j · s_j − s_j · Σ_i g_i · s_i
         = s_j · (g_j − ⟨g, s⟩)
```

→ 向量化: **`s ⊙ (g − ⟨g, s⟩)`**,只要 `O(M)` 乘加,不用建 J。

```python
# 显式 (慢, O(M²)):              vs       向量化 (快, O(M)):
J = np.diag(s) - np.outer(s, s)          out = s * (g - (g * s).sum(-1, keepdims=True))
out = g @ J                              # 直接对 batch 维广播
```

### Softmax + Cross-Entropy = `p − y` (经典化简)

CE loss: `L = -Σ_i y_i · log(s_i)`,其中 `y` 是 one-hot。

```
∂L/∂z_j = ∂L/∂s · ∂s/∂z = (-y/s) · J
        = Σ_i (-y_i/s_i) · s_i · (δ_ij − s_j)
        = -y_j + s_j · Σ_i y_i        ← y 是 one-hot, Σ y_i = 1
        = s_j − y_j
```

**所以 logits 上的梯度就是 `softmax(z) − onehot(target)`**,完全不用建 Jacobian。这就是为什么 deep learning 框架把 softmax + CE 融成一个 op (`F.cross_entropy(logits, targets)`),数值更稳、反向更快。

### Attention 反向链(QKV 完整推导)

前向: `S = QKᵀ/√D`, `A = softmax(S, dim=-1)`, `out = A·V`。

**已知**: `dout` (上游梯度, shape `[N, D]`)

**反向 1: `out = A·V` (普通矩乘)**

```
dV = Aᵀ · dout          ← shape [M, D]
dA = dout · Vᵀ          ← shape [N, M]
```

**反向 2: `A = softmax(S, dim=-1)` (逐行 softmax)**

```
dS[i] = softmax_backward(A[i], dA[i])     # 每行独立
     = A[i] ⊙ (dA[i] − ⟨dA[i], A[i]⟩)
```

**反向 3: `S = Q·Kᵀ/√D` (matmul + scale)**

```
dQ = dS · K / √D        ← shape [N, D]
dK = dSᵀ · Q / √D       ← shape [M, D]
```

**记忆法**: 反向是前向**4 个矩阵乘的逆序求转置**,中间夹一个 softmax_backward。`√D` 的位置和前向一样(标量,直接除)。

### 代码骨架

```python
def softmax_backward(s, grad_s):
    # 向量化版: s * (grad_s - (grad_s * s).sum(-1, keepdims=True))
    N, M = s.shape
    grad = np.zeros((N, M), dtype=np.float32)
    for i in range(N):
        J = np.diag(s[i]) - s[i][:, None] @ s[i][None, :]
        grad[i] = grad_s[i] @ J
    return grad

def sdpa_backward(grad, q, k, v):
    D = q.shape[1]
    A = softmax(q @ k.T / np.sqrt(D))
    dv = A.T @ grad
    dA = grad @ v.T
    dS = softmax_backward(A, dA)
    dq = dS @ k / np.sqrt(D)
    dk = dS.T @ q / np.sqrt(D)
    return dq, dk, dv
```

**验证**: 和 PyTorch autograd 对比,`atol=1e-4` 内 close。

### 为什么除 √D? (方差分析)

设 `Q[i], K[j]` 各元素 i.i.d. ~ `N(0, 1)`, 维度 D。

```
S[i,j] = Σ_k Q[i,k] · K[j,k]
E[S]   = 0
Var[S] = Σ_k Var[Q·K] = D · 1 = D     ← 方差线性增长
Std[S] = √D
```

D=64 时 `Std[S]≈8`,D=128 时 `≈11.3`。**进 softmax 前 logits 太大 → softmax 退化为 one-hot → 梯度消失**。

除 `√D`: `Var[S/√D] = D/D = 1`,无论 D 多大方差稳定。
**所以这个 `√D` 是为了梯度健康,不是为了 forward 数值稳定**。

### 数值稳定的 softmax(减 max 技巧)

```python
def softmax(z):
    z = z - z.max(-1, keepdims=True)   # ★ 减 max
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)
```

**为什么不影响结果**: `softmax(z) = softmax(z - c)`,因为 `exp(z_i - c) / Σ exp(z_k - c) = exp(z_i)·exp(-c) / (exp(-c)·Σ exp(z_k))`,`exp(-c)` 上下消去。

**为什么减最大值不溢出**: 减完后所有元素 ≤ 0,`exp(≤0) ∈ (0, 1]`,绝不会溢出。

### 训练时 attention 反向的显存代价

每一层 attention,正向产生:
- `S [B, H, N, M]` 中间矩阵
- `A [B, H, N, M]` softmax 输出

反向都要用,所以必须保存 → **显存 O(B·H·N²)**。
对 LLaMA-7B (32 层, 32 头),序列 4k → ~16 GB 仅 attention 中间量。
**这就是为什么 FlashAttention(§4)能让训练装下更大的模型**。

### 高频追问

| Q | A |
|---|---|
| 为什么 softmax 数值稳定要减 max? | `exp(z)` 大数会上溢;`softmax(z) == softmax(z - max(z))`(分子分母同乘常数) |
| 为什么 attention 要除 √D? | 点积方差 = D → softmax 饱和 → 梯度消失。除 √D 让 logit 方差稳定为 1 |
| softmax + cross-entropy 为什么简化成 `softmax - onehot`? | 见上面推导:`-y/s · J` 展开后 one-hot 把求和压到一个分量,剩 `p - y` |
| 为什么不显式建 Jacobian? | J 是 `O(M²)`,但 VJP 化简后 `s ⊙ (g − ⟨g,s⟩)` 只要 `O(M)` |
| 反向能 inplace 写 dq, dk, dv 吗? | 不行,共用 A 的话会乱;PyTorch 是 save_for_backward 保留前向中间量 |
| 训练时哪几个张量必须存? | 前向: Q, K, V, A(softmax 输出);反向重算 S 或直接复用 A。FlashAttention 只存 `(O, m, l)` 三个 |
| 为什么 PyTorch autograd 默认是 VJP 不是 JVP? | 神经网络 loss 是标量,VJP (反向模式) 1 次反向得所有参数梯度;JVP 要 `dim(params)` 次正向 |
| `softmax(z/T)` 中 T 的作用? | T → 0: one-hot(贪心);T → ∞: 均匀。LLM 推理用来控制采样随机性 |
| 推导一下 `dz = s · (g - g·s)`,为什么不是 `s · g - s² · g`? | `(g · s)` 是**内积标量** Σ g_i·s_i,不是逐元素 g⊙s。是同一个标量减回 g 的每个分量 |

---

## 4. Memory-Efficient Attention — Online Softmax → FlashAttention

> **Key Insight**: 数值稳定的 safe softmax 本身**最少要 2 pass**(必须先扫一遍找 `m_N` 才能减 max)。
> 但 attention 的最终目标是 `O = A·V`,**不是 A 本身** —— 对 `O` 再施一次 "surrogate" 技巧,可以把整个 attention 压到 **1 pass**。这就是 FlashAttention 的全部魔法。

> 参考: [From Online Softmax to FlashAttention](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf) (Zihao Ye, UW CSE 599M)

### 实现层次: 3 pass → 2 pass → 1 pass

```
══════════ Pass 1: Safe Softmax (3 passes) ══════════

   pass 1:  m_N = max(x_1, ..., x_N)          ← 扫一遍取全局 max
   pass 2:  d_N = Σ exp(x_i - m_N)             ← 再扫一遍算分母
   pass 3:  a_i = exp(x_i - m_N) / d_N         ← 第三遍得 softmax

   问题: 长序列里 logits {x_i} 装不下 SRAM,3 次 pass 就要 3 次重算 Q·Kᵀ


══════════ Pass 2: Online Softmax (Milakov 2018) ══════════

   关键: 用 surrogate d'_i 代替依赖 m_N 的 d_i

      d'_i := Σ_{j=1}^i exp(x_j - m_i)         ← 只用当前 max m_i,不用 m_N

   递推:
      m_i  = max(m_{i-1}, x_i)
      d'_i = d'_{i-1} · exp(m_{i-1} - m_i)     ← 旧累加用 exp(Δm) 重新缩放
             + exp(x_i - m_i)

   性质: 当 i = N 时 d'_N = d_N,所以可以用 d'_N 替换 d_N

   pass 1:  循环里同时算 m_i 和 d'_i
   pass 2:  a_i = exp(x_i - m_N) / d'_N

   但 softmax 本身**没法压到 1 pass** —— a_i 必须等 m_N 算完


══════════ Pass 3: FlashAttention (1 pass) ══════════

   关键观察: 我们不需要 a_i,只需要 O = Σ a_i · V[i,:]

   再施一次 surrogate,对 O 做递推:

      o'_i := Σ_{j=1}^i (exp(x_j - m_i) / d'_i) · V[j,:]

      o'_i = o'_{i-1} · (d'_{i-1} · exp(m_{i-1} - m_i)) / d'_i
                       + (exp(x_i - m_i) / d'_i) · V[i,:]

   性质: o'_N = O[k, :]

   一个循环里同时维护 (m_i, d'_i, o'_i) 三个状态,扫一遍 K, V 就完事
```

### 单 pass 算法 (核心 5 行)

```python
def flash_attention_row(q_k, K, V):
    """对 Q 的第 k 行计算 attention 输出 (single pass over K, V)."""
    D = V.shape[1]
    m, d, o = -np.inf, 0.0, np.zeros(D)
    for i in range(K.shape[0]):
        x_i   = q_k @ K[i] / np.sqrt(D)                        # logit
        m_new = max(m, x_i)
        d_new = d * np.exp(m - m_new) + np.exp(x_i - m_new)
        o     = o * (d * np.exp(m - m_new)) / d_new \
              + (np.exp(x_i - m_new) / d_new) * V[i]
        m, d  = m_new, d_new
    return o   # = O[k, :]
```

### 工程版: tiled (块大小 b 进 SRAM)

```python
def flash_attention_tiled(Q, K, V, b):
    L, D = Q.shape
    O = np.zeros_like(Q)
    for k in range(L):                                         # 行间天然并行
        m, d, o = -np.inf, 0.0, np.zeros(D)
        for start in range(0, L, b):                           # 沿 K, V 滑动
            Kc, Vc  = K[start:start+b], V[start:start+b]       # 一块进 SRAM
            x       = Q[k] @ Kc.T / np.sqrt(D)                 # [b]
            m_new   = max(m, x.max())
            scale   = np.exp(m - m_new)
            e       = np.exp(x - m_new)                        # [b]
            d_new   = d * scale + e.sum()
            o       = o * (d * scale) / d_new + (e @ Vc) / d_new
            m, d    = m_new, d_new
        O[k] = o
    return O
```

### 硬件视角: 为什么省 HBM

```
朴素:    HBM ↔ S [N, M] ↔ HBM ↔ A = softmax(S) [N, M] ↔ HBM ↔ A·V ↔ HBM
         中间矩阵 S, A 反复读写 HBM  (这是真正的瓶颈)

Flash:   HBM → load Q[k], K[i:i+b], V[i:i+b] → SRAM tile 内算完 (m,d,o) → HBM 只写 O[k]
         中间结果 (S 的 tile, A 的 tile) 永远不出 SRAM
```

H100 单个 SM 的 SRAM ≈ 228 KB(比 HBM 快 ~30×)。整体 SRAM footprint 只和 `b, D` 有关,**和序列长度 L 无关** → 这就是 FlashAttention 能撑 16k+ context 的原因。

### 反向传播: 重算 (recomputation) 而非缓存

朴素 attention 的反向(见 §3) 需要保留 `[N, M]` 的 `A` 矩阵;长序列下显存爆炸。
FlashAttention 的解法: **正向只保存每行的标量 `(m, l)`,反向时在每个 tile 内重新算一遍 softmax**。

```
正向保存:                反向重算:
  O   [L, D]              for each Kⱼ, Vⱼ tile:
  m   [L]   (running max)   重算  S_ij = Qᵢ Kⱼᵀ / √D
  l   [L]   (running sum)   重算  A_ij = exp(S_ij - mᵢ) / lᵢ   ← 用保存的 mᵢ, lᵢ
                            标准 chain rule:
                              dV_j  += A_ijᵀ · dO_i
                              dA_ij  = dO_i · V_jᵀ
                              dS_ij  = softmax_backward(A_ij, dA_ij)
                                     = A_ij ⊙ (dA_ij - D_i)
                                     其中  D_i = (dO_i · O_iᵀ).sum(-1)    ← 一个标量
                              dQ_i  += dS_ij · K_j / √D
                              dK_j  += dS_ijᵀ · Q_i / √D
```

**关键 trick: 那个 `D_i = (dO·Oᵀ).sum`** —— 朴素 softmax 反向要 `g·s` 内积,FlashAttention 里
`g = dA`, `s = A` 都不显式存。但代入 `O = A·V` 可以证明 `(dA·A).sum == (dO·O).sum` 是同一个标量,
**用已经存好的 `O, dO` 就能算出来**,完全不需要 materialize `A`。

### 正反向显存对比

| 方案 | 正向显存 | 反向额外显存 | 总计 |
|---|---|---|---|
| 朴素 attention | `O(N²)` 存 `S`, `A` | 0 (复用前向) | `O(N²)` |
| FlashAttention | `O(N·D)` 存 `O` + `O(N)` 存 `(m, l)` | tile 缓冲 `O(b²)` | **`O(N·D)`** |
| 序列 16k, D 128 | ~256 MB | — | ~8 MB (`32×`) |

### IO 复杂度证明梗概(为什么是 O(N²·D²/SRAM))

FlashAttention 论文给出: 朴素 attention 的 HBM 访问量 `O(N·D + N²)`,FlashAttention `O(N²·D² / SRAM_size)`。
当 `D ≪ √SRAM` 时(典型 D=128, SRAM=100KB → √SRAM≈300),Flash 的 HBM 访问 **少 1-2 个数量级**。
这正是 FlashAttention 实际**加速 2-4×** 的来源(不是 FLOPs,而是 IO)。

### 高频追问

| Q | A |
|---|---|
| 为什么 softmax 最少 2 pass? | 数值稳定要先有 `m_N` 才能减;`d'_i` 的 surrogate 让 `m_i` 替代 `m_N`,但 `a_i` 仍依赖 `m_N` |
| 为什么 FlashAttention 能 1 pass? | 目标是 `O = A·V` 而非 `A`,对 `O` 再施一次 surrogate trick (`o'_i`) 消去 `m_N` 依赖 |
| `exp(m_{i-1} - m_i)` 在做什么? | 把"用旧 max 累加的量"**修正成"用新 max 累加的"**,保持数值等价 |
| FlashAttention 为什么快?(常见误解) | **不是减 FLOPs,是减 HBM 读写**。tiling 让中间矩阵留在 SRAM |
| 反向怎么实现? | 只存 `(m, d')`,反向重算 softmax + V 乘积;FLOPs 换显存 |
| FlashAttention 2 / 3 改了什么? | v2: 交换内外循环 + 减 non-matmul FLOPs;v3: Hopper warp-specialization + FP8 异步 |
| 为什么 Flash 对 long-context LLM 关键? | SRAM 占用与 L 无关,只与 `b, D` 有关 → 64k context 也能稳跑 |

---

## 5. Reward Model — Bradley-Terry pairwise

> **Key Insight**: 不要让人打绝对分(尺度不一致),只让人选 A/B 哪个好。
> Bradley-Terry: `P(y_w ≻ y_l) = σ(r_w − r_l)`,损失 = `−log σ(r_w − r_l)`。

### 实现层次

```
            ┌──────── chosen 分支 ────────┐    ┌──────── rejected 分支 ────────┐
            │                              │    │                                │
input:      prompt + y_w  [B, T]                prompt + y_l  [B, T]
            │                                   │
            ▼ backbone (causal LM)              ▼ backbone (共享权重,前向 2 次)
            h_w [B, T, hidden]                  h_l [B, T, hidden]
            │                                   │
            ▼ 取最后一个 token (左 padding 保证)  ▼ 取最后一个 token
            h_w[:, -1, :] [B, hidden]           h_l[:, -1, :] [B, hidden]
            │                                   │
            ▼ reward_head (Linear → 1)          ▼ reward_head (同一个)
            r_chosen   [B]                      r_rejected [B]
                │                                   │
                └─────────────┬─────────────────────┘
                              ▼
                Bradley-Terry pairwise loss
                L = −log σ(r_chosen − r_rejected)
                              ▼
                          backward
                  (训练 backbone + reward_head)
```

### 公式

每个回答 `y` 有隐式效用 `r(x, y)`,偏好概率:

```
P(y_w ≻ y_l | x) = σ(r(x, y_w) − r(x, y_l))
```

最大化对数似然 → 损失:

```
L = -log σ(r(x, y_w) − r(x, y_l))
```

### 代码骨架

```python
class RewardModel(nn.Module):
    def __init__(self, backbone, hidden_dim):
        super().__init__()
        self.backbone = backbone
        self.reward_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.backbone(x)              # [B, T, hidden]
        last = h[:, -1, :]                # 最后一个 token(causal 模型看到整句)
        return self.reward_head(last).squeeze(-1)

def reward_model_loss(r_chosen, r_rejected):
    return -torch.log(torch.sigmoid(r_chosen - r_rejected)).sum()
```

**工程细节**:
- 用**左 padding**(`encode_batch_left_pad`),保证 `h[:, -1, :]` 是真实结尾
- Backbone 通常 init from SFT checkpoint,reward_head 随机初始化
- 训练时 chosen / rejected 共享 backbone 前向,只跑 2 次

### Bradley-Terry 的等价 logistic 写法

`-log σ(r_w - r_l)` 其实就是**二分类 logistic loss**,把"`y_w` ≻ `y_l`"当作 label = 1。
等价于在 reward 差上做二分类:

```
σ(r_w - r_l)   = P(chosen wins)         ← 模型预测
target         = 1                        ← 数据标签
binary CE      = -log σ(r_w - r_l)       ← 标准 logistic loss
```

→ 任何稳定的 logistic 损失 trick(如 label smoothing)都能直接套上。

### 奖励的尺度问题(reward scaling)

RM 训完后,raw reward 的**尺度任意**(取决于 backbone 初始化、训练步数等)。
不归一化直接送 PPO 会出问题:**reward 数量级波动 → KL 系数失效 → policy 训练不稳**。

常用归一化:

```python
# 方法 1: 跑 RM 在大批 prompt 上算 mean/std,做 z-score (InstructGPT 做法)
rewards = (rm(x) - reward_mean) / reward_std

# 方法 2: per-batch 归一化(简单)
rewards = (r - r.mean()) / (r.std() + 1e-8)

# 方法 3: 取 baseline (减 SFT 模型在同 prompt 上的 reward)
shaped = rm(policy_out) - rm(sft_out)
```

### Reward Hacking 是什么

policy 学会"骗"RM 拿高分但人类不喜欢的行为。常见模式:

| 模式 | 例子 | 修法 |
|---|---|---|
| **长度作弊** | RM 训练时 chosen 倾向更长 → policy 全写超长答案 | length penalty / SimPO |
| **格式作弊** | 加 "Sure!" "Of course." 这种 RM 喜欢的开头 | reward shaping / 多样化训练数据 |
| **拒答作弊** | 模糊回答避免被 RM 惩罚 | helpfulness vs harmlessness 多 RM |
| **重复回应** | 复述 prompt 占 token | repetition penalty in reward |
| **数学幻觉** | 装作算了一长串 (RM 看格式不看正确性) | 用 verifier 替代 RM (GRPO 数学路线) |

→ 这就是为什么 PPO 要加 KL 惩罚 `β·KL(π‖π_ref)`:**约束 policy 不要为了高 reward 偏离 SFT 太远**。

### 多目标 Reward Model (Helpfulness vs Harmlessness)

Anthropic Constitutional AI / Claude 路线:**两个 RM**,加权合并:

```
R = α · R_helpful + β · R_harmless
```

`α, β` 是动态调的: 检测到不安全输入 → β 提高;一般问答 → α 主导。
也可以训成**单个 RM 输出多维标量**,然后融合层学加权。

### Pairwise vs Pointwise vs Listwise

| 范式 | 数据 | Loss | 典型 |
|---|---|---|---|
| **Pointwise** | (prompt, response, score 1-5) | MSE | 早期人工评分,人标不一致 |
| **Pairwise (BT)** | (prompt, chosen, rejected) | `-log σ(r_w - r_l)` | InstructGPT, Claude |
| **Listwise (Plackett-Luce)** | (prompt, ranked list of k) | k 选 1 的概率乘积 | 排序信息更密集 |

Pairwise 工业标配 —— 标注成本 / 一致性 / 算法成熟度三方面最优。

### 评估 RM 的指标

**Accuracy on held-out pairs**: 给定 (chosen, rejected),RM 算出 `r_w > r_l` 的比例。
经验值: 训得好的 RM 在新数据上 ~65-75%(注意:这个 ceiling 比想象低,因为偏好本身不一致)。

**KL-controlled win rate**: 让 PPO 训完的 policy 在固定 KL 距离 `β` 下,
对比 baseline (SFT 或上一版 policy) 的人类胜率。这才是 RM 真正的"线上指标"。

### 高频追问

| Q | A |
|---|---|
| 为什么取最后一个 token? | Causal mask 下,最后一个位置的 hidden 看到了完整 prompt + response |
| Reward hacking 是什么? | Policy 学会钻 RM 漏洞(长度/格式/拒答作弊),拿到高 reward 但人类不喜欢。修: 加 KL,length penalty,多目标 RM |
| RM 和 DPO 的关系? | DPO 在数学上等价于"训 RM 然后 PPO",但消去 RM 这一步直接优化 policy(见 §10) |
| 多个 chosen 怎么处理? | (a) 列出所有 pairwise (chosen, rejected) 对逐对算损失 (b) 用 listwise Plackett-Luce |
| Reward 要不要归一化? | 必须。raw reward 尺度任意 → PPO 不稳。z-score 或减 SFT baseline |
| RM 训多少数据? | InstructGPT 用 ~33k pairs;开源 (Anthropic HH-RLHF) ~170k;实际工业线 100k-1M 量级 |
| 为什么 BT 假设可疑但还能用? | BT 假设 reward 是固定标量函数,实际人类偏好可能 cyclic。但平均下来够好 |
| RM 能直接做 inference rank 吗? | 可以(best-of-N 采样,挑 r 最高的)。计算成本: N 倍 inference |
| Pairwise 数据怎么收集? | 同 prompt 不同 temperature 采 2 个 → 人标 chosen/rejected。或 GPT-4 当 judge (RLAIF) |
| RM 会不会"偏向自己生成"? | 会。如果 RM 训练数据是 SFT 模型采样的,RM 会偏好 SFT 风格 → 限制了 PPO 的探索 |

---

## 6. RL Fine-tuning — 策略梯度 + 组内归一化

> **Key Insight**: 策略梯度的方差太大,必须减 baseline。
> **组内均值**当 baseline(同 prompt 才可比),既降方差又不用单独训 value 网络。

### 实现层次

```
                  ┌─────────── 采样阶段 ───────────┐
                  │                                 │
   prompt [B] ────┤  policy.sample × G             │
                  │  (一个 prompt 采 G 个 rollout)  │
                  └────────────────┬───────────────┘
                                   ▼
              rollouts [B, G, T]   ←─── 每个回答的 token 序列
                                   │
                  ┌────────────────┼────────────────┐
                  ▼                                 ▼
       Reward Model 打分                  policy 重新前向 (要梯度)
              rewards [B, G]                  logits [B, G, T, V]
                  │                                 │
                  ▼ compute_group_advantage         ▼ gather + (-log_softmax)
       (r - mean(-1)) / (std(-1) + ε)      neg_logp [B, G, T]
       advantage [B, G]                            │
                  │                                 │
                  └──────────────┬──────────────────┘
                                 ▼
                policy_gradient_loss
                = (neg_logp * advantage[..., None]).sum(-1).mean(1).sum()
                  − entropy_weight · entropy
                                 │
                                 ▼
                            backward → 更新 policy
```

### 为什么要 advantage,不直接用 reward?

```
∇L = -E[ ∇logπ(a) · R ]    ← 方差大
∇L = -E[ ∇logπ(a) · (R − b) ]  ← 减 baseline b 不改变期望但降方差
```

`b` 选什么?
- 经典: value network `V(s)`(actor-critic)
- **GRPO 路线**: 同 prompt 的**组内均值**(省一个 value 网络)

### 代码骨架

```python
def compute_group_advantage(rewards):     # [B, G]
    return (rewards - rewards.mean(-1, keepdim=True)) / \
           (rewards.std(-1, keepdim=True) + 1e-8)

def policy_gradient_loss(neg_logp, advantage, entropy, entropy_weight=1e-3):
    # neg_logp: [B, G, T]  ; advantage: [B, G]
    adv_loss = neg_logp * advantage.unsqueeze(-1)
    return adv_loss.sum(-1).mean(1).sum() - entropy_weight * entropy
```

**熵正则**: 鼓励探索,防止过早坍缩到一种回答模式。

### 策略梯度的"血脉" (REINFORCE → A2C → PPO → GRPO)

```
1992  Williams  REINFORCE        ∇L = -E[∇logπ · R]            ← 纯朴素,方差爆炸
                                                              
1999  Sutton    Policy Gradient  ∇L = -E[∇logπ · A]            ← 引入 advantage
      Theorem                       A = R - b(s)                   降方差
                                                              
2016  Mnih      A2C/A3C          A = Σ γ^k r_{t+k} - V(s_t)    ← 用 value 网络当 baseline
                                                              
2017  Schulman  PPO              加 ratio + clip + KL            ← 防策略跑飞
                                                              
2024  DeepSeek  GRPO             去掉 value,组内 z-score 当 A   ← 省一个模型
```

每代都在解决前一代的痛点。今天 RLHF 实际用的就是 **PPO** 和 **GRPO** 两条线。

### 完整训练循环 (伪代码)

```python
def rl_finetune(policy, ref, rm, prompts, epochs):
    for epoch in range(epochs):
        for batch_prompts in prompts:
            # ===== 采样阶段 (no grad) =====
            with torch.no_grad():
                rollouts = []
                for _ in range(group_size):                # G 个 rollout per prompt
                    out = policy.generate(batch_prompts)
                    rollouts.append(out)
                rewards = rm(rollouts)                     # [B, G]

            # ===== 优势归一化 =====
            advantage = (rewards - rewards.mean(-1, keepdim=True)) / \
                        (rewards.std(-1, keepdim=True) + 1e-8)

            # ===== 训练阶段 (需要 grad) =====
            for _ in range(ppo_epochs):                    # PPO 多轮利用同一批 rollout
                logits = policy(rollouts)                  # 重新前向
                neg_logp = F.cross_entropy(logits, rollouts, reduction='none')
                entropy  = compute_entropy(logits)

                loss = (neg_logp * advantage.unsqueeze(-1)).sum(-1).mean() \
                     - entropy_weight * entropy

                # 可选: KL 惩罚
                kl = compute_kl(policy, ref, rollouts)
                loss += kl_coef * kl

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
```

### On-Policy vs Off-Policy 的微妙差别

**On-policy**: 用**当前** π 采样的数据更新 π。每次梯度步后,旧数据就**作废**。
**Off-policy**: 用**别的** π (历史 / SFT / 别人的) 采样的数据更新 π。需要 importance sampling 修正。

| 范式 | 数据来源 | 修正 | 复杂度 | 算法 |
|---|---|---|---|---|
| 纯 on-policy | π_θ (当前) | 无 | 慢 (一步一采) | REINFORCE, vanilla PG |
| **近 on-policy** | π_θ_old (几步前) | importance ratio | 中 (一批数据多步) | **PPO, GRPO** |
| 纯 off-policy | replay buffer / SFT | 强 IS / Q-learning | 难 (分布外严重) | DDPG, SAC, **DPO** |

PPO 和 GRPO 的妙处: 用 importance ratio 复用同一批 rollout 训多步,**省了 90%+ 的采样成本**。

### 为什么减 baseline 不改变期望(严格证明)

```
E_π[∇log π(a|s) · b(s)] = b(s) · ∫ π(a|s) · ∇log π(a|s) da
                       = b(s) · ∫ ∇π(a|s) da                ← log 求导技巧
                       = b(s) · ∇ ∫ π(a|s) da
                       = b(s) · ∇ 1 = 0
```

`b(s)` 只要不依赖 action `a`(可以依赖 state `s`)都不改期望;但**会改方差**。
**最优 baseline**: `b*(s) = E[(∇log π)² · R] / E[(∇log π)²]` (但实践用 V(s) 已经够好)。

### 熵正则的具体作用

```python
loss = -E[logπ · A] - β_ent · H(π)        # H 越大,π 越分散
```

| β_ent | 效果 |
|---|---|
| 0 | 容易 mode collapse,最后只剩 1-2 个高 reward 答案 |
| 1e-3 ~ 1e-2 | 典型范围 (InstructGPT 用 1e-2 量级) |
| > 0.1 | π 太接近均匀分布,根本学不到偏好 |

对 LLM,熵指 token 级别 H = -Σ p_v · log p_v 在 vocab 维度上的均值。

### 高频追问

| Q | A |
|---|---|
| 为什么减 baseline 不改变期望? | `E[∇logπ · b] = b · ∇∫π = 0`,只要 b 不依赖 action a(可依赖 state s) |
| 为什么用组内均值? | 同 prompt 下 reward 才可比;跨 prompt 平均无意义,reward 尺度差太大 |
| on-policy vs off-policy? | on: 当前 π 采样,数据用 1 次;近 on: PPO/GRPO 用 IS ratio 重用;off: DPO 直接用离线偏好数据 |
| 熵正则太强 / 太弱? | 太强 → 不收敛,policy 一直瞎采;太弱 → 早期 mode collapse |
| 和 GRPO 关系? | 这一节就是 GRPO 的"裸版"(没有 PPO 的 ratio + clip);加上后变成 §9 |
| REINFORCE 为什么不实用? | 没有 baseline → 方差极大;每个梯度步都要 fresh 采样 → 慢 |
| 为什么 LLM 不用 actor-critic 的 V(s)? | (a) state 是整个 token 序列,V(s) 很难训 (b) GRPO 用组内 baseline 已经够好 |
| group size G 的影响? | G 太小 (≤2) 组内 std 不稳定;G 太大 (>64) 显存爆且边际收益递减;典型 8-16 |
| 奖励信号为什么稀疏(只在 EOS 给)? | 用 RM 打分时,RM 是 sequence-level 输出。token 级 reward 需要 step-wise RM 或 process RM |
| token-level 还是 sequence-level loss? | 序列级 advantage 广播到所有 token 是主流 (GRPO/PPO);DPO 直接序列级 logp 差 |

---

## 7. GAE — 广义优势估计

> **Key Insight**: 用 `λ` 在「**低偏差高方差**(蒙特卡洛回报)」和「**高偏差低方差**(一步 TD)」之间插值。
> 关键计算:**从后往前递推** `A_t = δ_t + γλ·A_{t+1}`。

### 实现层次

```
   输入: rewards [B, T] , values [B, T] (来自 value head)
                                │
   ════════ 从后往前递推 ════════
                                │
   t = T-1 ──┐  next_V = 0      │
             │  δ = r_{T-1} - V_{T-1}
             │  A_{T-1} = δ
             ▼
   t = T-2 ──┐  next_V = V_{T-1}
             │  δ = r_{T-2} + γ·V_{T-1} - V_{T-2}
             │  A_{T-2} = δ + γλ·A_{T-1}
             ▼
   ...        ↓ 不断累乘 γλ
   t = 0   ──┐  next_V = V_1
             │  δ = r_0 + γ·V_1 - V_0
             │  A_0 = δ + γλ·A_1
             ▼
   advantages [B, T]
                                │
                                ▼
   returns = advantages + values   (用于 value head 的 MSE)
   
   极端情况:
     λ=0  ⇒  A_t = δ_t                   (一步 TD,低方差高偏差)
     λ=1  ⇒  A_t = Σ γ^k·δ_{t+k}         (≈MC return − V,高方差低偏差)
```

### 公式

```
TD 残差:  δ_t = r_t + γ·V(s_{t+1}) − V(s_t)
GAE:      A_t = δ_t + γλ·A_{t+1}    ← 从后往前递推
```

### 代码骨架

```python
def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    advantages = torch.zeros_like(values)
    lastgae = 0.0
    T = rewards.shape[-1]
    for t in reversed(range(T)):
        next_value = values[:, t+1] if t < T-1 else torch.zeros_like(values[:, t])
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        lastgae = delta + gamma * lam * lastgae
        advantages[:, t] = lastgae
    return advantages
```

**数值验证**: `γ = λ = 1` 时 GAE 应该等于 `cumsum(rewards, from end) - values`。

### n-step Returns 复习 (GAE 的前身)

回报的几种估计:

```
1-step TD:   Â^(1)_t = r_t + γ·V(s_{t+1}) − V(s_t)               (= δ_t)
2-step:      Â^(2)_t = r_t + γ·r_{t+1} + γ²·V(s_{t+2}) − V(s_t)
k-step:      Â^(k)_t = Σ_{l=0}^{k-1} γ^l·r_{t+l} + γ^k·V(s_{t+k}) − V(s_t)
∞-step (MC): Â^(∞)_t = Σ_{l=0}^∞ γ^l·r_{t+l} − V(s_t)
```

**偏差-方差权衡**:
- k 小: bootstrapping V 多,V 不准 → **偏差大**;但 Var 小,因为只用了 k 步真实 reward
- k 大: 用更多真实 reward → **偏差小**;但 Var 大,因为长 trajectory 累加随机性

### GAE = n-step Advantages 的指数加权平均

Schulman 2015 的 GAE 定义:

```
A^GAE(γ,λ)_t = (1-λ) · Σ_{k=1}^∞ λ^{k-1} · Â^(k)_t       ← 各 k-step 的指数加权
```

代入 Â^(k) 展开,**化简后等于 δ_t 的指数加权**:

```
A^GAE_t = Σ_{l=0}^∞ (γλ)^l · δ_{t+l}                    ← 关键化简式!
        = δ_t + γλ·δ_{t+1} + (γλ)²·δ_{t+2} + ...
```

这就是为什么 GAE 的递推这么简单: **`A_t = δ_t + γλ · A_{t+1}`**(一行)。

### 两个极端的严格推导

**λ = 0**: 只保留 k=1 项
```
A^GAE_t = δ_t = r_t + γ·V(s_{t+1}) − V(s_t)             ← 一步 TD
```
**bias 高**(完全信任 V),**variance 低**(只有一项随机)。

**λ = 1**: 所有项加权
```
A^GAE_t = Σ_{l=0}^∞ γ^l · δ_{t+l}
        = Σ_{l=0}^∞ γ^l · r_{t+l} − V(s_t)              ← MC return − V
```
**bias 低**(用真实 reward),**variance 高**(累加无限项)。

→ 选 `λ ∈ (0, 1)` 在两者间插值,典型 `λ = 0.95`(略偏蒙特卡洛)。

### γ 和 λ 的实际差别

| 超参 | 含义 | 典型 | 调谁 |
|---|---|---|---|
| **γ (折扣)** | 未来 reward 衰减率 | 0.99 | 任务相关:长 horizon 任务接近 1,短 horizon 可低些 |
| **λ (GAE)** | n-step 加权 | 0.95 | bias-variance 调优,通常固定 0.9-0.97 |

LLM RLHF 里: γ ≈ 1(序列只到 EOS,无需折扣)、λ ≈ 0.95-1.0。

### Value Head 训练

Actor-critic 同时优化 policy 和 value:

```python
class PolicyValueModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone   = backbone
        self.policy_head = nn.Linear(d, vocab_size)
        self.value_head  = nn.Linear(d, 1)          # 标量 V(s)

    def forward(self, x):
        h = self.backbone(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

# 训练
logits, V = model(rollouts)
A_gae     = compute_gae(rewards, V, gamma=0.99, lam=0.95)
returns   = A_gae + V                                 # 即 V_target

policy_loss = -mean(logp * A_gae.detach())            # advantage detach!
value_loss  = F.mse_loss(V, returns.detach())         # returns detach!
loss = policy_loss + 0.5 * value_loss
```

**坑**: A 和 returns 都要 `.detach()`,否则 value 梯度会污染 policy 梯度。

### 稀疏 reward 是怎么传回去的

LLM RLHF 中,RM 只在 EOS 给一个 scalar reward。其他 token reward = 0。
GAE 把这个唯一的非零 reward 折扣回传:

```
r = [0, 0, 0, ..., 0, R_final]              # 只有最后一步
γ = 1, λ = 1:
A_t = R_final - V(s_t)                       # 所有 t 都拿到 R_final 信号

γ < 1, λ < 1:
A_t = γ^(T-t-1) · R_final + bootstrap(V)
```

→ 即使 reward 极稀疏,GAE + V 也能给每个 token 一个 advantage 信号。

### 高频追问

| Q | A |
|---|---|
| γ 和 λ 各管什么? | γ 是回报折扣(管"未来 reward 看多远");λ 管"信任 value 估计 vs 真实回报"的程度 |
| 为什么从后往前? | 递推式 `A_t = δ_t + γλ·A_{t+1}` 依赖 t+1 的值,所以反向遍历 |
| GAE 怎么从 n-step 推出来? | n-step advantages 的指数加权平均,化简得 `Σ (γλ)^l · δ_{t+l}`,等价于 `A_t = δ_t + γλ·A_{t+1}` |
| Actor 和 Critic 为什么共享 backbone? | 省显存 + 表征共享;value head 只是个 `Linear(dim, 1)` |
| Value loss 怎么算? | `MSE(V(s_t), returns_t)`,`returns = advantages + values` |
| value 和 reward 的区别? | reward 是即时奖励 r_t;value V(s_t) 是从 s_t 起的预期总折扣回报 |
| advantage 为什么要 detach? | 进 policy loss 时要,否则 value 反向梯度会污染 policy |
| 稀疏 reward (只有最后一步)怎么办? | GAE 会把最后一步的 reward 沿 γ^t 折扣回传到所有时间步 |
| GAE 在 LLM 里是 token 级还是序列级? | token 级:每个 token 有自己的 A_t。但 reward 一般只在 EOS 给 |
| 为什么 GRPO 不用 GAE? | GRPO 砍掉 value 网络,直接用组内 z-score。代价: 没法 token 级精细化 |

---

## 8. PPO — Clipped Surrogate + KL

> **Key Insight**: 用 importance sampling ratio 修正 off-policy + **clip 防止策略跑飞** + KL 约束别偏离 SFT 太远。
> 4 个模型 (policy / ref / reward / value) 是 PPO 工程痛点。

### 实现层次

```
══════════ RLHF 三步走 ══════════
   Step 1: SFT             → π_SFT
   Step 2: 训 Reward Model → RM   (见 §5)
   Step 3: PPO 优化 π             ← 本节

══════════ PPO 一步更新的 4 个模型 ══════════

                        ┌──────────────────────────────┐
                        │  采样: π_old.sample(prompt)   │
                        │  得到 (a, logp_old) 缓存       │
                        └────────────┬──────────────────┘
                                     │ rollout
                                     ▼
   ┌───────────────┬─────────────────┴────────────────┬───────────────┐
   ▼               ▼                                  ▼               ▼
policy(πθ)      ref(π_SFT)                     reward(RM)         value(V)
要梯度           冻结                            冻结                要梯度
   │               │                                  │               │
logp_new         logp_ref                          rewards          V(s)
   │               │                                  │               │
   │     KL(π_θ ‖ π_ref) per token                    │       GAE(rewards, V)
   │               │                                  │               │
   │               └─── β·KL ───┐                     │               │
   │                            ▼                     │               ▼
   │                  shaped_reward = r − β·KL        │           advantages
   │                                                  │
   └────── ratio = exp(logp_new − logp_old) ──────────┘
                       │
            ┌──────────┴──────────┐
            ▼                     ▼
       ratio·A          clip(ratio, 1±ε)·A
            └───────── min() ─────┘
                       ▼
                   −mean = PPO loss
                       ▼
                   backward → 更新 policy + value
```

### 公式

```
ratio_t = exp(logπ_θ(a_t) - logπ_old(a_t))    # 重要性采样比
L = -E[ min( ratio · A, clip(ratio, 1-ε, 1+ε) · A ) ]
```

**为什么 clip**: 防止单步更新太大让策略跑飞。`ε = 0.2` 是经典值。

### KL 惩罚

```
advantage = reward - β · KL(π_θ ‖ π_ref)
```

`π_ref` = SFT 的冻结副本,**全程不更新**。这个 KL 防止 policy 偏离语言模型本能(避免 reward hacking)。

### 代码骨架

```python
def ppo_loss(logp_new, logp_ref, advantages, clip=0.2):
    ratio = torch.exp(logp_new - logp_ref.detach())
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantages
    return -torch.mean(torch.min(unclipped, clipped))

def apply_kl_penalty(rewards, logp_new_dist, logp_ref_dist, kl_coef=0.01):
    kl = (torch.exp(logp_new_dist) * (logp_new_dist - logp_ref_dist.detach())).sum(-1)
    return rewards - kl_coef * kl
```

### Clip 的可视化(理解 min 的含义)

```
                Advantage > 0 (好动作,想拉高 logp)
                ┌──────────────────────────────┐
                │  ratio 增大方向                │
                │                              │
   ratio·A      │      ╱                       │ ← unclipped:无限抬升
                │     ╱                        │
                │    ╱                         │
                │   ╱                          │
   1+ε ─────────│──╱──────────── clipped       │ ← clipped:被天花板限制
                │ ╱            ────────────    │
                │╱                             │
   ratio=1 ─────┼──────────────────────────────│
                ↑                              ↓
                │  ratio 减小方向                │
                │                              │
              对 A > 0: 选 min → 取被 clip 的(保守,防过度拉高)


                Advantage < 0 (差动作,想压低 logp)
                ┌──────────────────────────────┐
   ratio=1 ─────┼──────────────────────────────│
                │\                             │
                │ \                            │
                │  \                           │
   1-ε ─────────│---\---────────── clipped     │
                │    \           ─────────     │
                │     \                        │
                │      \                       │
   ratio·A      │       \                      │ ← unclipped:无限下压
                │        \                     │
                └──────────────────────────────┘

              对 A < 0: 选 min → 取没被 clip 的(继续压低,不饶过坏动作)
```

**核心直觉**: 对正 A,**限上界防止过度乐观**;对负 A,**不限下界保留惩罚力度**。
这是 PPO 比 TRPO 简洁但等效的关键。

### Multi-Epoch 训练: 为什么能复用同一批 rollout

朴素 on-policy: 一批数据用 1 个梯度步,扔掉,重新采样 — 极慢。
PPO: 一批数据用 **k 个 epoch**(典型 k=4)的梯度更新,通过 `ratio` 修正 off-policy 偏差:

```python
for batch in batches:
    rollouts, logp_old = sample(policy)          # 采一次
    for ppo_epoch in range(4):                    # ← 复用 4 次
        logp_new = policy(rollouts)
        ratio = exp(logp_new - logp_old)         # ratio 起飞表示 policy 漂太远
        loss = clipped_surrogate(ratio, A)
        loss.backward(); step()
```

`ratio` 越接近 1 → 策略和 old 没差多远 → IS 估计准;`ratio` 离 1 远 → clip 生效避免崩。
**多 epoch 是 PPO 比 REINFORCE 快 10× 的根本原因**。

### Token-level vs Sequence-level PPO

LLM 里的 a_t 是 token,不是动作。两种实现:

```
Token-level (主流):
   logp(a_t)        每个 token 单独算 logπ(token | prefix)
   ratio_t          per-token ratio
   loss = mean_t mean_seq min(ratio_t · A_t, clip)

Sequence-level (DPO 风格):
   logp(seq) = Σ_t logp(a_t)
   ratio = exp(logp_new(seq) - logp_old(seq))
   loss = min(ratio · A_seq, clip)
```

Token-level 更细致,能给 GAE 提供 per-token A_t;sequence-level 更稳定,梯度信号简洁。

### 4 个模型 (PPO 工程痛点)

| 模型 | 角色 | 是否更新 | 显存代价 |
|---|---|---|---|
| **policy** | 主角,被 RL 训练 | ✅ | 1× + 优化器 ~5× |
| **ref** (= SFT 冻结) | 算 KL 惩罚 | ❌ | 1× (inference only) |
| **reward** | RM 给分 | ❌ | 1× (inference only) |
| **value** | GAE baseline (actor-critic) | ✅ (共享 backbone) | 0.1× (只是个 head) |

总显存 ≈ **3-4 倍模型大小**。这就是为什么 RLHF PPO 难做 70B+ 模型,GRPO 把 value 砍了缓解。

### PPO 完整 loss

```python
def ppo_total_loss(logp_new, logp_old, A, V, returns, dist_new, dist_ref):
    # 1. Policy loss (clipped surrogate)
    ratio   = torch.exp(logp_new - logp_old)
    policy  = -torch.min(ratio * A,
                          torch.clamp(ratio, 1-ε, 1+ε) * A).mean()

    # 2. Value loss
    value   = F.mse_loss(V, returns)

    # 3. Entropy bonus
    entropy = -(dist_new.exp() * dist_new).sum(-1).mean()

    # 4. KL penalty (optional, 也可以放在 reward shaping 里)
    kl      = (dist_new.exp() * (dist_new - dist_ref)).sum(-1).mean()

    return policy + c_v * value - c_ent * entropy + β_kl * kl
```

InstructGPT 经验值: `c_v = 0.5, c_ent = 0.01, β_kl = 0.1, ε = 0.2`。

### 高频追问

| Q | A |
|---|---|
| 为什么要 clip? | 单步更新太大会让 ratio 离 1 太远 → importance sampling 估计崩;clip 提供"信任域" |
| ε 太大/太小? | 太大失去 clip 意义,等同朴素 PG;太小更新太保守,收敛慢。0.2 是经验最优 |
| 为什么需要 KL? | RM 是不完美的代理,policy 全力对抗 RM 会 reward hacking;KL 限制偏离 SFT |
| Min(unclipped, clipped) 的直觉? | 选**保守**的那一边:正 A 限上界、负 A 不限下界(本来就要压) |
| Ratio 用 logp_new - logp_old 不是 logp_ref? | `old` 是**采样时的 policy 快照** (每个 minibatch 之前 detach);ref 是 SFT 副本,做 KL 用 |
| PPO 为什么能多 epoch? | 用 ratio 修正 off-policy,只要 ratio 没漂太远(被 clip 控住),数据可复用 4-8 次 |
| token-level 还是 sequence-level? | LLM 通常 token-level,提供 per-token advantage;sequence-level 适合稀疏 reward |
| value head 怎么初始化? | (a) 从 policy 复用 backbone + 随机初始 head (b) 单独训几个 epoch 让 value 先 warm up |
| PPO vs TRPO? | TRPO 用二阶 + KL 信任域 (严格但贵);PPO 用一阶 + clip 近似 (便宜 90% 效果) |
| 4 个模型显存怎么压? | (a) ref 用 LoRA delta (b) value 共享 backbone (c) reward 量化 (d) 用 GRPO 砍 value |
| `ratio.mean()` 应该等于多少? | 应该 ≈ 1。> 1.5 或 < 0.5 表示 policy 漂太远,该早停或减小 lr |

---

## 9. GRPO — PPO 砍掉 Value 网络

> **Key Insight**: 同 prompt 的 G 个 rollout 组内 z-score 就是个**很好的 baseline**,不再需要 value 网络。
> 省一个模型 = 省 25% 显存,适合**可验证 reward**(数学/代码)。

### 实现层次

```
══════════ PPO vs GRPO 差在哪 ══════════

PPO:    policy + ref + reward + VALUE  (4 个模型, GAE 算 A)
GRPO:   policy + ref + reward          (3 个模型, 组内归一化算 A)
                              ↑
                       省掉的就是 value


══════════ GRPO 一步更新 ══════════

  prompt
     │
     ▼ policy.sample × G             一个 prompt 采 G 个 rollout
  rollouts [B, G]
     │
     ├──► RM 打分 ─────► rewards [B, G]
     │                       │
     │                       ▼  组内 z-score (不需要 value!)
     │             advantage = (r - r.mean(-1)) / (r.std(-1) + ε)
     │                       │
     │                       ▼
     │             advantage [B, G]
     │
     ├──► policy 重新前向 ─► logp_policy [B, G]
     │
     └──► ref (本轮 deepcopy) ─► logp_ref [B, G]
                                     │
                  ratio = exp(logp_policy - logp_ref)
                                     │
                  ┌──────────────────┴──────────────────┐
                  ▼                                     ▼
            ratio · A                       clip(ratio, 1±ε) · A
                  └──────────── min() ──────────────────┘
                                ▼
                          −mean = GRPO loss
                                ▼
                          backward → 更新 policy
```

### 公式

```
A_i = (r_i - mean(r)) / (std(r) + eps)     ← 组内 z-score
ratio_i = exp(logπ_θ(rollout_i) - logπ_ref(rollout_i))
L = -E[ min(ratio_i · A_i, clip(ratio_i, 1-ε, 1+ε) · A_i) ]
```

`π_ref` 不再是 SFT 副本,而是**这一轮 RL step 开始时 deepcopy 的快照**(每步刷新)。

### 代码骨架

```python
def grpo_loss(logp_policy, logp_ref, rewards, epsilon=0.2):
    advantage = (rewards - rewards.mean(-1, keepdim=True)) / \
                (rewards.std(-1, keepdim=True) + 1e-8)
    ratio = torch.exp(logp_policy - logp_ref.detach())
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage
    return -torch.minimum(unclipped, clipped).mean()
```

### GRPO vs PPO 对比表

| 维度 | PPO | GRPO |
|---|---|---|
| Value 网络 | 必须 | 砍掉 |
| Advantage | GAE(rewards, V) | 组内归一化 reward |
| Ref policy | SFT 副本(冻结) | 每步 deepcopy 的快照 |
| 显存 | 4 模型 | 3 模型(policy + ref + RM/verifier) |
| 适合任务 | 通用 RLHF | 有可验证 reward(数学 / 代码) |
| 代表 | InstructGPT, ChatGPT 早期 | **DeepSeek-R1, DeepSeek-Math** |

### GRPO vs RLOO vs Vanilla PG

| 算法 | Baseline | Off-policy 修正 | 代表 |
|---|---|---|---|
| **REINFORCE** | 无 | 无 | 教科书 |
| **Vanilla PG with mean** | 全 batch 均值 | 无 | 早期 |
| **RLOO (REINFORCE Leave-One-Out)** | 同 prompt **其他 G-1 个** rollout 的均值 | 无 (on-policy) | Cohere 用 |
| **GRPO** | 组内 z-score (mean + std 归一化) | ratio + clip | DeepSeek |
| **PPO with value** | V(s) | ratio + clip | InstructGPT |

GRPO 比 RLOO 多了 std 归一化(让 advantage 尺度稳定)+ PPO 的 clip(允许 multi-epoch)。

### DeepSeek-R1 的具体配置

DeepSeek-R1 论文 (2025) 用 GRPO 训了一个**仅靠 RL 就能 reasoning** 的模型:

```
配置:
  Group size G = 64        ← 比 InstructGPT 大很多
  ε_clip      = 0.2
  β_KL        = 0          ← 不加 KL!直接让 policy 探索
  Reward      = 0/1 数学正确性 (verifier, 不用 RM)
  Ref         = 每个 RL step 之前 deepcopy 的快照

为什么能 work?
  (1) 数学/代码任务有 verifiable reward,不会 reward hacking → 不需要 KL
  (2) 大 G 让组内 z-score 估计稳定
  (3) 长 CoT 自动涌现:奖励正确就行,policy 学会"先想后答"
```

观察: SFT 不需要长 CoT 数据,RL 自己会涌现"先思考再答案"的链。

### 长度归一化问题 (length normalization)

GRPO 序列级 logp 会偏向短回答(每多一个 token 多一个 logp 项,绝对值变大):

```
logp_policy(seq_short) = log p(t1) + log p(t2)              # 2 项
logp_policy(seq_long)  = log p(t1) + ... + log p(t50)       # 50 项, 绝对值偏大
```

→ ratio 受序列长度影响。修法:

```python
# Token-mean normalization (DeepSeek-Math 用)
logp_seq = neg_logp.sum(-1) / mask.sum(-1)        # 除以序列长度

# Or 算 per-token 的 ratio 和 advantage,然后逐 token 加权
ratio_t = exp(logp_new_t - logp_old_t)            # 每个 token 独立
loss_t  = min(ratio_t · A, clip(ratio_t, 1±ε) · A)
loss    = loss_t.sum(-1).mean()                   # 沿 token 加和
```

DeepSeek-V3/R1 实际用的是 **token-level loss + length normalization** 的组合。

### Verifier vs Reward Model

GRPO 不一定要 RM,可以用**可验证的 verifier**:

| 任务 | Verifier | 优势 |
|---|---|---|
| 数学题 | 答案正则匹配 / SymPy 化简 | 无 reward hacking |
| 代码题 | 跑测试用例,通过率 | 客观 |
| 自然语言 | 用 RM (LLM-as-judge 或学习的 RM) | 灵活但有 hacking 风险 |

DeepSeek-R1 用了**纯 verifier**(`0/1` 准确率),所以才能砍掉 KL 还稳定。

### 完整 GRPO 训练循环

```python
def grpo_train_step(policy, prompts, verifier, G=16, eps=0.2):
    # === 采样阶段 ===
    ref = copy.deepcopy(policy).eval()          # 本轮 ref 快照
    with torch.no_grad():
        rollouts = [policy.generate(p) for p in prompts for _ in range(G)]
        rewards  = verifier(rollouts)            # [B, G],可验证 reward
        logp_old = compute_logp(ref, rollouts)   # ref = policy snapshot

    # === 组内归一化 ===
    A = (rewards - rewards.mean(-1, keepdim=True)) / \
        (rewards.std(-1, keepdim=True) + 1e-8)

    # === 多 epoch PPO 风格更新 ===
    for _ in range(ppo_epochs):
        logp_new = compute_logp(policy, rollouts)
        ratio    = torch.exp(logp_new - logp_old)
        loss     = -torch.min(ratio * A,
                              torch.clamp(ratio, 1-eps, 1+eps) * A).mean()
        loss.backward(); step()
```

### 高频追问

| Q | A |
|---|---|
| 为什么组内归一化能替代 value baseline? | 同 prompt 下 reward 才可比,组内均值是无偏 baseline 估计 |
| Group size G 怎么选? | InstructGPT 8;DeepSeek-R1 用 64。大 G 估计准但显存爆;小 G 方差大 |
| 没 value 网络怎么处理 token-level? | 序列级 A 广播到所有 token (typical);或 token-level loss + length norm |
| 为什么 ref 每步 deepcopy 而不是固定 SFT? | GRPO 的 ratio 主要做 off-policy 修正(单批数据多轮更新);不需要长期 KL 约束 |
| GRPO vs RLOO 区别? | RLOO 用 leave-one-out 均值;GRPO 多了 std 归一化 + PPO clip 支持多 epoch |
| DeepSeek-R1 为什么能不加 KL? | 数学/代码用 verifier (0/1 正确性),不会 reward hacking |
| 长 CoT 怎么自然涌现? | 大 G 探索充足 + 正确答案被加权 → policy 学会先想后答(无监督) |
| 长度偏置怎么处理? | Token-mean normalization (logp / 序列长度) 或 per-token loss |
| GRPO 能 multi-epoch 吗? | 能,和 PPO 一样靠 ratio + clip 控住 off-policy 漂移 |
| GRPO 适合什么、不适合什么? | 适合: 数学/代码/有 verifier 的任务。不适合: 主观对话 (需要 RM 的偏好) |

---

## 10. DPO — 把 RM + RL 合成一个分类损失

> **Key Insight**: 「最大化 reward + KL 约束」的最优解 → 反解出 `r = β·log(π/π_ref)` → 代入 Bradley-Terry → **RM 这一步消失**,变成一个 sigmoid 分类损失。

### 实现层次

```
══════════ 数据 ══════════
   (prompt, y_w (chosen tokens), y_l (rejected tokens))


══════════ 4 次前向 = 2 (policy/ref) × 2 (chosen/rejected) ══════════

                ┌──────────────────── policy(πθ) ────────────────────┐
                │                                                     │
   y_w ────────►│ logits_w ──► seq_logprob ──► π_w  (要梯度)         │
                │                                                     │
   y_l ────────►│ logits_l ──► seq_logprob ──► π_l  (要梯度)         │
                └─────────────────────────────────────────────────────┘

                ┌──────────────────── ref(π_SFT) ─────────────────────┐
                │  (冻结, with torch.no_grad())                       │
   y_w ────────►│ logits  ──► seq_logprob ──► r_w                    │
                │                                                     │
   y_l ────────►│ logits  ──► seq_logprob ──► r_l                    │
                └─────────────────────────────────────────────────────┘

══════════ 损失 ══════════

   margin = (π_w − r_w) − (π_l − r_l)        ← chosen 相对 ref 的 log-ratio
                                                减 rejected 相对 ref 的 log-ratio
   logits = β · margin
   loss   = −log σ(logits) = −F.logsigmoid(logits).sum()
                              │
                              ▼ backward → 只更新 policy

══════════ 模型数 ══════════
   policy (训练中) + ref (冻结) = 2 个    ← 比 PPO 的 4 个少一半
```

### 公式

把隐式 reward `r(x,y) = β·log(π_θ/π_ref)` 代入 Bradley-Terry:

```
L = -log σ( β · [ (logπ_θ(y_w) - logπ_ref(y_w))  -  (logπ_θ(y_l) - logπ_ref(y_l)) ] )
```

### 直觉

- 抬高 chosen 相对 ref 的 log-ratio,压低 rejected 的
- `β` 控制偏离 ref 的强度(≈ PPO 里的 KL 系数)
- 不需要 RM,不需要 RL,**纯监督训练**

### 代码骨架

```python
def sequence_logprob(logits, tokens):
    logp = torch.log_softmax(logits, dim=-1)
    # t 时刻 logits 预测 t+1 的 token
    chosen = logp[:, :-1, :].gather(-1, tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    return chosen.sum(-1)

def dpo_loss(policy_logits_w, policy_logits_l,
             ref_logits_w, ref_logits_l,
             tokens_w, tokens_l, beta=0.2):
    pi_w = sequence_logprob(policy_logits_w, tokens_w)
    pi_l = sequence_logprob(policy_logits_l, tokens_l)
    with torch.no_grad():
        ref_w = sequence_logprob(ref_logits_w, tokens_w)
        ref_l = sequence_logprob(ref_logits_l, tokens_l)
    logits = beta * ((pi_w - ref_w) - (pi_l - ref_l))
    return -F.logsigmoid(logits).sum()
```

### DPO 损失从 RLHF 目标的完整推导(必背)

**Step 1**: RLHF 的优化目标 = "最大化 reward + KL 不偏离 ref":

```
π* = argmax_π  E_{x, y~π}[ r(x,y) ]  − β · KL(π ‖ π_ref)
```

**Step 2**: 这个约束优化有**闭式最优解**(Lagrangian 配方法):

```
π*(y|x) ∝ π_ref(y|x) · exp(r(x,y) / β)
π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp(r(x,y) / β)
```

其中 `Z(x) = Σ_y π_ref(y|x)·exp(r(x,y)/β)` 是归一化常数。

**Step 3**: **反解出 reward**(从最优 policy 反推 reward):

```
r(x,y) = β · log(π*(y|x) / π_ref(y|x)) + β · log Z(x)
```

`β·log Z(x)` 是 prompt 相关的常数项,在 Bradley-Terry 的**差**里**消掉了**。

**Step 4**: 代入 Bradley-Terry: `P(y_w ≻ y_l) = σ(r(x,y_w) - r(x,y_l))`:

```
P(y_w ≻ y_l) = σ( β · log(π*(y_w)/π_ref(y_w)) − β · log(π*(y_l)/π_ref(y_l)) )
            = σ( β · [(logπ*(y_w) − logπ_ref(y_w)) − (logπ*(y_l) − logπ_ref(y_l))] )
```

**Step 5**: 最大化对数似然 → DPO loss:

```
L = -log σ( β · [(logπ_θ(y_w) − logπ_ref(y_w)) − (logπ_θ(y_l) − logπ_ref(y_l))] )
```

**关键观察**: 这个推导**完全等价于** RM + PPO,但**消去了 RM 这一步**。
代价: PPO 是 on-policy 自己采样,DPO 是 off-policy 用固定偏好数据。

### DPO 梯度的直觉(看公式知道在干啥)

对 DPO loss 求 `θ` 梯度:

```
∇_θ L = -β · σ(-β·margin) · ∇_θ [logπ_θ(y_w) - logπ_θ(y_l)]
                ↑                          ↑
           当前模型预测错的概率           "推高 chosen, 压低 rejected"
```

`σ(-β·margin)` 是个**置信度调节子**: 模型已经把 chosen 拉高很多了,这个 σ 很小,梯度自动减弱。
**自动 hard mining**: 关注那些当前还分不清的 pair。

### 长度偏置 (DPO 的著名痛点)

```
logπ(seq) = Σ_t log π(t_t | t_<t)         # 项数 = 序列长度
```

如果 chosen 平均比 rejected 长 50 tokens,`logπ(chosen) - logπ(rejected)` 会有
**~50 个额外 logp 项贡献**,模型学到"输出长 = 高 reward"的**虚假关联**。

实测: 普通 DPO 训完后,输出长度比 SFT 长 2-3 倍,但人类胜率提升有限。

**修法**:
- **SimPO**: 用 `logπ / |seq|` 归一化,去掉 ref 模型
- **Length-normalized DPO**: 加 length penalty 项
- **R-DPO**: 在 reward 里减 length term

### DPO 的几个进化版本

| 算法 | 改了什么 | 解决了什么 |
|---|---|---|
| **DPO (Rafailov 2023)** | RM + PPO → 直接分类 loss | 简化 RLHF |
| **IPO (Azar 2023)** | sigmoid → identity | 防止过拟合 / 偏好概率饱和 |
| **KTO (Ethayarajh 2024)** | pairwise → 单样本 like/dislike | 不需要成对数据,只要 thumbs up/down |
| **SimPO (Meng 2024)** | 去掉 ref 模型 + length normalize | 省显存 + 解长度偏置 |
| **RPO (Pal 2024)** | DPO + 显式 SFT loss 项 | 防 logp 整体下降的坍塌 |
| **R-DPO** | 加 length penalty 到 reward | 解长度偏置 |
| **β-DPO** | 动态调 β | 不同样本难度自适应 |

### DPO 训练中的"logp 双降"现象 (常见 bug)

实际训完 DPO 后画图,常发现:

```
logπ_θ(chosen):    -25   (训前)  →   -40   (训后)    ← 居然降了!
logπ_θ(rejected):  -27   (训前)  →   -55   (训后)    ← 降得更多
margin:              2                  15            ← margin 增大了
```

`margin` 在涨,所以 DPO loss 在降,**但 chosen 的概率也降了**!原因: DPO 只关心 margin,不关心绝对概率。

**修法 (RPO)**: 加一个 SFT loss 项强制 `logπ_θ(chosen)` 不要下降:

```python
L_RPO = L_DPO + λ · (-logπ_θ(y_w))     # 第二项就是普通 SFT loss
```

### DPO vs PPO 实操对比

| 维度 | PPO | DPO |
|---|---|---|
| 训练阶段 | SFT → RM → PPO | SFT → DPO(一步到位) |
| Reward model | 显式训 | 隐式 |
| RL 采样 | on-policy 采 | 用离线偏好数据 |
| 显存 | 4 模型 (policy + ref + reward + value) | 2 模型 (policy + ref) |
| 实现复杂度 | 高 (4 模型 + GAE + clip) | 低 (一个 sigmoid loss) |
| 训练稳定性 | 难调 | 稳定(分类损失) |
| 数据分布敏感性 | 低 (自己采样) | 高 (OOD 偏好学不好) |
| 工业线落地 | Anthropic, OpenAI 早期 | Llama 3, Mistral, Zephyr |
| 性能上限 | 略高 | 接近但有差距 |

### 完整 DPO 训练代码

```python
def dpo_train_step(policy, ref, batch, beta=0.1):
    # 4 次前向: 2 (policy/ref) × 2 (chosen/rejected)
    logp_pol_w = sequence_logprob(policy(batch['chosen']), batch['chosen'])
    logp_pol_l = sequence_logprob(policy(batch['rejected']), batch['rejected'])

    with torch.no_grad():
        logp_ref_w = sequence_logprob(ref(batch['chosen']), batch['chosen'])
        logp_ref_l = sequence_logprob(ref(batch['rejected']), batch['rejected'])

    # margin: chosen 的 log-ratio 减 rejected 的 log-ratio
    chosen_logratio   = logp_pol_w - logp_ref_w
    rejected_logratio = logp_pol_l - logp_ref_l
    margin            = chosen_logratio - rejected_logratio

    loss = -F.logsigmoid(beta * margin).mean()

    # 监控指标
    chosen_reward   = beta * chosen_logratio.detach()
    rejected_reward = beta * rejected_logratio.detach()
    reward_gap      = (chosen_reward - rejected_reward).mean()    # 应该上涨
    accuracy        = (chosen_reward > rejected_reward).float().mean()  # 训练 accuracy

    return loss, {'reward_gap': reward_gap, 'acc': accuracy}
```

### 高频追问

| Q | A |
|---|---|
| DPO 为什么不需要 RM? | 反解 RLHF 最优解: `r = β·log(π/π_ref) + const`,代入 BT,const 抵消,直接得分类 loss |
| β 怎么选? | 0.1-0.5 常见;太大 → 几乎不动 ref;太小 → 偏离 ref 太远易崩 |
| 为什么还要 ref? | 限制 policy 不要离 SFT 太远;ref 同时给 chosen / rejected 提供归一化基准 |
| DPO 的缺点? | (a) Off-policy → OOD 偏好学不好 (b) 长度偏置 (c) chosen logp 可能整体下降 |
| 怎么改进 DPO? | IPO (防过拟合), KTO (单点偏好), SimPO (去 ref + length norm), RPO (加 SFT) |
| DPO 训练中 chosen logp 应该上升吗? | 不一定!DPO 只优化 margin,实测 chosen 和 rejected 经常一起下降 (用 RPO 修) |
| DPO 能 multi-epoch 吗? | 能,但易过拟合;经验 1-3 epoch 最佳。IPO 提出就是为了缓解 |
| 为什么 SimPO 去掉 ref 还 work? | 用 `logπ/|seq|` 自带 length norm + ref 的归一化作用被 length scale 替代 |
| DPO 数据从哪来? | 同 prompt 不同 temperature 采样的两个回复 → 人标 chosen/rejected;或 GPT-4 当 judge |
| DPO 训练时 ref 要不要也 grad? | **绝对不要**!ref 必须 `with torch.no_grad()`,否则梯度全乱 |

---

## 11. 一图全景: 四种对齐算法对比

```
              数据形式                    需要 Reward Model?           是否需要在线采样?
──────────────────────────────────────────────────────────────────────────────────
SFT       (prompt, response)              ─                            ─
RM        (prompt, chosen, rejected)      训出来 ←                     ─
PPO       (prompt) + RM 打分              是 (独立训)                  是 (on-policy)
GRPO      (prompt) + RM 或 verifier       是 或 否 (可验证 reward)      是 (组内采样)
DPO       (prompt, chosen, rejected)      ─                            ─ (纯监督)
```

```
                            ┌─ value baseline ─► PPO  (4 模型)
            策略梯度 + RM ──┤
                            └─ 组内归一化 ────► GRPO (3 模型, 无 value)

            Bradley-Terry + 最优策略解析解 ───► DPO  (2 模型, 无 RM 无 RL)
```

```
显存:  PPO  ≫  GRPO  >  DPO  ≫  SFT
难度:  PPO  ≫  GRPO  ≈  DPO  >  SFT
质量:  PPO  >  GRPO  ≳  DPO    (DPO 在 OOD 数据上掉得快)
```

---

## 附: 复习自检清单

- [ ] **MinHash**: 能口述"为什么 min h(A) = min h(B) 的概率 = Jaccard"
- [ ] **BPE**: 能在白纸上跑完一个 3 词例子的前 3 轮合并
- [ ] **BPE vs WordPiece vs Unigram**: 三个准则、三种方向、三种 OOV 处理
- [ ] **Byte-level BPE**: GPT-2 的 Ġ 是什么,为什么用字节
- [ ] **Softmax 反向**: 写出 `s ⊙ (g - (g·s))`,解释为什么
- [ ] **Attention 反向**: dq/dk/dv 三个公式 + √D 的位置
- [ ] **Memory-Efficient Attention**: 分块 num/den 累加;online softmax 的 running max 怎么维护;FlashAttention 为什么快(HBM,不是 FLOPs)
- [ ] **RM loss**: `-log σ(r_w - r_l)`,从 Bradley-Terry 推
- [ ] **GAE**: 写出递推式 + γ/λ 极值含义
- [ ] **PPO**: clipped surrogate + KL,4 个模型
- [ ] **GRPO**: 比 PPO 少了什么(value),advantage 怎么算
- [ ] **DPO**: 写出损失,解释为什么不需要 RM
- [ ] **四者对比**: 数据形式 / RM / on-policy / 显存 4 个维度

---

