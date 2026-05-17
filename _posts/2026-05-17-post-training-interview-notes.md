---
layout: post
title: "Post-Training 面试速记: 从分词到对齐"
date: 2026-05-16 23:00:00-0700
description: "从 MinHash 去重到 BPE/WordPiece/Unigram、Softmax 手推、RM/PPO/GRPO/DPO，每节给出 Key Insight、实现层次图、代码骨架和高频追问。"
tags: post-training rlhf tokenizer ppo grpo dpo
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
4. [Reward Model — Bradley-Terry pairwise](#4-reward-model--bradley-terry-pairwise)
5. [RL Fine-tuning — 策略梯度 + 组内归一化](#5-rl-fine-tuning--策略梯度--组内归一化)
6. [GAE — 广义优势估计](#6-gae--广义优势估计)
7. [PPO — Clipped Surrogate + KL](#7-ppo--clipped-surrogate--kl)
8. [GRPO — PPO 砍掉 Value 网络](#8-grpo--ppo-砍掉-value-网络)
9. [DPO — 把 RM + RL 合成一个分类损失](#9-dpo--把-rm--rl-合成一个分类损失)
10. [一图全景: 四种对齐算法对比](#10-一图全景-四种对齐算法对比)

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

### 数值例子

```
d1 = "The quick brown fox jumps over the lazy dog."
d2 = "The quick brown fox jumps over the lazy dog!"   # 只差一个标点
d3 = "Completely unrelated sentence about cats."

sim(d1, d2) ≈ 0.97   ← 近重复
sim(d1, d3) ≈ 0.00
```

### 高频追问

| Q | A |
|---|---|
| 为什么 min-hash 碰撞概率 = Jaccard? | 对随机排列,两集合并集中任意元素被映为最小的概率均等;最小元素同时属于交集的概率 = \|A∩B\| / \|A∪B\| |
| `num_perm` 怎么选? | 估计方差 ≈ J(1-J)/num_perm。128 误差 ~5%,256 ~3%。代价是签名内存和比较时间 |
| MinHash vs SimHash? | MinHash 估 **Jaccard**(集合相似),适合 shingle 集;SimHash 估 **cosine**(向量相似),适合特征向量 |
| 为什么去重对 LLM 重要? | 重复 → 记忆 / 浪费算力 / 评测泄漏。GPT-3、LLaMA 论文都强调过 |

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

### Softmax Jacobian(必背)

对一行 `s = softmax(z)`:

```
∂s_i / ∂z_j = s_i · (δ_ij − s_j)
J = diag(s) − s · sᵀ
```

给定上游梯度 `g`,**向量化公式**:

```
∂L/∂z = g · J = s ⊙ (g − (g·s))
```

注意 `g · s` 是 inner product(标量),广播减回去。

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

### 高频追问

| Q | A |
|---|---|
| 为什么 softmax 数值稳定要减 max? | `exp(z)` 大数会上溢;`softmax(z) == softmax(z - max(z))`(分子分母同乘常数) |
| 为什么 attention 要除 √D? | 点积方差随 D 增大 → softmax 进饱和区 → 梯度消失。除 √D 让方差稳定 |
| softmax + cross-entropy 为什么简化成 `softmax - onehot`? | log-softmax 求导和 softmax 的 Jacobian 抵消,只剩 `p - y` |
| 反向能 inplace 写 dq, dk, dv 吗? | 不行,共用 A 的话会乱;PyTorch 是 save_for_backward 保留前向中间量 |

---

## 4. Reward Model — Bradley-Terry pairwise

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

### 高频追问

| Q | A |
|---|---|
| 为什么取最后一个 token? | Causal mask 下,最后一个位置的 hidden 看到了完整 prompt + response |
| Reward hacking 是什么? | Policy 学会钻 RM 的漏洞(说套话 / 长度作弊),拿到高 reward 但人类不喜欢 |
| RM 和 DPO 的关系? | DPO 在数学上等价于"训 RM 然后 PPO",但消去了 RM 这一步,直接优化 policy |
| 多个 chosen 怎么处理? | 列出所有 pairwise (chosen, rejected) 对,逐对算损失求和 |
| Reward 要不要归一化? | RM 的输出会送进 PPO 当 advantage 一部分,通常会做 z-score 或 baseline 减均值 |

---

## 5. RL Fine-tuning — 策略梯度 + 组内归一化

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

### 高频追问

| Q | A |
|---|---|
| 为什么减 baseline 不改变期望? | `E[∇logπ · b] = b·∇E[1] = 0`(对常数 baseline 严格成立) |
| 为什么用组内均值? | 同 prompt 下 reward 才可比;跨 prompt 平均无意义 |
| on-policy vs off-policy? | RL fine-tune 通常 on-policy(策略采样自当前 π);PPO 用 ratio 修正一点 off |
| 熵正则太强 / 太弱? | 太强 → 不收敛,policy 一直瞎采;太弱 → 早期 mode collapse |
| 和 GRPO 关系? | 这一节就是 GRPO 的"裸版"(没有 PPO 的 ratio + clip);加上后变成 GRPO |

---

## 6. GAE — 广义优势估计

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

### 高频追问

| Q | A |
|---|---|
| γ 和 λ 各管什么? | γ 是回报折扣(管"未来 reward 看多远");λ 管"信任 value 估计 vs 真实回报"的程度 |
| 为什么从后往前? | 递推式 `A_t = δ_t + γλ·A_{t+1}` 依赖 t+1 的值 |
| Actor 和 Critic 为什么共享 backbone? | 省显存 + 表征共享;value head 只是个 `Linear(dim, 1)` |
| Value loss 怎么算? | `MSE(V(s_t), returns_t)`,`returns = advantages + values` |
| 稀疏 reward (只有最后一步)怎么办? | GAE 会把最后一步的 reward 沿 γ^t 折扣回传到所有时间步 |

---

## 7. PPO — Clipped Surrogate + KL

> **Key Insight**: 用 importance sampling ratio 修正 off-policy + **clip 防止策略跑飞** + KL 约束别偏离 SFT 太远。
> 4 个模型 (policy / ref / reward / value) 是 PPO 工程痛点。

### 实现层次

```
══════════ RLHF 三步走 ══════════
   Step 1: SFT             → π_SFT
   Step 2: 训 Reward Model → RM   (见 §4)
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

### 高频追问

| Q | A |
|---|---|
| 为什么要 clip? | 单步更新太大会让 ratio 离 1 太远 → importance sampling 估计崩 |
| ε 太大/太小? | 太大失去 clip 意义,等同朴素 PG;太小更新太保守,收敛慢 |
| 为什么需要 KL? | RM 是个不完美的代理,policy 全力对抗 RM 会 reward hacking |
| Min(unclipped, clipped) 的直觉? | 选**保守**的那一边,对正 advantage 限上界,对负 advantage 不限下界(本来就要压) |
| Ratio 用 logp_new - logp_old 不是 logp_ref? | `old` 通常是采样时的 policy 快照(每个 minibatch 之前 detach 的 logp);ref 是 SFT 副本,不一样 |

---

## 8. GRPO — PPO 砍掉 Value 网络

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

### 高频追问

| Q | A |
|---|---|
| 为什么组内归一化能替代 value baseline? | 同 prompt 下 reward 才可比,组内均值就是个很好的 baseline 估计 |
| Group size G 怎么选? | 经验值 8-64;太小方差大,太大显存爆 |
| 没 value 网络怎么处理 token-level advantage? | GRPO 是序列级 advantage,广播到所有 token(假设 reward 只在序列结束给) |
| 为什么 ref 每步 deepcopy 而不是固定 SFT? | GRPO 的 ratio 主要做 off-policy 修正(同 batch 内多轮更新),不需要长期约束 |

---

## 9. DPO — 把 RM + RL 合成一个分类损失

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

### DPO vs PPO

| 维度 | PPO | DPO |
|---|---|---|
| 训练阶段 | SFT → RM → PPO | SFT → DPO(一步到位) |
| Reward model | 显式训 | 隐式 |
| RL 采样 | on-policy 采 | 用离线偏好数据 |
| 显存 | 4 模型 | 2 模型(policy + ref) |
| 稳定性 | 难调 | 稳定(分类损失) |
| 数据分布敏感性 | 低(自己采样) | 高(off-policy,数据分布外效果差) |

### 高频追问

| Q | A |
|---|---|
| DPO 为什么不需要 RM? | 数学上证明了"最优 RM + 最优 PPO" = "DPO 损失" |
| β 怎么选? | 0.1-0.5 常见;太大不动,太小偏离 ref 太远 |
| 为什么还要 ref? | 限制 policy 不要离 SFT 太远;ref 同时给 chosen / rejected 提供归一化基准 |
| DPO 的缺点? | Off-policy → 数据分布外的偏好学不好;长输出容易 reward 减一变成 logp 减一,长度偏置 |
| 怎么改进 DPO? | IPO(替代 sigmoid 防过拟合)、KTO(无需 pairwise,只要单点 like/dislike)、SimPO(去 ref) |

---

## 10. 一图全景: 四种对齐算法对比

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
- [ ] **RM loss**: `-log σ(r_w - r_l)`,从 Bradley-Terry 推
- [ ] **GAE**: 写出递推式 + γ/λ 极值含义
- [ ] **PPO**: clipped surrogate + KL,4 个模型
- [ ] **GRPO**: 比 PPO 少了什么(value),advantage 怎么算
- [ ] **DPO**: 写出损失,解释为什么不需要 RM
- [ ] **四者对比**: 数据形式 / RM / on-policy / 显存 4 个维度

---

