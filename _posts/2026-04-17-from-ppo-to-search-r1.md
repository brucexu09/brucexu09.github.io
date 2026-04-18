---
layout: post
title: "From PPO to Search-R1: The Design Space of Reasoning and Agentic RL"
title_zh: "从 PPO 到 Search-R1：Reasoning 与 Agentic RL 的设计空间"
date: 2026-04-17
author: Boxun Xu
description: "A component-by-component decomposition of PPO through GRPO, verifiers, and retrieved-token masking to a complete instance of Agentic RL."
tags: [reasoning, agentic-rl]
bilingual: true
giscus_comments: false
related_posts: false
---

<div class="lang-content lang-en" markdown="1">


## Preface

The major advances in reasoning LLMs over the past year — o1, DeepSeek-R1, Search-R1 — share a common RL foundation, but each makes different choices at a few critical design points. Laying these choices side by side makes the design space much clearer than reading any single paper on its own.

That is what this post does. The main thread is Search-R1[^sr1]: it is a clean instance of Agentic RL, and it is simple enough to naturally tie together PPO, GRPO, verifiers, and token masking.

Readers are assumed to be familiar with transformers, SFT, and basic deep learning. Familiarity with RL internals is **not** assumed.

Not covered: RLHF preference-data construction, inference-time scaling (best-of-N, tree search), multi-agent, low-level hardware optimizations.

---

## 1. Problem Setup: Evolution of the Post-training Stack

Moving from a pretrained LLM to an agent that can autonomously call tools goes through several technical jumps. Each jump solves a limitation of the previous one:

```
Pretrain
   │ knows a lot, but doesn't follow instructions
   ↓
SFT (Supervised Fine-Tuning)
   │ imitates only; can't actively suppress wrong behavior
   ↓
RLHF (PPO + Reward Model)
   │ RM gets hacked; preference data is expensive; weak reasoning signal
   ↓
Reasoning RL (GRPO + Verifier)
   │ reward only comes from the model's own outputs; no external interaction
   ↓
Agentic RL (+ Tool / Environment)
```

The technical core of each jump:

- **SFT**: cross entropy + supervised learning
- **RLHF**: PPO + neural reward model
- **Reasoning RL**: GRPO + rule-based verifier; drops the reward model
- **Agentic RL**: on top of Reasoning RL, embeds environment interaction into the rollout

"Agentic RL" is not a standalone new algorithm but a training paradigm that "introduces environment interaction into the rollout." The algorithmic core is still PPO or GRPO. The key change is **extension of the environment**: tool calls are inserted mid-rollout, and execution / retrieval results are injected back into the sequence, becoming context for the next generation.

Search-R1 is the cleanest instance of this paradigm: it uses a search engine as the environment and lets the LLM autonomously decide when and what to search during reasoning.

Off this main thread there is a side branch: **DPO** turns RLHF directly into supervised learning — no RL rollout, no reward model. §7 discusses it separately.

The following sections unfold each component in turn.

---

## 2. From Cross Entropy to Policy Gradient

This section establishes the equivalence between RL loss and SFT loss, as groundwork for understanding PPO later.

### 2.1 Simplifying Cross Entropy under Next-Token Prediction

General definition:

$$
H(p, q) = -\sum_i p_i \log q_i
$$

In the next-token setting, the ground-truth label $$p$$ is one-hot with $$p_{y_t} = 1$$ and zero elsewhere. Summing over the vocabulary leaves a single term:

$$
H(p, q) = -\log \pi_\theta(y_t \mid y_{<t})
$$

SFT sums over the whole sequence:

$$
\mathcal{L}^{SFT} = -\sum_{t=1}^{T} \log \pi_\theta(y_t \mid y_{<t})
$$

### 2.2 The Subtle Distinction Between CE and KL

The identity:

$$
H(p, q) = H(p) + D_{KL}(p \| q)
$$

In supervised learning $$p$$ is the fixed ground truth, so $$H(p)$$ is a constant with respect to $$\theta$$ — meaning **optimizing CE and optimizing KL have identical gradients**. We default to CE only because it is cheaper.

In RL this equivalence breaks down: both sides of the KL are models ($$\pi_\theta$$ vs $$\pi_{ref}$$), neither is fixed. The distinction between CE and KL becomes genuinely important here. We will use this below when discussing PPO's KL penalty.

### 2.3 From SFT to REINFORCE

The SFT loss unconditionally pushes up the probability of every token — every token in the demo data is assumed to be good. That assumption is fine for imitation learning, but RL wants more: **push up good behaviors, and actively suppress bad ones**.

REINFORCE multiplies the SFT loss by a weight $$A_t$$:

$$
\mathcal{L}^{REINFORCE} = -\mathbb{E}\Big[\sum_t \log \pi_\theta(y_t \mid y_{<t}) \cdot A_t\Big]
$$

- $$A_t > 0$$: good token, push probability up
- $$A_t < 0$$: bad token, push probability down
- The larger $$\lvert A_t \rvert$$ is, the larger the adjustment

The three losses in unified form:

| Loss | Per-token form | Meaning of the weight |
|------|---------------|-----------------------|
| SFT | $$-\log \pi_\theta(y_t)$$ | 1 (push up every token) |
| REINFORCE | $$-\log \pi_\theta(y_t) \cdot A_t$$ | advantage-weighted |
| PPO | $$-\min(r_t A_t, \text{clip}(r_t) A_t)$$ | ratio replaces log (next section) |

**Key insight**: the RL loss is structurally "signed, weighted cross entropy." With $$A_t \equiv 1$$, SFT and RL coincide in **gradient form**. Note this is only a formal analogy — SFT samples from demo data while RL samples from $$\pi_\theta$$; the sample distributions are different.

### 2.4 Code Comparison

```python
# SFT loss
def sft_loss(logits, actions, mask):
    log_probs = F.log_softmax(logits, dim=-1)
    token_logprobs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    return -(token_logprobs * mask).sum() / mask.sum()

# REINFORCE loss: only differs by * advantages
def reinforce_loss(logits, actions, advantages, mask):
    log_probs = F.log_softmax(logits, dim=-1)
    token_logprobs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    return -(token_logprobs * advantages * mask).sum() / mask.sum()
```

The only difference is one `* advantages`. That one multiplication opens the gap between SFT and RL: **RL can suppress bad behavior with a negative weight; SFT cannot.**

---

## 3. PPO: Design Motivation and the Role of Each Component

PPO[^ppo] is the de-facto standard for LLM RL training today. This section breaks it down component by component, ordered as "component → what problem does it solve."

Below is the LLM-RLHF version of PPO — it adds a $$\pi_{ref}$$ KL penalty that the original PPO paper did not have. This addition is an engineering convention inherited from InstructGPT[^instructgpt], not part of the original PPO:

$$
\mathcal{L}^{PPO}(\theta) = -\mathbb{E}_t\Big[\min\big(r_t(\theta) A_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\big)\Big] + \beta \cdot D_{KL}\big[\pi_\theta \| \pi_{ref}\big]
$$

### 3.1 Starting Point: Two Pathologies of REINFORCE

**Pathology 1: high variance; a single update may blow up.** $$A_t$$ is estimated from a single trajectory, which is high variance. Occasional high-reward samples can blow up $$A_t$$ for certain tokens, and one gradient step can push $$\pi_\theta$$ into a weird region.

**Pathology 2: sample inefficiency.** REINFORCE is strictly on-policy: once policy updates, the previous rollouts are useless. Unacceptable for LLMs where rollouts are extremely expensive.

Each component of PPO cures these two pathologies.

### 3.2 Importance Ratio: Reusing the Same Batch of Rollouts

$$
r_t(\theta) = \frac{\pi_\theta(y_t \mid y_{<t})}{\pi_{old}(y_t \mid y_{<t})}
$$

$$\pi_{old}$$ is a snapshot of the policy at the time this batch of rollouts was sampled. The training loop becomes:

```python
for iteration in range(N):
    # snapshot current policy
    pi_old = copy(pi_theta).detach()
    
    # collect rollouts using pi_old
    rollouts = sample_rollouts(pi_old, batch_size=B)
    advantages = compute_gae(rollouts)
    
    # multiple gradient updates on the same rollouts
    for epoch in range(K):  # typically K = 2~4
        for minibatch in split(rollouts):
            loss = ppo_loss(pi_theta, pi_old, minibatch, advantages)
            loss.backward()
            optimizer.step()
```

The same batch is reused $$K$$ times. By the $$k$$-th epoch $$\pi_\theta \neq \pi_{old}$$ — that is off-policy. The importance ratio mathematically corrects this mismatch (standard application of importance sampling).

**Notational pitfall**: $$r_t$$ here is the importance ratio, not the reward. Reward becomes $$A_t$$ via GAE in PPO; the reward itself does not appear in the loss.

### 3.3 Clip: Single-step Magnitude Constraint

The ratio alone is not enough. If $$\pi_\theta$$ drifts far from $$\pi_{old}$$ (e.g. $$r_t = 5$$), importance sampling variance explodes. Clip hard-caps it:

$$
\text{clip}(r_t, 1-\epsilon, 1+\epsilon)
$$

Typically $$\epsilon = 0.2$$, meaning the single-token probability ratio can change by at most ±20%.

**But clip alone has a trap.** Say $$A_t = +1$$ (a good token) and $$r_t = 0.5$$ (the policy has been pushing it down — an error state). If we directly used $$\text{clip}(r_t) \cdot A_t = 0.8$$, the gradient gets **artificially inflated**, encouraging the policy to keep going the wrong way.

### 3.4 Min: Let Clip Only Brake, Not Block Corrections

The full policy term:

$$
L_t^{CLIP} = \min\big(r_t A_t,\ \text{clip}(r_t, 1-\epsilon, 1+\epsilon) \cdot A_t\big)
$$

With $$\epsilon = 0.2$$, four cases:

| Case | $$A_t$$ | $$r_t$$ | $$r_t A_t$$ | $$\text{clip}(r_t) A_t$$ | min picks | Meaning |
|------|--------|--------|------------|--------------------------|-----------|---------|
| B | $$+1$$ | $$1.5$$ (above upper bound) | $$1.5$$ | $$1.2$$ | clip term | good token pushed too far, **saturated** |
| C | $$+1$$ | $$0.5$$ (below lower bound) | $$0.5$$ | $$0.8$$ | **raw term** | good token being pushed down, **correction allowed** |
| D | $$-1$$ | $$0.5$$ (below lower bound) | $$-0.5$$ | $$-0.8$$ | clip term | bad token already suppressed enough, **saturated** |
| E | $$-1$$ | $$1.5$$ (above upper bound) | $$-1.5$$ | $$-1.2$$ | **raw term** | bad token being pushed up, **correction allowed** |

The pattern:

- Going too far in the **right** direction (B, D) → min picks the clip term, gradient saturates
- Going in the **wrong** direction (C, E) → min picks the raw term, gradient preserved, correction continues

Schulman calls this the **pessimistic bound**: take the smaller value as a conservative estimate. Two birds one stone: restrict single-step magnitude (B, D) without blocking error correction (C, E).

### 3.5 KL Penalty: Long-term Anchor Against Drift

Clip handles **single-step** stability. But even if every step stays within $$[1-\epsilon, 1+\epsilon]$$, accumulation over many iterations may still drift $$\pi_\theta$$ to a completely different distribution. The classic failure mode is **reward hacking**: the model discovers a generation pattern that reliably gets high reward while language quality collapses; clip never triggers at any single step, but the cumulative drift is catastrophic.

PPO adds the KL term:

$$
\beta \cdot D_{KL}\big[\pi_\theta(\cdot \mid s_t) \| \pi_{ref}(\cdot \mid s_t)\big]
$$

$$\pi_{ref}$$ is the policy at the **start of training** (usually the post-SFT model), frozen throughout RL. It is not the same as $$\pi_{old}$$:

| Dimension | Clip on $$r_t = \pi_\theta / \pi_{old}$$ | KL on $$\pi_\theta \| \pi_{ref}$$ |
|-----------|-----------------------------------------|-----------------------------------|
| Comparison target | policy at last rollout | policy at training start |
| Time scale | single iteration | entire training run |
| Form | hard constraint (saturating clip) | soft constraint (continuous penalty) |
| Problem solved | over-aggressive single step | cumulative drift |

The two are **complementary, not redundant**. Clip only, no KL → long-run reward hacking. KL only, no clip → a single big update can crash training.

**The k3 estimator.** Token-level KL by definition iterates over the whole vocabulary, which is expensive. In practice people use Schulman's k3[^k3]:

$$
\text{KL}_{k3} = \exp(\log\pi_{ref} - \log\pi_\theta) - (\log\pi_{ref} - \log\pi_\theta) - 1
$$

Strictly speaking, k3 is a low-variance unbiased estimator of $$D_{KL}[p \| q]$$ **when samples come from $$p$$**. In PPO we actually sample from $$\pi_{old}$$ (because the rollout uses $$\pi_{old}$$), so this is technically an approximation. But $$\pi_{old} \approx \pi_\theta$$ under the clip constraint, and in practice the approximation is good enough.

One-liner in code:

```python
log_ratio = ref_logprobs - new_logprobs
kl = torch.exp(log_ratio) - log_ratio - 1  # k3 estimator
```

### 3.6 Full Code

```python
def compute_ppo_loss(
    logits,          # [B, T, V] current policy outputs
    old_logprobs,    # [B, T] log prob from pi_old (detached)
    ref_logprobs,    # [B, T] log prob from pi_ref (frozen)
    actions,         # [B, T] sampled tokens
    advantages,      # [B, T] from GAE (Section 4)
    action_mask,     # [B, T] 1 for policy-generated positions
    clip_eps=0.2,
    kl_coef=0.001,
):
    log_probs = F.log_softmax(logits, dim=-1)
    new_logprobs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    
    # === importance ratio (3.2) ===
    ratio = torch.exp(new_logprobs - old_logprobs)
    
    # === clip + min (3.3, 3.4) ===
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2)
    
    # === KL penalty (3.5, k3 estimator) ===
    log_ratio_ref = ref_logprobs - new_logprobs
    kl = torch.exp(log_ratio_ref) - log_ratio_ref - 1
    
    total = (policy_loss + kl_coef * kl) * action_mask
    return total.sum() / action_mask.sum()
```

### 3.7 Five Regularization Mechanisms at a Glance

| Mechanism | Target | Problem it solves |
|-----------|--------|-------------------|
| Importance ratio | data / policy mismatch | allows rollouts to be reused |
| Clip on ratio | single-token single-step update | single step drifting too far from $$\pi_{old}$$ |
| Min on clip | side effects of clip | preserves correction gradient in the wrong direction |
| KL on $$\pi_{ref}$$ | cumulative drift | reward hacking |
| Value baseline | advantage estimation | variance reduction, credit assignment |

PPO's engineering stability is the sum of all five. Remove any one and some setting will start breaking.

---

## 4. Sparse Reward and Credit Assignment

### 4.1 Problem Definition

In LLM RL, reward typically only has a nonzero value at the end of the episode. A typical Search-R1 episode:

```
<think>...</think><search>...</search><information>...</information>
<think>...</think><search>...</search><information>...</information>
<think>...</think><answer>McComb, Mississippi</answer>   ← reward=1 only here
```

500 tokens, only the last one has a nonzero reward. Using the terminal reward as $$A_t$$ for every token causes two problems: all tokens get equal weight (can't distinguish the critical ones), and gradient variance is high (each episode contributes only one scalar signal).

This is the **credit assignment** problem.

### 4.2 Value Function

The standard fix introduces a value function $$V_\phi$$:

$$
V_\phi(s_t) = \mathbb{E}\Big[\sum_{k=0}^{T-t} \gamma^k r_{t+k}\Big]
$$

The advantage is defined as:

$$
A_t = R_t - V_\phi(s_t)
$$

- $$A_t > 0$$: actual return exceeds expectation → good choice
- $$A_t < 0$$: below expectation → bad choice
- $$A_t \approx 0$$: not critical

The value function **spreads** the sparse terminal reward across every token position.

### 4.3 Monte Carlo vs TD

Two ways to estimate $$R_t$$:

| Method | Formula | Property |
|--------|---------|----------|
| Monte Carlo | $$R_t^{MC} = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$$ | unbiased, high variance |
| TD | $$R_t^{TD} = r_t + \gamma V_\phi(s_{t+1})$$ | biased, low variance |

### 4.4 GAE: Exponential Interpolation of the Two

GAE[^gae] uses $$\lambda \in [0, 1]$$ to interpolate:

$$
A_t^{GAE} = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
$$

Recursive form:

$$
A_t^{GAE} = \delta_t + \gamma \lambda A_{t+1}^{GAE}
$$

```python
def compute_gae(rewards, values, gamma=1.0, lam=1.0):
    """
    rewards: [T] per-step rewards (mostly zero in LLM setting)
    values:  [T+1] value estimates; values[T] is terminal bootstrap
    """
    T = len(rewards)
    advantages = torch.zeros(T)
    gae = 0.0
    
    # iterate in reverse: GAE is an exponentially weighted sum of future deltas
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    
    returns = advantages + values[:T]
    return advantages, returns
```

Search-R1 uses $$\lambda = \gamma = 1$$. Under this setting the GAE recursion degenerates to $$A_t = \sum_l \delta_{t+l}$$, which telescopes to $$R_t - V(s_t)$$ — exactly the MC advantage.

### 4.5 Why Sparse Reward Works on LLMs

1. **Pretraining provides a strong prior**: RL doesn't start from scratch — it only does fine-grained adjustment
2. **Causal token structure**: early good choices correlate highly with final success
3. **Smoothing from the value function**: spreads terminal reward to each position
4. **Batch-size statistical power**: Search-R1 uses batch = 512, gradient is relatively stable

---

## 5. Reward Source: Reward Model vs Verifier

### 5.1 Two Orthogonal Dimensions

The design space of LLM RL has two orthogonal axes:

**Axis 1: Where does reward come from** (optimize what)
- Neural Reward Model (RM)
- Rule-based Verifier
- Generative Verifier

**Axis 2: How to do credit assignment** (propagate how)
- Critic + GAE (PPO)
- Group Normalization (GRPO)
- Offline data (DPO[^dpo])

| Reward source | Credit Assignment | Representative work |
|---------------|-------------------|----------------------|
| Neural RM | Critic + GAE | Classic RLHF (InstructGPT) |
| Rule Verifier | Critic + GAE | Search-R1 |
| Rule Verifier | Group Norm | DeepSeek-R1[^r1] |

### 5.2 Reward Model vs Critic: Two Concepts Frequently Confused

| Dimension | Reward Model | Critic (Value Model) |
|-----------|--------------|---------------------|
| What it evaluates | the **absolute quality** of the full response | the **expected future return** of the current state |
| Input | (prompt, response) | current state $$s_t$$ |
| Output | a single scalar $$r$$ | one $$V_\phi(s_t)$$ per position |
| Training signal | human preference data (or rules) | bootstrap from reward |
| When trained | **before** RL, frozen during RL | trained **together** with the policy |

Common misconception: "with an RM you don't need a Critic." In fact the two have completely different jobs: **the RM compresses (prompt, response) into a sparse scalar; the Critic spreads that scalar across each token.** One tells you "good or bad overall," the other tells you "good where, bad where."

### 5.3 Why Reasoning Loves Rule-based Verifiers

Four reasons:

**1. Hack-resistant.** The classic failure of a neural RM: policy learns to "fool" the RM. A rule-based verifier is deterministic — it cannot be hacked.

**2. Scalable.** Training an RM needs a lot of annotation. Writing a verifier function takes a few lines:

```python
def math_verifier(pred, gt):
    return sympy.simplify(parse(pred) - parse(gt)) == 0

def code_verifier(pred, tests):
    return all(run_test(pred, t) for t in tests)
```

**3. Signal is more direct.** An RM is a proxy indicator; the verifier is the true target. The more direct the training objective, the better RL converges.

**4. Matches the new outcome-only reward paradigm.** DeepSeek-R1 showed that outcome reward alone can elicit complex reasoning. Rule-based verifiers naturally fit this paradigm.

### 5.4 Limitations of Verifiers

Not every task admits a good verifier:

| Task | Easy to write a verifier? |
|------|---------------------------|
| Math | Yes, easy |
| Code | Yes, easy |
| Short-answer QA | Yes, moderate |
| Long-form writing | No, hard |
| Dialogue quality | No, hard |

This explains why the breakthroughs of reasoning LLMs are concentrated in math / code / QA — they have natural verifiers.

---

## 6. GRPO: The Trade-off of Dropping the Critic

GRPO[^grpo] is proposed by DeepSeek and is the RL algorithm used by DeepSeek-R1. **Core innovation: drop the Critic.**

### 6.1 Motivation

In LLM settings, the Critic is not cheap: it is typically the same scale as the policy and needs its own forward / backward, and its value estimates are inaccurate at cold-start. When reward is already outcome-level (one scalar per response), the token-level granularity a Critic provides may not be worth the compute.

### 6.2 Group Normalization

Sample $$G$$ responses for the same prompt, and compute advantage via **intra-group normalization**:

$$
\hat{A}_i = \frac{r_i - \text{mean}(r_1, ..., r_G)}{\text{std}(r_1, ..., r_G)}
$$

```python
def compute_grpo_advantages(rewards):
    """
    rewards: [G] rewards for G responses to the same prompt
    """
    mean = rewards.mean()
    std = rewards.std() + 1e-8
    return (rewards - mean) / std
```

Intuition: among $$G$$ responses to the same prompt, a response is "good" if it scores above the group mean. You don't need the Critic's absolute estimate — relative comparison within the group is enough.

### 6.3 Trade-off

**Benefits**:
- No Critic, about half the compute
- Simpler training pipeline
- Reasoning tasks already have outcome-level reward; token-level advantage isn't needed

**Costs**:
- Token-level information lost: can't distinguish "which tokens are critical within the same response"
- Depends on $$G$$ being large enough: GRPO does not work at $$G=1$$
- Reward collapse risk: late in training, when all $$G$$ responses are correct (or all wrong), std → 0

Search-R1's PPO vs GRPO comparison (Qwen2.5-7B):

| Method | Convergence speed | Stability | Final avg EM |
|--------|-------------------|-----------|--------------|
| PPO | slow (Critic needs warmup) | stable | 0.431 |
| GRPO | fast | tends to collapse late | 0.350–0.396 |

### 6.4 How to Choose

- **Reasoning + short training**: GRPO. reward is already outcome-level
- **Long training + chasing peak performance**: PPO. the Critic pays for itself
- **Compute-constrained**: GRPO. saves a model

---

## 7. DPO: Skipping the Reward Model and RL Altogether

DPO[^dpo] (Direct Preference Optimization) takes a completely different path: **no reward model, no RL rollout**. Preference pairs $$(x, y_w, y_l)$$ (where $$y_w$$ is the chosen response and $$y_l$$ the rejected one) are treated as plain supervised-learning data.

### 7.1 Motivation

Classic RLHF has a long pipeline: preference data → train RM → PPO rollout → policy update. Each step has its own failure modes: RM gets hacked, PPO is unstable, rollouts are expensive. The DPO question is: can we short-circuit this chain?

### 7.2 Key Derivation: The RLHF Optimum Has a Closed Form

The RLHF objective:

$$
\max_\pi \mathbb{E}_{x, y \sim \pi}[r(x, y)] - \beta D_{KL}[\pi \| \pi_{ref}]
$$

This problem has a closed-form optimal solution:

$$
\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{ref}(y \mid x) \exp\Big(\frac{1}{\beta} r(x, y)\Big)
$$

where $$Z(x) = \sum_y \pi_{ref}(y \mid x) \exp(r(x, y)/\beta)$$ is the normalization constant. Solving for the reward:

$$
r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{ref}(y \mid x)} + \beta \log Z(x)
$$

**Key observation**: reward can be written as a "log probability ratio." This is the pivot of the entire DPO derivation.

### 7.3 From Preference Probability to the DPO Loss

Plug into the Bradley-Terry preference model:

$$
P(y_w \succ y_l \mid x) = \sigma\big(r(x, y_w) - r(x, y_l)\big)
$$

The subtraction kills the intractable $$\log Z(x)$$ term. What remains only involves log probability ratios of the policy and the reference:

$$
\mathcal{L}^{DPO} = -\mathbb{E}_{(x, y_w, y_l)}\Big[\log \sigma\Big(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{ref}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{ref}(y_l \mid x)}\Big)\Big]
$$

This is just a binary classification loss (logistic regression on preference pairs) — no reward model, no rollout, no PPO.

### 7.4 Code

```python
def compute_dpo_loss(
    policy_chosen_logps,    # [B] log prob of y_w under pi_theta (sequence-level)
    policy_rejected_logps,  # [B] log prob of y_l under pi_theta
    ref_chosen_logps,       # [B] log prob of y_w under pi_ref
    ref_rejected_logps,     # [B] log prob of y_l under pi_ref
    beta=0.1,
):
    # log ratio of policy to reference for both chosen and rejected
    chosen_ratio = policy_chosen_logps - ref_chosen_logps
    rejected_ratio = policy_rejected_logps - ref_rejected_logps
    
    # how much more the policy prefers y_w over y_l, scaled by beta
    logits = beta * (chosen_ratio - rejected_ratio)
    
    # binary classification: push chosen above rejected
    return -F.logsigmoid(logits).mean()
```

**Implementation detail**: `logps` is the sum of log probabilities over the whole response (sequence-level), not per-token. This is the fundamental difference from PPO's token-level advantage.

### 7.5 DPO vs PPO: Trade-off

| Dimension | PPO | DPO |
|-----------|-----|-----|
| Data source | online rollout | offline preference pairs |
| Reward model | required | not required |
| Training stability | need to tune clip, KL, GAE, etc. | almost as simple as SFT |
| Data efficiency | rollouts are expensive | preferences collected once |
| OOD generalization | good (rollout enables exploration) | bounded by preference coverage |
| Granularity | token-level | sequence-level |
| Best fit | verifier available or online interaction | static preference data available |

**Key insight**: DPO reduces RL to supervised learning. The cost is losing "online exploration" — the policy can only learn from a static set of preference pairs and cannot self-improve via new rollouts. This is a genuine disadvantage for OOD generalization and reasoning tasks.

### 7.6 When to Choose DPO

- **Preference data already exists and rollouts are expensive**: engineering-friendly substitute for classic RLHF
- **Tasks like dialogue style or writing preference where verifiers are hard to write**
- **Engineering simplicity**: no desire to maintain an RM + PPO stack

Not a good fit:
- **Reasoning tasks with rule verifiers**: online exploration in PPO / GRPO is stronger
- **Agentic tasks requiring multi-turn environment interaction**: DPO has no concept of rollout and cannot handle this

DPO is complementary to PPO / GRPO, not a replacement. With this in mind, when you get to Search-R1 you won't ask "why not use DPO?" — Search-R1 fundamentally requires rollouts and environment interaction.

---

## 8. Search-R1: A Complete Instance of Agentic RL

All the pieces above come together in this section.

### 8.1 Task Formalization

A Search-R1 rollout is an alternating interaction between the LLM and a search engine:

```
<think>reasoning about what to search</think>
<search>query string</search>
<information>[retrieved passages injected by environment]</information>
<think>reasoning about retrieved info</think>
<search>follow-up query</search>
<information>[more retrieved passages]</information>
...
<answer>final answer</answer>
```

The template constrains only **structure**, not **strategy** — RL itself discovers the search policy.

### 8.2 Rollout Process

```python
def rollout_with_search(model, tokenizer, search_engine, prompt, max_turns=4):
    """
    Search-R1 rollout: interleave LLM generation with search engine calls.
    """
    tokens = tokenizer.encode(prompt)
    action_mask = [0] * len(tokens)  # prompt tokens are not LLM-generated
    
    for turn in range(max_turns):
        new_tokens, stop = model.generate(
            tokens,
            stop_sequences=["</search>", "</answer>"],
            max_new_tokens=500,
        )
        tokens.extend(new_tokens)
        action_mask.extend([1] * len(new_tokens))       # LLM-generated
        
        if stop == "</answer>":
            break
        elif stop == "</search>":
            query = extract_search_query(tokens)
            retrieved = search_engine.retrieve(query, top_k=3)
            retrieved_text = f"<information>{format_passages(retrieved)}</information>"
            retrieved_tokens = tokenizer.encode(retrieved_text)
            tokens.extend(retrieved_tokens)
            action_mask.extend([0] * len(retrieved_tokens))  # NOT LLM-generated
    
    return tokens, action_mask
```

### 8.3 Core Technical Contribution: Retrieved Token Masking

**Problem**: the rollout sequence mixes in tokens not produced by the LLM (retrieved passages). If you don't handle them, policy gradient will compute gradients on those tokens too — which is wrong: those tokens are not actions of the policy.

Worse, failing to mask lets the loss treat "the stylistic features of search-engine output" as something the policy should imitate. The policy ends up learning to mimic the retrieval text style instead of learning "how to search."

**Solution**: only compute loss over LLM-generated tokens:

$$
\mathcal{L}^{Search\text{-}R1} = \frac{\sum_t I(y_t) \cdot \mathcal{L}^{PPO}_t}{\sum_t I(y_t)}, \quad I(y_t) = \begin{cases} 1 & y_t \text{ is LLM-generated} \\ 0 & y_t \text{ is retrieved} \end{cases}
$$

In implementation, `action_mask` just gets passed into the PPO loss:

```python
def search_r1_loss(logits, old_logprobs, ref_logprobs, actions,
                   advantages, action_mask, clip_eps=0.2, kl_coef=0.001):
    """
    Identical to standard PPO loss, except action_mask now also
    excludes retrieved tokens in addition to padding.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    new_logprobs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    
    ratio = torch.exp(new_logprobs - old_logprobs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2)
    
    log_ratio_ref = ref_logprobs - new_logprobs
    kl = torch.exp(log_ratio_ref) - log_ratio_ref - 1
    
    # action_mask = 0 for padding OR retrieved tokens
    total = (policy_loss + kl_coef * kl) * action_mask
    return total.sum() / action_mask.sum()
```

**Effect** (Qwen2.5-7B, avg EM over 7 QA datasets):

| Setting | Avg EM |
|---------|--------|
| Do not mask retrieved tokens | 0.343 |
| Mask retrieved tokens | **0.431** |

This single change gives ~25% relative improvement. **The right definition of the action space can have a larger effect on RL training than the algorithmic changes themselves.**

More abstractly: **RL should only optimize variables the agent controls.** Retrieved content is the environment's response — the agent doesn't control it, so it shouldn't be shaped by policy gradient. This is a basic RL principle; it is easy to overlook in "pure LLM generation" settings, and Agentic RL makes it matter again.

### 8.4 Experimental Observations

**Main results** (avg EM over 7 QA datasets, Qwen2.5-7B):

| Method | Avg EM |
|--------|--------|
| Direct Inference | 0.181 |
| RAG | 0.304 |
| R1 (RL without search) | 0.276 |
| Rejection Sampling + SFT | 0.348 |
| **Search-R1 (PPO)** | **0.431** |

**Emergent behaviors**: the trained model exhibits patterns not explicitly taught by the paper:

1. **Self-verification**: after reaching an answer, issue one more query to confirm
2. **Query refinement**: if the first search fails, rewrite the query and retry
3. **Problem decomposition**: complex questions get automatically decomposed into sub-queries

Table 9 of the paper shows a case: asking "In which city and state was the singer of the perfume Curious born?" Search-R1 proceeds as:

1. `<search>Curious fragrance information</search>` → finds it's Britney Spears's
2. `<search>Britney Spears birthplace</search>` → McComb, Mississippi
3. `<search>McComb, Mississippi location</search>` → verification
4. `<answer>McComb, Mississippi</answer>`

Compare R1 without search: it guesses "Houston" (wrongly attributing Curious to Beyoncé).

---

## 9. A Panoramic Summary of the Design Space

### 9.1 Two Orthogonal Dimensions

```
                    │ Critic + GAE      │ Group Norm         │
                    │ (token-level adv) │ (response-level)   │
────────────────────┼───────────────────┼────────────────────┤
  Neural RM         │ classic RLHF      │ uncommon           │
────────────────────┼───────────────────┼────────────────────┤
  Rule Verifier     │ Search-R1         │ DeepSeek-R1        │
────────────────────┼───────────────────┼────────────────────┤
  Gen Verifier      │ emerging          │ emerging           │
```

**The horizontal axis governs cost and stability.** Critic + GAE needs one more model but is more stable; Group Norm saves a model at the cost of coarser granularity.

**The vertical axis governs applicable scope.** Neural RM covers more tasks but can be hacked; rule verifiers are hack-resistant but cover less; generative verifiers sit between the two.

### 9.2 The Essence of Agentic RL

Agentic RL is not a new algorithm — it is an **extension of the environment**. The environment of traditional RL for LLM has only prompt and terminal reward; the environment of Agentic RL also includes tools, intermediate feedback, and state changes.

The algorithm layer is still PPO or GRPO. The key change is in the rollout — no longer a one-shot generation but alternating interaction between policy and environment. This brings two new challenges:

1. **Non-policy tokens mixed into the rollout sequence** → need retrieved token masking
2. **Higher rollout cost** → higher bar for sample efficiency

### 9.3 Current Limitations and Open Problems

1. **Reward design for open-ended tasks**: rule verifiers only work for tasks with clear right and wrong. Writing, dialogue, and similar tasks still need neural RMs or human feedback.
2. **Long-horizon reasoning**: when the reasoning chain runs to tens of thousands of tokens, both Critic and Group Norm face challenges.
3. **Multi-tool coordination**: how to attribute reward across tools, how to manage the rollout of complex tool-call graphs — both still open.
4. **Theoretical explanation of emergent behaviors under outcome reward**: why do behaviors like self-verification emerge spontaneously? Under what conditions? No systematic answer yet.

---

## 10. Closing and References

### Quick Index

```
Cross Entropy                  -log π(y_t)              per-token base loss
SFT Loss                       Σ CE                     unweighted
REINFORCE                      Σ CE · A_t               advantage-weighted
PPO                            + clip + min + KL        stabilized policy gradient
GRPO                           + group norm             drops the Critic
DPO                            logistic on pref pairs   offline path skipping RL
Search-R1                      + retrieved token mask   Agentic RL instance

Policy (π_θ)                   LLM being trained
Reference (π_ref)              LLM at training start, frozen
Old (π_old)                    snapshot at this rollout
Critic (V_φ)                   estimates future return
Reward Model                   scalar on (prompt, response)
Verifier                       general concept: function/model judging correctness
```

</div>

<div class="lang-content lang-zh" markdown="1">


## 前言

过去一年 reasoning LLM 的主要进展——o1、DeepSeek-R1、Search-R1——共享一套 RL 技术基础，却在几个关键设计点上做了不同选择。把这些选择并列对比，比单独读任一篇论文都更能看清设计空间。

这篇文章做的就是这件事。主线是 Search-R1[^sr1]：它是 Agentic RL 的一个清晰实例，又足够简单，能把 PPO、GRPO、verifier、token masking 这些概念自然串起来。

假设读者熟悉 transformer、SFT 和基础 DL。不假设熟悉 RL 细节。

不覆盖：RLHF 偏好数据构造、inference-time scaling（best-of-N、tree search）、multi-agent、硬件工程优化。

---

## 1. 问题设定：post-training 的技术栈演进

从 pretrained LLM 走到能自主调用工具的 agent，中间经过几次技术跳跃。每一跳都在解决上一跳的局限：

```
Pretrain
   │ 知道很多，但不听话
   ↓
SFT (Supervised Fine-Tuning)
   │ 只会模仿，不会主动抑制错误
   ↓
RLHF (PPO + Reward Model)
   │ RM 易被 hack；偏好数据贵；reasoning 信号弱
   ↓
Reasoning RL (GRPO + Verifier)
   │ reward 只从模型自己的输出来，无法与外界交互
   ↓
Agentic RL (+ Tool / Environment)
```

每一跳的技术内核：

- **SFT**：cross entropy + 监督学习
- **RLHF**：PPO + 神经网络 reward model
- **Reasoning RL**：GRPO + rule-based verifier，砍掉 reward model
- **Agentic RL**：在 Reasoning RL 基础上，把环境交互嵌入 rollout

"Agentic RL" 并不是一个独立的新算法，而是"在 rollout 中引入环境交互"的训练范式。算法层仍然是 PPO 或 GRPO。关键变化在 **环境的扩展**：rollout 中间会插入工具调用，执行或检索的结果被注入到 sequence，成为后续 generation 的 context。

Search-R1 是这个范式最干净的实例：把搜索引擎作为 environment，让 LLM 在 reasoning 过程中自主决定何时搜、搜什么。

在这条线性主线之外还有一条支线：**DPO** 把 RLHF 直接化成监督学习，不跑 RL、不训 reward model。§7 会单独讨论。

下面的章节依次展开每个组件。

---

## 2. 从 Cross Entropy 到 Policy Gradient

本节建立 RL loss 和 SFT loss 的等价关系，作为后面理解 PPO 的地基。

### 2.1 Cross Entropy 在 next-token prediction 下的化简

一般定义：

$$
H(p, q) = -\sum_i p_i \log q_i
$$

next-token 场景下真实标签 $$p$$ 是 one-hot，$$p_{y_t} = 1$$、其他为 0。对 vocab 求和只剩一项：

$$
H(p, q) = -\log \pi_\theta(y_t \mid y_{<t})
$$

SFT 对整个序列求和：

$$
\mathcal{L}^{SFT} = -\sum_{t=1}^{T} \log \pi_\theta(y_t \mid y_{<t})
$$

### 2.2 CE 和 KL 的细微之处

两者的恒等式：

$$
H(p, q) = H(p) + D_{KL}(p \| q)
$$

监督学习里 $$p$$ 是 ground truth、固定不变，$$H(p)$$ 对 $$\theta$$ 是常数——所以 **优化 CE 和优化 KL 在梯度上等价**。我们默认用 CE 仅仅是因为它更便宜。

在 RL 里这个等价会失效：KL 的两边都是模型（$$\pi_\theta$$ vs $$\pi_{ref}$$），都不固定。这时 CE 和 KL 的差别才真正重要。后面 PPO 的 KL penalty 会用到这一点。

### 2.3 从 SFT 到 REINFORCE

SFT loss 对每个 token 都在"无条件推高它的概率"——demo data 里的 token 被默认是好的。在 imitation learning 里这个假设合理，但 RL 想做的更多：**既要推高好行为，也要主动压低坏行为**。

REINFORCE 在 SFT loss 上乘一个权重 $$A_t$$：

$$
\mathcal{L}^{REINFORCE} = -\mathbb{E}\Big[\sum_t \log \pi_\theta(y_t \mid y_{<t}) \cdot A_t\Big]
$$

- $$A_t > 0$$：好 token，推高概率
- $$A_t < 0$$：差 token，压低概率
- $$\lvert A_t \rvert$$ 越大，调整幅度越大

三个 loss 的统一形式：

| Loss | 单 token 形式 | 权重含义 |
|------|--------------|---------|
| SFT | $$-\log \pi_\theta(y_t)$$ | 1（所有 token 都推高）|
| REINFORCE | $$-\log \pi_\theta(y_t) \cdot A_t$$ | advantage 加权 |
| PPO | $$-\min(r_t A_t, \text{clip}(r_t) A_t)$$ | ratio 替代 log（下节解释）|

**核心洞察**：RL loss 在形式上就是"带符号、加权的 cross entropy"。SFT 在 $$A_t \equiv 1$$ 时和 RL 在 **梯度形式** 上重合。需注意这只是形式上的类比——SFT 采样自 demo data，RL 采样自 $$\pi_\theta$$，两者的样本分布并不相同。

### 2.4 代码对比

```python
# SFT loss
def sft_loss(logits, actions, mask):
    log_probs = F.log_softmax(logits, dim=-1)
    token_logprobs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    return -(token_logprobs * mask).sum() / mask.sum()

# REINFORCE loss: only differs by * advantages
def reinforce_loss(logits, actions, advantages, mask):
    log_probs = F.log_softmax(logits, dim=-1)
    token_logprobs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    return -(token_logprobs * advantages * mask).sum() / mask.sum()
```

两者差别就一个 `* advantages`。正是这个乘法打开了 SFT 和 RL 的分野：**RL 能用负权重抑制错误行为，SFT 不能**。

---

## 3. PPO 的设计动机与每个组件的作用

PPO[^ppo] 是当前 LLM RL 训练的事实标准。本节按"组件 → 解决什么问题"的顺序拆开。

下面写的是 LLM RLHF 版 PPO，比原 PPO 论文多了一项 $$\pi_{ref}$$ 的 KL 惩罚（这是 InstructGPT[^instructgpt] 起带进来的工程惯例，不是原始 PPO 的一部分）：

$$
\mathcal{L}^{PPO}(\theta) = -\mathbb{E}_t\Big[\min\big(r_t(\theta) A_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\big)\Big] + \beta \cdot D_{KL}\big[\pi_\theta \| \pi_{ref}\big]
$$

### 3.1 起点：REINFORCE 的两个病

**病一：方差大，单步更新可能崩。** $$A_t$$ 从单条 trajectory 估，方差大。偶然的高 reward 样本让某些 token 的 $$A_t$$ 异常大，一次 gradient step 就能把 $$\pi_\theta$$ 推到奇怪位置。

**病二：样本效率太低。** REINFORCE 严格 on-policy：每次更新后旧 rollout 作废。对 LLM 这种 rollout 极贵的场景不可接受。

PPO 的每个组件都在治这两个病。

### 3.2 Importance Ratio：让一批 rollout 多用几次

$$
r_t(\theta) = \frac{\pi_\theta(y_t \mid y_{<t})}{\pi_{old}(y_t \mid y_{<t})}
$$

$$\pi_{old}$$ 是采样这批 rollout 时的 policy 快照。训练循环变成：

```python
for iteration in range(N):
    # snapshot current policy
    pi_old = copy(pi_theta).detach()
    
    # collect rollouts using pi_old
    rollouts = sample_rollouts(pi_old, batch_size=B)
    advantages = compute_gae(rollouts)
    
    # multiple gradient updates on the same rollouts
    for epoch in range(K):  # typically K = 2~4
        for minibatch in split(rollouts):
            loss = ppo_loss(pi_theta, pi_old, minibatch, advantages)
            loss.backward()
            optimizer.step()
```

同批 rollout 被用 $$K$$ 次。更新到第 $$k$$ 个 epoch 时 $$\pi_\theta \neq \pi_{old}$$，这就是 off-policy 的。Importance ratio 在数学上修正这个 mismatch（importance sampling 的标准应用）。

**符号陷阱**：这里 $$r_t$$ 是 importance ratio，不是 reward。Reward 在 PPO 中通过 GAE 化成 $$A_t$$，loss 里根本没有 reward 出现。

### 3.3 Clip：单步幅度约束

只有 ratio 还不够。如果 $$\pi_\theta$$ 偏离 $$\pi_{old}$$ 太远（比如 $$r_t = 5$$），importance sampling 的方差会爆炸。Clip 硬压回来：

$$
\text{clip}(r_t, 1-\epsilon, 1+\epsilon)
$$

典型 $$\epsilon = 0.2$$，policy 在单个 token 上的概率比例变化不超过 ±20%。

**但单独用 clip 有坑**。设 $$A_t = +1$$（好 token），$$r_t = 0.5$$（policy 反而压低了它——这是错误状态）。如果直接用 $$\text{clip}(r_t) \cdot A_t = 0.8$$，梯度被 **人为放大**，反而鼓励 policy 继续走错方向。

### 3.4 Min：让 clip 只刹车、不阻止纠错

PPO 的完整 policy 项：

$$
L_t^{CLIP} = \min\big(r_t A_t,\ \text{clip}(r_t, 1-\epsilon, 1+\epsilon) \cdot A_t\big)
$$

设 $$\epsilon = 0.2$$，四种情况：

| 情况 | $$A_t$$ | $$r_t$$ | $$r_t A_t$$ | $$\text{clip}(r_t) A_t$$ | min 选谁 | 含义 |
|------|--------|--------|------------|--------------------------|---------|------|
| B | $$+1$$ | $$1.5$$（超上界）| $$1.5$$ | $$1.2$$ | clip 项 | 好 token 推过头，**饱和** |
| C | $$+1$$ | $$0.5$$（低于下界）| $$0.5$$ | $$0.8$$ | **原项** | 好 token 反被压低，**允许纠正** |
| D | $$-1$$ | $$0.5$$（低于下界）| $$-0.5$$ | $$-0.8$$ | clip 项 | 坏 token 压够了，**饱和** |
| E | $$-1$$ | $$1.5$$（超上界）| $$-1.5$$ | $$-1.2$$ | **原项** | 坏 token 反被推高，**允许纠正** |

规律：

- 朝 **正确方向** 走过头（B, D）→ min 选 clip 项，梯度饱和
- 朝 **错误方向** 走（C, E）→ min 选原项，梯度保留，继续纠错

Schulman 称此为 **pessimistic bound**：取较小者做保守估计。一举两得：限制单步幅度（B, D），但不阻断错误修复（C, E）。

### 3.5 KL Penalty：长期防漂移

Clip 管 **单步** 稳定性。但即使每步都在 $$[1-\epsilon, 1+\epsilon]$$，多 iteration 累积仍可能让 $$\pi_\theta$$ 漂到完全不同的分布。典型失败模式是 **reward hacking**：模型发现某种生成模式能稳定拿高 reward，但语言质量崩坏；每步 clip 都不触发，但累积后 policy 已面目全非。

PPO 加 KL 项：

$$
\beta \cdot D_{KL}\big[\pi_\theta(\cdot \mid s_t) \| \pi_{ref}(\cdot \mid s_t)\big]
$$

$$\pi_{ref}$$ 是 **训练起点** 的 policy（通常是 SFT 后的模型），整个 RL 过程冻结。它和 $$\pi_{old}$$ 不是一回事：

| 维度 | Clip on $$r_t = \pi_\theta / \pi_{old}$$ | KL on $$\pi_\theta \| \pi_{ref}$$ |
|------|----------------------------------------|-----------------------------------|
| 比较对象 | 上一轮 rollout 时的 policy | 训练起点的 policy |
| 时间尺度 | 单 iteration | 整个训练过程 |
| 形式 | 硬约束（饱和截断）| 软约束（连续惩罚）|
| 解决问题 | 单步更新过激 | 累积漂移 |

两者 **互补，不可替代**。只有 clip、没有 KL，长期会 reward hacking；只有 KL、没有 clip，单次大更新可能直接崩。

**k3 estimator**。Token 级 KL 按定义要遍历整个 vocab，开销大。实际用 Schulman 的 k3[^k3]：

$$
\text{KL}_{k3} = \exp(\log\pi_{ref} - \log\pi_\theta) - (\log\pi_{ref} - \log\pi_\theta) - 1
$$

严格讲 k3 是 $$D_{KL}[p \| q]$$ 在 "从 $$p$$ 采样" 时的低方差无偏估计。PPO 中我们实际从 $$\pi_{old}$$ 采样（因为 rollout 是用 $$\pi_{old}$$ 跑的），严格来说是一个近似。但 $$\pi_{old} \approx \pi_\theta$$（在 clip 的约束下），实践中这个近似足够好。

代码一行：

```python
log_ratio = ref_logprobs - new_logprobs
kl = torch.exp(log_ratio) - log_ratio - 1  # k3 estimator
```

### 3.6 完整代码

```python
def compute_ppo_loss(
    logits,          # [B, T, V] current policy outputs
    old_logprobs,    # [B, T] log prob from pi_old (detached)
    ref_logprobs,    # [B, T] log prob from pi_ref (frozen)
    actions,         # [B, T] sampled tokens
    advantages,      # [B, T] from GAE (Section 4)
    action_mask,     # [B, T] 1 for policy-generated positions
    clip_eps=0.2,
    kl_coef=0.001,
):
    log_probs = F.log_softmax(logits, dim=-1)
    new_logprobs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    
    # === importance ratio (3.2) ===
    ratio = torch.exp(new_logprobs - old_logprobs)
    
    # === clip + min (3.3, 3.4) ===
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2)
    
    # === KL penalty (3.5, k3 estimator) ===
    log_ratio_ref = ref_logprobs - new_logprobs
    kl = torch.exp(log_ratio_ref) - log_ratio_ref - 1
    
    total = (policy_loss + kl_coef * kl) * action_mask
    return total.sum() / action_mask.sum()
```

### 3.7 五个正则机制一览

| 机制 | 作用对象 | 解决问题 |
|------|---------|---------|
| Importance ratio | data / policy mismatch | 让 rollout 多次复用 |
| Clip on ratio | 单 token 单步更新 | 单次偏离 $$\pi_{old}$$ 过远 |
| Min on clip | clip 的副作用 | 错误方向保留纠错梯度 |
| KL on $$\pi_{ref}$$ | 累积漂移 | reward hacking |
| Value baseline | advantage 估计 | 降方差、实现 credit assignment |

PPO 的工程稳定性是这五件事叠加的结果。去掉其中任何一个，某种 setting 下训练会变脆。

---

## 4. 稀疏 Reward 与 Credit Assignment

### 4.1 问题定义

LLM RL 中 reward 通常只在 episode 末尾有值。Search-R1 典型 episode：

```
<think>...</think><search>...</search><information>...</information>
<think>...</think><search>...</search><information>...</information>
<think>...</think><answer>McComb, Mississippi</answer>   ← reward=1 只在这里
```

500 个 token，只有最后一位有非零 reward。直接把 terminal reward 当作所有 token 的 $$A_t$$ 会有两个问题：所有 token 等权（无法区分关键 token），梯度方差大（每条 episode 只贡献一个标量 signal）。

这就是 **credit assignment** 问题。

### 4.2 Value Function

标准解法是引入 value function $$V_\phi$$：

$$
V_\phi(s_t) = \mathbb{E}\Big[\sum_{k=0}^{T-t} \gamma^k r_{t+k}\Big]
$$

advantage 定义为：

$$
A_t = R_t - V_\phi(s_t)
$$

- $$A_t > 0$$：实际回报超预期 → 好选择
- $$A_t < 0$$：低于预期 → 差选择
- $$A_t \approx 0$$：不关键

value function 把稀疏的 terminal reward **摊平** 到每个 token 位置上。

### 4.3 Monte Carlo vs TD

$$R_t$$ 的两种估计方式：

| 方法 | 公式 | 性质 |
|------|------|------|
| Monte Carlo | $$R_t^{MC} = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$$ | 无偏、方差高 |
| TD | $$R_t^{TD} = r_t + \gamma V_\phi(s_{t+1})$$ | 有偏、方差低 |

### 4.4 GAE：两者的指数加权混合

GAE[^gae] 用 $$\lambda \in [0, 1]$$ 插值：

$$
A_t^{GAE} = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
$$

递推形式：

$$
A_t^{GAE} = \delta_t + \gamma \lambda A_{t+1}^{GAE}
$$

```python
def compute_gae(rewards, values, gamma=1.0, lam=1.0):
    """
    rewards: [T] per-step rewards (mostly zero in LLM setting)
    values:  [T+1] value estimates; values[T] is terminal bootstrap
    """
    T = len(rewards)
    advantages = torch.zeros(T)
    gae = 0.0
    
    # iterate in reverse: GAE is an exponentially weighted sum of future deltas
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    
    returns = advantages + values[:T]
    return advantages, returns
```

Search-R1 用 $$\lambda = \gamma = 1$$。此时 GAE 递推退化成 $$A_t = \sum_l \delta_{t+l}$$，telescope 后等于 $$R_t - V(s_t)$$——就是 MC advantage。

### 4.5 为什么稀疏 reward 在 LLM 上能 work

1. **预训练提供 strong prior**：RL 不是从零学，只是 fine-grained 调整
2. **Token 的因果结构**：早期好选择和最终成功高度相关
3. **Value function 的平滑**：把 terminal reward 摊到每个位置
4. **Batch size 的统计力量**：Search-R1 batch = 512，梯度相对稳定

---

## 5. Reward 来源：Reward Model vs Verifier

### 5.1 两个正交维度

LLM RL 的设计空间有两个正交轴：

**轴一：Reward 从哪来**（optimize what）
- Neural Reward Model（RM）
- Rule-based Verifier
- Generative Verifier

**轴二：Credit Assignment 怎么做**（propagate how）
- Critic + GAE（PPO）
- Group Normalization（GRPO）
- Offline data（DPO[^dpo]）

| Reward 来源 | Credit Assignment | 代表工作 |
|------------|-------------------|---------|
| Neural RM | Critic + GAE | 经典 RLHF（InstructGPT）|
| Rule Verifier | Critic + GAE | Search-R1 |
| Rule Verifier | Group Norm | DeepSeek-R1[^r1] |

### 5.2 Reward Model vs Critic：常被搞混的两个概念

| 维度 | Reward Model | Critic (Value Model) |
|------|--------------|---------------------|
| 评什么 | 整条 response 的 **绝对质量** | 当前状态的 **未来期望回报** |
| 输入 | (prompt, response) | 当前 state $$s_t$$ |
| 输出 | 单个 scalar $$r$$ | 每个位置一个 $$V_\phi(s_t)$$ |
| 训练信号 | 人类偏好数据（或规则）| 从 reward bootstrap |
| 训练时机 | RL **之前** 训好，RL 中冻结 | 和 policy **一起** 训 |

常见误解："有 RM 就不需要 Critic"。事实上两者职责完全不同：**RM 把 (prompt, response) 压成一个稀疏 scalar，Critic 把这个 scalar 摊平到每个 token**。一个管"整体是好是坏"，一个管"好在哪里坏在哪里"。

### 5.3 为什么 Reasoning 爱用 Rule-based Verifier

四个原因：

**1. 抗 reward hacking。** Neural RM 的经典问题：policy 学会 "骗过" RM。Rule-based verifier 是确定性的，无法被 hack。

**2. 可扩展。** 训 RM 要大量标注。写 verifier 函数只要几行：

```python
def math_verifier(pred, gt):
    return sympy.simplify(parse(pred) - parse(gt)) == 0

def code_verifier(pred, tests):
    return all(run_test(pred, t) for t in tests)
```

**3. 信号更直接。** RM 是 proxy indicator，verifier 是真实目标。训练目标越直接，RL 越好收敛。

**4. 匹配 outcome-only reward 的新范式。** DeepSeek-R1 证明了只用 outcome reward 就能涌现复杂 reasoning。Rule-based verifier 天然适合。

### 5.4 Verifier 的局限

不是所有任务都有好的 verifier：

| 任务 | Verifier 容易写吗 |
|------|-------------------|
| 数学题 | ✅ 容易 |
| 代码 | ✅ 容易 |
| 短答 QA | ✅ 中等 |
| 长文写作 | ❌ 难 |
| 对话质量 | ❌ 难 |

这解释了为什么 reasoning LLM 的突破集中在 math / code / QA——它们有天然 verifier。

---

## 6. GRPO：去掉 Critic 的 Trade-off

GRPO[^grpo] 由 DeepSeek 提出，是 DeepSeek-R1 用的 RL 算法。**核心创新：去掉 Critic**。

### 6.1 动机

LLM 场景下 Critic 不便宜：通常和 policy 同等规模，需要独立的 forward / backward，冷启动时 value 估计也不准。如果 reward 本来就是 outcome-level（一条 response 一个分），token-level Critic 的细粒度信号未必值这个计算开销。

### 6.2 Group Normalization

对同一个 prompt 采样 $$G$$ 个 response，用 **组内归一化** 得 advantage：

$$
\hat{A}_i = \frac{r_i - \text{mean}(r_1, ..., r_G)}{\text{std}(r_1, ..., r_G)}
$$

```python
def compute_grpo_advantages(rewards):
    """
    rewards: [G] rewards for G responses to the same prompt
    """
    mean = rewards.mean()
    std = rewards.std() + 1e-8
    return (rewards - mean) / std
```

直觉：同 prompt 下的 $$G$$ 个 response，好的相对于组内均值就是好。不需要 Critic 给绝对评分，组内相对比较就够了。

### 6.3 Trade-off

**好处**：
- ✅ 省掉 Critic，计算量约减半
- ✅ 训练流程更简单
- ✅ Reasoning 任务 reward 本就 outcome-level，不需要 token-level advantage

**代价**：
- ❌ Token-level 信息丢失：无法区分"同一 response 中哪些 token 关键"
- ❌ 依赖 $$G$$ 够大：$$G=1$$ 时 GRPO 无法工作
- ❌ Reward collapse 风险：训练后期 $$G$$ 个 responses 全对或全错时 std 趋零

Search-R1 的 PPO vs GRPO 对比（Qwen2.5-7B）：

| 方法 | 收敛速度 | 稳定性 | 最终 avg EM |
|------|---------|-------|------------|
| PPO | 慢（Critic 需 warmup）| 稳定 | 0.431 |
| GRPO | 快 | 后期易崩 | 0.350–0.396 |

### 6.4 怎么选

- **Reasoning + 短训练**：GRPO。reward 本就 outcome-level
- **长训练 + 追求最终性能**：PPO。Critic 开销值得
- **计算资源紧张**：GRPO。省一个模型

---

## 7. DPO：绕开 reward model 和 RL 的捷径

DPO[^dpo]（Direct Preference Optimization）走了完全不同的路：**不训 reward model，也不做 RL rollout**。直接把偏好对 $$(x, y_w, y_l)$$（$$y_w$$ 是 chosen、$$y_l$$ 是 rejected）当监督学习数据。

### 7.1 动机

经典 RLHF 训练链路很长：preference data → 训 RM → PPO rollout → policy update。每一步都有各自的失败模式：RM 被 hack、PPO 不稳、rollout 贵。DPO 的问题是：能不能把这条链短路？

### 7.2 核心推导：RLHF 的最优解有闭式

RLHF 的目标是：

$$
\max_\pi \mathbb{E}_{x, y \sim \pi}[r(x, y)] - \beta D_{KL}[\pi \| \pi_{ref}]
$$

这个问题有闭式最优解：

$$
\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{ref}(y \mid x) \exp\Big(\frac{1}{\beta} r(x, y)\Big)
$$

其中 $$Z(x) = \sum_y \pi_{ref}(y \mid x) \exp(r(x, y)/\beta)$$ 是归一化常数。反解出 reward：

$$
r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{ref}(y \mid x)} + \beta \log Z(x)
$$

**关键观察**：reward 可以写成 "log 概率比" 的形式。这是 DPO 整个推导的支点。

### 7.3 从偏好概率到 DPO loss

代入 Bradley-Terry 偏好模型：

$$
P(y_w \succ y_l \mid x) = \sigma\big(r(x, y_w) - r(x, y_l)\big)
$$

两个 reward 相减，难算的 $$\log Z(x)$$ 项消掉。剩下只包含 policy 和 reference 的 log 概率比：

$$
\mathcal{L}^{DPO} = -\mathbb{E}_{(x, y_w, y_l)}\Big[\log \sigma\Big(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{ref}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{ref}(y_l \mid x)}\Big)\Big]
$$

这就是一个普通的二分类 loss（logistic regression on preference pairs），不需要 reward model、不需要 rollout、不需要 PPO。

### 7.4 代码

```python
def compute_dpo_loss(
    policy_chosen_logps,    # [B] log prob of y_w under pi_theta (sequence-level)
    policy_rejected_logps,  # [B] log prob of y_l under pi_theta
    ref_chosen_logps,       # [B] log prob of y_w under pi_ref
    ref_rejected_logps,     # [B] log prob of y_l under pi_ref
    beta=0.1,
):
    # log ratio of policy to reference for both chosen and rejected
    chosen_ratio = policy_chosen_logps - ref_chosen_logps
    rejected_ratio = policy_rejected_logps - ref_rejected_logps
    
    # how much more the policy prefers y_w over y_l, scaled by beta
    logits = beta * (chosen_ratio - rejected_ratio)
    
    # binary classification: push chosen above rejected
    return -F.logsigmoid(logits).mean()
```

**实现细节**：`logps` 是整个 response 的 log 概率之和（sequence-level），不是单 token。这和 PPO token-level advantage 是本质区别。

### 7.5 DPO vs PPO：Trade-off

| 维度 | PPO | DPO |
|------|-----|-----|
| 数据来源 | online rollout | offline preference pairs |
| Reward model | 需要 | 不需要 |
| 训练稳定性 | 需调 clip、KL、GAE 等超参 | 几乎像 SFT 一样好训 |
| 数据效率 | rollout 昂贵 | 偏好数据一次性收集 |
| 分布外泛化 | 好（rollout 中能探索）| 依赖偏好数据覆盖 |
| 粒度 | token-level | sequence-level |
| 适用场景 | 有 verifier 或在线交互 | 有静态偏好数据 |

**核心洞察**：DPO 把 RL 问题化成了监督学习。代价是失去了 "online exploration"——policy 只能从静态偏好对学，无法从新的 rollout 自我改进。这在分布外泛化和 reasoning 任务上是实打实的 disadvantage。

### 7.6 什么时候选 DPO

- **偏好数据已有、rollout 贵**：经典 RLHF 的工程替代品
- **对话风格、写作偏好等 "难写 verifier" 的任务**
- **工程简单**：不想维护 RM + PPO 基础设施

不适合：
- **有 rule verifier 的 reasoning 任务**：PPO / GRPO 的在线探索更强
- **需要多轮环境交互的 Agentic 任务**：DPO 没有 rollout 的概念，无法处理

DPO 和 PPO / GRPO 互为补充，不是替代。理解清楚这点，后面看 Search-R1 时就不会问 "为什么不用 DPO"——因为 Search-R1 本质需要 rollout 和环境交互。

---

## 8. Search-R1：Agentic RL 的完整实例

前面所有组件在这一节组装起来。

### 7.1 任务形式化

Search-R1 的 rollout 是 LLM 和搜索引擎的交替交互：

```
<think>reasoning about what to search</think>
<search>query string</search>
<information>[retrieved passages injected by environment]</information>
<think>reasoning about retrieved info</think>
<search>follow-up query</search>
<information>[more retrieved passages]</information>
...
<answer>final answer</answer>
```

Template 只约束 **结构**，不约束 **策略**——让 RL 自己发现好的搜索 policy。

### 7.2 Rollout 过程

```python
def rollout_with_search(model, tokenizer, search_engine, prompt, max_turns=4):
    """
    Search-R1 rollout: interleave LLM generation with search engine calls.
    """
    tokens = tokenizer.encode(prompt)
    action_mask = [0] * len(tokens)  # prompt tokens are not LLM-generated
    
    for turn in range(max_turns):
        new_tokens, stop = model.generate(
            tokens,
            stop_sequences=["</search>", "</answer>"],
            max_new_tokens=500,
        )
        tokens.extend(new_tokens)
        action_mask.extend([1] * len(new_tokens))       # LLM-generated
        
        if stop == "</answer>":
            break
        elif stop == "</search>":
            query = extract_search_query(tokens)
            retrieved = search_engine.retrieve(query, top_k=3)
            retrieved_text = f"<information>{format_passages(retrieved)}</information>"
            retrieved_tokens = tokenizer.encode(retrieved_text)
            tokens.extend(retrieved_tokens)
            action_mask.extend([0] * len(retrieved_tokens))  # NOT LLM-generated
    
    return tokens, action_mask
```

### 7.3 核心技术贡献：Retrieved Token Masking

**问题**：rollout sequence 里混入了非 LLM 生成的 token（retrieved passages）。如果不处理，policy gradient 会对这些 token 也算梯度——这是错的：它们不是 policy 的 action。

更糟的是，不 mask 会让 loss 把 "搜索引擎返回的文本风格" 当成 policy 要学的东西，policy 学着模仿 retrieval 文风，而不是学 "如何搜索"。

**解法**：loss 只对 LLM 生成的 token 计算：

$$
\mathcal{L}^{Search\text{-}R1} = \frac{\sum_t I(y_t) \cdot \mathcal{L}^{PPO}_t}{\sum_t I(y_t)}, \quad I(y_t) = \begin{cases} 1 & y_t \text{ is LLM-generated} \\ 0 & y_t \text{ is retrieved} \end{cases}
$$

实现上就是把 `action_mask` 传进 PPO loss：

```python
def search_r1_loss(logits, old_logprobs, ref_logprobs, actions,
                   advantages, action_mask, clip_eps=0.2, kl_coef=0.001):
    """
    Identical to standard PPO loss, except action_mask now also
    excludes retrieved tokens in addition to padding.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    new_logprobs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    
    ratio = torch.exp(new_logprobs - old_logprobs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2)
    
    log_ratio_ref = ref_logprobs - new_logprobs
    kl = torch.exp(log_ratio_ref) - log_ratio_ref - 1
    
    # action_mask = 0 for padding OR retrieved tokens
    total = (policy_loss + kl_coef * kl) * action_mask
    return total.sum() / action_mask.sum()
```

**效果**（Qwen2.5-7B, avg EM over 7 QA datasets）：

| Setting | Avg EM |
|---------|--------|
| 不 mask retrieved tokens | 0.343 |
| Mask retrieved tokens | **0.431** |

单这一个改动带来约 25% 相对提升。**正确定义 action space 对 RL 训练的影响，可能比算法本身的改动更大**。

更抽象地说：**RL 应该只优化 agent 能控制的变量**。Retrieved content 是环境的 response，agent 不控制它，不该被 policy gradient 塑形。这是 RL 的基础原则，在 "纯 LLM generation" setting 下容易被忽略；Agentic RL 让它重新变得重要。

### 7.4 实验观察

**主结果**（avg EM over 7 QA datasets, Qwen2.5-7B）：

| 方法 | Avg EM |
|------|--------|
| Direct Inference | 0.181 |
| RAG | 0.304 |
| R1 (RL without search) | 0.276 |
| Rejection Sampling + SFT | 0.348 |
| **Search-R1 (PPO)** | **0.431** |

**涌现行为**：训出的模型展现了论文没有显式教授的 pattern：

1. **Self-verification**：拿到答案后再发一次 query 确认
2. **Query refinement**：首次搜索失败后改写 query 重试
3. **Problem decomposition**：复杂问题自动拆成多个 sub-query

论文 Table 9 的案例：问 "Curious 香水的歌手出生在哪个城市和州"。Search-R1 会：

1. `<search>Curious fragrance information</search>` → 发现是 Britney Spears 的
2. `<search>Britney Spears birthplace</search>` → McComb, Mississippi
3. `<search>McComb, Mississippi location</search>` → 验证确认
4. `<answer>McComb, Mississippi</answer>`

对比没有搜索能力的 R1：直接猜成 "Houston"（错把 Curious 归给 Beyoncé）。

---

## 9. 设计空间的全景总结

### 8.1 两个正交维度

```
                    │ Critic + GAE      │ Group Norm         │
                    │ (token-level adv) │ (response-level)   │
────────────────────┼───────────────────┼────────────────────┤
  Neural RM         │ 经典 RLHF          │ 少见                │
────────────────────┼───────────────────┼────────────────────┤
  Rule Verifier     │ Search-R1          │ DeepSeek-R1        │
────────────────────┼───────────────────┼────────────────────┤
  Gen Verifier      │ 新兴方向           │ 新兴方向            │
```

**横轴决定成本和稳定性**。Critic + GAE 多一个模型但更稳；Group Norm 省一个模型但粒度更粗。

**纵轴决定适用范围**。Neural RM 覆盖广但易 hack；Rule Verifier 抗 hack 但覆盖窄；Generative Verifier 介于两者之间。

### 8.2 Agentic RL 的本质

Agentic RL 不是新算法，是 **环境的扩展**。传统 RL for LLM 的环境只有 prompt 和 terminal reward；Agentic RL 的环境还包括工具、中间反馈、状态变化。

算法层仍是 PPO 或 GRPO。关键变化在 rollout——不再是一次性 generation，而是 policy 与环境的交替交互。这带来两个新挑战：

1. **Rollout sequence 里混入非 policy token** → 需要 retrieved token masking
2. **Rollout 成本更高** → 对 sample efficiency 要求更高

### 8.3 当前的局限与开放问题

1. **开放式任务的 reward 设计**：Rule verifier 只适用有明确对错的任务。写作、对话等仍需 Neural RM 或人类反馈。
2. **长 horizon reasoning**：推理链达几万 token 时，Critic 和 Group Norm 都面临挑战。
3. **Multi-tool 协同**：reward 如何跨工具分配、rollout 如何管理复杂工具调用图，都还 open。
4. **Outcome reward 下涌现行为的理论解释**：为什么 self-verification 等行为会自发出现？在什么条件下？缺乏系统回答。

---

## 10. 结语与参考资料

### 快速索引

```
Cross Entropy                  -log π(y_t)              单 token 基础 loss
SFT Loss                       Σ CE                     无权重
REINFORCE                      Σ CE · A_t               advantage 加权
PPO                            + clip + min + KL        稳定化的 policy gradient
GRPO                           + group norm             去掉 Critic
DPO                            logistic on pref pairs   跳过 RL 的离线路线
Search-R1                      + retrieved token mask   Agentic RL 实例

Policy (π_θ)                   被训练的 LLM
Reference (π_ref)              训练起点的 LLM，冻结
Old (π_old)                    本轮 rollout 时的快照
Critic (V_φ)                   估计 future return
Reward Model                   scalar on (prompt, response)
Verifier                       广义概念：判断正确性的函数/模型
```

</div>

### References

[^ppo]: Schulman et al. (2017). Proximal Policy Optimization Algorithms. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
[^gae]: Schulman et al. (2015). High-Dimensional Continuous Control Using Generalized Advantage Estimation. [arXiv:1506.02438](https://arxiv.org/abs/1506.02438)
[^grpo]: Shao et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
[^sr1]: Jin et al. (2025). Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning. [arXiv:2503.09516](https://arxiv.org/abs/2503.09516)
[^k3]: Schulman, J. (2020). Approximating KL Divergence. <http://joschu.net/blog/kl-approx.html>
[^r1]: Guo et al. (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
[^dpo]: Rafailov et al. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
[^instructgpt]: Ouyang et al. (2022). Training language models to follow instructions with human feedback. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
