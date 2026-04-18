---
layout: post
title: "From PPO to Search-R1: The Design Space of Reasoning and Agentic RL"
date: 2026-04-17
description: "A component-by-component decomposition of PPO, walking through GRPO, verifiers, and retrieved-token masking to a complete instance of Agentic RL"
tags: reinforcement-learning llm reasoning agentic-rl ppo grpo
categories: technical-notes
lang: en
alt: /blog/2026/from-ppo-to-search-r1-v2/
giscus_comments: false
related_posts: false
toc:
  sidebar: left
---

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

### References

[^ppo]: Schulman et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347
[^gae]: Schulman et al. (2015). High-Dimensional Continuous Control Using Generalized Advantage Estimation. arXiv:1506.02438
[^grpo]: Shao et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300
[^sr1]: Jin et al. (2025). Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning. arXiv:2503.09516
[^k3]: Schulman, J. (2020). Approximating KL Divergence. http://joschu.net/blog/kl-approx.html
[^r1]: Guo et al. (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv:2501.12948
[^dpo]: Rafailov et al. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model.
[^instructgpt]: Ouyang et al. (2022). Training language models to follow instructions with human feedback.
