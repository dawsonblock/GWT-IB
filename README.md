# GWT-IB: Gated Working Memory with Information Bottleneck

A production-ready implementation of **GWT-IB (Gated Working Memory with Information Bottleneck)** using **PPO (Proximal Policy Optimization)** for reinforcement learning on memory-intensive tasks.

This implementation fixes all training-breaking bugs while preserving critical architectural invariants, enabling stable learning on challenging memory benchmarks like POPGym's CountRecallEasy-v0.

---

## 🎯 Overview

**GWT-IB** combines:
- **Gated Working Memory**: A recurrent GRU-based architecture where a learned gate controls what information enters memory
- **Information Bottleneck**: Regularization that encourages the agent to compress information efficiently
- **TD-Error Gating**: The gate is driven by temporal difference (TD) errors, allowing the agent to selectively remember when prediction errors are high
- **Recurrent PPO**: On-policy reinforcement learning with proper backpropagation through time (BPTT)

### Key Innovation

The router learns to gate memory updates based on normalized TD errors:

```
prior = sigmoid(td_scale * |td_norm| + base_logit + bias)
```

This creates an adaptive memory system that:
- **Remembers** when prediction errors are high (important events)
- **Forgets** when prediction errors are low (redundant information)
- **Compresses** information to a target budget via IB loss

---

## 🐛 Bugs Fixed (v3)

This implementation addresses critical issues that caused "looks correlated but dead learning" failures:

### 1. **Gate Actually Gates Memory**
- **Problem**: Original implementation ran full GRU update before gating, so gate didn't control what entered memory
- **Fix**: Input-gated GRU with residual gating: `h = h + c * (h_prop - h)`
- **Impact**: Gate now truly freezes hidden state when `c → 0`

### 2. **Episode Boundary Handling**
- **Problem**: Hidden states weren't reset at episode boundaries
- **Fix**: `done_prev` properly resets hidden state in rollout and training
- **Impact**: Prevents information leakage across episodes

### 3. **Non-Circular TD Bootstrap**
- **Problem**: V(s_{t+1}) was computed with the same TD that defined the gate, creating circular dependencies
- **Fix**: Compute V(s_{t+1}) with `td_prev=0` for clean bootstrap
- **Impact**: Eliminates circular gating logic

### 4. **Stable TD Normalization**
- **Problem**: TD stats could pin at floor/ceiling, breaking learning
- **Fix**: EMA + quantile buffer with proper clamping `[td_std_min, td_std_max]`
- **Impact**: Stable normalization throughout training

### 5. **Correct Recurrent PPO Minibatching**
- **Problem**: Training used `h0=zeros` for all sequences, breaking credit assignment
- **Fix**: Store and carry `h0` from rollout for each BPTT sequence
- **Impact**: Proper temporal credit assignment

### 6. **Rollout Storage Correctness**
- **Problem**: Observations not properly converted to float32 tensors
- **Fix**: `obs_to_tensor()` utility handles dict/array obs robustly
- **Impact**: Works with diverse POPGym observation spaces

### 7. **GateCheck Diagnostics**
- **Problem**: No detection of "correlated but dead" failure mode
- **Fix**: Reports td_std, mean_c, EV, correlations; fails if td_std pinned or eval non-improving
- **Impact**: Catches subtle training failures

### 8. **Production Hardening**
- **Fix**: Seeds, assertions, grad clipping, bias clamping, robust obs handling
- **Impact**: Reproducible, stable training

---

## 🔒 Invariants Preserved

These architectural choices are **intentionally preserved** and should not be changed:

1. **Router Prior Equation Form**:
   ```python
   prior = sigmoid(td_scale * |td_norm| + base_logit + bias)
   ```

2. **PPO Algorithm**: Clipped surrogate objective + value loss + entropy bonus (on-policy)

3. **IB Loss Definition**: Per-sequence `mean_c → target_c`

4. **No Curriculum**: Task difficulty remains constant

5. **Minimal Dependencies**: Only `popgym`, `gymnasium`, `torch`, `tqdm`, `matplotlib`

---

## 📦 Installation

### Requirements
- Python 3.8+
- PyTorch 1.10+
- CUDA (optional, for GPU acceleration)

### Install Dependencies

```bash
pip install popgym gymnasium torch tqdm matplotlib
```

Or install from requirements:

```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Basic Training

```bash
python IB
```

This will:
- Train on POPGym's `CountRecallEasy-v0` environment
- Run for 500,000 timesteps (122 updates)
- Evaluate every 10 updates
- Display progress with td_std, mean_c, EV, and eval metrics
- Show final GateCheck diagnostics and plots

### Expected Output

```
upd   10/122 | td_mean +0.018 td_std 0.0474 | mean_c 0.429 (tgt 0.35) | EV 0.481 | ep100 nan | bias -1.512 | 1,967 steps/s | eval -0.94
upd   20/122 | td_mean -0.002 td_std 0.0149 | mean_c 0.339 (tgt 0.35) | EV 0.690 | ep100 nan | bias -1.510 | 2,017 steps/s | eval -0.91
...
```

### Final GateCheck

After training completes, you'll see:

```
GATECHECK
rho(td,c)=0.XXX  rho_tail=0.XXX | rho(td,Δc)=0.XXX rho_tailΔ=0.XXX
td_std=0.XXXXXX (std_ok=True) | eval_ok=True -> PASS
mean_c=0.XXXX  EV=0.XXXX  ep100=XX.XX
```

Plus plots showing:
- TD → Gate relationship (binned curves)
- Evaluation return over time

---

## ⚙️ Configuration

Edit the `Cfg` dataclass in the script to customize training:

### Environment
```python
env_id: str = "popgym-CountRecallEasy-v0"  # POPGym environment
num_envs: int = 16                          # Parallel environments
seed: int = 42                              # Random seed
```

### Training
```python
total_timesteps: int = 500_000  # Total training steps
rollout_steps: int = 256        # Steps per rollout
seq_len: int = 64               # BPTT sequence length
```

### PPO Hyperparameters
```python
epochs: int = 4                 # PPO epochs per update
minibatches: int = 4            # Minibatches per epoch
lr: float = 2.5e-4              # Learning rate
gamma: float = 0.99             # Discount factor
gae_lambda: float = 0.95        # GAE lambda
clip_eps: float = 0.2           # PPO clip epsilon
```

### Router & Gating
```python
td_scale: float = 5.0           # TD scaling in prior equation
base_logit: float = -1.0        # Base logit in prior equation
bias_init: float = -1.5         # Initial learnable bias
bias_clamp_min: float = -4.0    # Min bias value
bias_clamp_max: float = 2.0     # Max bias value
```

### Information Bottleneck
```python
target_c: float = 0.35          # Target gate probability
lambda_c: float = 0.05          # IB loss weight
budget_warmup_updates: int = 25 # Warmup period for lambda_c
```

### TD Normalization
```python
td_ema: float = 0.99            # EMA for td_mean
td_scale_ema2: float = 0.995    # EMA for td_std
td_quantile: float = 0.90       # Quantile for scaling
td_std_min: float = 1e-3        # Min td_std (prevents pinning)
td_std_max: float = 50.0        # Max td_std (prevents explosion)
```

---

## 📊 Monitoring Training

### Key Metrics

| Metric | Description | Healthy Range |
|--------|-------------|---------------|
| `td_std` | TD error standard deviation | 0.01 - 10.0 (not pinned) |
| `mean_c` | Average gate probability | Near `target_c` (0.35) |
| `EV` | Explained variance | 0.5 - 0.95 (improving) |
| `ep100` | Rolling 100-episode return | Task-dependent |
| `bias` | Learnable router bias | -4.0 to 2.0 |

### GateCheck Pass Criteria

Training is considered **successful** if:
1. `td_std > td_std_min * 5.0` (not pinned at floor)
2. Evaluation returns improve over time (early vs late comparison)

### Correlation Metrics

- `rho(td,c)`: Correlation between |td_norm| and gate probability `c`
- `rho_tail`: Correlation in top 20% of TD errors (should be positive)
- `rho(td,Δc)`: Correlation with residual gate component

---

## 🏗️ Architecture Details

### Model Components

```
Encoder (obs → z)
    ↓
Router (z, h, td_prev → c)  ← TD-driven gate
    ↓
GRU Cell (gated input)
    ↓
Actor (h → policy) & Critic (h → value)
```

### Forward Pass

```python
# 1. Encode observation
z = encoder(obs)

# 2. Compute gate from TD error
td_norm = normalize_td(td_prev)
prior = sigmoid(td_scale * |td_norm| + base_logit + bias)
c = sigmoid(logit(prior) + router_mlp(z, h))

# 3. Gated GRU update
z_in = c * z
h_prop = GRU(z_in, h)
h = h + c * (h_prop - h)  # Residual gating

# 4. Policy and value
policy = actor(h)
value = critic(h)
```

### Loss Function

```python
loss = (
    pg_loss                    # PPO clipped surrogate
    + vf_coef * vf_loss       # Value function loss
    - ent_coef * entropy      # Entropy bonus
    + lambda_c * ib_loss      # Information bottleneck
)

# IB loss: per-sequence mean_c → target_c
ib_loss = ((mean_c_per_seq - target_c) ** 2).mean()
```

---

## 🧪 Testing Other Environments

### POPGym Environments

```python
# Easy memory tasks
env_id = "popgym-CountRecallEasy-v0"
env_id = "popgym-RepeatFirstEasy-v0"

# Medium difficulty
env_id = "popgym-CountRecallMedium-v0"
env_id = "popgym-RepeatPreviousMedium-v0"

# Hard memory tasks
env_id = "popgym-CountRecallHard-v0"
env_id = "popgym-HigherLowerHard-v0"
```

### Custom Environments

The code works with any Gymnasium-compatible environment. For dict observations:

```python
# obs_to_tensor() automatically handles:
# - Dict observations (flattens and concatenates)
# - Array observations (flattens)
# - Ensures float32 dtype
```

---

## 📈 Performance Benchmarks

### CountRecallEasy-v0 (500k steps)

| Metric | Value |
|--------|-------|
| Final EV | ~0.69 |
| Mean Gate (c) | ~0.34 (target: 0.35) |
| TD Std | ~0.015 (stable) |
| Training Speed | ~2,000 steps/s (CPU) |

---

## 🔬 Research Background

### Key Papers

1. **Information Bottleneck**: Tishby et al. (1999)
2. **PPO**: Schulman et al. (2017)
3. **POPGym Benchmark**: Morad et al. (2023)

### Design Rationale

**Why TD-driven gating?**
- TD errors indicate when the agent's predictions are wrong
- High TD → important event → remember
- Low TD → predictable → forget

**Why Information Bottleneck?**
- Prevents the agent from memorizing everything
- Forces efficient compression of task-relevant information
- Regularizes the gate to a target budget

**Why Recurrent PPO?**
- On-policy learning is more stable than off-policy for memory tasks
- BPTT with proper h0 enables temporal credit assignment
- PPO's clipped objective prevents destructive updates

---

## 🛠️ Troubleshooting

### Training Not Improving

**Check GateCheck output:**
- If `td_std` is pinned near `td_std_min`: Increase `td_std_min` or adjust normalization
- If `mean_c` far from `target_c`: Adjust `lambda_c` or `bias_init`
- If `EV` not improving: Check learning rate, increase `rollout_steps`

### Memory Issues

**Reduce memory usage:**
```python
num_envs = 8           # Reduce parallel envs
rollout_steps = 128    # Reduce rollout length
seq_len = 32           # Reduce BPTT length
```

### Slow Training

**Speed up training:**
```python
device = "cuda"        # Use GPU if available
num_envs = 32          # Increase parallelism
eval_episodes = 10     # Reduce eval episodes
```

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{gwtib2026,
  title={GWT-IB: Gated Working Memory with Information Bottleneck},
  author={Block, Dawson},
  year={2026},
  url={https://github.com/dawsonblock/GWT-IB}
}
```

---

## 📄 License

MIT License - see LICENSE file for details

---

## 🤝 Contributing

Contributions welcome! Please:
1. Preserve the hard invariants (router equation, PPO, IB loss)
2. Add tests for new features
3. Update this README with any configuration changes

---

## 🙏 Acknowledgments

- **POPGym** team for the memory benchmark suite
- **OpenAI** for PPO algorithm
- **Tishby et al.** for Information Bottleneck theory

---

## 📧 Contact

For questions or issues, please open a GitHub issue or contact:
- GitHub: [@dawsonblock](https://github.com/dawsonblock)

---

**Happy Training! 🚀**
