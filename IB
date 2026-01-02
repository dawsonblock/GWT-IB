#!/usr/bin/env python3
"""
GWT-IB PRODUCTION-READY BUILD (FIX-ALL v3) — POPGYM CountRecallEasy-v0
=====================================================================

BUGS FIXED:
- Gate now actually gates memory (input-gated GRU with residual gating)
- Episode boundaries handled correctly (done_prev resets hidden state)
- Non-circular TD bootstrap (V(s_{t+1}) computed with td_prev=0)
- Stable TD normalization (EMA + quantile buffer, clamped std range)
- Correct recurrent PPO minibatching (BPTT sequences with stored h0)
- Correct rollout storage (float32 obs tensors, all required fields)
- GateCheck detects dead learning (td_std, correlations, eval trend)
- Production hardening (seeds, asserts, grad clip, bias clamp, obs flatten)

INVARIANTS PRESERVED:
- Router prior equation: prior = sigmoid(td_scale * |td_norm| + base_logit + bias)
- PPO remains PPO (clipped surrogate + value loss + entropy bonus, on-policy)
- IB loss remains per-sequence mean_c to target_c
- No curriculum or task changes
- Dependencies: popgym, gymnasium, torch, tqdm, matplotlib (optional)

Install:
    pip install popgym gymnasium torch tqdm matplotlib

Run:
    python IB_fixed.py
"""

from __future__ import annotations

import math
import time
import random
from dataclasses import dataclass
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import popgym  # noqa: F401
from tqdm import trange

try:
    import matplotlib.pyplot as plt
    PLOT_OK = True
except Exception:
    PLOT_OK = False


# ============================================================
# 0) CONFIG
# ============================================================
@dataclass
class Cfg:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Env
    env_id: str = "popgym-CountRecallEasy-v0"
    num_envs: int = 16
    seed: int = 42

    # Training
    total_timesteps: int = 500_000
    rollout_steps: int = 256
    seq_len: int = 64

    # PPO
    epochs: int = 4
    minibatches: int = 4
    lr: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5

    # Model
    enc_dim: int = 128
    hid_dim: int = 256
    router_hid: int = 64

    # Router (equation shape preserved)
    td_scale: float = 5.0
    base_logit: float = -1.0
    bias_init: float = -1.5
    bias_clamp_min: float = -4.0
    bias_clamp_max: float = 2.0

    # Router residual on top of prior
    use_router_residual: bool = True

    # Information bottleneck
    target_c: float = 0.35
    lambda_c: float = 0.05
    budget_warmup_updates: int = 25

    # TD stats / normalization
    td_ema: float = 0.99
    td_scale_ema2: float = 0.995
    td_quantile: float = 0.90
    td_scale_mode: str = "ema_quantile"
    td_center_mean: bool = True
    td_norm_clip: float = 5.0
    td_std_min: float = 1e-3
    td_std_max: float = 50.0
    use_percentile_buffer: bool = True
    td_buffer_size: int = 8192

    # Numeric safety
    td_clip: float = 20.0

    # Eval
    eval_every_updates: int = 10
    eval_episodes: int = 30

    # Optional ablation: force gate
    force_c: float | None = None


# ============================================================
# 1) UTILS
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_vec_env(cfg: Cfg):
    def thunk():
        env = gym.make(cfg.env_id)
        env = gym.wrappers.FlattenObservation(env)
        new_obs_space = gym.spaces.Box(
            low=env.observation_space.low,
            high=env.observation_space.high,
            shape=env.observation_space.shape,
            dtype=np.float32,
        )
        env = gym.wrappers.TransformObservation(
            env,
            lambda x: x.astype(np.float32),
            observation_space=new_obs_space
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return gym.vector.SyncVectorEnv([thunk for _ in range(cfg.num_envs)])


def obs_to_tensor(obs, device: torch.device) -> torch.Tensor:
    """Robust obs->tensor conversion handling dict/array obs."""
    if isinstance(obs, dict):
        parts = []
        for k in sorted(obs.keys()):
            v = obs[k]
            v = np.asarray(v, dtype=np.float32).ravel()
            parts.append(v)
        x = (np.concatenate(parts, axis=0) if parts
             else np.zeros((0,), dtype=np.float32))
    else:
        x = np.asarray(obs, dtype=np.float32).ravel()
    return torch.as_tensor(x, device=device, dtype=torch.float32)


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    var_y = torch.var(y_true)
    if var_y.item() < 1e-12:
        return float("nan")
    return float(1.0 - torch.var(y_true - y_pred) / (var_y + 1e-12))


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 50:
        return float("nan")
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    da = math.sqrt(float((a * a).mean()) + 1e-12)
    db = math.sqrt(float((b * b).mean()) + 1e-12)
    return float((a * b).mean() / (da * db + 1e-12))


# ============================================================
# 2) MODEL — GWT-IB (gate actually gates recurrence)
# ============================================================
class GWTIB(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, cfg: Cfg):
        super().__init__()
        self.cfg = cfg

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, cfg.enc_dim),
            nn.Tanh(),
            nn.Linear(cfg.enc_dim, cfg.enc_dim),
            nn.Tanh(),
        )

        # stepwise recurrence
        self.gru_cell = nn.GRUCell(cfg.enc_dim, cfg.hid_dim)

        # residual router MLP
        router_in = cfg.enc_dim + cfg.hid_dim
        self.router = nn.Sequential(
            nn.Linear(router_in, cfg.router_hid),
            nn.Tanh(),
            nn.Linear(cfg.router_hid, 1),
        )

        self.actor = nn.Sequential(
            nn.Linear(cfg.hid_dim, cfg.hid_dim),
            nn.Tanh(),
            nn.Linear(cfg.hid_dim, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(cfg.hid_dim, cfg.hid_dim),
            nn.Tanh(),
            nn.Linear(cfg.hid_dim, 1),
        )

        self.router_bias = nn.Parameter(
            torch.tensor([cfg.bias_init], dtype=torch.float32)
        )

        # TD running stats
        self.register_buffer("td_mean", torch.zeros(1))
        self.register_buffer("td_std", torch.ones(1))

        # percentile buffer
        self.register_buffer("td_buffer", torch.zeros(cfg.td_buffer_size))
        self.register_buffer("td_buf_idx", torch.zeros(1, dtype=torch.long))
        self.register_buffer(
            "td_buf_count", torch.zeros(1, dtype=torch.long)
        )

    @torch.no_grad()
    def update_td_stats(self, td: torch.Tensor):
        cfg = self.cfg
        ema = float(cfg.td_ema)
        td_flat = td.detach().view(-1)
        if td_flat.numel() == 0:
            return

        batch_mean = td_flat.mean()
        batch_rms = torch.sqrt(
            (td_flat * td_flat).mean().clamp(min=1e-12)
        )

        if cfg.td_center_mean:
            self.td_mean.mul_(ema).add_(batch_mean * (1 - ema))
        else:
            self.td_mean.mul_(ema)

        # write buffer BEFORE quantile
        if cfg.use_percentile_buffer:
            td_abs = td_flat.abs()
            n = int(td_abs.numel())
            buf_size = int(cfg.td_buffer_size)
            idx = int(self.td_buf_idx.item())
            end = idx + n
            if end <= buf_size:
                self.td_buffer[idx:end] = td_abs
            else:
                first = buf_size - idx
                self.td_buffer[idx:] = td_abs[:first]
                self.td_buffer[: (end % buf_size)] = td_abs[first:]
            self.td_buf_idx.fill_(end % buf_size)
            new_count = int(self.td_buf_count.item()) + n
            self.td_buf_count.fill_(
                buf_size if new_count >= buf_size else new_count
            )

        def qscale(q: float):
            if cfg.use_percentile_buffer:
                count = int(self.td_buf_count.item())
                if count > 64:
                    data = (self.td_buffer[:count] if count <
                            int(cfg.td_buffer_size) else self.td_buffer)
                    return torch.quantile(data, q)
            return torch.quantile(td_flat.abs(), q)

        if cfg.td_scale_mode == "ema":
            target = batch_rms
            self.td_std.mul_(ema).add_(target * (1 - ema))
        elif cfg.td_scale_mode == "ema_quantile":
            target = qscale(float(cfg.td_quantile))
            ema2 = float(cfg.td_scale_ema2)
            self.td_std.mul_(ema2).add_(target * (1 - ema2))
        elif cfg.td_scale_mode == "quantile":
            self.td_std.copy_(qscale(float(cfg.td_quantile)))
        else:
            target = batch_rms
            self.td_std.mul_(ema).add_(target * (1 - ema))

        self.td_std.clamp_(
            min=float(cfg.td_std_min), max=float(cfg.td_std_max)
        )

    def normalize_td(self, td: torch.Tensor) -> torch.Tensor:
        denom = self.td_std.clamp(min=self.cfg.td_std_min)
        td_norm = (td - self.td_mean) / denom
        return td_norm.clamp(-self.cfg.td_norm_clip, self.cfg.td_norm_clip)

    def forward(
        self,
        obs_seq: torch.Tensor,
        h0: torch.Tensor,
        td_prev_seq: torch.Tensor,
        done_prev_seq: torch.Tensor,
        force_c: float | None = None
    ):
        """
        FIX: stepwise recurrence with:
          - hidden reset on done_prev
          - gate controls GRU input + residual gating on hidden update
        """
        cfg = self.cfg
        T, B, _ = obs_seq.shape

        z = self.encoder(obs_seq)  # [T,B,enc]

        h = h0  # [B,H]
        logits_list = []
        values_list = []
        c_list = []
        prior_list = []
        td_abs_list = []

        bias = self.router_bias.clamp(
            cfg.bias_clamp_min, cfg.bias_clamp_max
        )

        for t in range(T):
            done_prev = done_prev_seq[t].squeeze(-1)  # [B]
            h = h * (1.0 - done_prev).unsqueeze(-1)

            td_norm = self.normalize_td(td_prev_seq[t])  # [B,1]
            td_abs = td_norm.abs()  # [B,1]
            td_abs_list.append(td_abs.squeeze(-1))

            # PRIOR: preserved form
            prior = torch.sigmoid(
                cfg.td_scale * td_abs + cfg.base_logit + bias
            )  # [B,1]
            prior_list.append(prior.squeeze(-1))

            if force_c is None and cfg.force_c is not None:
                force_c = float(cfg.force_c)

            if force_c is not None:
                c = torch.full_like(prior, float(force_c))
            else:
                if cfg.use_router_residual:
                    router_in = torch.cat([z[t], h], dim=-1)
                    resid = self.router(router_in)  # [B,1]
                    prior_logit = torch.logit(
                        prior.clamp(1e-5, 1 - 1e-5)
                    )
                    c = torch.sigmoid(prior_logit + resid)
                else:
                    c = prior

            c_list.append(c.squeeze(-1))

            # FIX: gate controls input + residual gating on update
            z_in = c * z[t]
            h_prop = self.gru_cell(z_in, h)
            h = h + c * (h_prop - h)

            logits_list.append(self.actor(h))
            values_list.append(self.critic(h).squeeze(-1))

        logits = torch.stack(logits_list, dim=0)  # [T,B,act]
        value = torch.stack(values_list, dim=0)  # [T,B]
        c = torch.stack(c_list, dim=0)  # [T,B]
        prior = torch.stack(prior_list, dim=0)  # [T,B]
        td_abs_norm = torch.stack(td_abs_list, dim=0)  # [T,B]

        return {
            "logits": logits,
            "value": value,
            "hT": h,
            "c": c,
            "prior": prior,
            "td_abs_norm": td_abs_norm,
            "bias": float(bias.item()),
        }


# ============================================================
# 3) PPO STORAGE
# ============================================================
class Rollout:
    def __init__(
        self, T: int, B: int, obs_dim: int, hid_dim: int, device: str
    ):
        self.T, self.B = T, B
        self.device = device

        self.obs = torch.zeros(T, B, obs_dim, device=device)
        self.actions = torch.zeros(T, B, dtype=torch.long, device=device)
        self.logp = torch.zeros(T, B, device=device)
        self.rewards = torch.zeros(T, B, device=device)
        self.dones = torch.zeros(T, B, device=device)
        self.values = torch.zeros(T, B, device=device)
        self.td_prev = torch.zeros(T, B, 1, device=device)
        self.done_prev = torch.zeros(T, B, 1, device=device)

        # store h BEFORE stepping (so PPO sequences get correct init)
        self.h0 = torch.zeros(T, B, hid_dim, device=device)

        # diagnostics
        self.c = torch.zeros(T, B, device=device)
        self.prior = torch.zeros(T, B, device=device)
        self.td_abs_norm = torch.zeros(T, B, device=device)

        self.advantages = torch.zeros(T, B, device=device)
        self.returns = torch.zeros(T, B, device=device)

    @torch.no_grad()
    def compute_gae(
        self, last_value: torch.Tensor, gamma: float, lam: float
    ):
        T = self.T
        adv = torch.zeros_like(last_value)
        for t in reversed(range(T)):
            next_nonterminal = 1.0 - self.dones[t]
            next_value = (
                last_value if t == T - 1 else self.values[t + 1]
            )
            delta = (
                self.rewards[t] + gamma * next_value * next_nonterminal
                - self.values[t]
            )
            adv = delta + gamma * lam * next_nonterminal * adv
            self.advantages[t] = adv
        self.returns = self.advantages + self.values


# ============================================================
# 4) EVAL
# ============================================================
@torch.no_grad()
def evaluate(cfg: Cfg, model: GWTIB) -> tuple[float, float]:
    env = gym.make(cfg.env_id)
    env = gym.wrappers.FlattenObservation(env)
    new_obs_space = gym.spaces.Box(
        low=env.observation_space.low,
        high=env.observation_space.high,
        shape=env.observation_space.shape,
        dtype=np.float32,
    )
    env = gym.wrappers.TransformObservation(
        env,
        lambda x: x.astype(np.float32),
        observation_space=new_obs_space
    )
    env = gym.wrappers.RecordEpisodeStatistics(env)

    model.eval()
    rets = []

    for ep in range(cfg.eval_episodes):
        obs, _ = env.reset(seed=cfg.seed + 10_000 + ep)
        done = False

        h = torch.zeros(1, cfg.hid_dim, device=cfg.device)
        td_prev = torch.zeros(1, 1, 1, device=cfg.device)
        done_prev = torch.zeros(1, 1, 1, device=cfg.device)

        ret = 0.0
        while not done:
            obs_t = obs_to_tensor(
                obs, torch.device(cfg.device)
            ).view(1, 1, -1)
            out = model(obs_t, h, td_prev, done_prev, force_c=cfg.force_c)
            logits = out["logits"][0, 0]
            a = torch.distributions.Categorical(
                logits=logits
            ).sample().item()

            obs2, r, term, trunc, _ = env.step(int(a))
            done = bool(term or trunc)
            ret += float(r)

            # clean TD: compute V(next) with td_prev=0
            obs2_t = obs_to_tensor(
                obs2, torch.device(cfg.device)
            ).view(1, 1, -1)
            out2 = model(
                obs2_t,
                out["hT"],
                torch.zeros_like(td_prev),
                torch.tensor(
                    [[[float(done)]]], device=cfg.device
                ),
                force_c=cfg.force_c,
            )
            v = out["value"][0, 0].item()
            v2 = out2["value"][0, 0].item()
            td_curr = float(
                r + cfg.gamma * (1.0 - float(done)) * v2 - v
            )
            td_curr = float(np.clip(td_curr, -cfg.td_clip, cfg.td_clip))

            obs = obs2
            h = out["hT"]
            td_prev = torch.tensor(
                [[[td_curr]]], device=cfg.device
            )
            done_prev = torch.tensor(
                [[[float(done)]]], device=cfg.device
            )

        rets.append(ret)

    model.train()
    return float(np.mean(rets)), float(np.std(rets))


# ============================================================
# 5) MAIN TRAIN
# ============================================================
def main(cfg: Cfg | None = None):
    if cfg is None:
        cfg = Cfg()

    assert cfg.rollout_steps % cfg.seq_len == 0, (
        "rollout_steps must be divisible by seq_len"
    )

    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    envs = make_vec_env(cfg)
    obs_dim = int(envs.single_observation_space.shape[0])
    act_dim = int(envs.single_action_space.n)

    model = GWTIB(obs_dim, act_dim, cfg).to(device)

    opt = optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)

    obs, _ = envs.reset(seed=cfg.seed)
    obs_t = torch.stack(
        [obs_to_tensor(obs[i], device) for i in range(cfg.num_envs)],
        dim=0
    )
    h = torch.zeros(cfg.num_envs, cfg.hid_dim, device=device)
    td_prev = torch.zeros(cfg.num_envs, 1, device=device)
    done_prev = torch.zeros(cfg.num_envs, 1, device=device)

    updates = cfg.total_timesteps // (cfg.num_envs * cfg.rollout_steps)

    # logging
    ep_ret_deque = deque(maxlen=100)
    gate_td_samples, gate_c_samples, gate_p_samples = [], [], []
    eval_updates, eval_means, eval_stds = [], [], []

    global_step = 0
    t0 = time.time()

    for update in trange(updates, desc="updates"):
        # lr anneal
        frac = 1.0 - (update / max(1, updates))
        for pg in opt.param_groups:
            pg["lr"] = cfg.lr * frac

        rollout = Rollout(
            cfg.rollout_steps, cfg.num_envs, obs_dim, cfg.hid_dim,
            cfg.device
        )

        # ---------- rollout ----------
        model.eval()
        for t in range(cfg.rollout_steps):
            global_step += cfg.num_envs

            rollout.obs[t] = obs_t
            rollout.td_prev[t] = td_prev
            rollout.done_prev[t] = done_prev
            rollout.h0[t] = h

            out = model(
                obs_t.view(1, cfg.num_envs, obs_dim),
                h,
                td_prev.view(1, cfg.num_envs, 1),
                done_prev.view(1, cfg.num_envs, 1),
                force_c=cfg.force_c,
            )

            logits = out["logits"][0]
            values = out["value"][0]
            h_next = out["hT"]

            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()
            logp = dist.log_prob(actions)

            obs2, r, term, trunc, infos = envs.step(
                actions.cpu().numpy()
            )
            done = np.logical_or(term, trunc)

            obs2_t = torch.stack(
                [obs_to_tensor(obs2[i], device)
                 for i in range(cfg.num_envs)],
                dim=0
            )
            r_t = torch.tensor(r, device=device, dtype=torch.float32)
            done_t = torch.tensor(
                done.astype(np.float32), device=device
            )

            # --- clean TD bootstrap (td_prev=0) ---
            with torch.no_grad():
                out2 = model(
                    obs2_t.view(1, cfg.num_envs, obs_dim),
                    h_next,
                    torch.zeros(1, cfg.num_envs, 1, device=device),
                    done_t.view(1, cfg.num_envs, 1),
                    force_c=cfg.force_c,
                )
                next_v = out2["value"][0]

            td_curr = (
                r_t + cfg.gamma * next_v * (1.0 - done_t) - values
            )
            td_curr = td_curr.clamp(
                -cfg.td_clip, cfg.td_clip
            ).view(cfg.num_envs, 1)

            model.update_td_stats(td_curr)

            rollout.actions[t] = actions
            rollout.logp[t] = logp.detach()
            rollout.rewards[t] = r_t
            rollout.dones[t] = done_t
            rollout.values[t] = values.detach()

            rollout.c[t] = out["c"][0].detach()
            rollout.prior[t] = out["prior"][0].detach()
            rollout.td_abs_norm[t] = out["td_abs_norm"][0].detach()

            # episode return logging
            if "final_info" in infos and infos["final_info"] is not None:
                for fi in infos["final_info"]:
                    if fi and "episode" in fi:
                        ep_ret_deque.append(float(fi["episode"]["r"]))

            obs_t = obs2_t
            h = h_next.detach()
            td_prev = td_curr
            done_prev = done_t.view(cfg.num_envs, 1)

        # bootstrap value for GAE (clean)
        model.eval()
        with torch.no_grad():
            out_last = model(
                obs_t.view(1, cfg.num_envs, obs_dim),
                h,
                torch.zeros(1, cfg.num_envs, 1, device=device),
                done_prev.view(1, cfg.num_envs, 1),
                force_c=cfg.force_c,
            )
            last_value = out_last["value"][0]
        rollout.compute_gae(last_value, cfg.gamma, cfg.gae_lambda)

        # sample gatecheck points
        with torch.no_grad():
            td_abs = rollout.td_abs_norm.flatten().cpu().numpy()
            c_all = rollout.c.flatten().cpu().numpy()
            p_all = rollout.prior.flatten().cpu().numpy()
            n = min(4096, td_abs.shape[0])
            idx = np.random.choice(td_abs.shape[0], size=n, replace=False)
            gate_td_samples.append(td_abs[idx])
            gate_c_samples.append(c_all[idx])
            gate_p_samples.append(p_all[idx])

        # ---------- PPO update (recurrent sequences with correct h0) -----
        model.train()

        T, B = cfg.rollout_steps, cfg.num_envs
        L = cfg.seq_len
        n_seq = T // L
        total_seqs = n_seq * B

        # shape into [S,L,...] with S = total_seqs
        def to_seqs(x, extra_dim=False):
            if extra_dim:
                x = x.view(n_seq, L, B, -1).permute(
                    0, 2, 1, 3
                ).reshape(total_seqs, L, -1)
            else:
                x = x.view(n_seq, L, B).permute(
                    0, 2, 1
                ).reshape(total_seqs, L)
            return x

        obs_seq = to_seqs(rollout.obs, extra_dim=True)
        act_seq = to_seqs(rollout.actions, extra_dim=False)
        logp_old_seq = to_seqs(rollout.logp, extra_dim=False)
        adv_seq = to_seqs(rollout.advantages, extra_dim=False)
        ret_seq = to_seqs(rollout.returns, extra_dim=False)
        val_old_seq = to_seqs(rollout.values, extra_dim=False)
        td_prev_seq = to_seqs(rollout.td_prev, extra_dim=True)
        done_prev_seq = to_seqs(rollout.done_prev, extra_dim=True)

        # IMPORTANT: correct h0 for each sequence
        h0_all = rollout.h0.view(
            n_seq, L, B, cfg.hid_dim
        ).permute(0, 2, 1, 3)[:, :, 0, :].reshape(total_seqs, cfg.hid_dim)

        adv_seq = (adv_seq - adv_seq.mean()) / (adv_seq.std() + 1e-8)

        warm = min(1.0, (update + 1) / max(1, cfg.budget_warmup_updates))
        lambda_c = cfg.lambda_c * warm

        batch_size = total_seqs
        mb_size = max(1, batch_size // cfg.minibatches)

        for epoch in range(cfg.epochs):
            perm = torch.randperm(batch_size, device=device)
            for i in range(cfg.minibatches):
                mb_idx = perm[i * mb_size: (i + 1) * mb_size]
                if mb_idx.numel() == 0:
                    continue

                # [S,L,...] -> model expects [L,S,...]
                obs_mb = obs_seq[mb_idx].transpose(0, 1).detach()
                act_mb = act_seq[mb_idx].transpose(0, 1).detach()
                logp_old_mb = logp_old_seq[mb_idx].transpose(0, 1).detach()
                adv_mb = adv_seq[mb_idx].transpose(0, 1).detach()
                ret_mb = ret_seq[mb_idx].transpose(0, 1).detach()
                val_old_mb = val_old_seq[mb_idx].transpose(0, 1).detach()
                td_prev_mb = td_prev_seq[mb_idx].transpose(0, 1).detach()
                done_prev_mb = done_prev_seq[mb_idx].transpose(
                    0, 1
                ).detach()
                h0_mb = h0_all[mb_idx].detach()

                out = model(
                    obs_mb, h0_mb, td_prev_mb, done_prev_mb,
                    force_c=cfg.force_c
                )
                logits = out["logits"]
                values = out["value"]
                c = out["c"]
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(act_mb)
                entropy = dist.entropy().mean()

                ratio = torch.exp(logp - logp_old_mb)
                pg1 = -adv_mb * ratio
                pg2 = -adv_mb * torch.clamp(
                    ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps
                )
                pg_loss = torch.max(pg1, pg2).mean()

                v_clipped = val_old_mb + torch.clamp(
                    values - val_old_mb, -cfg.clip_eps, cfg.clip_eps
                )
                vf1 = (values - ret_mb) ** 2
                vf2 = (v_clipped - ret_mb) ** 2
                vf_loss = 0.5 * torch.max(vf1, vf2).mean()

                mean_c_per_seq = c.mean(dim=0)  # [MB]
                ib_loss = ((mean_c_per_seq - cfg.target_c) ** 2).mean()

                loss = (
                    pg_loss + cfg.vf_coef * vf_loss
                    - cfg.ent_coef * entropy + lambda_c * ib_loss
                )

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.max_grad_norm
                )
                opt.step()

                with torch.no_grad():
                    model.router_bias.clamp_(
                        cfg.bias_clamp_min, cfg.bias_clamp_max
                    )

        # ---------- logging ----------
        with torch.no_grad():
            mean_c = float(rollout.c.mean().item())
            td_std = float(model.td_std.item())
            td_mean = float(model.td_mean.item())
            ev = explained_variance(
                rollout.values.flatten(), rollout.returns.flatten()
            )
            ep_mean = (
                float(np.mean(ep_ret_deque)) if len(ep_ret_deque)
                else float("nan")
            )

        do_eval = (
            ((update + 1) % cfg.eval_every_updates == 0) or (update == 0)
        )
        eval_m = eval_s = float("nan")
        if do_eval:
            eval_m, eval_s = evaluate(cfg, model)
            eval_updates.append(update + 1)
            eval_means.append(eval_m)
            eval_stds.append(eval_s)

        if (update + 1) % 10 == 0:
            elapsed = time.time() - t0
            sps = global_step / max(1e-6, elapsed)
            print(
                f"upd {update+1:4d}/{updates} | "
                f"td_mean {td_mean:+.3f} td_std {td_std:.4f} | "
                f"mean_c {mean_c:.3f} (tgt {cfg.target_c:.2f}) | "
                f"EV {ev:.3f} | ep100 {ep_mean:.2f} | "
                f"bias {model.router_bias.item():+.3f} | "
                f"{sps:,.0f} steps/s | eval {eval_m:.2f}"
            )

    # ---------- GateCheck ----------
    td_abs = (
        np.concatenate(gate_td_samples, axis=0) if gate_td_samples
        else np.array([])
    )
    c_all = (
        np.concatenate(gate_c_samples, axis=0) if gate_c_samples
        else np.array([])
    )
    p_all = (
        np.concatenate(gate_p_samples, axis=0) if gate_p_samples
        else np.array([])
    )
    d_all = c_all - p_all

    rho = safe_corr(td_abs, c_all) if td_abs.size else float("nan")
    rho_d = safe_corr(td_abs, d_all) if td_abs.size else float("nan")
    q = np.quantile(td_abs, 0.8) if td_abs.size else float("nan")
    mask = td_abs >= q if td_abs.size else np.array([], dtype=bool)
    rho_tail = (
        safe_corr(td_abs[mask], c_all[mask])
        if (td_abs.size and mask.any()) else float("nan")
    )
    rho_d_tail = (
        safe_corr(td_abs[mask], d_all[mask])
        if (td_abs.size and mask.any()) else float("nan")
    )

    td_std = float(model.td_std.item())
    std_ok = td_std > (cfg.td_std_min * 5.0)

    # eval sanity: trend check
    if len(eval_means) >= 3:
        k = max(2, len(eval_means) // 3)
        early = float(np.mean(eval_means[:k]))
        late = float(np.mean(eval_means[-k:]))
        eval_ok = late > early + 0.1
    elif len(eval_means) == 2:
        eval_ok = eval_means[1] > eval_means[0] + 0.1
    else:
        eval_ok = True

    gate_status = "PASS" if (std_ok and eval_ok) else "FAIL"

    print("\nGATECHECK")
    print(
        f"rho(td,c)={rho:.3f}  rho_tail={rho_tail:.3f} | "
        f"rho(td,Δc)={rho_d:.3f} rho_tailΔ={rho_d_tail:.3f}"
    )
    print(
        f"td_std={td_std:.6f} (std_ok={std_ok}) | "
        f"eval_ok={eval_ok} -> {gate_status}"
    )
    with torch.no_grad():
        mean_c = float(rollout.c.mean().item())
        ev = explained_variance(
            rollout.values.flatten(), rollout.returns.flatten()
        )
        ep_mean = (
            float(np.mean(ep_ret_deque)) if len(ep_ret_deque)
            else float("nan")
        )
    print(f"mean_c={mean_c:.4f}  EV={ev:.4f}  ep100={ep_mean:.2f}")

    # ---------- plots ----------
    if PLOT_OK and td_abs.size:
        # binned curves
        xmax = float(np.quantile(td_abs, 0.99))
        bins = np.linspace(0.0, max(1e-6, xmax), 25)
        inds = np.clip(np.digitize(td_abs, bins) - 1, 0, len(bins) - 2)

        bx, bc, bp, bd = [], [], [], []
        for k in range(len(bins) - 1):
            m = inds == k
            if m.sum() < 80:
                continue
            bx.append(0.5 * (bins[k] + bins[k + 1]))
            bc.append(float(c_all[m].mean()))
            bp.append(float(p_all[m].mean()))
            bd.append(float(d_all[m].mean()))

        plt.figure()
        plt.plot(bx, bc, label="c (binned mean)")
        plt.plot(bx, bp, label="prior (binned mean)")
        plt.plot(bx, bd, label="Δc (binned mean)")
        plt.xlabel("|td_norm|")
        plt.ylabel("gate")
        plt.title(
            f"TD→Gate | rho_tail={rho_tail:.3f}, td_std={td_std:.4f}"
        )
        plt.legend()
        plt.tight_layout()

        if len(eval_updates) > 0:
            plt.figure()
            plt.plot(eval_updates, eval_means)
            plt.fill_between(
                eval_updates,
                np.array(eval_means) - np.array(eval_stds),
                np.array(eval_means) + np.array(eval_stds),
                alpha=0.2,
            )
            plt.xlabel("update")
            plt.ylabel("eval return")
            plt.title("Evaluation")
            plt.tight_layout()

        plt.show()


if __name__ == "__main__":
    main()
