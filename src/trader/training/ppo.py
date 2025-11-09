from __future__ import annotations
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F  # optional; keep if you log extra metrics
from torch.amp import autocast, GradScaler

from trader.utils.logging import get_logger


def _sum_log_one_minus_a2(a: torch.Tensor) -> torch.Tensor:
    """
    Sum over action dims: log(1 - a^2 + eps).
    For a = tanh(z): log p(a) = log p(z) - sum log(1 - a^2).
    """
    # Slightly larger eps for bf16 safety
    eps = 1e-5
    return torch.log(1 - a.pow(2) + eps).sum(dim=-1)


def _flat_params(*modules):
    ps = []
    for m in modules:
        ps += [p for p in m.parameters() if p.requires_grad]
    return ps


class PPOTrainer:
    """
    PPO trainer with:
      - recurrent feature & policy nets (provided by caller)
      - GAE advantages (masked across episode boundaries)
      - minibatch SGD with value clipping
      - KL target with adaptive penalty and per-minibatch early stop
      - AMP (+ fp32 islands for fragile logprob math) + grad clipping
    """

    def __init__(
        self,
        env,
        feature,
        policy,
        cfg: Dict[str, Any],
        callbacks=None,
        device: torch.device | None = None,
        amp_dtype: str = "bfloat16",
    ):
        self.env = env
        self.feature = feature
        self.policy = policy
        self.cfg = cfg
        self.callbacks = callbacks

        self.device = device or torch.device("cpu")
        self.log = get_logger("ppo")

        # Optimizer
        self.opt = optim.Adam(
            _flat_params(feature, policy),
            lr=cfg.get("lr", 3e-4),
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # PPO knobs
        self.gamma = float(cfg.get("gamma", 0.99))
        self.lam = float(cfg.get("gae_lambda", 0.95))
        self.clip_eps = float(cfg.get("clip_eps", 0.2))
        self.value_coef = float(cfg.get("value_coef", 0.5))
        self.entropy_coef = float(cfg.get("entropy_coef", 0.01))

        self.batch_steps = int(cfg.get("batch_steps", 2048))
        self.epochs = int(cfg.get("epochs", 4))
        self.minibatch_size = int(cfg.get("minibatch_size", 256))
        self.value_clip = float(cfg.get("value_clip", 0.2))
        self.target_kl = float(cfg.get("target_kl", 0.03))
        self.entropy_target = cfg.get("entropy_target", None)

        # Stabilizers
        self.max_grad_norm = float(cfg.get("max_grad_norm", 0.5))
        self.kl_adapt = bool(cfg.get("kl_adapt", True))
        self.kl_beta = float(cfg.get("kl_beta", 1.0))
        self.kl_beta_inc = float(cfg.get("kl_beta_inc", 1.5))
        self.kl_beta_dec = float(cfg.get("kl_beta_dec", 0.75))
        self.kl_stop_multiplier = float(cfg.get("kl_stop_multiplier", 1.5))

        # AMP dtype selection
        if amp_dtype == "auto":
            self.amp_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        elif amp_dtype == "bfloat16":
            self.amp_dtype = torch.bfloat16
        elif amp_dtype == "float16":
            self.amp_dtype = torch.float16
        else:
            self.amp_dtype = torch.float32

        # NOTE: GradScaler for fp16 only. bf16 does not use it.
        self.scaler = GradScaler(
            "cuda",
            enabled=(self.device.type == "cuda" and self.amp_dtype == torch.float16),
        )

    @staticmethod
    def _torch_save(obj, path):
        torch.save(obj, path)

    # ---------------- rollout ----------------
    def _rollout(self, steps: int) -> Dict[str, torch.Tensor]:
        """
        Collect a flat trajectory of length `steps`.
        Recurrent states are reset across episode boundaries.
        """
        obs, _ = self.env.reset()
        traj = {k: [] for k in ["obs", "act", "rew", "val", "logp", "done"]}
        h_feat = None
        h_pol = None

        for _ in range(steps):
            x = torch.tensor(obs, dtype=torch.float32, device=self.device).view(1, 1, -1)

            # Run networks under AMP...
            with autocast("cuda", enabled=(self.device.type == "cuda"), dtype=self.amp_dtype):
                z, h_feat = self.feature(x, h_feat)
                mu, std, v, h_pol = self.policy(z, h_pol)

            # ...but compute distribution/logp/Jacobian in fp32 for stability
            mu32 = mu.float()
            std32 = std.float()
            base = torch.distributions.Normal(mu32, std32)
            z_samp = base.rsample()
            a = torch.tanh(z_samp)  # action in [-1, 1]

            base_logp = base.log_prob(z_samp).sum(dim=-1)
            # correct sign: subtract tanh-Jacobian term
            logp = base_logp - _sum_log_one_minus_a2(a.float())

            nxt, r, done, _, _ = self.env.step(a.detach().view(-1).cpu().numpy())

            traj["obs"].append(obs)
            traj["act"].append(a.detach().view(-1).cpu().numpy())
            traj["rew"].append(float(r))
            traj["val"].append(float(v.float().item()))
            traj["logp"].append(float(logp.float().item()))
            traj["done"].append(bool(done))

            obs = nxt
            if done:
                obs, _ = self.env.reset()
                h_feat = None
                h_pol = None

        for k in traj:
            traj[k] = torch.tensor(np.array(traj[k]), dtype=torch.float32, device=self.device)
        return traj

    # ---------------- gae ----------------
    def _gae(self, rew: torch.Tensor, val: torch.Tensor, done: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        GAE with episode masking. 'done' is 1.0 at terminal, 0.0 otherwise.
        """
        with torch.no_grad():
            not_done = 1.0 - done
            v_next = torch.cat([val[1:], val[-1:]]) * not_done + 0.0 * done
            deltas = rew + self.gamma * v_next - val
            adv = torch.zeros_like(rew)
            gae = 0.0
            for t in reversed(range(len(rew))):
                gae = deltas[t] + self.gamma * self.lam * gae * not_done[t]
                adv[t] = gae
            ret = adv + val
        # sanitize + normalize
        adv = torch.nan_to_num(adv, nan=0.0, posinf=0.0, neginf=0.0)
        ret = torch.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0)
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        return adv, ret

    # ---------------- forward over batch ----------------
    def _forward_batch(self, obs_tensor: torch.Tensor, act: torch.Tensor):
        """
        Returns a Normal distribution (mu,std) constructed in fp32, and v_pred.
        """
        with autocast("cuda", enabled=(self.device.type == "cuda"), dtype=self.amp_dtype):
            z, _ = self.feature(obs_tensor)                 # (T,1,E)
            mu, std, v_pred, _ = self.policy(z)            # (T,A),(T,A),(T,1)

        # Build distribution in fp32 for numerics
        base = torch.distributions.Normal(mu.float().view_as(act), std.float().view_as(act))
        return base, v_pred.squeeze(-1)

    # ---------------- utils ----------------
    def _compute_approx_kl(self, logp_old: torch.Tensor, logp_new: torch.Tensor) -> torch.Tensor:
        return (logp_old - logp_new).mean()

    # ---------------- ppo update ----------------
    def _update(self, traj: Dict[str, torch.Tensor]) -> Dict[str, float]:
        rew, val, logp_old, act, done = (
            traj["rew"],
            traj["val"],
            traj["logp"],
            traj["act"],
            traj["done"],
        )
        adv, ret = self._gae(rew, val, done)

        # (T,1,D) for recurrence
        obs_tensor = traj["obs"].float().view(-1, 1, traj["obs"].shape[-1])
        T = obs_tensor.shape[0]
        idx = torch.arange(T, device=self.device)

        last_logs: Dict[str, float] = {}

        for ep in range(self.epochs):
            perm = idx[torch.randperm(T)]
            running_kl, running_clipfrac = [], []
            kl_tripped = False

            for start in range(0, T, self.minibatch_size):
                mb = perm[start : start + self.minibatch_size]

                base, v_pred = self._forward_batch(obs_tensor[mb], act[mb])

                # robust atanh reconstruction of pre-tanh action (fp32)
                a = act[mb].clamp(-0.999999, 0.999999).float()
                z_recon = torch.atanh(a)

                # compute logp in fp32 (outside autocast for stability)
                base_logp = base.log_prob(z_recon).sum(dim=-1).float()
                logp = base_logp - _sum_log_one_minus_a2(a)

                # If old log-probs are non-finite in this minibatch, skip it safely
                lpo_raw = logp_old[mb]
                if not torch.isfinite(lpo_raw).all():
                    self.log.warning("Skipping minibatch due to non-finite old log-probs.")
                    continue
                lpo = lpo_raw.float()

                # safe ratio in log space
                log_ratio = torch.clamp(logp - lpo, min=-20.0, max=20.0)
                ratio = torch.exp(log_ratio)

                surr1 = ratio * adv[mb]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv[mb]
                policy_loss = -torch.min(surr1, surr2).mean()

                approx_kl = self._compute_approx_kl(lpo, logp)
                if self.kl_adapt:
                    policy_loss = policy_loss + self.kl_beta * approx_kl

                # value loss with clip
                v_pred = torch.nan_to_num(v_pred, nan=0.0, posinf=0.0, neginf=0.0)
                v_clip = val[mb] + torch.clamp(v_pred - val[mb], -self.value_clip, self.value_clip)
                value_loss = 0.5 * torch.max(
                    (ret[mb] - v_pred).pow(2),
                    (ret[mb] - v_clip).pow(2),
                ).mean()

                # base entropy (pre-tanh) proxy (fp32)
                entropy = base.entropy().sum(dim=-1).mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # skip non-finite minibatches
                if not torch.isfinite(loss):
                    self.log.warning("Skipping non-finite minibatch (loss/values NaN/Inf).")
                    continue

                # per-minibatch KL guard BEFORE optimizer step
                if self.target_kl and approx_kl >= self.target_kl * self.kl_stop_multiplier:
                    kl_tripped = True
                    break

                # optimize
                self.opt.zero_grad(set_to_none=True)
                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.opt)
                    if self.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            _flat_params(self.feature, self.policy), self.max_grad_norm
                        )
                    self.scaler.step(self.opt)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            _flat_params(self.feature, self.policy), self.max_grad_norm
                        )
                    self.opt.step()

                # stats
                running_kl.append(approx_kl.detach())
                clipfrac = (torch.abs(ratio - 1.0) > self.clip_eps).float().mean()
                running_clipfrac.append(clipfrac.detach())

                # Extra safety: if many samples are clipped, stop updates this epoch
                if clipfrac > 0.8:
                    kl_tripped = True
                    break

                # optional entropy target nudging
                if self.entropy_target is not None:
                    ent_err = float(entropy.item()) - float(self.entropy_target)
                    self.entropy_coef = float(np.clip(self.entropy_coef + 1e-3 * ent_err, 1e-4, 0.05))

                last_logs = {
                    "loss": float(loss.item()),
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(entropy.item()),
                }

            # epoch end stats
            if running_kl:
                mean_kl = torch.stack(running_kl).mean()
                mean_clipfrac = torch.stack(running_clipfrac).mean()
            else:
                # If all minibatches were skipped or KL-tripped immediately, set conservative stats
                mean_kl = torch.tensor(self.target_kl * self.kl_stop_multiplier, device=self.device)
                mean_clipfrac = torch.tensor(1.0, device=self.device)

            # adapt KL penalty
            if self.kl_adapt:
                if mean_kl > self.target_kl * 2.0:
                    self.kl_beta *= self.kl_beta_inc
                elif mean_kl < self.target_kl / 2.0:
                    self.kl_beta *= self.kl_beta_dec
                self.kl_beta = float(np.clip(self.kl_beta, 1e-4, 100.0))

            last_logs["approx_kl"] = float(mean_kl.item())
            last_logs["clipfrac"] = float(mean_clipfrac.item())
            last_logs["kl_beta"] = float(self.kl_beta)

            if hasattr(self.callbacks, "log_scalar"):
                self.callbacks.log_scalar("train/approx_kl", last_logs["approx_kl"])
                self.callbacks.log_scalar("train/clipfrac", last_logs["clipfrac"])
                self.callbacks.log_scalar("train/kl_beta", last_logs["kl_beta"])

            if kl_tripped:
                break

        return last_logs

    # ---------------- public train ----------------
    def train(self, episodes: int = 20):
        # fallback so this never fails even if __init__ is edited later
        batch_steps = getattr(self, "batch_steps", int(self.cfg.get("batch_steps", 2048)))
        for ep in range(episodes):
            traj = self._rollout(batch_steps)
            logs = self._update(traj)
            self.log.info(
                f"Episode {ep+1}/{episodes} | "
                f"loss={logs.get('loss', float('nan')):.4f} "
                f"pol={logs.get('policy_loss', float('nan')):.4f} "
                f"val={logs.get('value_loss', float('nan')):.4f} "
                f"ent={logs.get('entropy', float('nan')):.4f} "
                f"kl={logs.get('approx_kl', float('nan')):.4f} "
                f"clipfrac={logs.get('clipfrac', float('nan')):.3f} "
                f"kl_beta={logs.get('kl_beta', float('nan')):.3f}"
            )
            if self.callbacks:
                self.callbacks.on_episode_end(ep + 1, logs, self)

        if self.callbacks:
            self.callbacks.on_train_end(episodes)
