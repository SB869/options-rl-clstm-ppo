# src/trader/training/ppo.py
from __future__ import annotations
from typing import Dict, Any, Tuple, Optional
import os

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

from trader.utils.logging import get_logger


def _flat_params(*modules):
    ps = []
    for m in modules:
        ps += [p for p in m.parameters() if p.requires_grad]
    return ps


class PPOTrainer:
    """
    PPO with recurrent feature & policy nets, GAE, value clipping,
    KL target (adaptive penalty) and robust numerics for tanh-squashed Gaussians.

    NEW: Debug logging (YAML-controlled) to surface failure modes early:
      - ratio/log-ratio stats, clipfrac, KL
      - moments of advantages/returns/value predictions
      - action stats
      - gradient norms
      - optional CSV dump (debug_stats.csv in run dir)
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

        # ---- Optimizer & core PPO knobs ----
        self.opt = optim.Adam(
            _flat_params(feature, policy),
            lr=cfg.get("lr", 3e-4),
            betas=(0.9, 0.999),
            eps=1e-8,
        )
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

        # ---- KL penalty & early-stopping ----
        self.max_grad_norm = float(cfg.get("max_grad_norm", 0.5))
        self.kl_adapt = bool(cfg.get("kl_adapt", True))
        self.kl_beta = float(cfg.get("kl_beta", 1.0))
        self.kl_beta_inc = float(cfg.get("kl_beta_inc", 1.5))
        self.kl_beta_dec = float(cfg.get("kl_beta_dec", 0.75))
        self.kl_stop_multiplier = float(cfg.get("kl_stop_multiplier", 1.5))

        # ---- Stability block (YAML: trainer.stability.*) ----
        stab = (cfg.get("stability") or {})
        self.jacobian_eps = float(stab.get("jacobian_eps", 1e-5))
        self.atanh_clip = float(stab.get("atanh_clip", 0.999999))
        self.log_ratio_clip = float(stab.get("log_ratio_clip", 20.0))
        self.kl_beta_floor = float(stab.get("kl_beta_floor", 1e-4))
        self.clipfrac_tripwire = float(stab.get("clipfrac_tripwire", 0.80))
        self.use_huber_value = bool(stab.get("use_huber_value", True))
        self.huber_delta = float(stab.get("huber_delta", 1.0))
        self.scale_returns = bool(stab.get("scale_returns", True))
        self.scale_returns_mode = str(stab.get("scale_returns_mode", "std")).lower()  # "std"|"zscore"

        # ---- Debug block (YAML: trainer.debug.*) ----
        dbg = (cfg.get("debug") or {})
        self.debug_enabled = bool(dbg.get("enabled", False))
        self.debug_ep_freq = int(dbg.get("episode_freq", 1))  # log every N episodes
        self.debug_csv = bool(dbg.get("write_csv", True))
        self.debug_hist_k = int(dbg.get("hist_k", 0))  # 0 = off; else keep per-epoch K samples of ratios etc.

        # run_dir for debug CSV (lazily resolved in train())
        self._debug_run_dir: Optional[str] = None
        self._debug_csv_path: Optional[str] = None
        self._last_grad_norm: Optional[float] = None

        # ---- AMP dtype selection ----
        if amp_dtype == "auto":
            self.amp_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        elif amp_dtype == "bfloat16":
            self.amp_dtype = torch.bfloat16
        elif amp_dtype == "float16":
            self.amp_dtype = torch.float16
        else:
            self.amp_dtype = torch.float32

        # GradScaler: enabled only for float16 (bf16 does not use it).
        self.scaler = GradScaler(
            "cuda",
            enabled=(self.device.type == "cuda" and self.amp_dtype == torch.float16),
        )

    @staticmethod
    def _torch_save(obj, path):
        torch.save(obj, path)

    # ---------- math helpers ----------
    def _sum_log_one_minus_a2(self, a: torch.Tensor) -> torch.Tensor:
        # sum over action dims: log(1 - a^2 + eps)
        # (subtract this from base logp for tanh-squashed Gaussians)
        eps = self.jacobian_eps
        return torch.log(1 - a.pow(2) + eps).sum(dim=-1)

    # ---------- rollout ----------
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

            # Run nets under AMP...
            with autocast("cuda", enabled=(self.device.type == "cuda"), dtype=self.amp_dtype):
                z, h_feat = self.feature(x, h_feat)
                mu, std, v, h_pol = self.policy(z, h_pol)

            # ...but do distribution/logp in fp32 for stability
            mu32 = mu.float()
            std32 = std.float()
            base = torch.distributions.Normal(mu32, std32)
            z_samp = base.rsample()
            a = torch.tanh(z_samp)

            base_logp = base.log_prob(z_samp).sum(dim=-1)
            logp = base_logp - self._sum_log_one_minus_a2(a.float())

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

    # ---------- gae ----------
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

        # sanitize
        adv = torch.nan_to_num(adv, nan=0.0, posinf=0.0, neginf=0.0)
        ret = torch.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0)

        # optional return scaling for critic stability
        if self.scale_returns:
            if self.scale_returns_mode == "zscore":
                mean = ret.mean()
                std = ret.std(unbiased=False)
                ret = (ret - mean) / (std + 1e-6)
            else:
                std = ret.std(unbiased=False)
                ret = ret / (std + 1e-6)

        # normalize advantages only
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        return adv, ret

    # ---------- forward over batch ----------
    def _forward_batch(self, obs_tensor: torch.Tensor, act: torch.Tensor):
        """
        Returns a Normal distribution (mu,std) constructed in fp32, and v_pred.
        """
        with autocast("cuda", enabled=(self.device.type == "cuda"), dtype=self.amp_dtype):
            z, _ = self.feature(obs_tensor)                 # (T,1,E)
            mu, std, v_pred, _ = self.policy(z)            # (T,A),(T,A),(T,1)

        base = torch.distributions.Normal(mu.float().view_as(act), std.float().view_as(act))
        return base, v_pred.squeeze(-1)

    # ---------- utils ----------
    def _compute_approx_kl(self, logp_old: torch.Tensor, logp_new: torch.Tensor) -> torch.Tensor:
        return (logp_old - logp_new).mean()

    # ---------- ppo update ----------
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

        # debug accumulators per epoch
        dbg_ratio_stats = []
        dbg_logratio_stats = []
        dbg_act_stats = []
        dbg_adv_stats = []
        dbg_ret_stats = []
        dbg_vpred_stats = []

        last_logs: Dict[str, float] = {}
        last_grad_norm_for_epoch: Optional[float] = None

        for ep in range(self.epochs):
            perm = idx[torch.randperm(T)]
            running_kl, running_clipfrac = [], []
            kl_tripped = False

            for start in range(0, T, self.minibatch_size):
                mb = perm[start : start + self.minibatch_size]

                base, v_pred = self._forward_batch(obs_tensor[mb], act[mb])

                # robust atanh reconstruction of pre-tanh action (fp32)
                a = act[mb].clamp(-self.atanh_clip, self.atanh_clip).float()
                z_recon = torch.atanh(a)

                # logp in fp32
                base_logp = base.log_prob(z_recon).sum(dim=-1).float()
                logp = base_logp - self._sum_log_one_minus_a2(a)

                # old log-probs must be finite
                lpo_raw = logp_old[mb]
                if not torch.isfinite(lpo_raw).all():
                    self.log.warning("Skipping minibatch due to non-finite old log-probs.")
                    continue
                lpo = lpo_raw.float()

                # safe ratio in log space
                L = self.log_ratio_clip
                log_ratio = torch.clamp(logp - lpo, min=-L, max=L)
                ratio = torch.exp(log_ratio)

                surr1 = ratio * adv[mb]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv[mb]
                policy_loss = -torch.min(surr1, surr2).mean()

                approx_kl = self._compute_approx_kl(lpo, logp)
                if self.kl_adapt:
                    policy_loss = policy_loss + self.kl_beta * approx_kl

                # value loss with clip (Huber optional)
                v_pred = torch.nan_to_num(v_pred, nan=0.0, posinf=0.0, neginf=0.0)
                v_clip = val[mb] + torch.clamp(v_pred - val[mb], -self.value_clip, self.value_clip)

                if self.use_huber_value:
                    loss_uncl = F.smooth_l1_loss(v_pred, ret[mb], beta=self.huber_delta, reduction="mean")
                    loss_cl = F.smooth_l1_loss(v_clip, ret[mb], beta=self.huber_delta, reduction="mean")
                    value_loss = torch.max(loss_uncl, loss_cl)
                else:
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

                # per-minibatch KL guard BEFORE optimizer step (>= so equality trips)
                if self.target_kl and approx_kl >= self.target_kl * self.kl_stop_multiplier:
                    kl_tripped = True
                    break

                # optimize
                self.opt.zero_grad(set_to_none=True)
                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.opt)
                    # grad norm AFTER unscale (before step)
                    if self.max_grad_norm is not None:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            _flat_params(self.feature, self.policy), self.max_grad_norm
                        )
                        last_grad_norm_for_epoch = float(grad_norm.item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
                    self.scaler.step(self.opt)
                    self.scaler.update()
                else:
                    loss.backward()
                    # grad norm (before step)
                    if self.max_grad_norm is not None:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            _flat_params(self.feature, self.policy), self.max_grad_norm
                        )
                        last_grad_norm_for_epoch = float(grad_norm.item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
                    self.opt.step()

                # stats
                running_kl.append(approx_kl.detach())
                clipfrac = (torch.abs(ratio - 1.0) > self.clip_eps).float().mean()
                running_clipfrac.append(clipfrac.detach())

                # --- debug sample collection (cheap) ---
                if self.debug_enabled:
                    # ratios/log-ratios moments (mean/std/min/max)
                    r = ratio.detach()
                    lr = log_ratio.detach()
                    dbg_ratio_stats.append(
                        torch.tensor([r.mean(), r.std(unbiased=False), r.min(), r.max()], device=self.device)
                    )
                    dbg_logratio_stats.append(
                        torch.tensor([lr.mean(), lr.std(unbiased=False), lr.min(), lr.max()], device=self.device)
                    )
                    # action moments
                    a_stats = torch.tensor(
                        [a.mean(), a.std(unbiased=False), a.min(), a.max()],
                        device=self.device,
                    )
                    dbg_act_stats.append(a_stats)
                    # adv/ret/vpred moments
                    dbg_adv_stats.append(
                        torch.tensor([adv[mb].mean(), adv[mb].std(unbiased=False), adv[mb].min(), adv[mb].max()], device=self.device)
                    )
                    dbg_ret_stats.append(
                        torch.tensor([ret[mb].mean(), ret[mb].std(unbiased=False), ret[mb].min(), ret[mb].max()], device=self.device)
                    )
                    dbg_vpred_stats.append(
                        torch.tensor([v_pred.mean(), v_pred.std(unbiased=False), v_pred.min(), v_pred.max()], device=self.device)
                    )

                # Extra safety: if many samples are clipped, stop updates this epoch
                if clipfrac > self.clipfrac_tripwire:
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
                # If all minibatches were skipped or tripped immediately, set conservative stats
                mean_kl = torch.tensor(self.target_kl * self.kl_stop_multiplier, device=self.device)
                mean_clipfrac = torch.tensor(1.0, device=self.device)

            # adapt KL penalty with configurable floor
            if self.kl_adapt:
                if mean_kl > self.target_kl * 2.0:
                    self.kl_beta *= self.kl_beta_inc
                elif mean_kl < self.target_kl / 2.0:
                    self.kl_beta *= self.kl_beta_dec
                self.kl_beta = float(np.clip(self.kl_beta, self.kl_beta_floor, 100.0))

            last_logs["approx_kl"] = float(mean_kl.item())
            last_logs["clipfrac"] = float(mean_clipfrac.item())
            last_logs["kl_beta"] = float(self.kl_beta)
            if last_grad_norm_for_epoch is not None:
                last_logs["grad_norm"] = float(last_grad_norm_for_epoch)

            if hasattr(self.callbacks, "log_scalar"):
                self.callbacks.log_scalar("train/approx_kl", last_logs["approx_kl"])
                self.callbacks.log_scalar("train/clipfrac", last_logs["clipfrac"])
                self.callbacks.log_scalar("train/kl_beta", last_logs["kl_beta"])
                if "grad_norm" in last_logs:
                    self.callbacks.log_scalar("train/grad_norm", last_logs["grad_norm"])

            # emit debug line once per epoch loop end (kept compact)
            if self.debug_enabled:
                def _agg(tlist):
                    if not tlist:
                        return (np.nan, np.nan, np.nan, np.nan)
                    arr = torch.stack(tlist)
                    # average stats across minibatches
                    return tuple(map(float, arr.mean(dim=0).tolist()))

                r_mu, r_sd, r_min, r_max = _agg(dbg_ratio_stats)
                lr_mu, lr_sd, lr_min, lr_max = _agg(dbg_logratio_stats)
                a_mu, a_sd, a_min, a_max = _agg(dbg_act_stats)
                adv_mu, adv_sd, adv_min, adv_max = _agg(dbg_adv_stats)
                ret_mu, ret_sd, ret_min, ret_max = _agg(dbg_ret_stats)
                vp_mu, vp_sd, vp_min, vp_max = _agg(dbg_vpred_stats)

                self.log.info(
                    "[DBG] epoch_end "
                    f"KL={last_logs['approx_kl']:.4e} clipfrac={last_logs['clipfrac']:.3f} "
                    f"ratio[μ={r_mu:.3e},σ={r_sd:.3e},min={r_min:.3e},max={r_max:.3e}] "
                    f"logratio[μ={lr_mu:.3e},σ={lr_sd:.3e},min={lr_min:.3e},max={lr_max:.3e}] "
                    f"act[μ={a_mu:.3e},σ={a_sd:.3e},min={a_min:.3e},max={a_max:.3e}] "
                    f"adv[μ={adv_mu:.3e},σ={adv_sd:.3e},min={adv_min:.3e},max={adv_max:.3e}] "
                    f"ret[μ={ret_mu:.3e},σ={ret_sd:.3e},min={ret_min:.3e},max={ret_max:.3e}] "
                    f"vpred[μ={vp_mu:.3e},σ={vp_sd:.3e},min={vp_min:.3e},max={vp_max:.3e}] "
                    f"grad_norm={last_logs.get('grad_norm', float('nan')):.3e}"
                )

            if kl_tripped:
                break

        return last_logs

    # ---------- public train ----------
    def train(self, episodes: int = 20):
        # resolve run dir for debug CSV (once)
        if self.debug_enabled and self.callbacks and hasattr(self.callbacks, "get_run_paths"):
            paths = self.callbacks.get_run_paths()
            self._debug_run_dir = paths.get("run_dir")
            if self._debug_run_dir and self.debug_csv:
                self._debug_csv_path = os.path.join(self._debug_run_dir, "debug_stats.csv")
                if not os.path.exists(self._debug_csv_path):
                    # write header
                    with open(self._debug_csv_path, "w", encoding="utf-8") as f:
                        f.write(
                            "episode,loss,policy_loss,value_loss,entropy,approx_kl,clipfrac,kl_beta,grad_norm,"
                            "ratio_mu,ratio_sd,ratio_min,ratio_max,"
                            "logratio_mu,logratio_sd,logratio_min,logratio_max,"
                            "act_mu,act_sd,act_min,act_max,"
                            "adv_mu,adv_sd,adv_min,adv_max,"
                            "ret_mu,ret_sd,ret_min,ret_max,"
                            "vpred_mu,vpred_sd,vpred_min,vpred_max\n"
                        )

        batch_steps = getattr(self, "batch_steps", int(self.cfg.get("batch_steps", 2048)))
        for ep in range(episodes):
            traj = self._rollout(batch_steps)
            logs = self._update(traj)

            # emit one compact training line
            self.log.info(
                f"Episode {ep+1}/{episodes} | "
                f"loss={logs.get('loss', float('nan')):.4f} "
                f"pol={logs.get('policy_loss', float('nan')):.4f} "
                f"val={logs.get('value_loss', float('nan')):.4f} "
                f"ent={logs.get('entropy', float('nan')):.4f} "
                f"kl={logs.get('approx_kl', float('nan')):.4f} "
                f"clipfrac={logs.get('clipfrac', float('nan')):.3f} "
                f"kl_beta={logs.get('kl_beta', float('nan')):.3f}"
                + (f" grad={logs.get('grad_norm', float('nan')):.3e}" if "grad_norm" in logs else "")
            )

            # periodic debug CSV row
            if self.debug_enabled and self._debug_csv_path and ((ep + 1) % self.debug_ep_freq == 0):
                # NOTE: we don't recompute moments here (already logged at epoch_end).
                # For CSV, we use the last epoch_end moments stored in logs? We didn't store them.
                # To keep CSV simple, we only persist scalar logs here; rich moments are in console logs.
                with open(self._debug_csv_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"{ep+1},{logs.get('loss', float('nan'))},{logs.get('policy_loss', float('nan'))},"
                        f"{logs.get('value_loss', float('nan'))},{logs.get('entropy', float('nan'))},"
                        f"{logs.get('approx_kl', float('nan'))},{logs.get('clipfrac', float('nan'))},"
                        f"{logs.get('kl_beta', float('nan'))},{logs.get('grad_norm', float('nan'))},"
                        + ",".join(["nan"] * 24)  # placeholders for per-epoch moments to keep header consistent
                        + "\n"
                    )

            if self.callbacks:
                self.callbacks.on_episode_end(ep + 1, logs, self)

        if self.callbacks:
            self.callbacks.on_train_end(episodes)
