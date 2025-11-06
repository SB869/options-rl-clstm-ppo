from __future__ import annotations
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.optim as optim

from trader.utils.logging import get_logger


def _tanh_squash_correction(a: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    return torch.log(1 - a.pow(2) + eps).sum(dim=-1)


class PPOTrainer:
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

        params = list(feature.parameters()) + list(policy.parameters())
        self.opt = optim.Adam(params, lr=cfg.get("lr", 3e-4))

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

        # AMP dtype
        if amp_dtype == "auto":
            self.amp_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        elif amp_dtype == "bfloat16":
            self.amp_dtype = torch.bfloat16
        elif amp_dtype == "float16":
            self.amp_dtype = torch.float16
        else:
            self.amp_dtype = torch.float32

        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == "cuda" and self.amp_dtype != torch.float32))

    @staticmethod
    def _torch_save(obj, path):
        torch.save(obj, path)

    # ------------- rollout -------------
    def _rollout(self, steps: int) -> Dict[str, torch.Tensor]:
        obs, _ = self.env.reset()
        traj = {k: [] for k in ["obs", "act", "rew", "val", "logp"]}
        h_feat = None
        h_pol = None

        for _ in range(steps):
            x = torch.tensor(obs, dtype=torch.float32, device=self.device).view(1, 1, -1)

            with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda"), dtype=self.amp_dtype):
                z, h_feat = self.feature(x, h_feat)
                mu, std, v, h_pol = self.policy(z, h_pol)

                base = torch.distributions.Normal(mu, std)
                z_sample = base.rsample()
                a = torch.tanh(z_sample)

                base_logp = base.log_prob(z_sample).sum(dim=-1)
                corr = _tanh_squash_correction(a)
                logp = base_logp + corr

            next_obs, r, done, _, info = self.env.step(a.detach().view(-1).cpu().numpy())

            traj["obs"].append(obs)
            traj["act"].append(a.detach().view(-1).cpu().numpy())
            traj["rew"].append(float(r))
            traj["val"].append(float(v.item()))
            traj["logp"].append(float(logp.item()))

            obs = next_obs
            if done:
                obs, _ = self.env.reset()
                h_feat = None
                h_pol = None

        # to tensors on device
        for k in traj:
            traj[k] = torch.tensor(np.array(traj[k]), dtype=torch.float32, device=self.device)
        return traj

    # ------------- gae -------------
    def _gae(self, rew: torch.Tensor, val: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            v_next = torch.cat([val[1:], val[-1:]])
            deltas = rew + self.gamma * v_next - val
            adv = torch.zeros_like(rew)
            gae = 0.0
            for t in reversed(range(len(rew))):
                gae = deltas[t] + self.gamma * self.lam * gae
                adv[t] = gae
            ret = adv + val
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, ret

    # ------------- single forward over batch -------------
    def _forward_batch(self, obs_tensor: torch.Tensor, act: torch.Tensor):
        with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda"), dtype=self.amp_dtype):
            z, _ = self.feature(obs_tensor)                 # (T,1,E)
            mu, std, v_pred, _ = self.policy(z)            # (T,A),(T,A),(T,1)
            base = torch.distributions.Normal(mu.view_as(act), std.view_as(act))

        return base, v_pred.squeeze(-1)

    # ------------- ppo update -------------
    def _update(self, traj: Dict[str, torch.Tensor]) -> Dict[str, float]:
        rew, val, logp_old, act = traj["rew"], traj["val"], traj["logp"], traj["act"]
        adv, ret = self._gae(rew, val)

        # Prepare sequence as (T,1,D)
        obs_tensor = traj["obs"].float().view(-1, 1, traj["obs"].shape[-1])

        T = obs_tensor.shape[0]
        idx = torch.randperm(T, device=self.device)

        kl_stop = False
        last_logs = {}

        for ep in range(self.epochs):
            if kl_stop:
                break

            for start in range(0, T, self.minibatch_size):
                mb = idx[start : start + self.minibatch_size]
                base, v_pred = self._forward_batch(obs_tensor[mb], act[mb])

                # reconstruct pre-tanh
                a = act[mb].clamp(-0.999999, 0.999999)
                z_recon = 0.5 * (torch.log1p(a) - torch.log1p(-a)) * 2.0
                with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda"), dtype=self.amp_dtype):
                    base_logp = base.log_prob(z_recon).sum(dim=-1)
                    corr = _tanh_squash_correction(a)
                    logp = base_logp + corr

                    ratio = torch.exp(logp - logp_old[mb])
                    surr1 = ratio * adv[mb]
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv[mb]
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # value clip (like PPO2)
                    v_clip = val[mb] + torch.clamp(v_pred - val[mb], -self.value_clip, self.value_clip)
                    value_loss = 0.5 * torch.max((ret[mb] - v_pred).pow(2), (ret[mb] - v_clip).pow(2)).mean()

                    entropy = base.entropy().sum(dim=-1).mean()
                    loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.opt.zero_grad(set_to_none=True)
                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(list(self.feature.parameters()) + list(self.policy.parameters()), 1.0)
                    self.scaler.step(self.opt)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(self.feature.parameters()) + list(self.policy.parameters()), 1.0)
                    self.opt.step()

                # Approx KL (old-new)
                approx_kl = (logp_old[mb] - logp).mean().clamp_min(0.0)

                # Optional entropy target (nudges log_std)
                if self.entropy_target is not None:
                    # Simple proportional control on entropy to approach target
                    ent_err = float(entropy.item()) - float(self.entropy_target)
                    # tweak entropy_coef slightly (bounded)
                    self.entropy_coef = float(np.clip(self.entropy_coef + 1e-3 * ent_err, 1e-4, 0.05))

                last_logs = {
                    "loss": float(loss.item()),
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(entropy.item()),
                    "approx_kl": float(approx_kl.item()),
                }

                if self.target_kl and approx_kl > self.target_kl:
                    kl_stop = True
                    break

        return last_logs

    # ------------- public train -------------
    def train(self, episodes: int = 20):
        for ep in range(episodes):
            traj = self._rollout(self.batch_steps)
            logs = self._update(traj)
            self.log.info(
                f"Episode {ep+1}/{episodes} | "
                f"loss={logs.get('loss', float('nan')):.4f} "
                f"pol={logs.get('policy_loss', float('nan')):.4f} "
                f"val={logs.get('value_loss', float('nan')):.4f} "
                f"ent={logs.get('entropy', float('nan')):.4f} "
                f"kl={logs.get('approx_kl', float('nan')):.4f}"
            )
            if self.callbacks:
                self.callbacks.on_episode_end(ep + 1, logs, self)

        if self.callbacks:
            self.callbacks.on_train_end(episodes)
