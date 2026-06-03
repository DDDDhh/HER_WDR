# -*- coding: utf-8 -*-
"""
PPO (clipped surrogate) agent. Modified from the original to:
  * recognise done-code 4 (HER-relabeled terminal) as terminal,
  * support an entropy-coefficient hyper-parameter from cfg,
  * apply gradient clipping for numerical stability,
  * use a slightly deeper actor/critic with layer norm — these inputs are
    16-dim and the policy must learn a non-trivial nonlinear mapping
    (heading-error -> differential brake).
"""
import copy

import torch
from torch import nn
from torch.distributions import MultivariateNormal, Categorical


# Mirror of the done codes in worker_ppo.py / logger.py
DONE_FLYING = 0
DONE_DEAD = 1
DONE_TIMEOUT = 2
DONE_COMPLETE = 3
DONE_HER = 4


class RolloutBuffer:
    def __init__(self):
        self.actions, self.states, self.logprobs = [], [], []
        self.rewards, self.is_terminals = [], []

    def clear(self):
        for lst in (self.actions, self.states, self.logprobs, self.rewards, self.is_terminals):
            del lst[:]


class ActorCritic(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.has_continuous_action_space:
            self.action_var = torch.full(
                (args.action_dim,), args.action_std_init ** 2
            ).to(args.device)

        # Slightly deeper than the original 64-32-16 stack: HER throws more
        # diverse goal directions at the network, which benefits from a bit
        # more capacity. LayerNorm stabilises training in the early phase.
        if args.has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(args.state_dim, 128),
                nn.LayerNorm(128),
                nn.Tanh(),
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, args.action_dim),
                nn.Sigmoid(),
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(args.state_dim, 128),
                nn.LayerNorm(128),
                nn.Tanh(),
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, args.action_dim),
                nn.Softmax(dim=-1),
            )

        self.critic = nn.Sequential(
            nn.Linear(args.state_dim, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def set_action_std(self, new_action_std):
        if self.args.has_continuous_action_space:
            self.action_var = torch.full(
                (self.args.action_dim,), new_action_std ** 2
            ).to(self.args.device)

    def forward(self):
        raise NotImplementedError

    def act(self, state, sampling=True):
        if self.args.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
            if not sampling:
                return action_mean.detach(), dist.log_prob(action_mean)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        if self.args.has_continuous_action_space:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(self.args.device)
            dist = MultivariateNormal(action_mean, cov_mat)
            if self.args.action_dim == 1:
                action = action.reshape(-1, 1)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, cfg):
        self.args = copy.deepcopy(cfg.agent)
        defaults = dict(
            lr_actor=1e-4, lr_critic=2e-4, gamma=0.99, K_epochs=10, eps_clip=0.2,
            has_continuous_action_space=True, action_std_init=0.5, device='cpu',
            entropy_coef=0.02, max_grad_norm=0.5, lr_decay=0.999, seed=0,
        )
        self.args.state_dim = cfg.proc.state_dim
        self.args.action_dim = cfg.proc.action_dim
        for k, v in defaults.items():
            if not hasattr(self.args, k):
                setattr(self.args, k, v)

        # Coerce yaml-loaded scientific notation strings to floats.
        for key in ('lr_actor', 'lr_critic', 'gamma', 'eps_clip',
                    'action_std_init', 'entropy_coef', 'max_grad_norm', 'lr_decay'):
            setattr(self.args, key, float(getattr(self.args, key)))

        if self.args.has_continuous_action_space:
            self.args.action_std = self.args.action_std_init
        if self.args.device == 'auto':
            self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if self.args.device == 'cpu':
            self.args.device = torch.device('cpu')
            print('Device set to : cpu')
        elif self.args.device == 'cuda':
            if torch.cuda.is_available():
                self.args.device = torch.device('cuda:0')
                print(f'Device set to : {torch.cuda.get_device_name(self.args.device)}')
                torch.cuda.empty_cache()
            else:
                self.args.device = torch.device('cpu')
                print('cuda not available; device set to : cpu')
        else:
            raise NotImplementedError(f'device type {self.args.device} not defined.')

        self.policy = ActorCritic(self.args).to(self.args.device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': self.args.lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': self.args.lr_critic},
        ])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=self.args.lr_decay
        )

        self.policy_old = ActorCritic(self.args).to(self.args.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

        torch.manual_seed(int(self.args.seed))
        if next(self.policy.parameters()).device.type == 'cuda':
            torch.cuda.manual_seed_all(int(self.args.seed))

    # ------------------------------------------------------------------
    def set_action_std(self, new_action_std):
        if self.args.has_continuous_action_space:
            self.args.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        if self.args.has_continuous_action_space:
            self.args.action_std = self.args.action_std * action_std_decay_rate
            if self.args.action_std <= min_action_std:
                self.args.action_std = min_action_std
            self.set_action_std(self.args.action_std)

    def select_action(self, state, sampling=True):
        with torch.no_grad():
            state_t = torch.FloatTensor(state).to(self.args.device)
            action, action_logprob = self.policy_old.act(state_t, sampling)
        if self.args.has_continuous_action_space:
            return (
                action.detach().cpu().numpy().flatten(),
                action_logprob.detach().cpu().numpy().flatten(),
            )
        return action.item(), action_logprob.item()

    # ------------------------------------------------------------------
    def update(self, buffer):
        args = self.args
        if args.has_continuous_action_space:
            states = buffer[:, :args.state_dim]
            actions = buffer[:, args.state_dim:args.state_dim + args.action_dim]
            logprobs = buffer[:, args.state_dim + args.action_dim].reshape(-1)
        else:
            states = buffer[:, :args.state_dim]
            actions = buffer[:, args.state_dim:args.state_dim + 1]
            logprobs = buffer[:, args.state_dim + 1].reshape(-1)
        next_states = buffer[:, args.state_dim + args.action_dim + 1:
                                args.state_dim * 2 + args.action_dim + 1]
        rewards_done = buffer[:, -2:]

        old_states = torch.as_tensor(states).to(args.device)
        old_actions = torch.as_tensor(actions).to(args.device)
        old_logprobs = torch.as_tensor(logprobs).to(args.device)
        old_next_states = torch.as_tensor(next_states).to(args.device)

        td_target = []
        discounted_reward = 0.0
        for i, (reward, is_terminal) in enumerate(reversed(rewards_done), start=1):
            terminal_code = int(is_terminal)
            if terminal_code == DONE_DEAD:
                # Real failure: no future value.
                discounted_reward = 0.0
            elif terminal_code == DONE_TIMEOUT:
                # Timeout: bootstrap from critic's value of the next state.
                with torch.no_grad():
                    discounted_reward = self.policy_old.critic(old_next_states[-i]).item()
            elif terminal_code == DONE_COMPLETE:
                # Real success: terminal reward already captures the goal.
                discounted_reward = 0.0
            elif terminal_code == DONE_HER:
                # HER-relabeled terminal: treat like a real success.
                discounted_reward = 0.0
            discounted_reward = float(reward) + args.gamma * discounted_reward
            td_target.insert(0, discounted_reward)

        td_target = torch.tensor(td_target, dtype=torch.float32).to(args.device)
        # Add eps to std to avoid div-by-zero on degenerate batches.
        td_target = (td_target - td_target.mean()) / (td_target.std() + 1e-8)

        for _ in range(int(args.K_epochs)):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)

            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = td_target - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - args.eps_clip, 1 + args.eps_clip) * advantages

            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * self.MseLoss(state_values, td_target)
                - args.entropy_coef * dist_entropy
            )

            self.optimizer.zero_grad()
            loss.mean().backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=args.max_grad_norm)
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

    # ------------------------------------------------------------------
    def save(self, checkpoint_path, episode, time_step):
        checkpoint = {
            'Episode': episode,
            'Time_step': time_step,
            'Policy': self.policy.state_dict(),
            'Policy_old': self.policy_old.state_dict(),
            'Optimizer': self.optimizer.state_dict(),
            'action_std': self.args.action_std,
        }
        torch.save(checkpoint, checkpoint_path)

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.args.device)
        self.policy.load_state_dict(checkpoint['Policy'])
        self.policy_old.load_state_dict(checkpoint['Policy'])
        if 'action_std' in checkpoint:
            saved_std = float(checkpoint['action_std'])
            self.args.action_std = saved_std
            self.set_action_std(saved_std)
            print(f'  action_std restored: {saved_std:.4f}')
        if 'Episode' in checkpoint:
            print(f'  checkpoint Episode: {checkpoint["Episode"]}')
        print(f'{checkpoint_path} loaded.')
