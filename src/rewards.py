import torch

from trajectories import ListTrajectory


def discount_returns(episodes: ListTrajectory, gamma):
    """Reward discounting: Rₜ = rₜ + γRₜ₊₁"""
    discounted = []
    for er in episodes.rewards:
        er_flip = er.flip((0,))
        dr = [er_flip[0]]
        for r in er_flip[1:]:
            dr.append(r + gamma * dr[-1])
        dr = torch.tensor(dr).flip((0,))
        discounted.append(dr)
    return discounted


def gae(episodes: ListTrajectory, gamma, _lambda):
    """Generalized Advantage Estimate"""
    advantages = []
    for episode in episodes:
        r_flip = episode.rewards.flip((0,))
        v_flip = episode.values.flip((0,))

        ep_adv = []
        v_prev = 0
        a_prev = 0

        for r, v in zip(r_flip, v_flip):
            delta = r - v + gamma * v_prev  # δₜ = rₜ + γV(sₜ₊₁) - V(s)
            a = delta + gamma * _lambda * a_prev  # Aₜ = δₜ + λγAₜ₊₁
            ep_adv.append(a)
            v_prev = v
            a_prev = a
        ep_adv = torch.tensor(ep_adv).flip((0,))
        advantages.append(ep_adv)
    return advantages


# ======================================================================


def win_metrics(episodes: ListTrajectory, max_ep_len, key_prefix=None) -> dict:
    """Compute a dictionary of metrics for the given episodes. Used to evaluate the performance of the agent"""
    n_wins = 0
    n_losses = 0
    n_unfinished = 0
    for r in episodes.rewards:
        if r[-1] > 0:
            n_wins += 1
        elif len(r) == max_ep_len:
            n_unfinished += 1
        else:
            n_losses += 1

    metrics = dict(
        n_wins=n_wins,
        n_losses=n_losses,
        n_unfinished=n_unfinished,
        win_ratio=n_wins / (n_wins + n_losses + n_unfinished),
    )
    if key_prefix is None:
        return metrics
    return {f"{key_prefix}{k}": v for k, v in metrics.items()}


def reward_metrics(episodes: ListTrajectory, key_prefix=None) -> dict:
    """Compute a dictionary of metrics for the given episodes. Used to evaluate the performance of the agent"""
    rewards = torch.cat(episodes.rewards)
    returns = torch.cat(episodes.returns)
    advantages = torch.cat(episodes.advantages)
    metrics = dict(
        rewards_sum=rewards.sum().item(),
        rewards_std=rewards.std(0).item(),
        rewards_mean=rewards.mean().item(),
        #
        returns_sum=returns.sum().item(),
        returns_std=returns.std(0).item(),
        returns_mean=returns.mean().item(),
        #
        advantages_sum=advantages.sum().item(),
        advantages_std=advantages.std(0).item(),
        advantages_mean=advantages.mean().item(),
    )

    if key_prefix is None:
        return metrics
    return {f"{key_prefix}{k}": v for k, v in metrics.items()}


def action_metrics(episodes: ListTrajectory, key_prefix="") -> dict:
    """Compute a dict representing the distribution of actions in the episodes"""
    d = dict()
    actions, counts = torch.cat(episodes.actions).unique(return_counts=True)
    for a, c in torch.stack((actions, counts), dim=1):
        d[f"{key_prefix}action_{a.item()}"] = c.item()
    return d
