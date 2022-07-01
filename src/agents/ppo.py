import torch
from torch.distributions import Categorical
from torch.nn.functional import mse_loss

from agents.agent import TrainableAgent
from agents.impala import ImpalaNet
from agents.nature import NatureNet
from trajectories import TensorTrajectory


class PPOAgent(TrainableAgent):
    def __init__(self, act_space_size, epsilon=0.05, val_epsilon=0.1, clip_eps=0.25):
        super().__init__(act_space_size, val_epsilon=val_epsilon, epsilon=epsilon)
        self.clip_eps = clip_eps
        self.model = NatureNet(n_actions=act_space_size)

    def act(self, obs: torch.Tensor, add_rand=True):
        actions, state_values = self.model(obs)
        return Categorical(probs=actions, validate_args=False), state_values

    def loss(self, trajectory: "TensorTrajectory", logger=None):
        action_dist, values = self.act(trajectory.obs)

        # Compute the advantages (same as GAE A∞)
        advantage = trajectory.returns - trajectory.values
        # advantage = trajectory.returns
        # advantage = trajectory.advantages
        # advantage = (advantage - advantage.mean()) / (advantage.std(0) + 1e-8)

        # Compute the CLIP loss
        e = self.clip_eps
        r = action_dist.log_prob(trajectory.actions) - trajectory.log_prob_chosen
        r = r.exp()
        # r = action_dist.probs.gather(1, a) / trajectory.probs.gather(1, a)
        l_clip = torch.minimum(
            r * advantage,
            r.clip(1 - e, 1 + e) * advantage,
        )

        # Compute the Value Function loss. The procedure is similar to the CLIP loss, but with MSE and relative changes
        # returns = trajectory.returns
        # old_values = trajectory.values
        # l_vf = torch.minimum(
        #     (values - returns).square(),
        #     (old_values + (values - old_values).clip(-e, e) - returns).square(),
        # )
        l_vf = mse_loss(values, trajectory.returns, reduction='none')

        # Entropy regularization term
        l_s = action_dist.entropy()

        # Reduce the losses, scale and sum them. Negative terms are to be maximized
        l_clip = -l_clip.mean()
        l_vf = l_vf.mean() * 0.5
        l_s = -l_s.mean() * 0.01
        l_clip_vf_s = l_clip + l_vf + l_s
        # l_clip_vf_s = l_clip + l_s

        # Kullback-Leibler divergence (used to quantify the "difference" of the old vs new probability distribution
        kl = (r - 1 - r.log()).mean()

        if logger:
            logger.append(
                loss_clip=l_clip.item(),
                loss_value_mse=l_vf.item(),
                loss_entropy=l_s.item(),
                clip_fraction=((r - 1).abs() > e).float().mean().item(),
                kl=kl.item(),
            )

        return l_clip_vf_s