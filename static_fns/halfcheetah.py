import numpy as np
import torch


class StaticFns:
    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        done = torch.tensor([False]).repeat(len(obs))
        return done

    @staticmethod
    def recompute_reward_fn(obs, act, next_obs, rew):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        new_rew = -(rew + 0.1 * np.sum(np.square(act))) - 0.1 * np.sum(np.square(act))
        return new_rew
