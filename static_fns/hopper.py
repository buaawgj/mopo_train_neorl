import numpy as np
import torch


class StaticFns:
    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        height = next_obs[:, 0]
        angle = next_obs[:, 1]
        not_done = torch.isfinite(next_obs).all(dim=-1) \
                   * torch.abs(next_obs[:, 1:] < 100).all(dim=-1) \
                   * (height > .7) \
                   * (torch.abs(angle) < .2)

        done = ~not_done
        return done
