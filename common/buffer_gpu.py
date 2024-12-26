import numpy as np
import torch


class ReplayBuffer:
    def __init__(
            self,
            buffer_size,
            obs_shape,
            obs_dtype,
            action_dim,
            action_dtype,
            device
    ):
        self.max_size = buffer_size
        self.obs_shape = obs_shape
        self.obs_dtype = obs_dtype
        self.action_dim = action_dim
        self.action_dtype = action_dtype
        self.device = device

        self.ptr = 0
        self.size = 0

        self.observations = torch.zeros((self.max_size,) + self.obs_shape, dtype=obs_dtype).to(self.device)
        self.next_observations = torch.zeros((self.max_size,) + self.obs_shape, dtype=obs_dtype).to(self.device)
        self.actions = torch.zeros((self.max_size, self.action_dim), dtype=action_dtype).to(self.device)
        self.rewards = torch.zeros((self.max_size, 1), dtype=torch.float32).to(self.device)
        self.terminals = torch.zeros((self.max_size, 1), dtype=torch.float32).to(self.device)

    def add(self, obs, next_obs, action, reward, terminal):
        # Copy to avoid modification by reference
        self.observations[self.ptr] = torch.tensor(obs).to(self.device)
        self.next_observations[self.ptr] = torch.tensor(next_obs).to(self.device)
        self.actions[self.ptr] = torch.tensor(action).to(self.device)
        self.rewards[self.ptr] = torch.tensor(reward).to(self.device)
        self.terminals[self.ptr] = torch.tensor(terminal).to(self.device)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def load_dataset(self, dataset):
        observations = torch.tensor(dataset["observations"], dtype=self.obs_dtype).to(self.device)
        next_observations = torch.tensor(dataset["next_observations"], dtype=self.obs_dtype).to(self.device)
        actions = torch.tensor(dataset["actions"], dtype=self.action_dtype).to(self.device)
        rewards = torch.tensor(dataset["rewards"]).reshape(-1, 1).to(self.device)
        terminals = torch.tensor(dataset["terminals"], dtype=torch.float32).reshape(-1, 1).to(self.device)

        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals

        self.ptr = len(observations)
        self.size = len(observations)

    def add_batch(self, obs, next_obs, actions, rewards, terminals):
        batch_size = len(obs)
        if self.ptr + batch_size > self.max_size:
            begin = self.ptr
            end = self.max_size
            first_add_size = end - begin
            self.observations[begin:end] = obs[:first_add_size].clone().detach().to(self.device)
            self.next_observations[begin:end] = next_obs[:first_add_size].clone().detach().to(self.device)
            self.actions[begin:end] = actions[:first_add_size].clone().detach().to(self.device)
            self.rewards[begin:end] = rewards[:first_add_size].clone().detach().to(self.device)
            self.terminals[begin:end] = terminals[:first_add_size].clone().detach().to(self.device)

            begin = 0
            end = batch_size - first_add_size
            self.observations[begin:end] = obs[first_add_size:].clone().detach().to(self.device)
            self.next_observations[begin:end] = next_obs[first_add_size:].clone().detach().to(self.device)
            self.actions[begin:end] = actions[first_add_size:].clone().detach().to(self.device)
            self.rewards[begin:end] = rewards[first_add_size:].clone().detach().to(self.device)
            self.terminals[begin:end] = terminals[first_add_size:].clone().detach().to(self.device)
            
            self.ptr = end
            self.size = min(self.size + batch_size, self.max_size)

        else:
            begin = self.ptr
            end = self.ptr + batch_size
            self.observations[begin:end] = obs.clone().detach().to(self.device)
            self.next_observations[begin:end] = next_obs.clone().detach().to(self.device)
            self.actions[begin:end] = actions.clone().detach().to(self.device)
            self.rewards[begin:end] = rewards.clone().detach().to(self.device)
            self.terminals[begin:end] = terminals.clone().detach().to(self.device)

            self.ptr = end
            self.size = min(self.size + batch_size, self.max_size)

    def sample(self, batch_size):
        batch_indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "observations": self.observations[batch_indices],
            "actions": self.actions[batch_indices],
            "next_observations": self.next_observations[batch_indices],
            "terminals": self.terminals[batch_indices],
            "rewards": self.rewards[batch_indices]
        }

    def sample_all(self):
        return {
            "observations": self.observations[:self.size],
            "actions": self.actions[:self.size],
            "next_observations": self.next_observations[:self.size],
            "terminals": self.terminals[:self.size],
            "rewards": self.rewards[:self.size]
        }

    @property
    def get_size(self):
        return self.size
