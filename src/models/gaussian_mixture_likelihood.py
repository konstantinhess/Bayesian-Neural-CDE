import torch
from torch.distributions import Normal

class GaussianMixture:
    def __init__(self, means, stds): # input shape: n_obs x n_components
        self.n_obs = means.shape[0]
        self.num_components = means.shape[-1]
        self.components = [Normal(means[:, i], stds[:, i]) for i in range(self.num_components)]
    def sample(self, mc_samples): # n_samples = number of MC samples drawn from the mixture model
        component_idx = torch.randint(0, self.num_components,
                                      (mc_samples,))  # draw mc_samples from randomly chosen component for each obs

        return torch.stack([self.components[idx].sample() for idx in component_idx]).T # output shape: n_obs x mc_samples

    def log_prob(self, value): # input shape: either grid_shape or grid_shape x n_obs
        if len(value.shape) < self.n_obs:
            value = torch.stack([value for _ in range(self.n_obs)]).T

        component_log_probs = torch.stack([component.log_prob(value) for component in self.components])
        log_prob = torch.logsumexp(component_log_probs, dim=0) - torch.log(torch.tensor(self.num_components)).T
        return log_prob # grid_shape x n_obs
