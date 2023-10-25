import torch
from torchmetrics import Metric
from src.models.gaussian_mixture_likelihood import GaussianMixture

class CoverageMetric(Metric):
    def __init__(self, confidence_level=0.95, gaussian_mixture_components=False):
        super().__init__()
        self.confidence_level = confidence_level
        self.mixture = gaussian_mixture_components
        self.add_state("counter", default=torch.tensor(0))
        self.add_state("total_samples", default=torch.tensor(0))
        self.add_state("sizes", default=torch.tensor([0]))

    def update(self, y_pred, y_true):
        if self.mixture:
            gm = GaussianMixture(means=y_pred[:, :, 0].T, stds=y_pred[:, :, 1].T)
            y_pred = gm.sample(int(1e4))
        else:
            y_pred = y_pred.T
        lower = (1 - self.confidence_level) / 2
        upper = 1 - lower
        lower_quantile = torch.quantile(y_pred, lower, dim=1)
        upper_quantile = torch.quantile(y_pred, upper, dim=1)
        in_quantiles = torch.logical_and(y_true >= lower_quantile, y_true <= upper_quantile)

        self.counter += torch.sum(in_quantiles).item()
        self.total_samples += (y_true.shape[0]-torch.isnan(y_true).sum())
        self.sizes = torch.concatenate((self.sizes, (upper_quantile[~torch.isnan(y_true)]-lower_quantile[~torch.isnan(y_true)])))


    def compute(self):
        print('coverage:'+str(self.counter / self.total_samples))
        return self.counter / self.total_samples

    def size(self):
        return torch.median(self.sizes[1:]) # skip zero value from initialization
