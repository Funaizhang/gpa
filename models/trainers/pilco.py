import gin
import gpytorch.likelihoods
import torch
import tqdm

from models.gpmodel import GPModel


@gin.configurable
class PILCOTrainer:
    def __init__(self, data_size):
        inducing_points = torch.randn(10, 1)
        self.model = GPModel(inducing_points=inducing_points)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.training_iter = 200
        self.mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=data_size)

    def train(self, train_x_mean, train_x_stdv, train_y):
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.likelihood.parameters()}
        ], lr=0.01)
        iterator = tqdm.tqdm(range(self.training_iter))
        for i in iterator:
            train_x_sample = torch.distributions.Normal(train_x_mean, train_x_stdv).rsample()
            optimizer.zero_grad()
            output = self.model(train_x_sample)
            loss = -self.mll(output, train_y)
            iterator.set_postfix(loss=loss.item())
            loss.backward()
            optimizer.step()

    def eval(self, test_x):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            return self.likelihood(self.model(test_x))
