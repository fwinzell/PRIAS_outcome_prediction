import torch
import numpy as np

class GaussianNoise2Features(torch.nn.Module):
    def __init__(self, p, mu=0, sigma=0.01):
        super().__init__()
        self.p = p
        self.mu = mu
        self.sigma = sigma

    def forward(self, f):
        noise = torch.normal(mean=self.mu, std=self.sigma, size=f.size())
        out = (torch.rand(f.size()) > (1-self.p)).float()
        f_hat = f + noise*out
        return f_hat

if __name__ == "__main__":
    feature = torch.ones((1,256,2048))
    gn = GaussianNoise2Features(p=0.25)
    fat = gn(feature)

