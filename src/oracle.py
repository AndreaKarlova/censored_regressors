import torch
import numpy

class Oracle(): 
  def __init__(self, fcn, fcn_label, seed=42.): 
    self.fcn = fcn
    self.label = fcn_label
    self.seed = seed

  def set_grid(self, start, end, num_of_points):
    return torch.Tensor(numpy.linspace(
            start=start, stop=end, num=num_of_points))

  def evaluate_fcn(self, value):
    return self.fcn(value).squeeze()
    
  def get_sample(self, N, start, end, noise_scale=0.2, noisy=False):
    torch.manual_seed(self.seed)
    x = torch.distributions.Uniform(start, end).sample(sample_shape=(N,)).sort()
    y = self.evaluate_fcn(x)
    if noisy==True: 
      return x, self.get_noisy(y, noise_scale)
    else:
      return x, y

  def get_noisy(self, sample, noise_scale):
    torch.manual_seed(self.seed)
    return sample + torch.distributions.Normal(0.0, noise_scale).sample(sample_shape=(N,))

  def censor(self, sample, low, high, quantile_flag=False):
    if quantile_flag==True:
      return sample.clamp(min=(None if low is None else sample.quantile(low)), 
                          max=(None if high is None else sample.quantile(high)))
    else: 
      return sample.clamp(min=low, max=high)

  def censor_idx(self, sample, low, high, quantile_flag=False):
    mask = sample!= self.censor(sample, low, high, quantile_flag)
    return torch.where(mask, torch.ones(sample.shape), torch.zeros(sample.shape))