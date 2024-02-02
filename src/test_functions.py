import torch

# 1D test functions
# define function in the torch specific way
fcn1 = lambda x: 0.5 * torch.sin(3 * x)
fcn2 = lambda x: torch.pow((6 * x - 2),2)* torch.sin(2 * (6 * x - 2))
fcn3 = lambda x: torch.Tensor(x) * torch.sin(x)
fcn4 = lambda x: 0.5 * torch.sin(x) - 0.02 * torch.pow((10 - x), 2) + 2

fcn_dict_1D = {
  'fcn1': {
    'label': "$f(x) = 0.5 \sin(3x)$",
    'fcn': fcn1,
    'start': 0, 
    'end': 5, 
    'num_grid': 20, 
    'num_test': 200, 
    'pad_test': 1
    },
  'fcn2': {
    'label': "$f(x) = 0.5 \sin(3x)$",
    'fcn': fcn2,
    'start': 0, 
    'end': 1, 
    'num_grid': 30,
    'num_test': 300, 
    'pad_test': 0.2
    }, 
  'fcn3': {
    'label': "$f(x) = x \sin(x)$",
    'fcn': fcn3,
    'start': 0, 
    'end': 10, 
    'num_grid': 1000,
    'num_test': 5000, 
    'pad_test': 2
    }, 
  'fcn4': {
    'label': "$f(x) = x \sin(x)$", 
    'fcn': fcn4,
    'start': 0, 
    'end': 20, 
    'num_grid': 1000, 
    'num_test': 5000, 
    'pad_test': 5
    }
}
