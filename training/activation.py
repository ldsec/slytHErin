import torch

""" Approximated polynomial activation functions """

def approx_relu_2d(x):
  """2-degree approx of relu in [-6,6] from https://arxiv.org/pdf/2009.03727.pdf"""
  a = 0.563059
  b = 0.5
  c = 0.078047
  x_2 = torch.square(x)
  return a + b*x + c*x_2
  
def approx_relu_4d(x):
  """4-degree approx of relu in [-6,6] from https://arxiv.org/pdf/2009.03727.pdf"""
  a = 0.119782
  b = 0.5
  c = 0.147298
  d = -0.002015
  x_2 = torch.square(x)
  x_4 = torch.square(x_2)
  return a + b*x + c*x_2 + d*x_4