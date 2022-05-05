import torch
import torch.nn as nn
""" Approximated polynomial activation functions """

degree = 3
interval = 12

'''
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
'''

def relu_approx(x):
  if degree == 3:
    if interval == 3:
      return 0.7146 + 1.5000*torch.pow(x/interval,1)+0.8793*torch.pow(x/interval,2)
    if interval == 5:
      return 0.7865 + 2.5000*torch.pow(x/interval,1)+1.88*torch.pow(x/interval,2)
    if interval == 7:
      return 0.9003 + 3.5000*torch.pow(x/interval,1)+2.9013*torch.pow(x/interval,2)
    if interval == 10:
      return 1.1155 + 5*torch.pow(x/interval,1)+4.4003*torch.pow(x/interval,2)
    if interval == 12:
      return 1.2751 + 6*torch.pow(x/interval,1)+5.3803*torch.pow(x/interval,2)
  if degree == 5:
    if interval == 7:
      return 0.7521 + 3.5000*torch.pow(x/interval,1)+4.3825*torch.pow(x/interval,2)-1.7281*torch.pow(x/interval,4)
    if interval == 20:
      return 1.3127 + 10*torch.pow(x/interval,1)+15.7631*torch.pow(x/interval,2)-7.6296*torch.pow(x/interval,4)

def sigmoid_approx(x):
  if degree == 3:
    if interval == 3:
      return 0.5 + 0.6997*torch.pow(x/interval,1)-0.2649*torch.pow(x/interval,3)
    if interval == 5:
      return 0.5 + 0.9917*torch.pow(x/interval,1)-0.5592*torch.pow(x/interval,3)
    if interval == 7:
      return 0.5 + 1.1511*torch.pow(x/interval,1)-0.7517*torch.pow(x/interval,3)
    if interval == 8:
      return 0.5 + 1.2010*torch.pow(x/interval,1)-0.8156*torch.pow(x/interval,2)
    if interval == 10:
      return 0.5 + 1.2384*torch.pow(x/interval,1)-0.8647*torch.pow(x/interval,2)

def identity(x):
  return x
## Wrap as nn modules

class ReLUApprox(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self,x):
    return relu_approx(x)

class SigmoidApprox(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self,x):
    return sigmoid_approx(x)
