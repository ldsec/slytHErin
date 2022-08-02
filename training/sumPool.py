import torch.nn as nn

"""Define the ScaledAvgPool layer, a.k.a the Sum Pool"""
class ScaledAvgPool2d(nn.Module):
    def __init__(self, kernel_size, stride, padding=0):
      super().__init__()
      self.kernel_size = kernel_size
      self.AvgPool = nn.AvgPool2d(kernel_size=self.kernel_size, stride=stride, padding=padding)

    def forward(self,x):
      return (self.kernel_size**2)*self.AvgPool(x)
