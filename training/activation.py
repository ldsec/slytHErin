import torch
import torch.nn as nn
""" Approximated polynomial activation functions """

degree = 31
interval = 20

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

def softrelu_approx(x):
  if degree == 3:
    return 4.3068982183392716e-02 *torch.pow(x,2)+5.0000000000000011e-01 *torch.pow(x,1)+1.0736854302789589e+00 *torch.pow(x,0)
  if degree == 7:
    return 1.26341363091953e+000 + 499.999999999999e-003*x+36.3166180856353e-003*torch.pow(x,2)+1.96501614554230e-018*torch.pow(x,3)-39.5406244348814e-006*torch.pow(x,4)-1.15958661158139e-021*torch.pow(x,5)+ 17.2198428652718e-009*torch.pow(x,6) 
  if degree == 15:
    return +3.9201365685041496e-19 *torch.pow(x,14)-1.6759051753627716e-31 *torch.pow(x,13)-1.7605934897466414e-15 *torch.pow(x,12)+6.6437119862124193e-28 *torch.pow(x,11)+3.1613842673115711e-12 *torch.pow(x,10)-1.0151613940596730e-24 *torch.pow(x,9)-2.9006374846194302e-09 *torch.pow(x,8)+7.4751297971598446e-22 *torch.pow(x,7)+1.4480094446066676e-06 *torch.pow(x,6)-2.6954925739318276e-19 *torch.pow(x,5)-3.9583191537633959e-04 *torch.pow(x,4)+4.2272974147859158e-17 *torch.pow(x,3)+7.0773473160366113e-02 *torch.pow(x,2)+4.9999999999999806e-01 *torch.pow(x,1)+8.3757767749917544e-01 *torch.pow(x,0)
  if degree == 31:
    return -1.0040897579718860e-53 *torch.pow(x,31)+6.2085331754358028e-40 *torch.pow(x,30)+9.4522902777573076e-50 *torch.pow(x,29)-5.7963804324148821e-36 *torch.pow(x,28)-4.0131279328625271e-46 *torch.pow(x,27)+2.4410642683332394e-32 *torch.pow(x,26)+1.0153477706512291e-42 *torch.pow(x,25)-6.1290204181405624e-29 *torch.pow(x,24)-1.7039434123075587e-39 *torch.pow(x,23)+1.0216863193793685e-25 *torch.pow(x,22)+1.9976235851829888e-36 *torch.pow(x,21)-1.1917424918638167e-22 *torch.pow(x,20)-1.6781853595392470e-33 *torch.pow(x,19)+9.9891167268766684e-20 *torch.pow(x,18)+1.0196230261578948e-30 *torch.pow(x,17)-6.0833342283869143e-17 *torch.pow(x,16)-4.4658877204790776e-28 *torch.pow(x,15)+2.6909707871865122e-14 *torch.pow(x,14)+1.3889468322950614e-25 *torch.pow(x,13)-8.5600457797298628e-12 *torch.pow(x,12)-2.9800845828620543e-23 *torch.pow(x,11)+1.9200743786780711e-09 *torch.pow(x,10)+4.2045289670858245e-21 *torch.pow(x,9)-2.9487406547016763e-07 *torch.pow(x,8)-3.6043867162675355e-19 *torch.pow(x,7)+2.9886906932909647e-05 *torch.pow(x,6)+1.6307741516672765e-17 *torch.pow(x,5)-1.9601130409477464e-03 *torch.pow(x,4)-2.8618809778714450e-16 *torch.pow(x,3)+1.0678923596705732e-01 *torch.pow(x,2)+5.0000000000000022e-01 *torch.pow(x,1)+7.1225856852636027e-01 *torch.pow(x,0)

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


class SILU(nn.Module):
  def __init__(self):
    super().__init__()
    self.sig = nn.Sigmoid()

  def forward(self,x):
    return x*self.sig(x)


class SILUApprox(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self,x):
    return sigmoid_approx(x)*x

  
