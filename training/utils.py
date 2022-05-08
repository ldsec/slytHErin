import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
import time
## interactive off
plt.ioff()
## setup torch enviro

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.autograd.set_detect_anomaly(True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ScaledAvgPool2d(nn.Module):
    """Define the ScaledAvgPool layer, a.k.a the Sum Pool"""
    def __init__(self, kernel_size, stride, padding=0):
      super().__init__()
      self.kernel_size = kernel_size
      self.AvgPool = nn.AvgPool2d(kernel_size=self.kernel_size, stride=stride, padding=padding)

    def forward(self,x):
      return (self.kernel_size**2)*self.AvgPool(x)

def single_to_multi_label(y):
  """
    Input: labels in {0,...,9}
    Output: one-hot encoding of y
  """
  y_onehot = torch.FloatTensor(y.shape[0], 10)
  y = y.unsqueeze(1)
  y_onehot.zero_()
  y_onehot.scatter_(1, y, 1)
  return y_onehot

## PLOT HELPER
def plot_history(key, train, history):
  """ 
    Plot loss and accuracy history during model run
    Input:
          key : str => name of the model
          train : bool => training 1 or test 0
          history : dict{str : list of floats}
  """
  if train:
    when = "train"
  else:
    when = "test"

  fig, ax = plt.subplots( 1, 2, figsize = (12,4) )
  ax[0].plot(history['loss'], label = "loss")
  ax[0].set_title( "Loss" )
  ax[0].set_xlabel( "Epochs" )
  ax[0].set_ylabel( "Loss" )
  ax[0].grid( True )
  ax[0].legend()

  ax[1].plot(history['accuracy'], label = "accuracy")
  ax[1].set_title( "Accuracy" )
  ax[1].set_xlabel( "Epochs" )
  ax[1].set_ylabel( "Accuracy" )
  ax[1].grid( True )
  ax[1].legend()

  plt.savefig(f"./images/{key}_{when}.png")
  plt.close()

## TRAIN
def train(logger, model, dataHandler, num_epochs, lr=0.05, momentum=0.9, L1Ratio=0.5, optim_algo='SGD', loss='MSE', regularizer='None'):
  l2_penalty =(1-L1Ratio)/2
  if regularizer == 'None':
    l2_penalty = 0
    L1Ratio = 0
  elif regularizer == 'L1':
    L1Ratio = 1
  elif regularizer == 'L2':
    L1Ratio = 0
  elif regularizer != 'Elastic':
    print("regularizer is None, L1, L2 or Elastic")
    exit()

  if optim_algo == 'Adam':
    optimizer = optim.Adam(model.parameters(),lr=lr, weight_decay=l2_penalty, amsgrad=True)
  elif optim_algo == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=l2_penalty, nesterov=True)
  else:
    print("Optimizer is Adam or SGD")
    exit
  if loss == 'CSE':
    criterion = nn.CrossEntropyLoss()
  elif loss == 'MSE':
    criterion = nn.MSELoss()  
  else:
    print("Loss is CrossEntropy or MSE")
    exit
  
  print(f"Train summary:\n lr={lr}\n epochs={num_epochs}\n optim={optim_algo}\n loss={loss}\n regularizer={regularizer}")

  trainHistory = {}
  trainHistory['loss'] = []
  trainHistory['accuracy'] = []

  model.train()
  for epoch in range(num_epochs):
    start = time.time()
    epoch_loss = 0
    num_correct = 0.0
    num_samples = 0.0
    
    for i, (data, labels) in enumerate(dataHandler.train_dl):
      data = data.to(device=device)
      if loss == 'MSE':
        labels_one_hot = single_to_multi_label(labels)
        labels_one_hot = labels_one_hot.to(device=device)
      else:
        labels = labels.to(device=device)
      
      optimizer.zero_grad()
      
      predictions = model(data)
      if loss == 'MSE':
        predictions, labels_one_hot = predictions.type('torch.FloatTensor'), labels_one_hot.type('torch.FloatTensor')
        loss_value = criterion(predictions, labels_one_hot)
      else:
        predictions = model(data)
        #predictions, labels = predictions.type('torch.FloatTensor'), labels.type('torch.FloatTensor')
        loss_value = criterion(predictions, labels)
      
      l1_norm = sum(torch.linalg.norm(p) for p in model.parameters())
      loss_value = loss_value + L1Ratio*l1_norm
      epoch_loss += loss_value.item()
      
      loss_value.backward()
      optimizer.step()

      if model.verbose and (i+1)%100 == 0:
        print(f"[?] Step {i+1}/{len(dataHandler.train_dl)} Epoch {epoch+1}/{num_epochs} Loss {loss_value.item()}")
    _, predicted_labels = predictions.max(1)
    num_correct += (predicted_labels == labels).sum().item()
    num_samples += predicted_labels.size(0)

    end = time.time()
    print(f"[?] {logger.name} Epoch {epoch+1}/{num_epochs} Loss {epoch_loss/(i+1):.4f} Accuracy {num_correct/num_samples:.4f} Time: {end-start:.4f}s")
    logger.log_step(epoch, i, epoch_loss/(i+1), num_correct/num_samples)  
    trainHistory['loss'].append(loss_value.item())
    trainHistory['accuracy'].append(num_correct/num_samples)
    
  plot_history(logger.name, True, trainHistory)

## EVAL 
def eval(logger, model, dataHandler, loss='MSE'):
  num_correct = 0
  num_samples = 0

  if loss == 'CSE':
    criterion = nn.CrossEntropyLoss()
  elif loss == 'MSE':
    criterion = nn.MSELoss()  
  else:
    print("Loss is CrossEntropy or MSE")
    exit
  
  model.eval()
  
  testHistory = {}
  testHistory['loss'] = []
  testHistory['accuracy'] = []
  test_loss = 0
  test_accuracy = 0
  
  with torch.no_grad():
    for batch, (data,labels) in enumerate(dataHandler.test_dl):
        data = data.to(device=device)
        if loss == 'MSE':
          labels_one_hot = single_to_multi_label(labels)
          labels_one_hot = labels_one_hot.to(device=device)
        else:
          labels = labels.to(device=device)
        
        predictions = model(data)
        if loss == 'MSE':
          predictions, labels_one_hot = predictions.type('torch.FloatTensor'), labels_one_hot.type('torch.FloatTensor')
          loss_value = criterion(predictions, labels_one_hot).item()
        else:
          predictions = model(data)
          #predictions, labels = predictions.type('torch.FloatTensor'), labels.type('torch.FloatTensor')
          loss_value = criterion(predictions, labels).item()
        
        test_loss += loss_value
        _, predicted_labels = predictions.max(1)
        num_correct += (predicted_labels == labels).sum().item()
        num_samples += predicted_labels.size(0)
        
        testHistory['loss'].append(loss_value)
        testHistory['accuracy'].append(float(num_correct) / float(num_samples))
        
        logger.log_batch(batch+1, loss_value, float(num_correct) / float(num_samples))
    
    test_accuracy = float(num_correct) / float(num_samples)
    test_loss = test_loss/len(dataHandler.test_dl) ## avg loss
    logger.finalize(test_loss, test_accuracy)

    plot_history(logger.name, False, testHistory)

  return test_loss, test_accuracy
