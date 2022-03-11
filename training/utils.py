import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable

## interactive off
plt.ioff()
## setup torch enviro
torch.manual_seed(42)
torch.autograd.set_detect_anomaly(True)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
  ax[0].plot(history['loss'], label = when+"----"+key)
  ax[0].set_title( "Loss" )
  ax[0].set_xlabel( "Epochs" )
  ax[0].set_ylabel( "Loss" )
  ax[0].grid( True )
  ax[0].legend()

  ax[1].plot(history['accuracy'], label = when+"----"+key)
  ax[1].set_title( "Accuracy" )
  ax[1].set_xlabel( "Epochs" )
  ax[1].set_ylabel( "Accuracy" )
  ax[1].grid( True )
  ax[1].legend()

  plt.savefig(f"./images/{key}_{when}.png")
  plt.close()

## TRAIN
def train(logger, model, dataHandler, num_epochs, TPU=False):
  
  num_epochs = num_epochs
  optimizer = optim.SGD(model.parameters(), lr=5e-5, momentum=0.9)
  criterion = nn.MSELoss()

  trainHistory = {}
  trainHistory['loss'] = []
  trainHistory['accuracy'] = []

  model.train()
  for epoch in range(num_epochs):
    
    epoch_loss = 0
    num_correct = 0
    num_samples = 0
    
    for i, (data, labels) in enumerate(dataHandler.train_dl):
      data = data.to(device=device)
      labels = single_to_multi_label(labels)
      labels = labels.to(device=device)

      optimizer.zero_grad()
      
      predictions = model(data)
      predicted_labels, labels = predictions.type('torch.FloatTensor'), labels.type('torch.FloatTensor')
      
      loss = criterion(predicted_labels, labels)
      num_correct += (predicted_labels == labels).sum().item()
      num_samples += predicted_labels.size(0)
      epoch_loss += loss.item()
      
      if model.verbose and (i+1)%100 == 0:
        print(f"[?] Step {i+1}/{len(dataHandler.train_dl)} Epoch {epoch+1}/{num_epochs} Loss {loss.item()}")
      
      #loss = Variable(loss, requires_grad = True)
      loss.backward()
      
      if not TPU:
        optimizer.step()
      else:
        xm.optimizer_step(optimizer, barrier=True) ## if TPU 
    
    print(f"[?] Epoch {epoch+1}/{num_epochs} Loss {loss.item():.4f}")
    logger.log_step(epoch, i, epoch_loss/(i+1), num_correct/num_samples)  
    trainHistory['loss'].append(loss.item())
    trainHistory['accuracy'].append(num_correct/num_samples)
    
  plot_history(logger.name, True, trainHistory)

## EVAL 
def eval(logger, model, dataHandler):
  num_correct = 0
  num_samples = 0

  criterion = nn.MSELoss()
  model.eval()
  
  testHistory = {}
  testHistory['loss'] = []
  testHistory['accuracy'] = []
  test_loss = 0
  test_accuracy = 0
  
  with torch.no_grad():
    for batch, (data,labels) in enumerate(dataHandler.test_dl):
        data = data.to(device=device)
        labels = labels.to(device=device)
        
        predictions = model(data)
        _, predicted_labels = predictions.max(1)
        predicted_labels, labels = predicted_labels.type('torch.FloatTensor'), labels.type('torch.FloatTensor')
      
        loss = criterion(predicted_labels, labels).item()
        
        test_loss += loss
        num_correct += (predicted_labels == labels).sum().item()
        num_samples += predicted_labels.size(0)
        
        testHistory['loss'].append(loss)
        testHistory['accuracy'].append(float(num_correct) / float(num_samples))
        
        logger.log_batch(batch+1, loss, float(num_correct) / float(num_samples))
    
    test_accuracy = float(num_correct) / float(num_samples)
    test_loss = test_loss/len(dataHandler.test_dl) ## avg loss
    logger.finalize(test_loss, test_accuracy)

    plot_history(logger.name, False, testHistory)

  return test_loss, test_accuracy