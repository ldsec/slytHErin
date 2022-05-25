from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data.dataloader import DataLoader
import argparse
import numpy as np
from utils import single_to_multi_label
import torch.nn.functional as F
import json

class DataHandler():
  def __init__(self, dataset : str, batch_size : int, shuffle=True):
    if dataset == "MNIST":
      self.batch_size = batch_size
      if batch_size == None:
        drop_last = False
      else:
        drop_last = True
      
      ## images are PIL in range [0,1]
      ## make mean = 0 and std = 1
      transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
      train_ds = MNIST("data/", train=True, download=True, transform=transform)
      test_ds = MNIST("data/", train=False, download=True, transform=transform)
      self.train_dl = DataLoader(train_ds, batch_size = batch_size, shuffle=shuffle, drop_last=drop_last,num_workers=2, pin_memory=True)
      self.test_dl = DataLoader(test_ds, batch_size = batch_size, shuffle=shuffle, drop_last=drop_last,num_workers=2, pin_memory=True)

class DataHandlerAlex():
  def __init__(self, dataset : str, batch_size : int, shuffle=True):
    if dataset == "MNIST":
      self.batch_size = batch_size
      if batch_size == None:
        drop_last = False
      else:
        drop_last = True
      normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
      to_rgb = transforms.Lambda(lambda image: image.convert('RGB'))
      resize = transforms.Resize((227, 227))
      
      transform = transforms.Compose([resize, to_rgb, transforms.ToTensor(), normalize])

      train_ds = MNIST("data/", train=True, download=True, transform=transform)
      test_ds = MNIST("data/", train=False, download=True, transform=transform)

      self.train_dl = DataLoader(train_ds, batch_size = batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=2, pin_memory=True)
      self.test_dl = DataLoader(test_ds, batch_size = batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=2, pin_memory=True)

class DataHandlerNN():
  """
    Specific loader for NN
    Uses Zama validation set of .npz samples
  """
  def __init__(self, path : str, batchsize: int):
      self.path = path
      self.batchsize = batchsize
      self.data = []
      self.num_samples = 1000
      self.num_batches = self.num_samples // batchsize
      self.idx = 0
      
      for idx in range(self.num_samples):
        self.data.append(np.load(f"{path}/sample_{idx}.npz")["arr_0"].reshape(1,28,28))

      with open(f"{path}/expected_results.txt", "r", encoding="utf-8") as f:
        labels = f.readlines()
      self.labels = list(map(int,labels))

  def batch(self):
    while self.idx < self.num_batches:
      yield (self.data[self.idx*self.batchsize:(self.idx+1)*self.batchsize],
            self.labels[self.idx*self.batchsize:(self.idx+1)*self.batchsize])
      self.idx += 1


if __name__=="__main__":
  
  ## return the data for evaluation

  parser = argparse.ArgumentParser()
  parser.add_argument("--model", help="simplenet,alexnet or nn")
  
  args = parser.parse_args()
  if args.model == "simplenet" or args.model == "nn":
    dataHandler = DataHandler("MNIST", None, shuffle = False)
    dataset = {'X':[], 'Y':[]}
    for data,label in dataHandler.test_dl:
      if args.model == "simplenet":
        data = F.pad(data, (1,0,1,0)).numpy().flatten()
      else:
        #data = F.pad(data, (1,1,1,1)).numpy().flatten()
        data = data.numpy().flatten() #no pad if go training
      sample = [x.item() for x in data] 
      dataset['X'].append(sample)
      dataset['Y'].append(label)
    with open(f'./data/{args.model}_data.json','w') as f:
      json.dump(dataset,f)

