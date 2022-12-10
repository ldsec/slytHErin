from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data.dataloader import DataLoader
import argparse
import numpy as np
from utils import single_to_multi_label
import torch.nn.functional as F
import json

class DataHandler():
  #loads data for training
  def __init__(self, dataset : str, batch_size : int, shuffle=True, scale=True):
    if dataset == "MNIST":
      self.batch_size = batch_size
      if batch_size == None:
        drop_last = False
      else:
        drop_last = True
      
      ## images are PIL in range [0,1]
      ## make mean = 0 and std = 1
      if scale:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
      else:
        transform = transforms.Compose([transforms.ToTensor()])
      train_ds = MNIST("data/", train=True, download=True, transform=transform)
      test_ds = MNIST("data/", train=False, download=True, transform=transform)
      self.train_dl = DataLoader(train_ds, batch_size = batch_size, shuffle=shuffle, drop_last=drop_last,num_workers=2, pin_memory=True)
      self.test_dl = DataLoader(test_ds, batch_size = batch_size, shuffle=shuffle, drop_last=drop_last,num_workers=2, pin_memory=True)

    if dataset == "CIFAR":
      self.batch_size = batch_size
      if batch_size == None:
        drop_last = False
      else:
        drop_last = True
      transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
      ## images are PIL in range [0,1]
      train_ds = CIFAR10("data/", train=True, download=True, transform=transform)
      test_ds = CIFAR10("data/", train=False, download=True, transform=transform)
      self.train_dl = DataLoader(train_ds, batch_size = batch_size, shuffle=shuffle, drop_last=drop_last,num_workers=2, pin_memory=True)
      self.test_dl = DataLoader(test_ds, batch_size = batch_size, shuffle=shuffle, drop_last=drop_last,num_workers=2, pin_memory=True)

if __name__=="__main__":
  
  ## return the data for evaluation

  parser = argparse.ArgumentParser()
  parser.add_argument("--model", help="cryptonet or nn")
  
  args = parser.parse_args()
  if args.model == "cryptonet" or args.model == "nn":
    scale = True
    nopad = ""
    if args.model == "cryptonet":
      nopad = "_nopad"

    dataHandler = DataHandler("MNIST", None, shuffle = False, scale=scale)
    dataset = {'X':[], 'Y':[]}
    for data,label in dataHandler.test_dl:
      data = data.numpy().flatten()
      sample = [x.item() for x in data] 
      dataset['X'].append(sample)
      dataset['Y'].append(label)

    with open(f'./data/{args.model}_data{nopad}.json','w') as f:
      json.dump(dataset,f)

