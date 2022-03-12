from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data.dataloader import DataLoader

class DataHandler():
  def __init__(self, dataset : str, batch_size : int):
    if dataset == "MNIST":
      self.batch_size = batch_size
      
     
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
      
    train_ds = MNIST("data/", train=True, download=True, transform=transform)
    test_ds = MNIST("data/", train=False, download=True, transform=transform)
    self.train_dl = DataLoader(train_ds, batch_size = batch_size, shuffle=True, drop_last=True,num_workers=2, pin_memory=True)
    self.test_dl = DataLoader(test_ds, batch_size = batch_size, shuffle=True, drop_last=True,num_workers=2, pin_memory=True)

class DataHandlerAlex():
  def __init__(self, dataset : str, batch_size : int):
    if dataset == "MNIST":
      self.batch_size = batch_size
      
      normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
      to_rgb = transforms.Lambda(lambda image: image.convert('RGB'))
      resize = transforms.Resize((227, 227))
      
      transform = transforms.Compose([resize, to_rgb, transforms.ToTensor(), normalize])

      train_ds = MNIST("data/", train=True, download=True, transform=transform)
      test_ds = MNIST("data/", train=False, download=True, transform=transform)

      self.train_dl = DataLoader(train_ds, batch_size = batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
      self.test_dl = DataLoader(test_ds, batch_size = batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)

dataHandler = DataHandler("MNIST", 128)