from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import transformation
from torchvision import transforms
import torch
import numpy as np
import time

class CustomImageDataset(Dataset):
  def __init__(self, images, properties, transform=None):
    self.images = images
    self.properties = properties
    self.transform = transform
    self.transforms = transforms.Compose(
      [
        transformation.RandomRoll(),
        #transformation.RandomFlipLR(p=.5),
        #transformation.RandomFlipUD(p=0.5),
        #transformation.RandomTranspose(p=.5),
      ]
      )

  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    image = self.images[idx]
    properties = self.properties[idx]
    if self.transform:
      properties = transformation.vec_to_dict(properties.numpy())
      image, properties =  self.transforms({'image' : image, 'data' : properties}).values()
      properties = transformation.Q_to_vec(transformation.dict_to_Q(properties))
      properties = torch.from_numpy(np.asarray(properties, dtype=np.float32).copy())
      image = torch.from_numpy(np.asarray(image, dtype=np.float32).copy())
    return (image, properties[:6])


class Loader():
  def __init__(self, train_dataset, test_dataset, batch_size, train):
    self.train = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=train,drop_last=True)
    self.test = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=train,drop_last=True)