import os
import torch as torch
import numpy as np
from io import BytesIO
import scipy.misc
#import tensorflow as tf
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from skimage import io


def pil_loader(path):
  # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
  with open(path, 'rb') as f:
    img = Image.open(f)
    return img.convert('RGB')


class SatImageDataset(Dataset):
  def __init__(self, root='/raid.dell2/world', transform=None):
    self.root_dir = root
    self.transform = transform
    zooms = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
    self.samples = []
    for zoom in zooms[:1]:
      zoom_path = os.path.join(self.root_dir, zoom)
      for file_name in tqdm(os.listdir(zoom_path)):
        if file_name[-4:] == '.txt':
          meta_name = file_name
          img_name = file_name[:-4] + '.jpeg'
          meta_path = os.path.join(zoom_path, meta_name)
          img_path = os.path.join(zoom_path, img_name)
          if os.path.isfile(img_path):
            self.samples.append([img_path, meta_path])

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    image = pil_loader(self.samples[idx][0])
    with open(self.samples[idx][1]) as f:
      meta = f.read().split(',')
      meta = [float(x) for x in meta]

    if self.transform:
      image = self.transform(image)
      image = image.mul(2).add(-1)

    sample = {'image': image, 'meta': meta}

    return sample



class dataloader:
  def __init__(self, config):
    self.root = config.train_data_root
    self.batch_table = {4:32, 8:32, 16:32, 32:16, 64:16, 128:16, 256:12, 512:3, 1024:1}  # change this according to available gpu memory.
    self.batchsize = int(self.batch_table[pow(2,2)])        # we start from 2^2=4
    self.imsize = int(pow(2,2))
    self.num_workers = 0

  def renew(self, resl):
    print('[*] Renew dataloader configuration, load data from {}.'.format(self.root))
    self.batchsize = int(self.batch_table[pow(2,resl)])
    self.imsize = int(pow(2,resl))
    self.dataset = SatImageDataset(
                    root=self.root,
                    transform=transforms.Compose(   [
                                                    transforms.Resize(size=(self.imsize,self.imsize), interpolation=Image.NEAREST),
                                                    transforms.ToTensor(),
                                                    ]))

    self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batchsize,
            shuffle=True,
            num_workers=self.num_workers
        )

  def __iter__(self):
    return iter(self.dataloader)

  def __next__(self):
    return next(self.dataloader)

  def __len__(self):
    return len(self.dataloader.dataset)

  def get_batch(self):
    dataIter = iter(self.dataloader)
    return next(dataIter)
