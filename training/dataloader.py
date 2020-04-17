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


TILE_SIZE = 256
BASE_DIR = '/raid.dell1/world/'


def pil_loader(path):
  # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
  with open(path, 'rb') as f:
    img = Image.open(f)
    return np.array(img.convert('RGB'))


def xy_from_geodetic(lat, lon):
  lat = np.clip(lat, -85.05, 85.05)
  lon = np.clip(lon, -179.9999999, 179.9999999)
  x = lon/360.0 + .5
  sinlat = np.sin(np.radians(lat))
  y = 0.5 - np.log((1 + sinlat) / (1 - sinlat)) / (4 * np.pi)
  return x,y


def get_tile_idx(x, y, zoom):
  if (not 0 <= x < 1.0) or ((not 0 <= y < 1.0)):
    return 'NULL'
  if zoom == 0:
    return ''
  if x >= .5 and y >= .5:
    q = '3'
  elif y >= .5:
    q = '2'
  elif x >= .5:
    q = '1'
  else:
    q = '0'
  return q + get_tile_idx(x*2 % 1, y*2 % 1, zoom - 1)


def get_tile_pos(x, y, zoom):
  print(x,y)
  if zoom == 0:
    return x, y
  return get_tile_pos(x*2 % 2, y*2 % 2, zoom - 1)


def get_tile(idx):
  if idx == 'NULL':
    return np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
  sub_path = ''.join([a + '/' for a in idx])
  path = BASE_DIR + sub_path + 'img.jpeg'
  return pil_loader(path)
 

def get_custom_tile(x, y, zoom):
  assert 0 <= x < 1
  assert 0 <= y < 1
  assert zoom >= 1
  zoom = int(zoom)
  tiles = []
  delta = (.5)**(zoom + 1)
  for x_off in [-delta, delta]:
    tiles.append([])
    for y_off in [-delta, delta]:
      tiles[-1].append(get_tile(get_tile_idx(x + x_off, y + y_off, zoom)))
    tiles[-1] = np.vstack(tiles[-1])
  tiles = np.hstack(tiles)
  #x_pos, y_pos = get_tile_pos(x,y,zoom)
  for i in range(zoom):
    x = (x - .25) * 2 % 1.0 + .5
    y = (y - .25) * 2 % 1.0 + .5
  x_pix = int((x * TILE_SIZE))
  y_pix = int((y * TILE_SIZE))
  return tiles[y_pix - TILE_SIZE//2: y_pix + TILE_SIZE//2,
               x_pix - TILE_SIZE//2: x_pix + TILE_SIZE//2]


def get_custom_tile_geodetic(lat, lon, zoom):
  x,y = xy_from_geodetic(lat, lon)
  return get_custom_tile(x, y, zoom)


class SatImageDataset(Dataset):
  def __init__(self, transform=None, size=100000, max_zoom=8):
    self.size = size
    self.transform = transform
    self.samples = np.random.uniform(0.0, 1.0, size=(self.size, 3))
    self.samples[:,2] = np.floor(max_zoom*self.samples[:,2]) + 1.0

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    sample = self.samples[idx]

    image = Image.fromarray(get_custom_tile(*sample))
    if self.transform:
      image = self.transform(image)
    meta = torch.FloatTensor(sample)

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
                    transform=transforms.Compose([
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
