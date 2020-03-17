#!/usr/bin/env python
import requests
import json
import random
from tqdm import trange
import os.path
from os import path
import sys

if len(sys.argv) < 3:
  raise RuntimeError(" specify base dir and zoom level")

base_dir = sys.argv[1]
zoom = int(sys.argv[2])
if base_dir[-1] != '/':
  base_dir += '/'


with open('api_key', 'r') as file:
  key = file.read().replace('\n', '')


def save_img(latitude, longitude, zoom, save_path):
  api_request = ('https://dev.virtualearth.net/REST/v1/Imagery/Metadata/Aerial/' +
                  str(latitude) + ',' + str(longitude) + '?zoomLevel=' + str(zoom) +
                  '&key=' + key)
  r =requests.get(api_request)
  dic = json.loads(r.text)
  print(dic)
  image_url = dic['resourceSets'][0]['resources'][0]['imageUrl']

  receive = requests.get(image_url)
  with open(save_path,'wb') as f:
    f.write(receive.content)


directory = base_dir + '/zoom_' + str(zoom) + '/'
try:
  os.mkdir(directory)
except FileExistsError as e:
  pass

for i in trange(10*(4**zoom)):
  file_name = directory + str(i)
  if path.exists(file_name + '.txt'):
    continue
  lat = (180*random.random() - 90)
  lon = (360*random.random() - 180)
  with open(file_name + '.txt','w') as f:
    f.write(str(lat) + ', ' + str(lon) + ', ' + str(zoom))
  save_img(lat, lon, zoom, file_name + '.jpeg')
