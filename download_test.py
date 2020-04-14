#!/usr/bin/env python
import requests
import json
import random
from tqdm import trange
import os.path
from os import path
import sys
import re
import numpy as np

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


f = open("coord_test.txt", "r")
test_cases = f.readlines()

cnt = 0
for tc in test_cases:
  tc = tc.replace('\n', '')
  tc = tc.split(' ')
  lat = float(tc[0].replace('\U00002013', '-'))
  lon = float(tc[1].replace('\U00002013', '-'))
  zoom = int(float(tc[2]))
  save_img(lat, lon, zoom, 'test/' + str(cnt) + '.jpeg')
  cnt += 1

import cv2
test_imgs = np.zeros((16,256,256,3), dtype=np.uint8)
for i in range(0,16):
  test_imgs[i] = cv2.imread('test/' + str(i) + '.jpeg')
test_imgs = test_imgs.reshape((4,4,256,256,3))
test_imgs = np.concatenate(test_imgs, axis=1)
test_imgs = np.concatenate(test_imgs, axis=1)
print(test_imgs.shape)
cv2.imwrite('test/grid.jpeg', test_imgs)
