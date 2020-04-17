#!/usr/bin/env python
import requests
import os.path


BASE_DIR = '/raid.dell1/world/'
MAX_ZOOM = 10


def save_img(req, save_path):
  image_url = 'http://ecn.t1.tiles.virtualearth.net/tiles/a' + req + '.jpeg?g=8384'

  receive = requests.get(image_url)
  with open(save_path,'wb') as f:
    f.write(receive.content)



def get_quarters(base_dir, base_req, depth):
  for q in range(4):
    if not os.path.isdir(base_dir + str(q)):
      os.mkdir(base_dir + str(q))
    if not os.path.exists(base_dir + str(q) + '/img.jpeg'):
      save_img(base_req + str(q), base_dir + str(q) + '/img.jpeg')
    print(base_req)
    if depth > 1:
      get_quarters(base_dir + str(q) + '/',
                   base_req + str(q),
                   depth - 1)


get_quarters(BASE_DIR, '', depth=9)
