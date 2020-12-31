import os
from os.path import join
import time
import random
import PIL
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py

def load_bboxes(start, end):
  bboxes_path = "/content/drive/My Drive/list_bbox_celeba.csv"

  bbox_dict = {}
  bbox_df = pd.read_csv(bboxes_path)

  for i in range(start, end):
    bbox_obj = bbox_df.iloc[i, :]
    image_id = bbox_obj.image_id
    x1 = int(bbox_obj.x1)
    y1 = int(bbox_obj.y1)
    width = int(bbox_obj.w)
    height = int(bbox_obj.h)
    attribute_list = [x1, y1, width, height]
    bbox_dict[image_id] = attribute_list

  return bbox_dict

def crop_img(image_path, bbox, image_size):
  img = Image.open(image_path).convert('RGB')
  width, height = img.size
  R = int(np.maximum(bbox[2], bbox[3]) * 0.75)
  x_centre = int((bbox[0] + bbox[0] + bbox[2]) / 2)
  y_centre = int((bbox[1] + bbox[1] + bbox[3]) / 2)
  x1 = np.maximum(0, x_centre - R)
  y1 = np.maximum(0, y_centre - R)
  x2 = np.minimum(width, x_centre + R)
  y2 = np.minimum(height, y_centre + R)
  img = img.crop([x1, y1, x2, y2])
  img = img.resize(image_size, PIL.Image.BILINEAR)
  return img

def load_dataset(images_path, data_dir, skipthought_encodings_path, image_size, start, end):

  with open(images_path) as f:
    images_list = f.read().split('\n')

  bounding_boxes = load_bboxes(start, end)

  #print(bounding_boxes)
  h = h5py.File(skipthought_encodings_path)
  
  skipthought_encodings = h['vectors'][:]

  X = []
  
  for i in range(len(images_list)):
    if(i % 1000 == 0):
      print(i)
    #print(i)
    image = images_list[i]
    bbox = bounding_boxes[image]
    image_path = join(data_dir, image)
    image = crop_img(image_path, bbox, image_size)
    X.append(np.array(image))
  
  X = np.array(X)
  print(X.shape, X.shape[0])
  return X, skipthought_encodings









