import numpy as np
import pandas as pd
import os
def _load_image_set_idx(_data_root_path,_image_set):
    image_set_file = os.path.join(
        _data_root_path, 'ImageSets', _image_set+'.txt')
    assert os.path.exists(image_set_file), \
        'File does not exist: {}'.format(image_set_file)

    with open(image_set_file) as f:
      image_idx = [x.strip() for x in f.readlines()]
    return image_idx
    
def _get_idx_path(idx, img_path):
    image_path = os.path.join(img_path, idx+'.png')
    assert os.path.exists(image_path), \
        'Image does not exist: {}'.format(image_path)
    return image_path

def _read_annotation(idx, annotation_path):
    def _get_obj_level(obj):
      height = float(obj[7]) - float(obj[5]) + 1
      trucation = float(obj[1])
      occlusion = float(obj[2])
      if height >= 40 and trucation <= 0.15 and occlusion <= 0:
          return 1
      elif height >= 25 and trucation <= 0.3 and occlusion <= 1:
          return 2
      elif height >= 25 and trucation <= 0.5 and occlusion <= 2:
          return 3
      else:
          return 4
    idx2annotation = {}
    for index in idx:
      filename = os.path.join(annotation_path, index+'.txt')
      with open(filename, 'r') as f:
        lines = f.readlines()
      f.close()
      bboxes = []
      for line in lines:
        obj = line.strip().split(' ')
        cls = obj[0].lower().strip()
        if _get_obj_level(obj) > 3:
          continue
        xmin = float(obj[4])
        ymin = float(obj[5])
        xmax = float(obj[6])
        ymax = float(obj[7])
        assert xmin >= 0.0 and xmin <= xmax, \
            'Invalid bounding box x-coord xmin {} or xmax {} at {}.txt' \
                .format(xmin, xmax, index)
        assert ymin >= 0.0 and ymin <= ymax, \
            'Invalid bounding box y-coord ymin {} or ymax {} at {}.txt' \
                .format(ymin, ymax, index)
        #x, y, w, h = bbox_transform_inv([xmin, ymin, xmax, ymax])
        #bboxes.append([x, y, w, h, cls])
        bboxes.append([xmin, ymin, xmax, ymax, cls])
      idx2annotation[index] = bboxes

    return idx2annotation