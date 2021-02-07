import logging
import os
from collections import OrderedDict

import cv2
import h5py
import numpy as np
import scipy.io as sio
# import tensorflow as tf
import torch
from PIL import Image


def init_logger(name='x', filename='log.txt'):
  logger = logging.getLogger(name)
  logger.setLevel(logging.DEBUG)
  formatter = logging.Formatter(
      '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
      datefmt='%m-%d %H:%M:%S')

  fh = logging.FileHandler(filename, encoding='utf-8')
  fh.setLevel(logging.INFO)
  fh.setFormatter(formatter)

  ch = logging.StreamHandler()
  ch.setLevel(logging.INFO)
  ch.setFormatter(formatter)

  logger.addHandler(ch)
  logger.addHandler(fh)

  return logger


class BFM_model(object):

  def __init__(self, root_dir, path):
    super(BFM_model, self).__init__()

    self.root_dir = root_dir
    self.path = os.path.join(root_dir, path)
    if '09' in path:
      self.load_BFM09()
    elif '17' in path:
      self.load_BFM17()

    self.n_shape_coef = self.shapePC.shape[1]
    self.n_exp_coef = self.expressionPC.shape[1]
    self.n_color_coef = self.colorPC.shape[1]
    self.n_all_coef = self.n_shape_coef + self.n_exp_coef + self.n_color_coef

  def load_BFM09(self):
    model = sio.loadmat(self.path)
    self.shapeMU = model['meanshape'].astype(np.float32)  # mean face shape
    self.shapePC = model['idBase'].astype(np.float32)  # identity basis
    self.expressionPC = model['exBase'].astype(np.float32)  # expression basis
    self.colorMU = model['meantex'].astype(np.float32)  # mean face texture
    self.colorPC = model['texBase'].astype(np.float32)  # texture basis
    self.point_buf = model['point_buf'].astype(np.int32)
    # adjacent face index for each vertex, starts from 1 (only used for calculating face normal)
    self.triangles = model['tri'].astype(np.int32)
    # vertex index for each triangle face, starts from 1
    self.landmark = np.squeeze(model['keypoints']).astype(
        np.int32) - 1  # 68 face landmark index, starts from 0

  def load_BFM17(self):
    with h5py.File(self.path, 'r') as hf:
      self.triangles = np.transpose(np.array(hf['shape/representer/cells']),
                                    [1, 0])

      self.shapeMU = np.array(hf['shape/model/mean']) / 1e2
      shape_orthogonal_pca_basis = np.array(hf['shape/model/pcaBasis'])
      shape_pca_variance = np.array(hf['shape/model/pcaVariance']) / 1e4

      self.colorMU = np.array(hf['color/model/mean'])
      color_orthogonal_pca_basis = np.array(hf['color/model/pcaBasis'])
      color_pca_variance = np.array(hf['color/model/pcaVariance'])

      self.expressionMU = np.array(hf['expression/model/mean']) / 1e2
      expression_pca_basis = np.array(hf['expression/model/pcaBasis'])
      expression_pca_variance = np.array(
          hf['expression/model/pcaVariance']) / 1e4

      self.shapePC = shape_orthogonal_pca_basis * np.expand_dims(
          np.sqrt(shape_pca_variance), 0)
      self.colorPC = color_orthogonal_pca_basis * np.expand_dims(
          np.sqrt(color_pca_variance), 0)
      self.expressionPC = expression_pca_basis * np.expand_dims(
          np.sqrt(expression_pca_variance), 0)


def stitch_images(inputs, *outputs, im_size, img_per_row=2):
  gap = 5
  columns = len(outputs) + 1

  # width, height = inputs[0][:, :, 0].shape
  try:
    width = im_size[0]
    height = im_size[1]
  except TypeError:
    width = im_size
    height = im_size

  img = Image.new('RGB',
                  (width * img_per_row * columns + gap *
                   (img_per_row - 1), height * int(len(inputs) / img_per_row)))
  images = [inputs, *outputs]

  # for ix in range(len(inputs)):
  for ix, _ in enumerate(inputs):
    xoffset = int(ix % img_per_row) * width * columns + int(
        ix % img_per_row) * gap
    yoffset = int(ix / img_per_row) * height

    # for cat in range(len(images)):
    for cat, _ in enumerate(images):
      im = np.array((images[cat][ix]).cpu()).astype(np.uint8).squeeze()
      im = Image.fromarray(im)
      if im.size[0] != width:
        im = im.resize((width, height))
      img.paste(im, (xoffset + cat * width, yoffset))

  return img


def to_uint8(image):
  if np.min(image) < 0:
    return np.round((np.clip(image, -1, 1) + 1) * 127.5).astype(np.uint8)
  else:
    return np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)


def fix_state_dict(state_dict):
  new_state_dict = OrderedDict()
  for k, v in state_dict.items():
    name = k.replace('module.', '')
    new_state_dict[name] = v
  return new_state_dict


def to_uint8_torch(image, no_permute=False):
  if torch.min(image) < 0:
    image = torch.round((torch.clamp(image, -1., 1.) + 1) * 127.5)
  else:
    image = torch.round(torch.clamp(image, 0, 1) * 255)
  # return image.uint8().permute(0, 2, 3, 1)
  if no_permute:
    return image.type(torch.uint8)
  elif len(image.size()) == 4:
    return image.type(torch.uint8).permute(0, 2, 3, 1)
  elif len(image.size()) == 3:
    return image.type(torch.uint8).permute(1, 2, 0)
  else:
    return image.type(torch.uint8)


def center_crop_resize(image, img_size, crop=False):
  # set img_size to None will not resize image
  height, width, _ = image.shape
  if crop:
    if width > height:
      w_s = (width - height) // 2
      image = image[:, w_s:w_s + height]
    elif height > width:
      h_s = (height - width) // 2
      image = image[h_s:h_s + width, :]
  else:
    # padding
    if width > height:
      top = (width - height) // 2
      bottom = width - height - top
      image = cv2.copyMakeBorder(image, top, bottom, 0, 0, cv2.BORDER_REPLICATE)
    elif height > width:
      left = (height - width) // 2
      right = height - width - left
      image = cv2.copyMakeBorder(image, 0, 0, left, right, cv2.BORDER_REPLICATE)

  if img_size is not None:
    image = cv2.resize(image, (img_size, img_size))
  return image


def split_bfm09_coeff(coeff):
  shape_coef = coeff[:, 0:80]  # identity(shape) coeff of dim 80
  exp_coef = coeff[:, 80:144]  # expression coeff of dim 64
  color_coef = coeff[:, 144:224]  # texture(albedo) coeff of dim 80
  angles = coeff[:, 224:227]  # ruler angles(x,y,z) for rotation of dim 3
  gamma = coeff[:, 227:254]  # lighting coeff for 3 channel SH of dim 27
  translation = coeff[:, 254:257]  # translation coeff of dim 3
  return shape_coef, exp_coef, color_coef, angles, gamma, translation


def rotation_matrix_np(angles):
  angle_x = angles[:, 0]
  angle_y = angles[:, 1]
  angle_z = angles[:, 2]

  ones = np.ones_like(angle_x)
  zeros = np.zeros_like(angle_x)

  # yapf: disable
  rotation_X = np.array([[ones, zeros, zeros],
                         [zeros, np.cos(angle_x), -np.sin(angle_x)],
                         [zeros, np.sin(angle_x), np.cos(angle_x)]],
                        dtype=np.float32)
  rotation_Y = np.array([[np.cos(angle_y), zeros, np.sin(angle_y)],
                         [zeros, ones, zeros],
                         [-np.sin(angle_y), zeros, np.cos(angle_y)]],
                        dtype=np.float32)
  rotation_Z = np.array([[np.cos(angle_z), -np.sin(angle_z), zeros],
                         [np.sin(angle_z), np.cos(angle_z), zeros],
                         [zeros, zeros, ones]],
                        dtype=np.float32)
  # yapf: enable

  rotation_X = np.transpose(rotation_X, (2, 0, 1))
  rotation_Y = np.transpose(rotation_Y, (2, 0, 1))
  rotation_Z = np.transpose(rotation_Z, (2, 0, 1))
  rotation = np.matmul(np.matmul(rotation_Z, rotation_Y), rotation_X)
  # transpose row and column (dimension 1 and 2)
  rotation = np.transpose(rotation, (0, 2, 1))

  return rotation


def rgb2hsv(im, eps=1e-8):
  img = im * 0.5 + 0.5
  # import imageio
  # imageio.imsave('tmp/blur_uv.png', img.cpu()[0].permute(1,2,0).detach().numpy())

  hue = torch.Tensor(im.shape[0], im.shape[2], im.shape[3]).to(im.device)
  hue[img[:, 2] == img.max(1)[0]] = 4.0 + (
      (img[:, 0] - img[:, 1]) /
      (img.max(1)[0] - img.min(1)[0] + eps))[img[:, 2] == img.max(1)[0]]
  hue[img[:, 1] == img.max(1)[0]] = 2.0 + (
      (img[:, 2] - img[:, 0]) /
      (img.max(1)[0] - img.min(1)[0] + eps))[img[:, 1] == img.max(1)[0]]
  hue[img[:, 0] == img.max(1)[0]] = (0.0 + (
      (img[:, 1] - img[:, 2]) /
      (img.max(1)[0] - img.min(1)[0] + eps))[img[:, 0] == img.max(1)[0]]) % 6
  hue[img.min(1)[0] == img.max(1)[0]] = 0.0
  hue = hue / 6

  saturation = (img.max(1)[0] - img.min(1)[0]) / (img.max(1)[0] + eps)
  saturation[img.max(1)[0] == 0] = 0

  value = img.max(1)[0]

  hsv = torch.stack([hue, saturation, value], dim=-3)

  return hsv
