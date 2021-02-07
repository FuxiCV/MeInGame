import logging
import os

import cv2
import numpy as np
import torch
from skimage import io


class Dataset(torch.utils.data.Dataset):

  def __init__(self, config, flist_gt=None, flist=None, test=False,
               device=torch.device('cpu')):
    super(Dataset, self).__init__()

    self.device = device

    self.len_flist = len(flist)
    self.flist = flist
    self.flist_gt = flist_gt
    self.data_dir = config.data_dir

    self.is_train = not test
    self.root_dir = config.root_dir
    self.im_size = config.im_size
    self.uv_size = config.uv_size

    temp_uv = io.imread(
        os.path.join(config.root_dir, 'data/uv_param/masks/skin_mask.png'))
    self.temp_uv = cv2.resize(temp_uv[..., :3], (self.uv_size, self.uv_size),
                              interpolation=cv2.INTER_LANCZOS4).astype(
                                  np.float32)

    self.lip_mask = 1 - self.load_mask('data/uv_param/masks/lip_mask.png')
    brow_mask = 1 - self.load_mask('data/uv_param/masks/brow_mask.png')
    ear_mask = 1 - self.load_mask('data/uv_param/masks/ear_mask.png')
    eye_mask = 1 - self.load_mask('data/uv_param/masks/eye_mask.png')
    self.nose_shadow_mask = self.load_mask(
        'data/uv_param/masks/nose_shadow_mask.png')
    skin_mask = self.load_mask('data/uv_param/masks/skin_mask.png')
    self.temp_skin_mean = np.mean(self.temp_uv[skin_mask.astype(np.bool)],
                                  axis=0)
    self.temp_lip_mean = np.mean(self.temp_uv[self.lip_mask.astype(np.bool)],
                                 axis=0)
    self.skin_mask = skin_mask + brow_mask + ear_mask + eye_mask
    self.blur_skin_mask = cv2.GaussianBlur(self.skin_mask.astype(
        np.float32), (self.uv_size // 16 + 1, self.uv_size // 16 + 1),
                                           self.uv_size // 32)[..., None]
    lip_mask = cv2.erode(self.lip_mask, np.ones((9, 9)), 4)
    self.blur_lip_mask = cv2.GaussianBlur(lip_mask.astype(np.float32), (61, 61),
                                          0)[..., None]

    uv_mask = io.imread(
        os.path.join(config.root_dir, 'data/uv_param/masks/uv_mask.png'))[...,
                                                                          -1]
    uv_mask = cv2.resize(uv_mask, (self.uv_size, self.uv_size),
                         interpolation=cv2.INTER_NEAREST)
    self.uv_mask = uv_mask[..., None] // 255

    self.log = logging.getLogger('x')

  def __len__(self):
    if self.is_train:
      return len(self.flist_gt)
    else:
      return len(self.flist)

  def __getitem__(self, index):
    try:
      item = self.load_item_numpy(index)
    except Exception as e:
      self.log.error('Loading Error')
      self.log.error(e)
      item = self.load_item_numpy(0)
    return item

  def load_item_numpy(self, index):
    if self.is_train:
      uv_gt_path = self.flist_gt[index]
      data_id = os.path.split(uv_gt_path)[-1][:5]
      uv_path = os.path.join(self.data_dir, '{}_uv.png'.format(data_id))
      rand_uv_path = self.flist[np.random.randint(self.len_flist)]
      rand_data_id = os.path.split(rand_uv_path)[-1][:5]
      rand_im_path = os.path.join(self.data_dir,
                                  '{}_image.png'.format(rand_data_id))
      rand_vert_path = os.path.join(self.data_dir,
                                    '{}_nsh_vert.npy'.format(rand_data_id))
      rand_param_path = os.path.join(self.data_dir,
                                     '{}_params.npy'.format(rand_data_id))
    else:
      uv_path = self.flist[index]
      data_id = os.path.split(uv_path)[-1][:5]
    im_path = os.path.join(self.data_dir, '{}_image.png'.format(data_id))
    vert_path = os.path.join(self.data_dir, '{}_nsh_vert.npy'.format(data_id))
    param_path = os.path.join(self.data_dir, '{}_params.npy'.format(data_id))

    vertice = self.to_tensor(np.load(vert_path))
    param = self.to_tensor(np.load(param_path))
    image = resize(io.imread(im_path), self.im_size)

    if self.is_train:
      image = self.process_image(image)
      rand_image = resize(io.imread(rand_im_path), self.im_size)
      rand_image = self.process_image(rand_image)

      uvmap = resize(io.imread(uv_path), self.uv_size)
      rand_uvmap = resize(io.imread(rand_uv_path), self.uv_size)
      uvmap = self.process_uvmap(uvmap, rand_uvmap)
      rand_uvmap = self.process_uvmap(rand_uvmap)

      rand_vertice = self.to_tensor(np.load(rand_vert_path))
      rand_param = self.to_tensor(np.load(rand_param_path))

      uvmap_gt = resize(io.imread(uv_gt_path), self.uv_size) / 127.5 - 1
      uvmap_gt = self.to_tensor(uvmap_gt[..., :3].astype(np.float32))

      return image.permute((2, 0, 1)), uvmap.permute(
          (2, 0, 1)), uvmap_gt.permute(
              (2, 0, 1)), vertice, param, rand_image.permute(
                  (2, 0, 1)), rand_uvmap.permute(
                      (2, 0, 1)), rand_vertice, rand_param
    else:
      uvmap = resize(io.imread(uv_path), self.uv_size)
      uvmap = self.process_uvmap(uvmap)
      image = self.process_image(image, False)
      return image.permute((2, 0, 1)), uvmap.permute((2, 0, 1)), vertice, param

  def process_uvmap(self, uvmap, rand_uvmap=None, dark_brow=False):
    # uvmap dtype should be uint8
    rule_mask = get_rule_mask(uvmap)
    uv_mask = uvmap[..., -1:] > 127
    uv_seg = uvmap[..., -1] % 128
    uv_seg = np.eye(19, dtype=np.float32)[uv_seg]
    mask_idx = np.r_[1:10]
    uv_mask = uv_mask.astype(np.float32) * np.sum(uv_seg[..., mask_idx],
                                                  axis=-1, keepdims=True)
    uv_mask = uv_mask * self.uv_mask

    if rand_uvmap is not None:
      rand_uvmask = rand_uvmap[..., -1:] > 127
      rand_uvseg = rand_uvmap[..., -1:] % 128
      rand_uvmask = rand_uvmask * (rand_uvseg > 0) * (rand_uvseg < 10)
      uv_mask = uv_mask * rand_uvmask

    uv_mask_small = cv2.dilate(
        uv_mask, np.ones((self.uv_size // 32 + 1, self.uv_size // 32 + 1)), 2)
    uv_mask_small = cv2.erode(
        uv_mask_small, np.ones(
            (self.uv_size // 16 + 1, self.uv_size // 16 + 1)), 2)

    if not self.is_train:
      uv_mask_bottom = cv2.erode(
          uv_mask_small,
          np.ones((self.uv_size // 16 + 1, self.uv_size // 16 + 1)), 3)
      uv_mask_small[self.uv_size // 2:] = uv_mask_bottom[self.uv_size // 2:]

    mouth_idx = 8
    uv_mask = uv_mask_small * uv_mask[..., 0] * (1 - uv_seg[..., mouth_idx])

    mask_for_seam = 255 * uv_mask.astype(np.uint8)
    mask_for_seam[1, 1] = 255
    mask_for_seam[1, -2] = 255
    mask_for_seam[-2, 1] = 255
    mask_for_seam[-2, -2] = 255

    uv_mean = np.mean(uvmap[uv_seg[..., 1].astype(np.bool), :3],
                      axis=0).astype(np.float32)
    temp_uv = self.temp_uv - self.temp_skin_mean + uv_mean
    temp_uv = temp_uv * self.blur_skin_mask + self.temp_uv * (
        1 - self.blur_skin_mask)

    lip_idx = np.r_[7, 9]
    lip_mask = np.sum(uv_seg[..., lip_idx], axis=-1)
    lip_mask = lip_mask * rule_mask

    if not np.any(lip_mask):
      lip_mask = self.lip_mask
    lip_mean = np.mean(uvmap[lip_mask.astype(np.bool), :3],
                       axis=0).astype(np.float32)
    lip_uv = self.temp_uv.astype(np.int32)
    lip_uv += np.round(lip_mean - self.temp_lip_mean).astype(np.int32)
    temp_uv = lip_uv * self.blur_lip_mask + temp_uv * (1 - self.blur_lip_mask)

    temp_uv = np.clip(temp_uv, 0, 255).astype(np.uint8)
    k_size = (self.uv_size // 16 + 1, self.uv_size // 16 + 1)
    blur_uv_mask = cv2.GaussianBlur(uv_mask, k_size, 0)[..., None]

    use_seamless = True
    if use_seamless:
      # ! cv::NORMAL_CLONE, cv::MIXED_CLONE or cv::MONOCHROME_TRANSFER
      fused = cv2.seamlessClone(uvmap[..., :3], temp_uv, mask_for_seam,
                                (self.uv_size // 2, self.uv_size // 2),
                                cv2.MIXED_CLONE)
    else:
      fused = uvmap[..., :3] * blur_uv_mask + temp_uv * (1 - blur_uv_mask)
    if dark_brow:
      brow_mask = (uv_seg[..., 2] + uv_seg[..., 3]) * self.uv_mask[..., 0]
      k_size = (self.uv_size // 32 + 1, self.uv_size // 32 + 1)
      blur_brow_mask = cv2.GaussianBlur(brow_mask, k_size, 0)[..., None]
      fused = fused * (1 - blur_brow_mask * 0.4)
      fused = fused * 0.9

    uvmap = np.concatenate([(fused / 127.5 - 1).astype(np.float32),
                            blur_uv_mask.astype(np.float32)], axis=-1)
    uvmap = self.to_tensor(uvmap)

    return uvmap

  def process_image(self, image, mask=True):
    im_seg = 255 - image[..., -1:]
    image = image[..., :3].astype(np.float32) / 127.5 - 1
    if mask:
      im_seg_oh = np.eye(19, dtype=np.float32)[im_seg[..., 0]]
      # include face, ears and neck, exclude inner mouth
      skin_idx = np.r_[1:8, 9, 12, 13, 17]
      im_skin = np.sum(im_seg_oh[..., skin_idx], axis=-1, keepdims=True)
      k_size = (self.im_size // 32 + 1, self.im_size // 32 + 1)
      blur_im_skin = cv2.GaussianBlur(im_skin, k_size, 0)[..., None]
      image = np.concatenate([image, blur_im_skin, im_seg], axis=-1)
    else:
      image = np.concatenate([image, im_seg], axis=-1)

    image = self.to_tensor(image)
    return image

  def to_tensor(self, data, im_size=None):
    if im_size is not None:
      data = resize(data, im_size)
    return torch.from_numpy(data).to(self.device)

  def create_iterator(self, batch_size):
    while True:
      sample_loader = torch.utils.data.DataLoader(dataset=self,
                                                  batch_size=batch_size,
                                                  drop_last=True)

      for item in sample_loader:
        yield item

  def load_mask(self, path):
    mask = io.imread(os.path.join(self.root_dir, path))[..., -1]
    if self.uv_size != 1024:
      mask = cv2.resize(mask, (self.uv_size, self.uv_size),
                        interpolation=cv2.INTER_NEAREST)
    return mask // 255


def resize(image, im_size):
  if image.shape[0] != im_size or image.shape[1] != im_size:
    image = cv2.resize(image, (im_size, im_size),
                       interpolation=cv2.INTER_NEAREST)
  return image


def get_center(mask):
  rows = np.any(mask, axis=1)
  cols = np.any(mask, axis=0)
  rmin, rmax = np.where(rows)[0][[0, -1]]
  cmin, cmax = np.where(cols)[0][[0, -1]]

  return ((cmax + cmin) // 2, (rmax - rmin) // 2)


def get_rule_mask(image):
  R = image[..., 0]
  G = image[..., 1]
  B = image[..., 2]
  mask = (R > 95) & (G > 40) & (B > 20) & ((
      np.max(image, axis=-1) - np.min(image, axis=-1)) > 15) & (R > G) & (R > B)
  # mask = (R > 95) & (G > 40) & (B > 20) & (
  #     (np.max(image, axis=-1) - np.min(image, axis=-1)) >
  #     15) & ((R - G) > 20) & ((R - B) > 20)
  return mask
