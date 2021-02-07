import logging
import os
import random
from glob import glob

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.ops.points_alignment import corresponding_points_alignment
from pytorch3d.renderer import (MeshRasterizer, OpenGLPerspectiveCameras,
                                RasterizationSettings, look_at_view_transform)
from pytorch3d.structures import Meshes
from pytorch3d.transforms import Transform3d
from scipy.spatial.transform import Rotation
from skimage import io
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import utils
from lib import meshio
from lib.dataset import Dataset
from lib.deep3d import Deep3DFace
from lib.face_segment import Segment
from lib.image_cropper import ImageCropper
from lib.rbf import Shape_Transfer
from lib.uv_creator import UVCreator
from models import InpaintingModel


class UVInpainting():

  def __init__(self, config, device, sess=None, graph=None):
    '''
    if game_lm is not None, the result (mesh obj and UV texture map)
    will be convert from nsh to the game
    '''
    self.config = config
    self.name = config.name
    self.device = device
    self.sess = sess
    self.graph = graph
    self.log = logging.getLogger('x')
    self.rot_order = 'XYZ'

    self.debug = config.debug
    self.ex_idx = [4, 5, 8]

    self.inpaint_model = InpaintingModel(config, device, self.rot_order,
                                         debug=self.debug).to(device)
    # self.inpaint_model = InpaintingModel(config, device, self.debug)
    self.epoch = 0
    if config.restore:
      self.epoch = self.inpaint_model.load()
    # self.phase = config.phase

    if config.mode == 'train':
      num_test = 2048
      flist = glob(os.path.join(config.data_dir, '*_uv.png'))
      random.shuffle(flist)
      train_flist = flist[:-2 * num_test]
      val_flist = flist[-2 * num_test:-num_test]
      test_flist = flist[-num_test:]

      num_test = 300
      flist_gt = glob(os.path.join(config.data_gt_dir, '*_uv*.png'))
      random.shuffle(flist_gt)
      train_flist_gt = flist_gt[:-2 * num_test]
      val_flist_gt = flist_gt[-2 * num_test:-num_test]
      test_flist_gt = flist_gt[-num_test:]

      self.train_dataset = Dataset(config, train_flist_gt, train_flist)
      self.val_dataset = Dataset(config, val_flist_gt, val_flist)
      self.val_sample_iterator = self.val_dataset.create_iterator(
          config.batch_size)
      self.test_dataset = Dataset(config, test_flist_gt, test_flist, test=True)
      self.test_sample_iterator = self.test_dataset.create_iterator(
          config.batch_size)
      self.samples_dir = os.path.join('samples', config.name)
      os.makedirs(self.samples_dir, exist_ok=True)
    elif config.mode == 'test':
      self.test_dataset = Dataset(config, [], [], test=True)
      self.init_test()

  def train(self):
    train_loader = DataLoader(dataset=self.train_dataset,
                              batch_size=self.config.batch_size,
                              num_workers=self.config.workers, drop_last=True,
                              shuffle=True)

    if not self.train_dataset:
      self.log.info('No training data was provided!')
      return

    writer = SummaryWriter('logs/' + self.config.name)

    while self.epoch < self.config.epochs:
      self.log.info('Training epoch: %d', self.epoch)
      self.epoch += 1

      for items in train_loader:
        self.inpaint_model.train()
        iteration = self.inpaint_model.iteration

        images, uvmaps, uvmap_gts, vertices, coeffs, rand_images, rand_uvmaps, rand_verts, rand_coeffs = self.to_device(
            *items)

        _, gen_loss, im_dis_loss, uv_dis_loss, logs = self.inpaint_model.process(
            images, uvmaps, uvmap_gts, vertices, coeffs)
        for k, v in logs.items():
          writer.add_scalar(k, v, iteration)

        self.inpaint_model.backward(gen_loss=gen_loss, im_dis_loss=im_dis_loss,
                                    uv_dis_loss=uv_dis_loss)

        _, rand_gen_loss, rand_im_dis_loss, rand_uv_dis_loss, rand_logs = self.inpaint_model.process(
            rand_images, rand_uvmaps, uvmap_gts, rand_verts, rand_coeffs, False)
        self.inpaint_model.backward(gen_loss=rand_gen_loss,
                                    im_dis_loss=rand_im_dis_loss,
                                    uv_dis_loss=rand_uv_dis_loss)

        self.inpaint_model.iteration += 1

        # log model at checkpoints
        if self.config.log_interval and iteration % self.config.log_interval == 0:
          info = 'Epoch: {} Iter:{}\n'.format(self.epoch, iteration)
          info = create_log(logs, info)
          self.log.info(info)

          info = 'Epoch: {} Iter:{} RANDOM UVMAP\n'.format(
              self.epoch, iteration)
          info = create_log(rand_logs, info)
          self.log.info(info)

        # sample model at checkpoints
        if self.config.sample_interval and iteration % self.config.sample_interval == 0:
          self.val_sample()
          self.test_sample()

        if self.config.ckpt_interval and iteration % self.config.ckpt_interval == 0:
          self.inpaint_model.save(self.epoch)

    self.log.info('\nEnd training....')

  def val_sample(self, it=None):
    self.inpaint_model.eval()

    val_items = next(self.val_sample_iterator)
    images, uvmaps, uvmap_gts, vertices, coeffs, rand_images, rand_uvmaps, rand_verts, rand_coeffs = self.to_device(
        *val_items)

    gen_uvmaps, im_merged = self.sample(images, uvmaps, vertices, coeffs)
    rand_gen_uvmaps, rand_im_merged = self.sample(rand_images, rand_uvmaps,
                                                  rand_verts, rand_coeffs)

    iteration = self.inpaint_model.iteration
    if it is not None:
      iteration = it

    image_per_row = 2
    if self.config.batch_size <= 6:
      image_per_row = 1

    images = utils.stitch_images(
        utils.to_uint8_torch(images[:, :3]),
        utils.to_uint8_torch(uvmaps[:, :3]),
        utils.to_uint8_torch(gen_uvmaps[:, :3]),
        utils.to_uint8_torch(uvmap_gts),
        utils.to_uint8_torch(im_merged[:self.config.batch_size]),
        utils.to_uint8_torch(im_merged[self.config.batch_size:]),
        im_size=self.config.uv_size, img_per_row=image_per_row)
    name = os.path.join(self.samples_dir, str(iteration - 1).zfill(5) + ".png")
    images.save(name)
    self.log.info('Val Sample saved to %s', name)

    images = utils.stitch_images(
        utils.to_uint8_torch(rand_images[:, :3]),
        utils.to_uint8_torch(rand_uvmaps[:, :3]),
        utils.to_uint8_torch(rand_gen_uvmaps[:, :3]),
        utils.to_uint8_torch(rand_im_merged[:self.config.batch_size]),
        utils.to_uint8_torch(rand_im_merged[self.config.batch_size:]),
        im_size=self.config.uv_size, img_per_row=image_per_row)
    name = os.path.join(self.samples_dir,
                        str(iteration - 1).zfill(5) + "_r.png")
    images.save(name)
    self.log.info('Val Sample saved to %s', name)

  def test_sample(self, it=None):

    self.inpaint_model.eval()

    test_items = next(self.test_sample_iterator)
    images, uvmaps, vertices, coeffs = self.to_device(*test_items)

    gen_uvmaps, im_merged = self.sample(images, uvmaps, vertices, coeffs)

    iteration = self.inpaint_model.iteration
    if it is not None:
      iteration = it

    image_per_row = 2
    if self.config.batch_size <= 6:
      image_per_row = 1

    images = utils.stitch_images(
        utils.to_uint8_torch(images[:, :3]),
        utils.to_uint8_torch(uvmaps[:, :3]),
        utils.to_uint8_torch(gen_uvmaps[:, :3]),
        utils.to_uint8_torch(im_merged[:self.config.batch_size]),
        utils.to_uint8_torch(im_merged[self.config.batch_size:]),
        im_size=self.config.uv_size, img_per_row=image_per_row)

    # path = os.path.join(self.samples_dir, self.name)
    name = os.path.join(self.samples_dir,
                        str(iteration - 1).zfill(5) + "_t.png")
    os.makedirs(self.samples_dir, exist_ok=True)
    images.save(name)

    self.log.info('Test sample saved to %s\n', name)

  def sample(self, images, uvmaps, vertices, coeffs):
    gen_uvmaps, renders, _ = self.inpaint_model(images[:, :3], uvmaps, vertices,
                                                coeffs, fix_uv=True)
    # io.imsave('tmp/render.png', renders[0].permute(1,2,0).cpu().detach().numpy())

    double_images = torch.cat([images, torch.flip(images, (3,))], dim=0)
    no_l_eye = double_images[:, -1:] != self.ex_idx[0]
    no_r_eye = double_images[:, -1:] != self.ex_idx[1]
    no_mouth = double_images[:, -1:] != self.ex_idx[2]
    mask = renders[:, 3:4] * no_l_eye.float() * no_r_eye.float(
    ) * no_mouth.float()
    im_merged = double_images[:, :3] * (1 - mask) + renders[:, :3] * mask
    # io.imsave('tmp/mask.png', mask[0, 0].cpu().detach().numpy())

    return gen_uvmaps.cpu(), im_merged.cpu()

  def init_test(self):
    self.segmenter = Segment(self.device)

    up_line = 100
    bt_line = 80

    self.transfers = {}
    self.uv_creators = {}
    self.nsh_face_tris = {}
    self.nsh_meshes = {}
    self.nsh_face_meshes = {}
    for face_model in ['230']:
      self.transfers[face_model] = Shape_Transfer(face_model=face_model,
                                                  device=self.device)
      self.uv_creators[face_model] = UVCreator(
          face_model=face_model, bfm_version=self.config.bfm_version,
          device=self.device)

      self.nsh_face_meshes[face_model] = meshio.Mesh(
          'data/mesh/{}/nsh_bfm_face.obj'.format(face_model))
      self.nsh_face_tris[face_model] = self.to_tensor(
          self.nsh_face_meshes[face_model].triangles, torch.int64)
      self.nsh_meshes[face_model] = meshio.Mesh(
          'data/mesh/{}/nsh_std.obj'.format(face_model), group=True)

    self.up_line = int(up_line * (self.config.uv_size / 1024))
    self.bt_line = int(bt_line * (self.config.uv_size / 1024))

    self.eye_lm_idx = np.loadtxt('data/mesh/eye_lm_idx.txt', dtype=np.int32)

    self.cropper = ImageCropper(self.config.im_size, use_dlib=False)
    self.reconstructor = Deep3DFace(self.sess, self.graph)

    R, T = look_at_view_transform(10, 0, 0)
    self.cameras = OpenGLPerspectiveCameras(znear=0.001, zfar=30.0,
                                            aspect_ratio=1.0, fov=12.5936,
                                            degrees=True, R=R, T=T,
                                            device=self.device)

    raster_settings = RasterizationSettings(image_size=512, blur_radius=0.0,
                                            faces_per_pixel=1, bin_size=0,
                                            cull_backfaces=True)
    self.rasterizer = MeshRasterizer(cameras=self.cameras,
                                     raster_settings=raster_settings)

  def preprocess(self, image, face_model):
    #* input image should be uint8, in RGB order

    image = utils.center_crop_resize(image, self.config.im_size)
    image = self.cropper.crop_image(image, self.config.im_size)

    image = image[:, ::-1].copy()

    images_224 = cv2.resize(image, (224, 224),
                            interpolation=cv2.INTER_AREA).astype(
                                np.float32)[None]

    images = self.to_tensor(image[None])
    segments = self.segmenter.segment_torch(images)
    segments = center_crop(segments, images.shape[1])
    image_segment = torch.cat([images, segments[..., None]], dim=-1)
    image_segment = image_segment.permute(0, 3, 1, 2)

    coeff, bfm_vert, bfm_neu_vert = self.reconstructor.predict(
        images_224, neutral=True)
    bfm_neu_vert = self.to_tensor(bfm_neu_vert)

    #! using torch from now on -----------------------------
    bfm_vert = self.to_tensor(bfm_vert)
    nsh_vert = self.transfers[face_model].transfer_shape_torch(bfm_vert)
    nsh_neu_vert = None
    nsh_neu_vert = self.transfers[face_model].transfer_shape_torch(bfm_neu_vert)
    nsh_face_vert = nsh_vert[self.uv_creators[face_model].nsh_face_start_idx:]

    coeff = self.to_tensor(coeff[None])
    _, _, _, angles, _, translation = utils.split_bfm09_coeff(coeff)
    # angle = (angle / 180.0 * math.pi) if degrees else angle
    transformer = Transform3d(device=self.device)
    transformer = transformer.rotate_axis_angle(angles[:, 0], self.rot_order[0],
                                                False)
    transformer = transformer.rotate_axis_angle(angles[:, 1], self.rot_order[1],
                                                False)
    transformer = transformer.rotate_axis_angle(angles[:, 2], self.rot_order[2],
                                                False)
    transformer = transformer.translate(translation)

    nsh_trans_vert = transformer.transform_points(nsh_face_vert[None])

    nsh_shift_vert = nsh_trans_vert[0] - self.to_tensor([[0, 0, 10]])
    image_segment = torch.flip(image_segment, (3,)).type(torch.float32)

    nsh_trans_mesh = Meshes(nsh_trans_vert,
                            self.nsh_face_tris[face_model][None])

    fragment = self.rasterizer(nsh_trans_mesh)
    visible_face = torch.unique(fragment.pix_to_face)[1:]  # exclude face id -1
    visible_vert = self.nsh_face_tris[face_model][visible_face]
    visible_vert = torch.unique(visible_vert)
    vert_alpha = torch.zeros([nsh_shift_vert.shape[0], 1], device=self.device)
    vert_alpha[visible_vert] = 1
    nsh_shift_vert_alpha = torch.cat([nsh_shift_vert, vert_alpha], axis=-1)

    uvmap = self.uv_creators[face_model].create_nsh_uv_torch(
        nsh_shift_vert_alpha, image_segment, self.config.uv_size)

    uvmap[..., 3] = uvmap[..., 3] + uvmap[..., 4] * 128
    uvmap = uvmap[..., :4].cpu().numpy()
    uvmap = self.test_dataset.process_uvmap(uvmap.astype(np.uint8),
                                            dark_brow=True)

    images = images.permute(0, 3, 1, 2) / 127.5 - 1.0
    images = F.interpolate(images, size=self.config.im_size, mode='bilinear',
                           align_corners=False)
    segments = F.interpolate(segments[:, None], size=self.config.im_size,
                             mode='nearest')
    images = torch.cat([images, segments], dim=1)
    uvmaps = uvmap[None].permute(0, 3, 1, 2)

    return images, uvmaps, coeff, nsh_face_vert, nsh_neu_vert

  def predict(self, image, out_dir, idx=None, deploy=False, face_model='230'):
    '''deploy for nsh'''
    if not deploy and idx is None:
      idx = '{:>05d}'.format(idx)

    images, uvmaps, params, nsh_face_vert, nsh_neu_vert = self.preprocess(
        image, face_model)

    fnames = []

    gen_uvmaps = self.inpaint_model.forward(images[:, :3], uvmaps,
                                            nsh_face_vert[None], params,
                                            fix_uv=True, deploy=deploy,
                                            face_model=face_model)
    nsh_uv = F.interpolate(gen_uvmaps.detach(), size=1024, mode='bilinear',
                           align_corners=False)[0]

    fnames.append(os.path.join(out_dir, '{}_uv.png'.format(idx)))
    self.imsave(fnames[-1], nsh_uv, False, True)

    lm_idx = self.to_tensor(self.transfers[face_model].lm_icp_idx, torch.int64)
    nsh_vert_lm = nsh_neu_vert[None, lm_idx]
    nsh_std_lm = self.to_tensor(self.transfers[face_model].tgt_std_vert)[None,
                                                                         lm_idx]
    R, T, s = corresponding_points_alignment(nsh_vert_lm, nsh_std_lm,
                                             estimate_scale=True)
    s = s * 0.97

    nsh_neu_vert_trans = (s[:, None, None] * torch.bmm(nsh_neu_vert[None], R) +
                          T[:, None, :])[0]
    nsh_neu_vert = nsh_neu_vert_trans.cpu().numpy()
    nsh_neu_vert = self.transfers[face_model].normalize(nsh_neu_vert)
    fnames.append(os.path.join(out_dir, '{}_neu.obj'.format(idx)))
    meshio.write_obj(
        fnames[-1],
        nsh_neu_vert[self.uv_creators[face_model].nsh_face_start_idx:],
        self.nsh_face_meshes[face_model].triangles,
        texcoords=self.nsh_face_meshes[face_model].texcoords, mtllib=True,
        uv_name='{}_uv'.format(idx))

    fnames.append(os.path.join(out_dir, '{}_neu.mtl'.format(idx)))

    try:
      self.imsave(os.path.join(out_dir, '{}_input.jpg'.format(idx)),
                  images[0, :3], True)
    except:
      pass

  def to_device(self, *args):
    return (item.to(self.device) for item in args)

  def to_tensor(self, array, dtype=torch.float32):
    if not isinstance(array, np.ndarray):
      array = np.array(array)
    return torch.from_numpy(array).type(dtype).to(self.device)

  def imsave(self, path, image, h_flip=False, v_flip=False):
    image = utils.to_uint8_torch(image.cpu()).numpy()
    if h_flip:
      image = image[:, ::-1]
    if v_flip:
      image = image[::-1]
    io.imsave(path, image)

  def compute_eye_param(self, vertices, eye_lm_idx, face_model):
    nsh_vert_lm = vertices[None, eye_lm_idx]
    nsh_std_lm = self.to_tensor(
        self.transfers[face_model].tgt_std_vert)[None, eye_lm_idx]
    R, T, s = corresponding_points_alignment(nsh_vert_lm, nsh_std_lm,
                                             estimate_scale=True)
    R = R.cpu().numpy()[0]
    T = T.cpu().numpy()[0]
    s = s.cpu().numpy()
    angle = Rotation.from_matrix(R).as_euler('xyz')
    eye_param = np.concatenate([angle, T, s])

    return eye_param


def center_crop(image, img_size):
  # set img_size to None will not resize image
  _, height, width = image.shape
  if width > img_size:
    w_s = (width - img_size) // 2
    image = image[:, :, w_s:w_s + img_size]
  if height > img_size:
    h_s = (height - img_size) // 2
    image = image[:, h_s:h_s + img_size, :]

  return image


def create_log(inputs, info):
  for k, v in inputs.items():
    if k.endswith('_a') or k.endswith('_m'):
      info += ' {}:{:>.2f}'.format(k, v)
    else:
      info += ' {}:{:>.4e}'.format(k, v)
  info += '\n'
  return info
