import logging
import os
import math
from time import time
from glob import glob

import cv2
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (DirectionalLights, MeshRasterizer, MeshRenderer,
                                OpenGLPerspectiveCameras, RasterizationSettings,
                                TexturedSoftPhongShader, look_at_view_transform)
from pytorch3d.structures import Meshes, Textures
from pytorch3d.transforms import Transform3d
from skimage import io

import utils
# from lib import meshio
from lib.gaussian import gaussian_blur
from lib.loss import AdversarialLoss, PerceptualLoss, StyleLoss
from networks import Generator, ImageDiscriminator, UVMapDiscriminator


class InpaintingModel(nn.Module):

  def __init__(self, config, device, rot_order, debug=False):
    super(InpaintingModel, self).__init__()

    self.debug = debug
    self.rot_order = rot_order
    self.device = device
    # if torch.cuda.device_count() > 1:
    #   self.device = torch.device('cuda:1')
    # else:
    #   self.device = self.device
    self.config = config
    # self.parser = 'hq' in config.name
    self.parser = False
    self.name = config.name
    self.batch_size = config.batch_size
    self.im_size = config.im_size
    self.uv_size = config.uv_size
    self.log = logging.getLogger('x')
    self.iteration = 0

    # small_image = io.imread(os.path.join(config.root_dir, 'data/uv_param/small_mask.png'))
    # small_mask = small_image[..., :3] == [0, 255, 0]
    # small_mask = np.all(small_mask, axis=-1)
    # self.small_mask = torch.from_numpy(small_mask)
    # self.ds_scale = self.uv_size // small_image.shape[0]

    self.mask_dir = 'data/uv_param/masks'
    # self.brow_mask = 1 - self.load_mask('data/uv_param/masks/brow_mask.png')
    ear_mask = 1 - self.load_mask(os.path.join(self.mask_dir, 'ear_mask.png'),
                                  -self.uv_size // 16)
    eye_mask = 1 - self.load_mask(os.path.join(self.mask_dir, 'eye_mask.png'))
    self.hair_mask = 1 - self.load_mask(
        os.path.join(self.mask_dir, 'hair_mask.png'))
    self.lip_mask = 1 - self.load_mask(
        os.path.join(self.mask_dir, 'lip_mask.png'), -self.uv_size // 32)
    # self.tone_mask = 1 - self.load_mask('data/uv_param/masks/tone_mask.png')
    self.skin_mask = self.load_mask(
        os.path.join(self.mask_dir, 'skin_mask_for_loss.png'),
        self.uv_size // 32)
    self.skin_ear_mask = torch.clamp(self.skin_mask + ear_mask, min=0, max=1)
    self.face_mask = self.load_mask(
        os.path.join(self.mask_dir, 'face_mask.png'), self.uv_size // 16)
    self.face_mask = torch.clamp(self.face_mask - eye_mask - self.lip_mask,
                                 min=0, max=1)

    self.meshes = {}
    for face_model in ['230']:
      mesh_path = os.path.join(config.root_dir, 'data', 'mesh', face_model,
                               'nsh_bfm_face.obj')
      mesh = load_objs_as_meshes([mesh_path], self.device)
      self.meshes[face_model] = mesh.extend(self.batch_size * 2)

    self.ckpt_dir = os.path.join('checkpoints', self.name)
    os.makedirs(self.ckpt_dir, exist_ok=True)
    self.gen_weights_name = os.path.join(
        self.ckpt_dir, '{}_{}_gen'.format(self.im_size, self.uv_size))
    self.im_dis_weights_name = os.path.join(
        self.ckpt_dir, '{}_{}_im_dis'.format(self.im_size, self.uv_size))
    self.uv_dis_weights_name = os.path.join(
        self.ckpt_dir, '{}_{}_uv_dis'.format(self.im_size, self.uv_size))
    # self.gen_weights_path = os.path.join(self.ckpt_dir, self.name + '_gen.pth')
    # self.im_dis_weights_path = os.path.join(self.ckpt_dir, self.name + '_im_dis.pth')
    # self.uv_dis_weights_path = os.path.join(self.ckpt_dir, self.name + '_uv_dis.pth')

    self.generator = Generator(3, 8, config).to(self.device)
    if self.config.use_cuda:
      self.generator = nn.parallel.DataParallel(self.generator)

    if config.mode == 'train':
      if config.adv_weight > 0:
        self.image_disc = ImageDiscriminator(3, config).to(self.device)
        self.uvmap_disc = UVMapDiscriminator(3, config).to(self.device)

      # if self.config.use_cuda:
      # # if torch.cuda.device_count() > 1:
      #   self.generator = nn.parallel.DistributedDataParallel(self.generator)
      #   self.image_disc = nn.parallel.DistributedDataParallel(self.image_disc)
      #   self.uvmap_disc = nn.parallel.DistributedDataParallel(self.uvmap_disc)

      self.l1_loss = nn.L1Loss()
      # self.l1_loss = nn.SmoothL1Loss()
      # self.smooth_l1_loss = nn.SmoothL1Loss()
      # self.l2_loss = nn.MSELoss()
      self.perceptual_loss = PerceptualLoss()
      self.style_loss = StyleLoss()
      # self.facial_loss = FacialLoss()
      if self.config.gan_loss != 'wgan':
        self.adversarial_loss = AdversarialLoss(types=config.gan_loss)

      self.gen_optimizer = optim.Adam(params=self.generator.parameters(),
                                      lr=config.learning_rate,
                                      betas=(config.beta1, config.beta2),
                                      weight_decay=0.0001)
      if config.adv_weight > 0:
        self.im_dis_optimizer = optim.Adam(params=self.image_disc.parameters(),
                                           lr=config.learning_rate * 0.1,
                                           betas=(config.beta1, config.beta2),
                                           weight_decay=0.001)
        self.uv_dis_optimizer = optim.Adam(params=self.uvmap_disc.parameters(),
                                           lr=config.learning_rate * 0.1,
                                           betas=(config.beta1, config.beta2),
                                           weight_decay=0.001)

    self.load_uvmasks()
    self.init_renderer()

  def load_uvmasks(self):

    def to_torch(x):
      x = cv2.resize(x, (self.uv_size, self.uv_size),
                     interpolation=cv2.INTER_NEAREST)
      return torch.from_numpy(x).to(self.device)

    uv_tmp = io.imread(os.path.join(self.mask_dir, 'uvmap.png'))[..., :3]
    uv_tmp = to_torch(uv_tmp).float() / 127.5 - 1

    skin_mask = io.imread(os.path.join(self.mask_dir,
                                       'skin_mask.png'))[..., -1] // 255
    ear_mask = 1 - io.imread(os.path.join(self.mask_dir,
                                          'ear_mask.png'))[..., -1] // 255
    skin_mask = np.clip(skin_mask + ear_mask, 0, 1)
    skin_mask = to_torch(skin_mask)
    self.tmp_mean = torch.mean(uv_tmp[skin_mask == 1], axis=0)[None, :, None,
                                                               None]
    self.uv_tmp = uv_tmp.permute(2, 0, 1)[None]

    hair_mask = 1 - io.imread(os.path.join(self.mask_dir,
                                           'hair_mask.png'))[..., -1] // 255
    tone_mask = 1 - io.imread(os.path.join(self.mask_dir,
                                           'tone_mask.png'))[..., -1] // 255
    hair_mask = cv2.GaussianBlur(hair_mask.astype(np.float32), (77, 77), 49)
    hair_tone_mask = np.clip(hair_mask + tone_mask, 0, 1)[..., None]
    self.hair_tone_mask = to_torch(hair_tone_mask)[None, None]

    face_mask = io.imread(os.path.join(self.mask_dir,
                                       'face_mask.png'))[..., -1] // 255
    blur_face_mask = cv2.GaussianBlur(face_mask.astype(np.float32), (99, 99),
                                      49)[..., None]
    # blur_face_mask_bt = cv2.GaussianBlur(face_mask.astype(np.float32), (99, 99), 49)[..., None]
    # blur_face_mask[self.uv_size // 2:] = blur_face_mask_bt[self.uv_size // 2:]
    self.blur_face_mask = to_torch(blur_face_mask)[None, None]

  def init_renderer(self):
    # nsh_face_mesh = meshio.Mesh('data/mesh/nsh_bfm_face.obj')
    # self.nsh_face_tri = torch.from_numpy(nsh_face_mesh.triangles).type(
    #     torch.int64).to(self.device)

    R, T = look_at_view_transform(10, 0, 0)
    cameras = OpenGLPerspectiveCameras(znear=0.001, zfar=30.0, aspect_ratio=1.0,
                                       fov=12.5936, degrees=True, R=R, T=T,
                                       device=self.device)
    raster_settings = RasterizationSettings(image_size=self.im_size,
                                            blur_radius=0.0, faces_per_pixel=1,
                                            bin_size=0, cull_backfaces=True)
    self.rasterizer = MeshRasterizer(cameras=cameras,
                                     raster_settings=raster_settings)
    lights = DirectionalLights(device=self.device)
    shader = TexturedSoftPhongShader(device=self.device, cameras=cameras,
                                     lights=lights)
    self.renderer = MeshRenderer(rasterizer=self.rasterizer, shader=shader)

    # if torch.cuda.device_count() > 1:
    # self.renderer = nn.parallel.DistributedDataParallel(self.renderer)

  def process(self, images_alpha, uvmaps_alpha, uvmap_gts, vertices, coeffs,
              uv_gt=True):

    # zero optimizers
    self.gen_optimizer.zero_grad()
    if self.config.adv_weight > 0:
      self.im_dis_optimizer.zero_grad()
      self.uv_dis_optimizer.zero_grad()

    # process outputs
    images = images_alpha[:, :3].contiguous()
    im_skins = images_alpha[:, 3:4].contiguous()

    gen_uvmaps, renders_alpha, lights = self(images, uvmaps_alpha, vertices,
                                             coeffs)
    renders = renders_alpha[:, :3]
    renders_mask = renders_alpha[:, 3:]

    gen_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
    im_dis_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
    uv_dis_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
    im_gen_gan_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
    uv_gen_gan_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
    gen_uv_std_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
    gen_uv_sym_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
    gen_rd_l1_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
    gen_rd_style_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
    gen_uv_style_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
    gen_uv_content_loss = torch.tensor(0, dtype=torch.float32,
                                       device=self.device)
    gen_uv_l1_loss = torch.tensor(0, dtype=torch.float32, device=self.device)

    double_image = torch.cat([images, torch.flip(images, (3,))], dim=0)
    double_skins = torch.cat([im_skins, torch.flip(im_skins, (3,))], dim=0)
    render_mask = renders_mask * double_skins
    uv_merged = double_image * (1 - render_mask) + renders * render_mask
    uv_merged = uv_merged.contiguous()
    self.imsave('tmp/train/full_render.png', uv_merged[0, :3])
    self.imsave('tmp/train/full_render_flip.png',
                uv_merged[self.batch_size, :3])
    self.imsave('tmp/train/image_mask.png', double_skins[0, :3])
    self.imsave('tmp/train/image_mask_flip.png',
                double_skins[self.batch_size, :3])
    self.imsave('tmp/train/renders_mask.png', renders_mask[0, 0])
    self.imsave('tmp/train/render_mask.png', render_mask[0, 0])
    self.imsave('tmp/train/image.png', double_image[0, :3])
    self.imsave('tmp/train/image_flip.png', double_image[self.batch_size, :3])

    if self.config.adv_weight > 0:
      # discriminator loss
      im_dis_input_real = images
      im_dis_input_fake = uv_merged[:self.batch_size].detach()
      self.imsave('tmp/train/im_dis_input_real.png', im_dis_input_real[0, :3])
      self.imsave('tmp/train/im_dis_input_fake.png', im_dis_input_fake[0, :3])
      im_dis_real = self.image_disc(im_dis_input_real)
      im_dis_fake = self.image_disc(im_dis_input_fake)

      if self.config.gan_loss == 'wgan':
        im_dis_real_loss = -torch.mean(im_dis_real)
        im_dis_fake_loss = torch.mean(im_dis_fake)
        im_dis_gp = self.calculate_gradient_penalty(self.image_disc,
                                                    im_dis_input_real,
                                                    im_dis_input_fake)
        im_dis_loss += (im_dis_real_loss + im_dis_fake_loss +
                        im_dis_gp) * self.config.adv_weight
      else:
        im_dis_real_loss = self.adversarial_loss(im_dis_real, True, True)
        im_dis_fake_loss = self.adversarial_loss(im_dis_fake, False, True)
        im_dis_loss += (im_dis_real_loss + im_dis_fake_loss) / 2

      uv_dis_input_real = uvmap_gts
      if not uv_gt:
        uv_dis_input_real = torch.flip(uv_dis_input_real, (3,))
      self.imsave('tmp/train/uv_dis_input_real.png', uv_dis_input_real[0, :3])
      uv_dis_real = self.uvmap_disc(uv_dis_input_real)

      uv_dis_input_fake_1 = gen_uvmaps.detach()
      self.imsave('tmp/train/uv_dis_input_fake_1.png',
                  uv_dis_input_fake_1[0, :3])
      uv_dis_fake_1 = self.uvmap_disc(uv_dis_input_fake_1)

      if self.config.gan_loss == 'wgan':
        uv_dis_real_loss = -torch.mean(uv_dis_real)
        uv_dis_fake_loss_1 = torch.mean(uv_dis_fake_1)
        uv_dis_gp = self.calculate_gradient_penalty(self.uvmap_disc,
                                                    uv_dis_input_real,
                                                    uv_dis_input_fake_1)
        uv_dis_loss += (uv_dis_real_loss + uv_dis_fake_loss_1 +
                        uv_dis_gp) * self.config.adv_weight
      else:
        uv_dis_real_loss = self.adversarial_loss(uv_dis_real, True, True)
        uv_dis_fake_loss_1 = self.adversarial_loss(uv_dis_fake_1, False, True)
        uv_dis_loss += (uv_dis_real_loss + uv_dis_fake_loss_1) / 2

      # generator adversarial loss
      im_gen_input_fake = uv_merged[:self.batch_size]
      self.imsave('tmp/train/im_gen_input_fake.png', im_gen_input_fake[0, :3])
      im_gen_fake = self.image_disc(im_gen_input_fake)
      if self.config.gan_loss == 'wgan':
        im_gen_gan_loss = -torch.mean(im_gen_fake)
      else:
        im_gen_gan_loss = self.adversarial_loss(im_gen_fake, True,
                                                False) * self.config.adv_weight
      gen_loss += im_gen_gan_loss

      uv_gen_input_fake = gen_uvmaps
      self.imsave('tmp/train/uv_gen_input_fake.png', uv_gen_input_fake[0, :3])
      uv_gen_fake = self.uvmap_disc(uv_gen_input_fake)
      if self.config.gan_loss == 'wgan':
        uv_gen_gan_loss = -torch.mean(uv_gen_fake)
      else:
        uv_gen_gan_loss = self.adversarial_loss(uv_gen_fake, True,
                                                False) * self.config.adv_weight
      gen_loss += uv_gen_gan_loss

    #* Other Losses
    if self.config.sym_weight > 0 or self.config.std_weight > 0:
      blur_gen_uvs = gaussian_blur(
          gen_uvmaps, (self.uv_size // 8 + 1, self.uv_size // 8 + 1),
          (self.uv_size // 32, self.uv_size // 32))
      self.imsave('tmp/train/blur_gen_uv.png', blur_gen_uvs[0, :3])

      # generator symmetry loss
      if self.config.sym_weight > 0:
        flipped_uv = torch.flip(blur_gen_uvs, dims=(3,))
        gen_uv_sym_loss = self.l1_loss(blur_gen_uvs, flipped_uv)
        self.imsave('tmp/train/uv_flip.png', flipped_uv[0, :3])
        gen_loss += gen_uv_sym_loss * self.config.sym_weight

      # generator variance loss
      if self.config.std_weight > 0:
        blur_uv_hsv = utils.rgb2hsv(blur_gen_uvs)
        gen_uv_std_loss = torch.mean(
            torch.std(blur_uv_hsv[:, :,
                                  self.skin_ear_mask.type(torch.bool)], dim=-1))
        blur_gen_uvs_for_lip = gaussian_blur(
            gen_uvmaps, (self.uv_size // 32 + 1, self.uv_size // 32 + 1),
            (self.uv_size // 64, self.uv_size // 64))
        gen_uv_std_loss += torch.mean(
            torch.std(
                blur_gen_uvs_for_lip[:, :, self.lip_mask.type(torch.bool)],
                dim=-1)) * 0.05

        gen_loss += gen_uv_std_loss * self.config.std_weight

    if uv_gt:
      self.imsave('tmp/train/uvmap_gens.png', gen_uvmaps[0, :3])
      self.imsave('tmp/train/uvmap_gts.png', uvmap_gts[0, :3])
      # generator l1 loss uvmap
      if self.config.l1_weight > 0:
        gen_uv_l1_loss = self.l1_loss(gen_uvmaps, uvmap_gts)
        gen_loss += gen_uv_l1_loss * self.config.l1_weight * 3

      # generator perceptual loss
      if self.config.con_weight > 0:
        gen_uv_content_loss = self.perceptual_loss(gen_uvmaps, uvmap_gts)
        gen_loss += gen_uv_content_loss * self.config.con_weight

      # generator style loss
      if self.config.sty_weight > 0:
        gen_uv_style_loss = self.style_loss(gen_uvmaps, uvmap_gts)
        gen_loss += gen_uv_style_loss * self.config.sty_weight

    # rendered L1 loss
    if self.config.l1_weight > 0:
      gen_rd_l1_loss = self.l1_loss(double_image, uv_merged)
      gen_loss += gen_rd_l1_loss * self.config.l1_weight

    if self.config.sty_weight > 0:
      gen_rd_style_loss = self.style_loss(double_image, uv_merged)
      gen_loss += gen_rd_style_loss * self.config.sty_weight

    gen_loss += torch.mean(
        torch.std(lights[:, 0:3], dim=-1) +
        torch.std(lights[:, 3:6], dim=-1) * 0.3)

    # create logs
    logs = {
        'im_d': im_dis_loss.item(),
        'uv_d': uv_dis_loss.item(),
        'im_g': im_gen_gan_loss.item(),
        'uv_g': uv_gen_gan_loss.item(),
        'uv_std': gen_uv_std_loss.item(),
        'uv_sym': gen_uv_sym_loss.item(),
        'rd_l1': gen_rd_l1_loss.item(),
        'rd_sty': gen_rd_style_loss.item()
    }
    if uv_gt:
      logs['uv_sty'] = gen_uv_style_loss.item()
      logs['uv_con'] = gen_uv_content_loss.item()
      logs['uv_l1'] = gen_uv_l1_loss.item()

    return gen_uvmaps, gen_loss, im_dis_loss, uv_dis_loss, logs

  def forward(self, images, uvmaps_alpha, vertices, coeffs, fix_uv=False,
              deploy=False, face_model='230'):
    # the input images should be 3 channel and uvmaps should be 4 channel

    uvmaps_flip = torch.flip(uvmaps_alpha, (3,))
    uvmaps_input = torch.cat([uvmaps_alpha, uvmaps_flip], dim=1)

    gen_uvmaps, light_params = self.generator(images, uvmaps_input)
    self.imsave('tmp/train/uv_before.png', gen_uvmaps[0, :3])

    if fix_uv:
      face_mean = torch.mean(gen_uvmaps[..., self.face_mask == 1], axis=-1)
      new_uv = self.uv_tmp - self.tmp_mean + face_mean[..., None, None]
      new_uv = self.uv_tmp * self.hair_tone_mask + new_uv * (
          1 - self.hair_tone_mask)
      gen_uvmaps = gen_uvmaps * self.blur_face_mask + new_uv * (
          1 - self.blur_face_mask)

      self.imsave('tmp/train/uv_fix.png', gen_uvmaps[0, :3])

    if deploy:
      return gen_uvmaps
    else:
      renders = self.rendering(light_params, coeffs, vertices, gen_uvmaps,
                               face_model)
      if self.config.mode == 'test':
        light_params[:, 0:3] = light_params[:, 0:3] + light_params[:, 3:6] + 1.0
        light_params[:, 3:9] = -1
        alb_rends = self.rendering(light_params, coeffs, vertices, gen_uvmaps,
                                   face_model)

      self.imsave('tmp/train/uv_flip.png', uvmaps_flip[0, :4])
      self.imsave('tmp/train/uv_input.png', uvmaps_input[0, :4])
      self.imsave('tmp/train/gen_uvmap.png', gen_uvmaps[0])
      self.imsave('tmp/train/renders0.png', renders[0, :3])
      self.imsave('tmp/train/renders1.png', renders[1, :3])
      self.imsave('tmp/train/rend_mask.png', renders[0, 3])

      if self.config.mode != 'test':
        return gen_uvmaps, renders, light_params
      else:
        return gen_uvmaps, renders, alb_rends

  def rendering(self, light_params, coeffs, vertices, gen_uvmaps, face_model):
    ambient_color = torch.clamp(0.5 + 0.5 * light_params[:, 0:3], 0, 1)
    diffuse_color = torch.clamp(0.5 + 0.5 * light_params[:, 3:6], 0, 1)
    specular_color = torch.clamp(0.2 + 0.2 * light_params[:, 6:9], 0, 1)
    direction = light_params[:, 9:12]
    directions = torch.cat([
        direction, direction *
        torch.tensor([[-1, 1, 1]], dtype=torch.float, device=self.device)
    ], dim=0)
    lights = DirectionalLights(ambient_color=ambient_color.repeat(2, 1),
                               diffuse_color=diffuse_color.repeat(2, 1),
                               specular_color=specular_color.repeat(2, 1),
                               direction=directions, device=self.device)
    self.renderer.shader.lights = lights

    _, _, _, angles, _, trans = utils.split_bfm09_coeff(coeffs)

    reflect_angles = torch.cat([
        angles, angles *
        torch.tensor([[1, -1, -1]], dtype=torch.float, device=self.device)
    ], dim=0)
    reflect_trans = torch.cat([
        trans, trans *
        torch.tensor([[-1, 1, 1]], dtype=torch.float, device=self.device)
    ], dim=0)
    rotated_vert = self.rotate_vert(vertices.repeat(2, 1, 1), reflect_angles,
                                    reflect_trans)

    fliped_uv = torch.flip(gen_uvmaps / 2 + 0.5,
                           (2, 3)).repeat(2, 1, 1, 1).permute(0, 2, 3, 1)
    texture = Textures(
        maps=fliped_uv,
        faces_uvs=self.meshes[face_model].textures.faces_uvs_padded(),
        verts_uvs=self.meshes[face_model].textures.verts_uvs_padded())
    meshes = Meshes(rotated_vert, self.meshes[face_model].faces_padded(),
                    texture)

    renders = self.renderer(meshes)

    renders[..., :3] = renders[..., :3] * 2 - 1
    renders[..., -1] = (renders[..., -1] > 0).float()
    renders = renders.permute(0, 3, 1, 2).contiguous()

    return renders

  def rotate_vert(self, vertices, angles, trans):
    transformer = Transform3d(device=self.device)
    transformer = transformer.rotate_axis_angle(angles[:, 0], self.rot_order[0],
                                                False)
    transformer = transformer.rotate_axis_angle(angles[:, 1], self.rot_order[1],
                                                False)
    transformer = transformer.rotate_axis_angle(angles[:, 2], self.rot_order[2],
                                                False)
    transformer = transformer.translate(trans)

    rotate_vert = transformer.transform_points(vertices)
    return rotate_vert

  def calculate_gradient_penalty(self, discrimiator, real_data, fake_data):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake data
    alpha = torch.rand(real_data.size(0), 1, 1, 1).to(self.device)

    interpolates = (alpha * real_data +
                    ((1 - alpha) * fake_data)).requires_grad_(True).to(
                        self.device)
    discrimiator_interpolates = discrimiator(interpolates)
    fake = torch.ones(
        discrimiator_interpolates.size()).requires_grad_(False).to(self.device)

    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(outputs=discrimiator_interpolates,
                              inputs=interpolates, grad_outputs=fake,
                              create_graph=True, retain_graph=True,
                              only_inputs=True)[0]
    # lambda for gradient penalty is set to 10
    gradient_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean() * 10

    return gradient_penalty

  def backward(self, gen_loss=None, im_dis_loss=None, uv_dis_loss=None):
    if self.config.adv_weight > 0:
      if im_dis_loss is not None:
        im_dis_loss.backward()
        self.im_dis_optimizer.step()

      if uv_dis_loss is not None:
        uv_dis_loss.backward()
        self.uv_dis_optimizer.step()

    if gen_loss is not None:
      gen_loss.backward()
      self.gen_optimizer.step()

  def load_pth(self, path):
    self.log.info('Loading checkpoint from %s ...', path)
    data = torch.load(path, map_location=self.device)
    return data

  def load(self):
    gen_weights_paths = sorted(
        glob(
            os.path.join(self.config.root_dir,
                         self.gen_weights_name + '_*.pth')))
    epoch = 0
    if gen_weights_paths:
      data = self.load_pth(gen_weights_paths[-1])
      epoch = int(
          os.path.split(gen_weights_paths[-1])[-1].split('.')[0].split('_')[-1])
      if not self.config.use_cuda:
        data['generator'] = utils.fix_state_dict(data['generator'])
      self.generator.load_state_dict(data['generator'])
      self.iteration = data['iteration']

    # load discriminator only when training
    if self.config.mode == 'train':
      im_dis_weights_paths = sorted(glob(self.im_dis_weights_name + '_*.pth'))
      if im_dis_weights_paths:
        data = self.load_pth(im_dis_weights_paths[-1])
        if not self.config.use_cuda:
          data['image_disc'] = utils.fix_state_dict(data['image_disc'])
        self.image_disc.load_state_dict(data['image_disc'])

      uv_dis_weights_paths = sorted(glob(self.uv_dis_weights_name + '_*.pth'))
      if uv_dis_weights_paths:
        data = self.load_pth(uv_dis_weights_paths[-1])
        if not self.config.use_cuda:
          data['uvmap_disc'] = utils.fix_state_dict(data['uvmap_disc'])
        self.uvmap_disc.load_state_dict(data['uvmap_disc'])
    return epoch

  def save(self, idx):
    torch.save(
        {
            'iteration': self.iteration,
            'generator': self.generator.state_dict()
        }, '{}_{:>04}.pth'.format(self.gen_weights_name, idx))

    if self.config.adv_weight > 0:
      torch.save({'image_disc': self.image_disc.state_dict()},
                 '{}_{:>04}.pth'.format(self.im_dis_weights_name, idx))
      torch.save({'uvmap_disc': self.uvmap_disc.state_dict()},
                 '{}_{:>04}.pth'.format(self.uv_dis_weights_name, idx))

    self.log.info('Saved checkpoint to %s.\n', self.name)

  def imsave(self, path, image, debug=False):
    if debug or self.debug:
      io.imsave(path, utils.to_uint8_torch(image.cpu()))

  def load_mask(self, path, erode=0):
    mask = io.imread(os.path.join(self.config.root_dir, path))[..., -1]
    mask = cv2.resize(mask, (self.uv_size, self.uv_size),
                      interpolation=cv2.INTER_NEAREST)

    mask = mask // 255
    if erode > 0:
      mask = cv2.erode(mask, np.ones((erode // 4, erode // 4)), iterations=4)
    elif erode < 0:
      mask = cv2.dilate(mask, np.ones((-erode // 4, -erode // 4)), iterations=4)

    mask = torch.from_numpy(mask).to(self.device)
    return mask.int()

  def to_tensor(self, array, dtype=torch.float32):
    if not isinstance(array, np.ndarray):
      array = np.array(array)
    return torch.from_numpy(array).type(dtype).to(self.device)
