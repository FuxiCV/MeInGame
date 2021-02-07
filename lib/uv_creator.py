import functools
import math
import operator
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from lib import meshio


class UVCreator():

  def __init__(self, face_model, bfm_version='face', im_size=1024, uv_size=1024,
               device=torch.device('cpu')):
    self.face_model = face_model
    self.bfm_version = bfm_version
    self.im_size = im_size
    self.uv_size = uv_size
    self.device = device

    self.load_face_mesh()

    nsh_prefix = 'data/uv_param/{}/{}'.format(face_model, self.uv_size)
    bfm_prefix = 'data/uv_param/bfm_{}/{}'.format(bfm_version, self.uv_size)
    for p in [nsh_prefix, bfm_prefix]:
      if not os.path.isdir(p):
        os.makedirs(p)

    self.nsh_vert_idxs, self.nsh_bary_weight = self.get_uv_idx(
        nsh_prefix, self.nsh_uv_coord, self.nsh_face_tri)
    self.nsh_bary_weight_np = self.nsh_bary_weight.cpu().numpy()
    self.bfm_vert_idxs, self.bfm_bary_weight = self.get_uv_idx(
        bfm_prefix, self.bfm_uv_coord, self.bfm_tri)

  def load_face_mesh(self):
    nsh_mesh = meshio.Mesh('data/mesh/{}/nsh_std.obj'.format(self.face_model),
                           group=True)

    self.nsh_face_face_idx = nsh_mesh.groups[1][-1]
    self.nsh_head_tri = nsh_mesh.triangles
    nsh_face_face = self.nsh_head_tri[self.nsh_face_face_idx:]
    nsh_face_vert_idx = functools.reduce(operator.iconcat, nsh_face_face, [])
    nsh_face_vert_idx = list(set(nsh_face_vert_idx))
    self.nsh_face_start_idx = min(nsh_face_vert_idx) - 1
    self.nsh_face_tri = np.array(nsh_face_face) - (self.nsh_face_start_idx + 1)
    self.num_face = self.nsh_face_tri.shape[0]

    nsh_uv_coord = process_uv(np.array(nsh_mesh.texcoords), self.uv_size)
    self.nsh_uv_coord = nsh_uv_coord[self.nsh_face_start_idx:]

    bfm_mesh = meshio.Mesh('data/mesh/bfm09_{}.obj'.format(self.bfm_version))
    self.bfm_tri = bfm_mesh.triangles
    self.bfm_uv_coord = process_uv(np.array(bfm_mesh.texcoords), self.uv_size)

    fov_y = 12.5936
    facial = np.tan(fov_y / 360.0 * math.pi)
    self.facial = np.reshape(facial, [1, 1])
    self.facial_torch = self.to_tensor(self.facial)

  def get_uv_idx(self, prefix, uv_coord, triangles):
    tri_buf_path = os.path.join(prefix, 'tri_buffer.npy')
    bary_wei_path = os.path.join(prefix, 'barycentric_weight.npy')

    if not os.path.isfile(tri_buf_path) or not os.path.isfile(bary_wei_path):
      print('rasterizing triangles...')
      _, tri_buff, bary_weight = rasterize_triangles(uv_coord, triangles,
                                                     self.uv_size, self.uv_size)
      tri_buff.tofile(tri_buf_path)
      bary_weight.tofile(bary_wei_path)
    else:
      tri_buff = np.fromfile(tri_buf_path, dtype=np.int32).reshape(
          [self.uv_size, self.uv_size])
      bary_weight = np.fromfile(bary_wei_path, dtype=np.float32).reshape(
          [self.uv_size, self.uv_size, 3])

    vert_idxs = triangles[tri_buff]
    bary_weight_torch = self.to_tensor(bary_weight)
    return vert_idxs, bary_weight_torch

  def create_nsh_uv_torch(self, nsh_shift_vert, input_image, uv_size):
    return self.create_uv_torch(nsh_shift_vert, input_image, uv_size,
                                self.nsh_vert_idxs, self.nsh_bary_weight)

  def create_bfm_uv_torch(self, bfm_shift_vert, input_image, uv_size):
    return self.create_uv_torch(bfm_shift_vert, input_image, uv_size,
                                self.bfm_vert_idxs, self.bfm_bary_weight)

  def create_uv_torch(self, shift_vert, input_image, uv_size, vert_idxs=None,
                      bary_weight=None, param_dir=None, tpt_mesh=None):
    # input_image = torch.flip(input_image, (3,)).type(torch.float32)
    # input image shape: [bs,c, h, w]
    if vert_idxs is None or bary_weight is None:
      assert param_dir is not None, 'uv params are not exist!'
      tri_buf_path = os.path.join(param_dir, 'tri_buffer.npy')
      bary_wei_path = os.path.join(param_dir, 'barycentric_weight.npy')
      tri_buff = np.fromfile(tri_buf_path, dtype=np.int32).reshape(
          [self.uv_size, self.uv_size])
      bary_weight = np.fromfile(bary_wei_path, dtype=np.float32).reshape(
          [self.uv_size, self.uv_size, 3])
      vert_idxs = tpt_mesh.triangles[tri_buff]
      bary_weight = self.to_tensor(bary_weight)

    in_size = input_image.size()
    if in_size[2] != self.im_size:
      image = torch.zeros((in_size[0], in_size[1], self.im_size, self.im_size),
                          device=self.device)
      image[:, :3] = F.interpolate(input_image[:, :3], size=self.im_size,
                                   mode='bilinear', align_corners=False)
      if in_size[1] > 3:
        image[:, 3:4] = F.interpolate(input_image[:, 3:4], size=self.im_size,
                                      mode='nearest')
    else:
      image = input_image

    pixel_uv_coord = shift_vert[vert_idxs]
    use_vis = pixel_uv_coord.shape[-1] > 3
    if use_vis:
      pixel_uv_visible = pixel_uv_coord[..., -1]
      pixel_uv_visible = pixel_uv_visible.bool().all(dim=-1)[None]
    pixel_uv_coord = pixel_uv_coord[..., :3]

    pixel_uv_coord = torch.sum(pixel_uv_coord * bary_weight[..., None],
                               axis=-2)[..., :3]
    pixel_uv_coord = pixel_uv_coord.view([-1, 3])

    if self.bfm_version == 'face':
      proj_coord = pixel_uv_coord[..., :3]
      proj_coord = (proj_coord[..., :2] +
                    1e-8) / (self.facial_torch * proj_coord[..., 2:3] + 1e-8)
      half_size = self.im_size // 2
      coords = torch.round(proj_coord * half_size + half_size).long()
    else:
      coords = torch.round(pixel_uv_coord[..., :2]).long()

    coords = torch.clamp(coords, 0, self.im_size - 1)
    proj_uv = image[0, :, coords[:, 1], coords[:, 0]]
    proj_uv = proj_uv.view([-1, self.uv_size, self.uv_size])

    if use_vis:
      proj_uv = torch.cat(
          [proj_uv, pixel_uv_visible.type(torch.float32)], axis=0)

    proj_uv = proj_uv[None]
    if uv_size != self.uv_size:
      proj_uv = F.interpolate(proj_uv, size=uv_size, mode='nearest')
    proj_uv = proj_uv.permute(0, 2, 3, 1)
    return torch.flip(proj_uv[0], (0, 1))

  def create_nsh_uv_np(self, nsh_shift_vert, input_image, segments,
                       visible=False):
    input_image_resize = cv2.resize(input_image, (self.im_size, self.im_size),
                                    interpolation=cv2.INTER_LANCZOS4)
    segment_resize = cv2.resize(segments, (self.im_size, self.im_size),
                                interpolation=cv2.INTER_NEAREST)
    input_image_ = np.concatenate(
        [input_image_resize, segment_resize[..., None]], axis=-1)

    input_image = input_image_[:, ::-1]
    pixel_uv_coord = nsh_shift_vert[self.nsh_vert_idxs]
    if visible:
      pixel_uv_visible = pixel_uv_coord[..., -1]
      pixel_uv_visible = np.all(pixel_uv_visible, axis=-1, keepdims=True)
      pixel_uv_coord = pixel_uv_coord[..., :3]
    pixel_uv_coord = np.sum(
        pixel_uv_coord * self.nsh_bary_weight_np[..., np.newaxis],
        axis=-2)[..., :3]
    pixel_uv_coord = pixel_uv_coord.reshape([-1, 3])

    proj_coord = pixel_uv_coord[..., :3] * [[1, -1, -1]]
    proj_coord = pixel_uv_coord[..., :3]
    proj_coord = (proj_coord[..., :2] / self.facial +
                  1e-8) / (proj_coord[..., 2:3] + 1e-8)

    half_size = self.im_size // 2
    coords = np.round(proj_coord * half_size + half_size).astype(np.int32)
    coords = np.clip(coords, 0, self.im_size - 1)
    proj_uv = input_image[coords[:, 1], coords[:, 0]]
    proj_uv = proj_uv.reshape([self.uv_size, self.uv_size, -1])
    if visible:
      proj_uv = np.concatenate(
          [proj_uv, pixel_uv_visible.astype(np.uint8)], axis=-1)

    return proj_uv[::-1, ::-1]

  def to_tensor(self, array, dtype=torch.float32):
    if not isinstance(array, np.ndarray):
      array = np.array(array)
    return torch.from_numpy(array).type(dtype).to(self.device)


def process_uv(uv_coords, img_size, need_reverse=True):
  uv_coords[:, 0] = uv_coords[:, 0] * (img_size - 1)
  uv_coords[:, 1] = uv_coords[:, 1] * (img_size - 1)
  if need_reverse:
    uv_coords[:, 1] = img_size - uv_coords[:, 1] - 1
  uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1))))  # add z
  return uv_coords


def isPointInTri(point, tri_points):
  tp = tri_points

  # vectors
  v0 = tp[2, :] - tp[0, :]
  v1 = tp[1, :] - tp[0, :]
  v2 = point - tp[0, :]

  # dot products
  dot00 = np.dot(v0.T, v0)
  dot01 = np.dot(v0.T, v1)
  dot02 = np.dot(v0.T, v2)
  dot11 = np.dot(v1.T, v1)
  dot12 = np.dot(v1.T, v2)

  # barycentric coordinates
  if dot00 * dot11 - dot01 * dot01 == 0:
    inverDeno = 0
  else:
    inverDeno = 1 / (dot00 * dot11 - dot01 * dot01)

  u = (dot11 * dot02 - dot01 * dot12) * inverDeno
  v = (dot00 * dot12 - dot01 * dot02) * inverDeno

  # check if point in triangle
  return (u >= 0) & (v >= 0) & (u + v < 1)


def get_point_weight(point, tri_points):

  tp = tri_points
  # vectors
  v0 = tp[2, :] - tp[0, :]
  v1 = tp[1, :] - tp[0, :]
  v2 = point - tp[0, :]

  # dot products
  dot00 = np.dot(v0.T, v0)
  dot01 = np.dot(v0.T, v1)
  dot02 = np.dot(v0.T, v2)
  dot11 = np.dot(v1.T, v1)
  dot12 = np.dot(v1.T, v2)

  # barycentric coordinates
  if dot00 * dot11 - dot01 * dot01 == 0:
    inverDeno = 0
  else:
    inverDeno = 1 / (dot00 * dot11 - dot01 * dot01)

  u = (dot11 * dot02 - dot01 * dot12) * inverDeno
  v = (dot00 * dot12 - dot01 * dot02) * inverDeno

  w0 = 1 - u - v
  w1 = v
  w2 = u

  return w0, w1, w2


def rasterize_triangles(vertices, triangles, h, w):
  # initial
  depth_buffer = np.zeros([h, w]) - 999999.
  #+ np.min(vertices[2,:]) - 999999. # set the initial z to the farest position
  triangle_buffer = np.zeros([h, w], dtype=np.int32) - 1
  # if tri id = -1, the pixel has no triangle correspondance
  barycentric_weight = np.zeros([h, w, 3], dtype=np.float32)  #

  for i in tqdm(range(triangles.shape[0])):
    tri = triangles[i, :]  # 3 vertex indices

    # the inner bounding box
    umin = max(int(np.ceil(np.min(vertices[tri, 0]))), 0)
    umax = min(int(np.floor(np.max(vertices[tri, 0]))), w - 1)

    vmin = max(int(np.ceil(np.min(vertices[tri, 1]))), 0)
    vmax = min(int(np.floor(np.max(vertices[tri, 1]))), h - 1)

    if umax < umin or vmax < vmin:
      continue

    for u in range(umin, umax + 1):
      for v in range(vmin, vmax + 1):
        if not isPointInTri([u, v], vertices[tri, :2]):
          continue
        w0, w1, w2 = get_point_weight([u, v],
                                      vertices[tri, :2])  # barycentric weight
        point_depth = w0 * vertices[tri[0], 2] + w1 * vertices[
            tri[1], 2] + w2 * vertices[tri[2], 2]
        if point_depth > depth_buffer[v, u]:
          depth_buffer[v, u] = point_depth
          triangle_buffer[v, u] = i
          barycentric_weight[v, u, :] = np.array([w0, w1, w2])

  return depth_buffer, triangle_buffer, barycentric_weight
