import os

import numpy as np
import torch

from lib import meshio


class Shape_Transfer():

  def __init__(self, face_model=None, tgt_dir=None, device=None):
    '''transfer shape from src to tgt'''
    if face_model is not None:
      tgt_ref = meshio.Mesh('data/mesh/{}/nsh_bfm.obj'.format(face_model))
      tgt_std = meshio.Mesh('data/mesh/{}/nsh_std.obj'.format(face_model))
      self.tgt_std_vert = tgt_std.vertices

      lm_path = 'data/mesh/lm_bfm_{}.txt'.format(face_model)
      self.lm_idx = np.loadtxt(lm_path, dtype=np.int32)
      self.lm_idx = self.lm_idx[np.r_[:53, 59:83]]
      self.lm_icp_idx = self.lm_idx[:, 1]

      self.nsh_neck1 = load_neck_idx(
          'data/mesh/{}/nsh_neck1.txt'.format(face_model))
      self.nsh_neck2 = load_neck_idx(
          'data/mesh/{}/nsh_neck2.txt'.format(face_model))
      self.nsh_neck3 = load_neck_idx(
          'data/mesh/{}/nsh_neck3.txt'.format(face_model))
      self.nsh_neck4 = load_neck_idx(
          'data/mesh/{}/nsh_neck4.txt'.format(face_model))
    else:
      tgt_ref = meshio.Mesh(os.path.join(tgt_dir, 'deformed.obj'))
      self.lm_idx = np.loadtxt(os.path.join(tgt_dir, 'landmarks.txt'), dtype=np.int32)

    self.tgt_vertice = tgt_ref.vertices
    self.tgt_landmarks = self.tgt_vertice[self.lm_idx[:, 1]]

    self.init_A()
    self.primary_value = get_primary_value(self.tgt_vertice[:, np.newaxis] -
                                           self.tgt_landmarks)[..., np.newaxis]
    if device is not None:
      self.device = device

      def to_tensor(array):
        return torch.from_numpy(array).float().to(self.device)

      self.inv_A_torch = to_tensor(self.inv_A)
      self.primary_value_torch = to_tensor(self.primary_value)
      self.tgt_vert_torch = to_tensor(self.tgt_vertice)

  def init_A(self):
    self.dim = np.shape(self.tgt_landmarks)[0]
    A = np.zeros((self.dim + 4, self.dim + 4))
    A[0, :self.dim] = 1
    A[1:4, :self.dim] = self.tgt_landmarks.transpose()
    pri_func_value = get_primary_value(
        np.repeat(self.tgt_landmarks, self.dim, axis=0) -
        np.tile(self.tgt_landmarks, (self.dim, 1)))
    A[4:, :self.dim] = pri_func_value.reshape((self.dim, self.dim))
    A[4:, self.dim:self.dim + 3] = self.tgt_landmarks
    A[4:, -1] = 1
    self.inv_A = np.linalg.inv(A)

  def transfer_shape(self, src_vert, normalize=False):
    # input vertice is BFM
    tgt_lm = src_vert[self.lm_idx[:, 0]]

    B = np.concatenate([np.zeros((4, 3)), tgt_lm], axis=0)
    X = np.dot(self.inv_A, B)
    weight = X[:self.dim]
    rotate = X[self.dim:self.dim + 3]
    transform = X[self.dim + 3]

    tgt_vert = weight * self.primary_value
    tgt_vert = np.sum(tgt_vert, axis=1) + np.dot(self.tgt_vertice,
                                                 rotate) + transform

    if not normalize:
      return tgt_vert
    else:
      return self.normalize(tgt_vert)

  def transfer_shape_torch(self, src_vert):
    # input vertice is BFM
    tgt_lm = src_vert[self.lm_idx[:, 0]]

    B = torch.cat([torch.zeros((4, 3)).to(self.device), tgt_lm], axis=0)
    X = torch.mm(self.inv_A_torch, B)
    weight = X[:self.dim]
    rotate = X[self.dim:self.dim + 3]
    transform = X[self.dim + 3]

    tgt_vert = weight * self.primary_value_torch
    tgt_vert = torch.sum(tgt_vert, axis=1) + torch.mm(self.tgt_vert_torch,
                                                      rotate) + transform

    return tgt_vert

  def normalize(self, input_vertice):
    save_nsh_vertice = input_vertice
    save_nsh_vertice[self.nsh_neck1] = self.tgt_std_vert[self.nsh_neck1]
    save_nsh_vertice[self.nsh_neck2] = 0.9 * self.tgt_std_vert[
        self.nsh_neck2] + 0.1 * save_nsh_vertice[self.nsh_neck2]
    save_nsh_vertice[self.nsh_neck3] = 0.7 * self.tgt_std_vert[
        self.nsh_neck3] + 0.3 * save_nsh_vertice[self.nsh_neck3]
    save_nsh_vertice[self.nsh_neck4] = 0.5 * self.tgt_std_vert[
        self.nsh_neck4] + 0.5 * save_nsh_vertice[self.nsh_neck4]
    return save_nsh_vertice


def get_primary_value(inputs, doubleSiamaSquare=1):
  r = np.linalg.norm(inputs, axis=-1)
  return np.exp(-(r / doubleSiamaSquare))


def get_primary_value_torch(inputs, doubleSiamaSquare=1):
  r = torch.norm(inputs, dim=-1)
  return torch.exp(-(r / doubleSiamaSquare))


def load_neck_idx(neck_file):
  nsh_neck = [
      x.strip() for x in open(neck_file, 'r').readlines() if len(x.strip()) > 1
  ]
  return [int(x) for x in nsh_neck]


def get_rotation_matrix(x, y):
  u = np.cross(x, y)
  a = np.arccos(np.dot(x, y) / np.linalg.norm(x) / np.linalg.norm(y))

  norm = np.linalg.norm(u)
  rotation_matrix = np.zeros([3, 3])
  u[0] /= norm
  u[1] /= norm
  u[2] /= norm

  sin = np.sin
  cos = np.cos

  rotation_matrix[0][0] = cos(a) + u[0] * u[0] * (1 - cos(a))
  rotation_matrix[0][1] = u[0] * u[1] * (1 - cos(a)) - u[2] * sin(a)
  rotation_matrix[0][2] = u[1] * sin(a) + u[0] * u[2] * (1 - cos(a))

  rotation_matrix[1][0] = u[2] * sin(a) + u[0] * u[1] * (1 - cos(a))
  rotation_matrix[1][1] = cos(a) + u[1] * u[1] * (1 - cos(a))
  rotation_matrix[1][2] = -u[0] * sin(a) + u[1] * u[2] * (1 - cos(a))

  rotation_matrix[2][0] = -u[1] * sin(a) + u[0] * u[2] * (1 - cos(a))
  rotation_matrix[2][1] = u[0] * sin(a) + u[1] * u[2] * (1 - cos(a))
  rotation_matrix[2][2] = cos(a) + u[2] * u[2] * (1 - cos(a))

  return rotation_matrix
