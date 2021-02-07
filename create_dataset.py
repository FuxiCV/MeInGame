import os

import cv2
import imageio
import numpy as np
import tensorflow as tf
import torch
from pytorch3d import renderer, transforms
from pytorch3d.structures import Meshes
from tqdm import tqdm

import utils
from lib import meshio
from lib.face_segment import Segment
from lib.image_cropper import ImageCropper
from lib.rbf import Shape_Transfer
from lib.uv_creator import UVCreator


class Deep3DFace():

  def __init__(self, sess, graph, img_size=224, batch_size=1):
    self.sess = sess
    self.graph = graph
    self.img_size = img_size
    self.batch_size = batch_size
    self.bfm = utils.BFM_model('.', 'data/models/bfm2009_face.mat')

    refer_mesh = meshio.Mesh('data/mesh/bfm09_face.obj')
    self.num_bfm_vert = refer_mesh.vertices.shape[0]
    bfm_offset = meshio.Mesh('data/mesh/bfm09_face_offset.obj').vertices.astype(
        np.float32)
    self.bfm_offset = bfm_offset - refer_mesh.vertices.astype(np.float32)
    self.vert_mean = np.reshape(self.bfm.shapeMU, [-1, 3])
    self.vert_mean += self.bfm_offset

    with tf.name_scope('inputs'):
      self.ph_images = tf.compat.v1.placeholder(
          tf.float32, (self.batch_size, self.img_size, self.img_size, 3),
          'input_rgbas')
      self.input_images = (self.ph_images + 1) * 127.5

    self.infer_bfm()

  def infer_bfm(self):
    with tf.io.gfile.GFile('data/models/FaceReconModel.pb', 'rb') as f:
      face_rec_graph_def = tf.compat.v1.GraphDef()
      face_rec_graph_def.ParseFromString(f.read())

    def get_emb_coeff(net_name, inputs):
      resized = inputs
      if self.img_size != 224:
        resized = tf.image.resize(inputs, [224, 224])
      bgr_inputs = resized[..., ::-1]
      tf.import_graph_def(face_rec_graph_def, name=net_name,
                          input_map={'input_imgs:0': bgr_inputs})
      coeff = self.graph.get_tensor_by_name(net_name + '/coeff:0')
      return coeff

    self.coeff_test = get_emb_coeff('facerec_test', self.input_images)

    shape_coef, exp_coef, _, _, _, _ = utils.split_bfm09_coeff(self.coeff_test)

    shapePC = tf.constant(self.bfm.shapePC, dtype=tf.float32)
    expPC = tf.constant(self.bfm.expressionPC, dtype=tf.float32)

    neu_vert = tf.einsum('ij,aj->ai', shapePC, shape_coef)
    vertice = neu_vert + tf.einsum('ij,aj->ai', expPC, exp_coef)
    neu_vert = tf.reshape(
        neu_vert, [self.batch_size, self.num_bfm_vert, 3]) + self.vert_mean
    vertice = tf.reshape(
        vertice, [self.batch_size, self.num_bfm_vert, 3]) + self.vert_mean
    self.vert_test = vertice - tf.reduce_mean(self.vert_mean, axis=0,
                                              keepdims=True)
    self.neu_vert_test = neu_vert - tf.reduce_mean(self.vert_mean, axis=0,
                                                   keepdims=True)

  def predict(self, images, neutral=False):
    feed_dict = {self.ph_images: images}
    if neutral:
      fetches = [self.coeff_test, self.vert_test, self.neu_vert_test]
      coeffs, vertices, neu_vert = self.sess.run(fetches, feed_dict)
      return coeffs.squeeze(0), vertices.squeeze(0), neu_vert.squeeze(0)
    else:
      fetches = [self.coeff_test, self.vert_test]
      coeffs, vertices = self.sess.run(fetches, feed_dict)
      return coeffs.squeeze(0), vertices.squeeze(0)


def main():

  device = 'cpu'
  device = 'cuda'

  img_size = 1024
  input_dir = 'data\\dataset\\CelebAMask-HQ\\CelebA-HQ-img'
  output_dir = 'data\\dataset\\celeba_hq'

  os.makedirs(output_dir, exist_ok=True)

  nsh_face_mesh = meshio.Mesh('data/mesh/230/nsh_bfm_face.obj')

  cropper = ImageCropper(img_size, use_dlib=False)
  segmenter = Segment(device)
  transfer = Shape_Transfer('230', device=device)
  uv_creator = UVCreator('230', im_size=1024, uv_size=2048, device=device)

  R, T = renderer.look_at_view_transform(10, 0, 0)
  cameras = renderer.OpenGLPerspectiveCameras(znear=0.001, zfar=30.0,
                                              aspect_ratio=1.0, fov=12.5936,
                                              degrees=True, R=R, T=T,
                                              device=device)
  raster_settings = renderer.RasterizationSettings(image_size=1024,
                                                   blur_radius=0.0,
                                                   faces_per_pixel=1,
                                                   bin_size=0)
  rasterizer = renderer.MeshRasterizer(cameras=cameras,
                                       raster_settings=raster_settings)

  gpu_config = tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                        log_device_placement=False)
  # pylint: disable=no-member
  gpu_config.gpu_options.allow_growth = True
  with tf.compat.v1.Graph().as_default() as graph, tf.compat.v1.device(
      '/cpu'), tf.compat.v1.Session(config=gpu_config) as sess:

    reconstructor = Deep3DFace(sess, graph, img_size)

    # for img_path in tqdm(img_paths):
    for index in tqdm(range(30000)):
      img_path = os.path.join(input_dir, '{}.jpg'.format(index))
      image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      image = utils.center_crop_resize(image, img_size)
      image = cropper.crop_image(image)

      image = image / 127.5 - 1.0
      images = np.expand_dims(image, axis=0).astype(np.float32)

      coeff, bfm_vert = reconstructor.predict(images)
      _, _, _, [angles], _, [translation] = utils.split_bfm09_coeff(coeff[None])
      nsh_vert = transfer.transfer_shape(bfm_vert)
      nsh_face_vert = nsh_vert[uv_creator.nsh_face_start_idx:]

      transformer = transforms.Transform3d(device=device)
      transformer = transformer.rotate_axis_angle(angles[0], 'X', degrees=False)
      transformer = transformer.rotate_axis_angle(angles[1], 'Y', degrees=False)
      transformer = transformer.rotate_axis_angle(angles[2], 'Z', degrees=False)
      transformer = transformer.translate(translation[0], translation[1],
                                          translation[2])

      nsh_trans_vert = transformer.transform_points(
          torch.from_numpy(nsh_face_vert[None].astype(np.float32)).to(device))
      nsh_shift_vert = cameras.get_world_to_view_transform().transform_points(
          nsh_trans_vert).data.cpu().numpy()[0] * [-1, 1, -1]

      nsh_trans_mesh = Meshes(
          nsh_trans_vert,
          torch.from_numpy(nsh_face_mesh.triangles[None].astype(
              np.int32)).to(device))

      fragment = rasterizer(nsh_trans_mesh)
      pix_to_face = fragment.pix_to_face.data.cpu().numpy()
      visible_face = np.unique(pix_to_face)[1:]
      visible_vert = nsh_face_mesh.triangles[visible_face]
      visible_vert = np.unique(visible_vert)
      vert_alpha = np.zeros([nsh_shift_vert.shape[0], 1])
      vert_alpha[visible_vert] = 1
      nsh_shift_vert_alpha = np.concatenate([nsh_shift_vert, vert_alpha],
                                            axis=-1)

      _, segments = segmenter.segment(images, batch_size=1, all_seg=True)
      segment = segments[0]

      uv_map = uv_creator.create_nsh_uv_np(nsh_shift_vert_alpha,
                                           utils.to_uint8(image), segment, True)
      uv_map = cv2.resize(uv_map, (1024, 1024), interpolation=cv2.INTER_AREA)

      uv_map[..., 3] = uv_map[..., 3] + uv_map[..., 4] * 128

      np.save(os.path.join(output_dir, '{:>05d}_params.npy'.format(index)),
              coeff.astype(np.float32))

      image_with_seg = np.concatenate(
          [utils.to_uint8(image), 255 - segment[..., None]], axis=-1)
      imageio.imwrite(
          os.path.join(output_dir, '{:>05d}_image.png'.format(index)),
          image_with_seg)
      imageio.imwrite(os.path.join(output_dir, '{:>05d}_uv.png'.format(index)),
                      uv_map[..., :4])
      np.save(os.path.join(output_dir, '{:>05d}_nsh_vert.npy'.format(index)),
              nsh_face_vert.astype(np.float32))


if __name__ == '__main__':
  main()
