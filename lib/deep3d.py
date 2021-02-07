import os

import numpy as np
import scipy.io as sio
# pylint: disable=import-error
import tensorflow.compat.v1 as tf

import utils
from lib import meshio


class Deep3DFace():

  def __init__(self, sess, graph, bfm_version='face', img_size=224,
               batch_size=1):
    self.sess = sess
    if graph is None:
      self.graph = tf.get_default_graph()
    else:
      self.graph = graph
    self.img_size = img_size
    self.batch_size = batch_size
    self.bfm = BFM_model('.', 'data/models/bfm2009_{}.mat'.format(bfm_version))

    self.refer_mesh = meshio.Mesh('data/mesh/bfm09_{}.obj'.format(bfm_version))
    self.num_bfm_vert = self.refer_mesh.vertices.shape[0]
    self.vert_mean = np.reshape(self.bfm.shapeMU, [-1, 3])
    bfm_eye_offset = meshio.Mesh(
        'data/mesh/bfm09_face_offset_eye.obj').vertices.astype(np.float32)
    bfm_eye_offset = bfm_eye_offset - self.refer_mesh.vertices.astype(
        np.float32)
    self.vert_mean += bfm_eye_offset * 0.7
    bfm_offset = meshio.Mesh('data/mesh/bfm09_face_offset.obj').vertices.astype(
        np.float32)
    bfm_offset = bfm_offset - self.refer_mesh.vertices.astype(np.float32)
    self.vert_mean += bfm_offset * 0.3

    with tf.name_scope('inputs'):
      self.ph_images = tf.placeholder(
          tf.float32, (self.batch_size, self.img_size, self.img_size, 3),
          'input_rgbas')
      self.input_images = (self.ph_images + 1) * 127.5

    self.infer_bfm()

  def infer_bfm(self):
    assert os.path.isfile('data/models/FaceReconModel.pb')
    with tf.io.gfile.GFile('data/models/FaceReconModel.pb', 'rb') as f:
      face_rec_graph_def = tf.GraphDef()
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

    shape_coef, exp_coef, color_coef, _, _, _ = utils.split_bfm09_coeff(
        self.coeff_test)

    shapePC = tf.constant(self.bfm.shapePC, dtype=tf.float32)
    expPC = tf.constant(self.bfm.expressionPC, dtype=tf.float32)
    colorMU = tf.constant(self.bfm.colorMU, dtype=tf.float32)
    colorPC = tf.constant(self.bfm.colorPC, dtype=tf.float32)

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

    colors = tf.einsum('ij,aj->ai', colorPC, color_coef) + colorMU
    colors = tf.clip_by_value(colors, 0.0, 255.0)
    self.colors = tf.reshape(colors, [self.batch_size, self.num_bfm_vert, 3])

  def predict(self, images, neutral=False, color=False):
    # images should be uint8, RGB order
    images = images.astype(np.float32) / 127.5 - 1.0
    feed_dict = {self.ph_images: images}
    if neutral:
      if color:
        fetches = [
            self.coeff_test, self.vert_test, self.neu_vert_test, self.colors
        ]
        coeffs, vertices, neu_vert, colors = self.sess.run(fetches, feed_dict)
        return coeffs.squeeze(0), vertices.squeeze(0), neu_vert.squeeze(
            0), colors.squeeze(0)
      else:
        fetches = [self.coeff_test, self.vert_test, self.neu_vert_test]
        coeffs, vertices, neu_vert = self.sess.run(fetches, feed_dict)
        return coeffs.squeeze(0), vertices.squeeze(0), neu_vert.squeeze(0)
    else:
      if color:
        fetches = [self.coeff_test, self.vert_test, self.colors]
        coeffs, vertices, colors = self.sess.run(fetches, feed_dict)
        return coeffs.squeeze(0), vertices.squeeze(0), colors.squeeze(0)
      else:
        fetches = [self.coeff_test, self.vert_test]
        coeffs, vertices = self.sess.run(fetches, feed_dict)
        return coeffs.squeeze(0), vertices.squeeze(0)


class BFM_model(object):

  def __init__(self, root_dir, path):
    super(BFM_model, self).__init__()

    self.root_dir = root_dir
    self.path = os.path.join(root_dir, path)
    self.load_BFM09()

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
