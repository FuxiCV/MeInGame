import os

import cv2
import numpy as np
import scipy.io as sio

from lib import face_align


class ImageCropper():

  # def __init__(self, predictor_path, img_size):
  def __init__(self, img_size, use_dlib=False):
    self.use_dlib = use_dlib
    if use_dlib:
      import dlib
      predictor_path = os.path.join('data', 'models',
                                    'shape_predictor_68_face_landmarks.dat')
      self.detector = dlib.get_frontal_face_detector()
      self.predictor = dlib.shape_predictor(predictor_path)
    else:
      self.predictor = face_align.FaceAlignment(face_align.LandmarksType._2D,
                                                device='cuda')
    self.load_lm3d()
    self.img_size = img_size

  def load_lm3d(self):
    Lm3D = sio.loadmat('data/models/similarity_Lm3D_all.mat')
    Lm3D = Lm3D['lm']

    # calculate 5 facial landmarks using 68 landmarks
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    Lm3D = np.stack([
        Lm3D[lm_idx[0], :],
        np.mean(Lm3D[lm_idx[[1, 2]], :], 0),
        np.mean(Lm3D[lm_idx[[3, 4]], :], 0), Lm3D[lm_idx[5], :],
        Lm3D[lm_idx[6], :]
    ], axis=0)
    self.lm3D = Lm3D[[1, 2, 0, 3, 4], :]

  def compute_lm_trans(self, lm):
    npts = lm.shape[1]
    A = np.zeros([2 * npts, 8])

    A[0:2 * npts - 1:2, 0:3] = self.lm3D
    A[0:2 * npts - 1:2, 3] = 1

    A[1:2 * npts:2, 4:7] = self.lm3D
    A[1:2 * npts:2, 7] = 1

    b = np.reshape(lm.transpose(), [2 * npts, 1])
    k, _, _, _ = np.linalg.lstsq(A, b, -1)

    R1 = k[0:3]
    R2 = k[4:7]
    sTx = k[3]
    sTy = k[7]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2
    t = np.stack([sTx, sTy], axis=0)

    return t, s

  def process_image(self, img, t, s, im_size):
    w0, h0, _ = img.shape
    # img = img.transform(img.size, Image.AFFINE,
    #                     (1, 0, t[0] - w0 / 2, 0, 1, h0 / 2 - t[1]))
    img = cv2.warpAffine(
        img, np.array([[1, 0, w0 / 2 - t[0, 0]], [0, 1, t[1, 0] - h0 / 2]]),
        (w0, h0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    half_size = im_size // 2
    scale = (102 / 224) * im_size

    w = (w0 / s * scale).astype(np.int32)
    h = (h0 / s * scale).astype(np.int32)
    img = cv2.resize(img, (w, h), cv2.INTER_LINEAR)
    # img = img.resize((w, h), resample=Image.BILINEAR)

    if w < im_size:
      # padding
      top = (im_size - h) // 2
      bottom = im_size - h - top
      left = (im_size - w) // 2
      right = im_size - w - left
      img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                   cv2.BORDER_REPLICATE)
    else:
      # cropping
      left = (w / 2 - half_size).astype(np.int32)
      right = left + im_size
      up = (h / 2 - half_size).astype(np.int32)
      below = up + im_size
      img = img[left:right, up:below].copy()

    return img

  def get_landmarks(self, image, need_bbox=False):
    if self.use_dlib:
      faces = self.detector(np.array(image[..., :3]), 1)
      landmarks = self.predictor(np.array(image[..., :3]), faces[0])
    else:
      # rects = self.detector.detectFaceOpenCVDnn(image)
      # rect = rects[0]
      # cur_landmark, roi_box = self.detector.detectLandmark(image, rect)
      # landmarks = self.detector.pts_480_to_68(cur_landmark).reshape(68, 2)
      # landmarks[:, 0] += roi_box[0]
      # landmarks[:, 1] += roi_box[1]
      im_size = np.array(image.shape[:2])
      im_256 = cv2.resize(image, (256, 256), cv2.INTER_AREA)
      landmarks, faces = self.predictor.get_landmarks(im_256)
      landmarks = landmarks[0]
      faces = faces[0]
      landmarks = np.round(landmarks * (im_size / 256).astype(np.float32))
    if need_bbox:
      return landmarks, faces
    else:
      return landmarks

  def upright_face(self, image, lm, im_size):
    # color = (0, 255, 0)
    # for _, (x, y) in enumerate(lm):
    #   cv2.circle(image, (int(x), int(y)), 1, color, -1, 8)
    # imageio.imwrite('tmp/rot_before.png', image)

    lm_top = np.mean(lm[:2], axis=0)
    lm_bottom = np.mean(lm[3:], axis=0)
    dist = lm_top - lm_bottom
    angle = np.math.atan(-dist[0] / dist[1])
    angle = np.rad2deg(angle)

    M = cv2.getRotationMatrix2D((im_size / 2, im_size / 2), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (im_size, im_size))

    # lm_rot = np.dot(M[:, :2], lm)
    lm_rot = np.einsum('ij,aj->ai', M[:, :2], lm)
    lm_rot += M[:, 2]
    lm_rot = np.round(lm_rot)

    # color = (0, 255, 255)
    # for _, (x, y) in enumerate(lm_rot):
    #   cv2.circle(rotated, (int(x), int(y)), 1, color, -1, 8)
    # imageio.imwrite('tmp/rot_after.png', rotated)

    return rotated, lm_rot

  def crop_image(self, image, im_size=None):
    # image should be uint8, channel order RGB
    if im_size is None:
      im_size = self.img_size

    landmarks = self.get_landmarks(image[..., :3])
    idxs = [[36, 37, 38, 39, 40, 41], [42, 43, 44, 45, 46, 47], [30], [48],
            [54]]
    lm = np.zeros([5, 2])
    for i in range(5):
      for j in idxs[i]:
        if self.use_dlib:
          lm[i] += np.array([landmarks.part(j).x, landmarks.part(j).y])
        else:
          lm[i] += np.array([landmarks[j, 0], landmarks[j, 1]])
      lm[i] = lm[i] // len(idxs[i])

    # image, lm = self.upright_face(image, lm, im_size)

    # new_image = Image.fromarray(image)
    # w0, h0 = new_image.size

    lm = np.stack([lm[:, 0], im_size - 1 - lm[:, 1]], axis=1)
    t, s = self.compute_lm_trans(lm.transpose())

    return self.process_image(image, t, s, im_size)

  def crop_images(self, images, im_size=None):
    outputs = []
    for image in images:
      outputs.append(self.crop_image(image, im_size))

    return np.array(outputs)
