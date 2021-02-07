import argparse
import os
import platform
from multiprocessing import set_start_method

import cv2
import torch

import utils
from uv_inpainting import UVInpainting


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-n', '--name', type=str, default='celeba_hq',
                      help='dataset name')
  parser.add_argument('-s', '--suffix', type=str, default='demo',
                      help='suffix for training name')
  parser.add_argument('-f', '--face_model', type=str, default='230',
                      choices=['230'], help='which NSH model is used')
  parser.add_argument('-b', '--bfm_version', type=str, default='face',
                      choices=['face', 'head'], help='which BFM model is used')
  parser.add_argument('-m', '--mode', type=str, default='test',
                      help='train or test')
  parser.add_argument('-c', '--cpu', default=False, action='store_true',
                      help='use cpu')
  parser.add_argument('-d', '--debug', default=False, action='store_true',
                      help='enable debug mode')
  parser.add_argument('-is', '--im_size', type=int, default=512,
                      help='image size')
  parser.add_argument('-us', '--uv_size', type=int, default=1024,
                      help='uvmap size')
  parser.add_argument('-rt', '--root_dir', type=str,
                      default='D:\\Codes\\uv_inpainting', help='root dir')
  parser.add_argument('-r', '--restore', default=False, action='store_true',
                      help='restore train')
  # parser.add_argument('-e', '--epochs', type=str, default='0,0,100', help='number of epochs')
  parser.add_argument('-e', '--epochs', type=str, default=400,
                      help='number of epochs')
  parser.add_argument('-bs', '--batch_size', type=int, default=2,
                      help='input batch size')
  parser.add_argument('-w', '--workers', type=int, default=8,
                      help='number of data loading threads')
  parser.add_argument('-li', '--log_interval', type=int, default=10,
                      help='log frequency')
  parser.add_argument('-si', '--sample_interval', type=int, default=1000,
                      help='sample frequency')
  parser.add_argument('-ci', '--ckpt_interval', type=int, default=1000,
                      help='save ckpt frequency')
  parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4,
                      help='Learning Rate')
  parser.add_argument('-b1', '--beta1', type=float, default=0.5,
                      help='Adam optimizer beta1')
  parser.add_argument('-b2', '--beta2', type=float, default=0.9,
                      help='Adam optimizer beta2')
  parser.add_argument('-gl', '--gan_loss', type=str, default='nsgan',
                      help='gan loss type')
  parser.add_argument('-wl', '--l1_weight', type=float, default=3,
                      help='l1 loss weight')
  parser.add_argument('-ws', '--sty_weight', type=float, default=1,
                      help='style loss weight')
  parser.add_argument('-wc', '--con_weight', type=float, default=1,
                      help='content loss weight')
  parser.add_argument('-wy', '--sym_weight', type=float, default=0.1,
                      help='symmetry loss weight')
  parser.add_argument('-wd', '--std_weight', type=float, default=3,
                      help='std loss weight')
  parser.add_argument('-wa', '--adv_weight', type=float, default=0.01,
                      help='adv loss weight')
  parser.add_argument('-sd', '--seed', type=int, default=1, help='random seed')
  parser.add_argument('-i', '--input', type=str, default='demo',
                      help='test input dir')
  parser.add_argument('-o', '--output', type=str, default=None,
                      help='test output dir')
  parser.add_argument('-st', '--start', type=int, default=0, help='start idx')
  parser.add_argument('-rn', '--rename', default=False, action='store_true',
                      help='rename file')

  return parser.parse_args()


def main():
  try:
    set_start_method('spawn')
  except RuntimeError:
    pass

  config = get_args()
  logger = utils.init_logger('x')

  # init device
  setattr(config, 'use_cuda', not config.cpu and torch.cuda.is_available())

  if config.use_cuda:
    device = 'cuda'
    torch.backends.cudnn.benchmark = True  # cudnn auto-tuner
  else:
    device = 'cpu'

  # set number of cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
  if config.mode == 'train':
    cv2.setNumThreads(0)

  if not os.path.isdir(config.root_dir):
    config.root_dir = '.'
  data_dir = os.path.join(config.root_dir, 'data', 'dataset', config.name)
  setattr(config, 'data_dir', data_dir)
  data_gt_dir = os.path.join(config.root_dir, 'data', 'dataset',
                             config.name + '_gt')
  setattr(config, 'data_gt_dir', data_gt_dir)

  if config.suffix is not None:
    config.name += '_' + config.suffix

  if config.ckpt_interval < config.sample_interval:
    config.ckpt_interval = config.sample_interval

  if config.mode == 'train':
    logger.info(config)
    logger.info('Start training...\n')

    model = UVInpainting(config, device)
    model.train()
  elif config.mode == 'test':
    # pylint: disable=import-error, import-outside-toplevel
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

    config.batch_size = 1
    config.workers = 0
    logger.info(config)
    print('Start testing...\n')

    gpu_config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False)
    # pylint: disable=no-member
    gpu_config.gpu_options.allow_growth = True
    if 'Windows' in platform.platform() or not config.use_cuda:
      tf_device = '/cpu'
    else:
      tf_device = '/gpu'

    with tf.Graph().as_default() as graph, tf.device(tf_device), tf.Session(
        config=gpu_config) as sess:

      model = UVInpainting(config, device, sess, graph)
      model.inpaint_model.load()
      model.inpaint_model.eval()

      if config.output is None:
        config.output = os.path.split(config.input)[-1]
        if config.suffix is not None:
          config.output += '_' + config.suffix
      if not os.path.isdir(config.input):
        config.input = os.path.join('data/test', config.input)
      if not config.output.startswith('results/'):
        config.output = os.path.join('results/', config.output)
      os.makedirs(config.output, exist_ok=True)

      test_image_paths = [
          os.path.join(config.input, x)
          for x in sorted(os.listdir(config.input))
      ]

      for i, path in enumerate(test_image_paths):
        if i < config.start:
          continue
        if config.rename:
          name = i
        else:
          name = os.path.split(path)[1].split('.')[0]
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        model.predict(image, config.output, name, deploy=True,
                      face_model=config.face_model)
        logger.info('Saved results from %s to %s/%s', path, config.output, name)


if __name__ == '__main__':
  main()
