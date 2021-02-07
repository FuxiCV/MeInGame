from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

import utils


class Segment():
  """
    Label 00: background
    Label 01: face skin (excluding ears and neck)
    Label 02: left eyebrow
    Label 03: right eyebrow
    Label 04: left eye
    Label 05: right eye
    Label 06: nose
    Label 07: upper lip
    Label 08: inner mouth
    Label 09: lower lip
    Label 10: hair
    Label 11: hat
    Label 12: right ear
    Label 13: left ear
    Label 14: eye_g (glasses)
    Label 15: ear_r (earring or eardrop)
    Label 16: neck_l (necklace)
    Label 17: neck
    Label 18: cloth
  """

  def __init__(self, device, ckpt_path='data/models/torch_FaceSegment_300.pkl'):
    self.device = device
    self.model = resnet50(num_classes=19).to(device)
    if 'cuda' in device:
      self.model = torch.nn.DataParallel(self.model)
    self.model.eval()

    checkpoint = torch.load(ckpt_path, map_location=device)
    if 'cpu' in device:
      checkpoint['model_state'] = utils.fix_state_dict(checkpoint['model_state'])
    self.model.load_state_dict(checkpoint['model_state'])

  def inference(self, inputs, all_seg):
    # inputs should be in [-1, 1]
    image = (inputs + 1) * 127.5
    image = image - 128
    if len(image.shape) < 4:
      image = np.expand_dims(image, 0)
    image = image.transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).float().to(self.device)
    result = self.model(image)
    result = torch.argmax(result, dim=1)
    alpha = result.data.cpu().numpy()

    segment = alpha.astype(np.uint8)
    for x in [1, 2, 3, 4, 5, 6, 7, 9]:
      alpha[alpha == x] = 1
    alpha = np.where(alpha == 1, 1, 0).astype(np.float32)
    if not all_seg:
      return alpha
    else:
      return alpha, segment

  def segment(self, inputs, batch_size=4, all_seg=False):
    # input should be [-1, 1]
    num_input = inputs.shape[0]
    if num_input <= batch_size:
      return self.inference(inputs, all_seg)
    else:
      alphas = []
      if not all_seg:
        for i in range(0, num_input, batch_size):
          alpha = self.inference(inputs[i:i + batch_size], all_seg)
          alphas.append(alpha)
        return np.concatenate(alphas, axis=0)
      else:
        segments = []
        for i in range(0, num_input, batch_size):
          alpha, segment = self.inference(inputs[i:i + batch_size], all_seg)
          alphas.append(alpha)
          segments.append(segment)
        return np.concatenate(alphas, axis=0), np.concatenate(segments, axis=0)

  def segment_torch(self, images):
    images = images - 128
    if len(images.shape) < 4:
      images = images[None]
    if images.shape[-1] == 3:
      images = images.permute((0, 3, 1, 2))
    segments = self.model(images)
    segments = torch.argmax(segments, dim=1)
    return segments.type(torch.float32)


def conv3x3(in_planes, out_planes, stride=1):
  """3x3 convolution with padding"""
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes * self.expansion)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class ResNet(nn.Module):

  def __init__(self, block, layers, num_classes):
    # print("create resnet for semantic segmantation with %d num_classes" % (num_classes))
    self.inplanes = 64
    super(ResNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
    self.layer2 = self._make_layer(block, 128, layers[1], stride=1)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=1)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
    self.output = nn.Conv2d(512 * block.expansion, num_classes, kernel_size=1)
    self.upsample = nn.UpsamplingBilinear2d(scale_factor=8)
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
          nn.Conv2d(self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
          nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    #x = nn.functional.dropout2d(x)
    x = self.layer4(x)
    #x = nn.functional.dropout2d(x)
    x = self.output(x)
    x = self.upsample(x)
    return x


def resnet50(**kwargs):
  """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
  model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
  return model
