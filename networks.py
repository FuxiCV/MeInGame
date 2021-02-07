import math

import torch
import torch.nn as nn


# pylint: disable=abstract-method
class BaseNetwork(nn.Module):

  def __init__(self):
    super(BaseNetwork, self).__init__()

  def init_weights(self, init_type='kaiming', gain=0.02):
    '''
    initialize network's weights
    init_type: normal | xavier | kaiming | orthogonal
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
    '''

    def init_func(m):
      classname = m.__class__.__name__
      if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
          nn.init.normal_(m.weight.data, 0.0, gain)
        elif init_type == 'xavier':
          nn.init.xavier_normal_(m.weight.data, gain=gain)
        elif init_type == 'kaiming':
          nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
          nn.init.orthogonal_(m.weight.data, gain=gain)

        if hasattr(m, 'bias') and m.bias is not None:
          nn.init.constant_(m.bias.data, 0.0)

      elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, gain)
        nn.init.constant_(m.bias.data, 0.0)

    self.apply(init_func)


class Generator(BaseNetwork):

  def __init__(self, im_in_chn, uv_in_chn, config, init_weights=True):
    super(Generator, self).__init__()
    self.im_size = config.im_size
    self.uv_size = config.uv_size
    n_out = 32

    self.im_enc1 = GatedConv2d(im_in_chn, n_out, 7, 2)
    self.im_enc2 = GatedConv2d(n_out, n_out * 2, 5, 2)
    self.im_enc3 = GatedConv2d(n_out * 2, n_out * 4, 5, 2)
    self.im_enc4 = GatedConv2d(n_out * 4, n_out * 8, 3, 2)
    self.im_enc5 = GatedConv2d(n_out * 8, n_out * 8, 3, 2)

    encoder_light = []
    encoder_light.append(nn.AvgPool2d((self.im_size // 32, self.im_size // 32)))
    encoder_light.append(Flatten())
    encoder_light.append(nn.Linear(n_out * 8, n_out * 8))
    encoder_light.append(nn.ReLU())
    encoder_light.append(nn.Linear(n_out * 8, 12))
    encoder_light.append(nn.Tanh())
    self.encoder_light = nn.Sequential(*encoder_light)

    if self.uv_size > self.im_size:
      self.uv_enc0 = GatedConv2d(uv_in_chn, n_out, 7, 2)
      self.uv_enc1 = GatedConv2d(n_out, n_out, 7, 2)
    else:
      self.uv_enc1 = GatedConv2d(uv_in_chn, n_out, 7, 2)
    self.uv_enc2 = GatedConv2d(n_out, n_out * 2, 5, 2)
    self.uv_enc3 = GatedConv2d(n_out * 2, n_out * 4, 5, 2)
    self.uv_enc4 = GatedConv2d(n_out * 4, n_out * 8, 3, 2)
    self.uv_enc5 = GatedConv2d(n_out * 8, n_out * 8, 3, 2)

    dilateds = []
    dilateds.append(GatedConv2d(n_out * 16, n_out * 16, 3, 1, dilation=2))
    if self.im_size // 32 >= 4:
      dilateds.append(GatedConv2d(n_out * 16, n_out * 16, 3, 1, dilation=4))
      if self.im_size // 32 >= 8:
        dilateds.append(GatedConv2d(n_out * 16, n_out * 16, 3, 1, dilation=8))
        if self.im_size // 32 >= 16:
          dilateds.append(GatedConv2d(n_out * 16, n_out * 16, 3, 1, dilation=16))
    self.dilateds = nn.Sequential(*dilateds)

    self.uv_dec1 = GatedDeConv2d(2, n_out * 16, n_out * 8, 5, 1)
    self.uv_dec2 = GatedDeConv2d(2, n_out * 8 * 3, n_out * 4, 5, 1)
    self.uv_dec3 = GatedDeConv2d(2, n_out * 4 * 3, n_out * 2, 5, 1)
    self.uv_dec4 = GatedDeConv2d(2, n_out * 2 * 3, n_out * 1, 5, 1)
    self.uv_dec5 = GatedDeConv2d(2, n_out * 1 * 3, n_out, 3, 1)
    if self.uv_size > self.im_size:
      self.uv_dec6 = GatedDeConv2d(2, n_out * 2, n_out, 3, 1)
      # self.uv_dec7 = GatedConv2d(n_out, 3, 3, 1, activation=nn.Tanh())
      self.uv_dec7 = GatedConv2d(n_out, 3, 3, 1)
      self.uv_dec8 = GatedConv2d(3, 3, 3, 1, activation=nn.Tanh())
    else:
      # self.uv_dec6 = GatedConv2d(n_out, 3, 3, 1, activation=nn.Tanh())
      self.uv_dec6 = GatedConv2d(n_out, 3, 3, 1)
      self.uv_dec7 = GatedConv2d(3, 3, 3, 1, activation=nn.Tanh())

    if init_weights:
      self.init_weights()

  def forward(self, im, uv):
    im1 = self.im_enc1(im)
    im2 = self.im_enc2(im1)
    im3 = self.im_enc3(im2)
    im4 = self.im_enc4(im3)
    im5 = self.im_enc5(im4)

    light = self.encoder_light(im5)

    if self.uv_size > self.im_size:
      uv0 = self.uv_enc0(uv)
      uv1 = self.uv_enc1(uv0)
    else:
      uv1 = self.uv_enc1(uv)
    uv2 = self.uv_enc2(uv1)
    uv3 = self.uv_enc3(uv2)
    uv4 = self.uv_enc4(uv3)
    uv5 = self.uv_enc5(uv4)

    cat = torch.cat([im5, uv5], dim=1)
    dilate = self.dilateds(cat)

    dec1 = self.uv_dec1(dilate)
    dec2 = torch.cat([dec1, im4, uv4], dim=1)
    dec2 = self.uv_dec2(dec2)
    dec3 = torch.cat([dec2, im3, uv3], dim=1)
    dec3 = self.uv_dec3(dec3)
    dec4 = torch.cat([dec3, im2, uv2], dim=1)
    dec4 = self.uv_dec4(dec4)
    dec5 = torch.cat([dec4, im1, uv1], dim=1)
    dec5 = self.uv_dec5(dec5)
    if self.uv_size > self.im_size:
      dec6 = torch.cat([dec5, uv0], dim=1)
      dec6 = self.uv_dec6(dec6)
      # out = self.uv_dec7(dec6)
      dec7 = self.uv_dec7(dec6)
      out = self.uv_dec8(dec7)
    else:
      # out = self.uv_dec6(dec5)
      dec6 = self.uv_dec6(dec5)
      out = self.uv_dec7(dec6)

    return out, light


class ImageDiscriminator(BaseNetwork):

  def __init__(self, in_channels, config, init_weights=True):
    super(ImageDiscriminator, self).__init__()
    self.config = config
    n_out = 32

    discriminator = []
    discriminator.append(GatedConv2d(in_channels, n_out, 5, 2))
    discriminator.append(GatedConv2d(n_out, n_out * 2, 5, 2))
    discriminator.append(GatedConv2d(n_out * 2, n_out * 4, 5, 2))
    discriminator.append(GatedConv2d(n_out * 4, n_out * 8, 5, 2))
    if config.gan_loss == 'wgan':
      discriminator.append(GatedConv2d(n_out * 8, 1, 5, 1))
      self.linear = nn.Linear((config.im_size // 16)**2, 1)
    else:
      discriminator.append(GatedConv2d(n_out * 8, 1, 5, 1, activation=None))
      discriminator.append(nn.Sigmoid())

    self.discriminator = nn.Sequential(*discriminator)

    if init_weights:
      self.init_weights()

  def forward(self, x):
    out = self.discriminator(x)
    if self.config.gan_loss == 'wgan':
      out = out.view(self.config.batch_size, -1)
      out = self.linear(out)
    return out


class UVMapDiscriminator(BaseNetwork):

  def __init__(self, in_channels, config, init_weights=True):
    super(UVMapDiscriminator, self).__init__()
    self.config = config
    n_out = 32

    discriminator = []
    discriminator.append(GatedConv2d(in_channels, n_out, 5, 2))
    discriminator.append(GatedConv2d(n_out, n_out * 2, 5, 2))
    discriminator.append(GatedConv2d(n_out * 2, n_out * 4, 5, 2))
    discriminator.append(GatedConv2d(n_out * 4, n_out * 8, 5, 2))
    if config.uv_size > config.im_size:
      discriminator.append(GatedConv2d(n_out * 8, n_out * 8, 5, 2))
    if config.gan_loss == 'wgan':
      discriminator.append(GatedConv2d(n_out * 8, 1, 5, 1))
      self.linear = nn.Linear((config.im_size // 16)**2, 1)
    else:
      discriminator.append(GatedConv2d(n_out * 8, 1, 5, 1, activation=None))
      discriminator.append(nn.Sigmoid())
    self.discriminator = nn.Sequential(*discriminator)

    if init_weights:
      self.init_weights()

  def forward(self, x):
    out = self.discriminator(x)
    if self.config.gan_loss == 'wgan':
      out = out.view(self.config.batch_size, -1)
      out = self.linear(out)
    return out


def _get_padding(ksize, stride, dilation):
  return math.ceil((1 - stride + dilation * (ksize - 1)) / 2)


class GatedConv2d(nn.Module):

  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, dilation=1,
               groups=1, bias=True, padding_mode='zeros', activation=nn.ELU(inplace=True)):
    super(GatedConv2d, self).__init__()

    if padding is None:
      padding = _get_padding(kernel_size, stride, dilation)

    self.conv2d = nn.utils.spectral_norm(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                  padding_mode))
    self.conv2d_mask = nn.utils.spectral_norm(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                  padding_mode))
    self.sigmoid = nn.Sigmoid()
    self.activation = activation

  def forward(self, x):
    mask = self.conv2d_mask(x)
    x = self.conv2d(x)
    if self.activation is not None:
      x = self.activation(x) * self.sigmoid(mask)
    else:
      x = x * self.sigmoid(mask)
    return x


class GatedDeConv2d(nn.Module):

  def __init__(self, scale, in_channels, out_channels, kernel_size, stride=1, padding=None,
               dilation=1, groups=1, bias=True, padding_mode='zeros'):
    super(GatedDeConv2d, self).__init__()
    self.scale = scale
    self.conv2d = GatedConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation,
                              groups, bias, padding_mode)

  def forward(self, x):
    x = nn.functional.interpolate(x, scale_factor=self.scale)
    return self.conv2d(x)


class Flatten(nn.Module):

  def forward(self, x):
    batch_size = x.shape[0]
    return x.view(batch_size, -1)
