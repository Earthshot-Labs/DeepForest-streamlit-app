from functools import partial
import tifffile
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

def configure(opt):
    opt.format = 'jpg'
    opt.n_df = 64
    opt.flip = False
    opt.VGG_loss = False
    if opt.image_height == 512:
        opt.half = True
    elif opt.image_height == 1024:
        opt.half = False
    opt.image_size = (512, 512) if opt.half else (1024, 1024)
    opt.n_gf = 64 if opt.half else 32

def get_grid(input, is_real=True):
    if is_real:
        grid = torch.FloatTensor(input.shape).fill_(1.0)

    elif not is_real:
        grid = torch.FloatTensor(input.shape).fill_(0.0)

    return grid


def get_norm_layer(type):
    if type == 'BatchNorm2d':
        layer = partial(nn.BatchNorm2d, affine=True)

    elif type == 'InstanceNorm2d':
        layer = partial(nn.InstanceNorm2d, affine=False)

    return layer


def get_pad_layer(type):
    if type == 'reflection':
        layer = nn.ReflectionPad2d

    elif type == 'replication':
        layer = nn.ReplicationPad2d

    elif type == 'zero':
        layer = nn.ZeroPad2d

    else:
        raise NotImplementedError(
            "Padding type {} is not valid. Please choose among ['reflection', 'replication', 'zero']".format(type))

    return layer


class Manager(object):
    def __init__(self, opt):
        self.opt = opt

    @staticmethod
    def adjust_dynamic_range(data, drange_in, drange_out):
        if drange_in != drange_out:
            scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (
                    np.float32(drange_in[1]) - np.float32(drange_in[0]))
            bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
            data = data * scale + bias
        return data

    def tensor2image(self, image_tensor):
        np_image = image_tensor.squeeze().cpu().float().numpy()
        if len(np_image.shape) == 3:
            np_image = np.transpose(np_image, (1, 2, 0))  # HWC
        else:
            pass

        np_image = self.adjust_dynamic_range(np_image, drange_in=[-1., 1.], drange_out=[0, 255])
        np_image = np.clip(np_image, 0, 255).astype(np.uint8)
        return np_image

    def save_image(self, image_tensor, path):
        np_image = self.tensor2image(image_tensor)
        if np_image.shape[-1] > 3:
            tifffile.imwrite(path.replace('.jpg', '.tif'), np_image)
            np_image_mask = np_image[:, :, 3]
            np_image = np_image[:, :, 0:3]
            tifffile.imwrite(path.replace('_input', '_mask').replace('.jpg', '.png'), np_image_mask)
        pil_image = Image.fromarray(np_image)
        pil_image.save(path)

    def multi_dil(self, im,num):
        from skimage.morphology import dilation
        for i in range(num):
            im = dilation(im)
        return im

    def match_colors(self, img, img_color_ref):
        # TODO refactor
        factors = []   # 1.1926961073858011, 0.8401638792459487, 0.7082300372761956
        if len(factors) > 0:
            factor_r = factors[0]
            factor_g = factors[1]
            factor_b = factors[2]
        else:
            factor_r = np.mean(img_color_ref[:, :, 0]) / np.mean(img[:, :, 0])
            # TODO have a slidder in app?
            if factor_r > 1.4:
                factor_r = 1.0
            factor_g = np.mean(img_color_ref[:, :, 1]) / np.mean(img[:, :, 1])
            if factor_g > 1.5:
                factor_g = 1.4
            factor_b = np.mean(img_color_ref[:, :, 2]) / np.mean(img[:, :, 2])
            if factor_b > 1.5:
                factor_b = 1.4

        img_fixed_color = img.copy()[:, :, 0:3]
        # R
        img_fixed_color[:, :, 0] = img[:, :, 0] * factor_r
        print(factor_r)

        # G
        img_fixed_color[:, :, 1] = img[:, :, 1] * factor_g
        print(factor_g)

        # B
        img_fixed_color[:, :, 2] = img[:, :, 2] * factor_b
        print(factor_b)
        # TODO remove bright spots
        return img_fixed_color

    def save_image_overlay(self, fake, input, path):
        np_image_fake = self.tensor2image(fake)
        np_image_input = self.tensor2image(input)
        img_fixed_color = self.match_colors(np_image_fake, np_image_input)

        mask = np_image_input[:, :, -1]
        mask = self.multi_dil(mask, 5)
        np_image_input_RGB = np_image_input[:, :, 0:3]

        img_fixed_color[mask == 0] = np_image_input_RGB[mask == 0]
        pil_image = Image.fromarray(img_fixed_color)
        pil_image.save(path)
