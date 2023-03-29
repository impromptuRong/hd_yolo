## copy some functions from utils_image.py
import numpy as np
import numbers
import random
import math
import torch
import cv2

import skimage
import skimage.io
import skimage.util
import skimage.transform
import skimage.morphology

from PIL import Image
from skimage.color import rgb2hsv, hsv2rgb, hed2rgb, rgb2hed, gray2rgb

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Wedge, Polygon, Rectangle
from matplotlib.collections import PatchCollection

SKIMAGE_VERSION = skimage.__version__


def get_size(x, default=None):
    # if x is None and default is None:
    #     raise ValueError(f"input and default cannot both be None.")
    x = x if x is not None else default
    
    return (x, x) if isinstance(x, numbers.Number) else x


def update_size(x, new_size):  # assign new size and keep type
    if isinstance(x, torch.Tensor):
        return x.new(new_size)
    elif isinstance(x, np.ndarray):
        return np.array(new_size)
    else:
        new_size = np.array(new_size)
        return type(x)([new_size[0], new_size[1]])


def unique_colors(x, channel_axis=None):
    if not channel_axis:
        return np.unique(x)
    else:
        return np.unique(x.reshape(channel_axis, x.shape[channel_axis]), axis=0)


def image_stats(x, channel_axis=None):
    if x is None:
        return None
    stats = [x.min(), x.max(), len(unique_colors(x, channel_axis))] if min(x.shape) > 0 else [None, None, None]
    return [x.shape, x.dtype] + stats


def img_as(dtype):
    """ Convert images between different data types. 
        (Note that: skimage.convert is not a public function. )
        If input image has the same dtype and range, function will do nothing.
        (This check is included in skimage.convert, so no need to implement it here. )
        https://github.com/scikit-image/scikit-image/blob/master/skimage/util/dtype.py
        dtype: a string or a python dtype or numpy.dtype: 
               'float', 'float32', 'float64', 'uint8', 'int32', 'int64', 'bool', 
               float, uint8, bool, int,
               np.floating, np.float32, np.uint8, np.int, np.bool, etc
    """
    dtype = np.dtype(dtype)
    # return lambda x: skimage.convert(x, dtype, force_copy=False)
    dtype_name = dtype.name
    if dtype_name.startswith('float'):
        # convert(image, np.floating, force_copy=False)
        if dtype_name == 'float32':
            return skimage.img_as_float32
        elif dtype_name == 'float64':
            return skimage.img_as_float64
        else:
            return skimage.img_as_float
    elif dtype_name == 'uint8':
        # convert(image, np.uint8, force_copy=False)
        return skimage.img_as_ubyte
    elif dtype_name.startswith('uint'):
        # convert(image, np.uint16, force_copy=False)
        return skimage.img_as_uint
    elif dtype_name.startswith('int'):
        # convert(image, np.int16, force_copy=False)
        return skimage.img_as_int
    elif dtype_name == 'bool':
        # convert(image, np.bool_, force_copy)
        return skimage.img_as_bool
    else:
        raise ValueError("%s is not a supported data type in skimage" % dtype_name)


def rgba2rgb(img, background=(0, 0, 0), binary_alpha=False):
    """ Remove alpha channel through alpha blending. 
        Equivalent but faster than skimage.color.rgba2rgb(img, background=(0, 0, 0))
    """
    if img.ndim < 3 or img.shape[-1] < 4:
        return img
    alpha_channel = img[..., -1:]
    if binary_alpha:
        res = img[..., :-1] & alpha_channel
    else:
        res = img[..., :-1] * (alpha_channel/255)
    if background is not None:
        res = np.where(alpha_channel, res, background)
    return res.astype(img.dtype)


def unpack_dict(kwargs, N):
    """ Unpack a dictionary of values into a list (N) of dictionaries. """
    return [dict((k, v[i]) for k, v in kwargs.items()) for i in range(N)]


def get_pad_width(input_size, output_size, pos='center'):
    output_size = output_size + input_size[len(output_size):]
    output_size = np.maximum(input_size, output_size)
    if pos == 'center':
        l = np.floor_divide(output_size - input_size, 2)
    elif pos == 'random':
        l = np.array([np.random.randint(0, _ + 1) for _ in output_size - input_size])
#     elif pos == 'tl':
#         l = np.array([0, 0])
#     elif pos == 'tr':
#         l = np.array([0, (output_size - input_size)[1]])
#     elif pos == 'bl':
#         l = np.array([(output_size - input_size)[0], 0])
#     elif pos == 'br':
#         l = (output_size - input_size)
    return list(zip(l, output_size - input_size - l))


def pad(img, size=None, pad_width=None, pos='center', mode='constant', **kwargs):
    """ Pad the input numpy array image with pad_width and to given size.
    Args:
        img (numpy array): Image to be resized.
        size (tuple): Desired output size. 
        pad_width (list of tuples): Desired pad_width. 
        pos: one of {'center, 'random'}, default is 'center'. if given
             size, the parameter will decide whether to put original 
             image in the center or a random location.
        mode: supported mode in skimage.util.pad
        kwargs: other parameters in skimage.util.pad
    
    pad_width and size can have same length as img, or 1d less than img.
    pad_width and size cannot be both None. If size = None, function will
    image with return img_size + pad_width. If pad_width = None, function 
    will return image with size. If both size and pad_width is not None,
    function will pad with pad_width first, then will try to meet size. 
    Function don't do any resize, rescale, crop process. Return img size 
    will be max(img.size+pad_width, size). 
    Returns:
        numpy array: Resized image.
    """
    if mode == 'constant':
        pars = {'constant_values': kwargs.setdefault('cval', 0.0)}
    elif mode == 'linear_ramp':
        pars = {'end_values': kwargs.setdefault('end_values', 0.0)}
    elif mode == 'reflect' or mode == 'symmetric':
        pars = {'reflect_type': kwargs.setdefault('reflect_type', 'even')}
    else:
        pars = {'stat_length': kwargs.setdefault('stat_length', None)}
    
    if pad_width is not None:
        pad_width = pad_width + [(0, 0)] * (img.ndim - len(pad_width))
        # img = skimage.util.pad(img, pad_width[:img.ndim], mode=mode, **pars)
        img = np.pad(img, pad_width[:img.ndim], mode=mode, **pars)
    
    if size is not None:
        pad_var = get_pad_width(img.shape, output_size=size, pos=pos)
        # img = skimage.util.pad(img, pad_var, mode=mode, **pars)
        img = np.pad(img, pad_var, mode=mode, **pars)
    
    return img


def get_crop_width(input_size, output_size, pos='center'):
    output_size = output_size + input_size[len(output_size):]
    output_size = np.minimum(input_size, output_size)
    if pos == 'center':
        l = np.floor_divide(input_size - output_size, 2)
    elif pos == 'random':
        l = [np.random.randint(0, _ + 1) for _ in input_size - output_size]        
    return list(zip(l, input_size - output_size - l))


def crop(img, size=None, crop_width=None, pos='center', **kwargs):
    """ Crop the input numpy array image with crop_width and to given size.
    Args:
        img (numpy array): Image to be resized.
        size (tuple): Desired output size. 
        crop_width (list of tuples): Desired crop_width. 
        pos: one of {'center, 'random'}, default is 'center'. if given
             size, the parameter will decide whether to put original 
             image in the center or a random location.
        kwargs: other parameters in skimage.util.crop, use default just fine.
    
    crop_width and size can have same length as img, or 1d less than img.
    crop_width and size cannot be both None. If size = None, function will
    return image with img_size - crop_width. If crop_width = None, function 
    will return image with size. If both size and crop_width is not None,
    function will crop with crop_width first, then will try to meet size. 
    Function don't do any resize, rescale and pad process. Return img size 
    will be min(img.size-pad_width, size). 
    Returns:
        numpy array: Resized image.
    """
    copy = kwargs.setdefault('copy', False)
    order = kwargs.setdefault('order', 'K')
    
    if crop_width is not None:
        crop_width = crop_width + [(0, 0)] * (img.ndim - len(crop_width))
        img = skimage.util.crop(img, crop_width[:img.ndim], copy=copy, order=order)
    
    if size is not None:
        crop_var = get_crop_width(img.shape, output_size=size, pos=pos)
        img = skimage.util.crop(img, crop_var, copy=copy, order=order)
    return img


def random_transform_pars(input_size, output_size, hyp):
    # hyp = {'degrees': 10, 'translate': .1, 'scale': .1, 'shear': 10, 'perspective': 0.0}
    pars = {
        'c_x': -input_size[1] / 2,
        'c_y': -input_size[0] / 2,
        'p_x': random.uniform(-hyp['perspective'], hyp['perspective']),
        'p_y': random.uniform(-hyp['perspective'], hyp['perspective']),
        'angle': random.uniform(-hyp['degrees'], hyp['degrees']),
        'scale': random.uniform(1 - hyp['scale'], 1 + hyp['scale']),
        'shear_x': random.uniform(-hyp['shear'], hyp['shear']),
        'shear_y': random.uniform(-hyp['shear'], hyp['shear']),
        't_x': random.uniform(0.5 - hyp['translate'], 0.5 + hyp['translate']) * output_size[1],
        't_y': random.uniform(0.5 - hyp['translate'], 0.5 + hyp['translate']) * output_size[0],
    }
    
    return pars


def estimate_matrix(pars):
    C = np.array([[1, 0, pars['c_x']], [0, 1, pars['c_y']], [0, 0, 1]])  # offset center
    P = np.array([[1, 0, 0], [0, 1, 0], [pars['p_x'], pars['p_y'], 1]])  # perspective

    # Affine, (rewrite)
    R = np.eye(3)
    R[:2] = cv2.getRotationMatrix2D(angle=pars['angle'], center=(0, 0), scale=pars['scale'])
    S = np.eye(3)
    S[0, 1] = math.tan(pars['shear_x'] * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(pars['shear_y'] * math.pi / 180)  # y shear (deg)
    A = S @ R

    T = np.array([[1, 0, pars['t_x']], [0, 1, pars['t_y']], [0, 0, 1]])  # translation
    
    return T @ A @ P @ C
    # return T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT


def warp_image(img, M, output_size=None, order=1, cval=0.):
    if (M == np.eye(3)).all() and img.shape[0] == output_size[0] and img.shape[1] == output_size[1]:
        return img
        
    h, w = output_size
    order = {0: cv2.INTER_NEAREST, 1: cv2.INTER_LINEAR, 3: cv2.INTER_CUBIC}[order]
    if isinstance(cval, numbers.Number):
        cval = (cval, cval, cval)
    if M[-1, :-1].any():
        return cv2.warpPerspective(img, M, dsize=(w, h), flags=order, borderValue=cval)
    else:  # affine
        return cv2.warpAffine(img, M[:2], dsize=(w, h), flags=order, borderValue=cval)


def warp_coords(p, M):
    xy = np.ones((len(p), 3))
    xy[:, :2] = p
    xy = xy @ M.T  # transform
    return xy[:, :2] / xy[:, 2:3] if M[-1, :-1].any() else xy[:, :2]  # perspective rescale or affine


def get_mask_bbox(mask):
    pos = np.where(mask > 0)
    if len(pos[0]):
        return [np.min(pos[1]), np.min(pos[0]), np.max(pos[1]), np.max(pos[0])]
    else:
        return None


def get_mask_area(mask):
    ## boolean mask, rle, polygon
    if isinstance(mask, np.ndarray):  # boolean mask
        return np.sum(mask > 0)
    elif isinstance(mask, dict):  # rle
        from pycocotools import mask as mask_utils
        return mask_utils.area(mask)
    else:  ## polygon
        return np.sum(np.ceil(polygon_areas(mask)))


def binary_mask_to_rle(x, compress=True):
    """ transfer a binary mask to rles. 
        compress = True will return compressed rle by pycocotools
        compress = False will return uncompressed rle
    """
    if compress:
        from pycocotools import mask as mask_utils
        return mask_utils.encode(np.asfortranarray(x.astype(np.uint8)))
    rle = {'counts': [], 'size': list(x.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(x.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle


## paired function for contiguous. The rle is actually transposed.
def rle_encode(x):
    from pycocotools import mask as mask_utils
    
    assert x.data.contiguous, f"input tensor need to be contiguous."
    return mask_utils.encode(x.astype(np.uint8).T)


def rle_decode(x, size):
    from pycocotools import mask as mask_utils
    m = mask_utils.decode({'size': size, 'counts': x}).T
    # m is a uint8 image with [0, 1], that's confusing.
    return m.astype(bool)


def binary_mask_to_polygon(x, level=0.5, flatten=False, mode='xy', scale=1.0):
    """ Transfer a binary mask to polygons. 
        flatten = False will return original result from skimage.measure.find_contours
        flatten = True will convert result to "coco" polygons
        skimage.measure.find_contours gives open contours when meet edge and corners.
        This will cause trouble when revert back, so always pad image by 1 pixel.
    """
    h, w = x.shape[0], x.shape[1]
    x_pad = pad(x, pad_width=[(1, 1), (1, 1)], mode='constant', cval=False)
    polygons = skimage.measure.find_contours(x_pad, level)  # old scrip use 0.5 for binary
    polygons = [np.stack([np.clip(p[:,0]-1, 0, h-1), 
                          np.clip(p[:,1]-1, 0, w-1)], axis=-1) 
                for p in polygons]
    
    if not mode.startswith('yx'):
        polygons = [p[..., -1::-1] for p in polygons]
    
    if scale != 1.0:
        polygons = [p * scale for p in polygons]
        
    if flatten == True:
        return [np.flip(_, axis=1).ravel().tolist() for _ in polygons]
    else:
        return polygons


def polygon_to_binary_mask(x, size, mode='xy'):
    masks = []
    for p in x:
        p = np.array(p) if mode.startswith('yx') else np.array(p)[:,::-1]
        if SKIMAGE_VERSION >= '0.16':  ## a function in scikit-learn 0.16
            from skimage.draw import polygon2mask
            mask = polygon2mask(image_shape=size, polygon=p)
        else:
            image_shape = size
            vertex_row_coords, vertex_col_coords = p.T
            fill_row_coords, fill_col_coords = skimage.draw.polygon(
                vertex_row_coords, vertex_col_coords, image_shape)
            mask = np.zeros(image_shape, dtype=np.bool)
            mask[fill_row_coords, fill_col_coords] = True
        masks.append(mask)
    
    return np.stack(masks).any(0)


def polygon_to_binary_mask_v2(x, size, mode='xy'):
    res = np.zeros(size)
    x = [_.round().astype(int) for _ in x]
    cv2.fillPoly(res, pts=x, color=1)
    
    return res > 0


class Mask(object):
    def __init__(self, x, size, mode, clip=True):
        assert mode in ['mask', 'masks', 'rle', 'rles', 'polygons', 'poly'], f"Unsupported mask_mode: {mode}."
        if mode.startswith('poly') and clip:
            x = [np.clip(_, 0, [size[1], size[0]]) for _ in x]
        self.m = x
        self.size = [size[0], size[1]]
        self.mode = mode
    
    def convert(self, mode=None, dtype=None):
        if mode is None:
            return self
        elif mode.startswith('poly'):
            return self.poly()
        elif mode.startswith('mask'):
            return self.mask(dtype)
        elif model.startswith('rle'):
            return self.rle()
        else:
            raise ValueError(f"{mode} is not supported.")
    
    def poly(self):
        if self.mode.startswith('poly'):
            m = self.m
        elif self.mode.startswith('rle'):
            m = rle_decode(self.m, self.size)
            m = binary_mask_to_polygon(m)
        elif self.mode.startswith('mask'):
            m = binary_mask_to_polygon(self.m)
        
        return Mask(m, self.size, 'poly', clip=False)
    
    def mask(self, dtype=None):
        if self.mode.startswith('poly'):
            m = polygon_to_binary_mask_v2(self.m, self.size)
        elif self.mode.startswith('rle'):
            m = rle_decode(self.m, self.size)
        elif self.mode.startswith('mask'):
            m = self.m
        
        if dtype is not None:
            m = img_as(dtype)(m)
        
        return Mask(m, self.size, 'mask')

    def rle(self):
        if self.mode.startswith('poly'):
            m = rle_encode(polygon_to_binary_mask_v2(self.m, self.size))
        elif self.mode.startswith('rle'):
            m = self.m
        elif self.mode.startswith('mask'):
            m = rle_encode(self.m)
        
        return Mask(m, self.size, 'rle')
    
    def box(self):
        if self.mode.startswith('poly'):
            x, y = np.concatenate(self.m, axis=0).T
            return np.array([x.min(), y.min(), x.max(), y.max()]) if any(x) else np.zeros((4,))  # xyxy
        elif self.mode.startswith('rle'):
            m = rle_decode(self.m, self.size)
            r, c = np.where(m > 0)
            return np.array([c.min(), r.min(), c.max(), r.max()])
        elif self.mode.startswith('mask'):
            r, c = np.where(self.m > 0)
            return np.array([c.min(), r.min(), c.max(), r.max()])
    
    def hflip(self):
        if self.mode.startswith('poly'):
            m = [np.abs(_ - [self.size[1], 0]) for _ in self.m]
        elif self.model.startswith('rle'):
            m = rle_decode(self.m, self.size)
            m = rle_encode(m[:, ::-1, ...])
        elif self.mode.startswith('mask'):
            m = self.m[:, ::-1, ...]
        
        return Mask(m, self.size, self.mode, clip=False)

    def vflip(self):
        if self.mode.startswith('poly'):
            m = [np.abs(_ - [0, self.size[0]]) for _ in self.m]
        elif self.model.startswith('rle'):
            m = rle_decode(self.m, self.size)
            m = rle_encode(m[::-1, ...])
        elif self.mode.startswith('mask'):
            m = self.m[::-1, ...]
        
        return Mask(m, self.size, self.mode, clip=False)

    def t(self):
        if self.mode.startswith('poly'):
            m = [_[:, [1,0]] for _ in self.m]
        elif self.model.startswith('rle'):
            m = rle_decode(self.m, self.size)
            m = rle_encode(m.T)
        elif self.mode.startswith('mask'):
            m = self.m.T
        
        return Mask(m, [self.size[1], self.size[0]], self.mode, clip=False)

    def __bool__(self):
        if self.mode.startswith('poly'):
            return len(self.m) > 0
        elif self.mode.startswith('rle'):
            return len(self.m) > 0
        elif self.mode.startswith('mask'):
            return m.sum() > 0


def random_adjust_color(img, global_mean=0.01, channel_mean=0.01, channel_sigma=0.2):
    """ A simple and effective random color augmentation (from Shidan). 
        The function treat last dimension as channel.
        global_mean: the relative mean add to all channel.
        channel_mean: the relative mean add to each channel.
        channel_sigma: the relative variace add to each channel.
    """
    dtype = img.dtype
    img = img_as('uint8')(img) * 1.0
    n_channel = img.shape[-1]
    # 1) add global mean and channel mean
    img += np.random.randn() * global_mean
    # 2) add a shifting & variance for each channel
    channel_means = np.random.randn(n_channel) * channel_mean
    channel_vars = np.clip(np.random.randn(n_channel) * channel_sigma, -channel_sigma, channel_sigma)
    img += img * channel_vars + channel_means
    # print(image_stats(img))
    
    return img_as(dtype)(np.clip(img/255., 0., 1.))


class ColorDodge(object):
    """ Randomly color augmentation with mean and std.
    Args:
        global_mean: the relative mean add to all channel.
        channel_mean: the relative mean add to each channel.
        channel_sigma: the relative variace add to each channel.
    """
    def __init__(self, global_mean=0.01, channel_mean=0.01, channel_sigma=0.2, p=0.5):
        self.global_mean = global_mean
        self.channel_mean = channel_mean
        self.channel_sigma = channel_sigma
        self.p = p
    
    def __call__(self, images, kwargs=None):
        return [random_adjust_color(img, self.global_mean, self.channel_mean, self.channel_sigma)
                if img is not None and np.random.random() < self.p else img
                for img in images]

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'global_mean={0}'.format(self.global_mean)
        format_string += ', channel_mean={0}'.format(self.channel_mean)
        format_string += ', channel_sigma={0}'.format(self.channel_sigma)
        return format_string


def adjust_brightness(img, brightness_factor):
    """ Adjust brightness of an Image. """
    min_val, max_val = skimage.dtype_limits(img)
    return np.clip(img * brightness_factor, min_val, max_val).astype(img.dtype)


def adjust_contrast(img, contrast_factor):
    """ Adjust contrast of an Image. """
    min_val, max_val = skimage.dtype_limits(img)
    degenerate = np.mean(rgb2gray(img))
    res = degenerate * (1-contrast_factor) + img * contrast_factor
    return np.clip(res, min_val, max_val).astype(img.dtype)


def adjust_saturation(img, saturation_factor):
    """ Adjust color saturation of an image (PIL ImageEnhance.Color). """
    min_val, max_val = skimage.dtype_limits(img)
    degenerate = rgb2gray(img, 1)
    res = degenerate * (1-saturation_factor) + img * saturation_factor
    return np.clip(res, min_val, max_val).astype(img.dtype)


def adjust_hue(img, hue_factor):
    """ Adjust hue of an image.
        hue_factor is the amount of shift in H channel [-0.5, 0.5].
    """
    if not(-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))
    
    hsv = rgb2hsv(img)
    hsv[..., 0] *= (1 + hue_factor)
    res = hsv2rgb(np.clip(hsv, 0.0, 1.0)) # float image
    return img_as(img.dtype)(res)


def adjust_gamma(img, gamma=1, gain=1):
    """ Perform gamma correction (Power Law Transform) on an image. """
    return skimage.exposure.adjust_gamma(img, gamma=gamma, gain=gain)


def random_color_jitter(img, factors):
    func_list = {'brightness': adjust_brightness, 'contrast': adjust_contrast,
                 'saturation': adjust_saturation, 'hue': adjust_hue}
    for key, val in factors:
        img = func_list[key](img, val)
    return img


# def random_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
#     # HSV color-space augmentation, slower than cv2
#     if hgain or sgain or vgain:
#         r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]
#         hsv = rgb2hsv(img) * (1 + r)
#         img = img_as(img.dtype)(hsv2rgb(np.clip(hsv, 0.0, 1.0)))
    
#     return img


def random_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5, p=0.5):
    # HSV color-space augmentation
    if random.random() < 0.5 and (hgain or sgain or vgain):
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_RGB2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        im = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB)

    return im


## Copy from torch.vision
class ColorJitter(object):
    """ Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0 <= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=0.5):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)
        self.p = p

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def get_params(self):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        pars = []
        
        if self.brightness is not None:
            pars.append(('brightness', np.random.uniform(self.brightness[0], self.brightness[1])))
        
        if self.contrast is not None:
            pars.append(('contrast', np.random.uniform(self.contrast[0], self.contrast[1])))

        if self.saturation is not None:
            pars.append(('saturation', np.random.uniform(self.saturation[0], self.saturation[1])))

        if self.hue is not None:
            pars.append(('hue', np.random.uniform(self.hue[0], self.hue[1])))
        
        np.random.shuffle(pars)
        return pars

    def __call__(self, images, kwargs=None):
        """
        Args:
            img (PIL Image): Input image.
        Returns:
            PIL Image: Color jittered image.
        """
        pars = self.get_params()
        return [random_color_jitter(img, pars)
                if img is not None and np.random.random() < self.p else img
                for img in images]

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string


# def overlay_detections(ax, bboxes=None, labels=None, masks=None, scores=None,
#                        labels_color=None, labels_text=None,
#                        show_bboxes=True, show_texts=True, show_masks=True, show_scores=True,
#                       ):
#     # sns.color_palette()
#     _cmap = [
#         (0.12156862745098039, 0.4666666666666667, 0.7058823529411765), 
#         (1.0, 0.4980392156862745, 0.054901960784313725), 
#         (0.17254901960784313, 0.6274509803921569, 0.17254901960784313), 
#         (0.8392156862745098, 0.15294117647058825, 0.1568627450980392), 
#         (0.5803921568627451, 0.403921568627451, 0.7411764705882353), 
#         (0.5490196078431373, 0.33725490196078434, 0.29411764705882354), 
#         (0.8901960784313725, 0.4666666666666667, 0.7607843137254902), 
#         (0.4980392156862745, 0.4980392156862745, 0.4980392156862745), 
#         (0.7372549019607844, 0.7411764705882353, 0.13333333333333333), 
#         (0.09019607843137255, 0.7450980392156863, 0.8117647058823529)
#     ]
    
#     if bboxes is None:
#         assert masks is not None
#         bboxs = [get_mask_bbox(_) for _ in masks]
    
#     if labels is None:
#         labels = [1] * len(bboxes)
#     labels = np.array(labels)  # compatible with torch tensor
    
#     if labels_color is None:
#         labels_color = {k: _cmap[i%10] for i, k in enumerate(np.unique(labels))}
    
#     if labels_text is None:
#         labels_text = {k: str(k) for k in np.unique(labels)}
    
#     if scores is None:
#         show_scores = False
    
#     if masks is None:
#         show_masks = False
    
#     # convert color to np.array and normalize to 0~1
#     tmp_c = {}
#     for k, v in labels_color.items():
#         if isinstance(v, str):
#             tmp_c[k] = matplotlib.colors.to_rgba(v)
#         else:
#             v = np.array(v) * 1.0
#             if (v > 1.0).any():
#                 v /= 255.
#             tmp_c[k] = v
#     labels_color = tmp_c
    
    
#     for i, (bbox, label) in enumerate(zip(bboxes, labels)):
#         if label not in labels_color:
#             continue
#         c = labels_color.get(label, np.array([0., 0., 0.]))
        
#         y1, x1, y2, x2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

#         if show_bboxes:
#             if x1 == x2 and y1 == y2:
#                 ax.plot([y1], [x1], marker='o', markersize=3, color=c)
#             else:
#                 b = Rectangle((y1, x1), y2-y1, x2-x1, linewidth=2,
#                               alpha=0.5, linestyle="solid", edgecolor=c, facecolor="none")
#                 ax.add_patch(b)
        
#         if show_texts and show_scores:
#             text = "{}={:.4f}".format(labels_text[label], scores[i])
#         elif show_texts:
#             text = "{}".format(labels_text[label])
#         elif show_scores:
#             text = "{:.4f}".format(scores[i])
#         else:
#             text = None
        
#         if text is not None:
#             text_pad = 4
#             ax.annotate(text, (y1, x1), xytext=(text_pad, text_pad), textcoords='offset pixels',
#                         bbox=dict(facecolor=c, edgecolor='none', alpha=0.3, pad=text_pad), color='b', # backgroundcolor=c, 
#                         fontsize=6, ha='left', va='bottom')
#             # ax.annotate(text, (y1, x1), color=c, weight='bold', ) # fontsize=6, ha='center', va='center')

#         if show_masks:
#             mask = masks[i]
#             if mask is not None:
#                 if isinstance(mask, Mask):
#                     polygon = mask.poly().m
#                 else:
#                     polygon = mask if isinstance(mask, list) else binary_mask_to_polygon(np.array(mask))
#                 for verts in polygon:
#                     p = Polygon(np.array(verts), edgecolor=c, facecolor=c, alpha=0.3)
#                     ax.add_patch(p)

def overlay_detections(ax, bboxes=None, labels=None, masks=None, scores=None,
                       labels_color=None, labels_text=None,
                       show_bboxes=True, show_texts=True, show_masks=True, show_scores=True,
                      ):
    # sns.color_palette()
    _cmap = [
        (0.12156862745098039, 0.4666666666666667, 0.7058823529411765), 
        (1.0, 0.4980392156862745, 0.054901960784313725), 
        (0.17254901960784313, 0.6274509803921569, 0.17254901960784313), 
        (0.8392156862745098, 0.15294117647058825, 0.1568627450980392), 
        (0.5803921568627451, 0.403921568627451, 0.7411764705882353), 
        (0.5490196078431373, 0.33725490196078434, 0.29411764705882354), 
        (0.8901960784313725, 0.4666666666666667, 0.7607843137254902), 
        (0.4980392156862745, 0.4980392156862745, 0.4980392156862745), 
        (0.7372549019607844, 0.7411764705882353, 0.13333333333333333), 
        (0.09019607843137255, 0.7450980392156863, 0.8117647058823529)
    ]
    
    if bboxes is None:
        assert masks is not None
        bboxs = [get_mask_bbox(_) for _ in masks]
    
    if labels is None:
        labels = [1] * len(bboxes)
    labels = np.array(labels)  # compatible with torch tensor
    
    if labels_color is None:  # give unique color to each type
        unique_labels = np.unique(labels)
        _cmap = sns.color_palette(n_colors=len(unique_labels))
        palette = {label: color for label, color in enumerate(unique_labels, _cmap)}
        color_array = [palette[label] for label in labels]
    elif isinstance(labels_color, str) and labels_color.startswith('object'):
        color_array = sns.color_palette(n_colors=len(labels))
    elif isinstance(labels_color, dict):
        # convert color to np.array and normalize to 0~1
        palette = {}
        for k, v in labels_color.items():
            if isinstance(v, str):
                palette[k] = matplotlib.colors.to_rgba(v)
            else:
                v = np.array(v) * 1.0
                if (v > 1.0).any():
                    v /= 255.
                palette[k] = v
        color_array = [palette.get(label, np.array([0., 0., 0.])) for label in labels]
    elif isinstance(labels_color, list):
        assert len(labels_color) == len(labels)
        color_array = []
        for v in labels_color:
            if isinstance(v, str):
                v = matplotlib.colors.to_rgba(v)
            else:
                v = np.array(v) * 1.0
                if (v > 1.0).any():
                    v /= 255.
            color_array.append(v)
    else:
        raise ValueError(f"Invalid labels_color for display.")
    
    if labels_text is None:
        labels_text = {k: str(k) for k in np.unique(labels)}
    
    if scores is None:
        show_scores = False
    
    if masks is None:
        show_masks = False

    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
#         if label not in labels_color:
#             continue
        # c = labels_color.get(label, np.array([0., 0., 0.]))
        c = color_array[i]

        y1, x1, y2, x2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        if show_bboxes:
            if x1 == x2 and y1 == y2:
                ax.plot([y1], [x1], marker='o', markersize=3, color=c)
            else:
                b = Rectangle((y1, x1), y2-y1, x2-x1, linewidth=2,
                              alpha=0.5, linestyle="solid", edgecolor=c, facecolor="none")
                ax.add_patch(b)
        
        if show_texts and show_scores:
            text = "{}={:.4f}".format(labels_text[label], scores[i])
        elif show_texts:
            text = "{}".format(labels_text[label])
        elif show_scores:
            text = "{:.4f}".format(scores[i])
        else:
            text = None
        
        if text is not None:
            text_pad = 4
            ax.annotate(text, (y1, x1), xytext=(text_pad, text_pad), textcoords='offset pixels',
                        bbox=dict(facecolor=c, edgecolor='none', alpha=0.3, pad=text_pad), color='b', # backgroundcolor=c, 
                        fontsize=6, ha='left', va='bottom')
            # ax.annotate(text, (y1, x1), color=c, weight='bold', ) # fontsize=6, ha='center', va='center')

        if show_masks:
            mask = masks[i]
            if mask is not None:
                if isinstance(mask, Mask):
                    polygon = mask.poly().m
                else:
                    polygon = mask if isinstance(mask, list) else binary_mask_to_polygon(np.array(mask))
                if show_masks == 'face':
                    p_params = {'edgecolor': 'none', 'facecolor': c, 'linewidth': 0.0, 'alpha': 0.3,}
                elif show_masks == 'edge':
                    p_params = {'edgecolor': c, 'facecolor': 'none', 'linewidth': 2.0, 'alpha': 1.0,}
                else:
                    p_params = {'edgecolor': c, 'facecolor': c, 'linewidth': 1.0, 'alpha': 0.3,}
                for verts in polygon:
                    p = Polygon(np.array(verts), **p_params)
                    ax.add_patch(p)


def debug_plot(img_1, objects_1, filename="plot.png"):
    labels_color = {
        1: np.array([0., 1., 0.]),
        2: np.array([1., 0., 0.]),
        3: np.array([0., 0., 1.]),
        4: np.array([1., 1., 0.]),
        5: np.array([0.        , 0.58039216, 0.88235294]),
        6: np.array([0.39215686, 0.        , 1.        ]),
        -100: np.array([0.58039216, 0.58039216, 0.58039216]),
    }
    labels_text = {
        1: 'tumor nuclei',
        2: 'stroma nuclei',
        3: 'lymphocyte nuclei',
        4: 'macrophage nuclei',
        5: 'dead nuclei',
        6: 'ductal epithelium',
        -100: 'unlabeled',
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(24, 12))
    ax = axes.ravel()
    ax[0].imshow(img_1)
    overlay_detections(ax[0], bboxes=objects_1['boxes'], labels=objects_1['labels']+1, 
                       labels_color=labels_color, labels_text=labels_text,
                       show_bboxes=True, show_texts=False, show_masks=True, show_scores=False,
                      )
    
    plt.savefig(filename)
