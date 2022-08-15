import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
import random
import numbers

#------- ColorJitter -------#
class ColorJitter(object):
    def __init__(self, color_adjust_para):
        """brightness [max(0, 1 - brightness), 1 + brightness] or the given [min, max]"""
        """contrast [max(0, 1 - contrast), 1 + contrast] or the given [min, max]"""
        """saturation [max(0, 1 - saturation), 1 + saturation] or the given [min, max]"""
        """hue [-hue, hue] 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5"""
        '''Ajust brightness, contrast, saturation, hue'''
        '''Input: PIL Image, Output: PIL Image'''
        self.brightness, self.contrast, self.saturation, self.hue = color_adjust_para

    def __call__(self, in_img):
        if self.brightness > 0:
            brightness_factor = np.random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
            in_img = F.adjust_brightness(in_img, brightness_factor)

        if self.contrast > 0:
            contrast_factor = np.random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
            in_img = F.adjust_contrast(in_img, contrast_factor)

        if self.saturation > 0:
            saturation_factor = np.random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
            in_img = F.adjust_saturation(in_img, saturation_factor)

        if self.hue > 0:
            hue_factor = np.random.uniform(-self.hue, self.hue)
            inputs = F.adjust_hue(in_img, hue_factor)

        in_img = np.asarray(in_img)
        in_img = in_img.clip(0,1)

        return in_img