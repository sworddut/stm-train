import numpy as np
import torch
import math
import cv2

import random
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from .data import convert_mask, convert_one_hot, MAX_TRAINING_OBJ

class Compose(object):
    """
    Combine several transformation in a serial manner
    """

    def __init__(self, transform=[]):
        self.transforms = transform

    def __call__(self, imgs, annos, features=None):
        for m in self.transforms:
            imgs, annos, features = m(imgs, annos, features)
        return imgs, annos, features

class Transpose(object):

    """
    transpose the image and mask
    """

    def __call__(self, imgs, annos, features=None):

        H, W, _ = imgs[0].shape
        if H <= W:
            return imgs, annos, features
        else:
            timgs = [np.transpose(img, [1, 0, 2]) for img in imgs]
            tannos = [np.transpose(anno, [1, 0, 2]) for anno in annos]

            if features is not None:
                tfeatures = [np.transpose(feature, [1, 0, 2]) for feature in features]
                return timgs, tannos, tfeatures
            return timgs, tannos, None

class RandomAffine(object):

    """
    Affine Transformation to each frame
    """

    def __call__(self, imgs, annos, features=None):

        seq = iaa.Sequential([
            iaa.Crop(percent=(0.0, 0.1), keep_size=True),
            iaa.Affine(scale=(0.95, 1.05), shear=(-10, 10), rotate=(-15, 15))
        ])

        seq = seq.to_deterministic()

        num = len(imgs)
        for idx in range(num):
            img = imgs[idx]
            anno = annos[idx]
            max_obj = anno.shape[2]-1

            anno = convert_one_hot(anno, max_obj)
            segmap = SegmentationMapsOnImage(anno, shape=img.shape)
            img_aug, segmap_aug = seq(image=img, segmentation_maps=segmap)
            imgs[idx] = img_aug
            annos[idx] = convert_mask(segmap_aug.get_arr(), max_obj)

            if features is not None:
                feature = features[idx]
                feature_aug = seq(image=feature)
                features[idx] = feature_aug

        return imgs, annos, features

class RandomCropPad(object):

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, imgs, annos, features=None):

        num = len(imgs)

        th, tw = self.output_size
        ih, iw = imgs[0].shape[:2]

        # pad
        pt = max(th - ih, 0) // 2
        pb = max(th - ih, 0) - pt
        pl = max(tw - iw, 0) // 2
        pr = max(tw - iw, 0) - pl

        for i in range(num):
            imgs[i] = np.pad(imgs[i], ((pt, pb), (pl, pr), (0, 0)), mode='constant')
            annos[i] = np.pad(annos[i], ((pt, pb), (pl, pr), (0, 0)), mode='constant')

            if features is not None:
                features[i] = np.pad(features[i], ((pt, pb), (pl, pr), (0, 0)), mode='constant')

        ih, iw = imgs[0].shape[:2]
        valid = False
        # crop
        while not valid:
            sl = random.randrange(iw - tw + 1)
            st = random.randrange(ih - th + 1)
            valid = np.sum(annos[0][st:st+th, sl:sl+tw, 1:]) > 0

        for i in range(num):
            imgs[i] = imgs[i][st:st+th, sl:sl+tw]
            annos[i] = annos[i][st:st+th, sl:sl+tw]

            if features is not None:
                features[i] = features[i][st:st+th, sl:sl+tw]

        for k in range(1, annos[0].shape[2]):
            if np.sum(annos[0][:, :, k]) == 0:
                for i in range(1, num):
                    annos[i][:, :, k] = 0

        return imgs, annos, features

class AdditiveNoise(object):
    """
    sum additive noise
    """

    def __init__(self, delta=5.0):
        self.delta = delta
        assert delta > 0.0

    def __call__(self, imgs, annos, features=None):
        v = np.random.uniform(-self.delta, self.delta)
        for id, img in enumerate(imgs):
            imgs[id] += v

        return imgs, annos, features


class RandomContrast(object):
    """
    randomly modify the contrast of each frame
    """

    def __init__(self, lower=0.97, upper=1.03):
        self.lower = lower
        self.upper = upper
        assert self.lower <= self.upper
        assert self.lower > 0

    def __call__(self, imgs, annos, features=None):
        v = np.random.uniform(self.lower, self.upper)
        for id, img in enumerate(imgs):
            imgs[id] *= v

        return imgs, annos, features


class RandomMirror(object):
    """
    Randomly horizontally flip the video volume
    """

    def __init__(self):
        pass

    def __call__(self, imgs, annos, features=None):

        v = random.randint(0, 1)
        if v == 0:
            return imgs, annos, features

        sample = imgs[0]
        h, w = sample.shape[:2]

        for id, img in enumerate(imgs):
            imgs[id] = img[:, ::-1, :]

        for id, anno in enumerate(annos):
            annos[id] = anno[:, ::-1, :]

        if features is not None:
            for id, feature in enumerate(features):
                features[id] = feature[:, ::-1, :]

        return imgs, annos, features

class ToFloat(object):
    """
    convert value type to float
    """

    def __init__(self):
        pass

    def __call__(self, imgs, annos, features=None):
        for idx, img in enumerate(imgs):
            imgs[idx] = img.astype(dtype=np.float32, copy=True)

        for idx, anno in enumerate(annos):
            annos[idx] = anno.astype(dtype=np.float32, copy=True)

        if features is not None:
            for idx, feature in enumerate(features):
                features[idx] = feature.astype(dtype=np.float32, copy=True)

        return imgs, annos, features

class Rescale(object):

    """
    rescale the size of image and masks
    """

    def __init__(self, target_size):
        assert isinstance(target_size, (int, tuple, list))
        if isinstance(target_size, int):
            self.target_size = (target_size, target_size)
        else:
            self.target_size = target_size

    def __call__(self, imgs, annos, features=None):

        h, w = imgs[0].shape[:2]
        new_height, new_width = self.target_size

        factor = min(new_height / h, new_width / w)
        height, width = int(factor * h), int(factor * w)
        pad_l = (new_width - width) // 2
        pad_t = (new_height - height) // 2

        for id, img in enumerate(imgs):
            canvas = np.zeros((new_height, new_width, 3), dtype=np.float32)
            rescaled_img = cv2.resize(img, (width, height))
            canvas[pad_t:pad_t+height, pad_l:pad_l+width, :] = rescaled_img
            imgs[id] = canvas

        for id, anno in enumerate(annos):
            canvas = np.zeros((new_height, new_width, anno.shape[2]), dtype=np.float32)
            rescaled_anno = cv2.resize(anno, (width, height), cv2.INTER_NEAREST)
            canvas[pad_t:pad_t + height, pad_l:pad_l + width, :] = rescaled_anno
            annos[id] = canvas

        if features is not None:
            for id, feature in enumerate(features):
                canvas = np.zeros((new_height, new_width, feature.shape[2]), dtype=np.float32)
                rescaled_feature = cv2.resize(feature, (width, height))
                canvas[pad_t:pad_t + height, pad_l:pad_l + width, :] = rescaled_feature
                features[id] = canvas

        return imgs, annos, features

class Stack(object):
    """
    stack adjacent frames into input tensors
    """

    def __call__(self, imgs, annos, features=None):

        num_img = len(imgs)
        num_anno = len(annos)

        assert num_img == num_anno, f'number of images {num_img} not equal to number of annotations {num_anno}'

        img_stack = np.stack(imgs, axis=0)
        anno_stack = np.stack(annos, axis=0)
        
        if features is not None:
            feature_stack = np.stack(features, axis=0)
            return img_stack, anno_stack, feature_stack
        return img_stack, anno_stack, None

class ToTensor(object):
    """
    convert to torch.Tensor
    """

    def __call__(self, imgs, annos, features=None):
        imgs = torch.from_numpy(imgs.copy())
        annos = torch.from_numpy(annos.astype(np.uint8, copy=True)).float()

        imgs = imgs.permute(0, 3, 1, 2).contiguous()
        annos = annos.permute(0, 3, 1, 2).contiguous()

        if features is not None:
            features = torch.from_numpy(features.copy())
            features = features.permute(0, 3, 1, 2).contiguous()
            return imgs, annos, features
        return imgs, annos, None

class Normalize(object):
    def __init__(self):
        # RGB mean and std from ImageNet, 第4通道使用0均值和1标准差
        self.mean = np.array([0.485, 0.456, 0.406, 0.0]).reshape([1, 1, 4]).astype(np.float32)
        self.std = np.array([0.229, 0.224, 0.225, 1.0]).reshape([1, 1, 4]).astype(np.float32)

    def __call__(self, imgs, annos, features=None):
        if features is not None:
            # 将RGB图像和特征组合成4通道输入
            for idx, (img, feat) in enumerate(zip(imgs, features)):
                combined = np.concatenate([img, feat[..., np.newaxis]], axis=-1)
                imgs[idx] = (combined - self.mean) / self.std
        else:
            # 如果没有特征,只处理RGB图像
            for idx, img in enumerate(imgs):
                imgs[idx] = (img / 255.0 - self.mean[:,:,:3]) / self.std[:,:,:3]
        return imgs, annos, features

class ReverseClip(object):

    def __call__(self, imgs, annos, features=None):

        return imgs[::-1], annos[::-1], features

class SampleObject(object):

    def __init__(self, num):
        self.num = num

    def __call__(self, imgs, annos, features=None):

        max_obj = annos[0].shape[2] - 1
        num_obj = 0
        while num_obj < max_obj and np.sum(annos[0][:, :, num_obj+1]) > 0:
            num_obj += 1

        if num_obj <= self.num:
            return imgs, annos, features

        sampled_idx = random.sample(range(1, num_obj+1), self.num)
        sampled_idx.sort()
        for idx, anno in enumerate(annos):
            new_anno = anno.copy()
            new_anno[:, :, self.num+1:] = 0.0
            new_anno[:, :, 1:self.num+1] = anno[:, :, sampled_idx]
            annos[idx] = new_anno

        return imgs, annos, features

class TrainTransform(object):

    def __init__(self, size=(384,384)):
        self.transform = Compose([
            ToFloat(),
            RandomAffine(),
            RandomCropPad(size=size),
            RandomMirror(),
            Rescale(size=size),
            Normalize(),
            Stack(),
            ToTensor(),
        ])

    def __call__(self, imgs, annos, features=None):
        return self.transform(imgs, annos, features)

class TestTransform(object):

    def __init__(self, size=(384,384)):
        self.transform = Compose([
            ToFloat(),
            Rescale(size=size),
            Normalize(),
            Stack(),
            ToTensor(),
        ])

    def __call__(self, imgs, annos, features=None):
        return self.transform(imgs, annos, features)
