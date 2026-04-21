import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import rasterio


def load_image(filename):
    ext = splitext(filename)[1]

    if ext == '.npy':
        return np.load(filename)

    elif ext in ['.pt', '.pth']:
        return torch.load(filename).numpy()

    else:  # TIFF / GeoTIFF
        with rasterio.open(filename) as ds:
            img = ds.read()   # (C, H, W)
        return img



def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = load_image(mask_file)
    if mask.ndim == 3:
        mask = mask[0]  # 只取第 1 波段
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(img, is_mask):
        if is_mask:
            if img.ndim == 3:
                img = img[0]
            return img.astype(np.int64)
        else:
            img = img.astype(np.float32)
            # 如果是 0-255 范围，建议先归一化
            if img.max() > 1:
                img /= 255.0
            # 标准化
            img = (img - img.mean()) / (img.std() + 1e-6)
            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        # 如果是 rasterio 加载的，形状通常是 (C, H, W)
        if img.ndim == 3:
            if img.shape[0] == 4:  # 如果是 4 通道 (RGBA 或多光谱)
                img = img[:3, :, :]  # 只取前 3 个通道 (RGB)
            elif img.shape[0] == 1:  # 如果是单通道 (灰度)
                img = np.concatenate([img] * 3, axis=0)  # 广播成 3 通道
        # 如果是 PIL 或其他方式加载的 (H, W, C)
        elif img.ndim == 2:
            img = np.stack([img] * 3, axis=0)

        # 确保最终形状是 (3, H, W) 供后续使用
        # ------------------------------

        assert img.shape[-2:] == mask.shape[-2:], \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, is_mask=False)
        mask = self.preprocess(mask, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')
