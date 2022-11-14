"""Datasets
"""

from torch.utils.data import Dataset
import numpy as np
import cv2
import os

class SegDataset(Dataset):
    """Dataset for image segmentation

    Attributs:
        x_dirs(list): 이미지 경로
        y_dirs(list): 마스크 이미지 경로
        input_size(list, tuple): 이미지 크기(width, height)
        scaler(obj): 이미지 스케일러 함수
        logger(obj): 로거 객체
        verbose(bool): 세부 로깅 여부
    """   
    def __init__(self, paths, input_size, scaler, mode='train', logger=None, verbose=False, transforms=None):
        
        self.x_paths = paths
        self.y_paths = list(map(lambda x : x.replace('x', 'y'),self.x_paths))
        self.input_size = input_size
        self.scaler = scaler
        self.logger = logger
        self.verbose = verbose
        self.mode = mode
        self.transforms = transforms


    def __len__(self):
        return len(self.x_paths)

    def __getitem__(self, id_: int):
        
        filename = os.path.basename(self.x_paths[id_]) # Get filename for logging
        image = cv2.imread(self.x_paths[id_], cv2.IMREAD_COLOR)
        orig_size = image.shape

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.input_size)

        if self.mode in ['train', 'valid']:

            mask = cv2.imread(self.y_paths[id_], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, self.input_size, interpolation=cv2.INTER_NEAREST)

            if self.transforms is not None:
                augmented = self.transforms(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
                image = np.transpose(image, (2, 0, 1))
                image = self.scaler(image)
                

                return image, mask, filename
            
            return image, mask, filename

        elif self.mode in ['test']:
            image = np.transpose(image, (2, 0, 1))
            image = self.scaler(image)
            return image, orig_size, filename

        else:
            assert False, f"Invalid mode : {self.mode}"


