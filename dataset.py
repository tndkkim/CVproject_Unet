import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2


class PneumothoraxDataset(Dataset):
    def __init__(self, image_ids, train_rle, dicom_path=None, file_paths_dict=None, 
                 transform=None, use_mirroring=False, pad_size=92):

        self.train_rle = train_rle
        self.transform = transform
        self.use_mirroring = use_mirroring
        self.pad_size = pad_size

        # 파일 경로 딕셔너리 설정
        if file_paths_dict is not None:
            self.file_paths = file_paths_dict
        elif dicom_path is not None:
            all_files = list(dicom_path.rglob('*.dcm'))
            
            self.file_paths = {}
            for file_path in tqdm(all_files, desc="인덱싱"):
                img_id = file_path.stem
                self.file_paths[img_id] = file_path
        else:
            raise ValueError("파일 경로 인")
        
        # 실제로 파일이 있는 image_id만 필터링
        valid_ids = []
        
        for img_id in image_ids:
            if img_id in self.file_paths:
                valid_ids.append(img_id)
        
        self.image_ids = valid_ids
        print(f"사용 가능한 이미지: {len(self.image_ids)}개")
        print()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        file_path = self.file_paths[img_id]
        
        # DICOM 파일 읽기
        dcm = pydicom.dcmread(file_path)
        image = dcm.pixel_array

        if image.max() > 0:
            image = image.astype(np.float32) / image.max()
        else:
            image = image.astype(np.float32)
        
        image = np.stack([image, image, image], axis=-1)  # RGB

        # 마스크 로드
        rle = self.train_rle[self.train_rle['ImageId'] == img_id][' EncodedPixels'].values[0]

        # Negative 샘플인 경우 빈 마스크 생성
        if rle == '-1' or pd.isna(rle):
            mask = np.zeros((1024, 1024), dtype=np.float32)
        else:
            mask = rle_decode(rle, 1024, 1024).astype(np.float32)

        # Mirror padding (선택적)
        if self.use_mirroring:
            image = self.mirror_pad(image, self.pad_size)
            mask = self.mirror_pad(mask, self.pad_size)

        # Augmentation
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        mask = mask.unsqueeze(0)  # (H, W) -> (1, H, W)
        return image, mask
    
    def mirror_pad(self, img, pad_size):
        if len(img.shape) == 3:  # RGB image
            return np.pad(img, 
                         ((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
                         mode='reflect')
        else:  # Grayscale
            return np.pad(img, 
                         ((pad_size, pad_size), (pad_size, pad_size)),
                         mode='reflect')


def build_file_paths_dict(dicom_path):
    """
    파일 경로 딕셔너리를 미리 생성하는 헬퍼 함수
    이 함수를 한 번 실행하고 결과를 저장해두면 매번 파일 스캔을 안 해도 됨
    """
    all_files = list(dicom_path.rglob('*.dcm'))
    print(f"전체 DICOM 파일 수: {len(all_files)}")
    
    file_paths = {}
    for file_path in tqdm(all_files, desc="인덱싱"):
        img_id = file_path.stem
        file_paths[img_id] = file_path

    return file_paths


def rle_decode(rle, width, height):
    if pd.isna(rle) or str(rle).strip() == '-1':
        return np.zeros((height, width), dtype=np.uint8)

    array = np.asarray([int(x) for x in str(rle).strip().split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    mask = np.zeros(width * height, dtype=np.uint8)

    for start, length in zip(starts, lengths):
        current_position += start
        mask[current_position:current_position+length] = 1
        current_position += length

    return mask.reshape(width, height).T


def get_train_transforms():
    return A.Compose([
        # Elastic deformation
        # A.ElasticTransform(
        #     alpha=720,
        #     sigma=24,
        #     alpha_affine=24,
        #     p=0.3
        # ),
        
        # Geometric transforms
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=20, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        
        # Pixel-level transforms
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        
        # Normalization
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_val_transforms():
    return A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])