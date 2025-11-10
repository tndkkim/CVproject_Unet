import torch
import numpy as np
from tqdm import tqdm


def predict_with_tiles(model, image, tile_size=512, overlap=92, device='cuda'):
#U-Net overlap-tile

    model.eval()
    h, w = image.shape[:2]
    
    # 출력 마스크 초기화
    output = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)
    
    stride = tile_size - overlap * 2
    
    # 타일 좌표 계산
    y_coords = list(range(0, h, stride))
    x_coords = list(range(0, w, stride))
    
    total_tiles = len(y_coords) * len(x_coords)
    
    with tqdm(total=total_tiles, desc="Predicting tiles") as pbar:
        for y in y_coords:
            for x in x_coords:
                # 타일 추출
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                
                tile = image[y:y_end, x:x_end]
                
                # Padding if needed
                pad_h = tile_size - tile.shape[0]
                pad_w = tile_size - tile.shape[1]
                
                if pad_h > 0 or pad_w > 0:
                    tile = np.pad(tile, 
                                ((0, pad_h), (0, pad_w), (0, 0)),
                                mode='reflect')
                
                # Normalize
                tile = (tile - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                
                # 예측
                tile_tensor = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).float().to(device)
                
                with torch.no_grad():
                    pred = model(tile_tensor)
                    pred = torch.sigmoid(pred).cpu().numpy()[0, 0]
                
                # Valid region (overlap 제외)
                valid_y_start = overlap if y > 0 else 0
                valid_x_start = overlap if x > 0 else 0
                valid_y_end = tile_size - overlap if y_end < h else pred.shape[0]
                valid_x_end = tile_size - overlap if x_end < w else pred.shape[1]
                
                valid_pred = pred[valid_y_start:valid_y_end, valid_x_start:valid_x_end]
                
                # 실제 이미지 좌표
                out_y_start = y + valid_y_start
                out_x_start = x + valid_x_start
                out_y_end = min(out_y_start + valid_pred.shape[0], h)
                out_x_end = min(out_x_start + valid_pred.shape[1], w)
                
                # 실제 유효한 크기로 자르기
                valid_pred = valid_pred[:out_y_end - out_y_start, :out_x_end - out_x_start]
                
                # 결과 누적
                output[out_y_start:out_y_end, out_x_start:out_x_end] += valid_pred
                count[out_y_start:out_y_end, out_x_start:out_x_end] += 1
                
                pbar.update(1)
    
    # 평균 계산
    output = output / (count + 1e-8)
    return output


def dice_coefficient(pred, target, smooth=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice.item()


def iou_score(pred, target, smooth=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    return iou.item()