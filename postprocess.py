import cv2
import numpy as np
from skimage.morphology import remove_small_objects

def postprocess_mask(mask, min_size=100, kernel_size=3, dilate_iter=1):
    """
    Thực hiện hậu xử lý cho mask nhị phân:
    - Nối vùng đứt gãy (dilate)
    - Xoá nhiễu nhỏ (remove small objects)
    
    Args:
        mask: numpy array, binary mask (0 hoặc 1)
        min_size: số pixel nhỏ nhất giữ lại
        kernel_size: kích thước kernel cho dilation
        dilate_iter: số lần lặp dilation
        
    Returns:
        mask đã hậu xử lý (np.uint8: 0 hoặc 1)
    """
    mask = mask.astype(np.uint8)

    # Morphological dilation
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=dilate_iter)

    # Remove small objects
    mask = remove_small_objects(mask.astype(bool), min_size=min_size)
    mask = mask.astype(np.uint8)

    return mask
