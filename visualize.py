import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_sample(image, mask=None, pred=None, overlay_alpha=0.4):
    """
    Hiển thị ảnh gốc, mask thật, mask dự đoán và overlay.
    Args:
        image: numpy array (H, W, 3)
        mask: ground truth mask (H, W)
        pred: predicted mask (H, W)
    """
    plt.figure(figsize=(10, 10))

    # 1. Ảnh gốc
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.title("Image")
    plt.axis("off")

    # 2. Mask thật
    if mask is not None:
        plt.subplot(2, 2, 2)
        plt.imshow(mask, cmap='gray')
        plt.title("Ground Truth")
        plt.axis("off")

    # 3. Mask dự đoán
    if pred is not None:
        plt.subplot(2, 2, 3)
        plt.imshow(pred, cmap='gray')
        plt.title("Prediction")
        plt.axis("off")

        # 4. Overlay
        plt.subplot(2, 2, 4)
        overlay = image.copy()
        red_mask = np.zeros_like(image)
        red_mask[:, :, 0] = pred * 255
        cv2.addWeighted(red_mask, overlay_alpha, image, 1 - overlay_alpha, 0, overlay)
        plt.imshow(overlay)
        plt.title("Overlay")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# def visualize_sample(image, gt=None, pred=None):
#     """
#     Hiển thị ảnh gốc, ground truth và prediction mask
#     """
#     plt.figure(figsize=(15, 5))
#     plt.subplot(1, 3, 1)
#     plt.imshow(image)
#     plt.title("Original Image")

#     if gt is not None:
#         plt.subplot(1, 3, 2)
#         plt.imshow(gt, cmap='gray')
#         plt.title("Ground Truth")

#     if pred is not None:
#         plt.subplot(1, 3, 3)
#         plt.imshow(pred, cmap='gray')
#         plt.title("Prediction")

#     plt.tight_layout()
#     plt.show()