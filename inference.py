import os
import cv2
import numpy as np
import torch
from model.dlinknet3 import DLinkNet34
from postprocess import postprocess_mask
from visualize import visualize_sample

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Cấu hình ===
model_path = "checkpoints/model_best.pth"
input_dir = "inference_input"
output_dir = "inference_output"
os.makedirs(output_dir, exist_ok=True)

threshold = 0.5
target_size = (1024, 1024)

# Postprocess config
POSTPROCESS = True
MIN_SIZE = 150
KERNEL_SIZE = 3
DILATE_ITER = 1

# === Load model ===
model = DLinkNet34(num_classes=1)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# === Duyệt qua tất cả ảnh
for filename in os.listdir(input_dir):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
        continue

    # Đọc ảnh gốc
    image_bgr = cv2.imread(os.path.join(input_dir, filename))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    orig_height, orig_width = image_rgb.shape[:2]

    # Resize ảnh → 1024x1024 (phù hợp với model)
    image_resized = cv2.resize(image_rgb, target_size, interpolation=cv2.INTER_LINEAR)

    # Chuẩn hóa ảnh
    input_tensor = image_resized.astype(np.float32) / 255.0
    input_tensor = input_tensor * 3.2 - 1.6
    input_tensor = np.transpose(input_tensor, (2, 0, 1))
    input_tensor = torch.tensor(input_tensor).unsqueeze(0).to(device)

    # === Dự đoán
    with torch.no_grad():
        output = model(input_tensor)
        pred = output.squeeze().cpu().numpy()
        pred_mask = (pred > threshold).astype(np.uint8)

    # === Hậu xử lý (tuỳ chọn)
    if POSTPROCESS:
        pred_mask = postprocess_mask(
            pred_mask,
            min_size=MIN_SIZE,
            kernel_size=KERNEL_SIZE,
            dilate_iter=DILATE_ITER
        )

    # === Resize mask về lại size ảnh gốc
    pred_mask_resized = cv2.resize(
        pred_mask.astype(np.uint8),
        (orig_width, orig_height),
        interpolation=cv2.INTER_NEAREST
    )

    # === Lưu kết quả
    save_path = os.path.join(output_dir, filename.rsplit('.', 1)[0] + "_mask.png")
    cv2.imwrite(save_path, pred_mask_resized * 255)

    # === Hiển thị overlay
    visualize_sample(image_rgb, pred=pred_mask_resized)
