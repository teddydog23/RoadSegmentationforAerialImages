import os
import cv2
import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation
from postprocess import postprocess_mask
from visualize import visualize_sample

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Cấu hình ===
model_path = "checkpoints/segformer_model_best.pth"
input_dir = "inference_input"
output_dir = "inference_output_segformer"
os.makedirs(output_dir, exist_ok=True)

threshold = 0.5
input_size = (512, 512)  # phù hợp với model SegFormer bạn đã train

# Postprocess config
POSTPROCESS = True
MIN_SIZE = 150
KERNEL_SIZE = 3
DILATE_ITER = 1

# === Load model ===
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b1-finetuned-ade-512-512",
    num_labels=1,
    ignore_mismatched_sizes=True
)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
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

    # Resize về input size
    image_resized = cv2.resize(image_rgb, input_size, interpolation=cv2.INTER_LINEAR)

    # Chuẩn hóa ảnh theo SegFormer: [-1, 1]
    input_tensor = image_resized.astype(np.float32) / 255.0
    input_tensor = input_tensor * 2.0 - 1.0
    input_tensor = np.transpose(input_tensor, (2, 0, 1))  # HWC → CHW
    input_tensor = torch.tensor(input_tensor).unsqueeze(0).to(device)

    # === Dự đoán
    with torch.no_grad():
        output = model(input_tensor)
        logits = output.logits
        logits = torch.nn.functional.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)
        pred = torch.sigmoid(logits).squeeze().cpu().numpy()
        pred_mask = (pred > threshold).astype(np.uint8)

    # === Hậu xử lý (tuỳ chọn)
    if POSTPROCESS:
        pred_mask = postprocess_mask(
            pred_mask,
            min_size=MIN_SIZE,
            kernel_size=KERNEL_SIZE,
            dilate_iter=DILATE_ITER
        )

    # === Resize mask về lại kích thước gốc
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
