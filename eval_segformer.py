import os
import torch
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from transformers import SegformerForSemanticSegmentation
from utils.metrics import compute_metrics
from visualize import visualize_sample
from postprocess import postprocess_mask

# === Cấu hình ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "checkpoints/segformer_model_best.pth"
image_root = "data/512/images"
mask_root = "data/512/masks"
csv_path = "data/512/split.csv"
threshold = 0.3
kernel_size = 1
num_images_to_show = 5

# === Load model ===
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b1-finetuned-ade-512-512",
    num_labels=1,
    ignore_mismatched_sizes=True,
)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# === Load file ảnh
df = pd.read_csv(csv_path)
test_df = df[df['split'] == 'test'].reset_index(drop=True)
image_filenames = test_df['filename'].tolist()
mask_filenames = test_df['maskname'].tolist()

all_preds = []
all_gts = []

for idx, (img_file, mask_file) in tqdm(enumerate(zip(image_filenames, mask_filenames)), total=len(image_filenames)):
    img_path = os.path.join(image_root, img_file)
    mask_path = os.path.join(mask_root, mask_file)

    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Resize về 512x512 (giống như trong training)
    image_resized = cv2.resize(image_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

    # Normalize [-1, 1]
    input_tensor = image_resized.astype(np.float32) / 255.0
    input_tensor = input_tensor * 2.0 - 1.0
    input_tensor = np.transpose(input_tensor, (2, 0, 1))  # HWC → CHW
    input_tensor = torch.tensor(input_tensor).unsqueeze(0).to(device)

    # === Dự đoán
    with torch.no_grad():
        output = model(input_tensor)
        logits = output.logits
        logits = torch.nn.functional.interpolate(logits, size=(512, 512), mode='bilinear', align_corners=False)
        pred = (torch.sigmoid(logits).squeeze().cpu().numpy() > threshold).astype(np.uint8)

    # Hậu xử lý (nếu muốn)
    pred = postprocess_mask(pred, min_size=500, kernel_size=kernel_size, dilate_iter=1)

    all_preds.append(torch.tensor(pred))
    all_gts.append(torch.tensor((mask_resized > 127).astype(np.uint8)))

    # Hiển thị một vài ảnh đầu tiên
    if idx < num_images_to_show:
        visualize_sample(image_resized, mask_resized, pred)

# === Tính Dice & IoU
all_preds = torch.stack(all_preds).unsqueeze(1)
all_gts = torch.stack(all_gts).unsqueeze(1)

dice, iou, miou = compute_metrics(all_preds, all_gts)

print("\n📊 Evaluation Metrics:")
print(f"🔹 Dice Score: {dice:.4f}")
print(f"🔹 IoU Score : {iou:.4f}")
print(f"🔹 mIoU Score: {miou:.4f}")

# === Confusion Matrix
flat_preds = all_preds.flatten().numpy()
flat_gts = all_gts.flatten().numpy()
cm = confusion_matrix(flat_gts, flat_preds, labels=[0, 1])

print("\n📊 Confusion Matrix:")
print(cm)
print(f"True Negatives (TN): {cm[0, 0]}")
print(f"False Positives (FP): {cm[0, 1]}")
print(f"False Negatives (FN): {cm[1, 0]}")
print(f"True Positives (TP): {cm[1, 1]}")

# Precision, Recall, F1
TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"🔹 Precision: {precision:.4f}")
print(f"🔹 Recall   : {recall:.4f}")
print(f"🔹 F1-Score : {f1:.4f}")

# Heatmap
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=['Non-Road', 'Road'], yticklabels=['Non-Road', 'Road'])
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix')
# plt.savefig('confusion_matrix_segformer.png')
# plt.show()
