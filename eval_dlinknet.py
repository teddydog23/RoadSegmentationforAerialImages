import os
import torch
import cv2
import numpy as np
import pandas as pd
from model.dlinknet3 import DLinkNet34
from utils.metrics import compute_metrics
from visualize import visualize_sample
from postprocess import postprocess_mask
from tqdm import tqdm

# === Thêm Confusion Matrix ===
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Cấu hình ===
model_path = "checkpoints/dlinknet_model_best.pth"
image_root = "data/1024/images"
mask_root = "data/1024/masks"
csv_path = "data/1024/split.csv"
threshold = 0.35
num_images_to_show = 5

# === Load model ===
model = DLinkNet34(num_classes=1)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# === Load file ảnh
df = pd.read_csv(csv_path)
test_df = df[df['split'] == 'test']

# Lấy danh sách file ảnh và mask
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

    input_tensor = image_rgb.astype(np.float32) / 255.0
    input_tensor = input_tensor * 3.2 - 1.6
    input_tensor = np.transpose(input_tensor, (2, 0, 1))
    input_tensor = torch.tensor(input_tensor).unsqueeze(0).to(device)

    # Dự đoán
    with torch.no_grad():
        output = model(input_tensor)
        pred = (output.squeeze().cpu().numpy() > threshold).astype(np.uint8)
    
    # === Hậu xử lý ===
    pred = postprocess_mask(pred, min_size=10, kernel_size=1, dilate_iter=1)

    all_preds.append(torch.tensor(pred))
    all_gts.append(torch.tensor((mask > 127).astype(np.uint8)))

    # if idx < num_images_to_show:
    #     visualize_sample(image_rgb, mask, pred)

# === Tính Dice & IoU
all_preds = torch.stack(all_preds).unsqueeze(1)
all_gts = torch.stack(all_gts).unsqueeze(1)

dice, iou, miou = compute_metrics(all_preds, all_gts)

print("\n📊 Evaluation Metrics:")
print(f"🔹 Dice Score: {dice:.4f}")
print(f"🔹 IoU Score : {iou:.4f}")
print(f"🔹 mIoU Score: {miou:.4f}")


# === Thêm Confusion Matrix ===
# Làm phẳng dự đoán và nhãn thực tế
flat_preds = all_preds.flatten().numpy()
flat_gts = all_gts.flatten().numpy()

# Tính confusion matrix
cm = confusion_matrix(flat_gts, flat_preds, labels=[0, 1])

# In confusion matrix
print("\n📊 Confusion Matrix:")
print(cm)
print(f"True Negatives (TN): {cm[0, 0]}")
print(f"False Positives (FP): {cm[0, 1]}")
print(f"False Negatives (FN): {cm[1, 0]}")
print(f"True Positives (TP): {cm[1, 1]}")

# Tính Precision, Recall, F1-Score
TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"🔹 Precision: {precision:.4f}")
print(f"🔹 Recall   : {recall:.4f}")
print(f"🔹 F1-Score: {f1:.4f}")

# Vẽ confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Road', 'Road'], yticklabels=['Non-Road', 'Road'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')  # Lưu ảnh
plt.show()