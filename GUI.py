import os
import torch
import cv2
import numpy as np
from tkinter import filedialog, Tk, Label, Button
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

from models.dlinknet3 import DLinkNet34
from postprocess import postprocess_mask
from visualize import visualize_sample

# === Config ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "checkpoints/model_best.pth"
threshold = 0.35

# === Load model ===
model = DLinkNet34(num_classes=1)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# === Preprocess function ===
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    input_tensor = image_rgb.astype(np.float32) / 255.0
    input_tensor = input_tensor * 3.2 - 1.6
    input_tensor = np.transpose(input_tensor, (2, 0, 1))
    input_tensor = torch.tensor(input_tensor).unsqueeze(0).to(device)

    return image_rgb, input_tensor

# === Predict function ===
def predict(image_path):
    image_rgb, input_tensor = preprocess_image(image_path)
    with torch.no_grad():
        output = model(input_tensor)
        pred = (output.squeeze().cpu().numpy() > threshold).astype(np.uint8)
    pred = postprocess_mask(pred, min_size=10, kernel_size=1, dilate_iter=1)
    return image_rgb, pred

# === GUI Application ===
class RoadSegApp:
    def __init__(self, master):
        self.master = master
        master.title("Road Segmentation App")

        self.label = Label(master, text="Select an image to predict road mask")
        self.label.pack(pady=10)

        self.select_button = Button(master, text="Select Image", command=self.load_image)
        self.select_button.pack(pady=5)

        self.quit_button = Button(master, text="Quit", command=master.quit)
        self.quit_button.pack(pady=5)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.png *.tif *.tiff *.jpeg")]
        )
        if file_path:
            image_rgb, pred_mask = predict(file_path)
            visualize_sample(image_rgb, pred=pred_mask)

# === Run ===
if __name__ == "__main__":
    root = Tk()
    root.geometry("300x150")
    app = RoadSegApp(root)
    root.mainloop()