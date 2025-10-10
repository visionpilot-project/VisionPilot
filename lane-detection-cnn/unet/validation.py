import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

IMG_SIZE = (256, 320)
MODEL_PATH = "lane_detection_unet.h5"
VAL_IMG_DIR = "val_img"

model = tf.keras.models.load_model(MODEL_PATH)

img_files = [os.path.join(VAL_IMG_DIR, f) for f in os.listdir(VAL_IMG_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
img_files.sort()

images = []
pred_masks = []

for img_path in img_files:
	img = cv2.imread(img_path)
	img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img_resized = cv2.resize(img_rgb, (IMG_SIZE[1], IMG_SIZE[0]))
	img_norm = img_resized / 255.0
	images.append(img_resized)
	input_tensor = np.expand_dims(img_norm, axis=0)
	pred = model.predict(input_tensor)[0]
	pred_mask = (pred.squeeze() >= 0.5).astype(np.uint8) * 255
	pred_masks.append(pred_mask)

num_imgs = len(images)
fig, axes = plt.subplots(num_imgs, 2, figsize=(8, 4 * num_imgs))
if num_imgs == 1:
	axes = np.expand_dims(axes, axis=0)
for i in range(num_imgs):
	axes[i, 0].imshow(images[i])
	axes[i, 0].set_title(f"Image {i+1}")
	axes[i, 0].axis('off')
	axes[i, 1].imshow(pred_masks[i], cmap='gray')
	axes[i, 1].set_title(f"Predicted Mask {i+1}")
	axes[i, 1].axis('off')
plt.tight_layout()
plt.show()