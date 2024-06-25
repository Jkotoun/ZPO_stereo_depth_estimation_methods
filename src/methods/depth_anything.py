#author: Vit Tlustos
import torch
import numpy as np
import cv2
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

class DepthAnything:

    def __init__(self, variant):
        self.image_processor = AutoImageProcessor.from_pretrained(f"LiheYoung/depth-anything-{variant}-hf")
        self.model = AutoModelForDepthEstimation.from_pretrained(f"LiheYoung/depth-anything-{variant}-hf")

    def process(self, img):
        # 1. prepare image for the model
        inputs = self.image_processor(
            images=img, 
            return_tensors="pt"
        )

        # 2. forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # 3. interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        )

        # 4. visualize
        depth = prediction.squeeze().cpu().numpy()
        depth = (depth * 255 / np.max(depth)).astype("uint8")
        depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)

        return depth
