from ultralytics import YOLO
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import cv2
import numpy as np
import re

def clean_plate_text(text):
    text = text.upper()
    text = re.sub(r"[^A-Z0-9\- ]", "", text)
    return text

class LicensePlateDetector:
    def __init__(self, yolo_path, device=None):
        self.model = YOLO(yolo_path)
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        # TrOCR
        self.trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
        self.trocr_model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-base-stage1"
        ).to(self.device)

    def detect(self, img):
        results = self.model(img, imgsz=640)
        return results[0].boxes if results[0].boxes is not None else None

    def run_ocr(self, crop):
        img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb).resize((256,64))
        inputs = self.trocr_processor(images=pil_img, return_tensors="pt").pixel_values.to(self.device)
        output = self.trocr_model.generate(inputs)
        text = self.trocr_processor.batch_decode(output, skip_special_tokens=True)[0]
        return clean_plate_text(text)
