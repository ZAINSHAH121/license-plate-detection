from ultralytics import YOLO
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import cv2
import re


def clean_plate_text(text):
    text = text.upper()
    text = re.sub(r"[^A-Z0-9\- ]", "", text)
    return text


class LicensePlateDetector:
    def __init__(self, yolo_path):
        self.model = YOLO(yolo_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
        self.ocr_model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-base-stage1"
        ).to(self.device)

    def detect(self, img):
        result = self.model(img, imgsz=640)
        return result[0].boxes

    def run_ocr(self, crop):
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb).resize((256, 64))
        inp = self.processor(images=pil_img, return_tensors="pt").pixel_values.to(self.device)
        out = self.ocr_model.generate(inp)
        text = self.processor.batch_decode(out, skip_special_tokens=True)[0]
        return clean_plate_text(text)
