import cv2
import numpy as np

class ImageProcessor:
    def annotate_frame(self, img, boxes):
        if boxes is None or len(boxes.data) == 0:
            return img, []

        annotated = img.copy()
        detections = []

        for box in boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box[:4])
            annotated = cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,0), 2)
            detections.append((x1,y1,x2,y2))

        return annotated, detections

    def show_annotated_and_crops_side_by_side(self, img, boxes):
        annotated, crops_coords = self.annotate_frame(img, boxes)
        return annotated, crops_coords
