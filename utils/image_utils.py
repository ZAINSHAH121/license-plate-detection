import cv2
class ImageProcessor:

    def annotate_and_crop(self, img, boxes, ocr_fn):
        annotated = img.copy()
        crops = []

        if boxes is None or boxes.xyxy is None:
            return annotated, crops

        for box in boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box[:4])

            crop = img[y1:y2, x1:x2]
            text = ocr_fn(crop)

            cv2.putText(annotated, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(annotated, (x1, y1), (x2, y2),
                          (0, 255, 0), 2)

            crops.append((crop, text))

        return annotated, crops

    def resize_for_screen(self, img, max_w=900, max_h=900):
        h, w = img.shape[:2]
        scale = min(max_w / w, max_h / h, 1.0)
        return cv2.resize(img, (int(w * scale), int(h * scale)))
