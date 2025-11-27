import cv2
class MediaProcessor:
    def __init__(self, detector):
        self.detector = detector

    def process(self, source):
        if source.lower().endswith((".jpg", ".jpeg", ".png")):
            self._process_image(source)
        elif source.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            self._process_video(source)
        else:
            raise ValueError("Unsupported file type!")

    def _process_image(self, path):
        img = cv2.imread(path)
        annotated = img.copy()
        boxes = self.detector.detect(img)
        crops = []

        if boxes and boxes.xyxy is not None:
            for box in boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box[:4])
                crop = img[y1:y2, x1:x2]
                crops.append(crop)
                text = self.detector.run_ocr(crop)
                cv2.putText(annotated, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)
                print("Detected Plate:", text)

        right_window = None
        if crops:
            resized_crops = []
            h = annotated.shape[0]
            for c in crops:
                scale = h / c.shape[0]
                w = int(c.shape[1] * scale)
                resized = cv2.resize(c, (w, h))
                text = self.detector.run_ocr(c)
                cv2.putText(resized, text, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                resized_crops.append(resized)
            right_window = cv2.hconcat(resized_crops)

        # Concatenate left + right (if any)
        side_by_side = cv2.hconcat([annotated, right_window]) if right_window is not None else annotated
        side_by_side = self._resize_for_screen(side_by_side)
        cv2.imshow("License Plate Detection", side_by_side)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _process_video(self, path):
        cap = cv2.VideoCapture(path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            annotated = frame.copy()
            boxes = self.detector.detect(frame)

            if boxes and boxes.xyxy is not None:
                for box in boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box[:4])
                    text = self.detector.run_ocr(frame[y1:y2, x1:x2])
                    cv2.putText(annotated, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)
                    print("Detected Plate:", text)

            annotated = self._resize_for_screen(annotated)
            cv2.imshow("Video - License Plate Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    def _resize_for_screen(self, img, max_width=1600, max_height=1000):
        h, w = img.shape[:2]
        scale = min(max_width / w, max_height / h, 1)
        return cv2.resize(img, (int(w*scale), int(h*scale))) if scale < 1 else img
